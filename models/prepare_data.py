
from __future__ import annotations
from datetime import time

import sys
from typing import Optional, Tuple
import numpy as np

sys.path.append('.')

from models.shopper_data import ShopperDataStreamer, ShopperData


def zero_lower_bound(v: int):
    """
    Returns 0 if value `v` is less than 0, else returns `v`.
    """
    return v if v > 0 else 0


def set_limit(v1: int, v2: Optional[int]) -> int:
    """
    """
    if v2 and v2 < v1 and v2 > 0:
        return v2
    else:
        return v1


class DataStreamer:
    """
    Usage:
    >>> data_streamer = DataStreamer(
            baskets_streamer, coupon_products_streamer, coupon_values_streamer, 
            time_window
        )
    
    >>> for history, frequencies, coupons, purchases in data_streamer:
            model.train(history, frequencies, coupons, purchases)
    
    >>> data_streamer.reset() # reset iterators if needed
    >>> data_streamer.close() # close connections once done

    To make predictions:
    >>> data_streamer.enter_prediction_mode(week=90)
    >>> data_streamer.last_shopper = 2000   

    >>> for history, frequencies, coupons, _ in data_streamer:
            coupons[:, :-1] = fill_coupons()
            model.predict(history, frequencies, coupons)

    """
    shopper_baskets: ShopperData
    shopper_coupon_products: ShopperData
    shopper_coupon_values: ShopperData

    shopper_purchase_history: np.ndarray
    shopper_coupon_history: np.ndarray

    def __init__(self, 
        baskets_streamer: ShopperDataStreamer,
        coupon_products_streamer: ShopperDataStreamer,
        coupon_values_streamer: ShopperDataStreamer,
        time_window_recent_history: int = 5,
        time_window_extended_history: int = 25,
        dimension_extended_history: int = 5,
        #first_week: int = 0,
        last_week: Optional[int] = None,
        last_shopper: Optional[int] = None,
        ) -> None:
        
        self.baskets_streamer = baskets_streamer
        self.coupon_products_streamer = coupon_products_streamer
        self.coupon_values_streamer = coupon_values_streamer

        self.TW_RECENT = time_window_recent_history
        self.TW_EXTENDED = time_window_extended_history
        self.DIM_EXTENDED = dimension_extended_history

        self.first_week = self.TW_RECENT + self.TW_EXTENDED*self.DIM_EXTENDED - 1
        self.last_week = set_limit(self.baskets_streamer.max_week, last_week)
        self.last_shopper = set_limit(self.baskets_streamer.max_shopper, last_shopper)

        self.NR_PRODUCTS = self.baskets_streamer.max_value + 1

        self.prediction_mode = False
        self.__reset_iterator_positions()

    def reset(self) -> None:
        """
        Reset all iterators.
        """
        self.baskets_streamer.reset()
        self.coupon_products_streamer.reset()
        self.coupon_values_streamer.reset()
        self.__reset_iterator_positions()

    def __reset_iterator_positions(self):
        self._active = False
        self._current_shopper = 0
        self._current_week = self.first_week

    def close(self) -> None:
        """
        Close all connections.
        """
        self.baskets_streamer.close()
        self.coupon_products_streamer.close()
        self.coupon_values_streamer.close()

    def enter_prediction_mode(self, week: int = 90) -> None:
        """
        Change the iterator mode to prediciton for a selected week.
        """
        self.prediction_mode = True
        self.first_week = week - 1 
        self.last_week = week - 1
        self.reset()

    def __iter__(self) -> DataStreamer:
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Iterator loops over shoppers and weeks and returns (for a shopper `s` and  week `t`):

        - recent purchase history, binary encoded, last `time_window` weeks before `t+1` (not incl.)
        - extended purchase history, frequencies of the preceding `time_window`*`frequency_par` weeks
        - recent coupon history, (0, 1)-encoded, last `time_window` weeks and `t+1` (incl.)
        - purchases in week `t+1` (prediction target)

        All matrices have products as row-dimension and weeks as column-dimension.

        E.g., for `time_window=5`, `frequency_par=5` and `t=89`:
        - recent purchase history in weeks `85, 86, 87, 88, 89`
        - extended purchase history in weeks `60-64, 65-69, 70-74, 75-79, 80-84` (avg frequencies)
        - recent coupon history in weeks `85, 86, 87, 88, 89, 90`
        - purchases in week `90`

        # TODO improve this text

        """
        if not self._active:
            self._query_next_shopper()
            self._active = True

        history = self._get_recent_history(
            week_from = self._current_week - self.TW_RECENT + 1,
            week_to = self._current_week
        )
        frequencies = self._get_extended_history(
            week_from = self._current_week - self.TW_RECENT - self.TW_EXTENDED*self.DIM_EXTENDED + 1,
            week_to = self._current_week - self.TW_RECENT,
            agg_by = self.DIM_EXTENDED
        )
        coupons = self._get_coupons(
            week_from = self._current_week - self.TW_RECENT + 1,
            week_to = self._current_week + 1
        )
        purchases = self._get_true_purchases(
            week = self._current_week + 1
        )

        assert history.shape == (self.NR_PRODUCTS, self.TW_RECENT)
        assert frequencies.shape == (self.NR_PRODUCTS, self.DIM_EXTENDED)
        assert coupons.shape == (self.NR_PRODUCTS, self.TW_RECENT+1)
        assert purchases.shape == (self.NR_PRODUCTS, )
        
        if self._current_week >= (self.last_week - 1):
            # if we reached the last week for the current shopper, load next shopper
            self._current_week = self.first_week
            self._query_next_shopper()
        else:
            # else get ready to return next week
            self._current_week += 1

        return history, frequencies, coupons, purchases

    def _query_next_shopper(self) -> None:
        """
        Queries shopper data (baskets, coupon products, and coupon values)
        for the next shopper and stores it in memory.
        """
        self.shopper_baskets = next(self.baskets_streamer)
        self.shopper_coupon_products = next(self.coupon_products_streamer)
        self.shopper_coupon_values = next(self.coupon_values_streamer)

        if self.shopper_baskets.shopper > self.last_shopper:
            raise StopIteration(f'Exceeded limit of {self.last_shopper} shoppers.')

        if not (self.shopper_baskets.shopper 
            == self.shopper_coupon_products.shopper 
            == self.shopper_coupon_values.shopper
        ):
            raise ValueError(
                f'internal integrity check failed at shopper combination {self.shopper_baskets.shopper} '
                f'{self.shopper_coupon_products.shopper} {self.shopper_coupon_values.shopper}')

        self._current_shopper = self.shopper_baskets.shopper

        # construct shopper purchase and coupon history

        self.shopper_purchase_history = np.zeros((self.NR_PRODUCTS, self.last_week+1))
        self.shopper_coupon_history = np.zeros((self.NR_PRODUCTS, self.last_week+1))
        
        for week in range(self.last_week+1):

            week_products = self.shopper_baskets.get(week=week)
            for product in week_products:
                self.shopper_purchase_history[product, week] = 1

            week_discounts_zipper = zip(
                self.shopper_coupon_products.get(week=week),
                self.shopper_coupon_values.get(week=week)
            )
            for product, discount in week_discounts_zipper:
                self.shopper_coupon_history[product, week] = discount / 100

    def _get_true_purchases(self, week: int) -> np.ndarray:
        """
        Returns a {0,1}-array with products purchased by the in-memory shopper in a given `week`.
        """
        if self.prediction_mode:
            return np.zeros((self.NR_PRODUCTS, ))
        else:
            return self.shopper_purchase_history[:, week]

    def _get_recent_history(self, week_from: int, week_to: int) -> np.ndarray:
        """
        Returns a {0,1}-matrix with products purchased by the in-memory shopper in weeks (incl.)
        `week_from` -- `week_to`
        """
        return self.shopper_purchase_history[:, week_from:week_to+1]

    def _get_extended_history(self, week_from: int, week_to: int, agg_by: int) -> np.ndarray:
        """
        Returns a [0,1]-matrix with product purchase frequencies by the in-memory shopper in weeks (incl.)
        `week_from` -- `week_to`, with each `agg_by` weeks aggregated (avg) into one column
        """
        extended_history = np.zeros((self.NR_PRODUCTS, self.DIM_EXTENDED))

        #history = self.shopper_purchase_history[:, week_from:week_to+1]

        for column in range(self.DIM_EXTENDED):
            extended_history[:, column] = (
                self.shopper_purchase_history[:, week_from:week_from+self.TW_EXTENDED]
                .sum(axis=1) / self.TW_EXTENDED
            )
            week_from = week_from + self.TW_EXTENDED

        return extended_history

    def _get_coupons(self, week_from: int, week_to: int) -> np.ndarray:
        """
        Returns a [0,1]-matrix with coupons given to the in-memory shopper in weeks (incl.)
        `week_from` -- `week_to`
        """
        if self.prediction_mode:
            coupons = np.zeros((self.NR_PRODUCTS, self.TW_RECENT+1))
            coupons[:, :self.TW_RECENT] = self.shopper_coupon_history[:, week_from:week_to]
        else:
            coupons = self.shopper_coupon_history[:, week_from:week_to+1]
        return coupons

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__} at shopper={self._current_shopper}/{self.last_shopper+1} '
            f'and week={self._current_week}/{self.last_week+1}')


class BatchStreamer:
    """
    A wrapper around `DataStreamer` to obtain mini-batches instead of single observations.

    Usage:
    >>> batch_streamer = BatchStreamer(data_streamer, batch_size)
    >>> for _ in batch_streamer:
            pass
    """

    def __init__(self, data_streamer: DataStreamer, batch_size: int):
        self.data_streamer = data_streamer
        self.batch_size = batch_size

    def __iter__(self) -> BatchStreamer:
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        """
        nr_products = self.data_streamer.NR_PRODUCTS

        recent_history = np.zeros((self.batch_size, nr_products, self.data_streamer.TW_RECENT))
        extended_history = np.zeros((self.batch_size, nr_products, self.data_streamer.DIM_EXTENDED))
        coupons = np.zeros((self.batch_size, nr_products, self.data_streamer.TW_RECENT+1))
        purchases = np.zeros((self.batch_size, nr_products))

        for i in range(self.batch_size):
            try:
                h, f, c, p = next(self.data_streamer)
                recent_history[i,:,:] = h
                extended_history[i,:,:] = f
                coupons[i,:,:] = c
                purchases[i,:] = p
            except StopIteration:
                raise StopIteration
        
        return recent_history, extended_history, coupons, purchases


if __name__ == '__main__':

    baskets_streamer = ShopperDataStreamer('baskets.csv')
    coupon_products_streamer = ShopperDataStreamer('coupon_products.csv')
    coupon_values_streamer = ShopperDataStreamer('coupon_values.csv')

    data_streamer = DataStreamer(
        baskets_streamer=baskets_streamer,
        coupon_products_streamer=coupon_products_streamer,
        coupon_values_streamer=coupon_values_streamer,
        time_window_recent_history=5,
        time_window_extended_history=5,
        dimension_extended_history=5
    )
    batch_streamer = BatchStreamer(
        data_streamer=data_streamer,
        batch_size=10
    )

    # to make predictions:
    # data_streamer.enter_prediction_mode()
    # data_streamer.last_shopper = 2000

    # test integrity
    #history0, frequencies0, _, purchases1 = next(data_streamer)
    #history1, *_ = next(data_streamer)

    #assert sum(sum(history0)) > 0 # type: ignore
    #assert all(history0[:,0] == frequencies0)
    #assert all(history1[:,0] == purchases1)
    #data_streamer.reset()
