
from __future__ import annotations
from datetime import time

import sys
from typing import Optional, Tuple
import numpy as np

sys.path.append('.')

from models.shopper_data import ShopperDataStreamer, ShopperData


class DataStreamer:
    """ Data streamer

    Kwargs:
        baskets_streamer, coupon_products_streamer, coupon_values_streamer: input `ShopperDataStreamer`s
        time_window_recent_history:     time window (col dimension) for the recent purchase history matrix
        time_window_extended_history:   time window for each column of the extended purchase history matrix
        dimension_extended_history:     column dimension for the extended purchase history matrix
        first_week:                     first week to be predicted
        last_week:                      last week to be predicted
        last_shopper:                   last shopper to be included
        after_last_week:                set to True when making final predictions

    Usage:
    >>> data_streamer = DataStreamer(
            baskets_streamer, coupon_products_streamer, coupon_values_streamer, 
            time_window
        )
    
    >>> for history, frequencies, coupons, purchases in data_streamer:
            model.train(history, frequencies, coupons, purchases)
    
    >>> data_streamer.reset() # reset iterators if needed
    >>> data_streamer.close() # close connections once done

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
        first_week: Optional[int] = None,
        last_week: Optional[int] = None,
        last_shopper: Optional[int] = None,
        after_last_week: bool = False
        ) -> None:
        
        self.baskets_streamer = baskets_streamer
        self.coupon_products_streamer = coupon_products_streamer
        self.coupon_values_streamer = coupon_values_streamer

        self.NR_PRODUCTS = self.baskets_streamer.max_value + 1
        self.TW_RECENT = time_window_recent_history
        self.TW_EXTENDED = time_window_extended_history
        self.DIM_EXTENDED = dimension_extended_history
        
        weeks_to_burn = self.TW_RECENT + self.TW_EXTENDED * self.DIM_EXTENDED
        self.first_week = weeks_to_burn - 1
        self.last_week = self.baskets_streamer.max_week
        self.last_shopper = self.baskets_streamer.max_shopper
        
        if first_week:
            assert first_week > weeks_to_burn
            assert first_week < self.last_week
            self.first_week = first_week

        if last_week:
            assert last_week > 0
            assert last_week < self.last_week
            self.last_week = last_week

        if last_shopper:
            assert last_shopper > 0
            assert last_shopper < self.last_shopper
            self.last_shopper = last_shopper

        # for final predictions
        if after_last_week:
            self.prediction_mode = True
            self.first_week = self.last_week
        else:
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
        #print(self) # uncode for debugging

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
            f'and week={self._current_week}/{self.last_week+1} ({self._current_week+1} to be predicted)')


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

    def reset(self) -> None:
        """
        """
        self.data_streamer.reset()


if __name__ == '__main__':

    from models.config import streamer_config, TRAIN_LAST_WEEK, BATCH_SIZE 

    baskets_streamer = ShopperDataStreamer('baskets.csv')
    coupon_products_streamer = ShopperDataStreamer('coupon_products.csv')
    coupon_values_streamer = ShopperDataStreamer('coupon_values.csv')

    data = {
        'baskets_streamer': baskets_streamer,
        'coupon_products_streamer': coupon_products_streamer,
        'coupon_values_streamer': coupon_values_streamer
    }

    # weeks 0-29 are used only as history
    # train: predict weeks 30 to 79
    # test:  predict weeks 80 to 89
    # assignment: predict week 90

    data_streamer_train = DataStreamer(**data, **streamer_config, last_week=TRAIN_LAST_WEEK)
    data_streamer_test = DataStreamer(**data, **streamer_config, first_week=TRAIN_LAST_WEEK+1)
    data_streamer_final = DataStreamer(**data, **streamer_config, after_last_week=True)

    batch_streamer_train = BatchStreamer(data_streamer_train, batch_size=BATCH_SIZE)
    batch_streamer_test = BatchStreamer(data_streamer_test, batch_size=BATCH_SIZE)
    batch_streamer_final = BatchStreamer(data_streamer_final, batch_size=1)


    # TRAIN LOOP EXAMPLE
    raise NotImplementedError('code below should not be executed as is')

    class Model:
        def train(self, *args): pass
        def predict(self, *args): pass
        def evaluate(self, *args): pass

    model = Model()

    def generate_coupons(): pass
    def evaluate_uplift(_): pass

    # training
    batch_streamer_train.reset()
    for H, F, C, P in batch_streamer_train:
        model.train(H, F, C, P)
    
    # testing
    batch_streamer_test.reset()
    for H, F, C, P in batch_streamer_train:
        pred = model.predict(H, F, C)
        model.evaluate(pred, P)

    # final preds for assignment

    batch_streamer_final.reset()
    for H, F, C, _ in batch_streamer_final:
        C[:, -1] = generate_coupons()
        pred = model.predict(H, F, C)
        evaluate_uplift(pred)
