
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
    """
    _shopper_baskets: ShopperData
    _shopper_coupon_products: ShopperData
    _shopper_coupon_values: ShopperData

    def __init__(self, 
        baskets_streamer: ShopperDataStreamer,
        coupon_products_streamer: ShopperDataStreamer,
        coupon_values_streamer: ShopperDataStreamer,
        time_window: int = 10,
        first_week: int = 0,
        last_week: Optional[int] = None,
        last_shopper: Optional[int] = None,
        ) -> None:
        
        self.baskets_streamer = baskets_streamer
        self.coupon_products_streamer = coupon_products_streamer
        self.coupon_values_streamer = coupon_values_streamer

        self.time_window = time_window
        self.first_week = first_week
        self.last_week = set_limit(self.baskets_streamer.max_week, last_week)
        self.last_shopper = set_limit(self.baskets_streamer.max_shopper, last_shopper)

        self.NR_PRODUCTS = self.baskets_streamer.max_value + 1

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
        Returns:
        - purchase history matrix (dimension: products, weeks)
        - purchase frequency vector up until week `t` (dim: products)
        - coupon values vector in week `t` (dim: products)
        - purchases in week `t+1` (dim: products)
        """
        if not self._active:
            self._get_next_shopper()
            self._active = True

        history = self._get_history(week=self._current_week)
        frequencies = self._get_frequencies(week=self._current_week)
        coupons = self._get_coupons(week=self._current_week)
        purchases = self._get_true_purchases(week=self._current_week+1)

        if self._current_week == (self.last_week - 1):
            # if we reached the last week for the current shopper, load next shopper
            self._current_week = self.first_week
            self._get_next_shopper()
        else:
            # else get ready to return next week
            self._current_week += 1

        return history, frequencies, coupons, purchases

    def _get_next_shopper(self) -> None:
        """
        Queries shopper data (baskets, coupon products, and coupon values)
        for the next shopper and stores it in memory.
        """
        self.shopper_baskets = next(self.baskets_streamer)
        self.shopper_coupon_products = next(self.coupon_products_streamer)
        self.shopper_coupon_values = next(self.coupon_values_streamer)

        if not (self.shopper_baskets.shopper 
            == self.shopper_coupon_products.shopper 
            == self.shopper_coupon_values.shopper
        ):
            raise ValueError(
                f'internal integrity check failed at shopper combination {self.shopper_baskets.shopper} '
                f'{self.shopper_coupon_products.shopper} {self.shopper_coupon_values.shopper}')

        self._current_shopper = self.shopper_baskets.shopper

    def _get_true_purchases(self, week: int) -> np.ndarray:
        """
        Returns a {0,1}-array with products purchased by the in-memory shopper in a given `week`.
        """
        purchased = np.zeros((self.NR_PRODUCTS), dtype=int)

        for product in self.shopper_baskets.get(week=week):
            purchased[product] += 1
        
        return purchased

    def _get_coupons(self, week: int) -> np.ndarray:
        """
        Returns a [0,1]-array of coupon amounts given to a shopper in a given week.
        """
        coupons = np.zeros((self.NR_PRODUCTS), dtype=int)
        zipper = zip(
            self.shopper_coupon_products.get(week=week),
            self.shopper_coupon_values.get(week=week)
        )
        for prod, discount in zipper:
            coupons[prod] = discount
        
        return coupons / 100

    def _get_history(self, week: int) -> np.ndarray:
        """
        Returns a {0,1}-matrix of purchases by a shopper in the given week.
        """
        return self._calc_history(
            week_from=week - self.time_window + 1,
            week_to=week
        )

    def _get_frequencies(self, week: int) -> np.ndarray:
        """
        Returns a [0,1]-array of purchase frequencies given to a shopper until (incl.) the given week.
        """
        return self._calc_history( # type: ignore
            week_from=0,
            week_to=week
        ).sum(axis=1) / (week+1)

    def _calc_history(self, week_from: int, week_to: int) -> np.ndarray:
        """
        Returns a 0-1 matrix with history of shopper purchases for weeks from `week_from`
        to `week_to` (incl.), with rows=products and cols=weeks.

        A zero padding is appended for weeks preceding the first known week.
        """
        shopper_history = np.zeros((self.NR_PRODUCTS, week_to-week_from+1), dtype=int)

        for column, week in enumerate(range(week_to, week_from-1, -1)):

            if week < 0:
                continue
            
            prods = self.shopper_baskets.get(week=week)
            for prod in prods:
                shopper_history[prod, column] = 1

        return shopper_history


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
        time_window = self.data_streamer.time_window

        history = np.zeros((self.batch_size, nr_products, time_window))
        frequencies = np.zeros((self.batch_size, nr_products))
        coupons = np.zeros((self.batch_size, nr_products))
        purchases = np.zeros((self.batch_size, nr_products))

        for i in range(self.batch_size):
            try:
                h, f, c, p = next(self.data_streamer)
                history[i,:,:] = h
                frequencies[i,:] = f
                coupons[i,:] = c
                purchases[i,:] = p
            except StopIteration:
                raise StopIteration
        
        return history, frequencies, coupons, purchases


if __name__ == '__main__':

    baskets_streamer = ShopperDataStreamer('baskets.csv')
    coupon_products_streamer = ShopperDataStreamer('coupon_products.csv')
    coupon_values_streamer = ShopperDataStreamer('coupon_values.csv')

    data_streamer = DataStreamer(
        baskets_streamer=baskets_streamer,
        coupon_products_streamer=coupon_products_streamer,
        coupon_values_streamer=coupon_values_streamer,
        time_window=10
    )
    batch_streamer = BatchStreamer(
        data_streamer=data_streamer,
        batch_size=10
    )

    # test integrity
    #history0, frequencies0, _, purchases1 = next(data_constructor)
    #history1, *_ = next(data_constructor)

    #assert sum(sum(history0)) > 0 # type: ignore
    #assert all(history0[:,0] == frequencies0)
    #assert all(history1[:,0] == purchases1)
    #data_constructor.reset()
