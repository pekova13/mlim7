
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from tensorflow.keras import Model


@dataclass
class Coupon:
    """ A coupon data representation """
    shopper: int
    coupon: int
    product: int
    discount: float
    is_random: bool = False


class CouponOptimizer:
    """
    # TODO write doc
    """
    WEEK: int = 90
    
    def __init__(self, 
            model: Model, 
            prices: np.ndarray, 
            discounts: Sequence[float], 
            nr_coupons: int,
            generate_random: bool = False,
            early_stop: bool = True
            ) -> None:

        self.model = model
        self.prices = prices
        self.discounts = discounts # order matters for early stopping!

        self.nr_products = self.prices.shape[0]
        self.nr_coupons = nr_coupons
        self.generate_random = generate_random
        self.early_stop = early_stop
        self._times_early_stop = 0

        self.revenue_baseline: List[float] = []
        self.revenue_optimal: List[float] = []
        self.revenue_uplift: List[float] = []
        self.assigned_coupons: List[Coupon] = []

        self._revenue_baseline_shopper: float
        self._revenue_optimal_shopper: float

    def optimize(self, 
            H: np.ndarray, F: np.ndarray, C: np.ndarray,
            shopper: int,
            #P: Optional[np.ndarray] = None
            ) -> None:
        """
        Find shopper coupons that optimally improve the expected revenue and store them in-memory.
        """
        self._revenue_baseline_shopper = self._predict_revenue(H, F, C) # expected revenue with no coupons
        self._revenue_optimal_shopper = self._revenue_baseline_shopper  # expected revenue with optimized coupons
        C = C.copy()

        for i in range(self.nr_coupons):
            C = self._optimize_coupon(H, F, C, shopper=shopper, i=i)

        self.revenue_baseline.append(self._revenue_baseline_shopper)
        self.revenue_optimal.append(self._revenue_optimal_shopper)
        self.revenue_uplift.append(self._revenue_optimal_shopper - self._revenue_baseline_shopper)

    def _optimize_coupon(self, 
            H: np.ndarray, F: np.ndarray, C: np.ndarray, 
            shopper: int, 
            i: int
            ) -> np.ndarray:
        """
        Find one coupon which optimally improves the expected revenue 
        and return the updated coupon matrix.
        """
        # create default random coupon
        coupon = self._generate_random_coupon(shopper=shopper, i=i)

        # find optimal coupon
        if not self.generate_random:

            # create mini-batches to speed up process
            tile_dims = (self.nr_products, 1, 1)
            H_repeated = np.tile(H, tile_dims)
            F_repeated = np.tile(F, tile_dims)
            C_repeated = np.tile(C, tile_dims)
            prices = np.tile(self.prices, (self.nr_products, 1))

            for discount in self.discounts:

                for product in range(self.nr_products):
                    C_repeated[product, product, -1] = discount
                
                pred_proba: np.ndarray = self.model(
                    [H_repeated, F_repeated, C_repeated], training=False
                    ).numpy() # type: ignore

                discounts = C_repeated[:, :, -1]
                pred_revenue_vector = pred_proba * prices * (1-discounts)
                assert pred_revenue_vector.shape == (self.nr_products, self.nr_products)

                pred_revenue = pred_revenue_vector.sum(axis=1) # expected revenue per discount combination
                best_product = pred_revenue.argmax()
                best_revenue = pred_revenue.max()

                if best_revenue > self._revenue_optimal_shopper:
                    #print(f'{best_revenue} - {self._revenue_optimal_shopper}')
                    # if revenue improved, replace random coupon with optimal
                    coupon = Coupon(shopper=shopper, coupon=i, product=best_product, discount=discount)
                    self._revenue_optimal_shopper = best_revenue

                elif self.early_stop and discount != self.discounts[-1]:
                    # if revenue not improved, stop iteration early
                    self._times_early_stop += 1
                    break 
        
        # save coupon, update coupon matrix for next iteration, update optimized extected revenue
        self.assigned_coupons.append(coupon)
        C[:, coupon.product, -1] = coupon.discount

        self._revenue_optimal_shopper = self._predict_revenue(H, F, C)
        
        return C

    def _predict_revenue(self, H: np.ndarray, F: np.ndarray, C: np.ndarray) -> float:
        """
        Predict purchase probabilities for a mini-batch of size 1, 
        calculate and return total expected revenue.
        """
        discounts = C[:, :, -1]
        sigmoid_proba: np.ndarray = self.model([H, F, C], training=False).numpy() # type: ignore
        assert sigmoid_proba.shape[0] == 1 # batch-size must be 1

        revenue = (np.squeeze(sigmoid_proba, axis=0) * self.prices * (1-discounts)).sum()
        return revenue

    def _generate_random_coupon(self, shopper: int, i: int) -> Coupon:
        """
        Private method to generate a random coupon for a specified `shopper` and coupon priority `i`.
        """
        return Coupon(
            shopper=shopper,
            coupon=i,
            product=random.randint(0, self.nr_products-1),
            discount=random.choice(self.discounts),
            is_random=True
        )

    def write_coupons(self, path: str) -> None:
        """
        Write a CSV file with assigned coupon index.
        """
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['shopper', 'week', 'coupon', 'product', 'discount'])
            
            for coupon in self.assigned_coupons:
                writer.writerow([coupon.shopper, self.WEEK, coupon.coupon, coupon.product, coupon.discount])

    def write_stats(self, path: str) -> None:
        """
        Write a CSV file with major stats.
        """
        random_coupons = sum(1 for coupon in self.assigned_coupons if coupon.is_random)
        nr_shoppers = len(self.revenue_baseline)

        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Baseline expected revenue', int(sum(self.revenue_baseline))])
            writer.writerow(['Optimized expected revenue', int(sum(self.revenue_optimal))])
            writer.writerow(['Baseline expected revenue p.c.', int(sum(self.revenue_baseline) / nr_shoppers)])
            writer.writerow(['Optimized expected revenue p.c.', int(sum(self.revenue_optimal) / nr_shoppers)])
            writer.writerow(['Uplift', int(sum(self.revenue_uplift))])
            writer.writerow(['Uplift %', round(100*sum(self.revenue_uplift) / sum(self.revenue_baseline), 4)])
            writer.writerow([])
            writer.writerow(['Nr. of shoppers', nr_shoppers])
            writer.writerow(['Nr. of coupons', len(self.assigned_coupons)])
            writer.writerow(['Nr. of randomized coupons', random_coupons])
            writer.writerow(['Nr. of early stops', self._times_early_stop])
            writer.writerow([])
            for discount in self.discounts:
                count = sum(1 for coupon in self.assigned_coupons if coupon.discount == discount)
                writer.writerow([discount, count])
