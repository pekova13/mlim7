
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
            generate_random: bool = False
            ) -> None:

        self.model = model
        self.prices = prices
        self.discounts = discounts

        self.nr_products = self.prices.shape[0]
        self.nr_coupons = nr_coupons
        self.generate_random = generate_random

        self.revenue_baseline: List[float] = []
        self.revenue_optimal: List[float] = []
        self.revenue_uplift: List[float] = []
        self.assigned_coupons: List[Coupon] = []

        self._revenue_baseline_shopper: float
        self._revenue_optimal_shopper: float

    def optimize(self, H: np.ndarray, F: np.ndarray, C: np.ndarray, shopper: int) -> None:
        """
        Find shopper coupons that optimally improve the expected revenue and store them in-memory.
        """
        self._revenue_baseline_shopper = self._predict_revenue(H, F, C) # no coupons
        self._revenue_optimal_shopper = self._revenue_baseline_shopper

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
        C_updated = C.copy()
        coupon = self._generate_random_coupon(shopper=shopper, i=i)

        # if looking for optimal coupon and not just randomizing
        if not self.generate_random:

            for discount in self.discounts:
                for product in range(self.nr_products):

                    C_updated[:, -1] = C[:, -1]         # reset coupons to original state
                    C_updated[product, -1] = discount   # try next coupon

                    revenue = self._predict_revenue(H, F, C_updated)

                    if revenue > self._revenue_optimal_shopper:
                        self._revenue_optimal_shopper = revenue
                        coupon = Coupon(shopper=shopper, coupon=i, product=product, discount=discount)
        
        # if no optimal coupon found
        if self.generate_random or coupon.is_random:

            C_updated[:, -1] = C[:, -1] 
            C_updated[coupon.product, -1] = coupon.discount
            self._revenue_optimal_shopper = self._predict_revenue(H, F, C_updated)
        
        self.assigned_coupons.append(coupon)
        return C_updated

    def _predict_revenue(self, H: np.ndarray, F: np.ndarray, C: np.ndarray) -> float:
        """
        Predict purchase probabilities, calculate and return total expected revenue.
        """
        sigmoid_proba = self.model([H, F, C], training=False)
        # TODO reshape
        return sum(sigmoid_proba * self.prices)

    def _generate_random_coupon(self, shopper: int, i: int) -> Coupon:
        """
        Private method to generate a random coupon for a specified `shopper` and coupon priority `i`.
        """
        return Coupon(
            shopper=shopper,
            coupon=i,
            product=random.randint(0, self.nr_products),
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
        random_coupons = sum([1 for coupon in self.assigned_coupons if coupon.is_random])

        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Baseline revenue', sum(self.revenue_baseline)])
            writer.writerow(['Optimized revenue', sum(self.revenue_optimal)])
            writer.writerow(['Uplift', sum(self.revenue_uplift)])
            writer.writerow(['Nr. of randomized coupons', random_coupons])
