
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm


# TODO: HistoryMaker.load() fails if BasketLookup not loaded in namespace


class BasketLookup:
    """
    Lookup utility
    """
    
    def __init__(self, df: pd.DataFrame, TARGET: str = 'product') -> None:
    
        self.baskets = {}
        row_iterator = df.iterrows()
        
        for _, row in tqdm(row_iterator, total=len(df)):
            
            self.baskets.setdefault(
                row['shopper'], {}
            ).setdefault(
                row['week'], []
            ).append(
                row[TARGET]
            )
    
    def lookup(self, shopper: int, week: int) -> List[int]:
        """
        Lookup
        """
        return self.baskets[shopper][week]


class HistoryMaker:
    """
    Make history
    """
    NR_SHOPPERS: int
    NR_PRODUCTS: int
    NR_WEEKS: int
    baskets: BasketLookup
    coupon_products: BasketLookup
    coupon_discounts: BasketLookup
    prices: Dict[int, int]
        
    def fit(self, 
            baskets_df: pd.DataFrame, 
            coupons_df: pd.DataFrame,
            price_agg_method: str = 'median',
            ) -> None:
        """
        Prepare lookups based on baskets and coupons dataframes.
        """
        self.NR_SHOPPERS = len(set(baskets_df['shopper']))
        self.NR_PRODUCTS = len(set(baskets_df['product']))
        self.NR_WEEKS = len(set(baskets_df['week']))

        self.baskets = BasketLookup(baskets_df, TARGET='product')
        self.coupon_products = BasketLookup(coupons_df, TARGET='product')
        self.coupon_discounts = BasketLookup(coupons_df, TARGET='discount')
        self.prices = (
            baskets_df.groupby('product')['price'].agg(price_agg_method).to_dict()
        )

    def get_shopper_history(self, shopper: int) -> np.ndarray:
        """
        Returns a matrix of purchases history for a shopper,
        where rows=products and columns=weeks
        """
        self._assert_input(shopper=shopper, week=0)

        shopper_history = np.zeros((self.NR_PRODUCTS, self.NR_WEEKS), dtype=int)

        for week in range(self.NR_WEEKS):
            prods = self.baskets.lookup(shopper=shopper, week=week)
            for prod in prods:
                shopper_history[prod, week] += 1

        return shopper_history
    
    def get_coupon_amounts(self, shopper: int, week: int) -> np.ndarray:
        """
        Returns an array of coupon amounts given to a shopper in a certain week.
        """
        self._assert_input(shopper=shopper, week=week)

        coupons = np.zeros((self.NR_PRODUCTS), dtype=int)
        zipper = zip(
            self.coupon_products.lookup(shopper=shopper, week=week),
            self.coupon_discounts.lookup(shopper=shopper, week=week)
        )
        for prod, discount in zipper:
            coupons[prod] = discount
        
        return coupons
        
    def get_purchased(self, shopper: int, week: int) -> np.ndarray:
        """
        Returns an array of products purchased by a shopper in a certain week.
        """
        self._assert_input(shopper=shopper, week=week)

        purchased = np.zeros((self.NR_PRODUCTS), dtype=int)
        
        for prod in self.baskets.lookup(shopper=shopper, week=week):
            purchased[prod] += 1
        
        return purchased

    def get_prices(self) -> np.ndarray:
        """
        Returns an array of product prices
        """
        return np.array(list(self.prices.values()))

    def save(self, path: str = 'history.pkl') -> None:
        """
        Save fitted history to pickle.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f) 

    @classmethod
    def load(cls, path: str = 'history.pkl') -> None:
        """
        Load previously fitted history instance from a pickle file.
        """
        with open(path, "rb") as f:
            history_instance = pickle.load(f)
        return history_instance

    def _assert_input(self, shopper: int, week: int) -> None:
        assert shopper < self.NR_SHOPPERS
        assert week < self.NR_WEEKS

        


if __name__ == '__main__':

    baskets_df = pd.read_parquet('data/baskets.parquet')
    coupons_df = pd.read_parquet('data/coupons.parquet')

    history = HistoryMaker()
    history.fit(baskets_df=baskets_df, coupons_df=coupons_df)
    history.save(path='history.pkl')
