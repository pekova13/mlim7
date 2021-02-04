
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

NR_WEEKS = 90
NR_PRODUCTS = 250


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
    baskets: BasketLookup
    coupon_products: BasketLookup
    coupon_discounts: BasketLookup
    prices: Dict[int, int]

    def __init__(self, load_from_path: Optional[str] = None) -> None:
        if load_from_path:
            self.load(path=load_from_path)
        
    def fit(self, 
            baskets_df: pd.DataFrame, 
            coupons_df: pd.DataFrame,
            price_agg_method: str = 'median',
            ) -> None:
        """
        Prepare lookups based on baskets and coupons dataframes.
        """
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
        shopper_history = np.zeros((NR_PRODUCTS, NR_WEEKS), dtype=int)

        for week in range(NR_WEEKS):
            prods = self.baskets.lookup(shopper=shopper, week=week)
            for prod in prods:
                shopper_history[prod, week] += 1

        return shopper_history
    
    def get_coupon_info(self, shopper: int, week: int) -> np.ndarray:
        """
        Returns an array of coupon amounts given to a shopper in a certain week.
        """
        coupons = np.zeros((NR_PRODUCTS), dtype=int)
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
        purchased = np.zeros((NR_PRODUCTS), dtype=int)
        
        for prod in self.baskets.lookup(shopper=shopper, week=week):
            purchased[prod] += 1
        
        return purchased

    def get_prices(self) -> np.ndarray:
        """
        Returns an array of product prices
        """
        return np.ndarray(list(self.prices))

    def save(self, path: str = 'history.pkl') -> None:
        """
        Save fitted history to pickle.
        """
        contents = (self.baskets, self.coupon_products, self.coupon_discounts, self.prices)
        with open(path, "wb") as f:
            pickle.dump(contents, f) 

    def load(self, path: str = 'history.pkl') -> None:
        """
        Load previously fitted history from pickle.
        """
        with open(path, "rb") as f:
            contents = pickle.load(f)
        (self.baskets, self.coupon_products, self.coupon_discounts, self.prices) = contents


if __name__ == '__main__':

    baskets_df = pd.read_parquet('data/baskets.parquet')
    coupons_df = pd.read_parquet('data/coupons.parquet')

    history = HistoryMaker()
    history.fit(baskets_df=baskets_df, coupons_df=coupons_df)
    history.save(path='history.pkl')



