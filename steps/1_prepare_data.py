"""
Step 1: prepare data

Transform data from parquet files into the custom ShopperData format.
(Why do this? It enables much quicker data streaming when working with big data.)

config:
- LIMIT_SHOPPERS_DATA_PREP: if -1, parquet files are fully transformed into ShopperData format
                            (this can take >2 hours depending on your machine)
                            if some positive integer value, then only first N shoppers will be taken

- BASKETS_PARQUET_PATH: path to baskets.parquet
- COUPONS_PARQUET_PATH: path to coupons.parquet
- BASKETS_PATH:         path where baskets ShopperData will be saved
- COUPON_PRODUCTS_PATH: path where coupon product ShopperData will be saved
- COUPON_VALUES_PATH:   path where coupon value ShopperData will be saved
- PRICES_PATH:          path where median prices will be saved
"""

import sys
from typing import Sequence
sys.path.append('.')

import pandas as pd

from models.shopper_data import ShopperDataWriter
import config


def assert_df_columns(df: pd.DataFrame, columns: Sequence[str]):
    assert all((column in df.columns) for column in columns)


if __name__ == '__main__':

    baskets_df = pd.read_parquet(config.BASKETS_PARQUET_PATH)
    coupons_df = pd.read_parquet(config.COUPONS_PARQUET_PATH)

    assert_df_columns(baskets_df, ('shopper', 'week', 'product', 'price'))
    assert_df_columns(coupons_df, ('shopper', 'week', 'product', 'discount'))

    if config.LIMIT_SHOPPERS_DATA_PREP > 0:

        print(f'sample limited to {config.LIMIT_SHOPPERS_DATA_PREP} shoppers')
        baskets_df = baskets_df[baskets_df['shopper'] < config.LIMIT_SHOPPERS_DATA_PREP]
        coupons_df = coupons_df[coupons_df['shopper'] < config.LIMIT_SHOPPERS_DATA_PREP]

        assert isinstance(baskets_df, pd.DataFrame) # for pylance
        assert isinstance(coupons_df, pd.DataFrame)

    ShopperDataWriter().fit(df=baskets_df, target='product').write(config.BASKETS_PATH)
    ShopperDataWriter().fit(df=coupons_df, target='product').write(config.COUPON_PRODUCTS_PATH)
    ShopperDataWriter().fit(df=coupons_df, target='discount').write(config.COUPON_VALUES_PATH)

    prices = baskets_df.groupby('product')['price'].agg('median')
    prices.to_csv(config.PRICES_PATH, index=False) 
