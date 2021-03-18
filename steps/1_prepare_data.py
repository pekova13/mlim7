"""
Step 1: 
prepare data from parquets into a easier to work with format
"""

import sys
sys.path.append('.')

import pandas as pd

from models.shopper_data import ShopperDataWriter
from steps import config


if __name__ == '__main__':

    baskets_df = pd.read_parquet(config.BASKETS_PARQUET_PATH)
    coupons_df = pd.read_parquet(config.COUPONS_PARQUET_PATH)

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
