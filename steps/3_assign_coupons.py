"""
Step 3: optimize coupons

config:
- LIMIT_SHOPPERS_COUPONS: for how many shoppers to optimize coupons
- NR_COUPONS:             how many coupons to select for each shopper
- DISCOUNTS:              possible discount values to select from (order matters if early stopping)
- COUPONS_EARLY_STOP:     if True, the first best discount value is selected for each shopper
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.coupon_assignment import CouponOptimizer
from models.model import build_model
from steps.load_data import batch_streamer_90
import config


if __name__ == '__main__':

    model = build_model(**config.model_parms)
    _ = model.load_weights(config.MODEL_WEIGHTS_PATH)

    prices: np.ndarray = pd.read_csv(config.PRICES_PATH).price.to_numpy()

    assert prices.shape[0] == batch_streamer_90.nr_products

    # make predictions for week 90
    coupon_optimizer = CouponOptimizer(
        model=model,
        prices=prices,
        **config.coupon_parms,
        generate_random=False
    )
    coupon_randomizer = CouponOptimizer(
        model=model,
        prices=prices,
        **config.coupon_parms,
        generate_random=True
    )

    np.random.seed(config.LIMIT_SHOPPERS_COUPONS)
    batch_streamer_90.reset()

    for H, F, C, _ in tqdm(batch_streamer_90, total=config.LIMIT_SHOPPERS_COUPONS):
        coupon_optimizer.optimize(H, F, C, shopper=batch_streamer_90.current_shopper)
        coupon_randomizer.optimize(H, F, C, shopper=batch_streamer_90.current_shopper)

        #if batch_streamer_90.current_shopper % 50 == 0:
        #    coupon_optimizer.write_stats(config.COUPONS_PRED_STATS_OPTIMAL_PATH)
        #    coupon_randomizer.write_stats(config.COUPONS_PRED_STATS_RANDOM_PATH)


    coupon_optimizer.write_coupons(config.COUPONS_PRED_OPTIMAL_PATH)
    coupon_optimizer.write_stats(config.COUPONS_PRED_STATS_OPTIMAL_PATH)

    coupon_randomizer.write_coupons(config.COUPONS_PRED_RANDOM_PATH)
    coupon_randomizer.write_stats(config.COUPONS_PRED_STATS_RANDOM_PATH)

    batch_streamer_90.close() # close all connections
