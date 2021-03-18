"""
Step 3

"""
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.coupon_assignment import CouponOptimizer
from models.model import build_model
from steps.load_data import batch_streamer_final, data_streamer_final
from steps import config


if __name__ == '__main__':

    model = build_model(**config.model_parms)
    model.load_weights(config.MODEL_WEIGHTS_PATH)

    prices: np.ndarray = pd.read_csv(config.PRICES_PATH).price.to_numpy()

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

    for H, F, C, _ in tqdm(batch_streamer_final):
        coupon_optimizer.optimize(H, F, C, shopper=data_streamer_final._current_shopper)
        coupon_randomizer.optimize(H, F, C, shopper=data_streamer_final._current_shopper)

    coupon_optimizer.write_coupons(config.COUPONS_PATH)
    coupon_optimizer.write_stats(config.COUPONS_STATS_PATH)

    coupon_randomizer.write_coupons(config.COUPONS_RANDOM_PATH)
    coupon_randomizer.write_stats(config.COUPONS_STATS_RANDOM_PATH)

