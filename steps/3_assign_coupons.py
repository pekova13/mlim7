"""
Step 3

"""
from tqdm import tqdm
from models.model import build_model
from models.coupon_assignment import CouponOptimizer


if __name__ == '__main__':

    from steps import config
    from steps.load_data import batch_streamer_final, data_streamer_final

    model = build_model(**config.model_parms)
    model.load_weights(config.MODEL_WEIGHTS_PATH)

    import numpy as np
    prices = np.zeros(250)

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

    coupon_optimizer.write_coupons('coupon_index.csv')
    coupon_optimizer.write_stats('coupons_stats_optimal.csv')

    coupon_randomizer.write_coupons('coupon_index_random.csv')
    coupon_randomizer.write_stats('coupons_stats_random.csv')

