"""
This module is used to initiate data/batch streamers to reduce code clutter in other places.
"""

from models.data_streamer import BatchStreamer, DataStreamer
from models.shopper_data import ShopperDataStreamer

from steps import config

baskets_streamer = ShopperDataStreamer(config.BASKETS_PATH)
coupon_products_streamer = ShopperDataStreamer(config.COUPON_PRODUCTS_PATH)
coupon_values_streamer = ShopperDataStreamer(config.COUPON_VALUES_PATH)

data = {
    'baskets_streamer': baskets_streamer, 
    'coupon_products_streamer': coupon_products_streamer,
    'coupon_values_streamer': coupon_values_streamer
}

data_streamer_train = DataStreamer(**data, **config.streamer_parms, last_week=config.TRAIN_LAST_WEEK)
data_streamer_test = DataStreamer(**data, **config.streamer_parms, first_week=config.TRAIN_LAST_WEEK+1)
data_streamer_final = DataStreamer(**data, **config.streamer_parms, after_last_week=True)
    
batch_streamer_train = BatchStreamer(data_streamer_train, batch_size=config.BATCH_SIZE)
batch_streamer_test = BatchStreamer(data_streamer_test, batch_size=config.BATCH_SIZE)
batch_streamer_final = BatchStreamer(data_streamer_final, batch_size=1) # 1 shopper at a time
