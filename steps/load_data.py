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

data_streamer_train = DataStreamer(**data, **config.streamer_train_parms, last_week=config.TRAIN_LAST_WEEK)
data_streamer_test = DataStreamer(**data, **config.streamer_train_parms, first_week=config.TRAIN_LAST_WEEK+1)
    
batch_streamer_train = BatchStreamer(data_streamer_train, batch_size=config.BATCH_SIZE)
batch_streamer_test = BatchStreamer(data_streamer_test, batch_size=config.BATCH_SIZE)

# TODO make par predict_week = x-> 
# first_week = x
# last_week = x+1
"""
data_streamer_89 = DataStreamer(**data, **config.streamer_coupon_parms, first_week=89)
batch_streamer_89 = BatchStreamer(data_streamer_89, batch_size=1) # 1 shopper at a time
"""

data_streamer_90 = DataStreamer(**data, **config.streamer_coupon_parms, after_last_week=True)
batch_streamer_90 = BatchStreamer(data_streamer_90, batch_size=1) # 1 shopper at a time 

# nr of products can't be specified in config but is observed from the data
NR_PRODUCTS = data_streamer_train.NR_PRODUCTS
