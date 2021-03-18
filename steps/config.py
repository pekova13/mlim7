"""

"""

### PATHS ###
import os
#data_path = lambda path: os.path.join('data', path)

def data_path(path: str) -> str:
    """ """
    return os.path.join('data', path)

# input files
BASKETS_PARQUET_PATH = data_path('baskets.parquet')
COUPONS_PARQUET_PATH = data_path('coupons.parquet')

# shopper data streamers
BASKETS_PATH = data_path('baskets.csv')
COUPON_PRODUCTS_PATH = data_path('coupon_products.csv')
COUPON_VALUES_PATH = data_path('coupon_values.csv')

PRICES_PATH = data_path('prices.csv')

# trained model weights
MODEL_WEIGHTS_PATH = data_path('model_weights.pkl')


### PARAMETERS ###

### Step 1: Data preparation
LIMIT_SHOPPERS_DATA_PREP = -1


### Step 2: Model training
TIME_WINDOW_RECENT_HISTORY = 5
TIME_WINDOW_EXTENDED_HISTORY = 5
DIMENSION_EXTENDED_HISTORY = 5
LIMIT_SHOPPERS_TRAINING = 5999

TRAIN_LAST_WEEK = 79

# weeks 0-29 are used only as history
# train: predict weeks 30 to 79
# test:  predict weeks 80 to 89
# assignment: predict week 90

LEARNING_RATE = 1e3
EPOCHS = 10
BATCH_SIZE = 10

streamer_parms= {
    'time_window_recent_history': TIME_WINDOW_RECENT_HISTORY,
    'time_window_extended_history': TIME_WINDOW_EXTENDED_HISTORY,
    'dimension_extended_history': DIMENSION_EXTENDED_HISTORY,
    'last_shopper': LIMIT_SHOPPERS_TRAINING
}

model_parms = {
    'NR_PRODUCTS': 250,
    'HISTORY_DIM': TIME_WINDOW_RECENT_HISTORY,
    'FREQUENCY_DIM': DIMENSION_EXTENDED_HISTORY,
    'kernel_size': 3,
    'nr_filters': 18,
}


### Step 3: Assign coupons
coupon_parms = {
    'nr_coupons': 5,
    'discounts': (0.15, 0.2, 0.25, 0.3)
}
