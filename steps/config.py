"""
This module contains all configurable parameters.
See each step for more details.
"""

# Step 1: Data preparation
LIMIT_SHOPPERS_DATA_PREP = -1       # -1 to keep all

# Step 2: Model training - model architecture
TIME_WINDOW_RECENT_HISTORY = 5
TIME_WINDOW_EXTENDED_HISTORY = 5
DIMENSION_EXTENDED_HISTORY = 5
KERNEL_SIZE = 3
NR_FILTERS = 18

# Step 2: Model training - model hyperparameters
LIMIT_SHOPPERS_TRAINING = 5999
LEARNING_RATE = 1e3
BATCH_SIZE = 10
NR_EPOCHS = 10
TRAIN_LAST_WEEK = 79

# Step 3: Assign coupons
LIMIT_SHOPPERS_COUPONS = 2000
NR_COUPONS = 5
DISCOUNTS = (0.3, 0.25, 0.2, 0.15)
COUPONS_EARLY_STOP = True


# parms dicts
streamer_train_parms= {
    'time_window_recent_history': TIME_WINDOW_RECENT_HISTORY,
    'time_window_extended_history': TIME_WINDOW_EXTENDED_HISTORY,
    'dimension_extended_history': DIMENSION_EXTENDED_HISTORY,
    'last_shopper': LIMIT_SHOPPERS_TRAINING,
}
streamer_coupon_parms= {
    'time_window_recent_history': TIME_WINDOW_RECENT_HISTORY,
    'time_window_extended_history': TIME_WINDOW_EXTENDED_HISTORY,
    'dimension_extended_history': DIMENSION_EXTENDED_HISTORY,
    'last_shopper': LIMIT_SHOPPERS_COUPONS,
}
model_parms = {
    'HISTORY_DIM': TIME_WINDOW_RECENT_HISTORY,
    'FREQUENCY_DIM': DIMENSION_EXTENDED_HISTORY,
    'kernel_size': KERNEL_SIZE,
    'nr_filters': NR_FILTERS,
}
coupon_parms = {
    'nr_coupons': NR_COUPONS,
    'discounts': DISCOUNTS,
    'early_stop': COUPONS_EARLY_STOP,
    'week': 90
}


# import/export paths
import os

_DATA_FOLDER = 'data'
_OUT_FOLDER = 'coupons'

if not os.path.exists(_DATA_FOLDER):
    os.mkdir(_DATA_FOLDER)
if not os.path.exists(_OUT_FOLDER):
    os.mkdir(_OUT_FOLDER)

def data_path(path: str) -> str:
    return os.path.join(_DATA_FOLDER, path)
def out_path(path: str) -> str:
    return os.path.join(_OUT_FOLDER, path)

# input files
BASKETS_PARQUET_PATH = data_path('baskets.parquet')
COUPONS_PARQUET_PATH = data_path('coupons.parquet')

# shopper data streamers
BASKETS_PATH = data_path('baskets.csv')
COUPON_PRODUCTS_PATH = data_path('coupon_products.csv')
COUPON_VALUES_PATH = data_path('coupon_values.csv')

PRICES_PATH = data_path('prices.csv')

# trained model weights
MODEL_WEIGHTS_PATH = data_path('CNN_weights')

# coupons
COUPONS_PRED_STATS_OPTIMAL_PATH = out_path('coupons_pred_stats_optimal.csv')
COUPONS_PRED_STATS_RANDOM_PATH = out_path('coupons_pred_stats_random.csv')

COUPONS_PRED_OPTIMAL_PATH = out_path('coupons_pred_optimal.csv')
COUPONS_PRED_RANDOM_PATH = out_path('coupons_pred_random.csv')
