
BATCH_SIZE = 10
TRAIN_LAST_WEEK = 79

# weeks 0-29 are used only as history
# train: predict weeks 30 to 79
# test:  predict weeks 80 to 89
# assignment: predict week 90

streamer_config = {
    'time_window_recent_history': 5,
    'time_window_extended_history': 5,
    'dimension_extended_history': 5,
    'last_shopper': 1999
}