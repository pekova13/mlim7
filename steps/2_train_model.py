"""
Step 2: train model

Model inputs:
H = recent purchase history, last TIME_WINDOW_RECENT_HISTORY weeks          (250x5)
F = extended purchase history, DIMENSION_EXTENDED_HISTORY columns containing average purchase 
    frequencies for TIME_WINDOW_EXTENDED_HISTORY weeks each                 (250x5)
C = coupons, last TIME_WINDOW_RECENT_HISTORY weeks + prediction week        (250x6)

Target:
P = product purchases in prediction week                                    (250x1)

Train: predicting weeks until TRAIN_LAST_WEEK                               (30-79)
Test:  predicting weeks from TRAIN_LAST_WEEK+1                              (80-89)

Hyperparameters:
NR_EPOCHS, LEARNING_RATE, KERNEL_SIZE, NR_FILTERS, BATCH_SIZE
LIMIT_SHOPPERS_TRAINING: consider only first N shoppers to speed up training
"""

import sys
sys.path.append('.')

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

from models.model import build_naive_model, build_model, train_model
from steps.load_data import batch_streamer_train, batch_streamer_test, NR_PRODUCTS
from steps import config


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-n', '--naive', dest='naive', action='store_true')
    parser.add_argument('fix_value', nargs='?', const=None, type=float)
    kwargs = parser.parse_args()

    if kwargs.naive:
        model = build_naive_model(fix_value=kwargs.fix_value)
    else:
        model = build_model(**config.model_parms, NR_PRODUCTS=NR_PRODUCTS)
    
    print(model.summary())

    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    loss_fn = BinaryCrossentropy(from_logits=False)

    model = train_model(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        batch_streamer_train=batch_streamer_train,
        batch_streamer_test=batch_streamer_test,
        epochs=1 if kwargs.naive else config.NR_EPOCHS
    )

    if not kwargs.naive:
        model.save_weights(config.MODEL_WEIGHTS_PATH)

    batch_streamer_train.close() # will close all file connections
