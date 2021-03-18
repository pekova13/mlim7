"""
Step 2: 
train model
"""

import sys
sys.path.append('.')

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

from models.model import build_model, train_model
from steps.load_data import batch_streamer_train, batch_streamer_test
from steps import config


if __name__ == '__main__':
    
    model = build_model(**config.model_parms)
    print(model.summary())

    # tf.keras.utils.plot_model(model)
    # requires pydot

    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    loss_fn = BinaryCrossentropy(from_logits=False)

    model = train_model(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        batch_streamer_train=batch_streamer_train,
        batch_streamer_test=batch_streamer_test,
        epochs=config.EPOCHS
    )

    model.save_weights(config.MODEL_WEIGHTS_PATH)

    batch_streamer_train.close() # will close all file connections
