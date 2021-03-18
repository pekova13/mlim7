
import numpy as np
from tensorflow import GradientTape
from tensorflow.keras import Model, layers
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

from models.data_streamer import BatchStreamer


def build_model(
        NR_PRODUCTS: int = 250,
        HISTORY_DIM: int = 5,
        FREQUENCY_DIM: int = 5,
        kernel_size: int = 3,
        nr_filters: int = 18,
        ) -> Model:
    """
    """

    input_H = layers.Input(shape=(NR_PRODUCTS, HISTORY_DIM))
    input_F = layers.Input(shape=(NR_PRODUCTS, FREQUENCY_DIM))
    input_C = layers.Input(shape=(NR_PRODUCTS, HISTORY_DIM+1))

    layer_H = layers.Conv1D(nr_filters, kernel_size)(input_H)
    layer_H = layers.LeakyReLU()(layer_H)
    layer_H = layers.MaxPooling1D()(layer_H)

    layer_F = layers.Conv1D(nr_filters, kernel_size)(input_F)
    layer_F = layers.LeakyReLU()(layer_F)
    layer_F = layers.MaxPooling1D()(layer_F)

    layer_C = layers.Conv1D(nr_filters, kernel_size)(input_C)
    layer_C = layers.LeakyReLU()(layer_C)
    layer_C = layers.MaxPooling1D()(layer_C)

    output_H_conv = layers.Flatten()(layer_H)
    output_F_conv = layers.Flatten()(layer_F)
    output_C_conv = layers.Flatten()(layer_C)

    head_H = Model(inputs=input_H, outputs=output_H_conv)
    head_F = Model(inputs=input_F, outputs=output_F_conv)
    head_C = Model(inputs=input_C, outputs=output_C_conv)

    merged = layers.concatenate([head_H.output, head_F.output, head_C.output])

    final_layer = layers.Dense(250, activation='sigmoid')(merged)

    model = Model(
        inputs=[head_H.input, head_F.input, head_C.input], 
        outputs=final_layer
    )

    return model


def train_model(
        model: Model,
        optimizer: Optimizer,
        loss_fn: Loss,
        batch_streamer_train: BatchStreamer,
        batch_streamer_test: BatchStreamer,
        epochs: int = 10,
        ) -> Model:
    """
    """
    loss_per_epoch_train = []
    loss_per_epoch_test = []

    for epoch in range(epochs):
        print(f'Current Epoch: {epoch}')

        # loss for each mini-batch
        loss_array_train = []
        loss_array_test = []

        # TRAIN
        batch_streamer_train.reset() # ensure that all iterators are reset
        for H, F, C, P in batch_streamer_train:

            with GradientTape() as tape:

                # generate and evaluate predictions
                sigmoid_proba = model([H, F, C, P], training=True)
                loss_value = loss_fn(P, sigmoid_proba)
                loss_array_train.append(loss_value)

                # update weights with computed gradients
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

        mean_loss_current_epoch_train = np.array(loss_array_train).mean()
        loss_per_epoch_train.append(mean_loss_current_epoch_train)
        print(f'Avg loss train: {mean_loss_current_epoch_train}')

        # TEST
        batch_streamer_test.reset()
        for H, F, C, P in batch_streamer_test:

            # generate and evaluate predictions
            sigmoid_proba=model([H, F, C, P], training=False)
            loss_value = loss_fn(P, sigmoid_proba)
            loss_array_test.append(loss_value)

        mean_loss_current_epoch_test = np.array(loss_array_test).mean()
        loss_per_epoch_test.append(mean_loss_current_epoch_test.copy())
        print(f'Avg loss train: {mean_loss_current_epoch_test}')

    return model
