
import sys
sys.path.append('.')

from typing import Optional, Sequence, Union

import numpy as np
from tqdm import tqdm
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
        dense_layer: int = 250,
        activation: str = 'sigmoid'
        ) -> Model:
    """
    Build a CNN model as described in the paper.
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

    final_layer = layers.Dense(dense_layer, activation=activation)(merged)

    model = Model(
        inputs=[head_H.input, head_F.input, head_C.input], 
        outputs=final_layer
    )

    return model


class NaiveModel:
    """
    A very naive model that returns random or fixed predictions
    """
    def __init__(self, fix_value: Optional[float] = None) -> None:
        self.fix_value = fix_value

    def __call__(self, input: Sequence[np.ndarray], *args, **kwargs) -> np.ndarray:
        batch_size = input[0].shape[0]
        nr_products = input[0].shape[1]
        
        if self.fix_value:
            pred = [self.fix_value for _ in range(batch_size*nr_products)]
        else:
            pred = [np.random.rand() for _ in range(batch_size*nr_products)]

        return np.array(pred).reshape((batch_size, nr_products))

    def save_weights(self, *args, **kwargs) -> None:
        pass

    def summary(self) -> str:
        return 'This is a naive model'


def build_naive_model(seed = 0, fix_value: Optional[float] = None, **kwargs):
    """
    Build a naive model to do a naive benchmark.
    """
    np.random.seed(seed)
    return NaiveModel(fix_value=fix_value)


def train_model(
        model: Union[Model, NaiveModel],
        optimizer: Optimizer,
        loss_fn: Loss,
        batch_streamer_train: BatchStreamer,
        batch_streamer_test: BatchStreamer,
        epochs: int = 10,
        ) -> Union[Model, NaiveModel]:
    """
    Train the previously built model as described in the paper.
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
        for H, F, C, P in tqdm(batch_streamer_train):

            with GradientTape() as tape:
                # generate and evaluate predictions
                sigmoid_proba = model([H, F, C], training=True)
                loss_value = loss_fn(P, sigmoid_proba)
                loss_array_train.append(loss_value)

            if isinstance(model, NaiveModel):
                pass
            else:
                # update weights with computed gradients
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

        mean_loss_current_epoch_train = np.array(loss_array_train).mean()
        loss_per_epoch_train.append(mean_loss_current_epoch_train)
        print(f'Avg loss train: {mean_loss_current_epoch_train}')

        # TEST
        batch_streamer_test.reset()
        for H, F, C, P in tqdm(batch_streamer_test):

            # generate and evaluate predictions
            sigmoid_proba=model([H, F, C], training=False)
            loss_value = loss_fn(P, sigmoid_proba)
            loss_array_test.append(loss_value)

        mean_loss_current_epoch_test = np.array(loss_array_test).mean()
        loss_per_epoch_test.append(mean_loss_current_epoch_test.copy())
        print(f'Avg loss train: {mean_loss_current_epoch_test}')

    return model


if __name__ == '__main__':

    # run me to generate the model plot

    from tensorflow.keras.utils import plot_model

    model = build_model()
    plot_model(model, to_file='model.png', show_layer_names=False, rankdir='LR')
