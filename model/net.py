import json
import os
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras

from loss_functions import embedding_loss
from loss_functions import masking_loss
from model import utils


class HubmapMasker(keras.models.Model):
    def __init__(
            self,
            model_name: str,
            image_size: int,
            num_channels: int,
            embedding_dim: int,
            filter_sizes: int,
            filters: List[int],
            strides: List[int],
            dropout_rate: float = 0.25,
    ):
        if len(filters) != len(strides):
            raise ValueError(f'filters and strides must be lists with the same lengths.')

        super(HubmapMasker, self).__init__()

        # store arguments for config
        self.model_name = model_name
        self.image_size = image_size
        self.num_channels = num_channels
        self.image_shape = (image_size, image_size, num_channels)
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.filters = filters
        self.strides = strides
        self.dropout_rate = dropout_rate

        # set up encoder
        input_layer = keras.layers.Input(shape=self.image_shape, name='input')
        x = input_layer

        units = list(zip(self.filters, self.strides))
        for f, s in units:
            x = keras.layers.Conv2D(f, self.filter_sizes, s, padding='same')(x)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(self.dropout_rate)(x)

        volume_size = keras.backend.int_shape(x)

        x = keras.layers.Flatten()(x)
        encoder_output = keras.layers.Dense(self.embedding_dim, name='embedding')(x)

        self.encoder = keras.models.Model(
            inputs=input_layer,
            outputs=encoder_output,
            name='encoder',
        )

        # set up decoder
        x = keras.layers.Dense(units=np.prod(volume_size[1:]))(encoder_output)
        x = keras.layers.Reshape(target_shape=(volume_size[1], volume_size[2], volume_size[3]))(x)

        for f, s in reversed(units):
            x = keras.layers.Conv2DTranspose(f, self.filter_sizes, s, padding='same')(x)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(self.dropout_rate)(x)

        x = keras.layers.Conv2DTranspose(1, self.filter_sizes, padding='same')(x)
        decoder_output = keras.layers.Activation('sigmoid', name='masking')(x)

        self.masker = keras.models.Model(
            inputs=input_layer,
            outputs=decoder_output,
            name='masker',
        )

        self.model = keras.models.Model(
            inputs=input_layer,
            outputs=[encoder_output, decoder_output],
            name='model',
        )

    def call(self, inputs, training=None, mask=None):
        return self.masker(inputs)

    def summary(self, **kwargs):
        return self.model.summary(**kwargs)

    # noinspection PyMethodOverriding
    def compile(
            self, *,
            optimizer='adam',
            loss=None,
            weights=None,
            metrics=None,
    ):
        if loss is None:
            loss = {
                'embedding': embedding_loss,
                'masking': masking_loss,
            }

        return self.model.compile(
            optimizer=optimizer,
            loss=loss,
            loss_weights=weights,
            metrics=metrics
        )

    def fit(self, **kwargs):
        if kwargs['callbacks'] is None:
            kwargs['callbacks'] = [
                keras.callbacks.TensorBoard(
                    log_dir=os.path.join(utils.LOGS_DIR, self.model_name),
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(utils.MODELS_DIR, f'{self.model_name}_epoch_' + '{epoch}.h5'),
                    save_weights_only=True,
                ),
            ]
        return self.model.fit(**kwargs)

    def encode(self, x, batch_size=None, verbose=1):
        return self.encoder.predict(x, batch_size, verbose)

    def decode(self, x, batch_size=None, verbose=1):
        return self.decoder.predict(x, batch_size, verbose)

    def mask(self, x, batch_size=None, verbose=1):
        return self.autoencoder.predict(x, batch_size, verbose)

    def get_config(self):
        return {
            'model_name': self.model_name,
            'image_size': self.image_size,
            'num_channels': self.num_channels,
            'embedding_dim': self.embedding_dim,
            'filter_sizes': self.filter_sizes,
            'filters': self.filters,
            'strides': self.strides,
            'dropout_rate': self.dropout_rate,
        }

    # noinspection PyMethodOverriding
    def save(self):
        config = self.get_config()
        model_path = os.path.join(utils.MODELS_DIR, self.model_name)
        with open(f'{model_path}_config.json', 'w') as fp:
            json.dump(config, fp)
        self.model.save_weights(f'{model_path}.h5')
        return

    @classmethod
    def load(cls, model_name: str):
        model_name = os.path.join(utils.MODELS_DIR, model_name)
        with open(f'{model_name}_config.json', 'r') as fp:
            config = json.load(fp)
        model = cls(**config)
        model.model.load_weights(f'{model_name}.h5')
        return model


def test_model_and_save():
    image_shape = (128, 1024, 1024, 3)
    images = tf.random.uniform(shape=image_shape)
    masks = tf.cast(tf.random.uniform(shape=tuple(image_shape[:-1])) > 0.5, dtype=tf.int32)

    ys = {
        'embedding': masks,
        'masking': masks,
    }

    model = HubmapMasker(
        model_name='test_model',
        image_size=image_shape[1],
        num_channels=image_shape[3],
        embedding_dim=64,
        filter_sizes=7,
        filters=[8, 16, 24, 32],
        strides=[2, 4, 4, 4],
        dropout_rate=0.25,
    )
    model.summary()

    model.compile()
    model.fit(
        x=images,
        y=ys,
        batch_size=16,
        epochs=2,
        verbose=1,
        callbacks=None,
    )
    model.save()
    return


# def test_load_model():
#     model = HubmapAutoencoder.load(os.path.join(utils.MODELS_DIR, 'test_model'))
#     model.summary()
#
#     test_data = tf.random.normal(shape=(100, model.input_shape), mean=5.0, stddev=1.0)
#     predictions = model.classify(test_data, batch_size=10)
#     return predictions


if __name__ == '__main__':
    import shutil

    for _dir in [utils.LOGS_DIR, utils.MODELS_DIR]:
        shutil.rmtree(_dir)
        os.makedirs(_dir, exist_ok=True)

    test_model_and_save()
    # print(test_load_model())
    pass
