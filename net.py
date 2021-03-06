import json
import os
from typing import List
from typing import Optional

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

import loss_functions
import utils


class HubmapMasker(keras.models.Model):
    def __init__(
            self,
            model_name: str,
            image_size: int,
            filter_sizes: int,
            filters: List[int],
            pool_size: int,
            dropout_rate: float,
    ):
        super(HubmapMasker, self).__init__()

        # store arguments for config
        self.model_name = model_name
        self.image_size = image_size
        self.image_shape = (image_size, image_size, 3)
        self.filter_sizes = filter_sizes
        self.filters = filters
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate

        # set up encoder
        input_layer = keras.layers.Input(shape=self.image_shape, name='input')
        x = input_layer

        skip_layers = list()
        for i, f in enumerate(self.filters[:-1]):
            x = self._conv_block(x, f, name=f'skip_{f}')
            skip_layers.append(x)
            x = keras.layers.MaxPool2D((self.pool_size, self.pool_size))(x)
            x = keras.layers.SpatialDropout2D(self.dropout_rate)(x)
        else:
            x = self._conv_block(x, self.filters[-1], name='embedding')

        # set up decoder
        args = list(zip(self.filters[:-1], skip_layers))
        for f, skip in reversed(args):
            x = keras.layers.UpSampling2D(size=(self.pool_size, self.pool_size))(x)
            x = keras.layers.Concatenate()([x, skip])
            x = self._conv_block(x, f)
            x = keras.layers.SpatialDropout2D(self.dropout_rate)(x)

        x = keras.layers.Conv2D(1, self.filter_sizes, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        output_layer = keras.layers.Activation('sigmoid', name='mask')(x)

        self.model = keras.models.Model(
            inputs=input_layer,
            outputs=output_layer,
            name='model',
        )

    def _conv_block(self, x, filters, name: str = None):
        x = keras.layers.Conv2D(filters, self.filter_sizes, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(filters, self.filter_sizes, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        if name is None:
            x = keras.layers.Activation('relu')(x)
        else:
            x = keras.layers.Activation('relu', name=name)(x)
        return x

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def summary(self, **kwargs):
        return self.model.summary(**kwargs)

    # noinspection PyMethodOverriding
    def compile(self, *, optimizer=None, loss=None, metrics=None):
        if optimizer is None:
            optimizer = keras.optimizers.Nadam(learning_rate=1e-2)

        if loss is None:
            loss = loss_functions.dice_bce

        if metrics is None:
            metrics = loss_functions.dice_coef

        return self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )

    def fit(self, **kwargs):
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

    def mask(self, images, verbose: int = 1):
        return tf.squeeze(self.masker.predict(images, verbose=verbose))

    def get_config(self):
        return {
            'model_name': self.model_name,
            'image_size': self.image_size,
            'filter_sizes': self.filter_sizes,
            'filters': self.filters,
            'pool_size': self.pool_size,
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


def test_model():
    filters = [
        utils.GLOBALS['base_filters'] * (2 ** i)
        for i in range(utils.GLOBALS['model_depth'])
    ]
    model = HubmapMasker(
        model_name='test_model',
        image_size=utils.GLOBALS['tile_size'],
        filter_sizes=3,
        filters=filters,
        pool_size=2,
        smoothing_size=5,
        dropout_rate=0.25,
    )
    model.summary()
    return model


def test_fit_and_save(model: HubmapMasker):
    image_shape = (
        utils.GLOBALS['batch_size'] * 4,
        utils.GLOBALS['tile_size'],
        utils.GLOBALS['tile_size'],
        3
    )
    images = tf.random.uniform(shape=image_shape, dtype=tf.float16)
    masks = tf.cast(tf.random.uniform(shape=tuple(image_shape[:-1])) > 0.5, dtype=tf.uint8)

    ys = {
        # 'embedding': masks,
        'mask': masks,
    }

    model.compile()
    model.fit(
        x=images,
        y=ys,
        batch_size=utils.GLOBALS['batch_size'],
        epochs=2,
        verbose=1,
        callbacks=None,
    )
    model.save()
    return model


if __name__ == '__main__':
    # utils.delete_old()
    _model = test_model()
    # _model = test_fit_and_save(_model)
    pass
