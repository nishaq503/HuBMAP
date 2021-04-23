from collections import Counter

import numpy as np
from tensorflow import keras

from model import utils


def preparse():
    pass


class JSData(keras.utils.Sequence):
    def __init__(self, mode: str, resp: int, batch_size: int, noisy: bool = False):
        if mode not in {'train', 'validate', 'test'}:
            raise ValueError(f'mode must be \'train\', \'validate\', or \'test\'. Got {mode} instead.')
        if resp not in {0, 1, 2, 3, 4}:
            raise ValueError(f'resp should be one of [0, 1, 2, 3, 4]. Got {resp} instead.')

        if mode == 'train':
            x_path = utils.TRAIN_X_PATH
            y_path = utils.TRAIN_Y_PATH
        elif mode == 'validate':
            x_path = utils.VAL_X_PATH
            y_path = utils.VAL_Y_PATH
        else:
            x_path = utils.TEST_X_PATH
            y_path = utils.TEST_Y_PATH

        self.x_data = np.load(x_path, mmap_mode='r')
        self.y_data = np.load(y_path)[:, resp]

        self.indices = list(range(len(self.y_data)))
        np.random.shuffle(self.indices)

        self.batch_size = batch_size
        self.noisy = noisy

    def __getitem__(self, index):
        batch_ids = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        x = self.x_data[batch_ids]
        noisy_x = np.copy(x)
        if self.noisy:
            # 20% chance to set each weight to 0
            signs = np.asarray(np.random.uniform(0, 1, size=len(batch_ids)) > 0.2, dtype=np.int)
            noisy_x[:, 0] *= signs

            # 2% chance to flip the sign of each feature value (other than weight)
            signs = np.random.uniform(0, 1, size=(len(batch_ids), noisy_x.shape[1] - 1)) > 0.02
            signs = np.asarray(signs * 2 - 1, dtype=np.int)
            noisy_x[:, 1:] *= signs

        # noinspection PyTypeChecker
        y = self.y_data[batch_ids]

        ys = {
            'encoder': y,
            'decoder': x,
            'classifier': y,
        }
        return noisy_x, ys

    def __len__(self):
        num_batches = self.x_data.shape[0] // self.batch_size
        if self.x_data.shape[0] % self.batch_size > 0:
            num_batches += 1
        return num_batches

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        return


if __name__ == '__main__':
    preparse()
    _datagen = JSData(
        mode='train',
        resp=0,
        batch_size=32,
        noisy=True
    )
    for _i in range(5):
        _v1, _v2 = _datagen[_i]
        print(dict(Counter(_v2['classifier'])))
