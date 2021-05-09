import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from glob import glob
from typing import List

import tensorflow as tf
from tensorflow import keras

import loss_functions
import utils
from datagen import TrainSequence
from net import HubmapMasker

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def train_model(model_name: str, initial_epoch: int, final_epoch: int, train_ids: List[str], val_ids: List[str]):
    if initial_epoch == 0:
        filters = [
            utils.GLOBALS['base_filters'] * (2 ** i)
            for i in range(utils.GLOBALS['model_depth'])
        ]
        model = HubmapMasker(
            model_name=model_name,
            image_size=utils.GLOBALS['tile_size'],
            filter_sizes=3,
            filters=filters,
            pool_size=2,
            dropout_rate=0.4,
        )
        model.save()
    else:
        model = HubmapMasker.load(model_name)
    model.summary()
    # exit(1)

    model.compile()

    model.fit(
        x=TrainSequence(train_ids),
        initial_epoch=initial_epoch,
        epochs=final_epoch,
        verbose=1,
        validation_data=TrainSequence(val_ids),
    )
    model.save()
    return


def train_fold(model_name: str, fold: int, fold_size: int, file_ids: List[str]):
    if len(file_ids) % fold_size != 0:
        raise ValueError(f'fold_size {fold_size} with {len(file_ids)} files results in uneven fold sizes.')

    if not 0 <= fold < len(file_ids) // fold_size:
        raise IndexError(f'fold {fold} is out of index at a fold_size of {fold_size} with {len(file_ids)} files.'
                         f'fold must be between 0 and {len(file_ids) // fold_size - 1}.')

    keras.backend.clear_session()

    val_start, val_end = fold * fold_size, (fold + 1) * fold_size
    train_ids = file_ids[:val_start] + file_ids[val_end:]
    val_ids = file_ids[val_start:val_end]

    model_name = f'{model_name}_{fold}'
    train_model(
        model_name=model_name,
        initial_epoch=0,
        final_epoch=10,
        train_ids=train_ids,
        val_ids=val_ids,
    )
    return


if __name__ == '__main__':
    # utils.delete_old()
    # exit(1)

    # epoch 1 started with dice coef = 0.1380

    _file_ids = [_name.split('/')[-1].split('.')[0] for _name in glob(f'{utils.TRAIN_DIR}/*.tiff')]
    _file_ids = list(sorted(_file_ids))
    train_fold(
        model_name='test_model',
        fold=0,
        fold_size=3,
        file_ids=_file_ids,
    )
