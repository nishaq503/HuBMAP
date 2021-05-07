import os
from glob import glob
from typing import List

from tensorflow import keras

import loss_functions
import utils
from datagen import TrainSequence
from net import HubmapMasker


def train_model(model_name: str, initial_epoch: int, final_epoch: int, train_ids: List[str], val_ids: List[str]):
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
        smoothing_size=3,
        dropout_rate=0.4,
    )
    if initial_epoch != 0:
        path = os.path.join(utils.MODELS_DIR, model_name)
        model.model.load_weights(f'{path}_epoch_{initial_epoch}.h5')
    model.summary()
    # model.save()
    # exit(1)

    optimizer = keras.optimizers.Nadam()
    loss = {
        'embedding': loss_functions.embedding_loss,
        'autoencoder': 'mse',
        'mask': loss_functions.dice_loss,
    }
    weights = {
        'embedding': 1e-6 / 256.,
        'autoencoder': 1e-6,
        'mask': 1.,
    }
    model.compile(optimizer=optimizer, loss=loss, weights=weights)

    train_gen = TrainSequence(train_ids)
    valid_gen = TrainSequence(val_ids)
    model.fit(
        x=train_gen,
        initial_epoch=initial_epoch,
        epochs=final_epoch,
        verbose=1,
        validation_data=valid_gen,
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
        initial_epoch=13,
        final_epoch=25,
        train_ids=train_ids,
        val_ids=val_ids,
    )
    return


if __name__ == '__main__':
    # utils.delete_old()

    _file_ids = [_name.split('/')[-1].split('.')[0] for _name in glob(f'{utils.TRAIN_DIR}/*.tiff')]
    _file_ids = list(sorted(_file_ids))
    _test_ids = _file_ids[12:]
    _file_ids = _file_ids[:12]
    train_fold(
        model_name='test_model',
        fold=2,
        fold_size=3,
        file_ids=_file_ids,
    )
