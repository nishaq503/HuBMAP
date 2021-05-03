from glob import glob
from typing import List

from tensorflow import keras

import utils
from datagen import TrainSequence
from net import HubmapMasker


def train_model(
        fold: int,
        initial_epoch: int,
        final_epoch: int,
        file_ids: List[str],
):
    if fold >= 4:
        raise ValueError(f'We only do 4 folds. Fold must be one of [0, 1, 2, 3]. Got {fold} instead.')

    keras.backend.clear_session()

    val_start, val_end = fold * 3, (fold + 1) * 3
    train_ids = file_ids[:val_start] + file_ids[val_end:]
    val_ids = file_ids[val_start:val_end]

    model_name = f'main_fold_{fold}'
    if initial_epoch == 0:
        filters = [
            utils.GLOBALS['base_filters'] * (1 + i)
            for i in range(utils.GLOBALS['model_depth'])
        ]
        model = HubmapMasker(
            model_name=model_name,
            image_size=utils.GLOBALS['tile_size'],
            filter_sizes=3,
            filters=filters,
            pool_size=2,
            smoothing_size=5,
            dropout_rate=0.25,
        )
    else:
        model = HubmapMasker.load(model_name)
    model.summary()
    # exit(1)

    weights = {
        'embedding': 4,
        'autoencoder': 4,
        'mask': 1,
    }
    model.compile(weights=weights)

    train_gen = TrainSequence(train_ids)
    valid_gen = TrainSequence(val_ids)
    model.fit(
        x=iter(train_gen),
        steps_per_epoch=len(train_gen),
        initial_epoch=initial_epoch,
        epochs=final_epoch,
        verbose=1,
        validation_data=iter(valid_gen),
        validation_steps=len(valid_gen),
    )
    model.save()
    return


if __name__ == '__main__':
    # utils.delete_old()

    _model_name = 'main_model'
    _file_ids = [_name.split('/')[-1].split('.')[0] for _name in glob(f'{utils.TRAIN_DIR}/*.tiff')]
    _file_ids = list(sorted(_file_ids))
    _train_ids = _file_ids[:12]
    _test_ids = _file_ids[12:]
    train_model(
        fold=0,
        initial_epoch=0,
        final_epoch=1,
        file_ids=_train_ids,
    )
