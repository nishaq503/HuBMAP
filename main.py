from glob import glob
from typing import List

import tensorflow as tf

from datagen import TrainSequence
from model import utils
from model.net import HubmapMasker

tf.keras.backend.set_floatx('float16')


def train_model(
        model_name: str,
        train_ids: List[str],
        val_ids: List[str],
        epochs: int,
):
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
    model.summary()
    model.compile()
    exit(1)

    train_gen = TrainSequence(train_ids)
    valid_gen = TrainSequence(val_ids)
    model.fit(
        x=iter(train_gen),
        steps_per_epoch=len(train_gen),
        epochs=epochs,
        verbose=1,
        validation_data=iter(valid_gen),
        validation_steps=len(valid_gen),
    )
    model.save()
    return


def resume_training(
        model_name: str,
        train_ids: List[str],
        val_ids: List[str],
        initial_epoch: int,
        final_epoch: int,
):
    model = HubmapMasker.load(model_name)
    model.summary()

    weights = {
        'embedding': 1,
        'autoencoder': 1,
        'mask': 4,
    }
    model.compile(weights=weights)
    exit(1)

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
    _train_ids = _file_ids[:12]
    _val_ids = _file_ids[12:]
    train_model(
        model_name=_model_name,
        train_ids=_train_ids,
        val_ids=_val_ids,
        epochs=1,
    )

    # resume_training(
    #     model_name=_model_name,
    #     train_ids=_train_ids,
    #     val_ids=_val_ids,
    #     initial_epoch=1,
    #     final_epoch=2,
    # )
