from model.datagen import TrainSequence
from model.net import HubmapMasker

from model import utils


def train_model(epochs: int):
    model = HubmapMasker(
        model_name='main_model',
        image_size=utils.GLOBALS['tile_size'],
        num_channels=3,
        filter_sizes=3,
        filters=[16 * (i + 1) for i in range(4)],
        pool_size=2,
        smoothing_size=5,
        dropout_rate=0.25,
    )
    model.summary()
    model.compile()
    # exit(1)

    train_gen = TrainSequence('train')
    valid_gen = TrainSequence('validate')
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


def resume_training(initial_epoch: int, final_epoch: int):
    model = HubmapMasker.load('main_model')

    # weights = {
    #     'embedding': 1,
    #     'autoencoder': 1,
    #     'mask': 2,
    # }
    model.compile()

    train_gen = TrainSequence('train')
    valid_gen = TrainSequence('validate')
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
    train_model(1)
