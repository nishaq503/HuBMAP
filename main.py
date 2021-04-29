from datagen import TrainSequence
from net import HubmapMasker

TILE_SIZE = 512


def train_model(epochs: int):
    model = HubmapMasker(
        model_name='main_model',
        image_size=TILE_SIZE,
        num_channels=3,
        filter_sizes=3,
        filters=[32 * (i + 1) for i in range(5)],
        pool_size=2,
        smoothing_size=5,
        dropout_rate=0.25,
    )
    model.summary()
    model.compile()

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

    weights = {
        'embedding': 1,
        'autoencoder': 1,
        'masking': 2,
    }
    model.compile(weights=weights)

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


if __name__ == '__main__':
    train_model(1)
