import os

DATA_DIR = '../input/hubmap-kidney-segmentation'
TRAIN_DIR = f'{DATA_DIR}/train'
TEST_DIR = f'{DATA_DIR}/test'
TRAIN_PATH = f'{DATA_DIR}/train.csv'
SAMPLE_SUBMISSION_PATH = f'{DATA_DIR}/sample_submission.csv'

ROOT_DIR = './'
LOGS_DIR = f'{ROOT_DIR}/logs'
MODELS_DIR = f'{ROOT_DIR}/saved_models'
LOCAL_DIR = f'{ROOT_DIR}/local'

TILES_PATH = f'{LOCAL_DIR}/tiles'
SUBMISSION_PATH = f'{ROOT_DIR}/submission.csv'

GLOBALS = {
    'tile_size': 1024,
    'batch_size': 4,
    's_threshold': 40,
    'model_depth': 7,
    'base_filters': 16,
}
GLOBALS['min_overlap'] = GLOBALS['tile_size'] // 8
GLOBALS['p_threshold'] = 1000 * (GLOBALS['tile_size'] // 256) ** 2


def verify_initial_data_presence():
    for _path in [TRAIN_DIR, TEST_DIR, TRAIN_PATH, SAMPLE_SUBMISSION_PATH]:
        assert os.path.exists(_path), f'path not found: {_path}'
    return


def create_local_dirs():
    for _dir in [LOGS_DIR, MODELS_DIR, LOCAL_DIR]:
        os.makedirs(_dir, exist_ok=True)
    return


def delete_old():
    import shutil

    for _dir in [LOGS_DIR, MODELS_DIR]:
        shutil.rmtree(_dir)
        os.makedirs(_dir, exist_ok=True)
    return


if __name__ == '__main__':
    verify_initial_data_presence()
    create_local_dirs()
