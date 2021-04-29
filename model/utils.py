import os

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

DATA_DIR = '/data/kaggle/hubmap'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
TF_TRAIN_DIR = os.path.join(DATA_DIR, 'tf_train')

TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TF_TRAIN_PATH = os.path.join(DATA_DIR, 'tf_train.csv')
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

GLOBALS = {
    'tile_size': 256,
    'batch_size': 32,
    's_threshold': 40,
}
GLOBALS['min_overlap'] = GLOBALS['tile_size'] // 8
GLOBALS['p_threshold'] = 1000 * (GLOBALS['tile_size'] // 256) ** 2


def verify_initial_data_presence():
    for _path in [TRAIN_PATH, SAMPLE_SUBMISSION_PATH]:
        assert os.path.exists(_path)
    return


def create_local_dirs():
    for _dir in [LOGS_DIR, MODELS_DIR, RESULTS_DIR]:
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
