import os

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

DATA_DIR = '/data/kaggle/hubmap'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
FEATURES_PATH = os.path.join(DATA_DIR, 'features.csv')
EXAMPLE_TEST_PATH = os.path.join(DATA_DIR, 'example_test.csv')
EXAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, 'example_sample_submission.csv')

TRAIN_X_PATH = os.path.join(DATA_DIR, 'train_x.npy')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'train_y.npy')

VAL_X_PATH = os.path.join(DATA_DIR, 'val_x.npy')
VAL_Y_PATH = os.path.join(DATA_DIR, 'val_y.npy')

TEST_X_PATH = os.path.join(DATA_DIR, 'test_x.npy')
TEST_Y_PATH = os.path.join(DATA_DIR, 'test_y.npy')


if __name__ == '__main__':
    for _path in [TRAIN_PATH, FEATURES_PATH, EXAMPLE_TEST_PATH, EXAMPLE_SUBMISSION_PATH]:
        assert os.path.exists(_path)

    for _dir in [LOGS_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(_dir, exist_ok=True)
