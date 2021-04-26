import os

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

DATA_DIR = '/data/kaggle/hubmap'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
TF_TRAIN_DIR = os.path.join(DATA_DIR, 'tf_train')
TF_TEST_DIR = os.path.join(DATA_DIR, 'tf_test')

TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')


if __name__ == '__main__':
    for _path in [TRAIN_PATH, SAMPLE_SUBMISSION_PATH]:
        assert os.path.exists(_path)

    for _dir in [LOGS_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(_dir, exist_ok=True)
