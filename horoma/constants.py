import os
from pathlib import Path
from enum import Enum
# data file paths
DATA_ROOT_FOLDER = Path('/rap/jvb-000-aa/COURS2019/etudiants/data/horoma')
TRAINING_DATA_FILE = DATA_ROOT_FOLDER / 'train_x.dat'
VALIDATION_DATA_FILE = DATA_ROOT_FOLDER / 'valid_x.dat'
VALIDATION_LABELS_FILE = DATA_ROOT_FOLDER / 'valid_y.txt'
DEF_LABELS_FILE = DATA_ROOT_FOLDER / 'definition_labels.txt'
TEST_DATA_FILE = DATA_ROOT_FOLDER / 'test_x.dat'

# project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODEL_DIR = Path(os.path.abspath(PROJECT_ROOT / '../saved_models'))
PCA_MODEL_DEFAULT_PATH = SAVED_MODEL_DIR / 'pca_model.sav'
TSNE_MODEL_DEFAULT_PATH = SAVED_MODEL_DIR / 'tsne_model.sav'
RESULT_DIR = Path(os.path.abspath(PROJECT_ROOT / '../results'))
PARAM_DIR = PROJECT_ROOT / 'params'

# data variables
TRAINING_DATA_SIZE = 1614214
VALIDATION_DATA_SIZE = 201778
TEST_DATA_SIZE = 201778


class TrainMode(Enum):
    '''
    Enum for training mode
    '''
    TRAIN_ALL = 0
    TRAIN_ONLY_EMBEDDING = 1
    TRAIN_ONLY_CLUSTER = 2
