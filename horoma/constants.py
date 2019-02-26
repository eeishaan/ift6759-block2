import os
from pathlib import Path
from enum import Enum
# data file paths
DATA_ROOT_FOLDER = Path('/rap/jvb-000-aa/COURS2019/etudiants/data/horoma')

# project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODEL_DIR = Path(os.path.abspath(PROJECT_ROOT / '../saved_models'))
PCA_MODEL_DEFAULT_PATH = SAVED_MODEL_DIR / 'pca_model.sav'
TSNE_MODEL_DEFAULT_PATH = SAVED_MODEL_DIR / 'tsne_model.sav'
RESULT_DIR = Path(os.path.abspath(PROJECT_ROOT / '../results'))
PARAM_DIR = PROJECT_ROOT / 'params'


class TrainMode(Enum):
    '''
    Enum for training mode
    '''
    TRAIN_ALL = 0
    TRAIN_ONLY_EMBEDDING = 1
    TRAIN_ONLY_CLUSTER = 2


class DataSplit(Enum):
    '''
    Data split enum
    '''
    TRAINING = 1
    TRAINING_OVERLAPPED = 2
    VALIDATION = 3
    VALIDATION_OVERLAPPED = 4
    TEST = 5
