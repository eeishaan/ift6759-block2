import os
from pathlib import Path

# data file paths
DATA_ROOT_FOLDER = Path('/rap/jvb-000-aa/COURS2019/etudiants/data/horoma')
TRAINING_DATA_FILE = DATA_ROOT_FOLDER / 'train_x.dat'
VALIDATION_DATA_FILE = DATA_ROOT_FOLDER / 'valid_x.dat'
VALIDATION_LABELS_FILE = DATA_ROOT_FOLDER / 'valid_y.txt'
DEF_LABELS_FILE = DATA_ROOT_FOLDER / 'definition_labels.txt'
TEST_DATA_FILE = DATA_ROOT_FOLDER / 'test_x.dat'

# project paths
PROJECT_ROOT = Path(os.path.abspath(os.path.realpath(__file__)))
PCA_MODEL_DEFAULT_PATH = PROJECT_ROOT / 'models/saved/pca_model.sav'
TSNE_MODEL_DEFAULT_PATH = PROJECT_ROOT / 'models/saved/tsne_model.sav'

# data variables
TRAINING_DATA_SIZE = 1614214
VALIDATION_DATA_SIZE = 201778
TEST_DATA_SIZE = 201778
