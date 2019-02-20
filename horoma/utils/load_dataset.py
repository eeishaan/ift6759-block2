import os
from enum import Enum

import numpy as np
from torch.utils.data import Dataset

from horoma.constants import (DATA_ROOT_FOLDER, DEF_LABELS_FILE,
                              TEST_DATA_FILE, TEST_DATA_SIZE,
                              TRAINING_DATA_FILE, TRAINING_DATA_SIZE,
                              VALIDATION_DATA_FILE, VALIDATION_DATA_SIZE,
                              VALIDATION_LABELS_FILE)


# Data type
class DataSplit(Enum):
    TRAINING = 1
    VALIDATION = 2
    TEST = 3


class HoromaDataset(Dataset):
    """A PyTorch Dataset to access the horoma project data.

    The parameters of the initialization methods are:
        data_type: Type of data (Enum). train, valid or test.
        dataset_dir: The absolute path to the folder containing the data to use.
        transform: A list of optional transformations to apply on the data.
    """

    def __init__(self, data_type=DataSplit.TRAINING, dataset_dir=DATA_ROOT_FOLDER, transform=None):

        self.labels = []
        self.label_definitions = []

        data_size = TRAINING_DATA_SIZE
        data_file_path = dataset_dir + TRAINING_DATA_FILE

        if data_type == DataSplit.VALIDATION:
            data_size = VALIDATION_DATA_SIZE
            data_file_path = dataset_dir + VALIDATION_DATA_FILE
            self.labels = np.loadtxt(
                dataset_dir + VALIDATION_LABELS_FILE, dtype=np.int64)
            self.label_definitions = np.genfromtxt(dataset_dir + DEF_LABELS_FILE, delimiter=",",
                                                   dtype=('U2', 'U3', 'U3'), names=["specie", "density", "height"])

        elif data_type == DataSplit.TEST:
            data_size = TEST_DATA_SIZE
            data_file_path = dataset_dir + TEST_DATA_FILE
            label_definitions_path = dataset_dir + DEF_LABELS_FILE

            # Make sure we can load the label definitions when file is not found in in test data folder
            if label_definitions_path is None or not os.path.exists(label_definitions_path):
                label_definitions_path = DATA_ROOT_FOLDER + DEF_LABELS_FILE

            self.label_definitions = np.genfromtxt(label_definitions_path, delimiter=",",
                                                   dtype=('U2', 'U3', 'U3'), names=["specie", "density", "height"])

        self.data = np.memmap(data_file_path, dtype='float32',
                              mode='r', shape=(data_size, 32, 32, 4))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = self.data[i]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def to_2d_array(self, nb_examples=None):

        if nb_examples:
            return self.data[:nb_examples].reshape((nb_examples, 32 * 32 * 4))

        return self.data.reshape((len(self.data), 32 * 32 * 4))

    def get_labels(self, nb_labels=None):

        if nb_labels:
            return self.labels[:nb_labels]

        return self.labels

    def get_label_definitions(self):

        return self.label_definitions
