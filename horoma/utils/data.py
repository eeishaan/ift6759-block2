import os
from enum import Enum

import numpy as np
import torch
from torch.utils.data import Dataset

from horoma.constants import DATA_ROOT_FOLDER, DataSplit


class HoromaDataset(Dataset):

    def __init__(self, data_dir=DATA_ROOT_FOLDER, split="train",
                 subset=None, skip=0, flattened=False, transform=None):
        """
        Args:
            data_dir: Path to the directory containing the samples.
            split: Which split to use. [train, valid, test]
            subset: How many elements will be used. Default: all.
            skip: How many element to skip before taking the subset.
            flattened: If True return the images in a flatten format.
            tranform: Transform data with supplied transformer
        """
        self.transform = transform
        nb_channels = 3
        height = 32
        width = 32
        datatype = "uint8"

        if split == "train":
            self.nb_exemples = 150700
        elif split == "valid":
            self.nb_exemples = 480
        elif split == "test":
            self.nb_exemples = 498
        elif split == "train_overlapped":
            self.nb_exemples = 544749
        elif split == "valid_overlapped":
            self.nb_exemples = 1331
        else:
            raise(
                "Dataset: Invalid split. Must be " +
                "[train, valid, test, train_overlapped, valid_overlapped]")

        filename_x = os.path.join(data_dir, "{}_x.dat".format(split))
        filename_y = os.path.join(data_dir, "{}_y.txt".format(split))

        self.targets = None
        if os.path.exists(filename_y) and not split.startswith("train"):
            pre_targets = np.loadtxt(filename_y, 'U2')

            if subset is None:
                pre_targets = pre_targets[skip: None]
            else:
                pre_targets = pre_targets[skip: skip + subset]

            self.map_labels = np.unique(pre_targets)
            self.targets = np.asarray(
                [np.where(self.map_labels == t)[0][0] for t in pre_targets])

        self.data = np.memmap(filename_x, dtype=datatype, mode="r", shape=(
            self.nb_exemples, height, width, nb_channels))
        if subset is None:
            self.data = self.data[skip: None]
        else:
            self.data = self.data[skip: skip + subset]

        if flattened:
            self.data = self.data.reshape(len(self.data), -1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.targets is not None:
            return self.transform(self.data[index]), torch.Tensor([self.targets[index]])
        return self.transform(self.data[index])
