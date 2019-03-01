#!/usr/bin/env python3
import argparse
import os
from copy import deepcopy

import PIL
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from horoma.constants import SAVED_MODEL_DIR, TrainMode
from horoma.experiments.factory import experiment_factory
from horoma.models.factory import (cluster_factory, embedding_factory,
                                   supported_cluster, supported_embedding)
from horoma.utils import get_param_file
from horoma.utils.data import HoromaDataset
from horoma.utils.factory import crit_factory, optim_factory


def get_train_parser(parent=None):
    '''
    Construct arg parser for train script
    '''
    if parent is None:
        parser = argparse.ArgumentParser()
    else:
        parser = parent.add_parser('train', help='Train models')

    parser.add_argument(
        '--embedding',
        type=str,
        help='Embedding model to train',
        choices=supported_embedding(),
        required=True,
    )

    parser.add_argument(
        '--cluster',
        type=str,
        help="Clustering algorithm to use",
        choices=supported_cluster(),
        required=True,
    )

    parser.add_argument(
        '--mode',
        type=str,
        help="Train mode to use",
        choices=[e.name for e in TrainMode],
        default=TrainMode.TRAIN_ALL.name,
        required=False,
    )

    parser.add_argument(
        '--params',
        type=str,
        help='Model param file location. '
        'For information about param file format refer README.md'
    )

    return parser


def train_model(embedding_name, cluster_method_name, mode, params):
    mode = TrainMode[mode]

    tranformer_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomRotation(30, resample=PIL.Image.NEAREST),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(hue=0.3),
        transforms.ToTensor(),
    ])
    # load data
    train_dataset = HoromaDataset(
        split='train', transform=tranformer_pipeline)
    valid_dataset = HoromaDataset(
        split='valid')

    # Split train dataset in two
    train_train_size = int(len(train_dataset) * 0.95)
    train_valid_size = len(train_dataset) - train_train_size
    train_train_dataset, train_valid_dataset = random_split(
        train_dataset, [train_train_size, train_valid_size])

    train_train_loader = DataLoader(
        train_train_dataset, batch_size=100, shuffle=True)
    train_valid_loader = DataLoader(
        train_valid_dataset, batch_size=100, shuffle=True)

    per_class_data = {}
    for data, label in valid_dataset:
        label = int(label)
        if label not in per_class_data:
            per_class_data[label] = []
        per_class_data[label].append(data)

    # manufacture more data
    max_data_per_class = 100
    augmented_data = {}
    for class_label, class_data in per_class_data.items():
        data_len = len(class_data)
        augmented_data[class_label] = []
        num_loops = int(max_data_per_class / data_len) + 1
        for _ in range(num_loops):
            augmented_data[class_label].extend(
                map(tranformer_pipeline, class_data))

    valid_split = 0.5
    valid_dataset = {label: train_test_split(
        data, test_size=valid_split) for label, data in augmented_data.items()}

    valid_valid_dataset = []
    valid_train_dataset = []
    for label, data in valid_dataset.items():
        valid_train_dataset.extend(
            zip(data[0], torch.Tensor(len(data[0]) * [label])))
        valid_valid_dataset.extend(
            zip(data[1], torch.Tensor(len(data[1]) * [label])))

    valid_train_loader = DataLoader(
        valid_train_dataset, batch_size=100, shuffle=False)
    valid_valid_loader = DataLoader(
        valid_valid_dataset, batch_size=100, shuffle=False)

    # get embedding model
    embedding_model = embedding_factory(
        embedding_name, params.get('embedding_params', {}))
    optim_parameters = params['optimizer_params']
    optim_parameters['params'] = embedding_model.parameters()
    embedding_optim = optim_factory(params['optimizer'], optim_parameters)
    crit_parameters = params.get('criterion_params', {})
    embedding_crit = crit_factory(params['criterion'], crit_parameters)

    # get cluster model
    cluster_model = cluster_factory(
        cluster_method_name, params['cluster_params'])

    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    exp_file = os.path.join(SAVED_MODEL_DIR, params['exp_file'])

    # set up exp parameters
    experiment_params = {
        "experiment_file": exp_file,
        "embedding_model": embedding_model,
        "cluster_obj": cluster_model,
        "embedding_optim": embedding_optim,
        "embedding_crit": embedding_crit,
    }

    # get experiment object
    experiment = experiment_factory(embedding_name, experiment_params)
    experiment.train(train_train_loader, train_valid_loader,
                     params['epochs'], valid_train_loader, valid_valid_loader, mode=mode)


def train(args):
    param_file = get_param_file(args.embedding, args.cluster)
    if args.params:
        param_file = args.params
    if param_file is None:
        exit(1)

    # load exp parameters
    with open(param_file) as fob:
        params = yaml.load(fob)

    # train model
    train_model(args.embedding, args.cluster, args.mode, params)


if __name__ == '__main__':
    parser = get_train_parser()
    args = parser.parse_args()
    train(args)
