#!/usr/bin/env python3
import argparse
import logging
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from horoma.constants import TrainMode, SAVED_MODEL_DIR
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
        help="Clustering algorithm to use",
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
    # load data
    train_dataset = HoromaDataset(
        split='train_overlapped', transform=transforms.ToTensor())
    valid_dataset = HoromaDataset(
        split='valid', transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

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
    experiment.train(train_loader, params['epochs'], valid_dataset, mode=mode)


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
