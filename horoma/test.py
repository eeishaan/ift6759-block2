#!/usr/bin/env python3

import argparse
import json

import numpy as np
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from horoma.constants import LABEL_MAPPING_FILE
from horoma.experiments.factory import experiment_factory
from horoma.models.factory import (cluster_factory, embedding_factory,
                                   supported_cluster, supported_embedding)
from horoma.utils import get_param_file
from horoma.utils.data import HoromaDataset


def get_test_parser(parent=None):
    '''
    Construct argparser for test script
    '''

    if parent is None:
        parser = argparse.ArgumentParser()
    else:
        parser = parent.add_parser('test', help='Test pre-trained models')

    parser.add_argument(
        '--embedding',
        type=str,
        help='Embedding model to test',
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
        '-d', '--data-dir',
        type=str,
        help='Directory of the test data',
        required=True
    )

    parser.add_argument(
        '-f', '--model-path',
        type=str,
        help='File path of the model',
        required=False
    )

    parser.add_argument(
        '--split',
        type=str,
        default='valid',
        choices=['valid', 'test', 'train'],
        help='Dataset split to be used',
    )

    return parser


def test(args):
    data_dir = args.data_dir
    model_file = args.model_path

    # if no model file is specified, use the one specified in params file
    if model_file is None:
        param_file = get_param_file(args.embedding, args.cluster)
        with open(param_file) as fob:
            params = yaml.load(fob)
        model_file = params['model_file']

    # load data
    test_dataset = HoromaDataset(
        data_dir=data_dir, split=args.split, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    # get embedding model
    embedding_model = embedding_factory(
        args.embedding, {})

    # get cluster model
    cluster_model = cluster_factory(
        args.cluster, {})

    # set up exp parameters
    experiment_params = {
        "experiment_file": model_file,
        "embedding_model": embedding_model,
        "cluster_obj": cluster_model,
    }

    # get experiment object
    experiment = experiment_factory(args.embedding, experiment_params)
    experiment.load_experiment()
    res = experiment.test(test_loader)

    # load label mapping
    with open(LABEL_MAPPING_FILE) as fob:
        map_labels = json.load(fob)
    map_labels = np.array(map_labels, dtype='<U2')
    return map_labels[res]


if __name__ == '__main__':
    parser = get_test_parser()
    args = parser.parse_args()
    print(test(args))
