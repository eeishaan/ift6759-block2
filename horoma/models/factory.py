'''
Supported model index
'''
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

from horoma.models.ae import AE
from horoma.models.cae import CAE
from horoma.models.caes import CAES
from horoma.models.squeezenet import SqueezeNet

EMBEDDING_MODELS = {
    'squeezenet': SqueezeNet,
    'ae': AE,
    'vae': AE,
    'cae': CAE,
    'caes': CAES
}

CLUSTER_MODELS = {
    'kmeans': KMeans,
    'gmm': GaussianMixture,
    'mini_batch_kmeans': MiniBatchKMeans,
}


def embedding_factory(name, params):
    return EMBEDDING_MODELS[name](**params)


def cluster_factory(name, params):
    return CLUSTER_MODELS[name](**params)


def supported_cluster():
    return CLUSTER_MODELS.keys()


def supported_embedding():
    return EMBEDDING_MODELS.keys()
