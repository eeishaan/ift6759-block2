'''
Supported model index
'''
from horoma.models.vae import VAE
from horoma.models.cvae import CVAE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

EMBEDDING_MODELS = {
    'vae': VAE,
    'cvae': CVAE
}

CLUSTER_MODELS = {
    'kmeans': KMeans,
    'gmm': GaussianMixture
}


def embedding_factory(name, params):
    return EMBEDDING_MODELS[name](**params)


def cluster_factory(name, params):
    return CLUSTER_MODELS[name](**params)


def supported_cluster():
    return CLUSTER_MODELS.keys()


def supported_embedding():
    return EMBEDDING_MODELS.keys()
