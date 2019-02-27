'''
Supported model index
'''
from horoma.models.vae import VAE
from horoma.models.cvae import CVAE
from horoma.models.caes import CAES
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

EMBEDDING_MODELS = {
    'vae': VAE,
    'cvae': CVAE,
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
