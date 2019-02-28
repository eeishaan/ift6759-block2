from horoma.experiments.vae import VAEExperiment
from horoma.experiments.caes import CAESExperiment

SUPPORTED_EXP = {
    'vae': VAEExperiment,
    'caes': CAESExperiment
}


def experiment_factory(embedding_name, params):
    """
    Embedding name has one-to-one map with experiment name
    """
    return SUPPORTED_EXP[embedding_name](**params)
