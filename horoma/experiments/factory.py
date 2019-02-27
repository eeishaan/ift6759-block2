from horoma.experiments.cae import CAEExperiment
from horoma.experiments.vae import VAEExperiment

SUPPORTED_EXP = {
    'vae': VAEExperiment,
    'cae': CAEExperiment
}


def experiment_factory(embedding_name, params):
    """
    Embedding name has one-to-one map with experiment name
    """
    return SUPPORTED_EXP[embedding_name](**params)
