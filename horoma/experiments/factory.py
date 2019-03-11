from horoma.experiments.ae import AEExperiment, VAEExperiment
from horoma.experiments.cae import CAEExperiment
from horoma.experiments.caes import CAESExperiment

SUPPORTED_EXP = {
    'ae': AEExperiment,
    'vae': VAEExperiment,
    'cae': CAEExperiment,
    'caes': CAESExperiment
}


def experiment_factory(embedding_name, params):
    """
    Embedding name has one-to-one map with experiment name
    """
    return SUPPORTED_EXP[embedding_name](**params)
