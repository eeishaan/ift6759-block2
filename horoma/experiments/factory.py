from horoma.experiments import HoromaExperiment
from horoma.experiments.cvae import CVAEExperiment
from horoma.experiments.caes import CAESExperiment

SUPPORTED_EXP = {
    'vae': HoromaExperiment,
    'cvae': CVAEExperiment,
    'caes': CAESExperiment
}


def experiment_factory(embedding_name, params):
    """
    Embedding name has one-to-one map with experiment name
    """
    return SUPPORTED_EXP[embedding_name](**params)
