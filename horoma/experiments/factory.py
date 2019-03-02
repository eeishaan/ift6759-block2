from horoma.experiments.ae import AEExperiment
from horoma.experiments.caes import CAESExperiment

SUPPORTED_EXP = {
    'ae': AEExperiment,
    'caes': CAESExperiment
}


def experiment_factory(embedding_name, params):
    """
    Embedding name has one-to-one map with experiment name
    """
    return SUPPORTED_EXP[embedding_name](**params)
