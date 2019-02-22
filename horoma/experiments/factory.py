from horoma.experiments import HoromaExperiment

SUPPORTED_EXP = {
    'vae': HoromaExperiment,
}


def experiment_factory(embedding_name, params):
    """
    Embedding name has one-to-one map with experiment name
    """
    return SUPPORTED_EXP[embedding_name](**params)
