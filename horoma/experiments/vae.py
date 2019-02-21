from horoma.experiments import HoromaExperiment


class VAEExperiment(HoromaExperiment):

    def __init__(
        self,
        experiment_file,
        embedding_model,
        cluster_obj,
        embedding_optim=None,
        embedding_crit=None,
    ):
        super(VAEExperiment, self).__init__(
            experiment_file,
            embedding_model,
            cluster_obj,
            embedding_optim,
            embedding_crit,
        )

    def compute_loss(self, ctx, outputs, data):
        return self._embedding_crit(data, *outputs)
