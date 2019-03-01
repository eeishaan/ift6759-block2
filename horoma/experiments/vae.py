import torch
from torch.nn import functional as F

from horoma.cfg import DEVICE
from horoma.experiments import HoromaExperiment


class VAEExperiment(HoromaExperiment):
    def after_train(self, ctx):
        super().after_train(ctx)
        print("BCE loss: {} Cluster loss: {}".format(
            ctx.bce.item(), ctx.cluster_error.item()))

    def before_forwardp(self, ctx, data):
        ctx.bce = 0
        ctx.cluster_error = 0

    def compute_loss(self, ctx, outputs, x):
        recon_x, mu, logvar, output_embedding = outputs
        BCE = F.binary_cross_entropy(
            recon_x, x.view(-1, 3072), reduction='sum')
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE
        numpy_embedding = output_embedding.detach().cpu().numpy()
        predicted_clusters = self._cluster_obj.predict(
            numpy_embedding).reshape(-1)
        cluster_centers = self._cluster_obj.cluster_centers_
        predicted_centers = cluster_centers[predicted_clusters]
        predicted_centers = torch.Tensor(predicted_centers).to(DEVICE)

        _lambda = 0.05
        cluster_error = _lambda * torch.norm(
            output_embedding - predicted_centers).pow(2)
        loss += cluster_error

        ctx.bce += BCE
        ctx.cluster_error += cluster_error
        return loss
