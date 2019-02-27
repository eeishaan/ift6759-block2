import torch
from torch.nn import functional as F

from horoma.cfg import DEVICE
from horoma.experiments import HoromaExperiment


class VAEExperiment(HoromaExperiment):
    def compute_loss(self, ctx, outputs, x):
        recon_x, mu, logvar, output_embedding = outputs
        BCE = F.binary_cross_entropy(
            recon_x, x.view(-1, 3072), reduction='sum')
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE
        # cluster is fit after first backward pass
        if ctx.batch != 0:
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
        return loss
