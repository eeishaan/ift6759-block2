import torch
from torch.nn import functional as F

from horoma.cfg import DEVICE
from horoma.experiments.ae import AEExperiment


class CAEExperiment(AEExperiment):
    def compute_loss(self, ctx, outputs, x):
        recon_x, _, _, output_embedding = outputs
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        loss = BCE

        if not self._is_naive():
            numpy_embedding = output_embedding.detach().cpu().numpy()
            predicted_clusters = self._cluster_obj.predict(
                numpy_embedding).reshape(-1)
            cluster_centers = self._cluster_obj.cluster_centers_ \
                if hasattr(self._cluster_obj, 'cluster_centers_') \
                else self._cluster_obj.means_
            predicted_centers = cluster_centers[predicted_clusters]
            predicted_centers = torch.Tensor(predicted_centers).to(DEVICE)

            _lambda = 0.05
            cluster_error = _lambda * torch.norm(
                output_embedding - predicted_centers).pow(2)
            loss += cluster_error
            ctx.cluster_error += cluster_error / len(x)

        ctx.bce += BCE / len(x)
        return loss
