from types import SimpleNamespace

import torch
from torch.nn import functional as F

from horoma.cfg import DEVICE
from horoma.experiments import HoromaExperiment


class VAEExperiment(HoromaExperiment):
    def after_train(self, ctx):
        super().after_train(ctx)
        print("BCE loss: {} Cluster loss: {}".format(
            ctx.bce.item(), ctx.cluster_error.item()))

        # compute validation loss
        self._embedding_model.eval()
        eval_ctx = SimpleNamespace(
            bce=0,
            cluster_error=0,
            running_loss=0
        )
        with torch.no_grad():
            for data in ctx.train_valid_loader:
                data = data.to(DEVICE)
                outputs = self._embedding_model(data)
                loss = self.compute_loss(eval_ctx, outputs, data)
                eval_ctx.running_loss += loss
        eval_ctx.running_loss /= len(ctx.train_valid_loader)
        print("Eval loss:{} BCE: {} Cluster loss: {}".format(
            ctx.running_loss.item(), ctx.bce.item(), ctx.cluster_error.item()))
        if eval_ctx.cluster_error > ctx.cluster_error:
            return False
        return True

    def before_forwardp(self, ctx, data):
        ctx.bce = 0
        ctx.cluster_error = 0
        return data

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

        _lambda = 50000
        cluster_error = _lambda * torch.norm(
            output_embedding - predicted_centers).pow(2)
        loss += cluster_error

        ctx.bce += BCE/len(x)
        ctx.cluster_error += cluster_error/len(x)
        return loss
