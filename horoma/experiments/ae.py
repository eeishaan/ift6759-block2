from types import SimpleNamespace

import torch
from torch.nn import functional as F

from horoma.cfg import DEVICE
from horoma.experiments import HoromaExperiment


class AEExperiment(HoromaExperiment):
    def after_train(self, ctx):
        super().after_train(ctx)
        epoch = ctx.epoch
        bce_loss = ctx.bce.item()
        cluster_loss = ctx.cluster_error.item()
        print("BCE loss: {} Cluster loss: {}".format(
            ctx.bce.item(), ctx.cluster_error.item()))
        self._summary_writer.add_scalar(
            'train_train_bce_loss', bce_loss, epoch)
        self._summary_writer.add_scalar(
            'train_train_cluster_loss', cluster_loss, epoch)
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
        eval_running_loss = eval_ctx.running_loss.item()
        eval_bce_loss = eval_ctx.bce.item()
        eval_cluster_loss = eval_ctx.cluster_error.item()
        print("BCE: {} Cluster loss: {} Eval loss:{} "
              .format(eval_bce_loss,
                      eval_cluster_loss,
                      eval_running_loss))
        self._summary_writer.add_scalar(
            'train_valid_bce_loss', eval_bce_loss, epoch)
        self._summary_writer.add_scalar(
            'train_valid_cluster_loss', eval_cluster_loss, epoch)
        self._summary_writer.add_scalar(
            'train_valid_loss', eval_running_loss, epoch)
        if eval_ctx.running_loss > ctx.running_loss:
            return False
        return True

    def before_train(self, ctx, train_train_no_aug_loader):
        super().before_train(ctx, train_train_no_aug_loader)
        ctx.bce = 0
        ctx.cluster_error = 0

    def compute_loss(self, ctx, outputs, x):
        recon_x, _, _, output_embedding = outputs
        BCE = F.binary_cross_entropy(
            recon_x, x.view(-1, 3072), reduction='sum')
        loss = BCE
        numpy_embedding = output_embedding.detach().cpu().numpy()
        predicted_clusters = self._cluster_obj.predict(
            numpy_embedding).reshape(-1)
        cluster_centers = self._cluster_obj.cluster_centers_ \
            if hasattr(self._cluster_obj, 'cluster_centers_') \
            else self._cluster_obj.means_
        predicted_centers = cluster_centers[predicted_clusters]
        predicted_centers = torch.Tensor(predicted_centers).to(DEVICE)

        _lambda = 50000
        cluster_error = _lambda * torch.norm(
            output_embedding - predicted_centers).pow(2)
        loss += cluster_error

        ctx.bce += BCE / len(x)
        ctx.cluster_error += cluster_error / len(x)
        return loss


class VAEExperiment(HoromaExperiment):
    def after_train(self, ctx):
        super().after_train(ctx)
        epoch = ctx.epoch
        bce_loss, kld_loss, cluster_loss = ctx.bce.item(
        ), ctx.kld.item(), ctx.cluster_error.item()
        print("BCE loss: {} KLD Loss: {}, Cluster loss: {} ".format(
            bce_loss, kld_loss, cluster_loss))
        self._summary_writer.add_scalar(
            'train_train_bce_loss', bce_loss, epoch)
        self._summary_writer.add_scalar(
            'train_train_cluster_loss', cluster_loss, epoch)
        self._summary_writer.add_scalar(
            'train_train_kld_loss', kld_loss, epoch)
        # compute validation loss
        self._embedding_model.eval()
        eval_ctx = SimpleNamespace(
            bce=0,
            cluster_error=0,
            running_loss=0,
            kld=0,
        )
        with torch.no_grad():
            for data in ctx.train_valid_loader:
                data = data.to(DEVICE)
                outputs = self._embedding_model(data)
                loss = self.compute_loss(eval_ctx, outputs, data)
                eval_ctx.running_loss += loss
        eval_ctx.running_loss /= len(ctx.train_valid_loader)
        eval_running_loss = eval_ctx.running_loss.item()
        eval_bce_loss = eval_ctx.bce.item()
        eval_cluster_loss = eval_ctx.cluster_error.item()
        eval_kld_loss = eval_ctx.kld.item()
        print("BCE: {} KDL: {} Cluster loss: {} Eval loss:{} "
              .format(eval_bce_loss,
                      eval_kld_loss,
                      eval_cluster_loss,
                      eval_running_loss))
        self._summary_writer.add_scalar(
            'train_valid_bce_loss', eval_bce_loss, epoch)
        self._summary_writer.add_scalar(
            'train_valid_cluster_loss', eval_cluster_loss, epoch)
        self._summary_writer.add_scalar(
            'train_valid_loss', eval_running_loss, epoch)
        self._summary_writer.add_scalar(
            'train_valid_kld_loss', eval_kld_loss, epoch)

        if eval_ctx.running_loss > ctx.running_loss:
            return False
        return True

    def before_train(self, ctx, train_train_no_aug_loader):
        super().before_train(ctx, train_train_no_aug_loader)
        ctx.bce = 0
        ctx.cluster_error = 0
        ctx.kld = 0

    def compute_loss(self, ctx, outputs, x):
        recon_x, mu, logvar, output_embedding = outputs
        BCE = F.binary_cross_entropy(
            recon_x, x.view(-1, 3072), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD
        numpy_embedding = output_embedding.detach().cpu().numpy()
        predicted_clusters = self._cluster_obj.predict(
            numpy_embedding).reshape(-1)
        cluster_centers = self._cluster_obj.cluster_centers_ \
            if hasattr(self._cluster_obj, 'cluster_centers_') \
            else self._cluster_obj.means_
        predicted_centers = cluster_centers[predicted_clusters]
        predicted_centers = torch.Tensor(predicted_centers).to(DEVICE)

        _lambda = 50000
        cluster_error = _lambda * torch.norm(
            output_embedding - predicted_centers).pow(2)
        loss += cluster_error

        ctx.bce += BCE / len(x)
        ctx.kld += KLD / len(x)
        ctx.cluster_error += cluster_error / len(x)
        return loss
