from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from horoma.cfg import DEVICE
from horoma.constants import TrainMode
from horoma.experiments import HoromaExperiment
from horoma.utils.score import compute_metrics


class CVAEExperiment(HoromaExperiment):

    def compute_loss(self, ctx, outputs, x, predicted_clusters):
        recon_x, mu, logvar, output_embedding = outputs
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        _lambda = 0.05
        cluster_error = _lambda * torch.norm(
            output_embedding - predicted_clusters).pow(2)
        return BCE + KLD + cluster_error

    def _train_embedding(self, train_loader, epochs, start_epoch, valid_loader):
        for epoch in range(start_epoch, epochs):

            # first train embedding model
            self._embedding_model.train()

            # prepare context for hooks
            ctx = SimpleNamespace(
                epoch=epoch,
                running_loss=0,
                valid_loader=valid_loader
            )

            self.before_train(ctx)
            for _, data in enumerate(train_loader):

                data = data.to(DEVICE)

                # zero out previous gradient
                self._embedding_model.zero_grad()

                outputs = self._embedding_model(data)
                fit_fn = getattr(self._cluster_obj, 'partial_fit', None) or \
                    getattr(self._cluster_obj, 'fit')

                output_embedding = outputs[3]
                numpy_data = output_embedding.detach().cpu().numpy()
                # fit the cluster
                fit_fn(numpy_data)

                # get the cluster center points
                predicted_clusters = self._cluster_obj.predict(
                    numpy_data).reshape(-1)
                cluster_centers = self._cluster_obj.cluster_centers_
                predicted_centers = cluster_centers[predicted_clusters]
                predicted_centers = torch.Tensor(predicted_centers).to(DEVICE)

                loss = self.compute_loss(
                    ctx, outputs, data, predicted_centers)
                ctx.running_loss += loss
                loss.backward()
                self._embedding_optim.step()

                self.after_forwardp(ctx, outputs, data)
            self.after_train(ctx)

    def _train_cluster(self, valid_loader, no_save=False):

        # get validation data embedding
        true_labels = []
        embeddings = []
        self._embedding_model.eval()
        for data, labels in valid_loader:
            data = data.to(DEVICE)
            true_labels.extend(labels.int().view(-1).tolist())
            data_embedding = self._embedding_model.embedding(data)
            embeddings.extend(data_embedding.tolist())
        true_labels = np.array(true_labels)

        # fit the cluster
        predicted_labels = self._cluster_obj.predict(embeddings)
        self._cluster_label_mapping = {}

        # get number of clusters for GMM or Kmeans
        n_clusters = getattr(self._cluster_obj, 'weights_', None)
        if n_clusters is None:
            n_clusters = getattr(self._cluster_obj, 'cluster_centers_', None)
        n_clusters = n_clusters.shape[0]

        # construct the cluster_label mapping
        for i in range(n_clusters):
            # filter data which was predicted to be in ith cluster and
            # get their true label
            idx = np.where(predicted_labels == i)[0]
            if len(idx) != 0:
                labels_freq = np.bincount(true_labels[idx])
                self._cluster_label_mapping[i] = np.argmax(labels_freq)
            else:
                # No validation point found on this cluster. We can't label it.
                self._cluster_label_mapping[i] = -1

        if not no_save:
            self.save_experiment(None, save_embedding=False)

        predicted_labels = [self._remap(x) for x in predicted_labels]
        acc, f1, ari = compute_metrics(true_labels, predicted_labels)
        print("Validation Acc: {:.4f} F1 score: {:.4f} ARI: {:.4f}".format(
            acc, f1, ari))
