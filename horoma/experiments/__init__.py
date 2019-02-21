#!/usr/bin/env python3

import torch
import numpy as np
from horoma.cfg import DEVICE

from types import SimpleNamespace
from horoma.constants import TrainMode


class HoromaExperiment(object):
    '''
    Base class for experiments
    '''

    def __init__(
            self,
            experiment_file,
            embedding_model,
            cluster_obj,
            embedding_optim=None,
            embedding_crit=None,
    ):
        self._embedding_file = experiment_file
        self._cluster_file = "{}_{}".format(experiment_file, '.cluster')
        self._embedding_model = embedding_model
        self._cluster_obj = cluster_obj
        self._embedding_optim = embedding_optim
        self._embedding_crit = embedding_crit
        self._cluster_label_mapping = {}
        # send model to device
        self._embedding_model.to(DEVICE)

        # initialize epoch var correctly
        self._start_epoch = 0

    def after_eval(self, ctx):
        pass

    def after_forwardp(self, ctx, outputs, labels):
        pass

    def after_minibatch_eval(self, ctx, outputs, labels):
        pass

    def after_minibatch_test(self, ctx, outputs):
        pass

    def after_test(self, ctx):
        pass

    def after_train(self, ctx):
        # save embedding after 10 epochs
        if ctx.epoch % 10 == 9:
            self.save_experiment(ctx, save_embedding=True, save_cluster=False)

    def before_minibatch_eval(self, ctx, data, labels):
        return data, labels

    def before_eval(self, ctx):
        pass

    def before_forwardp(self, ctx, data, labels):
        return data, labels

    def before_test(self, ctx):
        pass

    def before_train(self, ctx):
        pass

    def compute_loss(self, ctx, outputs, labels):
        loss = self._embedding_crit(outputs, labels)
        return loss

    def eval(self, dataloader):
        self._embedding_model.eval()
        with torch.no_grad():
            ctx = SimpleNamespace()
            self.before_eval(ctx)
            for _, (data, labels) in enumerate(dataloader):
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                data, labels = self.before_minibatch_eval(ctx, data, labels)
                outputs = self._embedding_model(data)
                self.after_minibatch_eval(ctx, outputs, labels)
            return self.after_eval(ctx)

    def load_experiment(self, load_embedding=True, load_cluster=True):
        if load_embedding:
            checkpoint = torch.load(self._embedding_file)
            self._embedding_model.load_state_dict(
                checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint and hasattr(self, '_embedding_optim'):
                self._embedding_optim.load_state_dict(
                    checkpoint['optimizer_state_dict'])
            self._start_epoch = checkpoint.get('epoch', 0)
        if load_cluster:
            checkpoint = torch.load(self._cluster_file)
            self._cluster_obj = checkpoint['cluster_obj']
            self._cluster_label_mapping = checkpoint['cluster_label_mapping']

    def save_experiment(self, ctx, save_embedding=True, save_cluster=True):
        if save_embedding:
            save_dict = {
                'epoch': ctx.epoch,
                'model_state_dict': self._embedding_model.state_dict(),
                'optimizer_state_dict': self._embedding_optim.state_dict(),
            }
            torch.save(save_dict, self._embedding_file)
        if save_cluster:
            save_dict = {
                'cluster_obj': self._cluster_obj,
                'cluster_label_mapping': self._cluster_label_mapping,
            }
            torch.save(save_dict, self._cluster_file)

    def test(self, dataloader):
        self._embedding_model.eval()
        with torch.no_grad():
            ctx = SimpleNamespace()
            self.before_test(ctx)
            for _, data in enumerate(dataloader):
                data = data.to(DEVICE)
                outputs = self._embedding_model(data)
                self.after_minibatch_test(ctx, outputs)
        return self.after_test(ctx)

    def _train_embedding(self, train_loader, epochs, start_epoch):
        for epoch in range(start_epoch, epochs):

            # first train embedding model
            self._embedding_model.train()

            # prepare context for hooks
            ctx = SimpleNamespace(
                epoch=epoch,
                running_loss=0,
            )

            self.before_train(ctx)
            for _, (data, labels) in enumerate(train_loader):

                data, labels = data.to(DEVICE), labels.to(DEVICE)

                # before_forwardp can add second layer of transformation
                data, labels = self.before_forwardp(ctx, data, labels)

                # zero out previous gradient
                self._embedding_model.zero_grad()

                outputs = self._embedding_model(data)
                loss = self.compute_loss(ctx, outputs, labels)
                ctx.running_loss += loss
                loss.backward()
                self._embedding_optim.step()

                self.after_forwardp(ctx, outputs, labels)
            self.after_train(ctx)

    def _train_cluster(self, valid_dataset):
        data, true_labels = valid_dataset

        # get validation data embedding
        self._embedding_model.eval()
        data_embedding = self._embedding_model(data)

        # fit the cluster
        predicted_labels = self._cluster_obj.fit_transform(data_embedding)

        self._cluster_label_mapping = {}

        # get number of clusters for GMM or Kmeans
        n_clusters = getattr(self._cluster_obj, 'weights_', None) or getattr(
            self._cluster_obj, 'cluster_centers_', None)
        n_clusters = n_clusters.shape[0]

        # construct the cluster_label mapping
        for i in range(n_clusters):
            # filter data which was predicted to be in ith cluster and
            # get their true label
            filtered_labels = true_labels[np.where(predicted_labels == i)]
            if len(filtered_labels) > 0:
                counts = np.bincount(filtered_labels)
                max_voted_label = np.argmax(counts)
                self._cluster_label_mapping[i] = max_voted_label
            else:
                # No validation point found on this cluster. We can't label it.
                self._cluster_label_mapping[i] = -1
        self.save_experiment(None, save_embedding=False)

    def train(self, train_loader, epochs, valid_dataset=None, start_epoch=None, mode=TrainMode.TRAIN_ALL):
        # set start_epoch differently if you want to resume training from a
        # checkpoint.
        start_epoch = start_epoch if start_epoch is not None else self._start_epoch

        if mode == TrainMode.TRAIN_ONLY_CLUSTER:
            self.load_experiment(load_embedding=True, load_cluster=False)
        else:
            self._train_embedding(train_loader, epochs, start_epoch)

        if mode == TrainMode.TRAIN_ONLY_EMBEDDING:
            return
        if valid_dataset is None:
            err = 'ERROR: Validation dataset is required for training cluster model'
            print(err)
            return

        self._train_cluster(valid_dataset)
