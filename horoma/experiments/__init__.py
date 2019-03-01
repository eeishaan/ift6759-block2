#!/usr/bin/env python3

from types import SimpleNamespace

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted

from horoma.cfg import DEVICE
from horoma.constants import TrainMode
from horoma.utils.score import compute_metrics


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

        # intialize a schedule for lr decay
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._embedding_optim) \
            if self._embedding_optim is not None else None

    def _remap(self, x):
        return self._cluster_label_mapping[x]

    def after_forwardp(self, ctx, outputs, data):
        pass

    def after_minibatch_test(self, ctx, outputs):
        # map them to correct labels
        predictions = [self._remap(x) for x in outputs]
        ctx.predictions.extend(predictions)

    def after_test(self, ctx):
        return np.array(ctx.predictions)

    def after_train(self, ctx):
        # save embedding model after 10 epochs
        if ctx.epoch % 10 != 9:
            self.save_experiment(ctx, save_embedding=True, save_cluster=False)

        # print loss
        message = "Epoch: {} Train Loss: {}".format(
            ctx.epoch, ctx.running_loss.item())
        print(message)

    def before_backprop(self, ctx, outputs, data):
        pass

    def before_forwardp(self, ctx, data):
        return data

    def before_test(self, ctx):
        ctx.predictions = []

    def before_train(self, ctx, train_train_loader):
        print("Starting epoch {}".format(ctx.epoch))

        # train cluster on learnt or random embedding
        v_f1 = self._cluster_obj(train_train_loader,
                                 ctx.valid_train_loader, ctx.valid_valid_loader)
        self.lr_scheduler.step(v_f1)

    def compute_loss(self, ctx, outputs, labels):
        loss = self._embedding_crit(outputs, labels)
        return loss

    def load_experiment(self, load_embedding=True, load_cluster=True):
        if load_embedding:
            checkpoint = torch.load(self._embedding_file)
            self._embedding_model.load_state_dict(
                checkpoint['model_state_dict'])
            if self._embedding_optim is not None and \
                    'optimizer_state_dict' in checkpoint \
                    and hasattr(self, '_embedding_optim'):
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
                embedding = self._embedding_model.embedding(data)
                predictions = self._cluster_obj.predict(embedding)
                self.after_minibatch_test(ctx, predictions)
        return self.after_test(ctx)

    def _train_embedding(self,
                         train_train_loader,
                         train_valid_loader,
                         epochs,
                         start_epoch,
                         valid_train_loader,
                         valid_valid_loader):
        for epoch in range(start_epoch, epochs):
            # first train embedding model
            self._embedding_model.train()

            # prepare context for hooks
            ctx = SimpleNamespace(
                epoch=epoch,
                batch=0,
                running_loss=0,
                valid_train_loader=valid_train_loader,
                valid_valid_loader=valid_valid_loader,
            )

            self.before_train(ctx, train_train_loader)
            for batch, data in enumerate(train_train_loader):
                ctx.batch = batch
                data = data.to(DEVICE)

                # before_forwardp can add second layer of transformation
                data = self.before_forwardp(ctx, data)

                # zero out previous gradient
                self._embedding_model.zero_grad()

                outputs = self._embedding_model(data)
                self.before_backprop(ctx, outputs, data)
                loss = self.compute_loss(ctx, outputs, data)
                ctx.running_loss += loss
                loss.backward()
                self._embedding_optim.step()

                self.after_forwardp(ctx, outputs, data)

            # Divide the loss by the number of batches
            ctx.running_loss /= len(train_train_loader)
            self.after_train(ctx)

    def _train_cluster(self,
                       train_train_loader,
                       valid_train_loader,
                       valid_valid_loader,
                       no_save=False):
        '''
        Train cluster and evaluate performance metrics on validation data
        '''
        # fit the cluster on train_data
        embeddings = []
        self._embedding_model.eval()
        with torch.no_grad():
            for data in train_train_loader:
                data = data.to(DEVICE)
                data_embedding = self._embedding_model.embedding(data)
                embeddings.extend(data_embedding.tolist())
        self._cluster_obj.fit(embeddings)

        # learn the cluster labels using valid_train data

        # get validation data embedding
        true_labels = []
        embeddings = []
        self._embedding_model.eval()
        with torch.no_grad():
            for data, labels in valid_train_loader:
                data = data.to(DEVICE)
                true_labels.extend(labels.int().view(-1).tolist())
                data_embedding = self._embedding_model.embedding(data)
                embeddings.extend(data_embedding.tolist())
        true_labels = np.array(true_labels)

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

        # coditionally save the model
        if not no_save:
            self.save_experiment(None, save_embedding=False)

        # evaluate cluster on valid_train set
        predicted_labels = [self._remap(x) for x in predicted_labels]
        acc, f1, ari = compute_metrics(true_labels, predicted_labels)
        print("Validation Train Acc: {:.4f} F1 score: {:.4f} ARI: {:.4f}".format(
            acc, f1, ari))

        # evaluate performance on valid_valid set
        true_labels = []
        embeddings = []
        self._embedding_model.eval()
        with torch.no_grad():
            for data, labels in valid_valid_loader:
                data = data.to(DEVICE)
                true_labels.extend(labels.int().view(-1).tolist())
                data_embedding = self._embedding_model.embedding(data)
                embeddings.extend(data_embedding.tolist())
        true_labels = np.array(true_labels)
        predicted_labels = [self._remap(x) for x in predicted_labels]
        acc, f1, ari = compute_metrics(true_labels, predicted_labels)
        print("Validation Valid Acc: {:.4f} F1 score: {:.4f} ARI: {:.4f}".format(
            acc, f1, ari))

        # return f1 for LR decay
        return f1

    def train(self, train_train_loader, train_valid_loader, epochs,
              valid_train_loader=None, valid_valid_loader=None,
              start_epoch=None, mode=TrainMode.TRAIN_ALL):
        # set start_epoch differently if you want to resume training from a
        # checkpoint.
        start_epoch = start_epoch \
            if start_epoch is not None \
            else self._start_epoch

        if mode == TrainMode.TRAIN_ONLY_CLUSTER:
            self.load_experiment(load_embedding=True, load_cluster=False)
        else:
            self._train_embedding(train_train_loader, train_valid_loader, epochs,
                                  start_epoch, valid_train_loader,
                                  valid_valid_loader)

        if mode == TrainMode.TRAIN_ONLY_EMBEDDING:
            return
        if valid_train_loader is None:
            err = 'ERROR: Validation dataset is required for' +\
                ' training cluster model'
            print(err)
            return

        self._train_cluster(train_train_loader,
                            valid_train_loader, valid_valid_loader)
