from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from horoma.cfg import DEVICE
from horoma.constants import TrainMode
from horoma.experiments import HoromaExperiment
from horoma.utils.score import compute_metrics


class CAESExperiment(HoromaExperiment):

    def _train_embedding(self, train_loader, epochs, start_epoch, valid_train_loader, valid_valid_loader):
        for epoch in range(start_epoch, epochs):

            # first train embedding model
            self._embedding_model.train()

            # prepare context for hooks
            ctx = SimpleNamespace(
                epoch=epoch,
                running_loss=0,
                valid_train_loader=valid_train_loader,
                valid_valid_loader=valid_valid_loader
            )

            self.before_train(ctx)
            for _, data in enumerate(train_loader):

                data = data.to(DEVICE)

                # before_forwardp can add second layer of transformation
                data = self.before_forwardp(ctx, data)

                # zero out previous gradient
                self._embedding_model.zero_grad()

                outputs = self._embedding_model(data)
                #loss = self.compute_loss(ctx, outputs, data)
                ctx.running_loss += self._embedding_model.get_loss()
                #loss.backward()
                #self._embedding_optim.step()

                self.after_forwardp(ctx, outputs, data)
            self.after_train(ctx)
