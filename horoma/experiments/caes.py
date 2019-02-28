from types import SimpleNamespace

from horoma.cfg import DEVICE
from horoma.experiments import HoromaExperiment


class CAESExperiment(HoromaExperiment):

    def _train_embedding(self, train_train_loader, train_valid_loader, epochs, start_epoch, valid_train_loader, valid_valid_loader):
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
            for batch, data in enumerate(train_train_loader):
                ctx.batch = batch
                data = data.to(DEVICE)

                # before_forwardp can add second layer of transformation
                data = self.before_forwardp(ctx, data)

                # zero out previous gradient
                self._embedding_model.zero_grad()

                outputs = self._embedding_model(data)
                ctx.running_loss += self._embedding_model.get_loss()

                self.after_forwardp(ctx, outputs, data)
            self.after_train(ctx)
