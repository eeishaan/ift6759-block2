from __future__ import print_function, division
from horoma.cfg import DEVICE
import torch.nn as nn
import torch.optim as optim
import time
import copy
import numpy as np
import torch
from sklearn import warnings
from horoma.experiments import HoromaExperiment
from horoma.utils.data import HoromaDataset
from torch.utils.data import DataLoader
from torchvision import transforms

class SqueezenetExperiment(HoromaExperiment):

    def squeezenet1_0(self, pretrained=False):
        """SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
        accuracy with 50x fewer parameters and <0.5MB model size"
        <https://arxiv.org/abs/1602.07360>`_ paper.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        if pretrained:
            self._embedding_model.load_state_dict(torch.load('/home/user5/horoma/models/squeezenet1_0-a815701f.pth'))

    def _train_embedding(self,
                         train_train_loader,
                         train_train_no_aug_loader,
                         train_valid_loader,
                         epochs,
                         start_epoch,
                         valid_train_loader,
                         valid_valid_loader):

        """ Train and evaluate the model.

        Parameters
        ----------
        train and validation sets: DataLoaders
        epochs: int

        Returns
        -------
        Best _embedding_model after training
        """


        train_loader = train_train_no_aug_loader
        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        feature_extract = False
        use_pretrained = True

        self.initialize_model(feature_extract, use_pretrained)
        self._embedding_model = self._embedding_model.to(DEVICE)

        params_to_update = self._embedding_model.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name, param in self._embedding_model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in self._embedding_model.named_parameters():
                if param.requires_grad:
                    print("\t", name)

        optimizer = optim.SGD(params_to_update, lr=0.0001, momentum=0.9)

        print("=================")
        print("Training model...")
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(self._embedding_model.state_dict())
        best_acc = 0.0

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)

            self._embedding_model.train()  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for _, inputs in enumerate(train_loader):

                inputs = inputs.to(DEVICE)

                # zero the parameter gradients
                self._embedding_model.zero_grad()

                with torch.set_grad_enabled(True):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.

                    outputs = self._embedding_model(inputs)
                    outputs_copy = outputs.to('cpu')
                    output_arr = outputs_copy.detach().numpy()
                    if len(output_arr) < 2:
                        arrtmp = []
                        arrtmp.append(int(outputs_copy.tolist()[0][0]))
                        labels2 = (torch.tensor(arrtmp)).type(torch.LongTensor)
                        labels = labels2.to(self.DEVICE)
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            labels2 = (torch.from_numpy(self._cluster_obj.fit(output_arr).predict(output_arr))).type(torch.LongTensor)
                            labels = labels2.to(DEVICE)

                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            print("Loader len: ", len(train_loader.dataset))
            time_elapsed = time.time() - since
            print('Time: {:.0f}m Loss: {:.4f} Acc: {:.4f}'.format(time_elapsed // 60, epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(self._embedding_model.state_dict())
                val_acc_history.append(epoch_acc)

        print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load last model weights
        self._embedding_model.load_state_dict(best_model_wts)

        valid_dataset = HoromaDataset(split='valid', transform=transforms.ToTensor())
        valid_loader = DataLoader(valid_dataset, batch_size=100, shuffle=False)
        print("=================")

        print("Evaluating model...")

        self.eval_model(self._embedding_model, valid_loader, optimizer)


    def set_parameter_requires_grad(self, feature_extracting):
        """  Gather the parameters to be optimized/updated in this run. If we are
           finetuning we will be updating all parameters. However, if we are
           doing feature extract method, we will only update the parameters
           that we have just initialized, i.e. the parameters with requires_grad
           is True.

           Parameters
           ----------
           feature_extracting: boolean

           Returns
           -------
           model parameters with requires_grad setup
           """
        if feature_extracting:
            for param in self._embedding_model.parameters():
                param.requires_grad = False

    def initialize_model(self, feature_extract, use_pretrained):
        """  Initialize the model
          Parameters
          ----------
          feature_extracting: boolean - Defines if the model parameters requires grad or not. For finetunning this
                                        parameter should be false.
          use_pretrained: boolean - Define if it will use a pre-trained model or not.

          Returns
          -------
          Model initialized
          """
        self.squeezenet1_0(pretrained=use_pretrained)
        self.set_parameter_requires_grad(feature_extract)
        self._embedding_model.classifier[1] = nn.Conv2d(512, self._cluster_obj.n_clusters, kernel_size=(1, 1), stride=(1, 1))
        self._embedding_model.num_classes = self._cluster_obj.n_clusters

    def eval_model(self, model, valid_loader, optimizer):

        """  Evaluate the model
             Parameters
             ----------
             Validation loader: Dataset - Validation dataset to evaluate the model.

             Returns
             -------
             Validation set length: int
             Accuracy of the Validation set: int
             """
        model.eval()  # Set model to evaluation  mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in valid_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                # Get model outputs and calculate loss
                # Special case for inception because in training it has an auxiliary output. In train
                #   mode we calculate the loss by summing the final output and the auxiliary output
                #   but in testing we only consider the final output.
                outputs = model(inputs)
                labels_copy = labels.to('cpu')
                labels2 = labels_copy.detach().numpy()
                labels2 = list(np.squeeze(labels2.astype('int64'), axis=1))
                labels = torch.from_numpy(np.asarray(labels2))
                labels = labels.type(torch.LongTensor)
                labels = labels.to(DEVICE)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_acc = running_corrects.double() / len(valid_loader.dataset)
        print("Loader len: ", len(valid_loader.dataset))

        print()

        print('Valid Acc: {:4f}'.format(epoch_acc))


