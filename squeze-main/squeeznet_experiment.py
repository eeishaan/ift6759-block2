# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
import numpy as np
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

DATA_ROOT_FOLDER = '/rap/jvb-000-aa/COURS2019/etudiants/data/horoma'


__all__ = ['SqueezeNet', 'squeezenet1_0']


class HoromaDataset(Dataset):

    def __init__(self, data_dir=DATA_ROOT_FOLDER, split="train",
                 subset=None, skip=0, flattened=False, transform=None):
        """
        Args:
            data_dir: Path to the directory containing the samples.
            split: Which split to use. [train, valid, test]
            subset: How many elements will be used. Default: all.
            skip: How many element to skip before taking the subset.
            flattened: If True return the images in a flatten format.
            tranform: Transform data with supplied transformer
        """
        self.transform = transform
        nb_channels = 3
        height = 32
        width = 32
        datatype = "uint8"
        print("0...")
        if split == "train":
            self.nb_exemples = 150700
        elif split == "valid":
            self.nb_exemples = 480
        elif split == "test":
            self.nb_exemples = 498
        elif split == "train_overlapped":
            self.nb_exemples = 544749
        elif split == "valid_overlapped":
            self.nb_exemples = 1331
        else:
            raise(
                "Dataset: Invalid split. Must be " +
                "[train, valid, test, train_overlapped, valid_overlapped]")
        print("0.1...")
        filename_x = os.path.join(data_dir, "{}_x.dat".format(split))
        filename_y = os.path.join(data_dir, "{}_y.txt".format(split))

        self.targets = None

        if os.path.exists(filename_y) and not split.startswith("train"):
            pre_targets = np.loadtxt(filename_y, 'U2')

            if subset is None:
                pre_targets = pre_targets[skip: None]
            else:
                pre_targets = pre_targets[skip: skip + subset]

            self.map_labels = np.unique(pre_targets)
            self.targets = np.asarray(
                [np.where(self.map_labels == t)[0][0] for t in pre_targets])

        self.data = np.memmap(filename_x, dtype=datatype, mode="r", shape=(
            self.nb_exemples, height, width, nb_channels))
        if subset is None:
            self.data = self.data[skip: None]
        else:
            self.data = self.data[skip: skip + subset]

        if flattened:
            self.data = self.data.reshape(len(self.data), -1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.targets is not None:
            return self.transform(self.data[index]), torch.Tensor([self.targets[index]])
        return self.transform(self.data[index])


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)



class SqueezeImplement(object):

    def __init__(self):

        # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
        self.model_name = "squeezenet"

        # Number of classes in the dataset
        self.num_classes = 16

        # Batch size for training (change depending on how much memory you have)
        self.batch_size = 8

        # Number of epochs to train for
        self.num_epochs = 15

        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        self.feature_extract = True

        self.use_pretrained = True

        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def squeezenet1_0(self, pretrained=False, **kwargs):
        r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
        accuracy with 50x fewer parameters and <0.5MB model size"
        <https://arxiv.org/abs/1602.07360>`_ paper.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = SqueezeNet(version=1.0, **kwargs)
        if pretrained:
            model.load_state_dict(torch.load('squeezenet1_0-a815701f.pth'))
        return model


    def train(self):

        print("Initializing Datasets and Dataloaders...")

        # load data
        train_dataset = HoromaDataset(split='train', transform=transforms.ToTensor())
        valid_dataset = HoromaDataset(split='valid', transform=transforms.ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=100, shuffle=False)


        # Initialize the model for this run
        model_ft, input_size = self.initialize_model(self.model_name, self.num_classes, self.feature_extract)
        # Send the model to GPU

        model_ft = model_ft.to(self.DEVICE)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if self.feature_extract:
            params_to_update = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model_ft, hist = self.train_model(model_ft, valid_loader, criterion, optimizer_ft, self.num_epochs,False,True)


    def train_model(self, model, valid_loader, criterion, optimizer, num_epochs=25,is_inception=False,use_kmeans=True):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            model.train()  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in valid_loader:
                inputs = inputs.to(self.DEVICE)
                labels = labels.to(self.DEVICE)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if use_kmeans:
                        #print("Initial:", labels)
                        outputs = model(inputs)
                        outputs_copy = outputs.to('cpu')
                        output_arr = outputs_copy.detach().numpy()
                        if (len(output_arr) < 2):
                            arrtmp=[]
                            arrtmp.append(int(outputs.tolist()[0][0]))
                            labels2 = torch.tensor(arrtmp)
                            labels2 = labels2.type(torch.LongTensor)
                            labels = labels2.to(self.DEVICE)
                        else:
                            clf = KMeans(n_clusters=self.num_classes)
                            clf.fit(output_arr)
                            labels2 = torch.from_numpy(clf.labels_)
                            labels2 = labels2.type(torch.LongTensor)
                            labels=labels2.to(self.DEVICE)
                        loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        print("Outputs: ", outputs)
                        labels_copy = labels.to('cpu')
                        labels2 = labels_copy.detach().numpy()
                        new_labels = list(np.squeeze(labels2.astype('int64'), axis=1))
                        labels = torch.from_numpy(np.asarray(new_labels))
                        labels = labels.type(torch.LongTensor)
                        labels = labels.to(self.DEVICE)
                        print("Last labels: ", labels)
                        loss = criterion(outputs, labels)
                        print("Loss: {}".format(loss.item()))


                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase

                    loss.backward()
                    optimizer.step()

                # statistics
                print(inputs.size(0))
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(valid_loader.dataset)
            epoch_acc = running_corrects.double() / len(valid_loader.dataset)
            print("Loader len: ", len(valid_loader.dataset))
            print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(self, model_name, num_classes, feature_extract):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft=self.squeezenet1_0(pretrained=self.use_pretrained)
            print(model_ft.eval())
            self.set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = num_classes
            input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

if __name__ == '__main__':


    plt.ion()  # interactive mode# License: BSD

    squeezimp=SqueezeImplement()
    squeezimp.train()