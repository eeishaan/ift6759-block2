import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class CDAutoEncoder(nn.Module):
    r"""
    Convolutional denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.
    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
        stride: Stride of the convolutional layers.
    """
    def __init__(self, input_size, output_size, stride):
        super(CDAutoEncoder, self).__init__()

        self.forward_pass = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=2, stride=stride, padding=0),
            nn.ReLU(),
        )
        self.backward_pass = nn.Sequential(
            nn.ConvTranspose2d(output_size, input_size, kernel_size=2, stride=stride, padding=0), 
            nn.ReLU(),
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        self.current_loss = 0

    def forward(self, x):
        # Train each autoencoder individually
        x = x.detach()
        # Add noise, but use the original lossless input as the target.
        x_noisy = x * (x.data.new(x.size()).normal_(0, 0.1) > -.1).type_as(x)
        y = self.forward_pass(x_noisy)
        if self.training:
            x_reconstruct = self.backward_pass(y)
            x.requires_grad = False
            loss = self.criterion(x_reconstruct, x)
            self.current_loss = loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return y.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)


class CAES(nn.Module):
    r"""
    A stacked autoencoder made from the convolutional denoising autoencoders.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self):
        super(CAES, self).__init__()

        self.ae1 = CDAutoEncoder(3, 16, 2)
        self.ae2 = CDAutoEncoder(16, 32, 2)
        self.ae3 = CDAutoEncoder(32, 64, 2)
        self.ae4 = CDAutoEncoder(64, 32, 2)
        self.ae5 = CDAutoEncoder(32, 32, 2)

    def forward(self, x):
        a1 = self.ae1(x)
        a2 = self.ae2(a1)
        a3 = self.ae3(a2)
        a4 = self.ae4(a3)
        a5 = self.ae5(a4)

        if self.training:
            return a5

        else:
            return a5, self.reconstruct(a5)

    def reconstruct(self, x):
        a4_reconstruct = self.ae5.reconstruct(x)
        a3_reconstruct = self.ae4.reconstruct(a4_reconstruct)
        a2_reconstruct = self.ae3.reconstruct(a3_reconstruct)
        a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
        x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
        return x_reconstruct
    
    def get_loss(self):
        return self.ae1.current_loss + self.ae2.current_loss + self.ae3.current_loss + self.ae4.current_loss + self.ae5.current_loss

    def embedding(self, x):
        self.eval()
        embedding, _ = self.forward(x)
        embedding = embedding.view(embedding.shape[0], -1)
        return embedding.cpu().numpy()
