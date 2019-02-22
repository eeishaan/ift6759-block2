import torch
import torch.nn.functional as F


def vae_loss(outputs, x):
    """
    Reference: https://github.com/pytorch/examples/blob/master/vae/main.py
    """
    recon_x, mu, logvar = outputs
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3072), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
