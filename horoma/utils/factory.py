
import torch.nn as nn
import torch.optim as optim

from horoma.utils.loss import vae_loss


def optim_factory(optimizer_name, params):
    return getattr(optim, optimizer_name)(**params)


def crit_factory(criterion_name, params):
    if criterion_name == 'VAE_LOSS':
        return vae_loss
    return getattr(nn, criterion_name)(**params)
