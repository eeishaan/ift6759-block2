
import torch.nn as nn
import torch.optim as optim

from horoma.utils.loss import vae_loss, cvae_loss


def optim_factory(optimizer_name, params):
    return getattr(optim, optimizer_name)(**params)


def crit_factory(criterion_name, params):
    loss_map = {
        'VAE_LOSS': vae_loss,
        'CVAE_LOSS': cvae_loss,
    }
    if criterion_name in loss_map:
        return loss_map[criterion_name]
    return getattr(nn, criterion_name)(**params)
