import torch
from torch import nn
from torch.nn import functional as F


class AE(nn.Module):
    """
    Reference: https://github.com/pytorch/examples/blob/master/vae/main.py
    """

    def __init__(self):
        super(AE, self).__init__()

        latent_dim = 8
        self.fc1 = nn.Linear(3072, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 3072)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 3072))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def embedding(self, x):
        with torch.no_grad():
            mu, logvar = self.encode(x.view(-1, 3072))
            z = self.reparameterize(mu, logvar)
            return z.detach().cpu().numpy()
