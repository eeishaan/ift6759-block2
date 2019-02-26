import torch
from torch import nn
from torch.nn import functional as F


class CVAE(nn.Module):
    """
    Reference: https://github.com/FelixMohr/Deep-learning-with-Python/blob/master/VAE.ipynb
    """

    def __init__(self):
        super(CVAE, self).__init__()

        self.latent_dim = 32
        self.hidden_size = 2
        self.hidden_filter = 128
        self.hidden_dim = self.hidden_size ** 2 * self.hidden_filter
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=4, stride=2),  # 15
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=4, stride=2),  # 6
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=4, stride=2),  # 2
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.fc11 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc12 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=4, stride=2),  # 6
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=5, stride=2),  # 15
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.ConvTranspose2d(in_channels=32, out_channels=3,
                               kernel_size=4, stride=2),  # 32
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, self.hidden_dim)
        return self.fc11(h), self.fc12(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h = F.relu(self.fc2(z))
        h = h.view(-1, self.hidden_filter, self.hidden_size, self.hidden_size)
        h = self.decoder(h)
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def embedding(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z.detach().cpu().numpy()
