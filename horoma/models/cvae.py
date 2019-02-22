import torch
from torch import nn
from torch.nn import functional as F


class CVAE(nn.Module):
    """
    Reference: https://github.com/FelixMohr/Deep-learning-with-Python/blob/master/VAE.ipynb
    """

    def __init__(self):
        super(CVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc11 = nn.Linear(8*8*64, 8)
        self.fc12 = nn.Linear(8*8*64, 8)
        self.fc2 = nn.Linear(8, 64*8)
        self.fc3 = nn.Linear(64*8, 64*8*8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64,
                               kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(in_channels=32, out_channels=3,
                               kernel_size=4, padding=1, stride=2),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, 8*8*64)
        return self.fc11(h), self.fc12(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h = F.leaky_relu(self.fc2(z))
        h = F.leaky_relu(self.fc3(h))
        h = h.view(-1, 64, 8, 8)
        h = self.decoder(h)
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def embedding(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z.detach().cpu().numpy()
