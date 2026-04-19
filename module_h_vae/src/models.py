import torch
import torch.nn as nn


class ConvVAE(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        base_channels: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # 28 -> 14 -> 7 -> 4
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, c1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),

            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),

            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

        self.feature_shape = (c3, 4, 4)
        self.flatten_dim = c3 * 4 * 4

        self.fc_hidden = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.flatten_dim),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, kernel_size=3, stride=2, padding=1, output_padding=0),  # 4 -> 7
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(c1, input_channels, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(x.size(0), -1)
        h = self.fc_hidden(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(z.size(0), *self.feature_shape)
        x_recon = self.decoder(h)
        if x_recon.size(-1) != 28 or x_recon.size(-2) != 28:
            x_recon = x_recon[:, :, :28, :28]
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar