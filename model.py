import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """Convolutional VAE for square grayscale images.

    The encoder halves the spatial size at every layer until it reaches 4x4,
    so img_size must be a power of 2 (e.g. 64, 128).
    """

    def __init__(self, img_size=128, in_channels=1, latent_dim=128, base_channels=32):
        super().__init__()
        n_layers = int(math.log2(img_size)) - 2
        assert 2 ** (n_layers + 2) == img_size, "img_size must be a power of 2"
        self.latent_dim = latent_dim

        # ----- Encoder: (B, 1, H, W) -> (B, C, 4, 4) -----
        channels = [min(base_channels * 2 ** i, 512) for i in range(n_layers)]
        enc, ch_in = [], in_channels
        for ch_out in channels:
            enc += [nn.Conv2d(ch_in, ch_out, 4, 2, 1), nn.BatchNorm2d(ch_out), nn.LeakyReLU(0.2)]
            ch_in = ch_out
        self.encoder = nn.Sequential(*enc)

        self.feat_shape = (channels[-1], 4, 4)
        feat_dim = channels[-1] * 4 * 4
        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)

        # ----- Decoder: z -> (B, C, 4, 4) -> (B, 1, H, W) -----
        self.fc_dec = nn.Linear(latent_dim, feat_dim)
        dec, ch_in = [], channels[-1]
        for ch_out in reversed(channels[:-1]):
            dec += [nn.ConvTranspose2d(ch_in, ch_out, 4, 2, 1), nn.BatchNorm2d(ch_out), nn.LeakyReLU(0.2)]
            ch_in = ch_out
        dec += [nn.ConvTranspose2d(ch_in, in_channels, 4, 2, 1)]  # outputs logits
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # z = mu + sigma * eps, eps ~ N(0, I)
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        """Returns logits; apply sigmoid to get pixel values in [0, 1]."""
        h = self.fc_dec(z).view(-1, *self.feat_shape)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        """Generate n images by sampling z from the prior N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return torch.sigmoid(self.decode(z))


def vae_loss(recon_logits, x, mu, logvar, beta=1.0):
    """Negative ELBO, averaged over the batch.

    recon: binary cross-entropy summed over pixels
    kl:    KL(q(z|x) || N(0, I)) in closed form
    """
    batch_size = x.size(0)
    recon = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="sum") / batch_size
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return recon + beta * kl, recon, kl


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
