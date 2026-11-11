"""generator and discriminator, and the GAN losses."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def num_stages(img_size):
    """Number of 2x resolution steps between 4x4 and img_size (4 for 64x64, 5 for 128x128)."""
    n = int(math.log2(img_size)) - 2
    assert img_size >= 16 and 2 ** (n + 2) == img_size, "img_size must be a power of 2, at least 16"
    return n


class Generator(nn.Module):
    """z (B, nz) -> image (B, 3, H, W) in [-1, 1].

    A 4x4 transposed conv turns z into a 4x4 map. Each stride-2
    transposed conv then doubles the size and halves the channels. For 64x64 and ngf=64:
        z -> 512x4x4 -> 256x8x8 -> 128x16x16 -> 64x32x32 -> 3x64x64
    BatchNorm + ReLU everywhere except the output, which is tanh.
    """

    def __init__(self, img_size=64, nz=100, ngf=64):
        super().__init__()
        self.nz = nz
        n = num_stages(img_size)
        chs = [ngf * 2 ** i for i in reversed(range(n))]  # 64x64: [512, 256, 128, 64]
        layers = [nn.ConvTranspose2d(nz, chs[0], 4, 1, 0, bias=False), nn.BatchNorm2d(chs[0]), nn.ReLU(True)]
        for c_in, c_out in zip(chs[:-1], chs[1:]):
            layers += [nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(chs[-1], 3, 4, 2, 1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z.view(z.size(0), self.nz, 1, 1))


class Discriminator(nn.Module):
    """image (B, 3, H, W) -> one real/fake logit per image, shape (B,).
    """

    def __init__(self, img_size=64, ndf=64, norm="batch"):
        super().__init__()
        assert norm in ("batch", "spectral")
        n = num_stages(img_size)
        chs = [ndf * 2 ** i for i in range(n)]  # 64x64: [64, 128, 256, 512]
        use_bn = norm == "batch"
        sn = spectral_norm if norm == "spectral" else (lambda m: m)

        layers = [sn(nn.Conv2d(3, chs[0], 4, 2, 1)), nn.LeakyReLU(0.2, True)]  # no norm on the input layer
        for c_in, c_out in zip(chs[:-1], chs[1:]):
            layers.append(sn(nn.Conv2d(c_in, c_out, 4, 2, 1, bias=not use_bn)))  # BN has its own bias
            if use_bn:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.LeakyReLU(0.2, True))
        layers.append(sn(nn.Conv2d(chs[-1], 1, 4, 1, 0)))  # 4x4 -> 1x1
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    """DCGAN initialization: conv weights ~ N(0, 0.02), BatchNorm scale ~ N(1, 0.02), biases 0."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # spectral_norm keeps the trainable weight in weight_orig; .weight is recomputed from it
        nn.init.normal_(getattr(m, "weight_orig", m.weight), 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# ----- Losses. All take raw discriminator logits. -----

def d_loss(real_logits, fake_logits, kind="bce", real_label=1.0):
    """Discriminator loss: push real logits up and fake logits down.

    bce:   -log D(x) - log(1 - D(G(z))), with D = sigmoid(logit). real_label < 1 is one-sided
           label smoothing: D is never rewarded for being overconfident on real data.
    hinge: max(0, 1 - D(x)) + max(0, 1 + D(G(z))). D stops being pushed once a sample is beyond the margin.
    """
    if kind == "bce":
        real = F.binary_cross_entropy_with_logits(real_logits, torch.full_like(real_logits, real_label))
        fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return real + fake
    if kind == "hinge":
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    raise ValueError(kind)


def g_loss(fake_logits, kind="bce"):
    """Generator loss.

    bce:   the non-saturating loss -log D(G(z)). The minimax form
           log(1 - D(G(z))) has almost no gradient early in training, when D easily rejects every fake.
    hinge: -D(G(z)).
    """
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    if kind == "hinge":
        return -fake_logits.mean()
    raise ValueError(kind)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
