"""FID and Inception Score with PyTorch-Ignite
"""
import torch
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
from PIL import Image
from torchvision import transforms

from dataset import to_unit


def load_inception(device):
    return InceptionModel(return_features=False, device=device)


def interpolate(batch):
    """(B, 3, h, w) in [0, 1] -> (B, 3, 299, 299) in [0, 1], exactly as the blog's interpolate()."""
    arr = []
    for img in batch.cpu():
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299, 299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)


@torch.no_grad()
def inception_outputs(inception, batches, device):
    """Inception softmax (N, 1000) of every image in an iterable of [0, 1] batches."""
    return torch.cat([inception(interpolate(x).to(device)).cpu() for x in batches])


@torch.no_grad()
def generate(G, n, batch_size, device, seed=None):
    """Yields n generated images in [0, 1], in batches. G is put in eval mode and restored afterwards."""
    was_training = G.training
    G.eval()
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    for i in range(0, n, batch_size):
        z = torch.randn(min(batch_size, n - i), G.nz, device=device, generator=gen)
        yield to_unit(G(z))
    G.train(was_training)


def real_batches(dataset, indices, batch_size):
    """Yields dataset[indices] in [0, 1], in batches (the dataset should have flip=False)."""
    for i in range(0, len(indices), batch_size):
        yield to_unit(torch.stack([dataset[j] for j in indices[i:i + batch_size]]))


def fid(a, b, batch_size=256, device="cpu"):
    """ignite FID between two equal-sized sets of precomputed Inception outputs."""
    assert len(a) == len(b), "ignite's FID needs the same number of samples on both sides"
    metric = FID(num_features=a.shape[1], feature_extractor=torch.nn.Identity(), device=device)
    for i in range(0, len(a), batch_size):
        metric.update((a[i:i + batch_size], b[i:i + batch_size]))
    return metric.compute()


def inception_score(probs, batch_size=256, device="cpu"):
    metric = InceptionScore(num_features=probs.shape[1], feature_extractor=torch.nn.Identity(), device=device)
    for i in range(0, len(probs), batch_size):
        metric.update(probs[i:i + batch_size])
    return metric.compute()
