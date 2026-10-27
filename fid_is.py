import argparse
import json
import os

import torch
import torch.nn.functional as F
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
from torchvision.utils import save_image

from dataset import get_dataloader
from model import VAE, get_device

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser(description="FID and Inception Score of VAE samples, with real-vs-real baselines.")
    p.add_argument("--ckpt", default="outputs/run2/best.pt")
    p.add_argument("--data_root", default="data/chest_xray")
    p.add_argument("--n_fake", type=int, default=None, help="number of generated images; default = size of train split")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def to_inception_input(x):
    """(B, 1, H, W) grayscale in [0, 1] -> (B, 3, 299, 299) ImageNet-normalized.

    Inception needs >= 299x299 RGB. The ignite blog resizes with PIL bilinear; since we only upsample
    (128 -> 299), torch bilinear is equivalent. torchvision's pretrained inception_v3 has
    transform_input=True, which expects ImageNet-normalized input.
    """
    x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def inception_probs(inception, batches, device):
    """Runs ignite's default Inception feature extractor (1000-way softmax) over an iterable of image batches."""
    return torch.cat([inception(to_inception_input(x.to(device))).cpu() for x in batches])


@torch.no_grad()
def generate(model, n, batch_size, device):
    for i in range(0, n, batch_size):
        z = torch.randn(min(batch_size, n - i), model.latent_dim, device=device)
        yield torch.sigmoid(model.decode(z))


def fid(a, b, batch_size, device):
    """ignite FID between two equal-sized sets of precomputed Inception outputs."""
    assert len(a) == len(b), "ignite's FID needs the same number of samples on both sides"
    metric = FID(num_features=a.shape[1], feature_extractor=torch.nn.Identity(), device=device)
    for i in range(0, len(a), batch_size):
        metric.update((a[i:i + batch_size], b[i:i + batch_size]))
    return metric.compute()


def inception_score(probs, batch_size, device):
    metric = InceptionScore(num_features=probs.shape[1], feature_extractor=torch.nn.Identity(), device=device)
    for i in range(0, len(probs), batch_size):
        metric.update(probs[i:i + batch_size])
    return metric.compute()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    targs = ckpt["args"]
    model = VAE(targs["img_size"], 1, targs["latent_dim"], targs["base_channels"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Real images go through the same 128x128 pipeline as training, so real and fake differ only in content
    load = lambda split: get_dataloader(args.data_root, split, targs["img_size"], args.batch_size, False, args.num_workers)
    train_loader, test_loader = load("train"), load("test")
    n_train, n_test = len(train_loader.dataset), len(test_loader.dataset)
    n_fake = args.n_fake or n_train

    inception = InceptionModel(return_features=False, device=device)
    print(f"device: {device} | train {n_train}, test {n_test}, generated {n_fake}")
    train_p = inception_probs(inception, (x for x, _ in train_loader), device)
    test_p = inception_probs(inception, (x for x, _ in test_loader), device)
    fake_p = inception_probs(inception, generate(model, n_fake, args.batch_size, device), device)

    # FID is biased upward for small N, so compare everything at matched sizes:
    # the full-size score, and a size-n_test score next to the real test-vs-train baseline.
    g = torch.Generator().manual_seed(args.seed)
    train_sub = train_p[torch.randperm(n_train, generator=g)[:n_test]]
    n_big = min(n_fake, n_train)
    results = {
        "ckpt": args.ckpt,
        "epoch": ckpt["epoch"],
        "fid": {
            f"vae_vs_train_n{n_big}": fid(fake_p[:n_big], train_p[:n_big], args.batch_size, device),
            f"vae_vs_train_n{n_test}": fid(fake_p[:n_test], train_sub, args.batch_size, device),
            f"test_vs_train_n{n_test} (real baseline)": fid(test_p, train_sub, args.batch_size, device),
        },
        "is": {
            f"vae_n{n_fake}": inception_score(fake_p, args.batch_size, device),
            f"train_n{n_train} (real)": inception_score(train_p, args.batch_size, device),
            f"test_n{n_test} (real)": inception_score(test_p, args.batch_size, device),
        },
    }

    print(f"{args.ckpt} (epoch {ckpt['epoch']})")
    for name, table in [("FID (lower is better)", results["fid"]), ("Inception Score", results["is"])]:
        print(name)
        for k, v in table.items():
            print(f"  {k:<40} {v:8.3f}")

    out_dir = os.path.dirname(args.ckpt)
    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    with open(os.path.join(out_dir, f"fid_is_{stem}.json"), "w") as f:
        json.dump(results, f, indent=2)
    save_image(next(generate(model, 64, 64, device)), os.path.join(out_dir, f"generated_{stem}.png"), nrow=8)
    print(f"wrote fid_is_{stem}.json and generated_{stem}.png to {out_dir}")


if __name__ == "__main__":
    main()
