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
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="one sampling run per seed; scores are reported as mean and std over seeds")
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
def inception_outputs(feature_model, prob_model, batches, device):
    """Runs both of ignite's default Inception extractors over an iterable of image batches:
    2048-d pool features (FID's default) and the 1000-way softmax (InceptionScore's default)."""
    feats, probs = [], []
    for x in batches:
        x = to_inception_input(x.to(device))
        feats.append(feature_model(x).cpu())
        probs.append(prob_model(x).cpu())
    return torch.cat(feats), torch.cat(probs)


@torch.no_grad()
def generate(model, n, batch_size, device):
    for i in range(0, n, batch_size):
        z = torch.randn(min(batch_size, n - i), model.latent_dim, device=device)
        yield torch.sigmoid(model.decode(z))


def fid(a, b, batch_size, device):
    """ignite FID between two equal-sized sets of precomputed Inception pool features."""
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


def mean_std(values):
    """Mean and sample std (0 for a single value) of a list of floats."""
    t = torch.tensor(values, dtype=torch.float64)
    return t.mean().item(), (t.std().item() if len(values) > 1 else 0.0)


def main():
    args = parse_args()
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    targs = ckpt["args"]
    model = VAE(targs["img_size"], 1, targs["latent_dim"], targs["base_channels"],
                targs.get("likelihood", "bernoulli")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Real images go through the same 128x128 pipeline as training, so real and fake differ only in content.
    # Checkpoints from before --resize existed were trained on center crops
    resize = targs.get("resize", "crop")
    load = lambda split: get_dataloader(args.data_root, split, targs["img_size"], args.batch_size, False,
                                        args.num_workers, resize)
    train_loader, test_loader = load("train"), load("test")
    n_train, n_test = len(train_loader.dataset), len(test_loader.dataset)
    n_fake = args.n_fake or n_train

    # Same extractors as ignite's FID() / InceptionScore() defaults (and the blog):
    # FID on 2048-d pool features, IS on the 1000-way softmax
    feature_model = InceptionModel(return_features=True, device=device)
    prob_model = InceptionModel(return_features=False, device=device)
    print(f"device: {device} | train {n_train}, test {n_test}, generated {n_fake} x {len(args.seeds)} seeds")
    # Real features don't depend on the seed, so they are extracted once
    train_f, train_p = inception_outputs(feature_model, prob_model, (x for x, _ in train_loader), device)
    test_f, test_p = inception_outputs(feature_model, prob_model, (x for x, _ in test_loader), device)
    is_train = inception_score(train_p, args.batch_size, device)
    is_test = inception_score(test_p, args.batch_size, device)

    # Each seed draws a fresh set of generated images and a fresh train subset, so the spread over
    # seeds shows how much of a FID/IS difference is just sampling noise.
    # FID is biased upward for small N, so compare everything at matched sizes:
    # the full-size score, and a size-n_test score next to the real test-vs-train baseline.
    n_big = min(n_fake, n_train)
    per_seed = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        fake_f, fake_p = inception_outputs(feature_model, prob_model, generate(model, n_fake, args.batch_size, device), device)
        g = torch.Generator().manual_seed(seed)
        train_sub = train_f[torch.randperm(n_train, generator=g)[:n_test]]
        scores = {
            "seed": seed,
            "fid": {
                f"vae_vs_train_n{n_big}": fid(fake_f[:n_big], train_f[:n_big], args.batch_size, device),
                f"vae_vs_train_n{n_test}": fid(fake_f[:n_test], train_sub, args.batch_size, device),
                f"test_vs_train_n{n_test} (real baseline)": fid(test_f, train_sub, args.batch_size, device),
            },
            "is": {
                f"vae_n{n_fake}": inception_score(fake_p, args.batch_size, device),
                f"train_n{n_train} (real)": is_train,
                f"test_n{n_test} (real)": is_test,
            },
        }
        per_seed.append(scores)
        print(f"seed {seed}: " + ", ".join(f"FID {k} {v:.2f}" for k, v in scores["fid"].items())
              + f", IS vae {scores['is'][f'vae_n{n_fake}']:.3f}", flush=True)

    # "fid" and "is" hold the means (same keys as before), "fid_std" and "is_std" the std over seeds
    results = {"ckpt": args.ckpt, "epoch": ckpt["epoch"], "seeds": args.seeds}
    for metric in ("fid", "is"):
        stats = {k: mean_std([s[metric][k] for s in per_seed]) for k in per_seed[0][metric]}
        results[metric] = {k: m for k, (m, _) in stats.items()}
        results[f"{metric}_std"] = {k: s for k, (_, s) in stats.items()}
    results["per_seed"] = per_seed

    print(f"{args.ckpt} (epoch {ckpt['epoch']}), mean ± std over {len(args.seeds)} seeds")
    for name, metric in [("FID (lower is better)", "fid"), ("Inception Score", "is")]:
        print(name)
        for k, v in results[metric].items():
            print(f"  {k:<40} {v:8.3f} ± {results[f'{metric}_std'][k]:.3f}")

    out_dir = os.path.dirname(args.ckpt)
    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    with open(os.path.join(out_dir, f"fid_is_{stem}.json"), "w") as f:
        json.dump(results, f, indent=2)
    torch.manual_seed(args.seeds[0])
    save_image(next(generate(model, 64, 64, device)), os.path.join(out_dir, f"generated_{stem}.png"), nrow=8)
    print(f"wrote fid_is_{stem}.json and generated_{stem}.png to {out_dir}")


if __name__ == "__main__":
    main()
