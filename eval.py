import argparse
import csv
import json
import os

import torch
import torch.nn.functional as F

from dataset import get_dataloader
from model import VAE, get_device


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained VAE checkpoint, broken down by class.")
    p.add_argument("--ckpt", default="outputs/run2/best.pt")
    p.add_argument("--data_root", default="data/chest_xray")
    p.add_argument("--split", default="test")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--au_threshold", type=float, default=0.01,
                   help="a latent dim is active if Var_x(E_q[z_d|x]) exceeds this (Burda et al., 2016)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def gaussian_window(size=11, sigma=1.5, device=None):
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def ssim(x, y, window, data_range=1.0):
    """Per-image SSIM for (B, 1, H, W) tensors in [0, data_range], Gaussian 11x11 window, sigma 1.5.
    Matches skimage.metrics.structural_similarity(gaussian_weights=True, use_sample_covariance=False),
    averaged over the valid (unpadded) region."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x, mu_y = F.conv2d(x, window), F.conv2d(y, window)
    var_x = F.conv2d(x * x, window) - mu_x ** 2
    var_y = F.conv2d(y * y, window) - mu_y ** 2
    cov = F.conv2d(x * y, window) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return s.flatten(1).mean(1)


def psnr(x, y, data_range=1.0):
    mse = (x - y).pow(2).flatten(1).mean(1)
    return 10 * torch.log10(data_range ** 2 / mse.clamp_min(1e-10))


def summarize(rows, mus, au_threshold):
    out = {"n": len(rows)}
    for key in ["loss", "recon", "kl", "psnr", "ssim"]:
        v = torch.tensor([r[key] for r in rows], dtype=torch.float64)
        out[key] = {"mean": v.mean().item(), "std": v.std().item() if len(v) > 1 else 0.0}
    mu_var = mus.var(0)
    out["active_units"] = int((mu_var > au_threshold).sum())
    out["mu_var_sorted"] = sorted(mu_var.tolist(), reverse=True)
    return out


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    targs = ckpt["args"]
    model = VAE(targs["img_size"], 1, targs["latent_dim"], targs["base_channels"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = get_dataloader(args.data_root, args.split, targs["img_size"], args.batch_size, False, args.num_workers)
    dataset = loader.dataset
    window = gaussian_window(device=device)

    rows, mus = [], []
    for x, y in loader:
        x = x.to(device)
        mu, logvar = model.encode(x)
        # Loss terms use a sampled z, as in training, so they're comparable with history.csv / test_metrics.json
        logits = model.decode(model.reparameterize(mu, logvar))
        recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
        loss = recon + targs["beta"] * kl
        # Image-quality metrics use the posterior mean, the usual deterministic reconstruction
        x_hat = torch.sigmoid(model.decode(mu))
        p, s = psnr(x_hat, x), ssim(x_hat, x, window)
        for i in range(x.size(0)):
            path, label = dataset.samples[len(rows)]
            rows.append({"path": os.path.relpath(path, args.data_root), "class": dataset.classes[label],
                         "loss": loss[i].item(), "recon": recon[i].item(), "kl": kl[i].item(),
                         "psnr": p[i].item(), "ssim": s[i].item()})
        mus.append(mu.cpu())
    mus = torch.cat(mus)

    results = {"ckpt": args.ckpt, "epoch": ckpt["epoch"], "split": args.split, "au_threshold": args.au_threshold,
               "latent_dim": targs["latent_dim"], "all": summarize(rows, mus, args.au_threshold)}
    for c in dataset.classes:
        idx = [i for i, r in enumerate(rows) if r["class"] == c]
        results[c] = summarize([rows[i] for i in idx], mus[idx], args.au_threshold)

    print(f"{args.ckpt} (epoch {ckpt['epoch']}) on {args.split}, latent_dim {targs['latent_dim']}")
    print(f"{'group':<10} {'n':>5} {'loss':>9} {'recon':>9} {'kl':>7} {'psnr':>13} {'ssim':>15} {'active':>7}")
    for g in ["all"] + dataset.classes:
        r = results[g]
        print(f"{g:<10} {r['n']:>5} {r['loss']['mean']:>9.1f} {r['recon']['mean']:>9.1f} {r['kl']['mean']:>7.1f} "
              f"{r['psnr']['mean']:>6.2f} ± {r['psnr']['std']:<4.2f} {r['ssim']['mean']:>7.4f} ± {r['ssim']['std']:<5.3f} "
              f"{r['active_units']:>7}")

    out_dir = os.path.dirname(args.ckpt)
    stem = f"eval_{args.split}_{os.path.splitext(os.path.basename(args.ckpt))[0]}"
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, f"{stem}_per_image.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {stem}.json and {stem}_per_image.csv to {out_dir}")


if __name__ == "__main__":
    main()
