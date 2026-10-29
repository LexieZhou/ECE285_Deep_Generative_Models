import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # no display on the server
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from dataset import get_dataloader, get_train_val_loaders
from model import VAE, gaussian_sigma, get_device, vae_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/chest_xray")
    p.add_argument("--out_dir", default="outputs/run2")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=100.0, help="max gradient norm; <= 0 disables clipping")
    p.add_argument("--val_frac", type=float, default=0.1, help="fraction of the train split held out for validation")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--likelihood", choices=["bernoulli", "gaussian"], default="bernoulli",
                   help="decoder likelihood: BCE, or a Gaussian whose shared sigma is fitted to each batch (sigma-VAE)")
    p.add_argument("--kl_warmup", type=int, default=0,
                   help="epochs over which the training KL weight ramps linearly from 0 to beta; 0 disables")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sample_every", type=int, default=10, help="save sample grids every N epochs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def run_epoch(model, loader, device, beta, optimizer=None, grad_clip=0.0):
    """One pass over loader. Trains if an optimizer is given, otherwise evaluates.
    Returns per-image averages of the total loss, reconstruction term and KL term (and of the decoder
    sigma for the gaussian likelihood), plus the largest pre-clipping gradient norm seen when training."""
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    if model.likelihood == "gaussian":
        totals["sigma"] = 0.0
    max_grad_norm = 0.0
    n = 0
    with torch.set_grad_enabled(training):
        for x, _ in loader:
            x = x.to(device)
            logits, mu, logvar = model(x)
            loss, recon, kl = vae_loss(logits, x, mu, logvar, beta, model.likelihood)
            if training:
                optimizer.zero_grad()
                loss.backward()
                max_norm = grad_clip if grad_clip > 0 else float("inf")
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                max_grad_norm = max(max_grad_norm, grad_norm.item())
                optimizer.step()
            b = x.size(0)
            totals["loss"] += loss.item() * b
            totals["recon"] += recon.item() * b
            totals["kl"] += kl.item() * b
            if "sigma" in totals:
                totals["sigma"] += gaussian_sigma(logits, x).item() * b
            n += b
    stats = {k: v / n for k, v in totals.items()}
    if training:
        stats["max_grad_norm"] = max_grad_norm
    return stats


def balanced_batch(dataset, n_per_class):
    """First n_per_class images of each class from an ImageFolder Subset, stacked class by class."""
    base = dataset.dataset
    picked = []
    for c in range(len(base.classes)):
        idx = [i for i, j in enumerate(dataset.indices) if base.targets[j] == c][:n_per_class]
        picked += [dataset[i][0] for i in idx]
    return torch.stack(picked)


class Tee:
    """Writes to several streams at once, so console output also lands in a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def plot_history(history, path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axes, ["loss", "recon", "kl"], ["Total loss (-ELBO)", "Reconstruction (BCE)", "KL divergence"]):
        ax.plot(epochs, [h[f"train_{key}"] for h in history], label="train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log_file = open(os.path.join(args.out_dir, "train.log"), "w", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)  # keep tracebacks in the log too
    print(f"device: {device}")

    train_loader, val_loader = get_train_val_loaders(
        args.data_root, args.img_size, args.batch_size, args.num_workers, args.val_frac, args.seed)
    # Test split is held out: only evaluated once, on the best checkpoint, after training
    test_loader = get_dataloader(args.data_root, "test", args.img_size, args.batch_size, False, args.num_workers)
    print(f"train images: {len(train_loader.dataset)}, val images: {len(val_loader.dataset)}, "
          f"test images: {len(test_loader.dataset)}")

    model = VAE(args.img_size, 1, args.latent_dim, args.base_channels, args.likelihood).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Fixed inputs so sample grids are comparable across epochs
    fixed_z = torch.randn(64, args.latent_dim, device=device)
    # 16 NORMAL + 16 PNEUMONIA, so the recon grid covers both classes
    fixed_x = balanced_batch(val_loader.dataset, 16).to(device)

    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        # Validation always uses the full beta, so best.pt is never picked from an epoch with a discounted KL
        train_beta = args.beta * min(1.0, epoch / args.kl_warmup) if args.kl_warmup > 0 else args.beta
        train_stats = run_epoch(model, train_loader, device, train_beta, optimizer, args.grad_clip)
        val_stats = run_epoch(model, val_loader, device, args.beta)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "beta": train_beta}
        scheduler.step()
        row.update({f"train_{k}": v for k, v in train_stats.items()})
        row.update({f"val_{k}": v for k, v in val_stats.items()})
        history.append(row)
        print(f"epoch {epoch:3d} | train loss {train_stats['loss']:.1f} "
              f"(recon {train_stats['recon']:.1f}, kl {train_stats['kl']:.1f}, max grad {train_stats['max_grad_norm']:.1f}) "
              f"| val loss {val_stats['loss']:.1f}"
              + (f" | beta {train_beta:.2f}" if args.kl_warmup > 0 else "")
              + (f" | sigma {train_stats['sigma']:.4f}" if "sigma" in train_stats else ""),
              flush=True)

        with open(os.path.join(args.out_dir, "history.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(history)
        plot_history(history, os.path.join(args.out_dir, "loss_curve.png"))
        ckpt = {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_loss": val_stats["loss"]}
        torch.save(ckpt, os.path.join(args.out_dir, "model.pt"))
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                samples = torch.sigmoid(model.decode(fixed_z))
                recon = torch.sigmoid(model(fixed_x)[0])
            save_image(samples, os.path.join(args.out_dir, "samples", f"sample_epoch{epoch:03d}.png"), nrow=8)
            # top rows: real images, bottom rows: reconstructions
            save_image(torch.cat([fixed_x, recon]), os.path.join(args.out_dir, "samples", f"recon_epoch{epoch:03d}.png"), nrow=8)

    best = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model.load_state_dict(best["model"])
    test_stats = run_epoch(model, test_loader, device, args.beta)
    test_stats["epoch"] = best["epoch"]
    print(f"best epoch {best['epoch']} (val loss {best['val_loss']:.1f}) | test loss {test_stats['loss']:.1f} "
          f"(recon {test_stats['recon']:.1f}, kl {test_stats['kl']:.1f})")
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_stats, f, indent=2)


if __name__ == "__main__":
    main()
