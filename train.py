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

from dataset import get_dataloader
from model import VAE, get_device, vae_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/chest_xray")
    p.add_argument("--out_dir", default="outputs/run1")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sample_every", type=int, default=10, help="save sample grids every N epochs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def run_epoch(model, loader, device, beta, optimizer=None):
    """One pass over loader. Trains if an optimizer is given, otherwise evaluates.
    Returns per-image averages of the total loss, reconstruction term and KL term."""
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0
    with torch.set_grad_enabled(training):
        for x, _ in loader:
            x = x.to(device)
            logits, mu, logvar = model(x)
            loss, recon, kl = vae_loss(logits, x, mu, logvar, beta)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            b = x.size(0)
            totals["loss"] += loss.item() * b
            totals["recon"] += recon.item() * b
            totals["kl"] += kl.item() * b
            n += b
    return {k: v / n for k, v in totals.items()}


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

    train_loader = get_dataloader(args.data_root, "train", args.img_size, args.batch_size, True, args.num_workers)
    # The official val split has only 16 images, so the test split serves as validation.
    val_loader = get_dataloader(args.data_root, "test", args.img_size, args.batch_size, False, args.num_workers)
    print(f"train images: {len(train_loader.dataset)}, val images: {len(val_loader.dataset)}")

    model = VAE(args.img_size, 1, args.latent_dim, args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Fixed inputs so sample grids are comparable across epochs
    fixed_z = torch.randn(64, args.latent_dim, device=device)
    fixed_x = next(iter(val_loader))[0][:32].to(device)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, device, args.beta, optimizer)
        val_stats = run_epoch(model, val_loader, device, args.beta)
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_stats.items()})
        row.update({f"val_{k}": v for k, v in val_stats.items()})
        history.append(row)
        print(f"epoch {epoch:3d} | train loss {train_stats['loss']:.1f} "
              f"(recon {train_stats['recon']:.1f}, kl {train_stats['kl']:.1f}) | val loss {val_stats['loss']:.1f}",
              flush=True)

        with open(os.path.join(args.out_dir, "history.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(history)
        plot_history(history, os.path.join(args.out_dir, "loss_curve.png"))
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                   os.path.join(args.out_dir, "model.pt"))

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                samples = torch.sigmoid(model.decode(fixed_z))
                recon = torch.sigmoid(model(fixed_x)[0])
            save_image(samples, os.path.join(args.out_dir, "samples", f"sample_epoch{epoch:03d}.png"), nrow=8)
            # top rows: real images, bottom rows: reconstructions
            save_image(torch.cat([fixed_x, recon]), os.path.join(args.out_dir, "samples", f"recon_epoch{epoch:03d}.png"), nrow=8)


if __name__ == "__main__":
    main()
