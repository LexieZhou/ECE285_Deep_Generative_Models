import argparse
import csv
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # no display on the server
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image

from dataset import get_loader, PaintingDataset, to_unit
from model import Discriminator, Generator, d_loss, g_loss, get_device, weights_init


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/wikiart_64.npy", help="written by prepare_data.py; sets the image size")
    p.add_argument("--out_dir", default="outputs/run1")
    p.add_argument("--nz", type=int, default=100, help="latent dimension")
    p.add_argument("--ngf", type=int, default=64, help="generator width")
    p.add_argument("--ndf", type=int, default=64, help="discriminator width")
    p.add_argument("--d_norm", choices=["batch", "spectral"], default="batch",
                   help="BatchNorm (original DCGAN) or spectral norm in the discriminator")
    p.add_argument("--loss", choices=["bce", "hinge"], default="bce",
                   help="bce: standard GAN with the non-saturating G loss; hinge: SNGAN/BigGAN hinge loss")
    p.add_argument("--real_label", type=float, default=1.0, help="< 1 for one-sided label smoothing (bce only)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5, help="Adam beta1; DCGAN uses 0.5 instead of 0.9")
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--no_flip", action="store_true", help="disable random horizontal flips")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=50, help="average losses over this many iterations per row")
    p.add_argument("--sample_every", type=int, default=1, help="save a sample grid every N epochs")
    p.add_argument("--fid_every", type=int, default=5, help="quick FID every N epochs to pick best.pt; 0 disables")
    p.add_argument("--fid_n", type=int, default=2000, help="images per side for the quick FID")
    p.add_argument("--ckpt_every", type=int, default=10, help="keep a copy of G every N epochs; 0 disables")
    p.add_argument("--resume", action="store_true", help="continue from out_dir/last.pt")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


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


def write_csv(rows, path):
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def smooth(y, k):
    """Moving average over k points (shorter at the start), for readable GAN curves."""
    y = np.asarray(y, dtype=float)
    c = np.cumsum(np.insert(y, 0, 0.0))
    idx = np.arange(1, len(y) + 1)
    lo = np.maximum(idx - k, 0)
    return (c[idx] - c[lo]) / (idx - lo)


def plot_history(history, fid_history, path, iters_per_epoch):
    epochs = np.array([h["iter"] for h in history]) / iters_per_epoch
    k = max(1, len(history) // 50)
    n_panels = 3 if fid_history else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    for key, label in [("loss_d", "D loss"), ("loss_g", "G loss")]:
        y = [h[key] for h in history]
        line, = axes[0].plot(epochs, y, alpha=0.25)
        axes[0].plot(epochs, smooth(y, k), color=line.get_color(), label=label)
    axes[0].set_title("Losses")
    for key, label in [("d_real", "D(x)"), ("d_fake", "D(G(z))")]:
        y = [h[key] for h in history]
        line, = axes[1].plot(epochs, y, alpha=0.25)
        axes[1].plot(epochs, smooth(y, k), color=line.get_color(), label=label)
    axes[1].set_title("Discriminator output, sigmoid(logit)")
    axes[1].set_ylim(0, 1)
    if fid_history:
        axes[2].plot([h["epoch"] for h in fid_history], [h["fid"] for h in fid_history], marker="o")
        axes[2].set_title("Quick FID (lower is better)")
    for ax in axes:
        ax.set_xlabel("epoch")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)
    last_path = os.path.join(args.out_dir, "last.pt")
    if not args.resume:
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    log_file = open(os.path.join(args.out_dir, "train.log"), "a" if args.resume else "w", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)  # keep tracebacks in the log too

    loader = get_loader(args.data, args.batch_size, args.num_workers, flip=not args.no_flip)
    img_size = loader.dataset.img_size
    print(f"device: {device} | {len(loader.dataset)} images of {img_size}x{img_size} | "
          f"{len(loader)} iterations per epoch")

    G = Generator(img_size, args.nz, args.ngf).to(device)
    D = Discriminator(img_size, args.ndf, args.d_norm).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    print(f"G: {sum(p.numel() for p in G.parameters()) / 1e6:.2f}M parameters, "
          f"D: {sum(p.numel() for p in D.parameters()) / 1e6:.2f}M parameters")

    # Same z for every sample grid, so the grids show how the generator evolves
    fixed_z = torch.randn(64, args.nz, generator=torch.Generator().manual_seed(1234)).to(device)
    history, fid_history = [], []
    start_epoch, it, best_fid = 1, 0, float("inf")
    if args.resume:
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        optG.load_state_dict(ckpt["optG"])
        optD.load_state_dict(ckpt["optD"])
        history, fid_history = ckpt["history"], ckpt["fid_history"]
        start_epoch, it, best_fid = ckpt["epoch"] + 1, ckpt["iter"], ckpt["best_fid"]
        torch.set_rng_state(ckpt["rng"].cpu())  # map_location moved it to the device
        print(f"resumed from epoch {ckpt['epoch']}")

    # Quick FID on a fixed real subset, to choose best.pt (a GAN has no validation loss)
    if args.fid_every > 0:
        from metrics import fid, generate, inception_outputs, inception_score, load_inception, real_batches
        inception = load_inception(device)
        real_set = PaintingDataset(args.data, flip=False)
        idx = torch.randperm(len(real_set), generator=torch.Generator().manual_seed(0))[:args.fid_n].tolist()
        real_feats = inception_outputs(inception, real_batches(real_set, idx, 100), device)

    acc = {k: 0.0 for k in ("loss_d", "loss_g", "d_real", "d_fake")}
    n_acc = 0
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        G.train()
        D.train()
        for x in loader:
            x = x.to(device, non_blocking=True)
            z = torch.randn(x.size(0), args.nz, device=device)
            fake = G(z)

            # (1) Discriminator step: real -> high, fake -> low. detach() keeps this step out of G.
            real_logits = D(x)
            fake_logits = D(fake.detach())
            loss_d = d_loss(real_logits, fake_logits, args.loss, args.real_label)
            optD.zero_grad(set_to_none=True)
            loss_d.backward()
            optD.step()

            # (2) Generator step: make the just-updated D call the same fakes real.
            # Gradients flow through D into G, but only G's optimizer steps.
            fake_logits_g = D(fake)
            loss_g = g_loss(fake_logits_g, args.loss)
            optG.zero_grad(set_to_none=True)
            loss_g.backward()
            optG.step()

            it += 1
            acc["loss_d"] += loss_d.item()
            acc["loss_g"] += loss_g.item()
            acc["d_real"] += torch.sigmoid(real_logits).mean().item()
            acc["d_fake"] += torch.sigmoid(fake_logits).mean().item()
            n_acc += 1
            if it % args.log_every == 0:
                history.append({"iter": it, "epoch": epoch, **{k: v / n_acc for k, v in acc.items()}})
                acc = {k: 0.0 for k in acc}
                n_acc = 0
            if not torch.isfinite(loss_d) or not torch.isfinite(loss_g):
                raise RuntimeError(f"non-finite loss at iteration {it}: D {loss_d.item()}, G {loss_g.item()}")

        recent = history[-max(1, len(loader) // args.log_every):]
        msg = (f"epoch {epoch:3d} | {time.time() - t0:5.1f}s | "
               + " | ".join(f"{k} {np.mean([h[k] for h in recent]):.3f}" for k in acc))

        if args.fid_every > 0 and (epoch % args.fid_every == 0 or epoch == args.epochs):
            fake_feats = inception_outputs(inception, generate(G, args.fid_n, 100, device, seed=0), device)
            score = fid(fake_feats, real_feats, device=device)
            is_score = inception_score(fake_feats, device=device)
            fid_history.append({"epoch": epoch, "iter": it, "fid": score, "is": is_score})
            write_csv(fid_history, os.path.join(args.out_dir, "fid_history.csv"))
            msg += f" | quick FID {score:.4f}, IS {is_score:.3f}"
            if score < best_fid:
                best_fid = score
                torch.save({"G": G.state_dict(), "args": vars(args), "img_size": img_size, "epoch": epoch,
                            "quick_fid": score}, os.path.join(args.out_dir, "best.pt"))
                msg += " (best)"
        print(msg, flush=True)

        write_csv(history, os.path.join(args.out_dir, "history.csv"))
        plot_history(history, fid_history, os.path.join(args.out_dir, "loss_curve.png"), len(loader))
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            G.eval()
            with torch.no_grad():
                samples = to_unit(G(fixed_z))
            save_image(samples, os.path.join(args.out_dir, "samples", f"epoch{epoch:03d}.png"), nrow=8)
        g_ckpt = {"G": G.state_dict(), "args": vars(args), "img_size": img_size, "epoch": epoch}
        if args.ckpt_every > 0 and epoch % args.ckpt_every == 0:
            torch.save(g_ckpt, os.path.join(args.out_dir, f"G_epoch{epoch:03d}.pt"))
        # Everything needed to resume (written last, so a crash mid-epoch resumes from the previous epoch)
        torch.save({**g_ckpt, "D": D.state_dict(), "optG": optG.state_dict(), "optD": optD.state_dict(),
                    "history": history, "fid_history": fid_history, "iter": it, "best_fid": best_fid,
                    "rng": torch.get_rng_state()}, last_path)

    if args.fid_every > 0:
        print(f"best quick FID {best_fid:.4f} -> {os.path.join(args.out_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
