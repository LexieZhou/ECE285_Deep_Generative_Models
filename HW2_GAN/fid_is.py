"""FID and Inception Score of a trained generator, with a real-vs-real baseline.

    python fid_is.py --ckpt outputs/run1/best.pt
"""
import argparse
import json
import os

import torch
from torchvision.utils import save_image

from dataset import PaintingDataset
from metrics import fid, generate, inception_outputs, inception_score, load_inception, real_batches
from model import Generator, get_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="outputs/run1/best.pt")
    p.add_argument("--data", default=None, help="default: the --data the checkpoint was trained on")
    p.add_argument("--n", type=int, default=10000, help="images per side (capped at half the dataset)")
    p.add_argument("--batch_size", type=int, default=100, help="Inception batch size")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="one run per seed; scores are reported as mean and std over seeds")
    return p.parse_args()


def mean_std(values):
    t = torch.tensor(values, dtype=torch.float64)
    return t.mean().item(), (t.std().item() if len(values) > 1 else 0.0)


def main():
    args = parse_args()
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    targs = ckpt["args"]
    G = Generator(ckpt["img_size"], targs["nz"], targs["ngf"]).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    real_set = PaintingDataset(args.data or targs["data"], flip=False)
    # Two disjoint real subsets per seed: one is the reference for the generator, the other gives the
    # real-vs-real baseline (the FID a perfect generator would get at this sample size)
    n = min(args.n, len(real_set) // 2)
    inception = load_inception(device)
    print(f"device: {device} | {args.ckpt} (epoch {ckpt['epoch']}) | {n} images per side x {len(args.seeds)} seeds")

    per_seed = []
    for seed in args.seeds:
        perm = torch.randperm(len(real_set), generator=torch.Generator().manual_seed(seed))
        ref_f = inception_outputs(inception, real_batches(real_set, perm[:n].tolist(), args.batch_size), device)
        other_f = inception_outputs(inception, real_batches(real_set, perm[n:2 * n].tolist(), args.batch_size), device)
        fake_f = inception_outputs(inception, generate(G, n, args.batch_size, device, seed=seed), device)
        scores = {
            "seed": seed,
            "fid_gan": fid(fake_f, ref_f, device=device),
            "fid_real_baseline": fid(other_f, ref_f, device=device),
            "is_gan": inception_score(fake_f, device=device),
            "is_real": inception_score(ref_f, device=device),
        }
        per_seed.append(scores)
        print(", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in scores.items()), flush=True)

    results = {"ckpt": args.ckpt, "epoch": ckpt["epoch"], "n_per_side": n, "seeds": args.seeds}
    for k in ("fid_gan", "fid_real_baseline", "is_gan", "is_real"):
        m, s = mean_std([r[k] for r in per_seed])
        results[k] = {"mean": m, "std": s}
        print(f"{k:<20} {m:8.4f} ± {s:.4f}")
    results["per_seed"] = per_seed

    out_dir = os.path.dirname(args.ckpt)
    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    with open(os.path.join(out_dir, f"fid_is_{stem}.json"), "w") as f:
        json.dump(results, f, indent=2)
    save_image(next(generate(G, 64, 64, device, seed=args.seeds[0])),
               os.path.join(out_dir, f"generated_{stem}.png"), nrow=8)
    print(f"wrote fid_is_{stem}.json and generated_{stem}.png to {out_dir}")


if __name__ == "__main__":
    main()
