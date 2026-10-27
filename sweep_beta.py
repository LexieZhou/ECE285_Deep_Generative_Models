import argparse
import csv
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")  # no display on the server
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Train one VAE per beta, compute FID/IS for each, and summarize. "
                    "Unrecognized arguments (e.g. --epochs 50) are passed through to train.py. "
                    "Finished steps are skipped, so the sweep can be rerun after an interruption.")
    p.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--existing", nargs="*", default=["1=outputs/run2"],
                   help="reuse finished runs instead of retraining, as beta=run_dir")
    p.add_argument("--out_root", default="outputs")
    p.add_argument("--data_root", default="data/chest_xray")
    return p.parse_known_args()


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def fmt_beta(b):
    return f"{b:g}"


def main():
    args, train_args = parse_args()
    existing = {float(k): v for k, v in (e.split("=", 1) for e in args.existing)}
    run_dirs = {b: existing.get(b, os.path.join(args.out_root, f"beta_{fmt_beta(b)}")) for b in args.betas}

    for b, d in run_dirs.items():
        # train.py writes test_metrics.json last, so its presence means training finished
        if os.path.exists(os.path.join(d, "test_metrics.json")):
            print(f"beta={fmt_beta(b)}: training already done in {d}, skipping")
        else:
            run([sys.executable, "train.py", "--beta", str(b), "--out_dir", d, "--data_root", args.data_root] + train_args)
        if os.path.exists(os.path.join(d, "fid_is_best.json")):
            print(f"beta={fmt_beta(b)}: FID/IS already done in {d}, skipping")
        else:
            run([sys.executable, "fid_is.py", "--ckpt", os.path.join(d, "best.pt"), "--data_root", args.data_root])

    # Only beta should differ between runs; anything else makes the comparison unfair
    configs = {b: json.load(open(os.path.join(d, "config.json"))) for b, d in run_dirs.items()}
    for b, d in run_dirs.items():
        if configs[b]["beta"] != b:
            sys.exit(f"{d} was trained with beta={configs[b]['beta']}, not {b}")
    ref_beta = args.betas[0]
    for b in args.betas[1:]:
        diff = {k: (configs[ref_beta].get(k), v) for k, v in configs[b].items()
                if k not in ("out_dir", "beta") and configs[ref_beta].get(k) != v}
        if diff:
            print(f"WARNING: beta={fmt_beta(b)} config differs from beta={fmt_beta(ref_beta)} beyond beta: {diff}")

    rows = []
    for b, d in run_dirs.items():
        test = json.load(open(os.path.join(d, "test_metrics.json")))
        fi = json.load(open(os.path.join(d, "fid_is_best.json")))
        row = {"beta": fmt_beta(b), "run_dir": d, "best_epoch": test["epoch"],
               "test_loss": test["loss"], "test_recon": test["recon"], "test_kl": test["kl"]}
        row.update({f"FID {k}": v for k, v in fi["fid"].items()})
        row.update({f"IS {k}": v for k, v in fi["is"].items()})
        rows.append(row)

    csv_path = os.path.join(args.out_root, "beta_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    keys = [k for k in rows[0] if k != "run_dir"]
    print("\n| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(f"{r[k]:.2f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |")

    # Side-by-side generated samples for the report
    fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 4.3))
    for ax, r in zip(axes if len(rows) > 1 else [axes], rows):
        ax.imshow(mpimg.imread(os.path.join(r["run_dir"], "generated_best.png")), cmap="gray")
        ax.set_title(f"β = {r['beta']}")
        ax.axis("off")
    fig.tight_layout()
    fig_path = os.path.join(args.out_root, "beta_sweep_samples.png")
    fig.savefig(fig_path, dpi=200)
    print(f"\nwrote {csv_path} and {fig_path}")


if __name__ == "__main__":
    main()
