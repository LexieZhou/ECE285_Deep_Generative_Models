"""Train a VAE, then score its best checkpoint with FID/IS, back to back.

All arguments are passed through to train.py, e.g.
    python run_all.py --out_dir outputs/run3 --epochs 50
    python run_all.py --out_dir outputs/run5 --likelihood gaussian --kl_warmup 10
fid_is.py reads the likelihood back from the checkpoint, so it needs no extra flags.
"""
import subprocess
import sys

from train import parse_args


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    # Parse with train.py's own parser so out_dir/data_root resolve exactly as train.py sees them
    args = parse_args()
    # Separate processes: train.py redirects stdout into its train.log
    run([sys.executable, "train.py"] + sys.argv[1:])
    run([sys.executable, "fid_is.py", "--ckpt", f"{args.out_dir}/best.pt", "--data_root", args.data_root,
         "--batch_size", str(args.batch_size), "--num_workers", str(args.num_workers)])


if __name__ == "__main__":
    main()
