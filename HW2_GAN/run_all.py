"""Train, then compute FID/IS on the best checkpoint. Takes the same arguments as train.py, e.g.
    python run_all.py --out_dir outputs/run1
    python run_all.py --out_dir outputs/run2 --d_norm spectral --loss hinge
"""
import os
import subprocess
import sys

from train import parse_args


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()  # train.py's own parser, so out_dir resolves exactly as train.py sees it
    run([sys.executable, "train.py"] + sys.argv[1:])  # separate process: train.py redirects stdout to train.log
    ckpt = "best.pt" if args.fid_every > 0 else "last.pt"
    run([sys.executable, "fid_is.py", "--ckpt", os.path.join(args.out_dir, ckpt)])


if __name__ == "__main__":
    main()
