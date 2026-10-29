import argparse
import glob
import os

import torch
from PIL import Image
from torchvision.utils import save_image

from dataset import RESIZE_MODES, get_transform


def parse_args():
    p = argparse.ArgumentParser(description="Aspect ratios of the X-rays, and squash vs. center crop vs. pad on the most extreme ones.")
    p.add_argument("--data_root", default="data/chest_xray")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--n_show", type=int, default=8, help="most extreme images to show")
    p.add_argument("--out", default="outputs/aspect_check.png")
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.data_root, "*", "*", "*.jpeg")))
    sizes = []
    for path in paths:
        with Image.open(path) as im:  # only reads the header
            sizes.append(im.size)
    w, h = torch.tensor(sizes, dtype=torch.float64).T
    aspect = w / h
    # Fraction of the long side the center crop throws away
    cropped = 1 - torch.minimum(w, h) / torch.maximum(w, h)

    q = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64)
    print(f"{len(paths)} images under {args.data_root}")
    print(f"width x height: median {w.median():.0f} x {h.median():.0f}")
    print("aspect w/h  min / p10 / median / p90 / max: " + " / ".join(f"{v:.2f}" for v in aspect.quantile(q)))
    print(f"wider than tall: {(aspect > 1).double().mean():.1%}")
    print("cropped share of long side  min / p10 / median / p90 / max: "
          + " / ".join(f"{v:.1%}" for v in cropped.quantile(q)))
    for t in (0.1, 0.2, 0.3):
        print(f"  more than {t:.0%} cropped: {(cropped > t).sum().item()} images ({(cropped > t).double().mean():.1%})")

    # One row per resize mode, in RESIZE_MODES order
    extreme = cropped.argsort(descending=True)[:args.n_show].tolist()
    imgs = [Image.open(paths[i]) for i in extreme]
    grid = torch.stack([get_transform(args.img_size, mode)(im) for mode in RESIZE_MODES for im in imgs])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_image(grid, args.out, nrow=len(imgs))
    print(f"wrote {args.out}: rows are {', '.join(RESIZE_MODES)}, {len(imgs)} most cropped images")


if __name__ == "__main__":
    main()
