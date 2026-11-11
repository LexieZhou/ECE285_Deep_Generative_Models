"""Resize every WikiArt painting once to size x size and store them all in one uint8 .npy file.

The originals are large JPEGs (often 1000-2000 px), so decoding them on every epoch would make the CPU,
not the GPU, the bottleneck. After this script, training reads a ~1 GB array (81k x 64 x 64 x 3).

    python prepare_data.py --src data/wikiart.zip --size 64     # read straight from the zip
    python prepare_data.py --src data/wikiart --size 64         # or from the unzipped folder
    python prepare_data.py --src data/wikiart.zip --size 64 --styles Impressionism   # one style only

Reading from the zip means it never has to be extracted, which halves the disk space needed (~30 GB
instead of ~60 GB). Images are found recursively, so it does not matter whether the style folders sit at
the top level or one level deeper. The style of an image is the name of the folder that contains it.
"""
import argparse
import io
import json
import os
import zipfile
from collections import Counter
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None  # some paintings are huge scans; we only shrink them
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/wikiart.zip", help="the Kaggle zip, or the folder it was unzipped to")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--styles", nargs="+", default=None, help="keep only these style folders (default: all)")
    p.add_argument("--out", default=None, help="default: data/wikiart_<size>[_<styles>].npy")
    p.add_argument("--workers", type=int, default=default_workers(),
                   help="processes; each holds one decoded image at a time, so more workers need more RAM")
    return p.parse_args()


def default_workers():
    """CPUs this process may actually use, at most 16. os.cpu_count() reports every core of the machine,
    which inside a container can be far more than were allocated (and each worker costs RAM)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return max(1, min(16, n or 1))


def is_image(name):
    base = os.path.basename(name)
    return os.path.splitext(base)[1].lower() in EXTS and not base.startswith("._") and "__MACOSX" not in name


def find_images(src, styles=None):
    """Sorted (path, style) pairs for every image under src (a folder, or the members of a zip)."""
    if src.endswith(".zip"):
        with zipfile.ZipFile(src) as zf:
            names = [n for n in zf.namelist() if is_image(n)]
        items = [(n, os.path.basename(os.path.dirname(n))) for n in names]
    else:
        items = [(os.path.join(d, f), os.path.basename(d))
                 for d, _, files in os.walk(src) for f in files if is_image(f)]
    items.sort()
    if styles:
        items = [(p, s) for p, s in items if s in set(styles)]
    return items


_zip = None  # each worker process opens the zip once (a ZipFile can't be shared across processes)


def init_worker(src):
    global _zip
    if src.endswith(".zip"):
        _zip = zipfile.ZipFile(src)


def load_square(args):
    """Center-crop to a square (the shorter side is kept whole), then resize to size x size.
    Returns a (size, size, 3) uint8 array, or None if the file can't be read."""
    path, size = args
    try:
        with Image.open(io.BytesIO(_zip.read(path)) if _zip else path) as img:
            # For JPEGs, decode at 1/2, 1/4 or 1/8 scale straight away (never below `size`): far faster
            img.draft("RGB", (size, size))
            img = img.convert("RGB")
            w, h = img.size
            s = min(w, h)
            left, top = (w - s) // 2, (h - s) // 2
            img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)
            return np.asarray(img, dtype=np.uint8)
    except Exception as e:  # truncated or corrupt files
        print(f"skipping {path}: {e}")
        return None


def main():
    args = parse_args()
    items = find_images(args.src, args.styles)
    assert items, f"no images found under {args.src}"
    if args.out is None:
        suffix = "_" + "-".join(args.styles) if args.styles else ""
        args.out = f"data/wikiart_{args.size}{suffix}.npy"
    print(f"{len(items)} images in {len(set(s for _, s in items))} styles, {args.workers} workers -> {args.out}")

    arr = np.empty((len(items), args.size, args.size, 3), dtype=np.uint8)
    kept_styles, skipped, k = [], [], 0
    with Pool(args.workers, initializer=init_worker, initargs=(args.src,)) as pool:
        jobs = pool.imap(load_square, [(p, args.size) for p, _ in items], chunksize=64)  # keeps the order
        for (path, style), img in zip(items, tqdm(jobs, total=len(items))):
            if img is None:
                skipped.append(path)
                continue
            arr[k] = img
            kept_styles.append(style)
            k += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, arr[:k])
    meta = {
        "src": args.src, "size": args.size, "n_images": k, "n_skipped": len(skipped),
        "styles": dict(sorted(Counter(kept_styles).items(), key=lambda kv: -kv[1])), "skipped": skipped,
    }
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {k} images ({arr[:k].nbytes / 1e9:.2f} GB), skipped {len(skipped)}")


if __name__ == "__main__":
    main()
