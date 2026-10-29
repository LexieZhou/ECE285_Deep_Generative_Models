import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


RESIZE_MODES = ("pad", "crop", "squash")


def pad_to_square(img):
    """Pads the shorter side of a PIL image with black, evenly on both sides, to make it square."""
    w, h = img.size
    side = max(w, h)
    left, top = (side - w) // 2, (side - h) // 2
    return transforms.functional.pad(img, [left, top, side - w - left, side - h - top], fill=0)


def get_transform(img_size, resize="pad"):
    """Grayscale img_size x img_size tensor in [0, 1]. The X-rays are not square, so resize picks how:
    pad: pad the shorter side with black to a square, then resize (keeps the whole image, undistorted)
    crop: resize the shorter side to img_size, then center crop (undistorted, but cuts the long side)
    squash: resize straight to img_size x img_size (keeps everything, but distorts the anatomy)
    check_aspect.py compares the three."""
    to_square = {
        "pad": [transforms.Lambda(pad_to_square), transforms.Resize((img_size, img_size))],
        "crop": [transforms.Resize(img_size), transforms.CenterCrop(img_size)],
        "squash": [transforms.Resize((img_size, img_size))],
    }[resize]
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # a few X-rays are stored as RGB
        *to_square,
        transforms.ToTensor(),  # pixel values in [0, 1]
    ])


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,  # drop the last incomplete batch only during training
    )


def get_dataloader(data_root, split, img_size=128, batch_size=64, shuffle=True, num_workers=4, resize="pad"):
    """Loads data_root/<split>/{NORMAL,PNEUMONIA}/*.jpeg. Labels are ignored by the VAE."""
    dataset = datasets.ImageFolder(os.path.join(data_root, split), transform=get_transform(img_size, resize))
    return make_loader(dataset, batch_size, shuffle, num_workers)


def get_train_val_loaders(data_root, img_size=128, batch_size=64, num_workers=4, val_frac=0.1, seed=0,
                          resize="pad"):
    """Splits data_root/train into train/val, stratified by class, so the test split stays held out.

    The official val split has only 16 images, so it is not used.
    """
    dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=get_transform(img_size, resize))
    targets = torch.tensor(dataset.targets)
    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = [], []
    for c in range(len(dataset.classes)):
        idx = torch.nonzero(targets == c).flatten()
        idx = idx[torch.randperm(len(idx), generator=g)].tolist()
        n_val = round(len(idx) * val_frac)
        val_idx += idx[:n_val]
        train_idx += idx[n_val:]
    train_set, val_set = Subset(dataset, train_idx), Subset(dataset, val_idx)
    return make_loader(train_set, batch_size, True, num_workers), make_loader(val_set, batch_size, False, num_workers)
