import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transform(img_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # a few X-rays are stored as RGB
        # Shorter side to img_size, then crop the center square, so the aspect ratio is kept
        # instead of squashing non-square X-rays (check_aspect.py shows how much this crops)
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
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


def get_dataloader(data_root, split, img_size=128, batch_size=64, shuffle=True, num_workers=4):
    """Loads data_root/<split>/{NORMAL,PNEUMONIA}/*.jpeg. Labels are ignored by the VAE."""
    dataset = datasets.ImageFolder(os.path.join(data_root, split), transform=get_transform(img_size))
    return make_loader(dataset, batch_size, shuffle, num_workers)


def get_train_val_loaders(data_root, img_size=128, batch_size=64, num_workers=4, val_frac=0.1, seed=0):
    """Splits data_root/train into train/val, stratified by class, so the test split stays held out.

    The official val split has only 16 images, so it is not used.
    """
    dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=get_transform(img_size))
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
