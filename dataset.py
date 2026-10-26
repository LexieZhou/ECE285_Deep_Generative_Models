import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transform(img_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # a few X-rays are stored as RGB
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # pixel values in [0, 1]
    ])


def get_dataloader(data_root, split, img_size=128, batch_size=64, shuffle=True, num_workers=4):
    """Loads data_root/<split>/{NORMAL,PNEUMONIA}/*.jpeg. Labels are ignored by the VAE."""
    dataset = datasets.ImageFolder(os.path.join(data_root, split), transform=get_transform(img_size))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,  # drop the last incomplete batch only during training
    )
