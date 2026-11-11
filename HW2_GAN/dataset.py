import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PaintingDataset(Dataset):
    """Paintings from the .npy written by prepare_data.py, as (3, H, W) float tensors in [-1, 1].

    [-1, 1] matches the tanh output of the generator. Labels (styles) are not used: the GAN is unconditional.
    """

    def __init__(self, npy_path, flip=True):
        # mmap: the array stays on disk and pages are read on demand, so DataLoader workers share it
        self.images = np.load(npy_path, mmap_mode="r")  # (N, H, W, 3) uint8
        self.flip = flip

    def __len__(self):
        return len(self.images)

    @property
    def img_size(self):
        return self.images.shape[1]

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.images[i])).permute(2, 0, 1).float() / 127.5 - 1.0
        if self.flip and torch.rand(()) < 0.5:  # a mirrored painting is still a plausible painting
            x = x.flip(-1)
        return x


def get_loader(npy_path, batch_size=128, num_workers=4, flip=True, shuffle=True):
    dataset = PaintingDataset(npy_path, flip)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,  # every training batch has the same size (BatchNorm statistics)
        persistent_workers=num_workers > 0,
    )


def to_unit(x):
    """[-1, 1] (model range) -> [0, 1] (what save_image and the FID/IS code expect)."""
    return ((x + 1) / 2).clamp(0, 1)
