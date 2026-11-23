
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HyperspectralDataset(Dataset):
    """Create labeled hyperspectral patches for 3D-CNN training."""

    def __init__(
        self,
        cube: np.ndarray,
        mask: np.ndarray,
        patch_size: int = 5,
    ):
        if cube.ndim != 3:
            raise ValueError("Cube must have shape [H, W, B].")

        if mask.shape != cube.shape[:2]:
            raise ValueError("Mask dimensions must match the cube.")

        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("Patch size must be a positive odd number.")

        if not np.all(np.isfinite(cube)):
            raise ValueError("Cube contains NaN or infinite values.")

        self.patch_size = patch_size
        self.radius = patch_size // 2

        # Convert [H, W, B] to [B, H, W].
        self.cube = np.transpose(cube, (2, 0, 1)).astype(
            np.float32
        )

        self.mask = mask.astype(np.int64)

        # Pad image borders so every pixel has a full patch.
        self.padded_cube = np.pad(
            self.cube,
            (
                (0, 0),
                (self.radius, self.radius),
                (self.radius, self.radius),
            ),
            mode="reflect",
        )

        self.coordinates = np.argwhere(
            np.ones(mask.shape, dtype=bool)
        )

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, index: int):
        row, col = self.coordinates[index]

        r = row + self.radius
        c = col + self.radius

        patch = self.padded_cube[
            :,
            r - self.radius:r + self.radius + 1,
            c - self.radius:c + self.radius + 1,
        ]

        # Add channel dimension: [1, B, P, P].
        patch = torch.from_numpy(
            patch.copy()
        ).unsqueeze(0)

        label = torch.tensor(
            self.mask[row, col],
            dtype=torch.long,
        )

        return patch, label


def create_dataloader(
    cube: np.ndarray,
    mask: np.ndarray,
    batch_size: int = 32,
    patch_size: int = 5,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for hyperspectral model training."""

    if batch_size < 1:
        raise ValueError("Batch size must be positive.")

    dataset = HyperspectralDataset(
        cube=cube,
        mask=mask,
        patch_size=patch_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
