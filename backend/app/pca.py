
from __future__ import annotations

import numpy as np


def reduce_dimensions(
    cube: np.ndarray,
    n_components: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce hyperspectral cube dimensions using PCA.

    Input:  [H, W, B]
    Output: [H, W, C] and explained variance ratios.
    """

    if cube.ndim != 3:
        raise ValueError("Expected cube with shape [H, W, B].")

    if not np.all(np.isfinite(cube)):
        raise ValueError("Cube contains NaN or infinite values.")

    height, width, bands = cube.shape

    if n_components < 1 or n_components > bands:
        raise ValueError(
            f"n_components must be between 1 and {bands}."
        )

    # Flatten spatial dimensions into individual pixels.
    pixels = cube.reshape(-1, bands).astype(np.float64)

    # Center each spectral band.
    mean = np.mean(pixels, axis=0)
    centered = pixels - mean

    # Compute PCA using singular value decomposition.
    _, singular_values, components = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    # Calculate explained variance.
    variance = singular_values ** 2
    total_variance = np.sum(variance)

    if total_variance == 0:
        raise ValueError("Cannot apply PCA to a constant cube.")

    explained_variance = variance / total_variance

    # Project pixels onto the selected principal components.
    selected_components = components[:n_components]

    reduced = centered @ selected_components.T

    # Restore the original spatial dimensions.
    reduced_cube = reduced.reshape(
        height,
        width,
        n_components,
    ).astype(np.float32)

    return (
        reduced_cube,
        explained_variance[:n_components].astype(np.float32),
    )


def get_pca_summary(
    original: np.ndarray,
    reduced: np.ndarray,
    explained_variance: np.ndarray,
) -> dict:
    """Generate a summary of PCA dimensionality reduction."""

    if original.ndim != 3 or reduced.ndim != 3:
        raise ValueError("Both cubes must have three dimensions.")

    if original.shape[:2] != reduced.shape[:2]:
        raise ValueError("Spatial dimensions must match.")

    return {
        "original_bands": int(original.shape[2]),
        "reduced_bands": int(reduced.shape[2]),
        "original_shape": list(original.shape),
        "reduced_shape": list(reduced.shape),
        "variance_retained": round(
            float(np.sum(explained_variance)) * 100,
            2,
        ),
        "dimensionality_reduction_percentage": round(
            (1 - reduced.shape[2] / original.shape[2]) * 100,
            2,
        ),
    }
