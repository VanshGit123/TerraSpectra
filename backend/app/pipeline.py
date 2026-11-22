
from __future__ import annotations

import numpy as np

from app.data import make_synthetic_cube
from app.preprocessing import normalize_cube
from app.pca import reduce_dimensions, get_pca_summary


def prepare_hyperspectral_data(
    height: int = 96,
    width: int = 96,
    bands: int = 32,
    n_components: int = 8,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate and preprocess a hyperspectral dataset.

    Returns:
        processed_cube: PCA-reduced hyperspectral data.
        mask: Simulated crop stress mask.
        metadata: Dataset and preprocessing statistics.
    """

    # Generate synthetic hyperspectral imagery.
    cube, mask = make_synthetic_cube(
        height=height,
        width=width,
        bands=bands,
        seed=seed,
    )

    # Normalize reflectance values.
    normalized_cube = normalize_cube(cube)

    # Reduce spectral dimensions using PCA.
    processed_cube, explained_variance = reduce_dimensions(
        normalized_cube,
        n_components=n_components,
    )

    # Collect preprocessing metadata.
    metadata = get_pca_summary(
        normalized_cube,
        processed_cube,
        explained_variance,
    )

    metadata["seed"] = seed
    metadata["stress_percentage"] = round(
        float(np.mean(mask)) * 100,
        2,
    )

    return processed_cube, mask, metadata
