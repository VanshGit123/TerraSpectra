
import numpy as np


def normalize_cube(cube: np.ndarray) -> np.ndarray:
    """Normalize hyperspectral reflectance values to [0, 1]."""

    if cube.ndim != 3:
        raise ValueError("Expected cube with shape [H, W, B].")

    if not np.all(np.isfinite(cube)):
        raise ValueError("Cube contains NaN or infinite values.")

    min_value = np.min(cube)
    max_value = np.max(cube)

    if max_value == min_value:
        return np.zeros_like(cube, dtype=np.float32)

    normalized = (cube - min_value) / (max_value - min_value)

    return normalized.astype(np.float32)


def standardize_bands(cube: np.ndarray) -> np.ndarray:
    """Standardize each spectral band independently."""

    if cube.ndim != 3:
        raise ValueError("Expected cube with shape [H, W, B].")

    if not np.all(np.isfinite(cube)):
        raise ValueError("Cube contains NaN or infinite values.")

    mean = np.mean(cube, axis=(0, 1), keepdims=True)
    std = np.std(cube, axis=(0, 1), keepdims=True)

    std = np.where(std < 1e-8, 1.0, std)

    return ((cube - mean) / std).astype(np.float32)


def get_spectral_summary(cube: np.ndarray) -> dict:
    """Generate summary information for spectral data."""

    if cube.ndim != 3:
        raise ValueError("Expected cube with shape [H, W, B].")

    return {
        "shape": list(cube.shape),
        "band_count": int(cube.shape[2]),
        "mean": float(np.mean(cube)),
        "std": float(np.std(cube)),
        "min": float(np.min(cube)),
        "max": float(np.max(cube)),
    }

