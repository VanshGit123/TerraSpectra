
from __future__ import annotations

import numpy as np


def make_synthetic_cube(
    height: int = 96,
    width: int = 96,
    bands: int = 32,
    seed: int = 17,
    stress_strength: float = 1.0,
    noise_std: float = 0.012,
    randomize_patches: bool = False,
):
    """
    Generate a reproducible hyperspectral cube [H,W,B]
    and simulated crop stress mask [H,W].
    """

    if height < 8 or width < 8:
        raise ValueError("Height and width must be at least 8.")

    if bands < 4:
        raise ValueError("At least 4 spectral bands are required.")

    if not 0 <= stress_strength <= 2:
        raise ValueError("Stress strength must be between 0 and 2.")

    if noise_std < 0:
        raise ValueError("Noise standard deviation cannot be negative.")

    rng = np.random.default_rng(seed)

    wavelengths = np.linspace(
        450, 950, bands, dtype=np.float32
    )

    base = 0.18 + 0.00035 * (wavelengths - 450)

    spectra = np.tile(
        base, (height, width, 1)
    ).astype(np.float32)

    yy, xx = np.mgrid[0:height, 0:width]

    spatial_texture = (
        0.018 * np.sin(xx / 8)
        + 0.014 * np.cos(yy / 11)
    )[..., None]

    spectra += spatial_texture

    noise = rng.normal(
        0,
        noise_std,
        size=spectra.shape
    ).astype(np.float32)

    spectra += noise

    # Configure two simulated stress regions.
    if randomize_patches:
        centers = [
            (
                rng.uniform(0.25, 0.45) * width,
                rng.uniform(0.30, 0.50) * height,
            ),
            (
                rng.uniform(0.60, 0.80) * width,
                rng.uniform(0.55, 0.75) * height,
            ),
        ]
    else:
        centers = [
            (width * 0.34, height * 0.42),
            (width * 0.72, height * 0.68),
        ]

    (cx1, cy1), (cx2, cy2) = centers

    rx1, ry1 = width * 0.16, height * 0.12
    rx2, ry2 = width * 0.12, height * 0.18

    patch1 = (
        ((xx - cx1) / rx1) ** 2
        + ((yy - cy1) / ry1) ** 2
    ) < 1

    patch2 = (
        ((xx - cx2) / rx2) ** 2
        + ((yy - cy2) / ry2) ** 2
    ) < 1

    mask = (patch1 | patch2).astype(np.uint8)

    # Simulated red-edge and NIR stress response.
    red_edge = (
        (wavelengths >= 680)
        & (wavelengths <= 760)
    )

    nir = wavelengths > 760

    stress_signature = (
        0.055 * red_edge + 0.035 * nir
    ).astype(np.float32)

    spectra[mask == 1] -= (
        stress_strength * stress_signature
    )

    spectra = np.clip(spectra, 0.01, 0.99)

    return spectra.astype(np.float32), mask


def get_dataset_statistics(
    spectra: np.ndarray,
    mask: np.ndarray,
):
    """Calculate spectral and crop-stress statistics."""

    if spectra.ndim != 3:
        raise ValueError("Spectra must have shape [H, W, B].")

    if mask.shape != spectra.shape[:2]:
        raise ValueError("Mask dimensions must match the image.")

    healthy_pixels = int(np.sum(mask == 0))
    stressed_pixels = int(np.sum(mask == 1))
    total_pixels = mask.size

    return {
        "height": int(spectra.shape[0]),
        "width": int(spectra.shape[1]),
        "spectral_bands": int(spectra.shape[2]),
        "total_pixels": int(total_pixels),
        "healthy_pixels": healthy_pixels,
        "stressed_pixels": stressed_pixels,
        "stress_percentage": round(
            stressed_pixels / total_pixels * 100, 2
        ),
        "mean_reflectance": round(
            float(np.mean(spectra)), 4
        ),
        "min_reflectance": round(
            float(np.min(spectra)), 4
        ),
        "max_reflectance": round(
            float(np.max(spectra)), 4
        ),
    }

def get_dataset_statistics(
    spectra: np.ndarray,
    mask: np.ndarray,
):
    """Calculate spectral and crop-stress statistics."""

    if spectra.ndim != 3:
        raise ValueError("Spectra must have shape [H, W, B].")

    if mask.shape != spectra.shape[:2]:
        raise ValueError("Mask dimensions must match the image.")

    healthy_pixels = int(np.sum(mask == 0))
    stressed_pixels = int(np.sum(mask == 1))
    total_pixels = mask.size

    return {
        "height": int(spectra.shape[0]),
        "width": int(spectra.shape[1]),
        "spectral_bands": int(spectra.shape[2]),
        "total_pixels": int(total_pixels),
        "healthy_pixels": healthy_pixels,
        "stressed_pixels": stressed_pixels,
        "stress_percentage": round(
            stressed_pixels / total_pixels * 100, 2
        ),
        "mean_reflectance": round(
            float(np.mean(spectra)), 4
        ),
        "min_reflectance": round(
            float(np.min(spectra)), 4
        ),
        "max_reflectance": round(
            float(np.max(spectra)), 4
        ),
    }
