from __future__ import annotations
import numpy as np


def make_synthetic_cube(height: int = 96, width: int = 96, bands: int = 32, seed: int = 17):
    """Create a reproducible cube [H,W,B] and simulated stress mask [H,W]."""
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(450, 950, bands, dtype=np.float32)
    base = 0.18 + 0.00035 * (wavelengths - 450)
    spectra = np.tile(base, (height, width, 1)).astype(np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    spatial_texture = (0.018 * np.sin(xx / 8) + 0.014 * np.cos(yy / 11))[..., None]
    spectra += spatial_texture
    spectra += rng.normal(0, 0.012, size=spectra.shape).astype(np.float32)

    # Two irregular simulated stress patches.
    cx1, cy1, rx1, ry1 = width * 0.34, height * 0.42, width * 0.16, height * 0.12
    patch1 = ((xx-cx1)/rx1)**2 + ((yy-cy1)/ry1)**2 < 1
    cx2, cy2, rx2, ry2 = width * 0.72, height * 0.68, width * 0.12, height * 0.18
    patch2 = ((xx-cx2)/rx2)**2 + ((yy-cy2)/ry2)**2 < 1
    mask = (patch1 | patch2).astype(np.uint8)

    # Stress changes simulated red-edge/NIR reflectance, not a real disease signature.
    red_edge = (wavelengths >= 680) & (wavelengths <= 760)
    nir = wavelengths > 760
    spectra[mask == 1] -= (0.055 * red_edge + 0.035 * nir).astype(np.float32)
    spectra = np.clip(spectra, 0.01, 0.99)
    return spectra, mask
