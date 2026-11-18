from __future__ import annotations
import numpy as np


def read_hyperspectral_geotiff(path: str):
    """Read a multi-band GeoTIFF as [H,W,B], plus geospatial metadata."""
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Rasterio is required to read GeoTIFF files. Install rasterio first.") from exc
    with rasterio.open(path) as src:
        cube = np.moveaxis(src.read().astype(np.float32), 0, -1)
        metadata = {"crs": str(src.crs) if src.crs else None,
                    "transform": tuple(src.transform), "width": src.width,
                    "height": src.height, "count": src.count, "nodata": src.nodata}
    return cube, metadata


def write_probability_geotiff(path: str, probability: np.ndarray, metadata: dict):
    """Write a [H,W] probability layer using source raster metadata."""
    try:
        import rasterio
        from affine import Affine
    except ImportError as exc:
        raise RuntimeError("Rasterio and affine are required to write GeoTIFF files.") from exc
    transform_values = metadata.get("transform")
    if not transform_values:
        raise ValueError("Source transform metadata is required.")
    with rasterio.open(path, "w", driver="GTiff", height=probability.shape[0],
                       width=probability.shape[1], count=1, dtype="float32",
                       crs=metadata.get("crs"), transform=Affine(*transform_values[:6])) as dst:
        dst.write(probability.astype(np.float32), 1)


def read_hyperspectral_hdf5(path: str, dataset_key: str = "cube"):
    """Read an HDF5 hyperspectral dataset; expected layout is [H,W,B] or [B,H,W]."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read HDF5 files.") from exc
    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise KeyError(f"Dataset '{dataset_key}' not found. Available keys: {list(handle.keys())}")
        cube = np.asarray(handle[dataset_key], dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D hyperspectral cube, received shape {cube.shape}")
    # Common satellite products store bands first. Heuristic moves that axis to the end.
    if cube.shape[0] < cube.shape[-1] and cube.shape[0] < cube.shape[1]:
        cube = np.moveaxis(cube, 0, -1)
    return cube
