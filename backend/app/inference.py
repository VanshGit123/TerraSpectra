from __future__ import annotations
import os
import numpy as np
import torch
from .model import TerraSpectraHybrid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "terraspectra_model.pt")
_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not os.path.exists(CHECKPOINT):
        return None
    model = TerraSpectraHybrid().to(DEVICE)
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _MODEL = model
    return _MODEL


def heuristic_probability(cube: np.ndarray) -> np.ndarray:
    """Transparent fallback using a normalized red-edge/NIR contrast heuristic."""
    bands = cube.shape[-1]
    red_idx = max(0, int(bands * 0.58))
    nir_idx = max(0, int(bands * 0.82))
    red_edge = cube[..., red_idx]
    nir = cube[..., min(nir_idx, bands - 1)]
    contrast = (red_edge - nir)
    scale = np.std(contrast) + 1e-6
    z = (np.mean(contrast) - contrast) / scale
    return (1.0 / (1.0 + np.exp(-np.clip(z, -12, 12)))).astype(np.float32)


def predict_tiled(cube: np.ndarray, tile_size: int = 32, overlap: int = 8):
    """Predict [H,W] stress probability with overlap averaging to reduce tile seams."""
    h, w, bands = cube.shape
    model = load_model()
    if bands > 32:
        # Fit PCA to the current scene and retain 32 components for manageable inference.
        from sklearn.decomposition import IncrementalPCA
        flat = cube.reshape(-1, bands)
        pca = IncrementalPCA(n_components=32, batch_size=max(256, min(4096, flat.shape[0])))
        pca.fit(flat)
        cube = pca.transform(flat).reshape(h, w, 32).astype(np.float32)
    if model is None:
        return heuristic_probability(cube), "spectral-heuristic (train model to enable CNN+Transformer)"
    stride = max(1, tile_size - overlap)
    total = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
                tile = cube[y:y2, x:x2, :]
                tensor = torch.from_numpy(tile).permute(2, 0, 1)[None, None].to(DEVICE)
                logits = model(tensor)
                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
                total[y:y2, x:x2] += probs
                count[y:y2, x:x2] += 1
    return total / np.maximum(count, 1), "hybrid-3dcnn-transformer"
