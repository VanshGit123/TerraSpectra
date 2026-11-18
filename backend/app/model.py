from __future__ import annotations
import torch
from torch import nn


class TerraSpectraHybrid(nn.Module):
    """Spatial-spectral 3D CNN followed by a spatial Transformer encoder.

    Input: [batch, 1, bands, height, width]
    Output: stress logits [batch, 1, height, width]
    """
    def __init__(self, width: int = 32, heads: int = 4, layers: int = 2):
        super().__init__()
        self.spectral_spatial = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.BatchNorm3d(16), nn.GELU(),
            nn.Conv3d(16, width, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(width), nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.spectral_spatial(x).mean(dim=2)  # spectral pooling -> [N,C,H,W]
        n, c, h, w = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        features = tokens.transpose(1, 2).reshape(n, c, h, w)
        return self.head(features)
