"""Use the spectral front-end from the sibling HSI-Adapter project with SAM.

The HSI-Adapter source is GPL-3.0; this module loads it at runtime instead of
copying its implementation into BalSAM.
"""
import importlib.util
from pathlib import Path

import torch
from torch import nn

from mmdet.registry import MODELS


def _load_spectral_transformer():
    source = (Path(__file__).resolve().parents[4] / 'HSI-Adapter' / 'mmseg'
              / 'models' / 'backbones' / 'adapter_hsi_modules_dino.py')
    if not source.is_file():
        raise FileNotFoundError(f'HSI-Adapter spectral module not found: {source}')
    spec = importlib.util.spec_from_file_location('_balsam_hsi_spectral', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SpectralTransformerWithSkipConnection


@MODELS.register_module()
class HSISpectralAdapter(nn.Module):
    """Map all HSI bands to the three channels expected by SAM's image encoder."""

    def __init__(self, input_channels=164, embed_dim=32, num_heads=2,
                 num_layers=1, mul=2, out_channels=3, pixel_chunk_size=16384):
        super().__init__()
        if pixel_chunk_size < 1:
            raise ValueError('pixel_chunk_size must be positive')
        spectral_cls = _load_spectral_transformer()
        self.spectral_transformer = spectral_cls(
            input_channels=input_channels, embed_dim=embed_dim,
            num_heads=num_heads, num_layers=num_layers, mul=mul,
            out_channels=out_channels)
        self.input_channels = input_channels
        self.pixel_chunk_size = pixel_chunk_size

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(
                f'Expected HSI tensor (B, {self.input_channels}, H, W), '
                f'got {tuple(x.shape)}')
        batch, _, height, width = x.shape
        pixels = x.flatten(2)
        outputs = []
        for chunk in pixels.split(self.pixel_chunk_size, dim=2):
            outputs.append(self.spectral_transformer(chunk.unsqueeze(2)))
        return torch.cat(outputs, dim=-1).reshape(
            batch, -1, height, width)
