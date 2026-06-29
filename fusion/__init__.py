"""SAR-Optical multi-modal fusion for LISS-IV cloud removal.

Components:
  - DCMFUNet: dual-encoder cross-modal fusion generator
  - PatchDiscriminator: conditional 70x70 PatchGAN
  - dataset / preprocess: paired + synthetic data and tiling utilities
  - train / infer / evaluate: end-to-end workflow
"""

from .discriminator import PatchDiscriminator
from .generator import DCMFUNet

__all__ = ["DCMFUNet", "PatchDiscriminator"]
