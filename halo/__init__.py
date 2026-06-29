"""HALO: Hybrid Atmospheric-Latent Optical reconstruction for LISS-IV cloud removal.

A physics-vs-generation framework driven by a continuous opacity field and
trained self-supervised via forged clouds (no paired data required).
"""

from .cartographer import CloudCartographer
from .cloudforge import ForgeConfig, curriculum_config, forge_batch, generate_clean_scene
from .pipeline import HaloPipeline
from .reclaimer import RadiometricReclaimer
from .synthesizer import LatentTerrainSynthesizer

__all__ = [
    "CloudCartographer",
    "RadiometricReclaimer",
    "LatentTerrainSynthesizer",
    "HaloPipeline",
    "ForgeConfig",
    "forge_batch",
    "generate_clean_scene",
    "curriculum_config",
]
