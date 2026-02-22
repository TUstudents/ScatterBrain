# scatterbrain/processing/__init__.py
"""
1D data processing utilities for ScatterBrain.
"""

from .background import subtract_background, normalize

__all__ = ["subtract_background", "normalize"]
