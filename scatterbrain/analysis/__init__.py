# scatterbrain/analysis/__init__.py
"""
The analysis subpackage for ScatterBrain.

This package provides modules for various SAXS/WAXS data analysis techniques.
"""

from .guinier import guinier_fit, GuinierResult
from .porod import porod_analysis, PorodResult

__all__ = [
    "guinier_fit",
    "GuinierResult",
    "porod_analysis",
    "PorodResult",
]
