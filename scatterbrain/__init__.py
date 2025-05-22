# ScatterBrain: A Python library for SAXS/WAXS data analysis and modeling.

"""
ScatterBrain
============

A Python library designed to facilitate the loading, processing, analysis,
modeling, and visualization of Small-Angle X-ray Scattering (SAXS)
and Wide-Angle X-ray Scattering (WAXS) data.

Modules
-------
core
    Core data structures like ScatteringCurve1D.
io
    Functions for data input and output.
reduction (placeholder)
    Modules for 2D to 1D data reduction.
processing
    Functions for 1D data processing (e.g., background subtraction).
analysis
    Modules for SAXS/WAXS analysis methods (e.g., Guinier, Porod).
modeling
    Tools for fitting scattering models to data, including submodules for
    form_factors and structure_factors.
visualization
    Plotting utilities.
utils
    General utility functions and constants.

Documentation is available at [Link to your documentation when ready]
"""

# Version of the scatterbrain package
try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("scatterbrain")
    except PackageNotFoundError:
        # package is not installed, e.g. when running from source checkout
        __version__ = "0.0.1.dev0" # Or a more dynamic way to get it if possible
except ImportError:
    # For Python < 3.8
    __version__ = "0.0.1.dev0" # Fallback

# Expose key classes or functions at the top level if desired.
# For now, we will encourage importing from submodules.
# Example:
# from .core import ScatteringCurve1D
# from .io import load_ascii_1d

# Define __all__ to control `from scatterbrain import *`
# This is generally good practice, even if explicit imports are preferred.
__all__ = [
    "core",
    "io",
    "reduction",
    "processing",
    "analysis",
    "modeling",
    "visualization",
    "utils",
    "__version__",
    # Add specific classes/functions here if re-exported, e.g. "ScatteringCurve1D"
]

# Optional: A simple print statement during development for feedback
# print(f"ScatterBrain package (version {__version__}) initialized.")