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

import logging

# Version of the scatterbrain package
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("scatterbrain")
except PackageNotFoundError:
    # Package is not installed (e.g. running from a source checkout).
    __version__ = "0.0.1.dev0"

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
    "configure_logging",
]


def configure_logging(
    level: int = logging.DEBUG,
    handler: logging.Handler = None,
) -> None:
    """Enable scatterbrain log output. Call once at application startup.

    By default the scatterbrain logger has a NullHandler attached, which
    silences all output. Call this function to route log records to a
    stream (or any other handler).

    Parameters
    ----------
    level : int, optional
        Logging level to set on the scatterbrain logger. Default is
        ``logging.DEBUG`` (all messages).
    handler : logging.Handler, optional
        A pre-configured handler to attach. If None (default), a
        ``StreamHandler`` with a timestamped formatter is created and used.
    """
    logger = logging.getLogger("scatterbrain")
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
    logger.addHandler(handler)
    logger.setLevel(level)
