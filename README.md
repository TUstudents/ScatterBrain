# ScatterBrain 🧠✨

**A Python library for SAXS/WAXS data analysis and modeling.**

*Dealing with scattered X-rays and the (sometimes) complex thought process of analyzing them.*

**Current Status:** Pre-Alpha (Under Active Development)

## Overview

`ScatterBrain` aims to provide a comprehensive, user-friendly, and extensible Python toolkit for scientists and researchers working with Small-Angle X-ray Scattering (SAXS) and Wide-Angle X-ray Scattering (WAXS) data. The library is designed to facilitate the entire workflow from data loading and processing to advanced analysis and model fitting.

## Guiding Principles

*   **Modularity:** Well-defined, testable components.
*   **Iterative Development:** Core functionality first, then incremental enhancements.
*   **Test-Supported Development:** Rigorous testing for reliability.
*   **Clear Documentation:** Comprehensive guides for users and developers.
*   **User-Centric Design:** Focus on intuitive and efficient workflows.
*   **Transparency:** Clear articulation of methods, assumptions, and limitations.
*   **Extensibility:** Designed for easy addition of new models and analysis routines.

## Planned Features (Iterative Implementation)

*   **Data I/O:**
    *   Loading 1D SAXS/WAXS data (e.g., `.dat`, `.txt`, `.csv`).
    *   (Future) 2D detector image loading (TIFF, EDF, CBF, HDF5).
*   **Data Reduction (Future):**
    *   Azimuthal integration, detector corrections, masking.
*   **Data Processing:**
    *   Background subtraction, normalization.
    *   (Future) Merging, smoothing, desmearing.
*   **SAXS Analysis:**
    *   Guinier analysis ($R_g$, $I(0)$).
    *   Porod analysis (Porod constant, exponent, specific surface area).
    *   (Future) Pair Distance Distribution Function $p(r)$, Kratky plots.
*   **WAXS Analysis (Future):**
    *   Peak fitting, $d$-spacing calculation (Bragg's Law).
    *   Crystallite size estimation (Scherrer equation), degree of crystallinity.
*   **Modeling:**
    *   Library of analytical form factors $P(q)$ (sphere, cylinder, etc.).
    *   (Future) Structure factors $S(q)$, polydispersity, global fitting.
*   **Visualization:**
    *   Standard scattering plots ($I(q)$ vs $q$, Guinier, Porod) using Matplotlib.
    *   (Future) Interactive plots, 2D image display.

## Installation

### From source (development)

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/TUstudents/ScatterBrain.git
cd ScatterBrain
uv sync --all-extras
```

Run commands inside the managed environment with `uv run`:

```bash
uv run python -c "import scatterbrain; print(scatterbrain.__version__)"
```

Or activate the virtual environment directly:

```bash
source .venv/bin/activate
```

### Stable release (when published to PyPI)

```bash
pip install scatterbrain
# or, if using uv:
uv add scatterbrain
```

**Requirements:** Python >= 3.10, numpy, scipy, matplotlib, pandas.

## Quick Start

```python
from scatterbrain.io import load_ascii_1d
from scatterbrain.processing import subtract_background
from scatterbrain.analysis.guinier import guinier_fit
from scatterbrain.visualization import plot_iq, plot_guinier

# Load a 1D SAXS data file
curve = load_ascii_1d("data.dat", err_col=2, skip_header=2, delimiter=r"\s+")

# Subtract a constant background
curve_bg = subtract_background(curve, 10.0)

# Guinier analysis for Rg and I(0)
result = guinier_fit(curve_bg, qrg_limit_max=1.3)
print(f"Rg = {result['Rg']:.3f} nm,  I(0) = {result['I0']:.3e}")

# Plot
fig, ax = plot_iq(curve_bg)
fig2, ax2 = plot_guinier(curve_bg, guinier_result=result)
```

See `notebooks/01_basic_workflow.ipynb` for a complete end-to-end tutorial.