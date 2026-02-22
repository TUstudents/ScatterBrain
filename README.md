# ScatterBrain

**A Python library for SAXS/WAXS data analysis and modeling.**

[![CI](https://github.com/TUstudents/ScatterBrain/actions/workflows/ci.yml/badge.svg)](https://github.com/TUstudents/ScatterBrain/actions/workflows/ci.yml)

> Status: Pre-Alpha (v0.1.0, under active development)

---

## Overview

ScatterBrain aims to provide a comprehensive, user-friendly, and extensible
Python toolkit for scientists and researchers working with Small-Angle X-ray
Scattering (SAXS) and Wide-Angle X-ray Scattering (WAXS) data. The library is
designed to facilitate the entire workflow from data loading and processing to
advanced analysis and model fitting.

---

## Features

### Implemented (v0.1.0)

**Data I/O**
- `load_ascii_1d` -- load 1D ASCII data files (`.dat`, `.txt`, `.csv`, any delimiter)
- `save_ascii_1d` -- write processed curves back to disk

**Processing**
- `subtract_background` -- subtract a constant or curve background with error propagation
- `normalize` -- divide intensity by a scalar factor with error propagation

**Analysis**
- `guinier_fit` -- Guinier analysis for Rg and I(0); weighted least-squares when errors are available
- `porod_analysis` -- Porod exponent and constant via log-log fit or average
- `scattering_invariant` -- Q* = integral q^2 I(q) dq with Guinier low-q and Porod high-q extrapolations

**Modeling**
- `sphere_pq` -- monodisperse sphere form factor P(q)
- `cylinder_pq` -- orientationally averaged cylinder form factor (64-point Gauss-Legendre quadrature)
- `core_shell_sphere_pq` -- spherically symmetric core-shell form factor
- `fit_model` -- fit any form factor to data with scale and background; uses lmfit internally for confidence intervals

**Visualization**
- `plot_iq` -- I(q) vs q (log-log or linear)
- `plot_guinier` -- Guinier plot (ln I vs q^2)
- `plot_porod` -- Porod plot (I*q^4 vs q or log-log)
- `plot_fit` -- data + model overlay with optional normalized residuals panel
- `plot_kratky` -- standard and dimensionless Kratky plots

### Planned (future phases)

- 2D detector image loading and azimuthal integration
- Pair distance distribution function p(r)
- Structure factors S(q) and polydispersity
- WAXS peak fitting and Scherrer crystallite size analysis
- Interactive visualization

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/TUstudents/ScatterBrain.git
cd ScatterBrain
uv sync --all-extras
```

Run commands in the managed environment:

```bash
uv run python -c "import scatterbrain; print(scatterbrain.__version__)"
```

Or activate the virtual environment directly:

```bash
source .venv/bin/activate
```

**Requirements:** Python >= 3.10, numpy, scipy, matplotlib, pandas, lmfit >= 1.2.

---

## Quick Start

```python
from scatterbrain.io import load_ascii_1d
from scatterbrain.processing import subtract_background
from scatterbrain.analysis import guinier_fit, porod_analysis
from scatterbrain.modeling.form_factors import sphere_pq
from scatterbrain.modeling.fitting import fit_model
from scatterbrain.visualization import plot_iq, plot_guinier, plot_fit
import numpy as np

# Load 1D SAXS data
curve = load_ascii_1d("data.dat", err_col=2, skip_header=2, delimiter=r"\s+")

# Background subtraction
curve_bg = subtract_background(curve, 10.0)

# Guinier analysis (weighted by errors when available)
g = guinier_fit(curve_bg, qrg_limit_max=1.3)
if g is not None:
    print(f"Rg  = {g['Rg']:.3f} +/- {g['Rg_err']:.3f} nm")
    print(f"I(0) = {g['I0']:.3e}")

# Sphere form factor fit
r_init = np.sqrt(5.0 / 3.0) * g["Rg"]
result = fit_model(
    curve_bg,
    model_func=sphere_pq,
    param_names=["radius"],
    initial_params=[curve_bg.intensity.max(), 0.0, r_init],
    param_bounds=([0, 0, 0.1], [1e6, 1e3, 100]),
)
if result is not None:
    fp = result["fitted_params"]
    print(f"radius = {fp['radius']:.3f} nm,  chi^2_red = {result['chi_squared_reduced']:.2f}")
    fig = plot_fit(curve_bg, result, plot_residuals=True)
```

---

## Tutorials

- `notebooks/01_basic_workflow.ipynb` -- data loading, background subtraction, Guinier and Porod analysis, sphere form factor fit
- `notebooks/02_form_factor_fitting.ipynb` -- cylinder and core-shell sphere fitting, lmfit confidence intervals, Kratky plot, scattering invariant

---

## Development

```bash
# Run tests
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=scatterbrain --cov-report=term-missing

# Format (black, 88-char lines)
uv run black scatterbrain tests

# Lint
uv run flake8 scatterbrain tests

# Type check
uv run mypy scatterbrain

# Build documentation
uv run sphinx-build -W -b html docs/source docs/_build/html
```

---

## License

[CC BY-NC-SA 4.0](LICENSE) -- Johannes Poms, 2026.
