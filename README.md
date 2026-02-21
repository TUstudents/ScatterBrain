# ScatterBrain 🧠✨

**A Python library for SAXS/WAXS data analysis and modeling.**

*Dealing with scattered X-rays and the (sometimes) complex thought process of analyzing them.*

**Current Status:** Pre-Alpha (Phase 1 MVP complete)

## Overview

`ScatterBrain` provides a user-friendly, extensible Python toolkit for scientists
and researchers working with Small-Angle X-ray Scattering (SAXS) and Wide-Angle
X-ray Scattering (WAXS) data.  The library covers the full Phase 1 MVP workflow:

1. Load 1D ASCII scattering data
2. Subtract backgrounds
3. Guinier analysis (Rg, I(0))
4. Porod analysis (exponent, Porod constant)
5. Sphere form factor fitting
6. Standard diagnostic plots (I(q), Guinier, Porod, fit overlay)
7. Save processed curves

## Installation

**From source (development):**

```bash
git clone https://github.com/TUstudents/ScatterBrain.git
cd ScatterBrain
pip install -e ".[dev]"
```

**Regular install (once a release is published):**

```bash
pip install scatterbrain
```

**Requirements:** Python ≥ 3.10, NumPy, SciPy, pandas, matplotlib.

## Quick Start

```python
from scatterbrain.io import load_ascii_1d, save_ascii_1d
from scatterbrain.processing import subtract_background
from scatterbrain.analysis.guinier import guinier_fit
from scatterbrain.analysis.porod import porod_analysis
from scatterbrain.modeling.form_factors import sphere_pq
from scatterbrain.modeling.fitting import fit_model
from scatterbrain.visualization import plot_iq, plot_guinier, plot_porod, plot_fit

# Load data
curve = load_ascii_1d("my_data.dat", err_col=2)

# Background subtraction
curve_bg = subtract_background(curve, background=10.0)

# Guinier analysis
g_result = guinier_fit(curve_bg)
print(f"Rg = {g_result['Rg']:.3f} ± {g_result['Rg_err']:.3f}")

# Porod analysis
p_result = porod_analysis(curve_bg, fit_log_log=True)

# Sphere model fit
import numpy as np
fit = fit_model(curve_bg, sphere_pq, ["radius"],
                initial_params=[curve_bg.intensity.max(), 0.0, 3.0])

# Plots
fig, ax = plot_guinier(curve_bg, guinier_result=g_result)
fig, ax = plot_porod(curve_bg, porod_result=p_result)
fig     = plot_fit(curve_bg, fit)
```

See `notebooks/01_basic_workflow.ipynb` for a complete end-to-end tutorial.

## Features

### Implemented (Phase 1)

- **Data I/O:** `load_ascii_1d`, `save_ascii_1d` — robust pandas-based ASCII loader
  and tab-delimited writer with auto-generated comment headers.
- **Processing:** `subtract_background` — constant or curve-based subtraction with
  error propagation in quadrature and optional q-grid interpolation.
- **Analysis:**
  - `guinier_fit` — automatic q-range selection (qRg ≤ 1.3), linear regression in
    Guinier space, full error propagation for Rg and I(0).
  - `porod_analysis` — log-log fit and Porod-constant (average-Kp) modes.
- **Modeling:** `sphere_pq` form factor; `fit_model` generic wrapper with scale,
  background, and arbitrary model parameters.
- **Visualization:** `plot_iq`, `plot_guinier`, `plot_porod`, `plot_fit` (with
  optional normalised-residuals panel).
- **Utilities:** q-unit conversion, custom exceptions, configurable logging.

### Roadmap (Phase 2+)

- Pair distance distribution p(r) via IFT
- Additional form factors: cylinder, core-shell sphere
- Structure factors S(q)
- 2D image loading and azimuthal integration (pyFAI)
- Interactive plots (plotly/bokeh)

## Development

```bash
# Install with all dev dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=scatterbrain -q

# Code formatting & linting
black scatterbrain tests
flake8 scatterbrain tests
```

CI runs automatically on push/PR via GitHub Actions (Python 3.10, 3.11, 3.12).

## Guiding Principles

- **Modularity:** Well-defined, independently testable components.
- **Test-supported development:** Comprehensive pytest suite (≥ 80% coverage).
- **Transparency:** Methods, assumptions, and limitations are clearly documented.
- **Extensibility:** Designed for easy addition of new models and routines.

## License

This project is licensed under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE).

## Authors

Johannes Poms — Johannes.Poms@tugraz.at
