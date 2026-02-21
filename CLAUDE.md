# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ScatterBrain

A Python library for loading, processing, analyzing, modeling, and visualizing SAXS/WAXS (Small/Wide-Angle X-ray Scattering) data. Development follows an iterative phase plan described in `PHASE0_PLAN.md` and `PHASE1_PLAN.md`. The design spec lives in `Design_document.md`.

## Environment & Installation

A virtual environment is pre-configured at `.venv/`. Always activate it before running commands:

```bash
source .venv/bin/activate
```

Install the package with all dev dependencies:

```bash
pip install -e ".[dev]"
```

## Common Commands

```bash
# Run full test suite
python -m pytest tests/

# Run a single test class or method
python -m pytest tests/test_core.py::TestScatteringCurve1D::test_init_basic
python -m pytest tests/test_analysis.py -k "TestGuinierFit"

# Run tests with coverage report
python -m pytest tests/ --cov=scatterbrain --cov-report=term-missing

# Format code (black, 88-char lines)
black scatterbrain tests

# Lint (configured in .flake8; max-line-length=88)
flake8 scatterbrain tests

# Type check (non-blocking; ignore_missing_imports=true)
mypy scatterbrain
```

## Architecture

### Central Data Object

`ScatteringCurve1D` (`scatterbrain/core.py`) is the central object passed between every module. It holds:
- `q`, `intensity`, `error` — numpy arrays
- `q_unit`, `intensity_unit` — strings (e.g. `"nm^-1"`, `"a.u."`)
- `metadata` — dict including `processing_history` (list of strings appended by each operation)

### Data Flow

```
io.load_ascii_1d()
    → ScatteringCurve1D
        → processing.subtract_background()   # returns NEW curve
        → analysis.guinier_fit()             # returns dict
        → analysis.porod_analysis()          # returns dict
        → modeling.fit_model()               # returns dict with "fit_curve" key
        → visualization.plot_*()             # returns (fig, ax) or fig
```

All processing functions return **new** `ScatteringCurve1D` objects and never modify their inputs. Each operation appends to `metadata["processing_history"]`.

### Module Layout

| Module | Key public API |
|--------|----------------|
| `scatterbrain/core.py` | `ScatteringCurve1D` |
| `scatterbrain/io.py` | `load_ascii_1d`, `save_ascii_1d` |
| `scatterbrain/processing/background.py` | `subtract_background` |
| `scatterbrain/analysis/guinier.py` | `guinier_fit` → `dict` with `Rg`, `I0`, fit stats |
| `scatterbrain/analysis/porod.py` | `porod_analysis` → `dict` with `porod_exponent`, `porod_constant_kp` |
| `scatterbrain/modeling/form_factors.py` | `sphere_pq(q, radius)` — normalized P(q) |
| `scatterbrain/modeling/fitting.py` | `fit_model` — wraps any form factor with `scale * model_func(q, ...) + background` |
| `scatterbrain/visualization.py` | `plot_iq`, `plot_guinier`, `plot_porod`, `plot_fit` |
| `scatterbrain/utils.py` | `convert_q_array`, `ProcessingError`, `AnalysisError`, `FittingError` |

`scatterbrain/reduction/` is a placeholder for future 2D→1D reduction; it is intentionally empty.

### fit_model Convention

`fit_model` always prepends `scale` and `background` to the parameter list, so `initial_params` and `param_bounds` must be ordered as `[scale, background, *model_params]`. `param_names` lists only the *model* parameters (e.g. `["radius"]`); scale and background are handled implicitly.

### Custom Exceptions

All custom exceptions (in `scatterbrain/utils.py`) inherit from `ScatterBrainError`:
- `ProcessingError` — raised in `processing/`
- `AnalysisError` — raised in `analysis/`
- `FittingError` — raised in `modeling/`

### Logging

All modules use `logging.getLogger(__name__)`. The root `scatterbrain` logger has a `NullHandler` by default (silent). To enable output call `scatterbrain.configure_logging()` once.

## Code Style

- Formatter: **black** (line length 88)
- Linter: **flake8** — config in `.flake8`; E501/E203/W503 are globally suppressed; T201 is suppressed per-file for `__main__` blocks
- Type annotations are used but `mypy` is run with `continue-on-error` in CI (not blocking)

## Testing Conventions

- Tests live in `tests/` and mirror the module structure (`test_core.py`, `test_io.py`, etc.)
- Test data files are in `tests/test_data/`
- Fixtures shared within a file are defined at the top of that file; there are no project-wide fixtures in `conftest.py`
- Matplotlib tests use `matplotlib.use("Agg")` at the top of the file before other imports (hence the E402 per-file ignore)
- The tutorial notebook is `notebooks/01_basic_workflow.ipynb` — it uses the bundled example file at `scatterbrain/examples/data/example_sphere_data.dat`
