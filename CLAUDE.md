# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ScatterBrain

A Python library for loading, processing, analyzing, modeling, and visualizing SAXS/WAXS (Small/Wide-Angle X-ray Scattering) data. Development follows an iterative phase plan described in `PHASE0_PLAN.md` through `PHASE5_PLAN.md`. The design spec lives in `Design_document.md`.

## Environment & Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install all dependencies (including dev extras) with:

```bash
uv sync --all-extras
```

To run commands within the managed environment, prefix them with `uv run`:

```bash
uv run python -c "import scatterbrain; print(scatterbrain.__version__)"
```

Alternatively, activate the `.venv` that uv manages:

```bash
source .venv/bin/activate
```

## Common Commands

```bash
# Run full test suite
uv run pytest tests/

# Run a single test class or method
uv run pytest tests/test_core.py::TestScatteringCurve1D::test_init_basic
uv run pytest tests/test_analysis.py -k "TestGuinierFit"

# Run tests with coverage report
uv run pytest tests/ --cov=scatterbrain --cov-report=term-missing

# Run tutorial notebook smoke test
uv run pytest --nbmake notebooks/01_basic_workflow.ipynb

# Format code (black, 88-char lines)
uv run black scatterbrain tests

# Lint (ruff; configured in pyproject.toml)
uv run ruff check scatterbrain tests

# Type check (non-blocking; ignore_missing_imports=true)
uv run mypy scatterbrain
```

## Architecture

### Central Data Object

`ScatteringCurve1D` (`scatterbrain/core.py`) is the central object passed between every module. It holds:
- `q`, `intensity`, `error` - numpy arrays
- `q_unit`, `intensity_unit` - strings (e.g. `"nm^-1"`, `"a.u."`)
- `metadata` - dict including `processing_history` (list of strings appended by each operation)

### Data Flow

```
io.load_ascii_1d()
    -> ScatteringCurve1D
        -> processing.subtract_background()   # returns NEW curve
        -> analysis.guinier_fit()             # returns GuinierResult
        -> analysis.porod_analysis()          # returns PorodResult
        -> modeling.fit_model()               # returns dict with "fit_curve" key
        -> visualization.plot_*()             # returns (fig, ax) or fig
```

All processing functions return **new** `ScatteringCurve1D` objects and never modify their inputs. Each operation appends to `metadata["processing_history"]`.

### Module Layout

| Module | Key public API |
|--------|----------------|
| `scatterbrain/core.py` | `ScatteringCurve1D` |
| `scatterbrain/io.py` | `load_ascii_1d`, `save_ascii_1d` |
| `scatterbrain/processing/background.py` | `subtract_background` |
| `scatterbrain/analysis/guinier.py` | `guinier_fit` -> `GuinierResult` with `Rg`, `I0`, fit stats |
| `scatterbrain/analysis/porod.py` | `porod_analysis` -> `PorodResult` with `porod_exponent`, `porod_constant_kp` |
| `scatterbrain/analysis/__init__.py` | re-exports `guinier_fit`, `GuinierResult`, `porod_analysis`, `PorodResult` |
| `scatterbrain/modeling/form_factors.py` | `sphere_pq(q, radius)` - normalized P(q) |
| `scatterbrain/modeling/fitting.py` | `fit_model` - wraps any form factor with `scale * model_func(q, ...) + background` |
| `scatterbrain/visualization.py` | `plot_iq`, `plot_guinier`, `plot_porod`, `plot_fit` |
| `scatterbrain/utils.py` | `convert_q_array`, `ScatterBrainError`, `ProcessingError`, `AnalysisError`, `FittingError` |

`scatterbrain/reduction/` is a placeholder for future 2D->1D reduction; it is intentionally empty.

### fit_model Convention

`fit_model` always prepends `scale` and `background` to the parameter list, so `initial_params` and `param_bounds` must be ordered as `[scale, background, *model_params]`. `param_names` lists only the *model* parameters (e.g. `["radius"]`); scale and background are handled implicitly.

### Return Types for Analysis Functions

`guinier_fit` returns `Optional[GuinierResult]` and `porod_analysis` returns `Optional[PorodResult]`. Both are `TypedDict` subclasses importable from `scatterbrain.analysis`. They provide IDE autocomplete and mypy validation on result field access.

`PorodResult` uses `total=False` because three keys (`log_kp_intercept`, `log_kp_intercept_err`, `r_value`) are only present when `fit_log_log=True`.

### Failure Mode Convention

Two distinct failure modes are used across the library:

- **Hard failure** (wrong input type, parameter count mismatch): raise a typed exception (`AnalysisError`, `FittingError`, `ProcessingError`) immediately.
- **Soft failure** (insufficient data points, no positive intensity in the selected range): emit a `logger.warning()` message and return `None`. Callers should check for `None` before using the result.

### Custom Exceptions

All custom exceptions (in `scatterbrain/utils.py`) inherit from `ScatterBrainError`:
- `ProcessingError` - raised in `processing/`
- `AnalysisError` - raised in `analysis/`
- `FittingError` - raised in `modeling/`

### Logging

All modules use `logging.getLogger(__name__)`. The root `scatterbrain` logger has a `NullHandler` by default (silent). To enable output call `scatterbrain.configure_logging()` once. Log levels: `DEBUG` for diagnostic internals, `WARNING` for soft failures.

## Code Style

- Formatter: **black** (line length 88)
- Linter: **ruff** - config in `pyproject.toml`; E203 and E501 are ignored globally; T201 is ignored per-file
- Type annotations are used but `mypy` is run with `continue-on-error` in CI (not blocking)
- All text in log messages, exception messages, and other terminal-rendered strings must use plain ASCII only (no Unicode arrows, Greek letters, superscripts, or symbols)

## Testing Conventions

- Tests live in `tests/` and mirror the module structure (`test_core.py`, `test_io.py`, etc.)
- Test data files are in `tests/test_data/`
- Fixtures shared within a file are defined at the top of that file; there are no project-wide fixtures in `conftest.py`
- Matplotlib tests use `matplotlib.use("Agg")` at the top of the file before other imports (hence the E402 per-file ignore)
- The tutorial notebook `notebooks/01_basic_workflow.ipynb` is executed in CI via `pytest --nbmake` with `MPLBACKEND=Agg`; it uses the bundled example file at `scatterbrain/examples/data/example_sphere_data.dat`
