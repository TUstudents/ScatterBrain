# ScatterBrain -- Phase 1 Plan: Core Skeleton Implementation

**Version:** 2.0
**Branch:** `main`
**Reference:** `Design_document.md` sec.8.2, `PHASE0_PLAN.md`

---

## Goal

Produce a minimally functional library that can:
1. Load 1D ASCII SAXS/WAXS data
2. Perform background subtraction
3. Run Guinier and Porod analysis
4. Fit a sphere form factor model
5. Generate standard diagnostic plots (I(q), Guinier, Porod, fit overlay)
6. Be installed via `uv` and exercised through a tutorial notebook

---

## Current State Assessment

All Phase 1 tasks audited against the live codebase. The test suite runs
**183 passing, 0 failing** with **93% total coverage**.

| Task | Component | Status | Notes |
|------|-----------|--------|-------|
| 1.1 Packaging | `pyproject.toml` + `uv_build` | Done | `uv_build` discovers all sub-packages automatically; no manual stanza needed |
| 1.2 I/O save | `scatterbrain/io.py` -> `save_ascii_1d` | Done | Tab-delimited, auto header, round-trip tested |
| 1.3 Processing | `scatterbrain/processing/background.py` -> `subtract_background` | Done | Constant + curve modes, interpolation, quadrature error propagation; 98% coverage |
| 1.4 Visualization | `plot_guinier` | Done | ln(I) vs q^2, fit overlay, Rg/I0 annotation, error bars |
| 1.5 Visualization | `plot_porod` | Done | Iq^4 and logI/logq modes, fit overlay |
| 1.6 Visualization | `plot_fit` | Done | Data + model overlay, normalised residual panel, chi^2 annotation |
| 1.7 Cleanup | `visualization_renamed/` artifact | Done | Directory removed |
| 1.8 Tutorial notebook | `notebooks/01_basic_workflow.ipynb` | Done | End-to-end MVP workflow |
| 1.9 GitHub Actions CI | `.github/workflows/ci.yml` | Done | `uv`-based; Python 3.10/3.11/3.12; black, flake8, mypy, pytest + coverage, sphinx |
| 1.10 Tests | `tests/` -- 183 passing | Done | All new components covered |
| -- | Core data structure | Done | Exceeds spec: `__getitem__`, `to_dict/from_dict`, `convert_q_unit` |
| -- | I/O load | Done | Robust pandas-based implementation; 96% coverage |
| -- | Guinier analysis | Done | Auto q-range, error propagation; 86% coverage |
| -- | Porod analysis | Done | Log-log fit + average-Kp mode; 95% coverage |
| -- | Sphere form factor | Done | Correct P(0)=1 limit; 100% coverage |
| -- | `fit_model` | Done | scale+bg wrapper, fixed params, chi^2; 92% coverage |
| -- | `plot_iq` | Done | Multi-curve, error bars, auto axis labels; 97% visualization coverage |
| -- | Utilities | Done | `convert_q_array`, exception hierarchy, NullHandler logging; 81% coverage |
| -- | Logging system | Done | Silent by default, `configure_logging()`, per-module propagation; tested in `test_logging.py` |

**Outstanding items (new tasks below):**

| Task | Component | Status | Notes |
|------|-----------|--------|-------|
| 1.11 | `plot_fit` tight_layout warning | Done | `constrained_layout=True` at figure creation; `tight_layout` guarded by `fig.get_constrained_layout()` |
| 1.12 | Typed analysis return types | Done | `GuinierResult` and `PorodResult` TypedDicts; exported from `scatterbrain.analysis` |
| 1.13 | Notebook smoke test in CI | Done | `nbmake>=1.4` in dev deps; `uv run pytest --nbmake` step added to CI with `MPLBACKEND=Agg` |
| 1.14 | Dead code in `__init__.py` | Done | Outer `except ImportError` removed; `__init__.py` coverage raised from 65% to ~85% |

---

## Completed Tasks (for record)

### 1.1 -- Packaging Done:
`uv_build` is declared as the build backend in `pyproject.toml [build-system]`.
It discovers all sub-packages (`scatterbrain.*`) automatically from the source
tree -- no `[tool.setuptools.packages.find]` stanza is needed. The original
setuptools gap no longer applies.

---

### 1.2 -- `save_ascii_1d` Done:
Implemented in `scatterbrain/io.py:291`. Writes tab-delimited output with
auto-generated comment header (`# q [unit]`, `# intensity [unit]`,
`# Saved by ScatterBrain vX.Y.Z`). Round-trip test in `tests/test_io.py`
verifies q, intensity, and error are recovered within floating-point tolerance.

---

### 1.3 -- `subtract_background` Done:
Implemented in `scatterbrain/processing/background.py`. Supports constant float
and `ScatteringCurve1D` backgrounds. Interpolation via `numpy.interp` when
q-grids differ. Quadrature error propagation. Appends to
`metadata["processing_history"]`. Returns a new curve; inputs unchanged.
Full test suite in `tests/test_processing.py`. Coverage: 98%.

---

### 1.4 -- `plot_guinier` Done:
Implemented in `scatterbrain/visualization.py:208`. Plots ln(I) vs q^2,
propagated error bars (sigma_lnI = sigma_I / I), optional fit overlay and Rg/I0
annotation, optional q-range shading.

---

### 1.5 -- `plot_porod` Done:
Implemented in `scatterbrain/visualization.py:341`. Two modes: `"Iq4_vs_q"`
(linear axes, Porod plateau) and `"logI_vs_logq"` (slope ~ -n). Optional
fit overlay with exponent and Kp annotation.

---

### 1.6 -- `plot_fit` Done:
Implemented in `scatterbrain/visualization.py:505`. Upper panel: data +
model overlay. Lower panel: normalised residuals (I_data - I_model) / sigma
with zero line (when errors available and `plot_residuals=True`). Annotates
with reduced chi^2 and fitted parameters.

---

### 1.7 -- `visualization_renamed/` cleanup Done:
Empty stale directory has been removed from the repository.

---

### 1.8 -- Tutorial Notebook Done:
`notebooks/01_basic_workflow.ipynb` exists and demonstrates the full MVP
workflow: load -> plot -> background subtract -> Guinier -> Porod -> sphere fit ->
save.

---

### 1.9 -- GitHub Actions CI Done:
`.github/workflows/ci.yml` uses `uv` (not `pip`) throughout:
- `astral-sh/setup-uv@v5` for environment setup
- `uv sync --all-extras` for dependency installation
- `uv run` prefix for all tool invocations
- Matrix: Python 3.10, 3.11, 3.12; `fail-fast: false`
- Steps: black --check -> flake8 -> mypy (continue-on-error) -> pytest + coverage -> codecov upload
- Separate `docs` job: sphinx-build with `-W` (warnings as errors)

**Note:** The original plan spec showed `pip install -e ".[dev]"` steps -- the
actual implementation uses `uv` throughout, which is correct per the tech
stack decision.

---

### 1.10 -- Tests Done:
183 tests passing, 0 failing. All new components have dedicated test files.
Total coverage: **93%**. Per-module coverage:

| Module | Coverage |
|--------|----------|
| `visualization.py` | 97% |
| `processing/background.py` | 98% |
| `io.py` | 96% |
| `core.py` | 94% |
| `modeling/fitting.py` | 92% |
| `analysis/guinier.py` | 86% |
| `utils.py` | 81% |
| `__init__.py` | 65% (dead code -- see task 1.14) |

---

## New Tasks

### 1.11 -- Fix `plot_fit` `tight_layout` UserWarning

**Problem:** `plot_fit` calls `fig.tight_layout()` after creating a subplot
layout with `gridspec_kw`. When the residuals panel is included, matplotlib
emits:
```
UserWarning: This figure includes Axes that are not compatible with
tight_layout, so results might be incorrect.
```
This appears in 3 test runs and will appear for every user call with residuals.

**Fix:** Replace `fig.tight_layout()` with `fig.set_layout_engine("constrained")`
(matplotlib >= 3.6, already required by our `matplotlib>=3.5` pin) which
handles complex subplot layouts correctly.

**Files:** `scatterbrain/visualization.py`
**Tests:** The existing `test_with_residuals` case should produce no warning
after the fix (use `pytest.warns(None)` or verify via `recwarn`).

---

### 1.12 -- Add Typed Return Types for Analysis Results

**Problem:** `guinier_fit` and `porod_analysis` return plain `dict`. Users
get no IDE autocomplete, no field name checking, and no stable API contract.
This was identified as the highest-usability gap in the design review.

**Specification:** Add `TypedDict` definitions in each analysis module:

```python
# scatterbrain/analysis/guinier.py
from typing import TypedDict

class GuinierResult(TypedDict):
    Rg: float
    Rg_err: float
    I0: float
    I0_err: float
    q_fit_min: float
    q_fit_max: float
    num_points_fit: int
    r_squared: float
    method: str
```

```python
# scatterbrain/analysis/porod.py
class PorodResult(TypedDict, total=False):
    porod_exponent: float          # present when fit_log_log=True
    porod_exponent_err: float
    porod_constant_kp: float       # always present
    porod_constant_kp_err: float
    log_kp_intercept: float
    log_kp_intercept_err: float
    r_value: float
    q_fit_min: float
    q_fit_max: float
    num_points_fit: int
    method: str
```

**Behaviour:**
- Change the return annotation of `guinier_fit` from `Optional[Dict[str, Any]]`
  to `Optional[GuinierResult]` and of `porod_analysis` to `Optional[PorodResult]`.
- `TypedDict` is a pure type annotation -- no runtime change, fully backwards
  compatible. All existing callers using `result["Rg"]` continue to work.
- Export `GuinierResult` and `PorodResult` from `scatterbrain.analysis`.

**Files:**
- `scatterbrain/analysis/guinier.py`
- `scatterbrain/analysis/porod.py`
- `scatterbrain/analysis/__init__.py`

**Tests:** No new tests needed -- type correctness is verified by `mypy`.
Add a `mypy` smoke check to CI once `disallow_untyped_defs` is enabled (Phase 2).

---

### 1.13 -- Notebook Smoke Test in CI

**Problem:** `notebooks/01_basic_workflow.ipynb` is never executed in CI.
The notebook could silently break when the API changes. The design document
(sec.10) states notebooks serve as end-to-end acceptance tests.

**Specification:** Add `nbmake` to dev dependencies and a CI step:

```toml
# pyproject.toml [tool.uv] dev-dependencies -- add:
"nbmake>=1.4",
```

```yaml
# .github/workflows/ci.yml -- add to test job steps:
- name: Run tutorial notebook
  run: uv run pytest --nbmake notebooks/01_basic_workflow.ipynb
```

**Alternative:** Use `jupyter nbconvert --to notebook --execute` if `nbmake`
is not preferred. Either approach is acceptable.

**Files:**
- `pyproject.toml`
- `.github/workflows/ci.yml`

**Tests:** The notebook itself serves as the acceptance test. It must run
end-to-end on each Python version in the matrix.

---

### 1.14 -- Remove Dead Python <3.8 Code from `__init__.py`

**Problem:** `scatterbrain/__init__.py` contains an `ImportError` fallback
for `importlib.metadata` that targets Python <3.8:

```python
try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("scatterbrain")
    except PackageNotFoundError:
        __version__ = "0.0.1.dev0"
except ImportError:        # <- unreachable: Python >=3.10 always has importlib.metadata
    __version__ = "0.0.1.dev0"
```

`importlib.metadata` is in the standard library since Python 3.8. Since
`requires-python = ">=3.10"`, the outer `except ImportError` branch is dead
code and suppresses the `PackageNotFoundError` path that should be the only
fallback. It also pulls `__init__.py` coverage down to 65%.

**Fix:** Simplify to:

```python
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("scatterbrain")
except PackageNotFoundError:
    __version__ = "0.0.1.dev0"
```

**Files:** `scatterbrain/__init__.py`
**Tests:** Existing `test_basic.py` covers `__version__` -- no new tests needed.
Coverage on `__init__.py` should rise from 65% to ~85%+ after removal.

---

## Priority Order

```
1.14  Remove dead __init__.py code     <- trivial cleanup, coverage win
1.11  Fix tight_layout warning         <- affects every plot_fit call
1.12  Typed return types               <- usability; pure annotation, no risk
1.13  Notebook CI smoke test           <- acceptance test coverage
```

---

## Phase 1 Completion Criteria

| Criterion | Status |
|-----------|--------|
| `uv sync && python -c "from scatterbrain.analysis.guinier import guinier_fit"` works in fresh env | Done    |
| `subtract_background` implemented and all tests pass | Done    |
| `save_ascii_1d` implemented and round-trip test passes | Done    |
| `plot_guinier`, `plot_porod`, `plot_fit` implemented (no `NotImplementedError`) | Done    |
| `visualization_renamed/` removed | Done    |
| `notebooks/01_basic_workflow.ipynb` exists | Done    |
| `pytest` >= 150 passing, 0 failing | Done: 183 passing |
| Test coverage >= 80% on `scatterbrain/` (excluding `reduction/` placeholder) | Done: 93% total |
| GitHub Actions CI on Python 3.10, 3.11, 3.12 | Done    |
| `README.md` installation section uses `uv` | Done    |
| `plot_fit` produces no `UserWarning` | Done: Task 1.11 |
| Typed return types for `guinier_fit` / `porod_analysis` | Done: Task 1.12 |
| Notebook executed in CI | Done: Task 1.13 |
| Dead Python <3.8 code removed from `__init__.py` | Done: Task 1.14 |

---

## What is Explicitly Out of Scope for Phase 1

These belong to Phase 2 and should not be started now:

- Guinier auto-range refinement (weighted regression, iterative Rg limit)
- Pair Distance Distribution Function `p(r)` / IFT
- Additional form factors (cylinder, core-shell sphere)
- `lmfit` integration for modeling
- Structure factors `S(q)`
- 2D image loading / azimuthal integration
- Normalization by concentration/thickness in processing
- Interactive (plotly/bokeh) plots
- GUI
- Deprecation/stability policy and versioning strategy (Phase 2 concern once
  the API stabilises)
