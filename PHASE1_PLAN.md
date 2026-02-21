# ScatterBrain — Phase 1 Plan: Core Skeleton Implementation

**Version:** 1.0
**Branch:** `claude/review-scatter-brain-docs-SwvfP`
**Reference:** `Design_document.md` §8.2, `PHASE0_PLAN.md`

---

## Goal

Produce a minimally functional library that can:
1. Load 1D ASCII SAXS/WAXS data
2. Perform background subtraction
3. Run Guinier and Porod analysis
4. Fit a sphere form factor model
5. Generate standard diagnostic plots (I(q), Guinier, Porod, fit overlay)
6. Be installed via `pip install` and exercised through a tutorial notebook

---

## Current State Assessment

All Phase 1 tasks were audited against the live codebase. The test suite runs
**137 passing, 3 skipped** (all skipped are intentional matplotlib-backend skips).

| Task | Component | Status | Notes |
|------|-----------|--------|-------|
| 1. Project setup | `pyproject.toml`, `README.md`, `LICENSE`, `.gitignore` | ✅ Done | Sub-package discovery fixed; README updated with install/usage |
| 2. Core data structure | `scatterbrain/core.py` → `ScatteringCurve1D` | ✅ Done | Exceeds spec: adds `__getitem__`, `to_dict/from_dict`, `convert_q_unit` |
| 3. I/O load | `scatterbrain/io.py` → `load_ascii_1d` | ✅ Done | Robust pandas-based implementation |
| 3. I/O save | `scatterbrain/io.py` → `save_ascii_1d` | ✅ Done | Tab-delimited with auto comment header; round-trip tested |
| 4. Processing | `scatterbrain/processing/` → `subtract_background` | ✅ Done | Constant + curve modes, error propagation, interpolation |
| 5a. Analysis | `scatterbrain/analysis/guinier.py` → `guinier_fit` | ✅ Done | Auto q-range, error propagation |
| 5b. Analysis | `scatterbrain/analysis/porod.py` → `porod_analysis` | ✅ Done | Log-log fit + average-Kp mode |
| 6a. Modeling | `scatterbrain/modeling/form_factors.py` → `sphere_pq` | ✅ Done | Correct P(0)=1 limit |
| 6b. Modeling | `scatterbrain/modeling/fitting.py` → `fit_model` | ✅ Done | scale+bg wrapper, fixed params, chi² |
| 7a. Visualization | `plot_iq` | ✅ Done | Multi-curve, error bars, auto axis labels |
| 7b. Visualization | `plot_guinier` | ✅ Done | ln(I) vs q², fit overlay, Rg/I(0) annotation |
| 7c. Visualization | `plot_porod` | ✅ Done | Iq4_vs_q and logI_vs_logq modes |
| 7d. Visualization | `plot_fit` | ✅ Done | Main panel + optional normalised residuals |
| 8. Utilities | `utils.py` | ✅ Done | `convert_q_array`, custom exceptions, logging |
| 9. Tests | `tests/` | ✅ Done | 183 passing, 93% coverage (target: ≥150, ≥80%) |
| 10. Docs infrastructure | `docs/source/conf.py`, `index.rst` | ✅ Done | Present; sub-module autodoc needs verification |
| 10. Tutorial notebook | `notebooks/01_basic_workflow.ipynb` | ✅ Done | Full end-to-end workflow with 7 steps |
| —  | GitHub Actions CI | ✅ Done | `.github/workflows/ci.yml` — Python 3.10/3.11/3.12 |
| —  | `visualization_renamed/` artifact | ✅ Done | Removed |

---

## Phase 1 Task List

### 1.1 — Fix Packaging: Sub-package Discovery

**Problem:** `pyproject.toml` has `packages = ["scatterbrain"]`. This tells setuptools
to install only the top-level package, missing `scatterbrain.analysis`,
`scatterbrain.modeling`, `scatterbrain.processing`, and `scatterbrain.reduction`.
The library appears to work in development mode (`pip install -e .`) because Python
resolves imports from the source tree, but a regular `pip install` would break.

**Fix required in `pyproject.toml`:**
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["scatterbrain*"]
```
Remove the existing `[tool.setuptools] packages = ["scatterbrain"]` stanza.

**Files:** `pyproject.toml`
**Tests:** Verify `import scatterbrain.analysis.guinier` works from an installed wheel.

---

### 1.2 — Implement `save_ascii_1d`

**Design doc reference:** `Design_document.md` §5.1

**Specification:**
```python
def save_ascii_1d(
    curve: ScatteringCurve1D,
    filepath: Union[str, pathlib.Path],
    include_error: bool = True,
    delimiter: str = "\t",
    header: Optional[str] = None,
    fmt: str = "%.6e",
) -> None
```

**Behaviour:**
- Write a tab (or user-specified delimiter) separated file with columns:
  `q`, `intensity`, and optionally `error` (if `curve.error` is not `None`
  and `include_error=True`).
- Prepend an auto-generated header comment block:
  `# q [{q_unit}]`, `# intensity [{intensity_unit}]`, `# Saved by ScatterBrain`,
  plus any user-supplied `header` string.
- Raise `ValueError` if `filepath` parent directory does not exist.

**Files:** `scatterbrain/io.py`
**Tests:** `tests/test_io.py` — round-trip test: `load_ascii_1d(save_ascii_1d(curve))` must
recover q, intensity, error within floating-point tolerance.

---

### 1.3 — Implement `subtract_background`

**Design doc reference:** `Design_document.md` §5.3

**Specification:**
```python
# scatterbrain/processing/background.py
def subtract_background(
    curve: ScatteringCurve1D,
    background: Union[ScatteringCurve1D, float],
    interpolate: bool = True,
    scale_factor: float = 1.0,
) -> ScatteringCurve1D
```

**Behaviour:**
- If `background` is a `float`: subtract it as a constant from all intensity values.
- If `background` is a `ScatteringCurve1D`:
  - If `interpolate=True`: interpolate background onto the signal curve's q-grid
    using `numpy.interp` before subtracting.
  - If `interpolate=False`: require identical q-grids; raise `ValueError` if they
    differ.
  - Apply optional `scale_factor` to the background before subtraction.
- Propagate errors in quadrature if both curves have error arrays.
- Return a **new** `ScatteringCurve1D`; do not modify inputs.
- Record the operation in `metadata["processing_history"]`.

**Files:**
- Create `scatterbrain/processing/background.py`
- Update `scatterbrain/processing/__init__.py` to expose `subtract_background`

**Tests:** `tests/test_processing.py` (new file):
- Constant background subtraction (with and without errors)
- Curve background subtraction with matching q-grids
- Curve background subtraction with interpolation
- `scale_factor` effect
- Error propagation correctness
- Type error for invalid background type

---

### 1.4 — Implement `plot_guinier`

**Design doc reference:** `Design_document.md` §5.6

**Specification:**
```python
def plot_guinier(
    curve: ScatteringCurve1D,
    guinier_result: Optional[Dict[str, Any]] = None,
    q_range_highlight: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = "Guinier Plot",
    **plot_kwargs: Any,
) -> Tuple[Figure, Axes]
```

**Behaviour:**
- Plot `ln(I(q))` vs `q²`, excluding points where `I(q) ≤ 0`.
- If `guinier_result` is provided (output of `guinier_fit`):
  - Overlay the fitted line over the fit q-range.
  - Annotate the axes with `Rg = X.XX ± Y.YY nm⁻¹` and `I(0) = Z.ZZ`.
- If `q_range_highlight` is provided, shade the fit region.
- Use error bars on ln(I) if `curve.error` is available (propagated:
  `σ_lnI = σ_I / I`).
- Return `(fig, ax)`.

**Files:** `scatterbrain/visualization.py`
**Tests:** `tests/test_visualization.py` — assert figure is returned, no exception
raised with/without `guinier_result`, with/without errors.

---

### 1.5 — Implement `plot_porod`

**Design doc reference:** `Design_document.md` §5.6

**Specification:**
```python
def plot_porod(
    curve: ScatteringCurve1D,
    porod_result: Optional[Dict[str, Any]] = None,
    plot_type: str = "Iq4_vs_q",   # or "logI_vs_logq"
    q_range_highlight: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = "Porod Plot",
    **plot_kwargs: Any,
) -> Tuple[Figure, Axes]
```

**Behaviour:**
- `"Iq4_vs_q"` mode: plot `I(q)·q⁴` vs `q` (linear axes); a flat plateau
  indicates Porod behaviour.
- `"logI_vs_logq"` mode: plot `log₁₀(I)` vs `log₁₀(q)`; slope ≈ −4 for
  smooth interfaces.
- If `porod_result` is provided, overlay the fitted line and annotate with
  Porod exponent and constant.
- If `q_range_highlight` is provided, shade the analysis region.
- Return `(fig, ax)`.

**Files:** `scatterbrain/visualization.py`
**Tests:** `tests/test_visualization.py` — both `plot_type` modes, with/without
result dict.

---

### 1.6 — Implement `plot_fit`

**Design doc reference:** `Design_document.md` §5.6

**Specification:**
```python
def plot_fit(
    curve: ScatteringCurve1D,
    fit_result_dict: Dict[str, Any],   # output of fit_model
    q_scale: str = "log",
    i_scale: str = "log",
    plot_residuals: bool = True,
    ax_main: Optional[Axes] = None,
    ax_res: Optional[Axes] = None,
    title: Optional[str] = "Model Fit",
    **plot_kwargs: Any,
) -> Figure
```

**Behaviour:**
- Upper panel: experimental `curve` (with error bars if available) + fitted
  `fit_result_dict["fit_curve"]` overlay.
- Lower panel (if `plot_residuals=True` and errors available): normalised
  residuals `(I_data − I_model) / σ` vs `q` with a zero line.
- Annotate with reduced chi-squared (`fit_result_dict["chi_squared_reduced"]`)
  and key fitted parameters (`scale`, `background`, model params).
- If `ax_main` is provided but `ax_res` is not (or `plot_residuals=False`),
  plot only on `ax_main`.
- Return the `Figure`.

**Files:** `scatterbrain/visualization.py`
**Tests:** `tests/test_visualization.py` — with/without residuals, with/without
supplied axes, with/without errors on curve.

---

### 1.7 — Clean Up `visualization_renamed/` Artifact

**Problem:** `scatterbrain/visualization_renamed/` is an empty directory with only
an `__init__.py`. It appears to be a leftover from an aborted rename and is
not referenced anywhere.

**Fix:** Remove the directory from the repository.

**Files:** `scatterbrain/visualization_renamed/` (delete)

---

### 1.8 — Tutorial Notebook

**Design doc reference:** `Design_document.md` §8.2, §11

**Requirement:** At least one Jupyter Notebook in `notebooks/` that demonstrates
the complete MVP workflow end-to-end.

**Suggested content (`notebooks/01_basic_workflow.ipynb`):**
1. Load example data from `scatterbrain/examples/data/example_sphere_data.dat`
2. Inspect and plot the raw curve (`plot_iq`)
3. Subtract a constant background
4. Run Guinier analysis and display results + Guinier plot
5. Run Porod analysis and display results + Porod plot
6. Fit a sphere form factor with `fit_model` + display fit plot
7. Save the processed curve with `save_ascii_1d`

**Files:** Create `notebooks/01_basic_workflow.ipynb`

---

### 1.9 — GitHub Actions CI

**Design doc reference:** `Design_document.md` §7, §10

**Specification:** Create `.github/workflows/ci.yml` that:
- Triggers on push and pull request to `main`/`master`.
- Tests on Python 3.10, 3.11, 3.12.
- Steps: checkout → install with `pip install -e ".[dev]"` → `black --check` →
  `flake8` → `mypy scatterbrain` → `pytest --cov=scatterbrain`.
- Optionally build docs: `sphinx-build docs/source docs/_build`.

**Files:** `.github/workflows/ci.yml` (new)

---

### 1.10 — Tests for New Components

Summary of new/expanded test files needed:

| File | New Tests Required |
|------|--------------------|
| `tests/test_io.py` | `save_ascii_1d` round-trip, header format, error column handling |
| `tests/test_processing.py` | All `subtract_background` cases (new file) |
| `tests/test_visualization.py` | `plot_guinier`, `plot_porod`, `plot_fit` (all modes) |

**Coverage target:** ≥ 80% branch coverage on `processing/background.py`,
`io.py` save section, and all three new visualization functions.

---

## Priority Order

Implement in this order to unblock downstream tasks:

```
1.1  pyproject.toml packaging fix       ← foundation; unblocks pip install
1.3  subtract_background                ← needed for tutorial notebook
1.2  save_ascii_1d                      ← needed for tutorial notebook
1.4  plot_guinier                       ← needed for tutorial notebook
1.5  plot_porod                         ← needed for tutorial notebook
1.6  plot_fit                           ← needed for tutorial notebook
1.7  Remove visualization_renamed/      ← housekeeping
1.8  Tutorial notebook                  ← requires 1.1–1.6
1.9  GitHub Actions CI                  ← independent, can be done anytime
1.10 Tests                              ← written alongside each task above
```

---

## Phase 1 Completion Criteria

Phase 1 is complete when **all** of the following are true:

- [x] `pip install .` (not editable) installs all sub-packages correctly
- [x] `from scatterbrain.analysis.guinier import guinier_fit` works in a fresh env
- [x] `subtract_background` is implemented and passes all tests
- [x] `save_ascii_1d` is implemented and round-trip test passes
- [x] `plot_guinier`, `plot_porod`, `plot_fit` are implemented (no `NotImplementedError`)
- [x] `visualization_renamed/` is removed
- [x] `notebooks/01_basic_workflow.ipynb` runs top-to-bottom without errors
- [x] `pytest` reports ≥ 150 passing tests, 0 failing — **183 passing, 93% coverage**
- [x] Test coverage ≥ 80% on `scatterbrain/` (excluding `reduction.py` placeholder)
- [x] GitHub Actions CI configured for Python 3.10, 3.11, 3.12
- [x] `README.md` installation section is updated with working `pip install` instructions

**Phase 1 is COMPLETE ✓**

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
