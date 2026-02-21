# ScatterBrain — Phase 2 Plan: Enhancing Core Functionality & Usability

**Version:** 1.0
**Branch:** `claude/phase2-plan-SwvfP`
**Reference:** `Design_document.md` §2.2, §8.3, `PHASE1_PLAN.md`

---

## Goal

Build on the complete Phase 1 MVP to deliver a richer, more capable library that
covers the most common SAXS/WAXS analysis workflows beyond the baseline:

1. Expand the processing pipeline (normalization, trimming, curve merging)
2. Add two more form factors (cylinder, core-shell sphere)
3. Integrate `lmfit` for parameter-rich, uncertainty-aware model fitting
4. Implement Kratky analysis and visualization
5. Implement the Pair Distance Distribution Function p(r) via Indirect Fourier
   Transform (IFT)
6. Refine the Guinier fit with an iterative auto-range algorithm
7. Build out Sphinx API documentation from the existing infrastructure
8. Deliver two additional tutorial notebooks

---

## Current State Assessment

Phase 1 is **fully complete** as of commit `7f7dde9` (merged in PR #6). All Phase 1
completion criteria are satisfied.

| Module | Phase 1 Status | Phase 2 Starting Point |
|--------|---------------|------------------------|
| `scatterbrain.core` | ✅ Complete | No changes required |
| `scatterbrain.io` | ✅ Complete | No changes required |
| `scatterbrain.processing` | ✅ `subtract_background` done | Add `normalize_curve`, `trim_curve`, `merge_curves` |
| `scatterbrain.analysis.guinier` | ✅ Basic implementation done | Refine auto-range selection |
| `scatterbrain.analysis.porod` | ✅ Complete | No changes required |
| `scatterbrain.analysis.kratky` | 🔲 Does not exist | Implement from scratch |
| `scatterbrain.analysis.ift` | 🔲 Does not exist | Implement p(r) from scratch |
| `scatterbrain.modeling.form_factors` | ✅ `sphere_pq` done | Add `cylinder_pq`, `core_shell_sphere_pq` |
| `scatterbrain.modeling.fitting` | ✅ `fit_model` (scipy) done | Add `fit_model_lmfit` |
| `scatterbrain.visualization` | ✅ All four plot types done | Add `plot_kratky` |
| `scatterbrain.reduction` | 🔲 Stub only | Deferred to Phase 3 |
| `docs/source/` | ⚠️ Infrastructure only | Populate RST API reference |
| `notebooks/` | ✅ `01_basic_workflow.ipynb` | Add notebooks 02, 03 |

---

## Phase 2 Task List

### 2.1 — Processing: `normalize_curve`

**Design doc reference:** `Design_document.md` §2.2, §5.3

**Specification:**
```python
# scatterbrain/processing/normalization.py
def normalize_curve(
    curve: ScatteringCurve1D,
    method: str = "i0",          # "i0" | "monitor" | "value"
    i0_result: Optional[Dict[str, Any]] = None,   # output of guinier_fit
    monitor_value: Optional[float] = None,
    target_value: Optional[float] = None,
) -> ScatteringCurve1D
```

**Behaviour:**
- `"i0"` mode: divide all intensities by `i0_result["i0"]` (requires
  `i0_result` from `guinier_fit`). Error propagated in quadrature using
  `σ_I/I0`.
- `"monitor"` mode: divide all intensities (and errors) by `monitor_value`.
  Raise `ValueError` if `monitor_value` is `None` or `≤ 0`.
- `"value"` mode: divide all intensities by `target_value`. Raise `ValueError`
  if `target_value` is `None` or `≤ 0`.
- All modes: return a **new** `ScatteringCurve1D`; do not modify the input.
- Record the normalization method and factor in
  `metadata["processing_history"]`.

**Files:**
- Create `scatterbrain/processing/normalization.py`
- Update `scatterbrain/processing/__init__.py` to export `normalize_curve`

**Tests:** `tests/test_processing.py` (extend existing file):
- Each of the three normalization modes with and without error arrays
- Error propagation correctness for `"i0"` mode
- `ValueError` raised for invalid `monitor_value` / `target_value`
- `ValueError` raised when `"i0"` mode called without `i0_result`
- Metadata `processing_history` entry present in result

---

### 2.2 — Processing: `trim_curve`

**Design doc reference:** `Design_document.md` §2.2, §5.3

**Specification:**
```python
# scatterbrain/processing/trimming.py
def trim_curve(
    curve: ScatteringCurve1D,
    q_min: Optional[float] = None,
    q_max: Optional[float] = None,
) -> ScatteringCurve1D
```

**Behaviour:**
- Return a new `ScatteringCurve1D` containing only points where
  `q_min <= q <= q_max` (inclusive, using `numpy` boolean indexing).
- If `q_min` is `None`, no lower bound is applied.
- If `q_max` is `None`, no upper bound is applied.
- If both are `None`, return a copy of the input unchanged.
- Raise `ValueError` if `q_min >= q_max` (when both are provided).
- Raise `ValueError` if the resulting curve has fewer than 2 points.
- Preserve `metadata`; add a `processing_history` entry noting the q-range
  applied.

**Files:**
- Create `scatterbrain/processing/trimming.py`
- Update `scatterbrain/processing/__init__.py` to export `trim_curve`

**Tests:** `tests/test_processing.py`:
- Trim with q_min only, q_max only, both
- No-op when both None
- Correct point count in result
- Errors preserved correctly (sliced in sync with q and intensity)
- `ValueError` on invalid bounds
- `ValueError` when result has fewer than 2 points

---

### 2.3 — Processing: `merge_curves`

**Design doc reference:** `Design_document.md` §2.2

**Specification:**
```python
# scatterbrain/processing/merging.py
def merge_curves(
    low_q_curve: ScatteringCurve1D,
    high_q_curve: ScatteringCurve1D,
    q_overlap_min: Optional[float] = None,
    q_overlap_max: Optional[float] = None,
    scale_to: str = "low_q",       # "low_q" | "high_q" | "none"
) -> ScatteringCurve1D
```

**Behaviour:**
- Concatenate `low_q_curve` (points with `q < q_overlap_min` or below
  the auto-detected overlap) and `high_q_curve` (points with `q > q_overlap_max`
  or above the auto-detected overlap) into a single sorted `ScatteringCurve1D`.
- If `q_overlap_min` and `q_overlap_max` are both `None`, auto-detect the
  overlap region as `[max(q_low.min(), q_high.min()), min(q_low.max(), q_high.max())]`.
  Raise `ValueError` if no overlap exists.
- `scale_to="low_q"`: compute a scale factor from the mean ratio of overlapping
  intensities and multiply `high_q_curve` intensities by it before merging.
- `scale_to="high_q"`: same but scale `low_q_curve`.
- `scale_to="none"`: no scaling applied.
- The returned curve's `q_unit` must match between both inputs; raise
  `ValueError` otherwise.
- Error arrays merged in the same order as q/intensity. If only one input has
  errors, errors are set to `None` in the merged result.
- Record scale factor and overlap range in `metadata["processing_history"]`.

**Files:**
- Create `scatterbrain/processing/merging.py`
- Update `scatterbrain/processing/__init__.py` to export `merge_curves`

**Tests:** `tests/test_processing.py`:
- Merge with explicit overlap bounds
- Merge with auto-detected overlap
- Scale-to-low-q, scale-to-high-q, and no-scaling modes
- `ValueError` for mismatched q-units
- `ValueError` when no overlap exists
- Output q-array is sorted
- Error handling when one or both inputs lack error arrays

---

### 2.4 — Modeling: `cylinder_pq`

**Design doc reference:** `Design_document.md` §8.3

**Specification:**
```python
# scatterbrain/modeling/form_factors.py  (extend existing file)
def cylinder_pq(
    q: np.ndarray,
    radius: float,
    length: float,
) -> np.ndarray
```

**Behaviour:**
- Compute the orientationally-averaged form factor P(q) for a monodisperse
  solid cylinder of circular cross-section using the standard double integral
  over orientation angle α:

  ```
  P(q) = ∫₀^{π/2} [2·J₁(q·R·sin α)/(q·R·sin α) · sin(q·L·cos α/2)/(q·L·cos α/2)]² sin α dα
  ```

  where `J₁` is the first-order Bessel function of the first kind
  (`scipy.special.j1`), `R = radius`, `L = length`.
- Use Gaussian quadrature (`scipy.integrate.quad` or `numpy` Gauss–Legendre
  nodes) for numerical integration over α. A minimum of 50 quadrature points
  is required for accuracy.
- P(q=0) = 1 by definition; handle this limit analytically.
- Raise `ValueError` if `radius ≤ 0` or `length ≤ 0`.
- Return a 1-D `np.ndarray` of shape `(len(q),)` with values in `[0, 1]`.

**Files:** `scatterbrain/modeling/form_factors.py`
**Tests:** `tests/test_modeling.py`:
- P(q=0) ≈ 1 (within 1e-6)
- Monotonically decreasing at low q for a long cylinder
- Output shape matches input q shape
- `ValueError` on non-positive radius or length
- Cross-check a few q-values against a known reference or analytical limit

---

### 2.5 — Modeling: `core_shell_sphere_pq`

**Design doc reference:** `Design_document.md` §8.3

**Specification:**
```python
# scatterbrain/modeling/form_factors.py  (extend existing file)
def core_shell_sphere_pq(
    q: np.ndarray,
    r_core: float,
    t_shell: float,
    sld_core: float,
    sld_shell: float,
    sld_solvent: float,
) -> np.ndarray
```

**Behaviour:**
- Compute the form factor amplitude `F(q)` for a core-shell sphere and return
  `P(q) = [F(q) / F(0)]²` (normalized to 1 at q=0):

  ```
  F(q) = (4π/3) · [(sld_core - sld_shell) · r_core³ · Φ(q·r_core)
                  + (sld_shell - sld_solvent) · (r_core+t_shell)³ · Φ(q·(r_core+t_shell))]
  ```

  where `Φ(x) = 3(sin x − x cos x)/x³` is the sphere amplitude function.
- `r_core` and `t_shell` are in the same length unit (nm or Å); SLD values
  are in consistent units (e.g., nm⁻²) — the function does **not** enforce
  units but is consistent internally.
- Handle `q → 0` analytically (limit of Φ is 1).
- Raise `ValueError` if `r_core ≤ 0` or `t_shell < 0`.

**Files:** `scatterbrain/modeling/form_factors.py`
**Tests:** `tests/test_modeling.py`:
- P(q=0) = 1
- When `sld_core == sld_shell` degenerates to a simple sphere with `r_core+t_shell`
- `ValueError` on non-positive `r_core` or negative `t_shell`
- Output shape matches input q

---

### 2.6 — Modeling: `fit_model_lmfit`

**Design doc reference:** `Design_document.md` §8.3

**Specification:**
```python
# scatterbrain/modeling/fitting_lmfit.py  (new file)
def fit_model_lmfit(
    curve: ScatteringCurve1D,
    model_func: Callable,
    params: "lmfit.Parameters",
    q_range: Optional[Tuple[float, float]] = None,
    fit_method: str = "leastsq",
    weights: str = "error",        # "error" | "none"
) -> Dict[str, Any]
```

**Behaviour:**
- Wrap `lmfit.minimize` (or `lmfit.Model`) around `model_func` to perform
  parameter-rich fitting with bounds, expressions, and uncertainty estimation.
- `params`: a fully specified `lmfit.Parameters` object (names, initial values,
  bounds, whether each is fixed) passed by the user. The function does **not**
  auto-add `scale` or `background` — the user declares all parameters via
  `params`.
- `weights="error"`: weight residuals by `1/σ_I` where σ is `curve.error`.
  Raise `ValueError` if `curve.error` is `None` and `weights="error"`.
- `weights="none"`: unweighted residuals.
- `q_range`: trim the curve before fitting (internally via `trim_curve`).
- Return a dictionary with keys:
  - `"lmfit_result"`: the raw `lmfit.MinimizerResult` object
  - `"params"`: `dict` of `{name: value}` for best-fit parameters
  - `"param_errors"`: `dict` of `{name: stderr}` (None if not estimated)
  - `"fit_curve"`: `ScatteringCurve1D` of the best-fit model evaluated on the
    fitted q-range
  - `"chi_squared_reduced"`: float
  - `"q_range_used"`: tuple of (q_min, q_max)
  - `"success"`: bool
- Raise `ImportError` with a clear message if `lmfit` is not installed.

**Files:**
- Create `scatterbrain/modeling/fitting_lmfit.py`
- Update `scatterbrain/modeling/__init__.py` to export `fit_model_lmfit`
  (conditional on `lmfit` availability)
- Add `lmfit` as an optional dependency in `pyproject.toml` under
  `[project.optional-dependencies]` key `lmfit`

**Tests:** `tests/test_modeling.py`:
- Fitting a sphere form factor with known parameters recovers them within
  uncertainty
- Output dictionary contains all required keys
- `ValueError` raised when `weights="error"` and no error array
- `ImportError` path tested by mocking `lmfit` as unavailable
- Fixed parameters respected (verify `stderr` is 0 or None for fixed params)

---

### 2.7 — Analysis: `kratky_analysis`

**Design doc reference:** `Design_document.md` §2.2

**Specification:**
```python
# scatterbrain/analysis/kratky.py  (new file)
def kratky_analysis(
    curve: ScatteringCurve1D,
    i0: Optional[float] = None,
    rg: Optional[float] = None,
    normalized: bool = False,
) -> Dict[str, Any]
```

**Behaviour:**
- Compute the Kratky representation `q² · I(q)` vs `q` (classical Kratky).
- If `normalized=True` and both `i0` and `rg` are provided, compute the
  dimensionless normalized Kratky plot:
  `(q · Rg)² · I(q) / I(0)` vs `q · Rg`.
  Raise `ValueError` if `normalized=True` but `i0` or `rg` is missing.
- Return a dictionary:
  - `"q"`: q-array used
  - `"kratky_y"`: `q² · I(q)` array (unnormalized) or normalized equivalent
  - `"kratky_y_error"`: propagated error array (or `None` if `curve.error` is `None`)
  - `"normalized"`: bool flag indicating which representation was computed
  - `"i0"`, `"rg"`: values used for normalization (or `None`)
  - `"peak_q"`: q-value at the maximum of `kratky_y` (simple `argmax`)

**Files:**
- Create `scatterbrain/analysis/kratky.py`
- Update `scatterbrain/analysis/__init__.py` to export `kratky_analysis`

**Tests:** `tests/test_analysis.py` (extend existing file):
- Unnormalized output: `kratky_y` matches `q² * curve.intensity`
- Normalized output: verify scale at `q·Rg = √3` is correct for Guinier model
- `ValueError` when `normalized=True` but `i0`/`rg` missing
- `peak_q` is within the input q-range
- Error propagation: `kratky_y_error ≈ q² * curve.error`

---

### 2.8 — Visualization: `plot_kratky`

**Design doc reference:** `Design_document.md` §5.6

**Specification:**
```python
# scatterbrain/visualization.py  (extend existing file)
def plot_kratky(
    curve: ScatteringCurve1D,
    kratky_result: Optional[Dict[str, Any]] = None,
    normalized: bool = False,
    i0: Optional[float] = None,
    rg: Optional[float] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = "Kratky Plot",
    **plot_kwargs: Any,
) -> Tuple[Figure, Axes]
```

**Behaviour:**
- If `kratky_result` is provided (output of `kratky_analysis`), plot from it
  directly. Otherwise compute `q² · I(q)` inline.
- If `normalized=True` (and `i0`, `rg` provided or taken from `kratky_result`):
  plot `(q·Rg)² · I/I₀` vs `q·Rg`. Add a visual reference line at
  `(q·Rg)² · I/I₀ = 1.104` (the theoretical Guinier peak maximum for a globular
  particle), drawn as a dashed grey line.
- If `normalized=False`: plot `q² · I(q)` vs `q`.
- Include error bars if `curve.error` is available (propagated as for
  `kratky_analysis`).
- x-axis label: `"q·Rg"` (normalized) or `f"q [{curve.q_unit}]"` (unnormalized).
- y-axis label: `"(q·Rg)²·I(q)/I(0)"` (normalized) or `"q²·I(q)"`.
- Return `(fig, ax)`.

**Files:** `scatterbrain/visualization.py`
**Tests:** `tests/test_visualization.py`:
- Unnormalized call returns `(fig, ax)`
- Normalized call with `i0`/`rg` returns `(fig, ax)`
- Supplying `kratky_result` dict works without additionally passing `i0`/`rg`
- Supplying an external `ax` does not create a new figure
- Error bars rendered when `curve.error` is not None

---

### 2.9 — Analysis: Guinier Auto-Range Refinement

**Design doc reference:** `Design_document.md` §8.3

**Target function:** `scatterbrain/analysis/guinier.py` → `guinier_fit`

**Current behaviour:** accepts a single `qrg_limit_max` threshold and
`auto_q_selection_fraction` to select the fit window. This can return a poor
window if data is noisy or the initial Rg guess is far from the true value.

**Refinement — iterative self-consistent algorithm:**

Add an optional `iterative` parameter:
```python
def guinier_fit(
    curve: ScatteringCurve1D,
    q_range: Optional[Tuple[float, float]] = None,
    qrg_limit_max: float = 1.3,
    auto_q_selection_fraction: float = 0.25,
    iterative: bool = False,           # NEW
    max_iter: int = 20,                # NEW
    convergence_tol: float = 1e-4,     # NEW
) -> Dict[str, Any]
```

**Iterative algorithm** (only active when `iterative=True`):
1. Run the current single-pass fit to get an initial `Rg_0`.
2. Recompute the q upper bound as `q_max_new = qrg_limit_max / Rg_0`.
3. Re-run the fit on `[q_min, q_max_new]`.
4. Repeat steps 2–3 until `|Rg_new − Rg_old| / Rg_old < convergence_tol`
   or `max_iter` is reached.
5. Return the converged result. If `max_iter` is reached without convergence,
   add a `"converged": False` key to the output dict and a warning via the
   module logger; otherwise `"converged": True`.

The returned dictionary gains:
- `"converged"`: bool (always present when `iterative=True`)
- `"n_iterations"`: int — number of iterations performed

**Files:** `scatterbrain/analysis/guinier.py`
**Tests:** `tests/test_analysis.py`:
- `iterative=False` (default) produces identical results to current behaviour
- `iterative=True` converges for clean synthetic data
- `converged=True` in result when tolerance is met
- `converged=False` set when `max_iter` is too small to converge
- `n_iterations` is within `[1, max_iter]`

---

### 2.10 — Analysis: `pair_distance_distribution` (p(r) via IFT)

**Design doc reference:** `Design_document.md` §2.2

**Specification:**
```python
# scatterbrain/analysis/ift.py  (new file)
def pair_distance_distribution(
    curve: ScatteringCurve1D,
    d_max: float,
    n_terms: int = 20,
    alpha: Optional[float] = None,
    alpha_range: Tuple[float, float] = (1e-10, 1e3),
    r_points: int = 100,
) -> Dict[str, Any]
```

**Behaviour:**
- Implement the Moore IFT method: represent p(r) as a truncated sine series
  `p(r) = Σ cₙ · sin(n·π·r/d_max)` over `[0, d_max]`, then determine
  coefficients `{cₙ}` by minimizing a regularized least-squares problem:

  ```
  χ²(c) = Σ [(I_data(qᵢ) − Ĩ(qᵢ, c))² / σᵢ²]  +  α · ||L·c||²
  ```

  where `Ĩ` is the Fourier transform of the trial p(r) evaluated at each qᵢ,
  `L` is a smoothness regularization matrix (second-difference), and `α` is
  the regularization parameter.
- If `alpha=None`, perform a scan over `alpha_range` (log-spaced, 50 points)
  and choose the α that minimizes the GCV (Generalized Cross-Validation) score.
- p(r) is evaluated on a uniform grid of `r_points` points over `[0, d_max]`.
- Return a dictionary:
  - `"r"`: `np.ndarray` — r values
  - `"pr"`: `np.ndarray` — p(r) values
  - `"pr_error"`: `np.ndarray` — propagated error on p(r) (from covariance of
    `{cₙ}`)
  - `"d_max"`: float — as supplied
  - `"alpha"`: float — regularization parameter used
  - `"i0"`: float — p(r) integral `4π ∫ p(r) dr` = I(0)
  - `"rg_from_pr"`: float — Rg from `sqrt(∫ r² p(r) dr / (2 ∫ p(r) dr))`
  - `"chi_squared_reduced"`: float — quality of fit back to I(q) data
  - `"n_terms"`: int — sine terms used
- Raise `ValueError` if `d_max ≤ 0` or `n_terms < 2`.
- Raise `ValueError` if `curve.error` is `None` (errors are required for
  proper regularization).

**Files:**
- Create `scatterbrain/analysis/ift.py`
- Update `scatterbrain/analysis/__init__.py` to export
  `pair_distance_distribution`

**Tests:** `tests/test_analysis.py`:
- For a sphere of known radius R, `d_max = 2R`: verify p(r) peaks near `r = R`
  and vanishes at `r = 0` and `r = d_max`
- `rg_from_pr` is within 5% of the known Rg for synthetic sphere data
- `i0` is consistent with Guinier extrapolation (within 10%)
- `ValueError` on `d_max ≤ 0`, `n_terms < 2`, missing error array
- Alpha scan produces a positive alpha when `alpha=None`
- `"pr_error"` has the same shape as `"pr"`

---

### 2.11 — Sphinx API Documentation

**Design doc reference:** `Design_document.md` §10, §11

**Current state:** `docs/source/conf.py` exists with Sphinx + RTD theme +
myst-parser configured. An `index.rst` stub is present but the API reference
contains no module-level RST pages.

**Required work:**

1. Create `docs/source/api/` directory with one RST page per top-level module:
   - `scatterbrain.rst` — top-level public API
   - `core.rst`
   - `io.rst`
   - `processing.rst`
   - `analysis.rst`
   - `modeling.rst`
   - `visualization.rst`
   - `utils.rst`
2. Each RST uses `.. automodule::` with `:members:`, `:undoc-members:`,
   `:show-inheritance:` directives.
3. Update `docs/source/index.rst` to include the API toctree.
4. Add a `docs/source/user_guide/` section with one page summarizing the
   Phase 2 workflow (cross-linking to notebooks).
5. Verify `sphinx-build -W -b html docs/source docs/_build/html` exits 0
   (no warnings treated as errors for missing references).

**Files:**
- `docs/source/api/*.rst` (8 new files)
- `docs/source/user_guide/phase2_workflow.rst` (1 new file)
- `docs/source/index.rst` (update)

---

### 2.12 — Tutorial Notebooks

**Design doc reference:** `Design_document.md` §8.3, §11

Two new notebooks, each runnable top-to-bottom with no user interaction:

#### `notebooks/02_advanced_analysis.ipynb`

Demonstrates Phase 2 analysis features:
1. Load example data (`scatterbrain/examples/data/example_sphere_data.dat`)
2. Trim to a useful q-range with `trim_curve`
3. Normalize by I(0) from `guinier_fit` using `normalize_curve`
4. Run `guinier_fit` with `iterative=True`; compare Rg with non-iterative result
5. Run `kratky_analysis`; call `plot_kratky` in both normalized and unnormalized
   modes; interpret the shape
6. Run `pair_distance_distribution` with `d_max` set to 2·Rg (a starting guess);
   inspect p(r), Rg_from_pr, I(0); plot p(r) manually

#### `notebooks/03_form_factor_fitting.ipynb`

Demonstrates expanded modeling:
1. Load data
2. Background subtract; trim; normalize
3. Fit `cylinder_pq` with `fit_model`; display fit plot
4. Fit `core_shell_sphere_pq` with `fit_model`; display fit plot
5. Fit `sphere_pq` using `fit_model_lmfit` with parameter bounds; compare
   uncertainties against `fit_model` (scipy) results
6. Merge two synthetic low-q and high-q datasets with `merge_curves`; refit

**Files:**
- `notebooks/02_advanced_analysis.ipynb`
- `notebooks/03_form_factor_fitting.ipynb`

---

### 2.13 — Tests for All New Components

Full test accounting for Phase 2 additions:

| File | Tests to Add |
|------|-------------|
| `tests/test_processing.py` | `normalize_curve` (3 modes, error prop), `trim_curve` (bounds, edge cases), `merge_curves` (scaling modes, q-sort, error handling) |
| `tests/test_analysis.py` | `kratky_analysis` (normalized/unnormalized, error prop), `guinier_fit` iterative refinement, `pair_distance_distribution` (known sphere, error cases) |
| `tests/test_modeling.py` | `cylinder_pq` (P(0)=1, shape, errors), `core_shell_sphere_pq` (degenerate case, P(0)=1), `fit_model_lmfit` (convergence, fixed params, error cases) |
| `tests/test_visualization.py` | `plot_kratky` (normalized/unnormalized, external ax, error bars) |

**Coverage target:** ≥ 85% branch coverage on all new modules. Overall
project coverage must not decrease below the Phase 1 baseline.

---

## Priority Order

Implement in this order to ensure each task unblocks the next and notebooks
can be written last:

```
2.1   normalize_curve          ← simple; used in notebooks
2.2   trim_curve               ← simple; used everywhere
2.3   merge_curves             ← moderate; used in notebook 03
2.4   cylinder_pq              ← self-contained; needed for notebook 03
2.5   core_shell_sphere_pq     ← self-contained; needed for notebook 03
2.6   fit_model_lmfit          ← depends on trim_curve (2.2); needed for notebook 03
2.7   kratky_analysis          ← depends on working curve; needed for notebook 02
2.8   plot_kratky              ← depends on kratky_analysis (2.7)
2.9   Guinier refinement       ← isolated change to existing function
2.10  pair_distance_distribution ← most complex; depends on trim_curve (2.2)
2.11  Sphinx API docs          ← independent; can be done in parallel with 2.1–2.3
2.12  Tutorial notebooks       ← requires all of 2.1–2.10
2.13  Tests                    ← written alongside each task above
```

---

## Phase 2 Completion Criteria

Phase 2 is complete when **all** of the following are true:

- [ ] `normalize_curve`, `trim_curve`, `merge_curves` implemented and tested
- [ ] `cylinder_pq` and `core_shell_sphere_pq` implemented and tested
- [ ] `fit_model_lmfit` implemented; `lmfit` listed as optional dependency
- [ ] `kratky_analysis` and `plot_kratky` implemented and tested
- [ ] `guinier_fit` with `iterative=True` converges on clean synthetic data
- [ ] `pair_distance_distribution` implemented; Rg from p(r) within 5% of
      known value for sphere test case
- [ ] `sphinx-build -W -b html docs/source docs/_build/html` exits 0
- [ ] `notebooks/02_advanced_analysis.ipynb` runs top-to-bottom without errors
- [ ] `notebooks/03_form_factor_fitting.ipynb` runs top-to-bottom without errors
- [ ] `pytest` reports 0 failing tests; ≥ 85% coverage on new modules
- [ ] GitHub Actions CI passes on Python 3.10, 3.11, 3.12

---

## What is Explicitly Out of Scope for Phase 2

These belong to Phase 3 and should not be started now:

- 2D detector image loading and azimuthal integration (`scatterbrain.reduction`)
- Structure factors S(q) and global fitting with polydispersity
- Interactive plots (plotly/bokeh)
- WAXS-specific analysis (peak fitting, d-spacing, crystallite size)
- Desmearing
- GUI
- lmfit-based global fitting across multiple datasets
- GISAXS/GIWAXS reduction
- Time-resolved data analysis
- Machine learning applications
