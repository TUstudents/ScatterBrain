# ScatterBrain -- Phase 2 Plan: Enhanced Core Functionality

**Version:** 1.0
**Branch:** `main`
**Reference:** `Design_document.md` sec.8.3, `PHASE1_PLAN.md`

---

## Goal

Extend the minimally functional Phase 1 library with:

1. Additional standard SAXS analysis (Kratky, scattering invariant)
2. Improved Guinier fitting (weighted regression)
3. Additional form factor models (cylinder, core-shell sphere)
4. Upgraded fitting engine (lmfit replacing scipy.optimize.curve_fit)
5. A normalize processing utility
6. Built-out Sphinx documentation and a second tutorial notebook
7. Stricter static type checking

Phase 2 does **not** tackle p(r)/IFT, structure factors, 2D reduction, or
interactive plots. Those are deferred to Phase 3.

---

## Current State

183 tests passing, 0 failing. Coverage 94% total.

| Module | Coverage | Notes |
|--------|----------|-------|
| `visualization.py` | 97% | Missing: Kratky plot |
| `processing/background.py` | 98% | Missing: normalize |
| `analysis/guinier.py` | 88% | Unweighted linregress; errors ignored |
| `analysis/porod.py` | 96% | Missing: scattering invariant |
| `modeling/form_factors.py` | 100% | sphere only |
| `modeling/fitting.py` | 92% | scipy.optimize.curve_fit; limited error reporting |
| `utils.py` | 81% | |
| `__init__.py` | 71% | |

---

## Tasks

### 2.1 -- Kratky Plot

**Value:** Standard diagnostic plot used to distinguish compact from disordered
(or unfolded) particles. Low implementation effort, immediate user value.

**Specification:**

Add `plot_kratky` to `scatterbrain/visualization.py`.

```
plot_kratky(
    curve: ScatteringCurve1D,
    guinier_result: Optional[GuinierResult] = None,
    normalized: bool = False,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    **plot_kwargs,
) -> Tuple[Figure, Axes]
```

- **Standard mode** (`normalized=False`): plots `I(q) * q^2` vs `q` on linear axes.
  y-axis label: `"I(q) * q^2 [intensity_unit * q_unit^2]"`.
- **Normalized (dimensionless) mode** (`normalized=True`, requires
  `guinier_result` with finite `Rg` and `I0`): plots
  `(q * Rg)^2 * I(q) / I0` vs `q * Rg`.
  y-axis label: `"(q Rg)^2 * I(q) / I(0)"`, x-axis label `"q * Rg"`.
  Raises `AnalysisError` if `normalized=True` but `guinier_result` is None
  or contains non-finite `Rg`/`I0`.
- Follows the same `ax`-injection pattern as existing plot functions.
- Returns `(fig, ax)`.

**Files:** `scatterbrain/visualization.py`

**Exports:** Add `plot_kratky` to `scatterbrain/__init__.py` imports if
visualization functions are re-exported there; otherwise just add to
`visualization.py`.

**Tests:** `tests/test_visualization.py`
- Standard mode: assert y-data equals `I * q^2`.
- Normalized mode with valid GuinierResult: assert y-data equals
  `(q * Rg)^2 * I / I0` and x-data equals `q * Rg`.
- Normalized mode without GuinierResult: assert `AnalysisError` is raised.

---

### 2.2 -- `normalize` Processing Function

**Value:** Dividing intensity by sample thickness, concentration, or
transmission is the most common absolute-scale pre-processing step. Currently
missing; users have no in-library way to do it.

**Specification:**

Add `normalize` to `scatterbrain/processing/background.py`
(or a new `scatterbrain/processing/normalize.py` if the file grows large).

```
normalize(
    curve: ScatteringCurve1D,
    factor: float,
    factor_error: Optional[float] = None,
) -> ScatteringCurve1D
```

- Returns a new curve with `intensity_new = curve.intensity / factor`.
- Error propagation (when `curve.error` is not None and `factor_error` is not None):
  `sigma_new(q) = sqrt( (sigma_I / factor)^2 + (I * factor_error / factor^2)^2 )`
- When `factor_error` is None: `sigma_new = curve.error / factor`.
- Appends to `metadata["processing_history"]`:
  `"Normalized by factor {factor:.4g} +/- {factor_error:.4g}."` (omit error if None).
- Raises `ProcessingError` if `factor <= 0`.
- Raises `ProcessingError` if `curve` is not `ScatteringCurve1D`.

Export `normalize` from `scatterbrain/processing/__init__.py` and the top-level
`scatterbrain` namespace.

**Files:**
- `scatterbrain/processing/background.py` (or new `normalize.py`)
- `scatterbrain/processing/__init__.py`

**Tests:** `tests/test_processing.py`
- Verify `intensity / factor` is correct.
- Verify error propagation with and without `factor_error`.
- Verify `ProcessingError` on `factor <= 0`.
- Verify original curve is not modified.

---

### 2.3 -- Weighted Guinier Regression

**Value:** When `curve.error` is available, the current `guinier_fit` ignores
it. Weighting by `1 / sigma_lnI` (where `sigma_lnI = sigma_I / I`) gives a
statistically correct fit and tighter, more credible Rg/I0 error estimates.
This is the most important scientific improvement in Phase 2.

**Specification:**

Add `use_errors: bool = True` parameter to `guinier_fit`.

When `use_errors=True` and `curve.error` is not None for the selected fit
points, switch from `scipy.stats.linregress` to `numpy.polynomial.polynomial
.Polynomial.fit` with weights, or equivalently use weighted OLS via the normal
equations:

```
w_i = I_i / sigma_I_i          # weight = 1 / sigma_lnI_i
slope, intercept = weighted_least_squares(q^2, ln_I, w)
```

Covariance of (slope, intercept) from the weighted fit:
```
X = [[q_i^2, 1], ...]
W = diag(w_i^2)
C = (X^T W X)^{-1}
stderr_slope = sqrt(C[0,0])
stderr_intercept = sqrt(C[1,1])
```

The manual WLS is straightforward with `numpy.linalg.lstsq` or the closed-form
2x2 inverse. Use the analytical form to avoid a numpy.linalg dependency change.

When `use_errors=False` or errors are unavailable, fall back to the current
`linregress` path (backward compatible).

Add `"weighted": bool` key to `GuinierResult` to indicate which path was used.

**Files:** `scatterbrain/analysis/guinier.py`

**Tests:** `tests/test_analysis.py`
- With noiseless data and no errors: results unchanged from current behavior.
- With data that has errors: verify slope/intercept from weighted fit differ
  from unweighted when errors are heterogeneous.
- With `use_errors=False`: verify unweighted path is taken.
- Verify `"weighted"` key in result.

---

### 2.4 -- Scattering Invariant Q*

**Value:** Q* = integral_0^inf q^2 * I(q) dq is a fundamental quantity in
two-phase scattering systems. It is needed to put Porod constants on an
absolute basis and to calculate volume fractions and specific surface areas.

**Specification:**

Add `scattering_invariant` to `scatterbrain/analysis/` (new file
`scatterbrain/analysis/invariant.py`).

```
class InvariantResult(TypedDict):
    Q_star: float          # integral q^2 I(q) dq over the measured range
    Q_star_low_q: float    # estimated contribution below q_min (Guinier extrapolation)
    Q_star_high_q: float   # estimated contribution above q_max (Porod extrapolation)
    Q_star_total: float    # Q_star + Q_star_low_q + Q_star_high_q
    q_min: float
    q_max: float
    num_points: int
    extrapolation_method: str

scattering_invariant(
    curve: ScatteringCurve1D,
    q_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    guinier_result: Optional[GuinierResult] = None,
    porod_result: Optional[PorodResult] = None,
) -> Optional[InvariantResult]
```

Core integral: `Q_star = numpy.trapz(curve.q**2 * curve.intensity, curve.q)`
over the selected q range.

Extrapolations (only when corresponding result is provided):
- **Low-q Guinier**: `int_0^{q_min} q^2 * I0 * exp(-(q*Rg)^2/3) dq`
  evaluated analytically or via `scipy.integrate.quad`.
- **High-q Porod**: `int_{q_max}^inf q^2 * (Kp * q^{-n}) dq`
  only converges for n > 3. For n = 4:
  `int_{q_max}^inf Kp * q^{-2} dq = Kp / q_max`.
  For general n > 3:
  `int_{q_max}^inf Kp * q^{2-n} dq = Kp * q_max^{3-n} / (n - 3)`.
  Log a WARNING and set `Q_star_high_q = nan` if `porod_exponent <= 3`.

Soft failure: return None with WARNING if fewer than 5 valid points in range.

**Files:**
- `scatterbrain/analysis/invariant.py` (new)
- `scatterbrain/analysis/__init__.py` (export `scattering_invariant`, `InvariantResult`)

**Tests:** `tests/test_analysis.py`
- Verify numerical integral on synthetic q^2*I(q) data matches `numpy.trapz`.
- Verify Porod high-q extrapolation formula for n=4.
- Verify low-q Guinier extrapolation adds a positive contribution.
- Verify soft failure on empty q range.
- Verify `Q_star_high_q = nan` warning when Porod exponent <= 3.

---

### 2.5 -- Additional Form Factors

**Value:** Cylinder and core-shell sphere are the two most frequently needed
models after the sphere.

#### 2.5a -- `cylinder_pq`

Form factor for a monodisperse cylinder of radius R and half-length L/2
(or full length L), randomly oriented in solution:

```
P(q, R, L) = integral_0^{pi/2}
    [ 2 * J1(q*R*sin(alpha)) / (q*R*sin(alpha)) ]^2
    * [ sin(q*L/2*cos(alpha)) / (q*L/2*cos(alpha)) ]^2
    * sin(alpha) d_alpha
```

where J1 is `scipy.special.j1` (Bessel function of the first kind, order 1).

Implementation: evaluate the integrand on a fixed Gauss-Legendre quadrature
grid over alpha in [0, pi/2] (e.g., 64 points via `numpy.polynomial.legendre
.leggauss`) and sum. This avoids a per-q `scipy.integrate.quad` call and is
fully vectorized over q.

Normalize so that P(q=0) = 1 (handle the 0/0 limits via a small-angle Taylor
expansion or the `_Q_EPSILON` guard used in `sphere_pq`).

```
cylinder_pq(q: np.ndarray, radius: float, length: float) -> np.ndarray
```

Raises `ValueError` if `radius <= 0` or `length <= 0`.

#### 2.5b -- `core_shell_sphere_pq`

Form factor for a spherically symmetric core-shell particle:

```
F(q) = (4*pi/3) * [
    R_core^3 * (contrast_core - contrast_shell) * f(q, R_core)
    + R_outer^3 * contrast_shell * f(q, R_outer)
]
where f(q, R) = 3*(sin(qR) - qR*cos(qR))/(qR)^3
and R_outer = radius_core + shell_thickness

P(q) = F(q)^2 / F(0)^2    (normalized to P(0) = 1)
```

```
core_shell_sphere_pq(
    q: np.ndarray,
    radius_core: float,
    shell_thickness: float,
    contrast_core: float = 1.0,
    contrast_shell: float = 0.5,
) -> np.ndarray
```

Raises `ValueError` if `radius_core <= 0` or `shell_thickness < 0`.
Returns `np.ones_like(q)` (P(q)=1 for all q) when `contrast_core == contrast_shell`
(degenerate case; log a DEBUG message).

**Files:** `scatterbrain/modeling/form_factors.py`

**Tests:** `tests/test_modeling.py`
- `cylinder_pq`: verify P(0)=1 limit, verify P(q) is real and non-negative,
  check against a known tabulated value (e.g., SasView or analytical limit).
- `core_shell_sphere_pq`: verify P(0)=1, verify limiting case
  `shell_thickness=0` recovers `sphere_pq`, test contrast inversion.
- Both: `ValueError` on invalid geometry parameters.

---

### 2.6 -- lmfit Integration

**Value:** `scipy.optimize.curve_fit` returns only a covariance matrix.
`lmfit` provides confidence intervals, correlation matrices, multiple
minimization methods, and cleaner parameter constraint syntax. It is a mature,
widely used library in the scientific Python ecosystem.

**Specification:**

Add `lmfit>=1.2` to `dependencies` in `pyproject.toml` (required, not optional;
lmfit is pure Python, ~150 kB, no heavy transitive dependencies).

Modify `fit_model` in `scatterbrain/modeling/fitting.py` to use `lmfit`
internally while keeping the existing public API fully backward compatible.
The return dict gains two new optional keys:

```python
"confidence_intervals": Optional[Dict[str, Tuple[float, float]]]
    # 1-sigma (68%) CI for each fitted parameter; None if CI calculation fails
"lmfit_result": lmfit.MinimizerResult
    # raw lmfit result object for advanced users
```

Migration strategy:
1. Replace the `curve_fit` call with `lmfit.minimize` (using the `leastsq`
   method by default for backward-compatible behavior).
2. Build a `lmfit.Parameters` object from `initial_params`, `param_bounds`,
   and `fixed_params`.
3. Extract `popt` and `pcov` from the lmfit result to preserve the existing
   return structure (no breaking change).
4. Attempt `lmfit.conf_interval` for the new `confidence_intervals` key;
   catch `MinimizerException` and set to None on failure.

If lmfit is unavailable at runtime (import fails), fall back to the current
`curve_fit` implementation and log a WARNING. Add a module-level
`_LMFIT_AVAILABLE` flag.

**Files:**
- `scatterbrain/modeling/fitting.py`
- `pyproject.toml` (add `lmfit>=1.2` to `dependencies`)

**Tests:** `tests/test_modeling.py`
- Verify all existing tests still pass (backward compatibility).
- Verify `"lmfit_result"` key is present and is a `lmfit.MinimizerResult`.
- Verify `"confidence_intervals"` key is present (may be None on degenerate fit).
- Verify fitted parameter values are consistent with the scipy path.

---

### 2.7 -- Sphinx Documentation Build-out and Notebook 02

**Value:** The `docs/` directory currently has a minimal `conf.py` and
`index.rst`. The sphinx-build step in CI builds successfully but produces an
essentially empty site. Users need a navigable API reference and at least one
additional tutorial.

**Specification:**

#### Documentation

Populate `docs/source/` with:

- `index.rst`: top-level toctree linking installation, quickstart, tutorials,
  api, changelog.
- `installation.rst`: uv-based dev install, PyPI install (when published).
- `quickstart.rst`: the same 10-line code block from README with prose.
- `tutorials/index.rst` + `tutorials/01_basic_workflow.rst` (thin wrapper
  pointing to notebook).
- `tutorials/02_form_factor_fitting.rst` (thin wrapper for notebook 02).
- `api/index.rst`: autodoc stubs for all public modules:
  `scatterbrain.core`, `scatterbrain.io`, `scatterbrain.processing`,
  `scatterbrain.analysis`, `scatterbrain.modeling`, `scatterbrain.visualization`,
  `scatterbrain.utils`.
- `changelog.rst`: version 0.0.1 (Phase 1) and 0.1.0 (Phase 2) entries.

Verify `sphinx-build -W -b html docs/source docs/_build/html` passes with
no warnings (the CI docs job already uses `-W`).

#### Notebook 02

`notebooks/02_form_factor_fitting.ipynb` demonstrating:
1. Load the example sphere data.
2. Fit `sphere_pq` with `fit_model` -- show fitted radius, chi-squared.
3. Fit `cylinder_pq` (misspecified model) -- show worse chi-squared.
4. Show `plot_fit` output with residuals panel.
5. Print fitted parameters and confidence intervals from lmfit.

Add to CI:
```yaml
- name: Run notebook 02
  env:
    MPLBACKEND: Agg
  run: uv run pytest --nbmake notebooks/02_form_factor_fitting.ipynb
```

**Files:**
- `docs/source/*.rst`
- `notebooks/02_form_factor_fitting.ipynb`
- `.github/workflows/ci.yml`

**Tests:** The Sphinx CI build job is the acceptance test for docs. Notebook 02
is its own acceptance test.

---

### 2.8 -- Stricter mypy

**Value:** Phase 1 runs mypy with `continue-on-error: true` and
`ignore_missing_imports = true`. This gives no enforcement. Phase 2 should
tighten this so that type errors in our own code become blocking.

**Specification:**

In `pyproject.toml [tool.mypy]`:
- Enable `disallow_untyped_defs = true`.
- Enable `check_untyped_defs = true`.
- Keep `ignore_missing_imports = true` (external libs may lack stubs).

In `.github/workflows/ci.yml`:
- Remove `continue-on-error: true` from the mypy step.

Annotate any public functions currently missing return type or parameter
annotations to make mypy pass. The new lmfit types can use `Any` for the
`lmfit_result` value until `lmfit` ships complete stubs.

**Files:**
- `pyproject.toml`
- `.github/workflows/ci.yml`
- Any source files with annotation gaps (audit with `uv run mypy scatterbrain`
  after enabling `disallow_untyped_defs`)

**Tests:** The CI mypy step is the acceptance test. All existing tests must
still pass after annotation additions.

---

## Priority Order

```
2.1  Kratky plot              -- easy, immediate user value
2.2  normalize                -- easy, fills an obvious processing gap
2.3  Weighted Guinier         -- moderate; most important scientific improvement
2.4  Scattering invariant Q*  -- moderate; unlocks absolute Porod analysis
2.5  Form factors             -- moderate-hard; cylinder then core-shell sphere
2.6  lmfit integration        -- moderate; improves all model fitting
2.7  Sphinx docs + notebook   -- moderate; needed before any public release
2.8  Stricter mypy            -- easy-moderate; do alongside annotation work
```

Tasks 2.1 and 2.2 are independent of everything else and can be done in
parallel. Task 2.7 (docs) is also largely independent; it should be started
after 2.5 and 2.6 are done so the API reference is complete.

---

## Phase 2 Completion Criteria

| Criterion | Status |
|-----------|--------|
| `plot_kratky` implemented, no `NotImplementedError` | Pending: Task 2.1 |
| `normalize` implemented in `scatterbrain.processing` | Pending: Task 2.2 |
| `guinier_fit` uses intensity errors as weights when available | Pending: Task 2.3 |
| `scattering_invariant` implemented and exported | Pending: Task 2.4 |
| `cylinder_pq` implemented with P(0)=1 | Pending: Task 2.5 |
| `core_shell_sphere_pq` implemented; shell=0 recovers sphere | Pending: Task 2.5 |
| `fit_model` uses lmfit; existing test suite passes unchanged | Pending: Task 2.6 |
| `fit_model` result includes `"confidence_intervals"` key | Pending: Task 2.6 |
| Sphinx docs build with no warnings; API reference populated | Pending: Task 2.7 |
| `notebooks/02_form_factor_fitting.ipynb` runs in CI | Pending: Task 2.7 |
| `mypy --disallow_untyped_defs` passes; CI mypy step is blocking | Pending: Task 2.8 |
| pytest >= 220 passing, 0 failing | Pending |
| Test coverage >= 90% (all modules) | Pending |

---

## What is Explicitly Out of Scope for Phase 2

These belong to Phase 3:

- Pair Distance Distribution Function p(r) via IFT (regularization)
- Structure factors S(q) and polydispersity
- Curve merging / rebinning to a common q-grid
- 2D detector image loading and azimuthal integration
- Interactive plots (plotly / bokeh)
- WAXS analysis (peak fitting, Scherrer equation, crystallinity)
- GISAXS / GIWAXS
- Time-resolved data support
- GUI
