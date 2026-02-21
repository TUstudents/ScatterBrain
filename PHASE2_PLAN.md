# ScatterBrain — Phase 2 Plan: Enhancing Core Functionality & Usability

**Version:** 1.0
**Branch:** `claude/plan-phase-2-qbWH7`
**Reference:** `Design_document.md` §8.3, `PHASE1_PLAN.md`

---

## Goal

Build on the completed Phase 1 MVP by deepening the analysis capabilities,
expanding the model library, introducing more robust fitting, and improving
documentation. At the end of Phase 2 the library should be useful for
real-world SAXS analysis workflows beyond a basic proof-of-concept.

---

## Current State Assessment

Phase 1 is complete: **183 passing tests, 0 failing**.

| Component | Phase 1 Status | Phase 2 Direction |
|-----------|---------------|-------------------|
| `analysis/guinier.py` | ✅ OLS only, heuristic auto-range | Weighted regression + iterative refinement |
| `analysis/porod.py` | ✅ Log-log fit + average Kp | Stable; no changes needed |
| `analysis/` — Kratky | ❌ Missing | New module |
| `analysis/` — IFT / p(r) | ❌ Missing | New module (complex) |
| `modeling/form_factors.py` | ✅ Sphere only | Add cylinder, core-shell sphere |
| `modeling/structure_factors.py` | ❌ Placeholder | Add hard-sphere S(q) |
| `modeling/fitting.py` | ✅ scipy.optimize.curve_fit | Add optional lmfit backend |
| `processing/background.py` | ✅ Complete | Stable |
| `processing/` — normalization | ❌ Missing | New module |
| `visualization.py` | ✅ plot_iq, Guinier, Porod, fit | Add plot_kratky, plot_pr |
| `notebooks/` | ✅ 01_basic_workflow | 3+ new notebooks |
| `docs/` | ✅ Scaffolded | Build out user guide content |

---

## Phase 2 Task List

### 2.1 — Guinier Analysis: Weighted Regression & Iterative q-range Refinement

**Design doc reference:** `Design_document.md` §8.3 ("Refine Guinier/Porod analysis")
**Phase 1 explicit out-of-scope:** "Guinier auto-range refinement (weighted regression,
iterative Rg limit)"

**Problem with current implementation:**
- Uses unweighted `scipy.stats.linregress`, ignoring `curve.error`.
- Auto-range is a single-pass heuristic: it estimates Rg from the lowest 10% of q-points,
  then sets q_max = qrg_limit_max / Rg_initial. This Rg_initial may differ from the
  Rg obtained in the final fit, so the limit is not self-consistent.

**Specification — additions to `guinier_fit`:**

```python
def guinier_fit(
    curve: ScatteringCurve1D,
    q_range=None,
    qrg_limit_max=1.3,
    qrg_limit_min=None,
    auto_q_selection_fraction=0.1,
    min_points=5,
    method: str = "wls",          # NEW: 'ols' or 'wls'
    max_iterations: int = 10,     # NEW: for iterative q-range refinement
    convergence_tol: float = 1e-3, # NEW: relative Rg change threshold
) -> Optional[Dict[str, Any]]
```

**`method='wls'` behaviour (when `curve.error` is available):**
- Transform propagated errors to ln-space: `σ_lnI = σ_I / I`.
- Perform weighted least squares on `ln(I)` vs `q²` using weights `w = 1 / σ_lnI²`.
- Use `numpy.polyfit(q², ln_I, deg=1, w=weights, cov=True)` to obtain slope, intercept,
  and their covariance.
- Fall back to OLS silently if no errors are available.

**Iterative q-range refinement (when `q_range` is None):**
1. Estimate Rg_0 from the initial heuristic (current behaviour).
2. Set q_max_1 = qrg_limit_max / Rg_0.
3. Fit in [q_min, q_max_1] → obtain Rg_1.
4. Set q_max_2 = qrg_limit_max / Rg_1.
5. Repeat until |Rg_n - Rg_{n-1}| / Rg_{n-1} < convergence_tol or max_iterations.
6. Report `num_iterations` in the result dict.

**New result dict keys:**
- `'method'`: `'wls'` or `'ols'`
- `'num_iterations'`: number of auto-range iterations performed (1 if q_range was manual)
- `'weights_used'`: bool — whether error-based weights were applied
- `'rg_qrg_max'`: qRg at q_fit_max (goodness-of-fit criterion; should be ≤ qrg_limit_max)

**Files:** `scatterbrain/analysis/guinier.py`
**Tests:** `tests/test_analysis.py` — verify WLS gives different (tighter) errors than OLS
on synthetic noisy data; verify iterative method converges to self-consistent Rg*qrg_limit.

---

### 2.2 — Kratky Analysis

**Design doc reference:** `Design_document.md` §2.2 (long-term), §5.4 future SAXS

**Motivation:** The Kratky plot (`q² · I(q)` vs `q`) is a routine diagnostic.
A bell-shaped peak indicates a compact (globular) particle; a plateau indicates
a polymer in solution; a monotonically rising signal suggests aggregation or
an unfolded/disordered system.

**Specification:**

```python
# scatterbrain/analysis/kratky.py

def kratky_analysis(
    curve: ScatteringCurve1D,
    i0: Optional[float] = None,
    rg: Optional[float] = None,
    q_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]
```

**Behaviour:**
- Compute the Kratky representation: `kratky_y = q² * I(q)`.
- If errors are present, propagate: `σ_kratky = q² * σ_I`.
- If `i0` and `rg` are provided (e.g. from `guinier_fit`), compute the
  dimensionless Kratky representation: `x = q * Rg`, `y = (q * Rg)² * I(q) / I(0)`.
  This normalised form allows comparison between samples.
- Detect the Kratky peak position (`q_peak`) and value (`kratky_peak_value`) if a
  local maximum exists in the q-range.
- Return:
  - `'q'`, `'kratky_y'`, `'kratky_y_err'` (may be None)
  - `'dimensionless'`: bool — True if i0/rg provided
  - `'q_rg'`, `'dimensionless_y'` — only when `dimensionless=True`
  - `'q_peak'`, `'kratky_peak_value'` — None if no peak detected

```python
# scatterbrain/visualization.py

def plot_kratky(
    curve: ScatteringCurve1D,
    kratky_result: Optional[Dict[str, Any]] = None,
    dimensionless: bool = False,
    ax: Optional[Axes] = None,
    title: Optional[str] = "Kratky Plot",
    **plot_kwargs,
) -> Tuple[Figure, Axes]
```

**Files:**
- Create `scatterbrain/analysis/kratky.py`
- Update `scatterbrain/analysis/__init__.py`
- `scatterbrain/visualization.py`

**Tests:** `tests/test_analysis_kratky.py` — verify q²I values; verify peak detection
on synthetic sphere data; verify dimensionless Kratky with known Rg and I0.

---

### 2.3 — Pair Distance Distribution Function p(r) via Indirect Fourier Transform (IFT)

**Design doc reference:** `Design_document.md` §8.3 ("Implement p(r) calculation (IFT)"),
§2.2 long-term scope, §5.4 ("Pair Distance Distribution Function")

**Background:**
`p(r)` is the Fourier transform of `I(q)`: `p(r) = (r²/2π²) ∫ I(q) q² sinc(qr) dq`.
In practice this is solved as a regularized ill-posed inverse problem.
The standard method (Glatter, 1977; Semenyuk & Svergun, 1991) is the
Indirect Fourier Transform with regularization (BIFT).

**Specification (simplified regularized IFT):**

```python
# scatterbrain/analysis/ift.py

def indirect_fourier_transform(
    curve: ScatteringCurve1D,
    d_max: float,
    n_terms: int = 20,
    alpha: Optional[float] = None,
    q_range: Optional[Tuple[float, float]] = None,
    n_r_points: int = 100,
) -> Dict[str, Any]
```

**Method — Moore IFT approach (appropriate for Phase 2 complexity):**
1. Expand `p(r)` in sine-series basis functions on `[0, d_max]`:
   `p(r) = Σ_k c_k * sin(k π r / d_max)` for k = 1 … n_terms.
2. The corresponding `I(q)` is analytic (sum of sinc integrals).
3. Fit the coefficients `c_k` using regularized least squares:
   `minimize ||A·c - I_data||² + α · ||L·c||²`
   where `A[i,k]` is the transform matrix and `L` is a smoothing matrix (finite differences).
4. If `alpha=None`, determine the regularisation parameter by generalized
   cross-validation (GCV) or L-curve criterion using a grid search over α.
5. Evaluate `p(r)` on a grid of `n_r_points` points in `[0, d_max]`.
6. Compute `Rg` and `I(0)` from moments of `p(r)`:
   `I(0) = 4π ∫ p(r) dr`
   `Rg² = ∫ r² p(r) dr / (2 ∫ p(r) dr)`

**Returns:**
- `'r'`: r-values at which p(r) is evaluated
- `'pr'`: p(r) values
- `'pr_err'`: estimated error on p(r) (propagated from fit uncertainties)
- `'Rg_pr'`: Rg calculated from p(r)
- `'I0_pr'`: I(0) calculated from p(r)
- `'d_max'`: maximum dimension used
- `'alpha'`: regularization parameter used
- `'chi_squared_reduced'`: goodness of fit to I(q) data
- `'n_terms'`: number of basis functions used

```python
# scatterbrain/visualization.py

def plot_pr(
    pr_result: Dict[str, Any],
    ax: Optional[Axes] = None,
    title: Optional[str] = "Pair Distance Distribution p(r)",
    **plot_kwargs,
) -> Tuple[Figure, Axes]
```

**Files:**
- Create `scatterbrain/analysis/ift.py`
- Update `scatterbrain/analysis/__init__.py`
- `scatterbrain/visualization.py`

**Tests:** `tests/test_analysis_ift.py`
- Verify `p(r)` is zero at r=0 and r=d_max (boundary conditions).
- Verify Rg from p(r) matches Guinier Rg within ~5% on noiseless sphere data.
- Verify I(0) from p(r) matches I(0) from Guinier on noiseless data.
- Verify `p(r)` is non-negative for sphere data.

**Note:** This is the most complex task in Phase 2. If the regularised IFT proves
too brittle in testing, a simpler trapezoid-rule direct integration on a
fine q-grid (limited to smooth, well-sampled data) is an acceptable fallback.

---

### 2.4 — Additional Form Factors: Cylinder & Core-Shell Sphere

**Design doc reference:** `Design_document.md` §8.3 ("Expand form factor library"),
§5.5 ("cylinder_pq, core_shell_sphere_pq")

#### 2.4a — `cylinder_pq`

```python
def cylinder_pq(
    q: np.ndarray,
    radius: float,
    length: float,
    n_integration_points: int = 76,
) -> np.ndarray
```

**Formula (orientationally averaged):**
```
P(q) = ∫₀^{π/2} P(q, α) sin(α) dα
P(q, α) = [2 J₁(qR sinα) / (qR sinα)]² · [sin(qL cosα / 2) / (qL cosα / 2)]²
```
where α is the angle between the cylinder axis and q, J₁ is the Bessel function
of the first kind order 1 (use `scipy.special.j1`), and L is the cylinder length.

**Numerical integration:** Gauss-Legendre quadrature via `numpy.polynomial.legendre.leggauss`
over [0, π/2], using 76 points (SasView convention). Handle `α = 0` and `α = π/2`
limits analytically to avoid 0/0.

**Normalisation:** P(q=0) = 1.

```python
def cylinder_pq(q, radius, length, n_integration_points=76) -> np.ndarray
```

#### 2.4b — `core_shell_sphere_pq`

```python
def core_shell_sphere_pq(
    q: np.ndarray,
    core_radius: float,
    shell_thickness: float,
    delta_rho_core: float = 1.0,
    delta_rho_shell: float = 0.0,
) -> np.ndarray
```

**Formula:**
```
F(q) = (4π/3) [Δρ_core · R_core³ · f(qR_core)
              + Δρ_shell · R_total³ · f(qR_total)
              - Δρ_shell · R_core³ · f(qR_core)]
f(x) = 3(sin x - x cos x) / x³
P(q) = F²(q) / F²(0)
```
where R_total = core_radius + shell_thickness.

**Normalisation:** P(q=0) = 1. Handle q→0 limit analytically.

**Files:** `scatterbrain/modeling/form_factors.py`
**Tests:** `tests/test_modeling.py`
- `cylinder_pq(q, R, L)` at q=0 returns 1.0 within numerical tolerance.
- Sphere limit: `cylinder_pq(q, R, R)` ≈ sphere P(q) for large q? (Not exact; this
  is just a sanity check on the amplitude fall-off scale.)
- `core_shell_sphere_pq` at q=0 returns 1.0.
- When `delta_rho_shell=0`, `core_shell_sphere_pq` reduces to a solid sphere scaled by
  its volume (verify shape, not absolute scale, matches `sphere_pq` of core radius).
- Validate cylinder against published tabulated values from SasView or literature.

---

### 2.5 — lmfit Integration for Robust Model Fitting

**Design doc reference:** `Design_document.md` §8.3 ("Integrate lmfit"),
§5.5 ("Strongly recommend lmfit")

**Motivation:** `scipy.optimize.curve_fit` has limitations: bounded parameters
can have poor convergence, there is no support for parameter constraints or
expressions, and error reporting is less transparent. `lmfit` wraps multiple
minimizers with a consistent parameter interface.

**New dependency:** add `lmfit>=1.2` to `pyproject.toml` `[project].dependencies`.

**Specification:**

```python
# scatterbrain/modeling/fitting.py  (new function, existing fit_model kept intact)

def fit_model_lmfit(
    curve: ScatteringCurve1D,
    model_func: Callable[..., np.ndarray],
    params: "lmfit.Parameters",
    q_range: Optional[Tuple[float, float]] = None,
    method: str = "leastsq",
    **minimizer_kwargs,
) -> Optional[Dict[str, Any]]
```

**Behaviour:**
- `params` is an `lmfit.Parameters` object. The caller constructs it directly,
  giving full control over bounds, expressions, and fixed flags.
- The scale and background parameters (`scale`, `background`) are expected in `params`
  unless the caller wishes to omit them.
- The residual function is `(I_data - I_model) / σ` if errors available, else
  `(I_data - I_model)`.
- Returns a dict with the same keys as `fit_model` for compatibility, plus:
  - `'lmfit_result'`: the raw `lmfit.MinimizerResult` object for advanced users.
  - `'fit_report'`: `lmfit.fit_report(result)` string for printing.

**Helper factory (convenience):**

```python
def make_lmfit_params(
    param_names: List[str],
    initial_values: Dict[str, float],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    fixed: Optional[List[str]] = None,
    expressions: Optional[Dict[str, str]] = None,
) -> "lmfit.Parameters"
```

Wraps `lmfit.Parameters` creation so users unfamiliar with lmfit can quickly
build a Parameters object from plain dicts.

**Files:**
- `scatterbrain/modeling/fitting.py`
- `pyproject.toml` (add `lmfit` dependency)

**Tests:** `tests/test_modeling.py`
- Fit sphere P(q) using `fit_model_lmfit` and verify radius recovered within 1%.
- Verify `make_lmfit_params` creates params with correct bounds.
- Verify fixed parameter is not adjusted during fit.
- Verify `fit_report` is a non-empty string.

---

### 2.6 — Structure Factor: Hard Sphere S(q)

**Design doc reference:** `Design_document.md` §5.5 (`structure_factors` sub-module)

**Specification:**

```python
# scatterbrain/modeling/structure_factors.py

def hard_sphere_sq(
    q: np.ndarray,
    volume_fraction: float,
    radius: float,
) -> np.ndarray
```

**Formula:** Percus-Yevick analytic solution (Ashcroft & Lekner, 1966).
The structure factor is:
```
S(q) = 1 / (1 - n · C(q))
```
where `n = 3φ / (4π R³)` is the number density, `φ` is the volume fraction,
and `C(q)` is the Fourier transform of the direct correlation function for
hard spheres (closed-form expression in terms of sin, cos).

The explicit expression for `C(q)` in terms of `x = 2qR`:
```
C(x) = -24φ [ α(sin x - x cos x)/x³
             + β(2x sin x - (x²-2) cos x - 2)/x⁴
             + γ(-x⁴ cos x + 4[(3x²-6)cos x + (x³-6x)sin x + 6])/x⁶ ]
```
where α = (1+2φ)²/(1-φ)⁴, β = -6φ(1+φ/2)²/(1-φ)⁴, γ = φα/2.

Handle `q → 0` analytically. Raise `ValueError` if `volume_fraction ≥ 0.64`
(beyond random-close-packing).

**Integration with `fit_model` / `fit_model_lmfit`:**
- Document (in docstring) the combined model:
  `I(q) = scale * P(q) * S(q) + background`
- Provide a worked example in a tutorial notebook.

**Files:** `scatterbrain/modeling/structure_factors.py`
**Tests:** `tests/test_modeling_structure_factors.py` (new file)
- `S(q → 0) → (1 - φ)⁴ / (1 + 2φ)²` (known analytic limit)
- `S(q)` at `φ = 0` equals 1 for all q.
- `ValueError` raised for `volume_fraction ≥ 0.64`.
- Validate against published tabulated values (SasView or Ashcroft & Lekner paper).

---

### 2.7 — Processing: Normalization Functions

**Design doc reference:** `Design_document.md` §8.3 ("Add more processing functions"),
§5.3 ("normalize_by_thickness")

**Specification:**

```python
# scatterbrain/processing/normalization.py

def normalize_by_concentration(
    curve: ScatteringCurve1D,
    concentration: float,
    concentration_unit: str = "mg/mL",
) -> ScatteringCurve1D

def normalize_by_thickness(
    curve: ScatteringCurve1D,
    thickness: float,
    thickness_unit: str = "mm",
) -> ScatteringCurve1D

def scale_to_absolute(
    curve: ScatteringCurve1D,
    reference_curve: ScatteringCurve1D,
    reference_value: float,
    q_range: Optional[Tuple[float, float]] = None,
) -> ScatteringCurve1D
```

**Behaviour (all functions):**
- Divide intensity (and error if present) by the normalization factor.
- Return a **new** `ScatteringCurve1D`; do not modify inputs.
- Record the operation in `metadata["processing_history"]`.
- Update `intensity_unit` string where possible (e.g., `"a.u."` → `"a.u. / (mg/mL)"`).
- Raise `ValueError` for non-positive normalization factors.

`scale_to_absolute` determines a scale factor by computing the ratio
`reference_value / mean(curve.intensity)` in the given q_range, then multiplies
the curve by this factor. Useful for normalization to water or glassy-carbon standards.

**Files:**
- Create `scatterbrain/processing/normalization.py`
- Update `scatterbrain/processing/__init__.py`

**Tests:** `tests/test_processing_normalization.py` (new file)
- Divide by concentration and verify values are halved for concentration=2.
- Divide by thickness and verify error propagation.
- Verify `ValueError` for zero or negative normalization factor.
- Round-trip: normalize then manually scale back; recover original.

---

### 2.8 — Tutorial Notebooks

**Design doc reference:** `Design_document.md` §8.3 ("Develop Jupyter Notebook tutorials"),
§11 (Documentation Strategy)

**Required notebooks (in `notebooks/`):**

| File | Topic | Depends on |
|------|-------|-----------|
| `02_kratky_analysis.ipynb` | Kratky plot for globular vs disordered particles | Task 2.2 |
| `03_pair_distance_distribution.ipynb` | IFT p(r) analysis workflow | Task 2.3 |
| `04_advanced_modeling.ipynb` | Cylinder/core-shell models, S(q), lmfit | Tasks 2.4–2.6 |

**Each notebook should:**
1. Start with a brief conceptual introduction (markdown cells).
2. Use simulated data generated within the notebook (no external file dependency).
3. Demonstrate the full workflow: load/generate → process → analyse → visualise.
4. Display all key results with interpretive comments.
5. Run top-to-bottom without errors.

---

### 2.9 — Sphinx Documentation Build-Out

**Design doc reference:** `Design_document.md` §8.3 ("Build out Sphinx documentation"),
§11 (Documentation Strategy)

**Tasks:**
- Add a `docs/source/user_guide/` directory with RST pages:
  - `guinier.rst` — theory, API usage example, interpretation guide.
  - `porod.rst` — theory and API usage.
  - `kratky.rst` — theory and API usage.
  - `form_factors.rst` — sphere, cylinder, core-shell; when to use each.
  - `ift.rst` — theory, regularization, practical notes.
  - `structure_factors.rst` — hard sphere PY; when concentrated suspensions matter.
- Update `docs/source/index.rst` to include the user guide section.
- Verify `sphinx-build docs/source docs/_build` completes without errors or warnings
  on the new modules.
- Ensure autodoc correctly picks up all public symbols in new modules
  (add them to `docs/source/api.rst` or equivalent).

**Files:** `docs/source/` (multiple new RST files)

---

### 2.10 — Tests for All New Components

Summary of new/expanded test files:

| File | New Tests |
|------|-----------|
| `tests/test_analysis.py` | WLS Guinier; iterative convergence; `num_iterations` key |
| `tests/test_analysis_kratky.py` | `kratky_analysis` (new file) |
| `tests/test_analysis_ift.py` | `indirect_fourier_transform` (new file) |
| `tests/test_modeling.py` | `cylinder_pq`, `core_shell_sphere_pq`, `fit_model_lmfit` |
| `tests/test_modeling_structure_factors.py` | `hard_sphere_sq` (new file) |
| `tests/test_processing_normalization.py` | All normalization functions (new file) |
| `tests/test_visualization.py` | `plot_kratky`, `plot_pr` |

**Coverage target:** ≥ 80% branch coverage on all new modules.
**Minimum passing count:** ≥ 250 tests total, 0 failing.

---

## Priority Order

Implement in this order to keep dependencies satisfied and unblock notebooks:

```
2.4   cylinder_pq, core_shell_sphere_pq      ← no new deps; extends existing module
2.1   Guinier weighted regression + iterative  ← refines existing; no new deps
2.2   Kratky analysis                         ← no new deps; simple new module
2.7   Normalization functions                 ← no new deps; simple new module
2.6   Hard sphere S(q)                        ← no new deps; completes modeling sub-package
2.5   lmfit integration                       ← new dep; adds lmfit to pyproject.toml
2.3   IFT p(r)                                ← complex; scipy only
2.10  Tests                                   ← written alongside each task above
2.8   Tutorial notebooks                      ← requires 2.1–2.7
2.9   Sphinx docs build-out                   ← can proceed in parallel with code tasks
```

---

## Phase 2 Completion Criteria

Phase 2 is complete when **all** of the following are true:

- [ ] `guinier_fit` supports `method='wls'` and iterative q-range refinement; result
      dict includes `'method'`, `'num_iterations'`, `'weights_used'`, `'rg_qrg_max'`
- [ ] `cylinder_pq` is implemented, normalised to P(0)=1, and validated against
      published reference values
- [ ] `core_shell_sphere_pq` is implemented, normalised to P(0)=1, and passes
      sphere-limit test
- [ ] `kratky_analysis` is implemented and returns correct `q², I·q²` data
- [ ] `plot_kratky` is implemented (no `NotImplementedError`)
- [ ] `indirect_fourier_transform` is implemented; Rg from p(r) matches Guinier Rg
      within 5% on noiseless sphere data; `p(r)` has correct boundary conditions
- [ ] `plot_pr` is implemented
- [ ] `hard_sphere_sq` is implemented; analytic limits are verified
- [ ] `fit_model_lmfit` is implemented; `make_lmfit_params` helper is available
- [ ] `normalize_by_concentration`, `normalize_by_thickness`, `scale_to_absolute`
      are implemented and update `metadata["processing_history"]`
- [ ] `pytest` reports ≥ 250 passing tests, 0 failing
- [ ] Test coverage ≥ 80% on all new modules
- [ ] `notebooks/02_kratky_analysis.ipynb`, `03_pair_distance_distribution.ipynb`,
      `04_advanced_modeling.ipynb` each run top-to-bottom without errors
- [ ] `sphinx-build docs/source docs/_build` completes without errors
- [ ] All public functions in new modules have NumPy-style docstrings

---

## What is Explicitly Out of Scope for Phase 2

These belong to Phase 3 and should not be started now:

- 2D detector image loading and azimuthal integration (`scatterbrain.reduction`)
- Advanced curve merging and desmearing
- WAXS-specific analysis (peak fitting, Scherrer equation, crystallinity)
- Polydispersity in form factor calculations
- Global fitting across multiple datasets
- Interactive (plotly/bokeh) visualizations
- GUI
- GISAXS/GIWAXS support
- Time-resolved data analysis
- Machine learning applications
