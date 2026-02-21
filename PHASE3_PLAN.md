# ScatterBrain -- Phase 3 Plan: Advanced Analysis, Modeling, and 2D Pipeline

**Version:** 1.0
**Branch:** `main`
**Reference:** `Design_document.md` sec.2.2, `PHASE2_PLAN.md`

---

## Goal

Phase 3 elevates ScatterBrain from a basic 1D toolkit to a comprehensive
SAXS/WAXS analysis library by adding:

1. Pair distance distribution function p(r) via Indirect Fourier Transform
2. Structure factors S(q) and polydispersity for model fitting
3. Curve merging and rebinning for multi-detector data
4. WAXS analysis: peak fitting, d-spacing, Scherrer equation
5. 2D detector data support: SAXSImage class, image loading, azimuthal integration

Phase 3 assumes Phase 2 is complete (lmfit in `fit_model`, weighted Guinier,
cylinder and core-shell sphere form factors, scattering invariant).

---

## Task Overview

| Task | Component | Complexity | Dependencies |
|------|-----------|------------|--------------|
| 3.1 | p(r) via IFT (Moore regularization) | High | numpy, scipy (already deps) |
| 3.2 | Hard-sphere structure factor S(q) | Moderate | numpy |
| 3.3 | Polydispersity for form factors | Moderate | numpy |
| 3.4 | Curve merging and rebinning | Moderate | numpy, scipy.interpolate |
| 3.5 | WAXS peak analysis | Moderate | scipy.signal, scipy.optimize |
| 3.6 | SAXSImage class and image loading | High | fabio (new optional dep) |
| 3.7 | Azimuthal integration | High | pyFAI (new optional dep) |

Tasks 3.1 through 3.5 require only existing dependencies (numpy, scipy, lmfit).
Tasks 3.6 and 3.7 introduce optional heavy dependencies (fabio, pyFAI) and
constitute the 2D data pipeline. They can be developed independently of 3.1-3.5.

---

## Tasks

### 3.1 -- Pair Distance Distribution Function p(r) via IFT

**Scientific context:** The pair distance distribution function p(r) is the
Fourier transform of I(q) and contains direct information about the maximum
particle dimension D_max, particle shape, and internal structure. It is
arguably the most important advanced SAXS analysis tool after Guinier/Porod.

**Method:** Moore (1980) indirect Fourier transform with Tikhonov regularization.

p(r) is expanded in a sine basis:

```
p(r) = sum_{s=1}^{N} c_s * sin(s * pi * r / D_max),   0 <= r <= D_max
```

The scattering intensity is then:

```
I(q) = 4 * pi * integral_0^{D_max} p(r) * sin(q*r) / (q*r) dr
     = sum_s c_s * T_s(q)

T_s(q) = 4*pi * integral_0^{D_max} sin(s*pi*r/D_max) * sin(q*r)/(q*r) dr
```

T_s(q) has an analytical form:

```
T_s(q) = 4 * pi * D_max * s * pi / (s^2*pi^2 - (q*D_max)^2)
         * [ (-1)^s * sin(q*D_max) / (q*D_max) ]
         (valid when q*D_max != s*pi; limit handled separately)
```

The matrix equation is `I_meas = A * c` where `A[j, s] = T_s(q_j)`.

Tikhonov regularization with second-derivative smoothness constraint:

```
minimize  || W * (A*c - I_meas) ||^2  +  alpha * c^T B c
```

where W is the diagonal weight matrix (W_jj = 1/sigma_j when errors available,
else 1), and B is the diagonal smoothness matrix:

```
B_ss = (s * pi / D_max)^4 * D_max / 2
```

(from the integral of the squared second derivative of the basis functions).

Solution:

```
c = (A^T W^2 A + alpha * B)^{-1} A^T W^2 I_meas
```

Covariance of c: `Cov_c = (A^T W^2 A + alpha * B)^{-1} A^T W^2 Cov_I W^2 A (A^T W^2 A + alpha * B)^{-1}`

The regularization parameter alpha is selected automatically by default using
Generalized Cross-Validation (GCV); users may override it manually.

Derived quantities from p(r):
- `I0_pr = 4*pi * integral_0^{D_max} p(r) dr`
- `Rg_pr = sqrt( integral_0^{D_max} r^2 * p(r) dr / (2 * integral_0^{D_max} p(r) dr) )`
- `D_max` (user-provided)
- Error estimates on I0 and Rg by propagating `Cov_c`.

**Specification:**

New file `scatterbrain/analysis/ift.py`:

```python
class IFTResult(TypedDict):
    r: np.ndarray            # r values of p(r) (length n_r_points)
    pr: np.ndarray           # p(r) values
    pr_err: np.ndarray       # 1-sigma errors on p(r)
    coefficients: np.ndarray # Moore basis coefficients c_s
    I0_pr: float             # I(0) from p(r)
    I0_pr_err: float
    Rg_pr: float             # Rg from p(r)
    Rg_pr_err: float
    D_max: float             # as provided by user
    alpha: float             # regularization parameter used
    alpha_selection: str     # "GCV" or "manual"
    n_terms: int             # number of sine terms used
    chi_squared_reduced: float
    q_min: float
    q_max: float
    num_points_fit: int


def indirect_fourier_transform(
    curve: ScatteringCurve1D,
    D_max: float,
    n_terms: int = 20,
    alpha: Optional[float] = None,
    q_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    n_r_points: int = 100,
) -> Optional[IFTResult]
```

- `D_max`: maximum particle dimension; required, must be positive.
- `n_terms`: number of sine basis functions; default 20, must satisfy
  `n_terms < len(fit_points)`.
- `alpha=None`: auto-select via GCV. GCV criterion:
  `GCV(alpha) = || (I - H(alpha)) * I_meas ||^2 / (N - trace(H(alpha)))^2`
  where `H(alpha) = A * (A^T W^2 A + alpha*B)^{-1} A^T W^2` is the hat matrix.
  Minimize GCV over a log-spaced grid of alpha values (e.g., 1e-10 to 1e5,
  50 points).
- Soft failure (return None + WARNING) if:
  - `len(fit_points) < n_terms + 2`
  - `D_max <= 0`
  - GCV minimization fails to converge

Exports: add to `scatterbrain/analysis/__init__.py`.

**Files:**
- `scatterbrain/analysis/ift.py` (new)
- `scatterbrain/analysis/__init__.py`

**Tests:** `tests/test_analysis.py`
- On synthetic sphere data: verify `Rg_pr` within 5% of known sphere Rg,
  and `D_max >= 2 * Rg_pr`.
- Verify `p(r) >= 0` for r near 0 and r near D_max (positivity is not enforced
  algebraically but should hold for well-behaved data).
- Verify `I0_pr` matches `I(q=0)` extrapolation within reasonable tolerance.
- Test GCV selects a finite alpha.
- Test manual alpha override.
- Test soft failure on insufficient points.

**Visualization:** Add `plot_pr(ift_result, ax=None, ...)` to
`scatterbrain/visualization.py`. Plots p(r) vs r with error band (shaded).

---

### 3.2 -- Hard-Sphere Structure Factor S(q)

**Scientific context:** Real samples are often concentrated; particle-particle
interactions modify I(q) from the dilute `I(q) = scale * P(q) + background`.
The full model is `I(q) = scale * P(q) * S(q) + background`. The hard-sphere
Percus-Yevick (PY) structure factor is the standard starting point.

**Method:** Percus-Yevick analytical solution for hard spheres
(Wertheim 1963, Ashcroft and Lekner 1966).

For hard spheres of radius R_hs at volume fraction phi:

```
S(q, phi, R_hs) = 1 / (1 - phi * C_hat(q, phi, R_hs))

where C_hat(q) = -24*phi * integral_0^1 c(r) * sin(q*R_hs*2*r) / (q*R_hs*2*r) * r^2 dr
```

The PY direct correlation function c(r) for r <= 2*R_hs has the analytical form:

```
c(r/2R_hs) = -(alpha + beta*s + gamma*s^3)  for s = r/(2*R_hs) in [0,1]
```

where:

```
alpha = (1 + 2*phi)^2 / (1 - phi)^4
beta  = -6*phi*(1 + phi/2)^2 / (1 - phi)^4
gamma = phi*alpha/2
```

The Fourier transform of c(r) is:

```
C_hat(q) = -24*phi * [alpha*A(x)/x^3 + beta*B(x)/x^4 + gamma*D(x)/x^6]
where x = 2*q*R_hs
A(x) = sin(x) - x*cos(x)
B(x) = 2*x*sin(x) - (x^2 - 2)*cos(x) - 2
D(x) = [4*x^3 - 24*x]*sin(x) - [x^4 - 12*x^2 + 24]*cos(x) + 24
```

Handle the x -> 0 limit via Taylor expansion to avoid numerical instability.

```
sphere_structure_factor_pq(
    q: np.ndarray,
    volume_fraction: float,
    radius_hs: float,
) -> np.ndarray
```

Returns S(q). Raises `ValueError` if `volume_fraction` not in (0, 1) or
`radius_hs <= 0`. Returns `np.ones_like(q)` for `volume_fraction = 0`
(non-interacting limit).

Modify `fit_model` to accept an optional `structure_factor` callable so that
the fitted model becomes `I(q) = scale * P(q) * S(q, *sf_params) + background`.
The structure factor parameters are appended to `param_names` and handled by
the existing fixed_params mechanism.

**Files:**
- `scatterbrain/modeling/structure_factors.py` (new)
- `scatterbrain/modeling/__init__.py`
- `scatterbrain/modeling/fitting.py` (add `structure_factor` parameter)

**Tests:** `tests/test_modeling.py`
- S(q=0, phi, R) should equal `1 / (1 + 24*phi*alpha/3 + ...)` (known limit).
- At phi -> 0, S(q) -> 1 for all q.
- At intermediate phi (e.g., 0.3), S(q) shows the expected suppression at
  low-q and correlation peak near q = pi/R_hs.
- `ValueError` on out-of-range phi.
- Test `fit_model` with `structure_factor=sphere_structure_factor_pq`.

---

### 3.3 -- Polydispersity for Form Factors

**Scientific context:** Real particles are never perfectly monodisperse. A
small degree of polydispersity smooths form factor oscillations and shifts
apparent radii. Implementing a polydisperse form factor wrapper is essential
for quantitative analysis of real samples.

**Method:** Numerical integration over a size distribution f(R; R_mean, sigma):

```
P_poly(q, R_mean, sigma_R) = integral_0^inf P(q, R) * f(R; R_mean, sigma_R) dR
```

Two supported distributions:
- **Gaussian**: `f(R) = Normal(R_mean, sigma_R)`, truncated at R=0.
- **Schulz-Zimm**: `f(R) propto R^z * exp(-(z+1)*R/R_mean)`, z = (R_mean/sigma_R)^2 - 1.
  This is the physically motivated distribution for polymer-controlled growth.

Numerical integration: use Gauss-Hermite quadrature (for Gaussian) or
Gauss-Laguerre quadrature (for Schulz-Zimm) over the size variable. Use
50-point quadrature by default (parameter `n_quad_points`).

```
polydisperse_form_factor(
    form_factor_func: Callable[[np.ndarray, float, ...], np.ndarray],
    q: np.ndarray,
    R_mean: float,
    sigma_R: float,
    distribution: str = "gaussian",
    n_quad_points: int = 50,
    **fixed_model_params,
) -> np.ndarray
```

- `form_factor_func`: any single-R form factor (e.g., `sphere_pq`, `cylinder_pq`
  with fixed `length`).
- The first structural parameter (after q) is always taken as R; additional
  parameters are passed through via `**fixed_model_params`.
- Returns weighted average P_poly(q) normalized to P_poly(0) = 1.
- Raises `ValueError` if `sigma_R < 0` or `R_mean <= 0`.

**Files:**
- `scatterbrain/modeling/polydispersity.py` (new)
- `scatterbrain/modeling/__init__.py`

**Tests:** `tests/test_modeling.py`
- At sigma_R = 0, `polydisperse_form_factor(sphere_pq, q, R, 0)` should equal
  `sphere_pq(q, R)`.
- With increasing sigma_R, form factor oscillations should be smoothed (verify
  the secondary maximum decreases for sphere).
- Verify P_poly(q=0) = 1 for both distributions.

---

### 3.4 -- Curve Merging and Rebinning

**Scientific context:** Multi-detector SAXS/WAXS experiments produce separate
curves for different angular ranges that must be merged into a single I(q).
Noisy high-q data also benefit from rebinning to reduce the effective number
of points while improving per-point statistics.

#### 3.4a -- Merging

```
merge_curves(
    curves: List[ScatteringCurve1D],
    overlap_method: str = "scale",
    reference_index: int = 0,
) -> ScatteringCurve1D
```

- Requires at least 2 curves; raises `ProcessingError` if any pair has no
  overlap.
- `overlap_method="scale"`: determines a scale factor for each curve to match
  the reference in the overlap region (weighted mean ratio in overlap).
- `overlap_method="stitch"`: no scaling; just concatenates at the overlap
  midpoint.
- In the overlap region, computes the weighted mean
  `I_merged = (I_1/sigma_1^2 + I_2/sigma_2^2) / (1/sigma_1^2 + 1/sigma_2^2)`
  when errors are available; simple mean otherwise.
- Outside overlap: use the appropriate curve directly.
- Appends to `metadata["processing_history"]`.

#### 3.4b -- Rebinning

```
rebin(
    curve: ScatteringCurve1D,
    n_bins: int,
    q_scale: str = "log",
    q_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> ScatteringCurve1D
```

- Creates `n_bins` equally spaced bins in q (linear) or log(q) space.
- Within each bin: weighted mean intensity (weight = 1/sigma^2 if errors
  available; else count-based).
- New error: `sigma_bin = 1/sqrt(sum 1/sigma_i^2)` when errors available;
  else `std(I_i) / sqrt(N_i)`.
- Bins with zero data points are dropped.
- Appends to `metadata["processing_history"]`.

**Files:**
- `scatterbrain/processing/merge.py` (new)
- `scatterbrain/processing/__init__.py`

**Tests:** `tests/test_processing.py`
- Merging two curves with known overlap: verify scale factor is recovered.
- Rebinning: verify bin centers are correct, total q range is preserved.
- Merging without overlap: verify `ProcessingError`.
- Rebin to n_bins > len(curve): verify soft failure or at most original N bins.

---

### 3.5 -- WAXS Peak Analysis

**Scientific context:** WAXS patterns contain Bragg peaks from crystalline
materials. Extracting peak positions, widths, and integrated intensities is
the basis for d-spacing, phase identification, and Scherrer crystallite size
estimation.

#### 3.5a -- Peak Finding and Fitting

New file `scatterbrain/analysis/waxs.py`:

```python
class PeakResult(TypedDict):
    q_peak: float          # fitted peak center
    q_peak_err: float
    intensity_peak: float  # peak amplitude
    intensity_peak_err: float
    fwhm_q: float          # full width at half maximum in q
    fwhm_q_err: float
    d_spacing: float       # 2*pi / q_peak
    integrated_intensity: float
    profile: str           # "gaussian" or "pseudo_voigt"
    r_squared: float       # quality of peak fit


def find_peaks(
    curve: ScatteringCurve1D,
    min_height: Optional[float] = None,
    min_prominence: Optional[float] = None,
    q_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> np.ndarray   # indices of found peaks
```

Uses `scipy.signal.find_peaks` with prominence and height filtering.

```python
def fit_peak(
    curve: ScatteringCurve1D,
    peak_index: int,
    window_width: Optional[float] = None,
    profile: str = "gaussian",
) -> Optional[PeakResult]
```

Fits a Gaussian or pseudo-Voigt profile around `curve.q[peak_index]` in a
window of width `window_width` (default: 4 * estimated FWHM from peak
neighbors). Uses `lmfit` (available from Phase 2) for fitting.

**Gaussian profile:** `I(q) = A * exp(-4*ln2 * (q-q0)^2 / FWHM^2) + bg`

**Pseudo-Voigt profile:** `I(q) = A * [eta * L(q) + (1-eta) * G(q)] + bg`
where eta is the Lorentzian fraction (0=pure Gaussian, 1=pure Lorentzian).

Soft failure: return None + WARNING if fewer than 5 points in window, or if
lmfit fails to converge.

#### 3.5b -- Scherrer Equation

```python
def scherrer_size(
    peak_result: PeakResult,
    wavelength: float,
    shape_factor: float = 0.9,
) -> Tuple[float, float]   # (L, L_err)
```

Scherrer formula in q-space:

```
L = 2 * pi * K / FWHM_q
L_err = 2 * pi * K * FWHM_q_err / FWHM_q^2
```

where K is the shape factor (default 0.9 for spherical crystallites).

Note: this formula assumes q = 4*pi*sin(theta)/lambda. Log a WARNING if the
input curve's q_unit is not "nm^-1" or "A^-1" (L units follow q_unit).

**Files:**
- `scatterbrain/analysis/waxs.py` (new)
- `scatterbrain/analysis/__init__.py`
- `scatterbrain/visualization.py` (add `plot_peak_fit` function)

**Tests:** `tests/test_analysis.py`
- Synthetic Gaussian peak: verify `fit_peak` recovers position, FWHM, amplitude.
- Verify `d_spacing = 2*pi / q_peak`.
- Verify `scherrer_size` returns L close to expected for known FWHM.
- Verify soft failure when window has too few points.

---

### 3.6 -- SAXSImage Class and 2D Image Loading

**Scientific context:** The full data pipeline starts from 2D detector images.
Phase 3 introduces the `SAXSImage` data structure and loading via `fabio`,
laying the groundwork for azimuthal integration in task 3.7.

**Note:** `fabio` is an optional dependency. All 3.6/3.7 code lives behind an
import guard; calling these functions without fabio installed raises an
informative `ImportError`.

#### SAXSImage class

Add to `scatterbrain/core.py` (or `scatterbrain/reduction/image.py`):

```python
class SAXSImage:
    """2D detector image with associated calibration metadata."""

    image: np.ndarray          # 2D float array, shape (n_rows, n_cols)
    mask: Optional[np.ndarray] # bool array, True = masked (bad pixel)
    wavelength: float          # X-ray wavelength in Angstrom
    distance: float            # sample-to-detector distance in mm
    pixel_size: float          # pixel size in mm (assumed square)
    beam_center_x: float       # beam center column (pixels)
    beam_center_y: float       # beam center row (pixels)
    metadata: dict
```

Methods: `__init__`, `__repr__`, `__str__`, `copy()`, `apply_mask(mask)`,
`to_dict()`.

#### Image loading

Add `load_image` to `scatterbrain/io.py`:

```python
def load_image(
    filepath: Union[str, Path],
    wavelength: float,
    distance: float,
    pixel_size: float,
    beam_center: Tuple[float, float],
    mask: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
) -> SAXSImage
```

Uses `fabio.open(filepath).data` to read the 2D array. Supported formats
(via fabio): TIFF, EDF, CBF, MarCCD, Pilatus HDF5.

Raises `ImportError` (with install hint) if fabio is not installed.
Raises `IOError` if the file cannot be read.

**Optional dependency declaration:**

```toml
# pyproject.toml
[project.optional-dependencies]
reduction = [
    "fabio>=0.14",
    "pyFAI>=2024.1",
]
```

Update `uv sync --extra reduction` instructions in CLAUDE.md and README.md.

**Files:**
- `scatterbrain/core.py` (add `SAXSImage`)
- `scatterbrain/io.py` (add `load_image`)
- `pyproject.toml` (new `reduction` optional extra)
- `scatterbrain/reduction/__init__.py` (currently placeholder; expose
  `azimuthal_integrate` when 3.7 is done)

**Tests:** `tests/test_core.py` and `tests/test_io.py`
- `SAXSImage` construction and attribute access.
- `SAXSImage.apply_mask`: verify masked pixels are set correctly.
- `load_image` without fabio: verify `ImportError` with helpful message.
- `load_image` with a synthetic TIFF (generated with `numpy + PIL` or just
  skip if fabio unavailable using `pytest.importorskip("fabio")`).

---

### 3.7 -- Azimuthal Integration

**Scientific context:** Converts a 2D `SAXSImage` into a 1D `ScatteringCurve1D`
by integrating intensity over azimuthal angle at each q value (or 2-theta bin).

**Note:** Uses pyFAI as the integration backend. pyFAI is highly optimized
(C extensions) and handles detector geometry, polarization, and solid-angle
corrections correctly. Wrapping it avoids re-implementing a complex algorithm.

```python
def azimuthal_integrate(
    image: SAXSImage,
    n_bins: int = 1000,
    q_range: Optional[Tuple[float, float]] = None,
    azimuth_range: Optional[Tuple[float, float]] = None,
    unit: str = "nm^-1",
    error_model: str = "poisson",
    polarization_factor: Optional[float] = None,
) -> ScatteringCurve1D
```

Internally:
1. Build a `pyFAI.AzimuthalIntegrator` from `image.wavelength`,
   `image.distance`, `image.pixel_size`, `image.beam_center_x/y`.
2. Call `ai.integrate1d(image.image, n_bins, ...)` with mask applied.
3. Wrap result in `ScatteringCurve1D` with appropriate q_unit and metadata.

`error_model="poisson"`: pyFAI estimates sigma = sqrt(I). Pass through to
pyFAI's `error_model` parameter.

`unit="nm^-1"` or `"A^-1"`: map to pyFAI unit strings `"q_nm^-1"` and
`"q_A^-1"` respectively.

Raises `ImportError` (with install hint) if pyFAI is not installed.

Add `plot_image(image: SAXSImage, ...)` to `scatterbrain/visualization.py`:
displays the 2D detector image with optional mask overlay, beam center marker,
and q-contour lines.

**Files:**
- `scatterbrain/reduction/__init__.py` (expose `azimuthal_integrate`)
- `scatterbrain/reduction/integration.py` (new)
- `scatterbrain/visualization.py` (add `plot_image`)

**Tests:** `tests/test_reduction.py` (new)
- Without pyFAI: `ImportError` with helpful message.
- With pyFAI (`pytest.importorskip("pyfai")`): integrate a synthetic 2D
  ring pattern and verify the output `ScatteringCurve1D` has the correct
  q range and approximate intensity profile (flat for isotropic ring).

---

## Priority Order

```
3.1  p(r) via IFT          -- highest scientific value; pure numpy/scipy
3.2  Hard-sphere S(q)      -- enables concentrated-system modeling
3.3  Polydispersity        -- improves accuracy of all form factor fits
3.4  Curve merging         -- needed for multi-detector experiments
3.5  WAXS peak analysis    -- extends library to WAXS users
3.6  SAXSImage + loading   -- foundation of 2D pipeline
3.7  Azimuthal integration -- completes 2D->1D pipeline (requires 3.6)
```

Tasks 3.1, 3.2, 3.3 can be developed in parallel (no mutual dependencies).
Task 3.4 is independent of 3.1-3.3. Task 3.5 is independent of all others.
Task 3.7 requires 3.6.

The 2D pipeline (3.6 + 3.7) is the most complex component and can be
developed on a separate branch in parallel with 3.1-3.5.

---

## New Module Layout After Phase 3

| Module | New API |
|--------|---------|
| `scatterbrain/analysis/ift.py` | `indirect_fourier_transform`, `IFTResult` |
| `scatterbrain/analysis/waxs.py` | `find_peaks`, `fit_peak`, `scherrer_size`, `PeakResult` |
| `scatterbrain/modeling/structure_factors.py` | `sphere_structure_factor_pq` |
| `scatterbrain/modeling/polydispersity.py` | `polydisperse_form_factor` |
| `scatterbrain/processing/merge.py` | `merge_curves`, `rebin` |
| `scatterbrain/reduction/integration.py` | `azimuthal_integrate` |
| `scatterbrain/core.py` | `SAXSImage` added |
| `scatterbrain/io.py` | `load_image` added |
| `scatterbrain/visualization.py` | `plot_pr`, `plot_peak_fit`, `plot_image` added |

---

## Phase 3 Completion Criteria

| Criterion | Status |
|-----------|--------|
| `indirect_fourier_transform` returns valid p(r) for sphere test data | Pending: Task 3.1 |
| `Rg_pr` from IFT matches Guinier `Rg` within 5% on sphere data | Pending: Task 3.1 |
| `sphere_structure_factor_pq` returns S->1 as phi->0 | Pending: Task 3.2 |
| `fit_model` accepts optional `structure_factor` argument | Pending: Task 3.2 |
| `polydisperse_form_factor` with sigma=0 equals monodisperse P(q) | Pending: Task 3.3 |
| `merge_curves` tested on two-curve overlap scenario | Pending: Task 3.4 |
| `rebin` produces correct bin centers and propagated errors | Pending: Task 3.4 |
| `find_peaks` locates Bragg peaks in synthetic WAXS data | Pending: Task 3.5 |
| `fit_peak` recovers FWHM within 1% for noiseless Gaussian | Pending: Task 3.5 |
| `SAXSImage` class instantiates and validates attributes | Pending: Task 3.6 |
| `load_image` raises `ImportError` with helpful message if fabio absent | Pending: Task 3.6 |
| `azimuthal_integrate` produces `ScatteringCurve1D` from ring image | Pending: Task 3.7 |
| `plot_pr`, `plot_peak_fit`, `plot_image` added to `visualization.py` | Pending: Tasks 3.1, 3.5, 3.7 |
| pytest >= 300 passing, 0 failing | Pending |
| Test coverage >= 88% (all non-reduction modules) | Pending |

---

## What is Explicitly Out of Scope for Phase 3

These belong to Phase 4 or later:

- Interactive plots (plotly / bokeh)
- Desmearing (Lake algorithm for slit-smearing correction)
- Advanced 2D corrections (polarization, flat field, solid angle beyond pyFAI)
- GISAXS / GIWAXS geometry and grazing-incidence analysis
- Global fitting across multiple datasets simultaneously
- Machine learning for parameter estimation or phase identification
- Time-resolved data series management
- GUI
- Absolute intensity calibration workflow (glassy carbon / water standards)
