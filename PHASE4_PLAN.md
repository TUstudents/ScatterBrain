# ScatterBrain -- Phase 4 Plan: Production Readiness and Specialized Techniques

**Version:** 1.0
**Branch:** `main`
**Reference:** `Design_document.md` sec.2.2, `PHASE3_PLAN.md`

---

## Goal

Phase 4 completes the library as a production-grade scientific tool by adding:

1. Slit-smearing correction (desmearing) for Kratky camera data
2. Absolute intensity calibration workflow (glassy carbon / water standards)
3. Multi-dataset container (`ScatteringExperiment`) and kinetic series management
4. Global fitting across multiple datasets with shared parameters
5. Interactive plots via plotly
6. Extended form factors: Debye function, ellipsoid, vesicle
7. Extended structure factors: sticky hard sphere, screened Coulomb (Yukawa)
8. GISAXS geometry conversion and standard map cuts

Phase 4 assumes Phase 3 is complete: p(r)/IFT, hard-sphere S(q), polydispersity,
curve merging, WAXS peak analysis, and the full 2D pipeline (SAXSImage + pyFAI).

---

## Task Overview

| Task | Component | Complexity | New Dependencies |
|------|-----------|------------|-----------------|
| 4.1 | Desmearing (Lake algorithm) | High | none |
| 4.2 | Absolute intensity calibration | Moderate | none |
| 4.3 | ScatteringExperiment + KineticSeries | Moderate | h5py (optional) |
| 4.4 | Global fitting | High | lmfit (already from Phase 2) |
| 4.5 | Interactive plots (plotly) | Moderate | plotly (new optional) |
| 4.6 | Extended form factors | Moderate | none |
| 4.7 | Extended structure factors | High | none |
| 4.8 | GISAXS geometry and cuts | Moderate | none |

---

## Tasks

### 4.1 -- Desmearing: Lake Algorithm

**Scientific context:** Kratky cameras and some synchrotron setups use a
line-focused beam. The resulting measured intensity is the convolution of the
true isotropic I(q) with the beam length profile W(l):

```
I_smeared(q) = integral_{-inf}^{+inf} I_true(sqrt(q^2 + l^2)) * W(l) dl
```

The Lake (1967) iterative deconvolution algorithm recovers I_true:

```
I_0(q) = I_smeared(q)
I_{n+1}(q) = I_n(q) * I_smeared(q) / I_n_smeared(q)
```

where `I_n_smeared(q)` is I_n convolved with the beam profile using the
same integral above.

**Beam profile:** For a rectangular beam of total length 2*L_half:
`W(l) = 1 / (2*L_half)` for `|l| <= L_half`, else 0.
Trapezoid and Gaussian profiles are supported as alternatives.

**Numerical implementation:**

For each output q_j:
1. Build an integration grid `l_k` (Gauss-Legendre on [-L_half, L_half],
   64 points default).
2. Compute `q_eff_k = sqrt(q_j^2 + l_k^2)` for each l_k.
3. Evaluate I_n(q_eff_k) by cubic spline interpolation over the measured q range.
4. For q_eff_k > q_max: extrapolate using the Porod law
   `I(q) = Kp * q^{-n}` with Kp and n from an optional `PorodResult`; if not
   provided, use a power-law fit over the last 20% of the measured q range.
5. Compute the numerical integral via the quadrature weights.
6. Convergence: stop when `max(|I_{n+1} - I_n| / I_n) < tolerance`.

```python
def desmear(
    curve: ScatteringCurve1D,
    beam_half_length: float,
    beam_profile: str = "rectangular",
    porod_result: Optional[PorodResult] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    n_integration_points: int = 64,
) -> ScatteringCurve1D
```

Returns a new `ScatteringCurve1D`. Appends to `metadata["processing_history"]`.
Raises `ProcessingError` if the curve is not a `ScatteringCurve1D`.
Soft failure (returns None + WARNING) if the q range is too narrow for
extrapolation.

Add `DesmearResult(TypedDict)` with `n_iterations`, `final_tolerance`,
`converged: bool`, stored in `metadata["desmear_info"]`.

**Files:**
- `scatterbrain/processing/desmear.py` (new)
- `scatterbrain/processing/__init__.py`

**Tests:** `tests/test_processing.py`
- Smeary a known analytical curve, then desmear it; verify recovery within 2%.
- Verify `converged=True` for well-behaved data.
- Verify `ProcessingError` on wrong input type.
- Verify Porod extrapolation is used when q_eff exceeds q_max.

---

### 4.2 -- Absolute Intensity Calibration

**Scientific context:** Relative intensities in arbitrary units are sufficient
for shape analysis (Guinier, Porod exponent, form factor shape), but absolute
units (cm^-1 sr^-1) are required for:
- Calculation of volume fractions and specific surface areas from Q* and Kp.
- Comparison between instruments, beamlines, and publications.
- Accurate molecular weight estimation.

**Calibration equation:**

```
k = I_ref(q) / [(I_std(q) * T_std - I_empty(q) * T_empty) / d_std]
I_abs(q) = k * (I_sample(q) * T_sample - I_empty(q) * T_empty) / d_sample
```

where T = transmission (fraction), d = sample thickness (cm).
k is averaged over a flat q-range of the reference where I_ref is constant.

**Supported standards:**

- **Glassy carbon** (NIST SRM 3600): pass a `reference_intensity` curve
  (loaded from file or embedded lookup table for common energies).
- **Water** (`reference_intensity=None` + `standard="water"`): uses the
  known water cross-section sigma = 0.01632 cm^-1 sr^-1 at 20 degC,
  independent of q; k is determined from the flat high-q region of the
  water measurement.

```python
class CalibrationResult(TypedDict):
    scale_factor: float         # k
    scale_factor_std: float     # standard deviation of k in calibration range
    q_calibration_min: float
    q_calibration_max: float
    standard: str               # "glassy_carbon" or "water" or "custom"
    units_out: str              # "cm^-1"


def calibrate_intensity(
    sample_curve: ScatteringCurve1D,
    standard_curve: ScatteringCurve1D,
    reference_intensity: Union[float, ScatteringCurve1D],
    sample_transmission: float = 1.0,
    standard_transmission: float = 1.0,
    empty_curve: Optional[ScatteringCurve1D] = None,
    empty_transmission: float = 1.0,
    sample_thickness: float = 1.0,
    standard_thickness: float = 1.0,
    q_calibration_range: Optional[Tuple[float, float]] = None,
) -> Tuple[ScatteringCurve1D, CalibrationResult]
```

Returns the calibrated sample curve (with `intensity_unit="cm^-1"`) and the
`CalibrationResult` dict.

Raises `ProcessingError` if k has relative standard deviation > 10% in the
calibration range (indicates a poor standard or incorrect inputs); log a
WARNING but do not prevent returning the result.

**Files:**
- `scatterbrain/processing/calibration.py` (new)
- `scatterbrain/processing/__init__.py`
- `scatterbrain/data/`: add `glassy_carbon_srm3600.dat` (tabulated I(q) for
  several common X-ray energies) as a bundled data file.

**Tests:** `tests/test_processing.py`
- With k=1 (reference_intensity == standard_curve.intensity): verify
  scale_factor ~ 1.0.
- Verify transmitted sample is scaled correctly.
- Verify `intensity_unit` of output is set to `"cm^-1"`.
- Verify WARNING when k_std/k > 0.10.

---

### 4.3 -- ScatteringExperiment Container and KineticSeries

**Scientific context:** Real experiments produce series of datasets:
concentration series (to separate P(q) and S(q)), temperature series (phase
transitions), kinetic series (time-resolved reactions). A container that
manages these datasets and supports batch operations is essential for
systematic analysis.

**Specification:**

Add to `scatterbrain/core.py`:

```python
class ScatteringExperiment:
    """
    Container for a collection of related ScatteringCurve1D objects.

    Curves are stored in an ordered dict keyed by user-provided names.
    Batch processing methods return a new ScatteringExperiment with the
    processed curves; inputs are never modified.
    """

    curves: Dict[str, ScatteringCurve1D]
    metadata: dict

    def __init__(self, metadata: Optional[dict] = None)
    def add_curve(self, curve: ScatteringCurve1D, name: Optional[str] = None) -> str
    def remove_curve(self, name: str) -> None
    def get_curve(self, name: str) -> ScatteringCurve1D
    def __len__(self) -> int
    def __iter__(self)           # yields (name, curve) pairs
    def __repr__(self) -> str
    def apply_to_all(
        self,
        func: Callable[[ScatteringCurve1D, ...], ScatteringCurve1D],
        **kwargs,
    ) -> ScatteringExperiment
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, data: dict) -> ScatteringExperiment
    def save(self, filepath: Union[str, Path], format: str = "hdf5") -> None
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> ScatteringExperiment
```

`save` / `load` in HDF5 format (via `h5py`, new optional dependency under a
`storage` extra). Fall back to JSON with base64-encoded arrays if h5py is
absent; raise `ImportError` with an install hint for HDF5.

**KineticSeries subclass:**

```python
class KineticSeries(ScatteringExperiment):
    """
    ScatteringExperiment with an associated time axis for kinetic SAXS.

    Curves must be added in order; time stamps are stored in seconds.
    """

    times: List[float]          # seconds from t=0

    def add_curve(self, curve, time: float, name=None) -> str
    def intensity_at_q(
        self,
        q_value: float,
        interpolation: str = "nearest",
    ) -> Tuple[np.ndarray, np.ndarray]  # (times, intensities)
    def parameter_series(
        self,
        analysis_func: Callable,
        param_key: str,
        **analysis_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]  # (times, param_values)
```

`parameter_series` applies `analysis_func` (e.g., `guinier_fit`) to each
curve, extracts `result[param_key]`, and returns the time series. Returns NaN
for any curve where the analysis returns None.

**Files:**
- `scatterbrain/core.py` (add `ScatteringExperiment`, `KineticSeries`)
- `pyproject.toml` (new `storage` optional extra: `h5py>=3.7`)

**Tests:** `tests/test_core.py`
- Add three curves, verify `len()` and iteration.
- `apply_to_all` with `subtract_background`: verify all curves are processed.
- `save` / `load` round-trip: verify curve data is preserved (skip if h5py absent).
- `KineticSeries.intensity_at_q`: verify correct q interpolation.
- `KineticSeries.parameter_series` with `guinier_fit` and `"Rg"`.

---

### 4.4 -- Global Fitting

**Scientific context:** A concentration series of the same sample should yield
the same structural parameters (radius, thickness) but different scales and
backgrounds. Global fitting enforces these constraints simultaneously, producing
more accurate shared parameters than fitting each dataset independently.

**Specification:**

```python
class GlobalFitResult(TypedDict):
    shared_params: Dict[str, float]
    shared_params_stderr: Dict[str, float]
    per_curve_params: Dict[str, Dict[str, float]]   # keyed by curve name/index
    per_curve_params_stderr: Dict[str, Dict[str, float]]
    fit_curves: List[ScatteringCurve1D]
    chi_squared_reduced: float
    success: bool
    message: str
    lmfit_result: Any  # lmfit.MinimizerResult


def global_fit(
    curves: List[ScatteringCurve1D],
    model_func: Callable[..., np.ndarray],
    param_names: List[str],
    initial_params: Dict[str, float],
    shared_params: List[str],
    per_curve_params: List[str],
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    q_ranges: Optional[List[Optional[Tuple[Optional[float], Optional[float]]]]] = None,
    structure_factor: Optional[Callable] = None,
) -> Optional[GlobalFitResult]
```

Implementation using lmfit:

1. Build a single `lmfit.Parameters` object:
   - For each shared parameter: one `Parameter` (e.g., `radius`).
   - For each per-curve parameter and each curve i: one `Parameter`
     (e.g., `scale_0`, `scale_1`, `background_0`, ...).
2. Define a combined residual function that concatenates the weighted residuals
   from all datasets.
3. Call `lmfit.minimize(residual, params, method="leastsq")`.
4. Extract shared and per-curve parameter values and uncertainties from the
   result.

`q_ranges`: per-curve q-range selection. If None, use the full range of each
curve. Length must equal `len(curves)` or be None.

Soft failure (return None + WARNING) if any curve has fewer than 5 points.
Raises `FittingError` on parameter name mismatches.

**Files:**
- `scatterbrain/modeling/fitting.py` (add `global_fit`, `GlobalFitResult`)
- `scatterbrain/modeling/__init__.py`

**Tests:** `tests/test_modeling.py`
- Fit a sphere model to three synthetic datasets with the same radius but
  different scales; verify the shared `radius` is recovered within 1%.
- Verify per-curve `scale` values match the generating values.
- Verify `FittingError` on parameter name mismatch.
- Verify soft failure when curves are too short.

---

### 4.5 -- Interactive Plots (plotly)

**Scientific context:** Static matplotlib plots are sufficient for publication
but not for interactive data exploration (zooming, hovering, toggling curves).
plotly provides web-native interactive figures that can be saved as standalone
HTML and embedded in Jupyter notebooks.

**Specification:**

New file `scatterbrain/visualization_interactive.py`:

```python
def plot_iq_interactive(
    curves: Union[ScatteringCurve1D, List[ScatteringCurve1D]],
    q_scale: str = "log",
    i_scale: str = "log",
    labels: Optional[List[str]] = None,
    title: str = "Scattering Intensity",
    show_errorbars: bool = True,
) -> plotly.graph_objects.Figure

def plot_guinier_interactive(
    curve: ScatteringCurve1D,
    guinier_result: Optional[GuinierResult] = None,
    title: str = "Guinier Plot",
) -> plotly.graph_objects.Figure

def plot_porod_interactive(
    curve: ScatteringCurve1D,
    porod_result: Optional[PorodResult] = None,
    mode: str = "logI_vs_logq",
    title: str = "Porod Plot",
) -> plotly.graph_objects.Figure

def plot_pr_interactive(
    ift_result: IFTResult,
    title: str = "Pair Distance Distribution",
) -> plotly.graph_objects.Figure

def plot_kratky_interactive(
    curve: ScatteringCurve1D,
    guinier_result: Optional[GuinierResult] = None,
    normalized: bool = False,
) -> plotly.graph_objects.Figure
```

Each function:
- Raises `ImportError` with an install hint if plotly is not installed.
- Returns a `plotly.graph_objects.Figure`. The caller uses `.show()` or
  `.write_html(path)`.
- Error bars are represented as plotly `error_y` traces.
- Hover tooltips show `(q, I, sigma)` or the appropriate axis quantities.

**Optional dependency declaration:**

```toml
# pyproject.toml
[project.optional-dependencies]
interactive = [
    "plotly>=5.0",
]
```

Export all five functions from `scatterbrain.visualization_interactive`.
Do NOT import from this module at the top of `scatterbrain/__init__.py`
(would cause ImportError for users without plotly); expose it as
`scatterbrain.visualization_interactive`.

**Files:**
- `scatterbrain/visualization_interactive.py` (new)
- `pyproject.toml`

**Tests:** `tests/test_visualization_interactive.py` (new)
- All five functions: with plotly installed, verify the return type is
  `plotly.graph_objects.Figure` and the figure contains at least one trace.
- Without plotly: verify `ImportError` with helpful message
  (mock the import using `monkeypatch` or `pytest.importorskip`).

---

### 4.6 -- Extended Form Factors

Three additional analytically well-defined form factors that cover a broad
range of soft-matter morphologies.

#### 4.6a -- Debye Function (Gaussian Chain / Polymer)

The Debye (1947) function for a Gaussian polymer chain:

```
P(q, Rg) = 2 * (exp(-u) - 1 + u) / u^2,   u = (q * Rg)^2
```

As u -> 0: P(q) -> 1 (via L'Hopital: 2*(0 - 1 + 0)/0 -> use Taylor: 2*(u/2 - u^2/6 + ...)/u^2 -> 1)

```python
def gaussian_chain_pq(q: np.ndarray, rg: float) -> np.ndarray
```

Raises `ValueError` if `rg <= 0`.
Handle u < 1e-6 with the Taylor approximation `P ~ 1 - u/3` to avoid
numerical cancellation.

#### 4.6b -- Ellipsoid of Revolution

For a prolate (a > b) or oblate (a < b) ellipsoid of revolution with semi-axes
a along the symmetry axis and b perpendicular, randomly oriented in solution:

```
P(q, a, b) = integral_0^1
    P_sphere(q, r_eff(x)) dx

where r_eff(x) = sqrt(a^2 * x^2 + b^2 * (1 - x^2)),
x = cos(alpha), and P_sphere uses the sphere amplitude formula
with argument q * r_eff(x).
```

Evaluate the integral using 64-point Gauss-Legendre quadrature over x in
[0, 1] (same strategy as `cylinder_pq`). The sphere amplitude for arbitrary
radius R is:

```
f(q, R) = 3 * (sin(qR) - qR * cos(qR)) / (qR)^3
```

P(q) = integral_0^1 f(q, r_eff(x))^2 dx, normalized to P(0) = 1.

```python
def ellipsoid_pq(q: np.ndarray, semi_axis_a: float, semi_axis_b: float) -> np.ndarray
```

Raises `ValueError` if either semi-axis is <= 0. When a == b, recovers
`sphere_pq(q, a)` (verify in tests).

#### 4.6c -- Vesicle (Hollow Sphere)

A unilamellar vesicle: hollow sphere of inner radius R_inner and shell
thickness t_shell. Special case of `core_shell_sphere_pq` with `contrast_core=0`
(solvent inside). Provided as a dedicated function for clarity.

```
F(q) = V_outer * f(q, R_outer) - V_inner * f(q, R_inner)
P(q) = F(q)^2 / F(0)^2
where R_outer = R_inner + t_shell
      V_outer = (4/3)*pi*R_outer^3
      V_inner = (4/3)*pi*R_inner^3
      F(0) = V_outer - V_inner = (4/3)*pi*(R_outer^3 - R_inner^3)
```

```python
def vesicle_pq(q: np.ndarray, radius_inner: float, shell_thickness: float) -> np.ndarray
```

Raises `ValueError` if `radius_inner <= 0` or `shell_thickness <= 0`.

**Files:** `scatterbrain/modeling/form_factors.py`

**Tests:** `tests/test_modeling.py`
- `gaussian_chain_pq`: P(0)=1; monotonically decreasing; Guinier limit
  matches `exp(-(q*Rg)^2/3)` at low q.
- `ellipsoid_pq`: P(0)=1; at a==b recovers sphere P(q) within 1e-6.
- `vesicle_pq`: P(0)=1; for thin shell (t_shell -> 0) approaches hollow sphere
  limit.

---

### 4.7 -- Extended Structure Factors

#### 4.7a -- Sticky Hard Sphere (Baxter Model)

The Baxter (1968) sticky hard sphere model adds a short-range attractive
square-well perturbation to the hard sphere potential. The analytical PY
solution (Menon, Regnaut, and Rajagopalan 1991) gives:

```
S(q, phi, tau, R_hs)
```

where tau is the "stickiness" parameter (tau -> inf: pure hard sphere;
tau ~ 0.1: strong aggregation tendency).

The structure factor is:

```
S(q) = 1 / |1 - C_hat(q)|^2  (in the PY approximation)
```

The direct correlation function c_hat(q) for the Baxter model has the
analytical form in terms of coefficients that satisfy a set of coupled
equations (quadratic in the sticky limit). Specifically, define:

```
lambda = 12 * tau * phi * mu  (where mu satisfies a quadratic equation)
mu = [1 + phi/2 + sqrt((1 + phi/2)^2 - 3*phi*(1 + phi/6)/(12*tau*(1+phi/4)^2))]
     / (3 * phi / (12*tau*(1+phi/4)))
```

The full expression follows from Baxter (1968) Eq. 4.13 and the PY
approximation. Reference: Menon et al. (1991) J. Chem. Phys. 95, 9186.

This is more algebraically involved than the PY hard sphere. Implement from
the published expressions in the cited reference.

```python
def sticky_sphere_structure_factor(
    q: np.ndarray,
    volume_fraction: float,
    stickiness: float,
    radius_hs: float,
) -> np.ndarray
```

Raises `ValueError` if `volume_fraction` not in (0, 1) or `stickiness <= 0`.
At `stickiness = inf` (numerically: `stickiness > 1e6`), returns the PY hard
sphere `sphere_structure_factor_pq` result.

#### 4.7b -- Screened Coulomb Structure Factor (Yukawa / DLVO)

For charged colloidal particles in solution, the effective pair potential is
approximated by a screened Coulomb (Yukawa) potential. The structure factor
in the Mean Spherical Approximation (MSA):

```
S(q, phi, epsilon, kappa * sigma)
```

where epsilon is the contact potential strength in units of kT, and
kappa * sigma is the reduced inverse Debye screening length
(sigma = 2 * R_hs, kappa = 1/Debye_length).

The MSA for the Yukawa fluid has an analytical solution in terms of
coefficients (Hayter and Penfold 1981, Hansen and Hayter 1982). This is
implemented in e.g. SasView as `HayterMSAStructure`. The analytical
expressions are lengthy; reference: Hansen and Hayter (1982) Mol. Phys. 46,
651.

```python
def screened_coulomb_structure_factor(
    q: np.ndarray,
    volume_fraction: float,
    contact_potential: float,
    screening_length: float,
    radius_hs: float,
) -> np.ndarray
```

Raises `ValueError` if `volume_fraction` not in (0, 1) or `radius_hs <= 0`.

**Files:** `scatterbrain/modeling/structure_factors.py`

**Tests:** `tests/test_modeling.py`
- `sticky_sphere_structure_factor` at large stickiness approaches hard sphere
  S(q) within 1%.
- `screened_coulomb_structure_factor` at large `screening_length` (weak
  screening) shows strong low-q suppression for repulsive charges.
- Both: `ValueError` on invalid volume_fraction.

---

### 4.8 -- GISAXS Geometry and Cuts

**Scientific context:** Grazing-Incidence SAXS (GISAXS) and GIWAXS probe
structure in thin films and at surfaces. The geometry is fundamentally
different from transmission SAXS: the beam hits the sample at a grazing angle
alpha_i (typically 0.1-0.5 degrees), and the 2D detector records both in-plane
(q_y) and out-of-plane (q_z) scattering simultaneously.

**Specification:**

New file `scatterbrain/reduction/gisaxs.py`:

```python
class GISAXSGeometry:
    """
    Encapsulates the geometry parameters for a GISAXS experiment.

    Attributes:
        alpha_i: float       -- grazing incidence angle in degrees
        wavelength: float    -- X-ray wavelength in Angstrom
        distance: float      -- sample-to-detector distance in mm
        pixel_size: float    -- pixel size in mm (assumed square)
        beam_center_x: float -- beam center column (pixels)
        beam_center_y: float -- beam center row (pixels, direct beam)
    """

    def critical_angle(self, delta: float) -> float
        # alpha_c = sqrt(2*delta) in radians, returned in degrees
        # delta = real part of refractive index decrement

    def pixel_to_angles(
        self,
        col: Union[float, np.ndarray],
        row: Union[float, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]
        # Returns (two_theta_f_deg, alpha_f_deg) for each pixel

    def angles_to_q(
        self,
        two_theta_f: Union[float, np.ndarray],
        alpha_f: Union[float, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]
        # Returns (q_y, q_z) in nm^-1 (or A^-1 if wavelength in A)
        # q_y = (2*pi/lambda) * cos(alpha_f) * sin(2*theta_f)
        # q_z = (2*pi/lambda) * (sin(alpha_f) + sin(alpha_i))
```

Standard cuts:

```python
def extract_yoneda_cut(
    image: SAXSImage,
    geometry: GISAXSGeometry,
    delta: float,
    width_deg: float = 0.05,
) -> ScatteringCurve1D
    # Horizontal cut at alpha_f = alpha_c(delta), averaged over +/-width_deg
    # Returns I vs q_y

def extract_inplane_cut(
    image: SAXSImage,
    geometry: GISAXSGeometry,
    alpha_f_deg: float,
    width_deg: float = 0.05,
) -> ScatteringCurve1D
    # Horizontal cut at fixed alpha_f, averaged over +/-width_deg
    # Returns I vs q_y

def extract_outofplane_cut(
    image: SAXSImage,
    geometry: GISAXSGeometry,
    two_theta_f_deg: float = 0.0,
    width_deg: float = 0.05,
) -> ScatteringCurve1D
    # Vertical cut at fixed 2*theta_f, averaged over +/-width_deg
    # Returns I vs q_z
```

Visualization:

```python
def plot_gisaxs_map(
    image: SAXSImage,
    geometry: GISAXSGeometry,
    q_range_y: Optional[Tuple[float, float]] = None,
    q_range_z: Optional[Tuple[float, float]] = None,
    log_scale: bool = True,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]
    # 2D color map with q_y (x-axis) and q_z (y-axis) labels
    # Uses pcolormesh with the q-coordinate grid from geometry.angles_to_q
```

All geometry functions raise `ValueError` if `alpha_i < 0` or `wavelength <= 0`.
Cuts use bilinear interpolation of the 2D image onto the cut row/column.

**Files:**
- `scatterbrain/reduction/gisaxs.py` (new)
- `scatterbrain/reduction/__init__.py`
- `scatterbrain/visualization.py` (add `plot_gisaxs_map`)

**Tests:** `tests/test_reduction.py`
- `GISAXSGeometry.angles_to_q`: verify q_y = 0 at 2*theta_f = 0 for any
  alpha_f; verify q_z = (2*pi/lambda)*(sin(alpha_f)+sin(alpha_i)).
- `critical_angle`: verify value for a known material (e.g., Si at 0.154 nm).
- Cut functions: verify output is a `ScatteringCurve1D` with correct q_unit.

---

## Priority Order

```
4.1  Desmearing              -- unblocks Kratky camera users; pure 1D
4.2  Absolute calibration    -- needed for any quantitative inter-lab comparison
4.3  ScatteringExperiment    -- foundation for global fit and kinetic series
4.4  Global fitting          -- requires 4.3 (container) and Phase 2 (lmfit)
4.5  Interactive plots       -- usability; independent of all other tasks
4.6  Extended form factors   -- independent; extends modeling coverage
4.7  Extended S(q)           -- requires Phase 3 (hard-sphere S(q) baseline)
4.8  GISAXS                  -- requires Phase 3 (SAXSImage); niche user base
```

Tasks 4.1, 4.2, 4.5, 4.6 are fully independent and can be developed in
parallel. Task 4.4 has a soft dependency on 4.3 (container provides the
multi-dataset input). Task 4.8 requires the Phase 3 `SAXSImage` class.

---

## New Module Layout After Phase 4

| Module | New API |
|--------|---------|
| `scatterbrain/processing/desmear.py` | `desmear` |
| `scatterbrain/processing/calibration.py` | `calibrate_intensity`, `CalibrationResult` |
| `scatterbrain/core.py` | `ScatteringExperiment`, `KineticSeries` added |
| `scatterbrain/modeling/fitting.py` | `global_fit`, `GlobalFitResult` added |
| `scatterbrain/modeling/form_factors.py` | `gaussian_chain_pq`, `ellipsoid_pq`, `vesicle_pq` added |
| `scatterbrain/modeling/structure_factors.py` | `sticky_sphere_structure_factor`, `screened_coulomb_structure_factor` added |
| `scatterbrain/visualization_interactive.py` | `plot_iq_interactive`, `plot_guinier_interactive`, `plot_porod_interactive`, `plot_pr_interactive`, `plot_kratky_interactive` |
| `scatterbrain/reduction/gisaxs.py` | `GISAXSGeometry`, `extract_yoneda_cut`, `extract_inplane_cut`, `extract_outofplane_cut` |
| `scatterbrain/visualization.py` | `plot_gisaxs_map` added |
| `scatterbrain/data/` | `glassy_carbon_srm3600.dat` added |

---

## Phase 4 Completion Criteria

| Criterion | Status |
|-----------|--------|
| `desmear` recovers a synthetic slit-smeared curve within 2% | Pending: Task 4.1 |
| `calibrate_intensity` returns curve with `intensity_unit="cm^-1"` | Pending: Task 4.2 |
| `ScatteringExperiment.apply_to_all` processes all curves correctly | Pending: Task 4.3 |
| `KineticSeries.parameter_series` returns time-ordered Rg values | Pending: Task 4.3 |
| `global_fit` recovers shared radius within 1% on synthetic data | Pending: Task 4.4 |
| `plot_iq_interactive` returns a plotly Figure; raises ImportError without plotly | Pending: Task 4.5 |
| `gaussian_chain_pq` satisfies Guinier limit at low q | Pending: Task 4.6 |
| `ellipsoid_pq` recovers sphere at a==b within 1e-6 | Pending: Task 4.6 |
| `vesicle_pq` gives P(0)=1 for any valid geometry | Pending: Task 4.6 |
| `sticky_sphere_structure_factor` approaches hard sphere at large stickiness | Pending: Task 4.7 |
| `GISAXSGeometry.angles_to_q` gives correct q_y=0 at 2*theta_f=0 | Pending: Task 4.8 |
| `plot_gisaxs_map` produces a labeled 2D map | Pending: Task 4.8 |
| pytest >= 400 passing, 0 failing | Pending |
| Test coverage >= 85% (all non-reduction modules) | Pending |

---

## What is Explicitly Out of Scope for Phase 4

These belong to Phase 5 or are explicitly deferred:

- GUI (Qt, web-based, or otherwise)
- Machine learning for parameter estimation or particle shape classification
- Full DWBA modeling for GISAXS (thin film interference, effective medium)
- GIWAXS fiber texture analysis
- Bayesian fitting and Markov Chain Monte Carlo uncertainty quantification
- LIMS / instrument database integration
- Absolute calibration using primary standards (direct flux measurement)
- Time-resolved data beyond batch parameter tracking (e.g. rapid kinetics streaming)
- Pair potential inversion from S(q)
