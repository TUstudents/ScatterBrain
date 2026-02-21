# ScatterBrain -- Phase 5 Plan: Expert Analysis, ML Assistance, and GUI

**Version:** 1.0
**Branch:** `main`
**Reference:** `Design_document.md` sec.2.2 and sec.12, `PHASE4_PLAN.md`

---

## Preface

Phase 5 represents the long-term completion of the library vision described in
`Design_document.md`. Implementation specs here are more high-level than
in Phases 1-4 because the exact requirements will be refined as earlier phases
complete and the user community matures. Each task is still scoped with enough
detail to plan work, but should be re-reviewed before implementation begins.

Phase 5 assumes Phase 4 is complete: desmearing, absolute calibration,
ScatteringExperiment container, global fitting, plotly interactive plots,
extended form factors and structure factors, and basic GISAXS geometry.

---

## Goal

Phase 5 reaches the full long-term vision of the design document:

1. DWBA modeling for GISAXS (thin film interference, Fresnel amplitudes)
2. GIWAXS texture analysis (fiber orientation, Herman's parameter, pole figures)
3. Bayesian fitting via MCMC (credible intervals, posterior sampling)
4. Machine learning tools (shape classification, auto-parameterization)
5. Web-based interactive dashboard GUI (Panel)
6. NeXus / HDF5 standard file I/O for beamline integration
7. Advanced time-resolved analysis (kinetic modeling, SVD, component analysis)

---

## Task Overview

| Task | Component | Complexity | New Dependencies |
|------|-----------|------------|-----------------|
| 5.1 | DWBA GISAXS modeling | Very high | none |
| 5.2 | GIWAXS fiber texture | High | none |
| 5.3 | Bayesian fitting (MCMC) | High | emcee (new optional) |
| 5.4 | ML-assisted analysis | High | scikit-learn (new optional) |
| 5.5 | Web GUI (Panel) | Very high | panel, param (new optional) |
| 5.6 | NeXus / HDF5 I/O | Moderate | h5py (already optional from Phase 4) |
| 5.7 | Advanced time-resolved | High | none |

---

## Tasks

### 5.1 -- DWBA Modeling for GISAXS

**Scientific context:** The GISAXS geometry from Phase 4 (task 4.8) treats
scattering in the Born approximation (BA). For samples near their critical
angle, refraction and specular reflection at the substrate are significant
and the Born approximation fails. The Distorted Wave Born Approximation
(DWBA) accounts for these effects and is the standard framework for
quantitative GISAXS modeling.

**Theory (two-layer system: vacuum / substrate):**

The Fresnel reflection and transmission amplitudes at a flat interface for
s-polarization (relevant for SAXS) depend on the normal wavevector components:

```
k_z_vac = (2*pi/lambda) * sin(alpha)
k_z_sub = sqrt(k_z_vac^2 - k_c^2)   where k_c = (2*pi/lambda)*sin(alpha_c)
                                       and alpha_c = sqrt(2*delta_sub)

T(alpha) = 2*k_z_vac / (k_z_vac + k_z_sub)
R(alpha) = (k_z_vac - k_z_sub) / (k_z_vac + k_z_sub)
```

The DWBA scattering intensity from a monolayer of objects on the substrate is:

```
I_DWBA(q_y, q_z) propto
  |T(alpha_i)|^2 * |T(alpha_f)|^2 * |F(q_y, q_z^+)|^2
+ |T(alpha_i)|^2 * |R(alpha_f)|^2 * |F(q_y,-q_z^-)|^2
+ |R(alpha_i)|^2 * |T(alpha_f)|^2 * |F(q_y, q_z^-)|^2
+ |R(alpha_i)|^2 * |R(alpha_f)|^2 * |F(q_y,-q_z^+)|^2
```

where F(q_y, q_z) is the 3D Fourier transform of the object's scattering
length density, and:

```
q_z^+ = k_z_sub(alpha_f) + k_z_sub(alpha_i)
q_z^- = k_z_sub(alpha_f) - k_z_sub(alpha_i)
```

For objects partially embedded in the film, additional terms appear from the
film/substrate interface; a three-layer model (vacuum / film / substrate)
is specified below.

**Three-layer extension:**

For a thin film of thickness d_film and SLD delta_film on a substrate:
- Two interfaces: vacuum/film and film/substrate.
- Fabry-Perot interference oscillations in alpha_f (Kiessig fringes).
- The transmission and reflection amplitudes for both interfaces must be
  combined via the transfer matrix method.

```
T_total = T_01 * T_12 / (1 + R_01 * R_12 * exp(2*i*k_z_film*d_film))
R_total = (R_01 + R_12 * exp(2*i*k_z_film*d_film)) / (...)
```

**Specification:**

New file `scatterbrain/reduction/dwba.py`:

```python
class SubstrateModel:
    """Optical model for the substrate stack (vacuum / [film /] substrate)."""

    substrate_delta: float          # SLD decrement of substrate
    substrate_beta: float           # absorption of substrate
    film_delta: Optional[float]     # SLD decrement of film (None = no film)
    film_beta: Optional[float]
    film_thickness: Optional[float] # nm


class DWBAResult(TypedDict):
    intensity_2d: np.ndarray    # 2D DWBA intensity map, shape (n_qz, n_qy)
    q_y: np.ndarray             # 1D q_y axis
    q_z: np.ndarray             # 1D q_z axis
    fresnel_Ti: np.ndarray      # Fresnel transmission at alpha_i, per q_z
    fresnel_Tf: np.ndarray      # Fresnel transmission at alpha_f, per q_z
    substrate_model: SubstrateModel
    alpha_i_deg: float


def compute_dwba_intensity(
    form_factor_2d: Callable[[np.ndarray, np.ndarray], np.ndarray],
    geometry: GISAXSGeometry,
    substrate: SubstrateModel,
    q_y_grid: np.ndarray,
    q_z_grid: np.ndarray,
    wavelength: float,
) -> DWBAResult
```

`form_factor_2d(q_y, q_z)` is a user-supplied function returning the 2D
object form factor amplitude F(q_y, q_z) (e.g., a cylinder standing on the
substrate, or a sphere partially embedded).

Provide built-in 2D form factor functions for common morphologies:

```python
def dwba_cylinder_form_factor(
    q_y: np.ndarray, q_z: np.ndarray,
    radius: float, height: float,
) -> np.ndarray

def dwba_truncated_sphere_form_factor(
    q_y: np.ndarray, q_z: np.ndarray,
    radius: float, truncation_factor: float,
) -> np.ndarray
```

**Files:**
- `scatterbrain/reduction/dwba.py` (new)
- `scatterbrain/reduction/__init__.py`

**Tests:** `tests/test_reduction.py`
- Fresnel T(alpha) -> 1 as alpha >> alpha_c (all transmitted in vacuum limit).
- At alpha_i = 0 (grazing), R -> -1, T -> 0.
- DWBA intensity with vacuum substrate (delta=0) should recover Born
  approximation result.
- Three-layer model: verify Kiessig fringe period matches d_film.

**Reference:** Renaud et al. (2009) Surface Science Reports 64, 255-380.

---

### 5.2 -- GIWAXS Fiber Texture Analysis

**Scientific context:** Organic semiconductors, block copolymers, and
semicrystalline polymers form films with preferred crystallographic orientation
(fiber texture). GIWAXS reveals this texture through the azimuthal distribution
of Bragg ring intensity. Quantifying orientation guides materials design.

**Specification:**

Extend `scatterbrain/reduction/gisaxs.py` with:

#### Pole Figure Extraction

```python
def extract_pole_figure(
    image: SAXSImage,
    geometry: GISAXSGeometry,
    q_center: float,
    q_width: float,
    chi_bins: int = 180,
    chi_range: Tuple[float, float] = (-90.0, 90.0),
) -> Tuple[np.ndarray, np.ndarray]   # (chi_deg, intensity)
```

Integrates intensity in the annulus `[q_center - q_width/2, q_center + q_width/2]`
at each azimuthal angle chi. Uses the GISAXSGeometry to compute q_y, q_z at each
pixel and then converts to q = sqrt(q_y^2 + q_z^2) and chi = atan2(q_z, q_y).

#### Orientation Analysis

```python
class TextureResult(TypedDict):
    chi_deg: np.ndarray              # azimuthal angles (degrees)
    pole_figure: np.ndarray          # I(chi)
    herman_orientation_factor: float # S = (3*<cos^2(chi)> - 1) / 2
    chi_max: float                   # chi at peak intensity (dominant orientation)
    fwhm_chi: float                  # angular width of orientation distribution
    orientation_label: str           # "isotropic" / "edge-on" / "face-on" / "tilted"


def analyze_texture(
    chi_deg: np.ndarray,
    intensity: np.ndarray,
    reference_direction: str = "out_of_plane",
) -> TextureResult
```

Herman's orientation parameter:

```
S = (3 * <cos^2(chi)> - 1) / 2

<cos^2(chi)> = integral I(chi) * cos^2(chi) * sin(chi) d_chi
               / integral I(chi) * sin(chi) d_chi
```

where the sin(chi) weighting accounts for the solid angle element in 3D.
For `reference_direction="out_of_plane"`, chi is measured from the substrate
normal (q_z axis).

- S = 1: perfect alignment along the reference direction.
- S = -0.5: perfect alignment perpendicular to the reference direction.
- S = 0: isotropic.

Classify orientation: `|S| < 0.1` -> isotropic; S > 0.5 -> edge-on;
S < -0.3 -> face-on; intermediate -> tilted.

Add `plot_pole_figure(chi_deg, intensity, texture_result=None, ax=None)`
to `scatterbrain/visualization.py`.

**Files:**
- `scatterbrain/reduction/gisaxs.py` (extend)
- `scatterbrain/visualization.py` (add `plot_pole_figure`)

**Tests:** `tests/test_reduction.py`
- Isotropic ring: verify S = 0 within numerical tolerance.
- Delta-function at chi=0: verify S = 1.
- Delta-function at chi=90 deg: verify S = -0.5.

---

### 5.3 -- Bayesian Parameter Estimation (MCMC)

**Scientific context:** Least-squares fitting gives a single best-fit
parameter vector and a covariance matrix that is only a valid approximation
to the uncertainty for linear models near a Gaussian posterior. For nonlinear
models (e.g., core-shell sphere with correlated radius and shell thickness),
the posterior is often non-Gaussian and the lmfit covariance can be misleading.
Markov Chain Monte Carlo sampling of the posterior gives exact credible
intervals and reveals parameter correlations.

**Method:** Affine-invariant ensemble sampler (Foreman-Mackey et al. 2013,
emcee). Reference implementation: https://emcee.readthedocs.io/

**Specification:**

```python
class BayesianFitResult(TypedDict):
    param_names: List[str]
    chain: np.ndarray               # shape (n_samples, n_params); after burn-in
    log_probability: np.ndarray     # shape (n_samples,)
    median: Dict[str, float]        # median of each parameter
    credible_interval_16: Dict[str, float]   # 16th percentile (1-sigma lower)
    credible_interval_84: Dict[str, float]   # 84th percentile (1-sigma upper)
    gelman_rubin: Dict[str, float]  # per-parameter; < 1.01 indicates convergence
    n_walkers: int
    n_steps: int
    burn_in: int
    acceptance_fraction: float      # fraction of accepted proposals (target ~0.25)


def bayesian_fit(
    curve: ScatteringCurve1D,
    model_func: Callable[..., np.ndarray],
    param_names: List[str],
    initial_params: Dict[str, float],
    priors: Dict[str, Tuple[float, float]],
    prior_types: Dict[str, str],
    structure_factor: Optional[Callable] = None,
    q_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    n_walkers: int = 32,
    n_steps: int = 3000,
    burn_in: int = 500,
    n_threads: int = 1,
) -> Optional[BayesianFitResult]
```

Prior types: `"uniform"` (bounds = (low, high)) or `"gaussian"` (bounds =
(mean, sigma)).

Log-posterior:

```
ln_P = ln_likelihood + sum_params ln_prior
ln_likelihood = -0.5 * sum_q [(I_obs - I_model)^2 / sigma^2 + ln(2*pi*sigma^2)]
```

When `curve.error` is None: use the homoscedastic approximation with sigma
as a free hyperparameter (Jeffrey's prior on sigma).

Convergence diagnostics:
- Gelman-Rubin statistic R_hat across independent walker groups.
- Integrated autocorrelation time tau; recommend `n_steps > 50 * tau`.
- Log a WARNING if R_hat > 1.05 for any parameter after burn-in.

Add `plot_corner(bayesian_result, truths=None, ...)` to `visualization.py`:
a corner (pairplot) figure showing 1D histograms on the diagonal and 2D
contour plots of the posterior in each parameter pair. Internally uses
matplotlib (not the `corner` package, to avoid a dependency).

Optional dependency:
```toml
[project.optional-dependencies]
mcmc = ["emcee>=3.1"]
```

**Files:**
- `scatterbrain/modeling/bayesian.py` (new)
- `scatterbrain/modeling/__init__.py`
- `scatterbrain/visualization.py` (add `plot_corner`)
- `pyproject.toml` (add `mcmc` optional extra)

**Tests:** `tests/test_modeling.py`
- On synthetic sphere data with known radius: verify median radius within 3%
  and true value inside the 16th-84th credible interval.
- Verify G-R statistic < 1.05 after sufficient steps on a simple 2-parameter
  problem.
- Without emcee: verify `ImportError` with install hint.

---

### 5.4 -- Machine Learning Assisted Analysis

**Scientific context:** Manual selection of analysis methods and initial fit
parameters is a bottleneck for high-throughput SAXS. Three targeted ML tools
accelerate the analysis pipeline without replacing rigorous numerical fitting.

**Tool 5.4a -- Shape Classifier**

Classifies the dominant morphology of a scattering curve from a set of
handcrafted features.

**Features extracted from I(q):**
- Porod exponent n (from Porod analysis)
- Presence and height of Kratky plateau (q^2 * I / I0 at q*Rg = sqrt(2))
- Dimensionless Rg * q_max
- Ratio I(q_low) / I(q_high) at fixed fraction of q range
- d(ln I)/d(ln q) at low, mid, and high q
- Whether a secondary minimum is visible in I(q)

**Classes:** `sphere`, `cylinder`, `disk`, `gaussian_chain`, `vesicle`,
`core_shell_sphere`, `unknown`.

**Model:** scikit-learn `RandomForestClassifier` trained on 10,000 synthetic
curves per class, generated from ScatterBrain's own form factors with varied
parameters, noise levels, and backgrounds.

```python
class ShapeClassifier:
    """Pre-trained random forest classifier for SAXS morphology."""

    def predict(self, curve: ScatteringCurve1D) -> str
    def predict_proba(self, curve: ScatteringCurve1D) -> Dict[str, float]
    def predict_batch(self, experiment: ScatteringExperiment) -> Dict[str, str]
```

The pre-trained model weights are bundled with the package as a pickle file
in `scatterbrain/data/shape_classifier_v1.pkl`.

**Tool 5.4b -- Initial Parameter Estimator**

Provides data-driven initial guesses for `fit_model` / `bayesian_fit`:

```python
def estimate_sphere_params(curve: ScatteringCurve1D) -> Dict[str, float]
    # Uses Guinier Rg and Rg->R conversion for a sphere: R = sqrt(5/3) * Rg
    # Estimates scale from I(0) / sphere_pq(0, R) = I(0) (since P(0)=1)
    # Estimates background from last 5% of q-range

def estimate_cylinder_params(curve: ScatteringCurve1D) -> Dict[str, float]
    # Estimates radius from Porod knee and length from Guinier Rg
    # Uses R_cyl = Rg_cross * sqrt(2) for the cross-section Rg

def estimate_core_shell_params(curve: ScatteringCurve1D) -> Dict[str, float]
    # Sequential Guinier + Porod + shoulder detection
```

**Tool 5.4c -- Outlier Detection**

Flags anomalous curves in a `ScatteringExperiment`:

```python
def detect_outliers(
    experiment: ScatteringExperiment,
    method: str = "pca",
    threshold: float = 3.0,
) -> Dict[str, bool]   # keyed by curve name; True = outlier
```

Methods:
- `"pca"`: PCA on the log-I(q) matrix interpolated to a common q grid;
  flag curves with reconstruction error > threshold * sigma.
- `"isolation_forest"`: scikit-learn `IsolationForest` on the same feature
  matrix.

Optional dependency:
```toml
[project.optional-dependencies]
ml = ["scikit-learn>=1.3"]
```

Training data generation is done in a separate script
`scripts/train_shape_classifier.py` (not bundled with the package; only the
trained model binary is bundled).

**Files:**
- `scatterbrain/ml/__init__.py` (new sub-package)
- `scatterbrain/ml/classifier.py` (ShapeClassifier)
- `scatterbrain/ml/estimators.py` (parameter estimators)
- `scatterbrain/ml/outliers.py` (OutlierDetector)
- `scatterbrain/data/shape_classifier_v1.pkl`
- `pyproject.toml` (add `ml` optional extra)

**Tests:** `tests/test_ml.py` (new)
- `ShapeClassifier.predict` on synthetic sphere: verify returns `"sphere"`.
- `estimate_sphere_params`: verify R within 10% of true value on ideal data.
- `detect_outliers`: inject one curve with added noise x10; verify it is flagged.
- Without scikit-learn: verify `ImportError` with install hint.

---

### 5.5 -- Web-Based Interactive Dashboard (Panel)

**Scientific context:** A significant portion of the SAXS user community
is not comfortable writing Python code. A browser-based GUI lowers the barrier
to entry and enables lab-level data QC workflows without programming.

**Method:** Panel (holoviz.org) for the dashboard framework, combined with
the plotly interactive plots from Phase 4 (task 4.5) for rendering.

**Scope:** The GUI is a wrapper around the library's existing API. It exposes
the most common single-curve workflow and does not attempt to replicate every
feature. Advanced users should use the Python API directly.

**Panels:**

1. **Load data:** Drag-and-drop or path entry to load one or more
   `ScatteringCurve1D` from ASCII. Shows a preview table (q range, n points,
   has_errors). Supports multi-file load into a `ScatteringExperiment`.

2. **Visualize:** Tabbed I(q) / Guinier / Porod / Kratky plots, with a curve
   selector. Uses plotly for zoom/hover. Error bars optional.

3. **Analysis:** Guinier and Porod analysis tabs with auto q-range or manual
   sliders. Shows result table (Rg, I0, Kp, n) and fit overlay.

4. **Model fitting:** Dropdown for form factor selection; sliders for initial
   parameter guesses; "Fit" button; shows fitted parameters, chi^2 reduced,
   and residuals panel.

5. **Export:** Save processed curves to ASCII; export analysis results to CSV;
   save figures as PNG/HTML.

**Entry point:**

```python
# pyproject.toml
[project.scripts]
scatterbrain-gui = "scatterbrain.gui:launch"
```

```python
# scatterbrain/gui/__init__.py
def launch(port: int = 5006, show: bool = True) -> None:
    """Launch the ScatterBrain Panel dashboard in the default browser."""
```

Also importable in Jupyter: `import scatterbrain.gui; scatterbrain.gui.app.servable()`.

Optional dependency:
```toml
[project.optional-dependencies]
gui = ["panel>=1.3", "param>=2.0", "plotly>=5.0"]
```

**Files:**
- `scatterbrain/gui/__init__.py` (new sub-package)
- `scatterbrain/gui/app.py` (main Panel application)
- `scatterbrain/gui/panels/load.py`, `visualize.py`, `analysis.py`,
  `fitting.py`, `export.py`
- `pyproject.toml`

**Tests:** `tests/test_gui.py`
- Import test: `from scatterbrain.gui import launch` succeeds with panel installed.
- Smoke test: instantiate the Panel app and verify it is a Panel object.
- Without panel: `ImportError` with install hint.
- Full UI interaction tests are deferred; they require a Playwright or
  Selenium fixture and are outside the pytest unit-test scope.

---

### 5.6 -- NeXus / HDF5 Standard File I/O

**Scientific context:** Major synchrotron and neutron facilities (ESRF, DLS,
NIST NCNR, PSI SLS) store raw and processed data in NeXus/HDF5 format.
Supporting NeXus I/O allows ScatterBrain to load beamline-reduced 1D curves
directly without manual column mapping, and to save results in a format
accepted by facility data repositories.

**NeXus application definition:** NXsas (1D SAXS/SANS).
Reference: https://manual.nexusformat.org/classes/applications/NXsas.html

**Key NeXus paths for 1D SAXS:**

```
/entry/
    @NX_class = NXentry
    instrument/
        @NX_class = NXinstrument
        beam/
            incident_wavelength    (Angstrom)
        detector/
            distance               (m or mm)
    sample/
        @NX_class = NXsample
        name
        thickness                  (m or mm)
        transmission
    data/
        @NX_class = NXdata
        I                          (1D array, cm^-1)
        Q                          (1D array, nm^-1 or A^-1)
        Idev                       (1D array; optional errors on I)
        Qdev                       (1D array; optional errors on Q)
```

**Specification:**

Add to `scatterbrain/io.py`:

```python
def load_nexus_1d(
    filepath: Union[str, Path],
    entry: str = "entry",
) -> ScatteringCurve1D
```

Reads I, Q, Idev from the NXsas structure. Reads wavelength and distance into
`metadata`. Raises `IOError` if the file is not a valid NeXus file or the
NXsas paths are missing. Raises `ImportError` with install hint if h5py is
absent (note: h5py is the same optional dependency introduced in Phase 4 for
`ScatteringExperiment.save`; here it becomes the same `storage` extra).

```python
def save_nexus_1d(
    curve: ScatteringCurve1D,
    filepath: Union[str, Path],
    entry: str = "entry",
    overwrite: bool = False,
) -> None
```

Writes a minimal NXsas-compliant file. Raises `FileExistsError` if the file
exists and `overwrite=False`.

**Facility-specific loaders:**

Many facilities produce NeXus-like HDF5 that deviates from the standard.
Provide thin wrappers:

```python
def load_esrf_biosaxs(filepath) -> ScatteringCurve1D
    # BM29 / ID02 reduced 1D format (h5 with non-standard paths)

def load_nist_sans(filepath) -> ScatteringCurve1D
    # NIST NCNR 6-column ASCII format (not NeXus; plain text)
```

The NIST SANS loader is a thin wrapper around `load_ascii_1d` with
column-order defaults specific to the NIST format. No h5py needed.

**Files:**
- `scatterbrain/io.py` (add `load_nexus_1d`, `save_nexus_1d`,
  `load_esrf_biosaxs`, `load_nist_sans`)
- `pyproject.toml` (h5py already in `storage` extra from Phase 4)

**Tests:** `tests/test_io.py`
- Round-trip `save_nexus_1d` / `load_nexus_1d`: verify q, I, error preserved.
- Missing NXsas paths: verify `IOError`.
- Without h5py: verify `ImportError`.
- NIST SANS loader: verify correct column mapping on a sample file.

---

### 5.7 -- Advanced Time-Resolved Analysis

**Scientific context:** Phase 4's `KineticSeries` tracks scalar parameters
(Rg, Kp) over time. Phase 5 adds tools for the full I(q,t) matrix: kinetic
model fitting, noise reduction via singular value decomposition, and spectral
component decomposition to separate coexisting scattering species.

#### 5.7a -- Kinetic Model Fitting

```python
class KineticModelResult(TypedDict):
    param_names: List[str]
    fitted_params: Dict[str, float]
    fitted_params_stderr: Dict[str, float]
    times_fit: np.ndarray
    param_fit: np.ndarray
    chi_squared_reduced: float
    model: str


def fit_kinetic_model(
    times: np.ndarray,
    param_values: np.ndarray,
    model: str,
    initial_params: Optional[Dict[str, float]] = None,
) -> Optional[KineticModelResult]
```

Supported models:
- `"exponential"`: `P(t) = P_inf + (P_0 - P_inf) * exp(-t / tau)`
- `"biexponential"`: two exponential components with amplitudes A1, A2 and rates k1, k2
- `"power_law"`: `P(t) = A * t^alpha`
- `"sigmoidal"`: `P(t) = P_max / (1 + exp(-k * (t - t0)))`
- `"avrami"`: `P(t) = P_inf * (1 - exp(-K * t^n))` (crystallization kinetics)

Uses lmfit (already from Phase 2). Provides a `plot_kinetics` function in
`visualization.py`.

#### 5.7b -- SVD Denoising and Component Analysis

```python
def svd_analysis(
    experiment: KineticSeries,
    q_grid: Optional[np.ndarray] = None,
    n_components: Optional[int] = None,
) -> SVDResult


class SVDResult(TypedDict):
    U: np.ndarray           # left singular vectors (spectral), shape (n_q, n_sv)
    S: np.ndarray           # singular values, shape (n_sv,)
    Vt: np.ndarray          # right singular vectors (time), shape (n_sv, n_t)
    n_significant: int      # estimated number of significant components (by scree test)
    q_grid: np.ndarray
    reconstructed: KineticSeries  # denoised dataset using n_significant components
```

The scree test for `n_significant`: log(S_i) vs i; find the "elbow" by
largest gap in log-singular-value spectrum.

#### 5.7c -- Multivariate Curve Resolution (MCR-ALS)

For a mixture of N components evolving over time, MCR-ALS decomposes the
I(q, t) matrix into spectral profiles and concentration profiles:

```
I(q, t) ~ sum_k c_k(t) * S_k(q)
```

subject to non-negativity constraints on both c_k(t) and S_k(q).

```python
def mcr_als(
    experiment: KineticSeries,
    n_components: int,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
    initial_spectra: Optional[np.ndarray] = None,
) -> MCRResult


class MCRResult(TypedDict):
    spectra: np.ndarray         # shape (n_components, n_q)
    concentrations: np.ndarray  # shape (n_components, n_t)
    residual: float             # final relative residual norm
    n_iterations: int
    converged: bool
    q_grid: np.ndarray
    times: np.ndarray
```

Uses alternating least squares with non-negativity projection at each step.
No new dependencies (standard numpy operations).

**Files:**
- `scatterbrain/analysis/kinetics.py` (new: `fit_kinetic_model`, `KineticModelResult`)
- `scatterbrain/analysis/svd.py` (new: `svd_analysis`, `SVDResult`, `mcr_als`, `MCRResult`)
- `scatterbrain/analysis/__init__.py`
- `scatterbrain/visualization.py` (add `plot_kinetics`, `plot_svd_scree`,
  `plot_mcr_components`)

**Tests:** `tests/test_analysis.py`
- `fit_kinetic_model` with `"exponential"` on synthetic data: verify tau within 1%.
- `svd_analysis` on a 2-component mixture: verify `n_significant = 2`.
- `mcr_als` on a 2-component synthetic mixture: verify recovered spectra
  within 5% of true component spectra.

---

## Priority Order

```
5.1  DWBA GISAXS        -- completes the Phase 4 GISAXS geometry; builds on SAXSImage
5.2  GIWAXS texture     -- natural extension of 5.1; independent of Bayesian/ML
5.3  Bayesian fitting   -- scientifically critical; requires only emcee
5.6  NeXus I/O          -- beamline integration; moderate complexity; h5py already optionally available
5.7  Time-resolved      -- builds on Phase 4 KineticSeries; no new dependencies
5.4  ML tools           -- requires training pipeline; scikit-learn optional dep
5.5  Web GUI            -- highest complexity; requires panel + all earlier features stable
```

Tasks 5.2, 5.3, 5.6, and 5.7 are independent of each other and can be
developed in parallel. Task 5.1 (DWBA) requires the Phase 4 SAXSImage and
GISAXSGeometry. Task 5.2 builds on 5.1. Task 5.5 (GUI) requires all other
Phase 5 features to be stable before the GUI can expose them, so it should
be started last.

---

## New Module Layout After Phase 5

| Module | New API |
|--------|---------|
| `scatterbrain/reduction/dwba.py` | `SubstrateModel`, `DWBAResult`, `compute_dwba_intensity`, `dwba_cylinder_form_factor`, `dwba_truncated_sphere_form_factor` |
| `scatterbrain/reduction/gisaxs.py` | `extract_pole_figure`, `analyze_texture`, `TextureResult` added |
| `scatterbrain/modeling/bayesian.py` | `bayesian_fit`, `BayesianFitResult` |
| `scatterbrain/ml/__init__.py` | `ShapeClassifier`, `detect_outliers` |
| `scatterbrain/ml/estimators.py` | `estimate_sphere_params`, `estimate_cylinder_params`, `estimate_core_shell_params` |
| `scatterbrain/gui/__init__.py` | `launch` entry point |
| `scatterbrain/io.py` | `load_nexus_1d`, `save_nexus_1d`, `load_esrf_biosaxs`, `load_nist_sans` added |
| `scatterbrain/analysis/kinetics.py` | `fit_kinetic_model`, `KineticModelResult` |
| `scatterbrain/analysis/svd.py` | `svd_analysis`, `SVDResult`, `mcr_als`, `MCRResult` |
| `scatterbrain/visualization.py` | `plot_corner`, `plot_pole_figure`, `plot_kinetics`, `plot_svd_scree`, `plot_mcr_components` added |

---

## New Optional Extras After Phase 5

| Extra | New Dependencies | Enables |
|-------|-----------------|---------|
| `mcmc` | `emcee>=3.1` | `bayesian_fit` |
| `ml` | `scikit-learn>=1.3` | `ShapeClassifier`, `detect_outliers` |
| `gui` | `panel>=1.3`, `param>=2.0`, `plotly>=5.0` | `scatterbrain-gui` entry point |
| `storage` | `h5py>=3.7` | NeXus I/O, `ScatteringExperiment.save` (already from Phase 4) |

---

## Phase 5 Completion Criteria

| Criterion | Status |
|-----------|--------|
| `compute_dwba_intensity` recovers Born approximation when substrate delta=0 | Pending: Task 5.1 |
| Fresnel T + R amplitudes are self-consistent at all angles | Pending: Task 5.1 |
| Herman's orientation factor S=0 for isotropic ring | Pending: Task 5.2 |
| `bayesian_fit` credible interval contains true radius on sphere data | Pending: Task 5.3 |
| G-R convergence statistic < 1.05 after adequate MCMC steps | Pending: Task 5.3 |
| `ShapeClassifier.predict` returns "sphere" for synthetic sphere curve | Pending: Task 5.4 |
| `detect_outliers` flags a 10x-noise-amplified curve | Pending: Task 5.4 |
| `scatterbrain-gui` entry point launches Panel app in browser | Pending: Task 5.5 |
| `load_nexus_1d` / `save_nexus_1d` round-trip preserves all data | Pending: Task 5.6 |
| `fit_kinetic_model` recovers tau within 1% on synthetic exponential data | Pending: Task 5.7 |
| `mcr_als` recovers 2-component spectra within 5% | Pending: Task 5.7 |
| `n_significant` from SVD equals 2 for a 2-component synthetic series | Pending: Task 5.7 |
| pytest >= 500 passing, 0 failing | Pending |

---

## What is Explicitly Out of Scope for Phase 5

These are either research-grade features beyond the library's scope or require
resources not available in an academic library project:

- **Ab initio shape reconstruction** (DAMMIN/GASBOR equivalent): requires
  simulated annealing on a bead model; specialist software (ATSAS) already
  exists and ScatterBrain should interface with it rather than reimplement it.
- **Pair potential inversion** (iterative Boltzmann inversion from S(q)):
  research-grade; no stable standard method.
- **Anomalous / resonant SAXS (ASAXS/RSAXS)**: requires multi-energy data
  and SLD dispersion corrections; specialist technique.
- **Contrast variation SAXS**: requires multiple solvent-matching conditions
  and joint analysis; specialist technique.
- **Cloud / HPC distributed computation**: ScatterBrain targets single-machine
  use; cluster integration is a deployment concern, not a library feature.
- **Desktop GUI (Qt / Tkinter)**: the Panel web app (task 5.5) is sufficient;
  a second Qt-based GUI adds maintenance burden without clear benefit.
- **LIMS / electronic lab notebook integration**: too facility-specific to
  implement generically; expose a clean API and let facilities write adapters.
- **Full Bayesian network / hierarchical Bayesian modeling**: requires PyMC or
  Stan; beyond the scope of a SAXS-specific library.
