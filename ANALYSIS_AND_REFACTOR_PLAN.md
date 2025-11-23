# ScatterBrain: Analysis and Refactoring Plan

**Date:** 2025-11-23
**Version:** 1.0
**Author:** Claude (Analysis Agent)

---

## Executive Summary

ScatterBrain is a Python library for SAXS/WAXS (Small-Angle/Wide-Angle X-ray Scattering) data analysis. The project has successfully completed **Phase 1** of development with a functional core architecture implementing basic 1D scattering analysis. This document analyzes the theoretical foundation, current achievements, identifies gaps, and proposes a comprehensive refactoring plan.

---

## 1. Theoretical Foundation

### 1.1 SAXS/WAXS Theory Implemented

ScatterBrain implements fundamental scattering theory for analyzing X-ray scattering data:

#### **Guinier Analysis**
- **Theory:** For small q-values (qRg < 1.3), globular particles follow:
  `I(q) = I(0) · exp(-(qRg)²/3)`
- **Linearization:** `ln[I(q)] = ln[I(0)] - (Rg²/3)·q²`
- **Parameters Extracted:**
  - Rg (Radius of Gyration): characteristic size of the scattering object
  - I(0): Forward scattering intensity (proportional to molecular weight)

#### **Porod Analysis**
- **Theory:** At high q-values, scattering from smooth interfaces follows:
  `I(q) ~ Kp · q^(-n) + Background`
- **Parameters:**
  - n: Porod exponent (typically 4 for smooth 3D interfaces, <4 for rough/fractal surfaces)
  - Kp: Porod constant (related to specific surface area)

#### **Form Factor Modeling**
- **Sphere Form Factor:** `P(q) = [3(sin(qR) - qR·cos(qR))/(qR)³]²`
- **Total Intensity:** `I(q) = Scale · P(q) + Background`
- Normalized such that P(q=0) = 1

### 1.2 Data Processing Theory
- **Background Subtraction:** Linear or constant background removal
- **Error Propagation:** Standard propagation through mathematical operations
- **Weighted Fitting:** Uses intensity errors for chi-squared minimization

---

## 2. Current Achievements

### 2.1 Core Architecture ✅

**Implemented Components:**

1. **Data Structures** (`scatterbrain/core.py`)
   - `ScatteringCurve1D`: Robust class for 1D scattering data
   - Attributes: q, intensity, error, metadata, units
   - Methods: copy(), slicing/indexing, unit conversion, serialization
   - **Quality:** Well-designed with ~490 LOC, comprehensive error handling

2. **I/O Module** (`scatterbrain/io.py`)
   - ASCII 1D data loading (CSV, DAT, TXT formats)
   - Flexible column mapping and delimiter handling
   - Metadata extraction from headers
   - **Quality:** Functional, handles common formats

3. **Analysis Module**
   - `guinier_fit()` (scatterbrain/analysis/guinier.py): 251 LOC
     - Automatic q-range selection with qRg limits
     - Manual q-range override capability
     - Linear regression with error estimation
     - Comprehensive validation and warnings
   - `porod_analysis()` (scatterbrain/analysis/porod.py): 213 LOC
     - Log-log fitting for exponent determination
     - Average Porod constant calculation
     - Flexible q-range selection

4. **Modeling Module**
   - `sphere_pq()` (scatterbrain/modeling/form_factors.py): 147 LOC
     - Analytical sphere form factor
     - Proper q=0 handling
     - Validated against theory
   - `fit_model()` (scatterbrain/modeling/fitting.py)
     - Generic fitting framework using scipy.optimize.curve_fit
     - Scale + Background + Model parameters
     - Fixed parameter support
     - Chi-squared calculation with error weighting

5. **Visualization Module** (`scatterbrain/visualization.py`)
   - `plot_iq()`: I(q) vs q with log/linear scales
   - `plot_guinier()`: Guinier plot (ln I vs q²)
   - `plot_porod()`: Porod plot (I·q⁴ vs q⁴)
   - Publication-quality matplotlib figures

6. **Utilities** (`scatterbrain/utils.py`)
   - q-unit conversion (nm⁻¹ ↔ Å⁻¹)
   - Custom exception classes

### 2.2 Testing Infrastructure ✅

- **Test Coverage:** 7 test modules with comprehensive unit tests
- **Test Files Located:** `tests/` directory
- **Test Data:** Synthetic and fixture data in `tests/test_data/`
- **Framework:** pytest with parametric testing

### 2.3 Documentation ✅

- **README.md:** Project overview, installation, planned features
- **Design_document.md:** 24KB comprehensive design specification
- **phase0.md:** Phase 0 completion documentation
- **Docstrings:** Google-style docstrings throughout codebase
- **Example Scripts:**
  - `examples/basic_usage.py`
  - `scatterbrain/examples/basic_workflow_example.py` (194 LOC, full pipeline demo)

### 2.4 Development Infrastructure ✅

- **Build System:** pyproject.toml with setuptools
- **Dependencies:** numpy, scipy, matplotlib, pandas
- **Code Quality Tools Configured:** black, flake8, isort, mypy
- **Version Control:** Git repository with clean history
- **License:** CC-BY-NC-SA-4.0

---

## 3. Identified Gaps

### 3.1 Critical Gaps

#### **A. Missing Core Functionality**

1. **Processing Module Incomplete**
   - `scatterbrain/processing/` exists but is mostly placeholder
   - Background subtraction not fully implemented
   - Missing: normalization, merging, smoothing, interpolation
   - No desmearing functionality

2. **Reduction Module Empty**
   - `scatterbrain/reduction/` is placeholder only
   - No 2D→1D reduction capabilities
   - Missing: azimuthal integration, detector corrections, masking

3. **Limited Form Factor Library**
   - Only sphere implemented
   - Missing: cylinder, ellipsoid, core-shell structures, sheet, rod
   - No polydispersity models

4. **No Structure Factors**
   - `scatterbrain/modeling/structure_factors.py` not implemented
   - Missing: hard sphere, Percus-Yevick, paracrystal models

5. **Incomplete Visualization**
   - Guinier/Porod plots partially implemented
   - Missing: residual plots, Kratky plots, p(r) plots
   - No interactive visualizations
   - No 2D image display

#### **B. Architecture Issues**

1. **Module Organization**
   - Inconsistent import structure
   - `visualization_renamed/` suggests refactoring in progress
   - Processing module structure undefined

2. **API Inconsistency**
   - Some functions return dicts, others return None on failure
   - Inconsistent parameter naming across modules
   - Mixed use of q_range tuple formats

3. **Error Handling**
   - Warnings used extensively but inconsistent patterns
   - Some functions return None, others raise exceptions
   - No custom exception hierarchy fully utilized

4. **Metadata Management**
   - Processing history tracked but not standardized
   - No provenance tracking for fitted parameters
   - Unit handling could be more robust

#### **C. Missing Advanced Features**

1. **Advanced Analysis**
   - No p(r) calculation (Inverse Fourier Transform)
   - No Kratky analysis
   - No correlation function analysis
   - No Zimm/Debye plots

2. **Global Fitting**
   - No multi-dataset fitting
   - No linked parameters across datasets
   - No contrast variation analysis

3. **WAXS Capabilities**
   - No peak fitting
   - No d-spacing calculations
   - No Scherrer equation implementation
   - No crystallinity analysis

4. **Data Management**
   - No `ScatteringExperiment` class for multi-dataset projects
   - No batch processing utilities
   - No data export beyond basic ASCII

### 3.2 Technical Debt

1. **Testing**
   - Dependencies not installed in test environment (numpy import errors)
   - No integration tests beyond unit tests
   - No performance benchmarking

2. **Documentation**
   - No tutorial notebooks (notebooks/ directory empty)
   - API reference not built/published
   - No developer contribution guide

3. **Performance**
   - No optimization or profiling conducted
   - Pure Python/NumPy (no Numba/Cython for hotspots)
   - No caching of expensive calculations

4. **CI/CD**
   - No GitHub Actions or CI configuration visible
   - No automated testing on push
   - No automated documentation builds

---

## 4. Better Architectures & Design Patterns

### 4.1 Recommended Architecture Improvements

#### **A. Result Objects Pattern**

**Current:** Functions return dictionaries with varying keys
```python
guinier_results = guinier_fit(curve)  # Returns Dict[str, Any]
print(guinier_results['Rg'])  # Prone to typos, no autocomplete
```

**Improved:** Use dataclasses/NamedTuples for type safety
```python
@dataclass
class GuinierResult:
    rg: float
    rg_err: float
    i0: float
    i0_err: float
    q_range: Tuple[float, float]
    r_squared: float
    metadata: Dict[str, Any]

    def to_dict(self) -> dict: ...
    def __str__(self) -> str: ...  # Nice summary

guinier_result = guinier_fit(curve)  # Returns GuinierResult
print(guinier_result.rg)  # Type-safe, autocomplete-friendly
```

**Benefits:**
- Type hints for IDE support
- Immutable results (frozen dataclass)
- Clear API contract
- Easy serialization

#### **B. Analysis Pipeline Pattern**

**Current:** Users chain function calls manually
```python
curve = load_ascii_1d(...)
guinier = guinier_fit(curve)
porod = porod_analysis(curve)
fit = fit_model(curve, ...)
```

**Improved:** Fluent API with method chaining
```python
analysis = (AnalysisPipeline(curve)
    .guinier(q_range=(0.01, 0.1))
    .porod(q_fraction=0.3)
    .fit_sphere(initial_radius=5.0)
    .plot_results()
    .save_report("analysis_report.pdf"))

# Access results
print(analysis.guinier_result.rg)
```

**Alternative:** Declarative pipeline configuration
```python
pipeline = Pipeline([
    GuinierAnalysis(qrg_max=1.3),
    PorodAnalysis(q_fraction=0.3),
    SphereModelFit(initial_params={'radius': 5.0}),
    ExportResults(format='json')
])
results = pipeline.run(curve)
```

#### **C. Model Registry Pattern**

**Current:** Direct function imports for models
```python
from scatterbrain.modeling.form_factors import sphere_pq, cylinder_pq
fit_model(curve, model_func=sphere_pq, ...)
```

**Improved:** Registry-based model management
```python
from scatterbrain.modeling import ModelRegistry

# Built-in models auto-registered
model = ModelRegistry.get('sphere')
fit_result = model.fit(curve, radius=5.0, scale=1e5)

# Easy to extend
@ModelRegistry.register('my_custom_model')
class MyModel(BaseFormFactor):
    parameters = ['param1', 'param2']
    def compute(self, q, param1, param2): ...
```

**Benefits:**
- Plugin architecture for user models
- Metadata attached to models (citations, parameter bounds)
- Easier testing and validation

#### **D. Processing Chain Pattern**

**Current:** No clear processing abstraction
```python
# Hypothetical current approach
curve_bg = subtract_background(curve, bg=10)
curve_norm = normalize(curve_bg, ...)
```

**Improved:** Composable processing operations
```python
from scatterbrain.processing import ProcessingChain, SubtractBackground, Normalize

chain = ProcessingChain([
    SubtractBackground(method='constant', value=10),
    Normalize(method='concentration', concentration=1.5),
    SmoothData(method='savgol', window=5)
])

processed_curve = chain.apply(raw_curve)
# Metadata tracks all operations
print(processed_curve.metadata['processing_history'])
```

**Benefits:**
- Reproducible processing
- Easy to serialize/deserialize pipelines
- Undo/redo capabilities
- Clear provenance tracking

#### **E. Unit-Aware Calculations**

**Current:** Manual unit tracking in metadata
```python
curve.q_unit = "nm^-1"
converted = curve.convert_q_unit("A^-1")
```

**Improved:** Use `pint` library for physical quantities
```python
import pint
ureg = pint.UnitRegistry()

curve.q = ureg.Quantity(q_array, 'nm^-1')
curve_A = curve.with_q_unit('angstrom^-1')  # Automatic conversion
# Units propagate through calculations
rg = guinier_fit(curve).rg  # Has units attached
print(f"Rg = {rg:.2f~P}")  # "Rg = 5.0 nm"
```

### 4.2 Suggested Dependencies/Libraries

1. **`lmfit`** instead of `scipy.optimize.curve_fit`
   - Better parameter handling (bounds, fixing, expressions)
   - Built-in confidence intervals
   - More robust fitting algorithms

2. **`pint`** for unit handling
   - Physical quantity support
   - Automatic unit conversion and validation
   - Dimensional analysis

3. **`xarray`** for multi-dimensional data
   - Better than nested dicts for multi-dataset management
   - NetCDF/HDF5 integration
   - Labeled arrays with metadata

4. **`plotly`** or `bokeh`** for interactive visualization
   - Zoom, pan, hover tooltips
   - Export to HTML
   - Better for exploratory analysis

5. **`pydantic`** for configuration validation
   - Type validation for parameters
   - JSON schema generation
   - Better than plain dataclasses for user input

6. **`numba`** for performance-critical code
   - JIT compilation of tight loops
   - Minimal code changes
   - Significant speedup for form factor calculations

---

## 5. Gap Analysis Summary

### 5.1 Feature Completeness Matrix

| Feature Category | Planned | Implemented | Tested | Documented | Priority |
|-----------------|---------|-------------|---------|------------|----------|
| **Core Data Structures** | ✓ | ✓ | ✓ | ✓ | ✓✓✓ |
| **1D Data I/O** | ✓ | ✓ | ✓ | ✓ | ✓✓✓ |
| **Guinier Analysis** | ✓ | ✓ | ✓ | ✓ | ✓✓✓ |
| **Porod Analysis** | ✓ | ✓ | ✓ | ✓ | ✓✓✓ |
| **Sphere Model Fitting** | ✓ | ✓ | ✓ | ✓ | ✓✓✓ |
| **Basic Visualization** | ✓ | ✓ | ✓ | ✓ | ✓✓✓ |
| **Background Subtraction** | ✓ | ✗ | ✗ | ✗ | ✓✓✓ |
| **Data Normalization** | ✓ | ✗ | ✗ | ✗ | ✓✓ |
| **Additional Form Factors** | ✓ | ✗ | ✗ | ✗ | ✓✓✓ |
| **Structure Factors** | ✓ | ✗ | ✗ | ✗ | ✓✓ |
| **p(r) Calculation** | ✓ | ✗ | ✗ | ✗ | ✓✓ |
| **2D Data I/O** | ✓ | ✗ | ✗ | ✗ | ✓ |
| **Azimuthal Integration** | ✓ | ✗ | ✗ | ✗ | ✓ |
| **Global Fitting** | ✓ | ✗ | ✗ | ✗ | ✓ |
| **Interactive Plots** | ✓ | ✗ | ✗ | ✗ | ✓ |

**Legend:** ✓✓✓ Critical | ✓✓ Important | ✓ Nice-to-have

---

## 6. Detailed Refactoring Plan

### Phase 2: Enhance Core Functionality (8-10 weeks)

#### **Sprint 2.1: Complete Processing Module** (2 weeks)

**Objectives:**
1. Implement background subtraction methods
2. Add normalization functions
3. Create processing pipeline infrastructure

**Tasks:**
```
[ ] Design ProcessingOperation base class
[ ] Implement SubtractBackground
    [ ] Constant background
    [ ] Linear background
    [ ] Curve-based background
[ ] Implement Normalize
    [ ] By concentration
    [ ] By transmission
    [ ] By thickness
[ ] Implement SmoothData (Savitzky-Golay, moving average)
[ ] Implement InterpolateData (linear, cubic spline)
[ ] Create ProcessingChain class for composable operations
[ ] Write unit tests for all processing functions
[ ] Write integration tests for processing chains
[ ] Document with example notebooks
```

**Files to Create/Modify:**
- `scatterbrain/processing/base.py` (new)
- `scatterbrain/processing/background.py` (new)
- `scatterbrain/processing/normalize.py` (new)
- `scatterbrain/processing/smooth.py` (new)
- `tests/test_processing.py` (expand)

#### **Sprint 2.2: Expand Form Factor Library** (2 weeks)

**Objectives:**
1. Implement 5+ additional form factors
2. Create unified form factor interface
3. Add polydispersity support

**Tasks:**
```
[ ] Design BaseFormFactor abstract class
[ ] Implement form factors:
    [ ] Cylinder (with orientation average)
    [ ] Ellipsoid (prolate/oblate)
    [ ] Core-shell sphere
    [ ] Rectangular parallelepiped
    [ ] Flexible cylinder (Debye function)
[ ] Implement Schulz polydispersity
[ ] Implement Gaussian polydispersity
[ ] Create ModelRegistry class
[ ] Add model metadata (parameter bounds, citations)
[ ] Write comprehensive tests for each model
[ ] Validate against literature/SasView
[ ] Document with example fits
```

**Files to Create/Modify:**
- `scatterbrain/modeling/base.py` (new)
- `scatterbrain/modeling/form_factors.py` (expand)
- `scatterbrain/modeling/polydispersity.py` (new)
- `scatterbrain/modeling/registry.py` (new)
- `tests/test_form_factors.py` (expand)

#### **Sprint 2.3: Refactor Analysis Results** (1 week)

**Objectives:**
1. Replace dict returns with typed result objects
2. Improve API consistency
3. Add result serialization

**Tasks:**
```
[ ] Create results module: scatterbrain/results.py
[ ] Define result dataclasses:
    [ ] GuinierResult
    [ ] PorodResult
    [ ] ModelFitResult
    [ ] AnalysisResult (container for multiple)
[ ] Refactor guinier_fit() to return GuinierResult
[ ] Refactor porod_analysis() to return PorodResult
[ ] Refactor fit_model() to return ModelFitResult
[ ] Add to_dict() and from_dict() to all result classes
[ ] Add __str__() for nice summaries
[ ] Update all tests
[ ] Update examples and documentation
```

**Files to Create/Modify:**
- `scatterbrain/results.py` (new)
- `scatterbrain/analysis/guinier.py` (modify)
- `scatterbrain/analysis/porod.py` (modify)
- `scatterbrain/modeling/fitting.py` (modify)
- All test files (update)

#### **Sprint 2.4: Advanced Analysis Methods** (2 weeks)

**Objectives:**
1. Implement p(r) calculation
2. Add Kratky analysis
3. Improve automatic q-range selection

**Tasks:**
```
[ ] Implement Indirect Fourier Transform (IFT) for p(r)
    [ ] Moore's method
    [ ] Regularization parameter selection
[ ] Create pr_analysis() function
[ ] Implement Kratky analysis
[ ] Implement dimensionless Kratky plot
[ ] Improve Guinier auto q-range with iterative refinement
[ ] Add quality metrics for Guinier/Porod fits
[ ] Write tests with synthetic data
[ ] Validate against GNOM/ATSAS tools
[ ] Create example notebook demonstrating p(r) workflow
```

**Files to Create/Modify:**
- `scatterbrain/analysis/ift.py` (new)
- `scatterbrain/analysis/kratky.py` (new)
- `scatterbrain/analysis/guinier.py` (improve)
- `tests/test_ift.py` (new)
- `notebooks/pr_analysis_tutorial.ipynb` (new)

#### **Sprint 2.5: Enhance Visualization** (1.5 weeks)

**Objectives:**
1. Complete Guinier/Porod plot implementations
2. Add residual plots
3. Create plot presets for publications

**Tasks:**
```
[ ] Complete plot_guinier() with fit overlay
[ ] Complete plot_porod() with fit overlay
[ ] Implement plot_fit() with residuals
[ ] Implement plot_kratky()
[ ] Implement plot_pr()
[ ] Add plot style presets (publication, presentation, notebook)
[ ] Create PlotManager class for multi-panel figures
[ ] Add export presets (PNG 300dpi, SVG, PDF)
[ ] Write visualization tests (image comparison)
[ ] Create gallery notebook with all plot types
```

**Files to Create/Modify:**
- `scatterbrain/visualization.py` (expand significantly)
- `scatterbrain/plotting/` (new submodule?)
- `tests/test_visualization.py` (expand)
- `notebooks/visualization_gallery.ipynb` (new)

---

### Phase 3: Advanced Features (6-8 weeks)

#### **Sprint 3.1: lmfit Integration** (1.5 weeks)

**Objectives:**
1. Replace scipy.optimize.curve_fit with lmfit
2. Improve parameter handling
3. Add confidence intervals

**Tasks:**
```
[ ] Add lmfit to dependencies
[ ] Create LMFitModel wrapper class
[ ] Refactor fit_model() to use lmfit
[ ] Add parameter constraints (min, max, expr)
[ ] Add parameter fixing with expr syntax
[ ] Implement confidence interval calculation
[ ] Add fit report generation
[ ] Update all fitting tests
[ ] Update documentation with new API
```

**Files to Create/Modify:**
- `scatterbrain/modeling/fitting.py` (major refactor)
- `pyproject.toml` (add lmfit dependency)
- `tests/test_modeling.py` (update)

#### **Sprint 3.2: Structure Factors** (2 weeks)

**Objectives:**
1. Implement basic structure factor models
2. Integrate S(q) into fitting framework
3. Add combined P(q)·S(q) fitting

**Tasks:**
```
[ ] Design BaseStructureFactor class
[ ] Implement structure factors:
    [ ] Hard sphere (Percus-Yevick)
    [ ] Hayter-Penfold MSA (charged spheres)
    [ ] Sticky hard sphere
[ ] Extend fit_model() for S(q) combination
[ ] Add decoupling approximation
[ ] Write tests with known analytical limits
[ ] Validate against SasView
[ ] Document theory and usage
```

**Files to Create/Modify:**
- `scatterbrain/modeling/structure_factors.py` (implement)
- `scatterbrain/modeling/fitting.py` (extend)
- `tests/test_structure_factors.py` (new)

#### **Sprint 3.3: Multi-Dataset Management** (1.5 weeks)

**Objectives:**
1. Create ScatteringExperiment class
2. Implement batch processing utilities
3. Add contrast variation support

**Tasks:**
```
[ ] Design ScatteringExperiment class
    [ ] Container for multiple curves
    [ ] Shared metadata (sample, conditions)
    [ ] Methods: add, remove, filter curves
[ ] Implement batch analysis functions
[ ] Add global fitting framework
[ ] Implement contrast variation analysis
[ ] Create project save/load (JSON, HDF5)
[ ] Write tests for multi-dataset operations
[ ] Document with example notebook
```

**Files to Create/Modify:**
- `scatterbrain/core.py` (add ScatteringExperiment)
- `scatterbrain/batch.py` (new)
- `scatterbrain/io.py` (extend for HDF5)
- `notebooks/multi_dataset_analysis.ipynb` (new)

#### **Sprint 3.4: Interactive Visualization** (1.5 weeks)

**Objectives:**
1. Add plotly backend for interactive plots
2. Create interactive fit parameter exploration
3. Add hover tooltips with metadata

**Tasks:**
```
[ ] Add plotly as optional dependency
[ ] Create interactive plotting module
[ ] Implement interactive_plot_iq()
[ ] Add interactive Guinier region selector
[ ] Add interactive fit parameter sliders
[ ] Create dashboard for exploration
[ ] Write examples for Jupyter notebook usage
[ ] Document interactive features
```

**Files to Create/Modify:**
- `scatterbrain/interactive/` (new submodule)
- `pyproject.toml` (add plotly to optional deps)
- `notebooks/interactive_fitting.ipynb` (new)

---

### Phase 4: Production Readiness (4-6 weeks)

#### **Sprint 4.1: Performance Optimization** (2 weeks)

**Objectives:**
1. Profile code for bottlenecks
2. Optimize hot paths with Numba
3. Add caching for expensive operations

**Tasks:**
```
[ ] Profile analysis and fitting code
[ ] Identify hotspots (likely form factor calculations)
[ ] Apply @numba.jit to form factor functions
[ ] Implement caching for repeated calculations
[ ] Add optional parallel processing for batch ops
[ ] Benchmark improvements
[ ] Document performance tips
```

#### **Sprint 4.2: CI/CD Pipeline** (1 week)

**Objectives:**
1. Set up GitHub Actions
2. Automate testing and builds
3. Add code coverage reporting

**Tasks:**
```
[ ] Create .github/workflows/tests.yml
    [ ] Matrix testing (Python 3.10, 3.11, 3.12)
    [ ] Run pytest with coverage
    [ ] Upload to codecov.io
[ ] Create .github/workflows/docs.yml
    [ ] Build Sphinx docs
    [ ] Deploy to GitHub Pages
[ ] Create .github/workflows/release.yml
    [ ] Build wheels
    [ ] Publish to PyPI (on tag)
[ ] Add status badges to README
```

#### **Sprint 4.3: Documentation Polish** (1.5 weeks)

**Objectives:**
1. Build comprehensive Sphinx docs
2. Write 5+ tutorial notebooks
3. Create contribution guide

**Tasks:**
```
[ ] Set up Sphinx with RTD theme
[ ] Write user guide sections:
    [ ] Installation and quickstart
    [ ] Data loading and processing
    [ ] Analysis workflow
    [ ] Model fitting
    [ ] Visualization
    [ ] Advanced topics
[ ] Create tutorial notebooks:
    [ ] Basic workflow
    [ ] Guinier and Porod analysis
    [ ] Form factor fitting
    [ ] p(r) analysis
    [ ] Multi-dataset analysis
[ ] Write API reference (auto-generated)
[ ] Create CONTRIBUTING.md
[ ] Set up ReadTheDocs hosting
```

#### **Sprint 4.4: Package Polish & v0.1.0 Release** (0.5 weeks)

**Objectives:**
1. Final testing and bug fixes
2. Prepare for first public release
3. Write release notes

**Tasks:**
```
[ ] Final code review and cleanup
[ ] Ensure test coverage >90%
[ ] Write CHANGELOG.md
[ ] Update README with installation instructions
[ ] Tag v0.1.0 release
[ ] Publish to PyPI
[ ] Announce release
```

---

## 7. Architecture Refactoring Checklist

### 7.1 High-Priority Refactors

```
[ ] Replace dict returns with typed result objects (Sprint 2.3)
[ ] Create ProcessingChain pattern for data operations (Sprint 2.1)
[ ] Implement ModelRegistry for extensible models (Sprint 2.2)
[ ] Integrate lmfit for improved fitting (Sprint 3.1)
[ ] Create ScatteringExperiment for multi-dataset management (Sprint 3.3)
[ ] Standardize error handling (create exception hierarchy)
[ ] Add pydantic for configuration validation
[ ] Implement unit-aware calculations with pint
```

### 7.2 Code Quality Improvements

```
[ ] Achieve >90% test coverage
[ ] Add type hints throughout (currently sparse)
[ ] Run mypy in strict mode and fix issues
[ ] Add docstring examples for all public functions
[ ] Create benchmark suite
[ ] Add logging framework (replace print statements)
[ ] Standardize parameter naming conventions
```

### 7.3 Module Reorganization

**Current Structure:**
```
scatterbrain/
├── analysis/
│   ├── guinier.py
│   └── porod.py
├── modeling/
│   ├── fitting.py
│   └── form_factors.py
├── processing/  (empty)
├── reduction/   (empty)
└── ...
```

**Proposed Structure:**
```
scatterbrain/
├── core/
│   ├── curve.py          (ScatteringCurve1D)
│   ├── experiment.py     (ScatteringExperiment)
│   └── results.py        (result dataclasses)
├── io/
│   ├── ascii.py
│   ├── images.py         (2D data)
│   └── project.py        (HDF5/JSON)
├── processing/
│   ├── base.py
│   ├── background.py
│   ├── normalize.py
│   └── chains.py
├── analysis/
│   ├── guinier.py
│   ├── porod.py
│   ├── kratky.py
│   └── ift.py
├── modeling/
│   ├── base.py
│   ├── registry.py
│   ├── form_factors/
│   │   ├── spheres.py
│   │   ├── cylinders.py
│   │   └── ...
│   ├── structure_factors.py
│   ├── polydispersity.py
│   └── fitting.py
├── visualization/
│   ├── static.py
│   ├── interactive.py
│   └── presets.py
├── utils/
│   ├── units.py
│   ├── constants.py
│   └── validation.py
└── batch/
    └── pipeline.py
```

---

## 8. Success Metrics

### 8.1 Technical Metrics

- **Test Coverage:** >90% line coverage
- **Performance:** Form factor calculations <1ms for 1000 points
- **API Stability:** Semantic versioning, deprecation warnings
- **Documentation:** All public API documented, 10+ tutorial notebooks

### 8.2 User Experience Metrics

- **Ease of Use:** Common workflow in <10 lines of code
- **Error Messages:** Clear, actionable error messages
- **Flexibility:** Support 90% of use cases without custom code
- **Reproducibility:** All operations logged and reproducible

### 8.3 Community Metrics

- **Contributors:** 3+ external contributors by v1.0
- **Citations:** Used in 5+ publications
- **Downloads:** 1000+ PyPI downloads/month
- **GitHub Stars:** 50+ stars

---

## 9. Recommended Next Steps

### Immediate Actions (This Sprint)

1. **Set up development environment properly**
   ```bash
   # Fix test environment
   pip install -e ".[dev]"
   pytest  # Should pass all tests
   ```

2. **Create Sprint 2.1 branch and start processing module**
   ```bash
   git checkout -b feature/processing-module
   ```

3. **Begin with ProcessingOperation base class**
   - Define interface
   - Write skeleton implementations
   - Write first tests

### Week 1-2 Focus

- Complete Sprint 2.1 (Processing Module)
- Draft result dataclasses for Sprint 2.3
- Create ModelRegistry design doc

### Month 1 Goal

- Complete all Sprint 2 objectives (Phase 2)
- Have 15+ form factors implemented
- Processing pipeline functional
- Typed result objects throughout

### Quarter 1 Goal

- Complete Phase 2 and Phase 3
- lmfit integration complete
- Interactive visualization working
- Documentation at 80%

### Release v0.1.0 Target

- 6 months from now
- All Phase 2-4 features complete
- Published on PyPI
- Documentation on ReadTheDocs
- First community contributions

---

## 10. Conclusion

ScatterBrain has a **solid Phase 1 foundation** with well-designed core architecture, comprehensive testing, and clear documentation. The theoretical implementation is sound for basic SAXS analysis.

**Key Strengths:**
- Clean, modular architecture
- Comprehensive docstrings and documentation
- Thoughtful design following best practices
- Good test coverage for implemented features

**Critical Next Steps:**
1. Complete processing module (highest priority)
2. Expand form factor library
3. Refactor to typed result objects
4. Add advanced analysis (p(r), Kratky)

**Timeline to v0.1.0:** ~6 months with focused development

The project is **well-positioned** to become a valuable tool for the SAXS/WAXS community with consistent execution of this refactoring plan.

---

**Document Version:** 1.0
**Next Review:** After completion of Phase 2
**Contact:** Johannes Poms (Johannes.Poms@tugraz.at)
