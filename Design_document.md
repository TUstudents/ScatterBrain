Okay, here is a complete Design Document for the `ScatterBrain` Python library, synthesizing your `ScatterBrain` proposal and your "SOP for Developing Complex Python Libraries."

---

# `ScatterBrain`: Design Document

**Version:** 0.1
**Date:** October 26, 2023
**Author/AI Assistant:** Based on user prompts and SOP.

**Reference:** This document is developed in accordance with the "Standard Operating Procedure for Developing Complex Python Libraries and Knowledge Bases" (SOP_Python_libs.md).

## 1. Introduction

### 1.1. Project Name
`ScatterBrain`

### 1.2. Pun Intended
The name "ScatterBrain" is a playful reference to the physical phenomenon of X-ray scattering and the sometimes complex cognitive processes involved in analyzing the resulting data. While the library aims for robust and clear functionality, the name adds a touch of lightheartedness to the project.

### 1.3. Project Goal
To develop a comprehensive, user-friendly, and extensible Python library for loading, processing, analyzing, modeling, and visualizing Small-Angle X-ray Scattering (SAXS) and Wide-Angle X-ray Scattering (WAXS) data. The project will follow a structured, iterative, and well-documented approach.

### 1.4. Target Audience
Material scientists, chemists, physicists, and biologists who utilize SAXS/WAXS techniques for structural characterization of various materials, including (but not limited to) nanomaterials, polymers, proteins, and colloids. The library aims to be accessible to both experienced scattering practitioners and researchers newer to the field.

### 1.5. Guiding Principles
The development of `ScatterBrain` will adhere to the following principles, as outlined in the SOP:
1.  **Modularity:** Logical, manageable, testable components.
2.  **Iterative Development:** Start with a core, functional skeleton; incrementally add features.
3.  **Test-Supported Development:** Write tests concurrently with or immediately after implementation.
4.  **Clear Documentation:** Comprehensive user and developer documentation.
5.  **User-Centric Design:** Focus on the end-user's workflow and needs.
6.  **Transparency:** Clearly state assumptions, limitations, and the nature of implemented models/calculations.
7.  **Extensibility:** Design for future expansion and user contributions.

## 2. Scope

### 2.1. Initial Scope (Minimum Viable Product - MVP / First Major Release)
*   **Data Input:** Loading 1D SAXS/WAXS data from common text-based formats (e.g., `.dat`, `.txt`, `.csv` with $q, I(q), \text{error}$ columns).
*   **Core Data Structure:** A robust `ScatteringCurve1D` class to hold 1D data and essential metadata.
*   **Basic Processing:** Simple background subtraction (constant or curve).
*   **Core SAXS Analysis:**
    *   Guinier analysis for Radius of Gyration ($R_g$) and forward scattering $I(0)$.
    *   Porod analysis for Porod constant and exponent.
*   **Basic Modeling:** Fitting a simple spherical form factor.
*   **Visualization:** Static plots for $I(q)$ vs $q$, Guinier plots, Porod plots using `matplotlib`.
*   **Documentation:** Basic installation guide, API reference for implemented features, and at least one tutorial notebook.
*   **Project Infrastructure:** `pyproject.toml`, `README.md`, `LICENSE`, `.gitignore`, basic test suite.

### 2.2. Long-Term Scope / Vision
*   Comprehensive 2D detector image loading and reduction (azimuthal integration, corrections).
*   Advanced 1D data processing (merging, desmearing, sophisticated background subtraction).
*   Expanded SAXS analysis (Pair Distance Distribution Function $p(r)$, Kratky analysis, correlation function).
*   WAXS-specific analysis (peak fitting, $d$-spacing, crystallite size, degree of crystallinity).
*   Extensive library of form factors $P(q)$ and structure factors $S(q)$ with support for polydispersity.
*   Global fitting capabilities for multiple datasets and models.
*   Interactive visualizations (e.g., using `plotly` or `bokeh`).
*   Support for advanced/specialized SAXS/WAXS techniques (e.g., GISAXS/GIWAXS, time-resolved).
*   Potential for a simple GUI.
*   Tools for error propagation and uncertainty quantification.
*   Interfaces to external modeling tools or databases if beneficial.

## 3. High-Level Architecture

`ScatterBrain` will be designed as a modular library. The main components (Python modules) will interact primarily through well-defined data structures, particularly the `ScatteringCurve1D` object for 1D data and, later, `SAXSImage`/`WAXSImage` and `ScatteringExperiment` objects.

**Conceptual Data Flow (1D focus initially):**

1.  **`scatterbrain.io`**: Loads raw 1D data file $\rightarrow$ `ScatteringCurve1D` object.
2.  **`scatterbrain.processing`**: Takes `ScatteringCurve1D` $\rightarrow$ Modifies it (e.g., background subtraction) $\rightarrow$ Returns processed `ScatteringCurve1D`.
3.  **`scatterbrain.analysis`**: Takes `ScatteringCurve1D` $\rightarrow$ Performs calculations (e.g., Guinier fit) $\rightarrow$ Returns analysis results (e.g., a dictionary or `AnalysisResult` object).
4.  **`scatterbrain.modeling`**: Takes `ScatteringCurve1D` and model parameters $\rightarrow$ Fits model $\rightarrow$ Returns fit results and best-fit curve.
5.  **`scatterbrain.visualization`**: Takes `ScatteringCurve1D` and/or analysis/modeling results $\rightarrow$ Generates plots.

(A more detailed diagram would be added as development progresses, especially with 2D data.)

## 4. Core Data Structures

### 4.1. `ScatteringCurve1D`
*   **Purpose:** Represents a 1D scattering intensity curve ($I(q)$ vs $q$). This will be the central object for most 1D operations.
*   **Attributes:**
    *   `q`: `numpy.ndarray` - Scattering vector values.
    *   `intensity`: `numpy.ndarray` - Scattering intensity values.
    *   `error`: `numpy.ndarray` (optional) - Error/uncertainty in intensity values.
    *   `metadata`: `dict` - Experimental parameters (wavelength, sample-detector distance, sample name, etc.), processing history, units.
    *   `q_unit`: `str` (e.g., "nm^-1", "A^-1").
    *   `intensity_unit`: `str` (e.g., "cm^-1", "a.u.").
*   **Key Methods (Initial):**
    *   `__init__(self, q, intensity, error=None, metadata=None, q_unit="nm^-1", intensity_unit="a.u.")`
    *   `__str__`, `__repr__`
    *   `copy()`
    *   `to_dict()`, `from_dict()` (for serialization)
    *   `convert_q_unit(new_unit)`
    *   Placeholder methods for common operations that will call functions from other modules (e.g., `curve.guinier_fit()`, `curve.plot()`).

### 4.2. `SAXSImage` / `WAXSImage` (Future)
*   **Purpose:** Represents 2D detector data.
*   **Attributes:** Raw image data (`numpy.ndarray`), mask, metadata (beam center, detector distance, wavelength, pixel size).

### 4.3. `ScatteringExperiment` (Future)
*   **Purpose:** A container object to manage multiple related scattering datasets (e.g., series of concentrations, different temperatures, raw 2D data and its 1D reduction).
*   **Attributes:** List/dictionary of `ScatteringCurve1D` and/or `SAXSImage` objects, global metadata, project information.

## 5. Module Specifications

### 5.1. `scatterbrain.io`
*   **Purpose:** Input/Output operations for scattering data.
*   **Functionality:**
    *   Load 1D data from text files (CSV, DAT, TXT) with flexible column mapping.
    *   (Future) Load 2D detector images (TIFF, EDF, CBF, HDF5).
    *   (Future) Parse metadata from headers or associated files.
    *   Save processed `ScatteringCurve1D` objects or analysis results.
*   **Key Classes/Functions (Initial):**
    *   `load_ascii_1d(filepath, q_col=0, i_col=1, err_col=None, **kwargs)`: Returns `ScatteringCurve1D`.
    *   `save_ascii_1d(curve, filepath)`
*   **Dependencies:** `numpy`, `pandas` (for flexible CSV loading). (Future: `fabio`, `h5py`).
*   **Phase 1 Implementation:** Focus on `load_ascii_1d` for simple column-based text files.

### 5.2. `scatterbrain.reduction` (Primarily Future)
*   **Purpose:** Convert 2D detector data to 1D $I(q)$ vs $q$ curves.
*   **Functionality:** Detector corrections, masking, beam center determination, azimuthal integration, $q$-conversion.
*   **Key Classes/Functions:** `AzimuthalIntegrator`, `BeamCenterFinder`.
*   **Dependencies:** `numpy`, `scipy.ndimage`. (Future: Consider `pyFAI` for performance).
*   **Phase 1 Implementation:** Placeholder module. Core functionality deferred.

### 5.3. `scatterbrain.processing`
*   **Purpose:** Process 1D scattering curves.
*   **Functionality:** Background subtraction, normalization, error propagation, (Future: merging, smoothing, interpolation, desmearing).
*   **Key Classes/Functions (Initial):**
    *   `subtract_background(curve, bg_curve_or_value)`: Returns new `ScatteringCurve1D`.
    *   (Future) `normalize_by_thickness(curve, thickness)`
*   **Dependencies:** `numpy`.
*   **Phase 1 Implementation:** Simple constant background subtraction. Basic error propagation (if errors are present).

### 5.4. `scatterbrain.analysis`
*   **Purpose:** Implement core SAXS/WAXS analysis methods.
*   **Functionality (SAXS - Initial):**
    *   Guinier analysis: $\ln[I(q)]$ vs $q^2$, $R_g$, $I(0)$.
    *   Porod analysis: $I(q)q^4$ vs $q$, Porod constant, Porod exponent.
    *   (Future SAXS) Kratky plots, Pair Distance Distribution Function ($p(r)$ via IFT).
    *   (Future WAXS) Peak finding, fitting, $d$-spacing, Scherrer equation, crystallinity.
*   **Key Classes/Functions (Initial):**
    *   `guinier_fit(curve, q_range=None, auto_q_limit_factor=1.3)`: Returns `dict` or `GuinierResult` object with $R_g, I(0)$, errors, fit quality.
    *   `porod_analysis(curve, q_range=None)`: Returns `dict` or `PorodResult` object with Porod constant, exponent.
*   **Dependencies:** `numpy`, `scipy.optimize` (for fitting), `scipy.stats` (for linear regression).
*   **Phase 1 Implementation:** Implement `guinier_fit` with basic linear regression and optional automatic $q$-range selection. Implement `porod_analysis` based on fitting or slope in Porod plot.

### 5.5. `scatterbrain.modeling`
*   **Purpose:** Fit theoretical models (form factors, structure factors) to scattering data.
*   **Functionality:**
    *   Library of analytical form factors $P(q)$ (sphere, cylinder, etc.).
    *   (Future) Library of structure factors $S(q)$.
    *   Fitting $I(q) = N \cdot P(q) \cdot S(q) + Bkg$.
    *   (Future) Polydispersity, global fitting.
*   **Sub-modules:**
    *   `scatterbrain.modeling.form_factors`
    *   `scatterbrain.modeling.structure_factors` (Future)
*   **Key Classes/Functions (Initial):**
    *   `form_factors.sphere_pq(q, R)`: Returns $P(q)$ for a sphere.
    *   `fit_model(curve, model_func, initial_params, q_range=None, **kwargs)`: Generic fitting utility.
*   **Dependencies:** `numpy`, `scipy.special` (for Bessel functions etc.), `scipy.optimize.curve_fit` or `lmfit`. (Strongly recommend `lmfit` for better parameter handling).
*   **Phase 1 Implementation:** Implement `form_factors.sphere_pq`. A simple `fit_model` function using `scipy.optimize.curve_fit` to fit the sphere model (scale, radius, background) to a `ScatteringCurve1D`.

### 5.6. `scatterbrain.visualization`
*   **Purpose:** Generate plots for data exploration and publication.
*   **Functionality:** $I(q)$ vs $q$ (various scales), Guinier plots, Porod plots, $p(r)$ plots, model fit overlays. (Future: 2D image display, interactive plots).
*   **Key Classes/Functions (Initial):**
    *   `plot_iq(curve, q_scale='log', i_scale='log', **kwargs)`
    *   `plot_guinier(curve, guinier_result=None, q_range_highlight=None, **kwargs)`
    *   `plot_porod(curve, porod_result=None, q_range_highlight=None, **kwargs)`
    *   `plot_fit(curve, model_func, fit_params, **kwargs)`
*   **Dependencies:** `matplotlib`. (Future: `plotly`, `bokeh`).
*   **Phase 1 Implementation:** Basic static plots for $I(q)$, Guinier, and Porod representations, using `matplotlib`.

### 5.7. `scatterbrain.utils`
*   **Purpose:** Utility functions and common constants.
*   **Functionality:** Unit conversions ($q$, wavelength), physical constants, logging setup, custom error classes.
*   **Key Classes/Functions (Initial):**
    *   `convert_q(q_values, current_unit, target_unit)`
    *   `ScatterBrainError(Exception)` base class.
*   **Dependencies:** `numpy`.
*   **Phase 1 Implementation:** Basic q-unit conversion (nm⁻¹ $\leftrightarrow$ Å⁻¹).

## 6. Directory Structure
(As per SOP_Python_libs.md, Section IV, and user's `ScatterBrain` proposal)
```
ScatterBrain/
├── scatterbrain/               # Main library source code package
│   ├── __init__.py
│   ├── core.py                # Core data structures (e.g., ScatteringCurve1D)
│   ├── io.py
│   ├── reduction.py           # Placeholders initially
│   ├── processing.py
│   ├── analysis.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── form_factors.py
│   │   └── structure_factors.py # Placeholders initially
│   ├── visualization.py
│   ├── utils.py
│   └── data/                  # Default data files (e.g., example .dat for tests)
│
├── docs/                      # Sphinx documentation
│   ├── source/
│   │   ├── conf.py
│   │   └── index.rst
│   └── Makefile
│
├── examples/                  # Example Python scripts using the library
│
├── notebooks/                 # Jupyter notebook tutorials/case studies
│
├── tests/                     # Pytest unit and integration tests
│   ├── fixtures/
│   └── test_core.py
│   └── test_io.py
│   └── ...
│
├── .gitignore
├── LICENSE                    # (e.g., MIT, BSD-3-Clause)
├── MANIFEST.in                # If needed for sdist
├── pyproject.toml             # PEP 517/518/621
├── README.md
└── requirements-dev.txt       # (or manage via pyproject.toml optional groups)
```
*(Note: Moved core data structures to `scatterbrain/core.py` for better separation from I/O operations).*

## 7. Technology Stack & Tooling

*   **Programming Language:** Python (>=3.8 recommended)
*   **Core Libraries:**
    *   `numpy`: Fundamental for numerical operations and array handling.
    *   `scipy`: For scientific computing (optimization, special functions, signal processing, statistics).
    *   `matplotlib`: For static plotting.
    *   `pandas`: For flexible data loading (especially CSVs) and tabular data manipulation.
*   **Specialized Libraries (Considerations):**
    *   `fabio`: For I/O of 2D detector image formats (Future).
    *   `pyFAI`: For high-performance azimuthal integration (Future, consider as optional dependency or internal simplified version first).
    *   `lmfit`: For robust model fitting (Highly recommended over plain `scipy.optimize.curve_fit` for `scatterbrain.modeling`).
    *   `scikit-image`: For image processing tasks in 2D data reduction (Future).
*   **Testing Framework:** `pytest`.
*   **Documentation Generator:** `Sphinx` with `sphinx_rtd_theme` (or `furo`), `sphinx.ext.autodoc`, `sphinx.ext.napoleon`, `myst_parser`.
*   **Version Control:** Git, hosted on GitHub (or similar).
*   **Packaging:** `pyproject.toml` (using `setuptools` or `flit`/`poetry` as build backend). `twine` for distribution.
*   **Code Quality:**
    *   `black` for code formatting.
    *   `flake8` for linting (with plugins like `flake8-bugbear`, `flake8-comprehensions`).
    *   `isort` for import sorting.
    *   `mypy` for optional static type checking.
*   **Continuous Integration (CI):** GitHub Actions (to run tests, build docs, lint on pushes/PRs).

## 8. Iterative Development Plan (Initial Phases)

This plan follows the SOP, Section II.

### 8.1. Phase 0: Foundation & Planning
*   **Status:** Largely complete (SOP document, this Design Document).
*   **Recap Existing Knowledge:**
    *   SAXS/WAXS is a well-established field. Key existing software includes SasView, RAW, Irena/Nika (Igor Pro), Scatter, various beamline-specific tools.
    *   Python ecosystem has `pyFAI`, `Dioptas`, parts of `SciKit-GISAXS`.
    *   Common challenges: data format diversity, usability for non-experts, integrating analysis steps, robust error handling, advanced modeling accessibility.
*   **Define Scope:** As per Section 2 of this document.
*   **High-Level Design & Core Data Structures:** As per Sections 3 & 4.
*   **Directory Structure:** As per Section 6.

### 8.2. Phase 1: Core Skeleton Implementation (First Iteration)
*   **Goal:** Create a minimally functional library capable of loading 1D ASCII data, performing basic Guinier and Porod analysis, fitting a sphere model, and generating simple plots.
*   **Tasks:**
    1.  **Project Setup:** Initialize Git repository, `pyproject.toml`, `README.md`, `LICENSE`, `.gitignore`, basic `docs/source/conf.py` and `index.rst`.
    2.  **Core Data Structures:** Implement `scatterbrain.core.ScatteringCurve1D` with basic attributes, `__init__`, `__str__`, `__repr__`, `copy()`.
    3.  **I/O:** Implement `scatterbrain.io.load_ascii_1d` for simple delimited text files.
    4.  **Analysis (Simplified/Placeholders initially, then refine):**
        *   Implement `scatterbrain.analysis.guinier_fit` (initial: basic linear fit, manual q-range).
        *   Implement `scatterbrain.analysis.porod_analysis` (initial: simple slope/fit).
    5.  **Modeling (Simplified):**
        *   Implement `scatterbrain.modeling.form_factors.sphere_pq`.
        *   Implement a basic `scatterbrain.modeling.fit_model` using `scipy.optimize.curve_fit` for the sphere model.
    6.  **Visualization:** Implement `scatterbrain.visualization.plot_iq`, `plot_guinier`, `plot_porod`, `plot_fit` using `matplotlib`.
    7.  **Utilities:** Implement `scatterbrain.utils.convert_q` and `ScatterBrainError`.
    8.  **Basic Pipeline/Orchestration:** Create an example script in `examples/` demonstrating the load-analyze-plot workflow.
    9.  **Unit Tests:** Write `pytest` tests for all implemented classes and functions. Start with `tests/test_core.py`, `tests/test_io.py`. Use simple, verifiable input data.
    10. **Documentation:** Write docstrings for all public code. Populate `README.md` with installation and basic usage.

### 8.3. Phase 2: Enhancing Core Functionality & Usability (Outlook)
*   Refine Guinier/Porod analysis (automated $q$-range selection, error estimation).
*   Implement $p(r)$ calculation (IFT).
*   Expand form factor library (`cylinder_pq`, `core_shell_sphere_pq`).
*   Integrate `lmfit` for more robust modeling and parameter handling.
*   Add more processing functions (e.g., curve normalization).
*   Develop Jupyter Notebook tutorials for each major feature.
*   Build out Sphinx documentation (user guide, API reference).
*   Implement basic error propagation in `processing` and `analysis`.

## 9. API Design Philosophy

*   **User-Centric:** The API should be intuitive for users familiar with SAXS/WAXS concepts.
*   **Object-Oriented Core:** The `ScatteringCurve1D` (and later `SAXSImage`, `ScatteringExperiment`) object will be central, allowing for a fluent API (e.g., `curve.guinier_fit()`, `curve.plot()`).
*   **Functional Components:** Underlying calculations (e.g., `guinier_fit` function in `analysis` module) will be well-defined functions that can also be used independently if needed.
*   **Consistency:**
    *   Naming conventions for functions, methods, and parameters.
    *   Consistent return types for similar operations (e.g., analysis functions returning a dictionary or a dedicated `Result` dataclass/object).
    *   Standardized parameter names (e.g., `q_range=(q_min, q_max)`).
*   **Extensibility:** Design interfaces (e.g., for models or analysis routines) that allow users to easily add their own custom components.
*   **Error Handling:** Clear and informative error messages, using custom exceptions derived from `ScatterBrainError`.
*   **Progressive Disclosure:** Simple interfaces for common tasks, with options for more advanced control.

## 10. Testing Strategy

*   **Unit Tests:** `pytest` will be used. Each module, class, and public function should have corresponding unit tests.
    *   Test pure functions with various inputs, including edge cases and expected failures.
    *   Test class instantiation, methods, and attribute access.
*   **Reference Data:**
    *   Use simulated data generated from known analytical expressions (e.g., perfect sphere scattering) to validate analysis and modeling functions.
    *   (Potentially) Use well-characterized, published experimental datasets as benchmarks.
*   **Integration Tests:** Test workflows involving multiple modules (e.g., load data $\rightarrow$ process $\rightarrow$ analyze $\rightarrow$ plot).
*   **Test Coverage:** Aim for high test coverage, tracked using tools like `coverage.py`.
*   **Notebooks as User Acceptance Tests:** Jupyter notebooks in `notebooks/` will serve as examples and can be run as part of testing to ensure end-to-end functionality and illustrate correct usage.
*   **CI:** Automated testing on every push/PR via GitHub Actions.

## 11. Documentation Strategy

*   **Audience:** End-users (scientists performing SAXS/WAXS analysis) and developers (those wishing to contribute to or extend `ScatterBrain`).
*   **Tools:** `Sphinx` with recommended extensions.
*   **Content:**
    *   **Installation Guide:** Clear instructions for installing `ScatterBrain` and its dependencies.
    *   **Quick Start Guide:** A brief tutorial to get users started with a simple workflow.
    *   **User Guide / Tutorials:**
        *   Detailed explanations of SAXS/WAXS concepts relevant to the library's features (e.g., Guinier theory, Porod's Law, form factor basics). This is crucial for transparency and user understanding.
        *   Step-by-step Jupyter Notebooks demonstrating how to use different functionalities with example data.
    *   **API Reference:** Auto-generated from docstrings (`sphinx.ext.autodoc`, `sphinx.ext.napoleon`) for all public modules, classes, and functions.
    *   **Examples:** Collection of example scripts in `examples/`.
    *   **Developer Guide (Future):** Information on contributing to the library, coding style, running tests, building documentation.
    *   **License & Changelog.**
*   **Docstrings:** Comprehensive docstrings in Google or NumPy style for all code.
*   **Build Regularly:** Documentation will be built and reviewed frequently.

## 12. Potential Future Enhancements
(As identified in the initial `ScatterBrain` proposal)
*   GUI (e.g., using `Qt`, `Tkinter`, or a web framework like `Dash`).
*   Integration with LIMS or experiment databases.
*   Advanced modeling (e.g., Reverse Monte Carlo, Bayesian methods).
*   Machine learning applications (e.g., automated phase identification, parameter estimation).
*   Support for Grazing Incidence SAXS/WAXS (GISAXS/GIWAXS) reduction and analysis.
*   Time-resolved data analysis.

## 13. Risks and Mitigation (Preliminary)

*   **Complexity of 2D Reduction:** Implementing robust 2D data reduction is complex and computationally intensive.
    *   **Mitigation:** Defer to later phase. Consider leveraging `pyFAI` (via wrapper or direct use) to avoid re-implementing highly optimized algorithms. Start with 1D data focus.
*   **Performance Bottlenecks:** Some calculations (IFT for $p(r)$, complex model fitting, 2D integration) can be slow in pure Python.
    *   **Mitigation:** Profile code. Optimize critical sections using `numpy` vectorization. Consider `numba` or Cython for specific compute-heavy functions if necessary.
*   **Scope Creep:** The field of SAXS/WAXS is vast; there's a temptation to include too many features too soon.
    *   **Mitigation:** Strictly adhere to the iterative development plan. Prioritize core functionalities. Maintain a clear backlog of future features.
*   **Maintaining User-Friendliness with Advanced Features:** As complexity grows, keeping the API simple can be challenging.
    *   **Mitigation:** Employ progressive disclosure in API design. Provide clear documentation and tutorials for advanced features.
*   **Accuracy of Implementations:** Ensuring scientific correctness of all analysis and modeling routines.
    *   **Mitigation:** Rigorous testing against known analytical solutions, simulated data, and reference implementations/literature values. Clearly document assumptions and limitations of each method.

---

This design document provides a comprehensive plan for the `ScatterBrain` library. The next step, following the SOP and this document, would be to begin Phase 1 implementation tasks, starting with setting up the project structure and implementing the `ScatteringCurve1D` class.