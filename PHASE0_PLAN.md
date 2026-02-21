# ScatterBrain — Phase 0 Plan: Foundation & Planning

**Version:** 1.1
**Branch:** `main`
**Reference:** `Design_document.md` §8.1, `phase0.md`, SOP_Python_libs.md

---

## Objectives

Phase 0 establishes the intellectual and structural foundation for all subsequent development. No production code is written in this phase. The outputs are documents, decisions, and a skeleton project scaffold.

---

## Deliverables & Status

| # | Deliverable | Description | Status |
|---|-------------|-------------|--------|
| 0.1 | Domain survey | Summary of existing SAXS/WAXS software and Python ecosystem | ✅ Done (`Design_document.md` §8.1) |
| 0.2 | Scope definition | MVP scope and long-term vision | ✅ Done (`Design_document.md` §2) |
| 0.3 | High-level design document | Architecture, modules, data flow, API philosophy | ✅ Done (`Design_document.md`) |
| 0.4 | Core data structure spec | `ScatteringCurve1D`, future `SAXSImage`, `ScatteringExperiment` | ✅ Done (`Design_document.md` §4) |
| 0.5 | Directory structure | Canonical project layout | ✅ Done (`Design_document.md` §6) |
| 0.6 | Technology stack decision | Language, core libraries, tooling choices | ✅ Done (`Design_document.md` §7) |
| 0.7 | Testing strategy | Framework, coverage targets, reference data approach | ✅ Done (`Design_document.md` §10) |
| 0.8 | Documentation strategy | Tools, audience, content outline | ✅ Done (`Design_document.md` §11) |
| 0.9 | Risk register | Identified risks and mitigations | ✅ Done (`Design_document.md` §13) |
| 0.10 | This Phase 0 plan | Formal task checklist | ✅ Done (this file) |

---

## Detailed Tasks

### 0.1 — Domain Survey

**Goal:** Understand the existing landscape to avoid reinventing the wheel and to identify gaps ScatterBrain fills.

**Tasks:**
- [x] List existing SAXS/WAXS analysis software (SasView, RAW, Irena/Nika, Scatter, beamline tools)
- [x] Survey relevant Python libraries (`pyFAI`, `Dioptas`, `SciKit-GISAXS`)
- [x] Identify common pain points: format diversity, usability, integrated workflows, accessibility of advanced modeling
- [x] Document findings in Design Document §8.1

---

### 0.2 — Scope Definition

**Goal:** Define a concrete MVP and an aspirational long-term vision.

**MVP (Phase 1 target):**
- [x] 1D ASCII data loading (`.dat`, `.txt`, `.csv` with q, I, error columns)
- [x] `ScatteringCurve1D` core object
- [x] Background subtraction (constant or curve)
- [x] Guinier analysis (R_g, I(0))
- [x] Porod analysis (Porod constant, exponent)
- [x] Sphere form factor fitting
- [x] Static matplotlib plots (I(q), Guinier, Porod)
- [x] Basic docs and at least one tutorial notebook
- [x] Project infrastructure (pyproject.toml, README, LICENSE, .gitignore, test suite)

**Long-Term Vision (future phases):**
- [ ] 2D detector image loading and azimuthal integration
- [ ] Advanced processing (merging, desmearing, normalization)
- [ ] p(r) via IFT, Kratky analysis
- [ ] WAXS peak fitting, Scherrer equation, crystallinity
- [ ] Expanded form factor / structure factor library with polydispersity
- [ ] Interactive plots (plotly/bokeh)
- [ ] GISAXS/GIWAXS, time-resolved data
- [ ] Optional GUI

---

### 0.3 — High-Level Architecture

**Goal:** Define how modules interact and data flows through the library.

**Conceptual pipeline:**
```
scatterbrain.io
    ↓ ScatteringCurve1D
scatterbrain.processing
    ↓ ScatteringCurve1D (processed)
scatterbrain.analysis     scatterbrain.modeling
    ↓ AnalysisResult           ↓ FitResult
scatterbrain.visualization
    ↓ matplotlib Figure
```

**Module map:**
- [x] `scatterbrain.io` — file loading/saving
- [x] `scatterbrain.core` — `ScatteringCurve1D` and future data structures
- [x] `scatterbrain.processing` — background subtraction, normalization
- [x] `scatterbrain.analysis` — Guinier, Porod (sub-package: `analysis/guinier.py`, `analysis/porod.py`)
- [x] `scatterbrain.modeling` — form factors, fitting (sub-package)
- [x] `scatterbrain.visualization` — matplotlib plots
- [x] `scatterbrain.utils` — unit conversion, constants, custom exceptions
- [x] `scatterbrain.reduction` — placeholder for future 2D reduction

---

### 0.4 — Core Data Structure Specification

**Goal:** Nail down the `ScatteringCurve1D` interface before any code is written.

**`ScatteringCurve1D` spec:**

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `q` | `np.ndarray` | Yes | Scattering vector values |
| `intensity` | `np.ndarray` | Yes | Intensity I(q) |
| `error` | `np.ndarray` | No | Uncertainty in intensity |
| `metadata` | `dict` | No | Experiment params, processing history |
| `q_unit` | `str` | Yes | `"nm^-1"` or `"A^-1"` |
| `intensity_unit` | `str` | Yes | `"cm^-1"`, `"a.u."`, etc. |

**Initial methods:** `__init__`, `__str__`, `__repr__`, `copy()`, `to_dict()`, `from_dict()`, `convert_q_unit()`

**Status:** [x] Specified in `Design_document.md` §4.1

---

### 0.5 — Directory Structure

**Goal:** Agree on canonical project layout.

**Agreed structure (from Design_document.md §6):**
```
ScatterBrain/
├── scatterbrain/
│   ├── __init__.py          # exposes configure_logging()
│   ├── core.py
│   ├── io.py
│   ├── utils.py
│   ├── visualization.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── guinier.py
│   │   └── porod.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── form_factors.py
│   │   └── fitting.py
│   ├── processing/
│   │   ├── __init__.py
│   │   └── background.py
│   ├── reduction/           # placeholder
│   │   └── __init__.py
│   ├── data/
│   │   └── __init__.py
│   └── examples/
│       └── data/
│           └── example_sphere_data.dat
├── docs/source/
├── examples/
├── notebooks/
│   └── 01_basic_workflow.ipynb
├── tests/
├── .github/workflows/ci.yml
├── .flake8
├── pyproject.toml           # build backend: uv_build
├── README.md
├── LICENSE                  # CC-BY-NC-SA-4.0
├── uv.lock
└── .gitignore
```

**Status:** [x] Defined; actual scaffold created in codebase

---

### 0.6 — Technology Stack Decisions

**Goal:** Lock in tool choices before Phase 1.

| Concern | Decision | Rationale |
|---------|----------|-----------|
| Language | Python ≥ 3.10 | Target audience norm; 3.8 EOL Oct 2024 |
| Numerical | `numpy` | Standard for scientific Python |
| Scientific | `scipy` | Optimization, special functions, stats |
| Plotting | `matplotlib` (initial) | Ubiquitous, publication-quality |
| Data loading | `pandas` | Flexible CSV/text parsing |
| Fitting | `scipy.optimize.curve_fit` → `lmfit` | Start simple, migrate for robustness |
| Testing | `pytest` + `coverage.py` | Industry standard |
| Docs | `Sphinx` + `sphinx_rtd_theme` + `myst_parser` | Auto-API + Markdown support |
| Formatting | `black` | Opinionated, zero-config |
| Linting | `flake8` | Broad compatibility |
| Type checking | `mypy` (optional) | Progressive adoption |
| Package manager | `uv` | Fast, lockfile-based (`uv sync --all-extras`) |
| Build backend | `uv_build` | Declared in `pyproject.toml [build-system]` |
| CI | GitHub Actions | Free for open source; `.github/workflows/ci.yml` |
| Future 2D I/O | `fabio` | Standard detector image formats |
| Future 2D reduction | `pyFAI` | High-performance azimuthal integration |

**Status:** [x] Decided in `Design_document.md` §7

---

### 0.7 — Testing Strategy

**Goal:** Define how correctness will be verified.

- [x] Framework: `pytest`
- [x] Unit tests for every public class and function
- [x] Reference data: simulated data from known analytical expressions (e.g., perfect sphere I(q))
- [x] Integration tests: full load → process → analyze → plot workflows
- [x] Coverage tracking: `coverage.py`, aim for ≥80% on core modules
- [x] Notebooks as end-to-end acceptance tests (run via `nbmake` or `nbconvert`)
- [x] CI: GitHub Actions on every push/PR

---

### 0.8 — Documentation Strategy

**Goal:** Define documentation structure and tooling.

- [x] Tool: `Sphinx` with `autodoc`, `napoleon`, `myst_parser`
- [x] Theme: `sphinx_rtd_theme`
- [x] Docstring style: NumPy or Google (pick one and enforce consistently — **recommendation: NumPy style**)
- [x] Sections: Installation, Quick Start, User Guide (tutorials), API Reference, Developer Guide, Changelog
- [x] Jupyter Notebooks: `notebooks/` for tutorials/case studies
- [x] Build regularly; include in CI

---

### 0.9 — Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 2D reduction complexity | High | High | Defer to later phase; wrap pyFAI |
| Performance bottlenecks (IFT, complex fitting) | Medium | Medium | Profile first; numpy vectorization; consider numba/Cython |
| Scope creep | High | Medium | Strict iterative plan; maintain backlog |
| API complexity growing | Medium | Medium | Progressive disclosure; thorough docs |
| Scientific accuracy | Medium | High | Test vs. known analytical solutions and literature |

---

## Phase 0 → Phase 1 Transition Criteria

Phase 0 is complete and Phase 1 may begin when:

1. [x] Design document is finalized and reviewed
2. [x] All technology stack decisions are made
3. [x] Core data structure interface is specified
4. [x] Directory structure is agreed upon
5. [x] Testing and documentation strategies are defined
6. [x] This Phase 0 plan document exists in the repository

**Phase 0 is complete. Proceed to Phase 1: Core Skeleton Implementation.**

---

## Phase 1 First Steps (for reference)

Per `Design_document.md` §8.2:

1. Project setup: verify `pyproject.toml`, `README.md`, `LICENSE`, `.gitignore`, `docs/source/conf.py`
2. Implement `ScatteringCurve1D` in `scatterbrain/core.py`
3. Implement `load_ascii_1d` in `scatterbrain/io.py`
4. Implement `guinier_fit` in `scatterbrain/analysis/guinier.py`
5. Implement `porod_analysis` in `scatterbrain/analysis/porod.py`
6. Implement `sphere_pq` in `scatterbrain/modeling/form_factors.py`
7. Implement `fit_model` in `scatterbrain/modeling/fitting.py`
8. Implement visualization functions in `scatterbrain/visualization.py`
9. Implement `convert_q_array`, `ScatterBrainError` hierarchy, and `NullHandler` logging init in `scatterbrain/utils.py`; expose `configure_logging()` from `scatterbrain/__init__.py`
10. Write `pytest` unit tests for all of the above
11. Write docstrings for all public API
12. Create `examples/basic_workflow.py` and at least one tutorial notebook
