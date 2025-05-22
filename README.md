# ScatterBrain 🧠✨

**A Python library for SAXS/WAXS data analysis and modeling.**

*Dealing with scattered X-rays and the (sometimes) complex thought process of analyzing them.*

**Current Status:** Pre-Alpha (Under Active Development)

## Overview

`ScatterBrain` aims to provide a comprehensive, user-friendly, and extensible Python toolkit for scientists and researchers working with Small-Angle X-ray Scattering (SAXS) and Wide-Angle X-ray Scattering (WAXS) data. The library is designed to facilitate the entire workflow from data loading and processing to advanced analysis and model fitting.

## Guiding Principles

*   **Modularity:** Well-defined, testable components.
*   **Iterative Development:** Core functionality first, then incremental enhancements.
*   **Test-Supported Development:** Rigorous testing for reliability.
*   **Clear Documentation:** Comprehensive guides for users and developers.
*   **User-Centric Design:** Focus on intuitive and efficient workflows.
*   **Transparency:** Clear articulation of methods, assumptions, and limitations.
*   **Extensibility:** Designed for easy addition of new models and analysis routines.

## Planned Features (Iterative Implementation)

*   **Data I/O:**
    *   Loading 1D SAXS/WAXS data (e.g., `.dat`, `.txt`, `.csv`).
    *   (Future) 2D detector image loading (TIFF, EDF, CBF, HDF5).
*   **Data Reduction (Future):**
    *   Azimuthal integration, detector corrections, masking.
*   **Data Processing:**
    *   Background subtraction, normalization.
    *   (Future) Merging, smoothing, desmearing.
*   **SAXS Analysis:**
    *   Guinier analysis ($R_g$, $I(0)$).
    *   Porod analysis (Porod constant, exponent, specific surface area).
    *   (Future) Pair Distance Distribution Function $p(r)$, Kratky plots.
*   **WAXS Analysis (Future):**
    *   Peak fitting, $d$-spacing calculation (Bragg's Law).
    *   Crystallite size estimation (Scherrer equation), degree of crystallinity.
*   **Modeling:**
    *   Library of analytical form factors $P(q)$ (sphere, cylinder, etc.).
    *   (Future) Structure factors $S(q)$, polydispersity, global fitting.
*   **Visualization:**
    *   Standard scattering plots ($I(q)$ vs $q$, Guinier, Porod) using Matplotlib.
    *   (Future) Interactive plots, 2D image display.

## Installation (Placeholder)

Once a stable version is released, `ScatterBrain` will be installable via pip:

```bash
pip install scatterbrain