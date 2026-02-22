Tutorial 01 -- Basic Workflow
==============================

This tutorial demonstrates the complete basic workflow:
loading data, background subtraction, Guinier and Porod analysis,
sphere form factor fitting, and saving results.

The notebook is located at ``notebooks/01_basic_workflow.ipynb``.

Topics covered
--------------

* :func:`scatterbrain.io.load_ascii_1d` -- loading 1D ASCII data
* :func:`scatterbrain.processing.subtract_background` -- constant background
* :func:`scatterbrain.analysis.guinier_fit` -- Rg and I(0)
* :func:`scatterbrain.analysis.porod_analysis` -- Porod exponent and constant
* :func:`scatterbrain.modeling.fitting.fit_model` -- sphere form factor fit
* :func:`scatterbrain.visualization.plot_iq` -- I(q) vs q
* :func:`scatterbrain.visualization.plot_guinier` -- Guinier plot
* :func:`scatterbrain.visualization.plot_fit` -- data + model + residuals
