Tutorial 02 -- Form Factor Fitting
====================================

This tutorial demonstrates fitting multiple form factor models to SAXS data
and comparing them using reduced chi-squared and confidence intervals from lmfit.

The notebook is located at ``notebooks/02_form_factor_fitting.ipynb``.

Topics covered
--------------

* :func:`scatterbrain.modeling.form_factors.sphere_pq` -- sphere model fit
* :func:`scatterbrain.modeling.form_factors.cylinder_pq` -- cylinder model fit
* :func:`scatterbrain.modeling.fitting.fit_model` -- lmfit-backed fitting
* Confidence intervals via ``result["confidence_intervals"]``
* :func:`scatterbrain.visualization.plot_fit` -- comparing fit quality
* :func:`scatterbrain.visualization.plot_kratky` -- Kratky plot diagnostic
