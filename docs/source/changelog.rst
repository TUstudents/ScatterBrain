Changelog
=========

0.1.0 -- Phase 2 (February 2026)
----------------------------------

New features
~~~~~~~~~~~~

* ``plot_kratky``: standard and dimensionless Kratky plot
  (:func:`scatterbrain.visualization.plot_kratky`).
* ``normalize``: divide intensity by a scalar factor with error propagation
  (:func:`scatterbrain.processing.normalize`).
* Weighted Guinier regression: ``guinier_fit`` now accepts ``use_errors=True``
  (default) to weight the linear regression by ``1 / sigma_lnI``
  (:func:`scatterbrain.analysis.guinier_fit`).
* Scattering invariant Q*: new ``scattering_invariant`` function with
  Guinier low-q and Porod high-q extrapolations
  (:func:`scatterbrain.analysis.scattering_invariant`).
* ``cylinder_pq``: orientationally averaged cylinder form factor via
  64-point Gauss-Legendre quadrature
  (:func:`scatterbrain.modeling.form_factors.cylinder_pq`).
* ``core_shell_sphere_pq``: spherically symmetric core-shell form factor
  (:func:`scatterbrain.modeling.form_factors.core_shell_sphere_pq`).
* lmfit integration: ``fit_model`` now uses lmfit internally; result dict
  gains ``confidence_intervals`` and ``lmfit_result`` keys
  (:func:`scatterbrain.modeling.fitting.fit_model`).
* Sphinx documentation: populated API reference and tutorials.
* Notebook 02: ``notebooks/02_form_factor_fitting.ipynb``.

Bug fixes
~~~~~~~~~

* Fixed pre-existing flakiness in ``test_noisy_curve_auto_q_range`` by
  seeding the noise fixture.

0.0.1 -- Phase 1 (February 2026)
----------------------------------

Initial release.

* Core data object: ``ScatteringCurve1D``.
* I/O: ``load_ascii_1d``, ``save_ascii_1d``.
* Processing: ``subtract_background``.
* Analysis: ``guinier_fit`` (``GuinierResult``), ``porod_analysis`` (``PorodResult``).
* Modeling: ``sphere_pq``, ``fit_model``.
* Visualization: ``plot_iq``, ``plot_guinier``, ``plot_porod``, ``plot_fit``.
* Utilities: ``convert_q_array``, ``ScatterBrainError`` hierarchy.
* Logging: ``configure_logging()``.
