Quickstart
==========

The following example shows the complete Phase 1 workflow in about ten lines:

.. code-block:: python

   import scatterbrain
   from scatterbrain.io import load_ascii_1d
   from scatterbrain.processing import subtract_background
   from scatterbrain.analysis import guinier_fit, porod_analysis
   from scatterbrain.modeling.form_factors import sphere_pq
   from scatterbrain.modeling.fitting import fit_model
   from scatterbrain.visualization import plot_iq, plot_guinier, plot_fit

   # Load data
   curve = load_ascii_1d("my_data.dat", q_col=0, i_col=1, err_col=2)

   # Subtract constant background
   curve_bg = subtract_background(curve, 10.0)

   # Guinier analysis
   g = guinier_fit(curve_bg)
   if g:
       print(f"Rg = {g['Rg']:.3f} +/- {g['Rg_err']:.3f}")

   # Sphere model fit
   result = fit_model(
       curve_bg, sphere_pq, ["radius"],
       initial_params=[1e4, 0.0, 5.0],
   )
   if result:
       print(f"R = {result['fitted_params']['radius']:.3f}")

See the :doc:`tutorials/index` for step-by-step walkthroughs.
