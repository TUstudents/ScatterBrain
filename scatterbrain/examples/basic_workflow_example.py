# examples/basic_workflow_example.py
"""
Example script demonstrating a basic workflow using the ScatterBrain library:
1. Load 1D SAXS data.
2. Perform Guinier analysis.
3. Perform Porod analysis.
4. Fit a sphere model.
5. Plot results.
"""

import pathlib
import matplotlib.pyplot as plt
import numpy as np # For np.inf in bounds

# Import necessary components from ScatterBrain
# Assuming ScatterBrain is installed or PYTHONPATH is set to include its root
try:
    from scatterbrain.core import ScatteringCurve1D
    from scatterbrain.io import load_ascii_1d
    from scatterbrain.analysis import guinier_fit, porod_analysis
    from scatterbrain.modeling import sphere_pq, fit_model
    from scatterbrain.visualization import plot_iq # plot_guinier, plot_porod, plot_fit (when implemented)
except ImportError:
    # Fallback for running script directly from examples directory if ScatterBrain not installed
    import sys
    # Add the parent directory (ScatterBrain root) to the Python path
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from scatterbrain.core import ScatteringCurve1D
    from scatterbrain.io import load_ascii_1d
    from scatterbrain.analysis import guinier_fit, porod_analysis
    from scatterbrain.modeling import sphere_pq, fit_model
    from scatterbrain.visualization import plot_iq


def run_basic_workflow():
    """
    Executes the example SAXS analysis workflow.
    """
    print("--- ScatterBrain Basic Workflow Example ---")

    # --- 1. Load Data ---
    print("\n[1] Loading data...")
    # Construct path to the example data file relative to this script
    script_dir = pathlib.Path(__file__).resolve().parent
    data_file = script_dir / "data" / "example_sphere_data.dat"

    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        print("Please ensure 'example_sphere_data.dat' is in an 'examples/data/' directory.")
        return

    try:
        # skip_header=3 because of: #comment, #comment, #header_names
        curve = load_ascii_1d(data_file, q_col=0, i_col=1, err_col=2, skip_header=3, delimiter=r'\s+')
        print(f"Loaded data: {curve}")
        print(f"  Number of points: {len(curve)}")
        print(f"  q range: {curve.q.min():.3g} - {curve.q.max():.3g} {curve.q_unit}")
        print(f"  Intensity range: {curve.intensity.min():.3g} - {curve.intensity.max():.3g} {curve.intensity_unit}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Plot Initial Data ---
    fig_data, ax_data = plot_iq(curve, title=f"Raw Data: {data_file.name}", errorbars=True)
    fig_data.show() # Use plt.show() if running in a script outside interactive env that shows plots

    # --- 2. Guinier Analysis ---
    print("\n[2] Performing Guinier analysis...")
    # For Rg ~ 3nm, qRg_max=1.3 -> q_max ~ 1.3/3 ~ 0.43.
    # Our data goes down to q=0.02. Let's try auto q-range.
    # A manual range might be: q_range=(0.02, 0.1) for Rg~3nm (q_max=0.1 gives qRg_max ~0.3, too small)
    # q_max should be more like 0.3/3 = 0.1 for qRg=0.3 -> Rg_max ~ 1.3/0.1 = 13
    # For Rg=3nm, qmax should be around 1.3/3 = 0.43.
    # Let's test q_range = (0.02, 0.15) which is qRg_max = 0.15 * 3 = 0.45 for Rg=3.
    # The auto q-range should find something reasonable.
    # q_guinier_range = (0.03, 0.1) # A possible manual range
    guinier_results = guinier_fit(
        curve,
        qrg_limit_max=1.3,
        min_points=5,
        auto_q_selection_fraction=0.2 # INCREASED from default 0.1
    )

    if guinier_results:
        print("  Guinier Fit Results:")
        print(f"    Rg            : {guinier_results['Rg']:.3f} +/- {guinier_results['Rg_err']:.3f} {curve.q_unit[:-3] if curve.q_unit.endswith('^-1') else 'units'}")
        print(f"    I(0)          : {guinier_results['I0']:.3e} +/- {guinier_results['I0_err']:.3e} {curve.intensity_unit}")
        print(f"    Fit q-range   : {guinier_results['q_fit_min']:.3g} - {guinier_results['q_fit_max']:.3g} {curve.q_unit}")
        print(f"    Points used   : {guinier_results['num_points_fit']}")
        print(f"    R-value       : {guinier_results['r_value']**2:.4f} (R-squared)") # R-squared
        print(f"    Range criteria: {guinier_results['valid_guinier_range_criteria']}")
        # TODO: Add plot_guinier(curve, guinier_results) when implemented
    else:
        print("  Guinier fit failed or was not applicable.")

    # --- 3. Porod Analysis ---
    print("\n[3] Performing Porod analysis...")
    # Use high-q region, e.g., last 30% of data points or q > 1.0 nm^-1
    # porod_q_range = (1.0, curve.q.max())
    porod_results = porod_analysis(curve, q_fraction_high=0.3, fit_log_log=True) # Auto high-q

    if porod_results:
        print("  Porod Fit Results (log-log):")
        print(f"    Porod Exponent: {porod_results['porod_exponent']:.3f} +/- {porod_results['porod_exponent_err']:.3f}")
        print(f"    Porod Constant: {porod_results['porod_constant_kp']:.3e} +/- {porod_results['porod_constant_kp_err']:.3e}")
        print(f"    Fit q-range   : {porod_results['q_fit_min']:.3g} - {porod_results['q_fit_max']:.3g} {curve.q_unit}")
        print(f"    Points used   : {porod_results['num_points_fit']}")
        # TODO: Add plot_porod(curve, porod_results) when implemented
    else:
        print("  Porod analysis failed.")

    # --- 4. Fit Sphere Model ---
    print("\n[4] Fitting sphere model...")
    # Model function: sphere_pq(q, radius)
    # Parameters for fit_model: scale, background, radius
    
    # Initial guesses - these can be critical!
    # From Guinier: Rg ~ 3nm => Radius_sphere = sqrt(5/3)*Rg ~ sqrt(1.66)*3 ~ 1.29*3 ~ 3.87 nm
    # From Guinier: I0 ~ 1e5 (this is scale * P(0)=scale, if P(0)=1)
    # Background: look at high-q intensity, maybe around 1-10
    initial_radius_guess = guinier_results['Rg'] * np.sqrt(5/3) if guinier_results and not np.isnan(guinier_results['Rg']) else 4.0
    initial_scale_guess = guinier_results['I0'] if guinier_results and not np.isnan(guinier_results['I0']) else 1e5
    initial_background_guess = np.median(curve.intensity[curve.q > curve.q.max() * 0.8]) # Median of last 20% of I
    
    param_names_sphere = ['radius'] # For the sphere_pq function itself
    initial_params_fit = [
        initial_scale_guess,      # Initial guess for scale
        initial_background_guess, # Initial guess for background
        initial_radius_guess      # Initial guess for radius
    ]
    print(f"  Initial guesses for sphere fit: Scale={initial_scale_guess:.2e}, BG={initial_background_guess:.2f}, Radius={initial_radius_guess:.2f}") # Add print

    # Bounds: ([scale_low, bg_low, r_low], [scale_high, bg_high, r_high])
    # Adjust radius upper bound to be more accommodating or dynamic
    radius_upper_bound = max(20.0, initial_radius_guess * 1.5 + 5.0) # Make upper bound more dynamic
    param_bounds_fit = (
        [1e2, 0, 0.5],                                 # Lower bounds
        [1e7, max(50.0, initial_background_guess * 2 + 10), radius_upper_bound]  # Upper bounds, also make BG upper bound more dynamic
    )
    print(f"  Parameter bounds for sphere fit: Lower={param_bounds_fit[0]}, Upper={param_bounds_fit[1]}") # Add print
    
    # Fit over a q-range that covers the main scattering features
    # Avoid very high q if it's too noisy or dominated by background not handled by the simple model
    fit_q_range = (0.02, 2.0) # Example range, might need tuning

    sphere_fit_results = fit_model(
        curve=curve,
        model_func=sphere_pq,
        param_names=param_names_sphere,
        initial_params=initial_params_fit,
        param_bounds=param_bounds_fit,
        q_range=fit_q_range
    )

    if sphere_fit_results and sphere_fit_results["success"]:
        print("  Sphere Model Fit Successful:")
        fp = sphere_fit_results["fitted_params"]
        fpe = sphere_fit_results["fitted_params_stderr"]
        print(f"    Fitted Scale    : {fp['scale']:.3e} +/- {fpe['scale']:.2e}")
        print(f"    Fitted Background: {fp['background']:.3f} +/- {fpe['background']:.2f}")
        print(f"    Fitted Radius   : {fp['radius']:.3f} +/- {fpe['radius']:.2f} {curve.q_unit[:-3] if curve.q_unit.endswith('^-1') else 'units'}")
        print(f"    Reduced Chi^2   : {sphere_fit_results['chi_squared_reduced']:.3f}")
        print(f"    Fit q-range     : {sphere_fit_results['q_fit_min']:.3g} - {sphere_fit_results['q_fit_max']:.3g}")

        # --- 5. Plot Fit Results ---
        print("\n[5] Plotting fit results...")
        fig_fit, ax_fit = plot_iq(
            curve,
            label="Experimental Data",
            title="Sphere Model Fit",
            errorbars=True,
            markersize=3  # Pass plot kwargs directly, not nested
        )
        # Plot the fitted model curve
        fit_curve_obj = sphere_fit_results["fit_curve"]
        ax_fit.plot(fit_curve_obj.q, fit_curve_obj.intensity, label="Sphere Fit", color='red', linewidth=1.5)
        ax_fit.legend()
        fig_fit.show()
        # TODO: Add plot_fit(curve, sphere_fit_results) when implemented for residuals etc.

    elif sphere_fit_results:
        print("  Sphere Model Fit did not converge successfully.")
        print(f"  Message: {sphere_fit_results.get('message', 'No message')}")
    else:
        print("  Sphere Model Fit failed.")

    print("\n--- End of Workflow ---")
    if __name__ == '__main__': # Only call plt.show() if script is run directly
        print("\nDisplaying all plots. Close plot windows to exit script.")
        plt.show()


if __name__ == '__main__':
    run_basic_workflow()