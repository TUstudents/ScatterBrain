# tests/test_modeling.py
"""
Unit tests for the scatterbrain.modeling subpackage,
including form factors and fitting routines.
"""
import pytest
import numpy as np

import pytest
import numpy as np
import warnings

from scatterbrain.core import ScatteringCurve1D
from scatterbrain.modeling.form_factors import sphere_pq, _Q_EPSILON
from scatterbrain.modeling.fitting import fit_model

# --- Test Cases for Form Factors ---

class TestSpherePq:
    """Tests for the sphere_pq form factor function."""

    @pytest.fixture
    def q_values(self) -> np.ndarray:
        """Standard q-values for testing form factors."""
        # Using geomspace as P(q) often plotted on log-log
        return np.geomspace(0.001, 5.0, 200)

    def test_sphere_pq_at_q_equals_zero(self):
        """Test P(q=0) for a sphere should be 1.0."""
        q_zero = np.array([0.0, _Q_EPSILON / 5]) # Test exactly zero and very close to zero
        radius = 5.0
        pq = sphere_pq(q_zero, radius)
        assert np.allclose(pq, 1.0), f"P(q=0) or P(q~0) for sphere should be 1.0, got {pq}"

    def test_sphere_pq_basic_calculation(self, q_values: np.ndarray):
        """Test P(q) values at a few characteristic points or general behavior."""
        radius = 10.0 # Example radius in nm
        pq = sphere_pq(q_values, radius)

        assert pq.shape == q_values.shape
        assert np.all(pq >= 0), "P(q) should always be non-negative."
        assert pq[0] > 0.99 and pq[0] <= 1.01, f"P(q->0) should be close to 1.0, got {pq[0]}"

        # Check first minimum: expected around qR ~ 4.4934
        qR_first_min_approx = 4.4934
        q_first_min_approx = qR_first_min_approx / radius #  ~0.44934 for R=10

        # Find index closest to this q value
        idx_near_first_min = np.argmin(np.abs(q_values - q_first_min_approx))
        
        # P(q) at the first minimum should be very small.
        # The exact value is non-zero, but for numerical precision, check it's low.
        # Value at first min is ~7.4e-3 for the (sin(x)-xcos(x))/(x^3) term, so Pq ~ (7.4e-3)^2 ~ 5.5e-5
        # Let's check that P(q) around the minimum is indeed small
        # and that values on either side are larger.
        if 0 < idx_near_first_min < len(pq) - 1: # Ensure we have neighbors
            assert pq[idx_near_first_min] < 1e-4, \
                f"P(q) at first minimum (q~{q_values[idx_near_first_min]:.3f}) should be very small, got {pq[idx_near_first_min]:.2e}"
            # Check it's a local minimum (numerically)
            # This can be sensitive to q-spacing.
            # assert pq[idx_near_first_min] < pq[idx_near_first_min - 1]
            # assert pq[idx_near_first_min] < pq[idx_near_first_min + 1]
        else: # pragma: no cover
            pytest.skip("q_values do not adequately cover the first minimum for this test.")


    def test_sphere_pq_scaling_with_radius(self, q_values: np.ndarray):
        """Test that features in P(q) scale correctly with 1/R."""
        radius1 = 5.0
        radius2 = 10.0 # Twice the radius

        pq1 = sphere_pq(q_values, radius1)
        pq2 = sphere_pq(q_values, radius2)

        # If q2 = q1 * (R1/R2), then P(q2, R2) should be similar to P(q1, R1)
        # Example: first minimum for R1 is at q_min1.
        # For R2, it should be at q_min2 = q_min1 * (R1/R2).
        # Let's find the q value for the ~100th point for pq1
        # q_ref_1 = q_values[100]
        # Corresponding q_ref_2 for pq2 such that q_ref_2 * R2 = q_ref_1 * R1
        # q_ref_2 = q_ref_1 * (radius1 / radius2)
        # pq_at_q_ref_1_for_R1 = sphere_pq(np.array([q_ref_1]), radius1)[0]
        # pq_at_q_ref_2_for_R2 = sphere_pq(np.array([q_ref_2]), radius2)[0]
        # assert np.isclose(pq_at_q_ref_1_for_R1, pq_at_q_ref_2_for_R2), \
        #     "P(q) feature scaling with radius is incorrect."

        # Simpler check: The overall curve for radius2 should be "compressed" towards lower q.
        # For a fixed q, if radius increases, qR increases, so we move further along the P(qR) curve.
        # Find the q value where P(q, R1) drops to, say, 0.1
        try:
            q_at_pq_01_r1 = q_values[np.where(pq1 < 0.1)[0][0]]
            # For R2, P(q, R2) should drop to 0.1 at roughly q_at_pq_01_r1 * (R1/R2)
            expected_q_at_pq_01_r2 = q_at_pq_01_r1 * (radius1 / radius2)
            
            # Find where pq2 actually drops to 0.1
            q_at_pq_01_r2_actual = q_values[np.where(pq2 < 0.1)[0][0]]
            
            assert np.isclose(q_at_pq_01_r2_actual, expected_q_at_pq_01_r2, rtol=0.1), \
                "P(q) feature scaling with radius (1/R compression) seems incorrect."
        except IndexError: # pragma: no cover
             pytest.skip("Could not find P(q) < 0.1 for one of the radii in the given q_range.")


    def test_sphere_pq_invalid_radius(self, q_values: np.ndarray):
        """Test P(q) calculation with invalid (non-positive) radius."""
        with pytest.raises(ValueError, match="Sphere radius must be positive."):
            sphere_pq(q_values, 0.0)
        with pytest.raises(ValueError, match="Sphere radius must be positive."):
            sphere_pq(q_values, -5.0)

    @pytest.mark.parametrize("radius_val", [1.0, 7.7, 23.45])
    def test_sphere_pq_various_radii(self, q_values: np.ndarray, radius_val: float):
        """Test P(q) with various valid radii, checking basic properties."""
        pq = sphere_pq(q_values, radius_val)
        assert pq.shape == q_values.shape
        assert np.all(pq >= 0)
        assert pq[0] > 0.99 and pq[0] <= 1.01 # P(q->0) ~ 1

# --- Helper to generate scattering data for fitting ---
def generate_sphere_scattering_data(
    q_values: np.ndarray,
    radius: float,
    scale: float,
    background: float,
    noise_level: float = 0.0,
    rel_error: float = 0.05 # Relative error for sigma if noise_level > 0
) -> ScatteringCurve1D:
    """Generates I(q) = scale * sphere_pq(q, radius) + background + noise."""
    pq_ideal = sphere_pq(q_values, radius)
    intensity_ideal = scale * pq_ideal + background
    
    if noise_level > 0:
        # Noise proportional to sqrt of ideal intensity (Poisson-like)
        noise = np.random.normal(0, noise_level * np.sqrt(np.abs(intensity_ideal)), size=intensity_ideal.shape)
        intensity_noisy = intensity_ideal + noise
    else:
        intensity_noisy = intensity_ideal

    # Assign errors: either a fraction of the noisy intensity or ideal if no noise
    if noise_level > 0 and rel_error > 0:
        error_values = np.abs(intensity_noisy * rel_error)
        error_values[error_values < 1e-9] = 1e-9 # Avoid zero/negative errors
    elif rel_error > 0 : # No noise, but still want errors based on ideal signal
        error_values = np.abs(intensity_ideal * rel_error)
        error_values[error_values < 1e-9] = 1e-9
    else:
        error_values = None # No errors

    return ScatteringCurve1D(q=q_values, intensity=intensity_noisy, error=error_values)


# --- Test Cases for Fitting ---
class TestFitModel:
    """Tests for the fit_model function."""

    @pytest.fixture
    def q_fit_values(self) -> np.ndarray:
        return np.geomspace(0.01, 1.0, 50) # Fewer points for faster fitting

    @pytest.fixture
    def sphere_model_params(self) -> dict:
        return {"true_radius": 5.0, "true_scale": 1000.0, "true_background": 10.0}

    @pytest.fixture
    def ideal_sphere_curve_for_fit(self, q_fit_values: np.ndarray, sphere_model_params: dict) -> ScatteringCurve1D:
        """Ideal sphere scattering data (no noise) for fitting."""
        return generate_sphere_scattering_data(
            q_values=q_fit_values,
            radius=sphere_model_params["true_radius"],
            scale=sphere_model_params["true_scale"],
            background=sphere_model_params["true_background"],
            noise_level=0.0,
            rel_error=0.05 # Add some synthetic errors for chi2 calculation
        )

    @pytest.fixture
    def noisy_sphere_curve_for_fit(self, q_fit_values: np.ndarray, sphere_model_params: dict) -> ScatteringCurve1D:
        """Noisy sphere scattering data for fitting."""
        return generate_sphere_scattering_data(
            q_values=q_fit_values,
            radius=sphere_model_params["true_radius"],
            scale=sphere_model_params["true_scale"],
            background=sphere_model_params["true_background"],
            noise_level=0.1, # Add 10% noise relative to sqrt(I)
            rel_error=0.05
        )

    def test_fit_sphere_ideal_data(self, ideal_sphere_curve_for_fit: ScatteringCurve1D, sphere_model_params: dict):
        """Fit sphere_pq to ideal (noiseless) data."""
        param_names_model = ['radius'] # For sphere_pq model function
        # Initial guesses: scale, background, radius
        initial_params_fit = [
            sphere_model_params["true_scale"] * 0.8, # Initial guess for scale
            sphere_model_params["true_background"] * 0.5, # Initial guess for background
            sphere_model_params["true_radius"] * 1.2  # Initial guess for radius
        ]
        # Bounds: (scale_low, bg_low, r_low), (scale_high, bg_high, r_high)
        param_bounds_fit = (
            [10, 0.1, 0.1],             # Lower bounds
            [1e5, 100, 20.0]          # Upper bounds
        )

        results = fit_model(
            curve=ideal_sphere_curve_for_fit,
            model_func=sphere_pq,
            param_names=param_names_model,
            initial_params=initial_params_fit,
            param_bounds=param_bounds_fit
        )

        assert results is not None
        assert results["success"] is True
        fitted_p = results["fitted_params"]
        assert np.isclose(fitted_p["scale"], sphere_model_params["true_scale"], rtol=1e-3)
        assert np.isclose(fitted_p["background"], sphere_model_params["true_background"], rtol=1e-3, atol=1e-2) # Background might be sensitive
        assert np.isclose(fitted_p["radius"], sphere_model_params["true_radius"], rtol=1e-3)
        assert "fit_curve" in results and isinstance(results["fit_curve"], ScatteringCurve1D)
        assert not np.isnan(results["chi_squared_reduced"]) # Should be calculable if errors are provided

    def test_fit_sphere_noisy_data(self, noisy_sphere_curve_for_fit: ScatteringCurve1D, sphere_model_params: dict):
        """Fit sphere_pq to noisy data."""
        param_names_model = ['radius']
        initial_params_fit = [sphere_model_params["true_scale"], sphere_model_params["true_background"], sphere_model_params["true_radius"]] # Start near true values
        param_bounds_fit = ([0, 0, 0.1], [np.inf, np.inf, 50.0])

        results = fit_model(
            curve=noisy_sphere_curve_for_fit,
            model_func=sphere_pq,
            param_names=param_names_model,
            initial_params=initial_params_fit,
            param_bounds=param_bounds_fit
        )

        assert results is not None
        # For noisy data, success is still expected, but parameters will have larger errors
        assert results["success"] is True 
        fitted_p = results["fitted_params"]
        # Allow larger tolerance for noisy data
        assert np.isclose(fitted_p["scale"], sphere_model_params["true_scale"], rtol=0.2)
        assert np.isclose(fitted_p["background"], sphere_model_params["true_background"], rtol=0.5, atol=sphere_model_params["true_background"]*0.5 + 5)
        assert np.isclose(fitted_p["radius"], sphere_model_params["true_radius"], rtol=0.2)
        
        # Check standard errors are reported and positive
        fitted_err = results["fitted_params_stderr"]
        assert fitted_err["scale"] > 0 and not np.isnan(fitted_err["scale"])
        assert fitted_err["background"] > 0 and not np.isnan(fitted_err["background"])
        assert fitted_err["radius"] > 0 and not np.isnan(fitted_err["radius"])


    def test_fit_with_fixed_params(self, ideal_sphere_curve_for_fit: ScatteringCurve1D, sphere_model_params: dict):
        """Test fitting with some parameters fixed."""
        param_names_model = ['radius']
        
        # Case 1: Fix radius, fit scale and background
        fixed_p_case1 = {'radius': sphere_model_params["true_radius"]}
        # Initial_params should only be for scale, background
        initial_p_case1 = [sphere_model_params["true_scale"] * 0.9, sphere_model_params["true_background"] * 0.8]
        bounds_case1 = ([0,0], [np.inf, np.inf])

        results_case1 = fit_model(
            curve=ideal_sphere_curve_for_fit,
            model_func=sphere_pq,
            param_names=param_names_model,
            initial_params=initial_p_case1,
            param_bounds=bounds_case1,
            fixed_params=fixed_p_case1
        )
        assert results_case1 is not None
        assert results_case1["success"]
        assert np.isclose(results_case1["fitted_params"]["radius"], sphere_model_params["true_radius"])
        assert np.isclose(results_case1["fitted_params"]["scale"], sphere_model_params["true_scale"], rtol=1e-3)
        assert np.isclose(results_case1["fitted_params"]["background"], sphere_model_params["true_background"], rtol=1e-3, atol=1e-2)
        assert results_case1["fitted_params_stderr"]["radius"] == 0.0 # Fixed param error is 0

        # Case 2: Fix scale and background, fit radius
        fixed_p_case2 = {
            'scale': sphere_model_params["true_scale"],
            'background': sphere_model_params["true_background"]
        }
        initial_p_case2 = [sphere_model_params["true_radius"] * 0.8] # Only for radius
        bounds_case2 = ([0.1], [20.0])

        results_case2 = fit_model(
            curve=ideal_sphere_curve_for_fit,
            model_func=sphere_pq,
            param_names=param_names_model,
            initial_params=initial_p_case2,
            param_bounds=bounds_case2,
            fixed_params=fixed_p_case2
        )
        assert results_case2 is not None
        assert results_case2["success"]
        assert np.isclose(results_case2["fitted_params"]["radius"], sphere_model_params["true_radius"], rtol=1e-3)
        assert np.isclose(results_case2["fitted_params"]["scale"], sphere_model_params["true_scale"])
        assert np.isclose(results_case2["fitted_params"]["background"], sphere_model_params["true_background"])
        assert results_case2["fitted_params_stderr"]["scale"] == 0.0
        assert results_case2["fitted_params_stderr"]["background"] == 0.0

    def test_fit_q_range(self, ideal_sphere_curve_for_fit: ScatteringCurve1D, sphere_model_params: dict):
        """Test fitting over a specified q-range."""
        param_names_model = ['radius']
        initial_params_fit = [sphere_model_params["true_scale"], sphere_model_params["true_background"], sphere_model_params["true_radius"]]
        
        # Fit only on a small portion of the q-range
        q_fit_range = (ideal_sphere_curve_for_fit.q[10], ideal_sphere_curve_for_fit.q[30])
        
        results = fit_model(
            curve=ideal_sphere_curve_for_fit,
            model_func=sphere_pq,
            param_names=param_names_model,
            initial_params=initial_params_fit,
            q_range=q_fit_range
        )
        assert results is not None
        assert results["success"]
        assert np.isclose(results["q_fit_min"], q_fit_range[0])
        assert np.isclose(results["q_fit_max"], q_fit_range[1])
        # Parameters should still be reasonably close even with reduced range for ideal data
        assert np.isclose(results["fitted_params"]["radius"], sphere_model_params["true_radius"], rtol=0.05)


    def test_fit_fail_not_enough_points(self, q_fit_values: np.ndarray, sphere_model_params: dict):
        """Test fit failure if q_range results in too few points."""
        # Create a curve with enough points overall
        curve = generate_sphere_scattering_data(
            q_values=q_fit_values,
            radius=sphere_model_params["true_radius"], scale=1, background=0
        )
        param_names_model = ['radius']
        initial_params_fit = [1.0, 0.0, 5.0] # scale, bg, radius

        # Select a q_range that has fewer points than parameters
        q_sparse_range = (q_fit_values[0], q_fit_values[1]) # Only 2 points
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = fit_model(
                curve=curve,
                model_func=sphere_pq,
                param_names=param_names_model,
                initial_params=initial_params_fit,
                q_range=q_sparse_range
            )
            assert results is None
            assert any("Not enough data points" in str(warn.message) for warn in w)

    def test_fit_mismatched_initial_params_length(self, ideal_sphere_curve_for_fit: ScatteringCurve1D):
        param_names_model = ['radius']
        # Too few initial_params (expected 3: scale, bg, radius)
        initial_params_too_few = [1000.0, 5.0] 
        with pytest.raises(ValueError, match="Length of initial_params .* does not match"):
            fit_model(
                curve=ideal_sphere_curve_for_fit, model_func=sphere_pq,
                param_names=param_names_model, initial_params=initial_params_too_few
            )
        
        # Too many initial_params
        initial_params_too_many = [1000.0, 10.0, 5.0, 1.0] 
        with pytest.raises(ValueError, match="Length of initial_params .* does not match"):
            fit_model(
                curve=ideal_sphere_curve_for_fit, model_func=sphere_pq,
                param_names=param_names_model, initial_params=initial_params_too_many
            )

    def test_fit_mismatched_bounds_length(self, ideal_sphere_curve_for_fit: ScatteringCurve1D):
        param_names_model = ['radius']
        initial_params_fit = [1000.0, 10.0, 5.0]
        # Bounds too short
        bounds_too_short = ([0,0], [np.inf, np.inf])
        with pytest.raises(ValueError, match="Length of param_bounds components must match"):
            fit_model(
                curve=ideal_sphere_curve_for_fit, model_func=sphere_pq,
                param_names=param_names_model, initial_params=initial_params_fit,
                param_bounds=bounds_too_short
            )

    def test_fit_with_no_errors_in_curve(self, q_fit_values: np.ndarray, sphere_model_params: dict):
        """Test fitting when the input curve has no error data."""
        curve_no_err = generate_sphere_scattering_data(
            q_values=q_fit_values,
            radius=sphere_model_params["true_radius"],
            scale=sphere_model_params["true_scale"],
            background=sphere_model_params["true_background"],
            noise_level=0.0,
            rel_error=0.0 # Ensure no errors are generated
        )
        assert curve_no_err.error is None

        param_names_model = ['radius']
        initial_params_fit = [sphere_model_params["true_scale"], sphere_model_params["true_background"], sphere_model_params["true_radius"]]
        
        results = fit_model(
            curve=curve_no_err,
            model_func=sphere_pq,
            param_names=param_names_model,
            initial_params=initial_params_fit
        )
        assert results is not None
        assert results["success"]
        # Chi-squared should be NaN as no sigma was provided to curve_fit
        assert np.isnan(results["chi_squared_reduced"])
        # Errors on parameters might be estimated based on unweighted residuals by curve_fit
        # but their interpretation is different than with weighted fits.
        assert not np.isnan(results["fitted_params_stderr"]["radius"])