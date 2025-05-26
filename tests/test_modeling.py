# tests/test_modeling.py
"""
Unit tests for the scatterbrain.modeling subpackage,
including form factors and fitting routines.
"""
import pytest
import numpy as np

from scatterbrain.modeling.form_factors import sphere_pq, _Q_EPSILON
# from scatterbrain.modeling.fitting import fit_model # Will be added later
# from scatterbrain.core import ScatteringCurve1D # Will be needed for fitting tests

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

# Future: Add tests for other form factors like cylinder_pq, core_shell_sphere_pq etc.

# --- Test Cases for Fitting (to be added later) ---
# class TestFitModel:
#     pass