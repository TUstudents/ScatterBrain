# tests/test_analysis.py
"""
Unit tests for the scatterbrain.analysis subpackage.
"""
import logging
import pytest
import numpy as np
import warnings

from scatterbrain.core import ScatteringCurve1D
from scatterbrain.analysis import guinier_fit, porod_analysis, scattering_invariant
from scatterbrain.utils import AnalysisError


# --- Helper function to generate ideal Guinier data ---
def generate_guinier_data(
    rg: float, i0: float, q_values: np.ndarray, noise_level: float = 0.0
) -> ScatteringCurve1D:
    """Generates ideal or noisy Guinier scattering data."""
    intensity = i0 * np.exp(-(q_values**2 * rg**2) / 3.0)
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * intensity, size=intensity.shape)
        intensity += noise
        intensity[intensity <= 0] = 1e-9  # Ensure positive intensity for log

    # Generate some plausible errors if noise is present
    error = noise_level * intensity if noise_level > 0 else np.sqrt(intensity) * 0.01
    error[error <= 0] = 1e-9  # Ensure positive error

    return ScatteringCurve1D(
        q=q_values,
        intensity=intensity,
        error=error,
        metadata={"ideal_Rg": rg, "ideal_I0": i0},
    )


# --- Helper function to generate ideal Porod data ---
def generate_porod_data(
    kp: float,
    n: float,
    q_values: np.ndarray,
    background: float = 0.0,
    noise_level: float = 0.0,
) -> ScatteringCurve1D:
    """Generates ideal or noisy Porod scattering data I(q) = Kp * q^-n + Bkg."""
    intensity = kp * (q_values**-n) + background
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * intensity, size=intensity.shape)
        intensity += noise
        intensity[intensity <= 0] = 1e-9  # Ensure positive intensity

    error = (
        noise_level * intensity
        if noise_level > 0
        else np.sqrt(np.abs(intensity)) * 0.05
    )  # Error even on background
    error[error <= 0] = 1e-9

    return ScatteringCurve1D(
        q=q_values,
        intensity=intensity,
        error=error,
        metadata={"ideal_Kp": kp, "ideal_n": n, "background": background},
    )


# --- Fixtures ---
@pytest.fixture
def ideal_guinier_curve() -> ScatteringCurve1D:
    """An ideal Guinier curve with known Rg and I0."""
    q = np.linspace(
        0.01, 0.5, 100
    )  # q up to q*Rg ~ 0.5 * 5 = 2.5 (beyond typical Guinier)
    return generate_guinier_data(rg=5.0, i0=1000.0, q_values=q)


@pytest.fixture
def noisy_guinier_curve() -> ScatteringCurve1D:
    """A noisy Guinier curve."""
    np.random.seed(7)  # Deterministic noise for reproducible auto-range selection
    q = np.linspace(0.01, 0.3, 100)  # Reduced q-range to avoid high-q noise effects
    return generate_guinier_data(
        rg=3.0, i0=500.0, q_values=q, noise_level=0.03
    )  # Reduced noise level


@pytest.fixture
def flat_curve() -> ScatteringCurve1D:
    """A curve with flat intensity, should yield positive slope or NaN Rg."""
    q = np.linspace(0.1, 1.0, 50)
    intensity = np.full_like(q, 100.0)
    return ScatteringCurve1D(q=q, intensity=intensity, error=np.full_like(q, 10.0))


@pytest.fixture
def very_few_points_curve() -> ScatteringCurve1D:
    """A curve with too few points."""
    q = np.array([0.1, 0.2, 0.3])
    intensity = np.array([100, 90, 80])
    return ScatteringCurve1D(q=q, intensity=intensity)


# --- Fixtures for Porod ---
@pytest.fixture
def ideal_porod_curve_n4() -> ScatteringCurve1D:
    """Ideal Porod curve with n=4, Kp=1.0, no background."""
    # Use higher q values for Porod region
    q = np.geomspace(0.5, 10.0, 100)
    return generate_porod_data(kp=1.0, n=4.0, q_values=q, background=0.0)


@pytest.fixture
def ideal_porod_curve_n3() -> ScatteringCurve1D:
    """Ideal Porod curve with n=3 (e.g., mass fractal), Kp=0.5, no background."""
    q = np.geomspace(0.5, 10.0, 100)
    return generate_porod_data(kp=0.5, n=3.0, q_values=q, background=0.0)


@pytest.fixture
def porod_curve_with_bkg() -> ScatteringCurve1D:
    """Porod curve with n=4 and a constant background."""
    q = np.geomspace(0.5, 10.0, 100)
    # Note: current porod_analysis doesn't subtract background, so fit will be affected.
    # This fixture is for testing how it behaves with background.
    return generate_porod_data(kp=1.0, n=4.0, q_values=q, background=0.1)


# --- Test Cases for guinier_fit ---


class TestGuinierFit:
    """Tests for the guinier_fit function.

    Test Categories:
    - Basic Functionality: Tests with ideal and noisy data
    - Error Handling: Tests for various error conditions
    - Q-Range Selection: Tests for q-range selection logic
    - Input Validation: Tests for input parameter validation

    Each test validates the Guinier fit parameters (Rg, I0) and checks
    appropriate error handling and warning generation.
    """

    # --- Basic Functionality Tests ---

    def test_ideal_curve_manual_q_range(self, ideal_guinier_curve: ScatteringCurve1D):
        """Test Guinier fit with ideal data and manual q-range.

        Validates:
        - Accurate Rg and I0 recovery (within 0.1% tolerance)
        - Q-range constraints are respected
        - Fit metadata is properly populated
        """
        # For Rg=5, qRg_max=1.3 means q_max ~ 1.3/5 = 0.26
        q_min_fit, q_max_fit = 0.01, 0.25
        results = guinier_fit(ideal_guinier_curve, q_range=(q_min_fit, q_max_fit))

        assert results is not None
        assert "Rg" in results and "I0" in results
        assert np.isclose(
            results["Rg"], 5.0, rtol=1e-3
        )  # Ideal data, should be very close
        assert np.isclose(results["I0"], 1000.0, rtol=1e-3)
        assert results["q_fit_min"] >= q_min_fit
        assert results["q_fit_max"] <= q_max_fit
        assert results["num_points_fit"] > 0
        assert "Manual q_range" in results["valid_guinier_range_criteria"]

    def test_ideal_curve_auto_q_range(self, ideal_guinier_curve: ScatteringCurve1D):
        """Test Guinier fit with ideal data and automatic q-range selection.

        Validates:
        - Accurate parameter recovery with auto q-range
        - Q-range selection respects qRg limits
        - Fit metadata reflects automatic selection
        """
        # For Rg=5, qRg_max=1.3 -> q_max around 0.26
        results = guinier_fit(ideal_guinier_curve, qrg_limit_max=1.3)

        assert results is not None
        assert np.isclose(
            results["Rg"], 5.0, rtol=1e-2
        )  # Auto range might be slightly less precise
        assert np.isclose(results["I0"], 1000.0, rtol=1e-2)
        assert results["num_points_fit"] > 0
        assert "Automatic q_range selection" in results["valid_guinier_range_criteria"]
        assert results["q_fit_max"] < 0.3  # Based on qRg_max=1.3 for Rg=5

    def test_noisy_curve_auto_q_range(self, noisy_guinier_curve: ScatteringCurve1D):
        """Test with noisy data and automatic q-range."""
        results = guinier_fit(
            noisy_guinier_curve, qrg_limit_max=1.3, auto_q_selection_fraction=0.3
        )  # Use more points for initial estimate

        assert results is not None
        assert np.isclose(
            results["Rg"], 3.0, rtol=0.15
        )  # Increased tolerance for noisy data
        assert np.isclose(results["I0"], 500.0, rtol=0.15)
        assert results["Rg_err"] > 0 and not np.isnan(results["Rg_err"])
        assert results["I0_err"] > 0 and not np.isnan(results["I0_err"])
        assert "Auto" in results["valid_guinier_range_criteria"]

    # --- Error Handling Tests ---

    def test_positive_slope_warning(self, flat_curve: ScatteringCurve1D, caplog):
        """Test handling of data yielding positive Guinier slope.

        Validates:
        - NaN results for invalid fit
        - Appropriate warning generation
        - Correct flagging in fit criteria
        """
        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            # use_errors=False forces OLS so slope=0 exactly for flat data
            results = guinier_fit(
                flat_curve, q_range=(0.1, 0.5), use_errors=False
            )  # Manual range to ensure fit attempt

        assert results is not None
        assert np.isnan(results["Rg"])
        assert np.isnan(results["I0"])
        assert results["slope"] >= 0  # or very close to 0
        assert any("non-negative slope" in r.message for r in caplog.records)
        assert "(Warning: Positive Slope)" in results["valid_guinier_range_criteria"]

    def test_insufficient_points_initial(
        self, very_few_points_curve: ScatteringCurve1D, caplog
    ):
        """Test case with too few points overall."""
        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            results = guinier_fit(very_few_points_curve, min_points=5)
        assert results is None
        assert any("Insufficient data points" in r.message for r in caplog.records)

    def test_insufficient_points_after_q_range_selection(
        self, ideal_guinier_curve: ScatteringCurve1D, caplog
    ):
        """Test when q-range selection results in too few points."""
        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            # Select a q_range that will contain very few points from the ideal curve
            results = guinier_fit(
                ideal_guinier_curve, q_range=(0.001, 0.005), min_points=5
            )
        assert results is None
        assert any(
            "Selected q-range resulted in" in r.message
            and "less than min_points" in r.message
            for r in caplog.records
        )

    def test_no_positive_intensity(self, caplog):
        """Test curve with no positive intensity values."""
        q = np.array([0.1, 0.2, 0.3])
        intensity = np.array([-1.0, -0.5, 0.0])
        curve = ScatteringCurve1D(q, intensity)
        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            results = guinier_fit(curve)
        assert results is None
        assert any("No positive intensity values" in r.message for r in caplog.records)

    def test_qrg_limits_effect(self, ideal_guinier_curve: ScatteringCurve1D):
        """Test the effect of qRg_limit_min and qRg_limit_max."""
        # Ideal Rg = 5.0
        # Test with tight qRg_max
        results_tight_max = guinier_fit(
            ideal_guinier_curve, qrg_limit_max=0.8
        )  # Should yield q_max ~ 0.8/5 = 0.16
        assert results_tight_max is not None
        assert (
            results_tight_max["q_fit_max"] < 0.20
        )  # Check if q_max was indeed restricted

        # Test with qRg_min
        results_with_min = guinier_fit(
            ideal_guinier_curve, qrg_limit_min=0.5, qrg_limit_max=1.3
        )  # q_min ~ 0.5/5 = 0.1
        assert results_with_min is not None
        assert results_with_min["q_fit_min"] > 0.09  # Check if q_min was respected

    def test_manual_q_range_partially_none(
        self, ideal_guinier_curve: ScatteringCurve1D
    ):
        """Test manual q_range with one boundary as None."""
        # Test q_min specified, q_max auto (effectively from data max if no qRg constraint applied from auto)
        results_min_only = guinier_fit(
            ideal_guinier_curve, q_range=(0.1, None), qrg_limit_max=None
        )
        assert results_min_only is not None
        assert np.isclose(results_min_only["q_fit_min"], 0.1, rtol=1e-1)
        assert np.isclose(
            results_min_only["q_fit_max"], ideal_guinier_curve.q.max()
        )  # Uses max of data

        # Test q_max specified, q_min auto
        results_max_only = guinier_fit(
            ideal_guinier_curve, q_range=(None, 0.2), qrg_limit_max=None
        )
        assert results_max_only is not None
        assert np.isclose(results_max_only["q_fit_min"], ideal_guinier_curve.q.min())
        assert np.isclose(results_max_only["q_fit_max"], 0.2, rtol=1e-1)

    def test_auto_q_selection_fallback_positive_slope(self, caplog):
        """Test auto q-selection fallback when initial fit has positive slope."""
        # Create data that will definitely have positive slope initially
        q = np.linspace(0.01, 0.5, 50)
        # Create base intensity that decreases with q
        intensity = 1000 * np.exp(-(q**2 * 5**2) / 3.0)
        # Create strong upturn at low-q
        intensity[:5] = intensity[0] * np.exp(
            np.linspace(0.0, 1.0, 5)
        )  # Exponential upturn
        curve_mod = ScatteringCurve1D(q, intensity)

        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            results = guinier_fit(
                curve_mod,
                auto_q_selection_fraction=0.05,  # Use very few points initially
                min_points=3,
            )

        assert results is not None
        assert any(
            "non-negative slope" in r.message for r in caplog.records
        ), "No positive slope warning found"
        assert (
            "Fallback" in results["valid_guinier_range_criteria"]
        ), "No fallback mentioned in criteria"
        assert not np.isnan(results["Rg"]), "Rg should not be NaN after fallback"

    # --- Input Validation Tests ---

    def test_invalid_input_type(self):
        """Test type validation for input curve.

        Validates:
        - AnalysisError for non-ScatteringCurve1D input
        - Descriptive error message
        """
        with pytest.raises(
            AnalysisError, match="Input 'curve' must be a ScatteringCurve1D object."
        ):
            guinier_fit("not_a_curve")


class TestGuinierFitWeighted:
    """Tests for the weighted (WLS) path in guinier_fit."""

    def test_weighted_key_present_with_errors(self) -> None:
        q = np.linspace(0.01, 0.25, 50)
        i = 1000.0 * np.exp(-(q**2) * 25.0 / 3.0)
        e = i * 0.02
        curve = ScatteringCurve1D(q, i, e)
        result = guinier_fit(curve, q_range=(0.01, 0.25))
        assert result is not None
        assert "weighted" in result
        assert result["weighted"] is True

    def test_weighted_key_false_when_use_errors_false(self) -> None:
        q = np.linspace(0.01, 0.25, 50)
        i = 1000.0 * np.exp(-(q**2) * 25.0 / 3.0)
        e = i * 0.02
        curve = ScatteringCurve1D(q, i, e)
        result = guinier_fit(curve, q_range=(0.01, 0.25), use_errors=False)
        assert result is not None
        assert result["weighted"] is False

    def test_weighted_key_false_when_no_errors(self) -> None:
        q = np.linspace(0.01, 0.25, 50)
        i = 1000.0 * np.exp(-(q**2) * 25.0 / 3.0)
        curve = ScatteringCurve1D(q, i)
        result = guinier_fit(curve, q_range=(0.01, 0.25))
        assert result is not None
        assert result["weighted"] is False

    def test_weighted_and_unweighted_differ_for_heterogeneous_errors(self) -> None:
        """With heterogeneous errors and noisy data, WLS and OLS give different slopes."""
        rng = np.random.default_rng(0)
        q = np.linspace(0.01, 0.25, 40)
        i_true = 1000.0 * np.exp(-(q**2) * 25.0 / 3.0)
        # Strongly heterogeneous errors: large at low q, tiny at high q
        e = np.linspace(i_true[0] * 0.5, i_true[-1] * 0.001, 40)
        # Add noise scaled to the error so the noisy data is dominated by low-q deviations
        i = i_true + rng.normal(0, e)
        i = np.maximum(i, 1e-3)
        curve = ScatteringCurve1D(q, i, e)

        result_wls = guinier_fit(curve, q_range=(0.01, 0.25), use_errors=True)
        result_ols = guinier_fit(curve, q_range=(0.01, 0.25), use_errors=False)
        assert result_wls is not None
        assert result_ols is not None
        # The slopes must differ when errors are strongly heterogeneous and data is noisy
        assert not np.isclose(result_wls["slope"], result_ols["slope"], rtol=1e-4)

    def test_weighted_rg_accurate_for_ideal_data(self) -> None:
        """WLS recovers correct Rg for ideal Guinier data with proportional errors."""
        q = np.linspace(0.01, 0.25, 100)
        rg_true = 5.0
        i = 1000.0 * np.exp(-(q**2) * rg_true**2 / 3.0)
        e = i * 0.01  # proportional errors (homogeneous in ln space)
        curve = ScatteringCurve1D(q, i, e)
        result = guinier_fit(curve, q_range=(0.01, 0.25))
        assert result is not None
        assert result["weighted"] is True
        assert np.isclose(result["Rg"], rg_true, rtol=1e-3)
        assert np.isclose(result["I0"], 1000.0, rtol=1e-3)


# --- Test Cases for porod_analysis ---
class TestPorodAnalysis:
    """Tests for the porod_analysis function."""

    def test_ideal_porod_n4_log_log_fit(self, ideal_porod_curve_n4: ScatteringCurve1D):
        """Test log-log fit on ideal Porod data with n=4."""
        # Use a reasonable q-range for Porod (high q)
        q_min_fit, q_max_fit = 2.0, 8.0
        results = porod_analysis(
            ideal_porod_curve_n4, q_range=(q_min_fit, q_max_fit), fit_log_log=True
        )

        assert results is not None
        assert "porod_exponent" in results and "porod_constant_kp" in results
        assert np.isclose(results["porod_exponent"], 4.0, rtol=1e-3)
        assert np.isclose(results["porod_constant_kp"], 1.0, rtol=1e-3)
        assert results["q_fit_min"] >= q_min_fit
        assert results["q_fit_max"] <= q_max_fit
        assert "Log-log fit" in results["method"]

    def test_ideal_porod_n3_log_log_fit_auto_q(
        self, ideal_porod_curve_n3: ScatteringCurve1D
    ):
        """Test log-log fit on ideal Porod data with n=3 and auto q-range."""
        results = porod_analysis(
            ideal_porod_curve_n3, fit_log_log=True, q_fraction_high=0.5
        )  # Use last 50%

        assert results is not None
        assert np.isclose(
            results["porod_exponent"], 3.0, rtol=1e-2
        )  # Auto range might be less precise
        assert np.isclose(results["porod_constant_kp"], 0.5, rtol=1e-2)
        # On perfectly noiseless data linregress correctly returns stderr=0
        # (SSR=0, no residuals), so assert >= 0 rather than > 0.
        assert results["porod_exponent_err"] >= 0 and not np.isnan(
            results["porod_exponent_err"]
        )
        assert results["porod_constant_kp_err"] >= 0 and not np.isnan(
            results["porod_constant_kp_err"]
        )
        assert "Automatic q_range" in results["method"]

    def test_average_kp_mode(self, ideal_porod_curve_n4: ScatteringCurve1D):
        """Test average Kp calculation mode with expected_exponent=4."""
        q_min_fit, q_max_fit = 2.0, 8.0
        results = porod_analysis(
            ideal_porod_curve_n4,
            q_range=(q_min_fit, q_max_fit),
            fit_log_log=False,
            expected_exponent=4.0,
        )
        assert results is not None
        assert np.isclose(results["porod_constant_kp"], 1.0, rtol=1e-3)
        assert results["porod_exponent"] == 4.0  # Should return the expected_exponent
        assert np.isnan(results["porod_exponent_err"])  # No fit for exponent error
        assert "Average Kp" in results["method"]

    def test_porod_with_background_log_log_fit(
        self, porod_curve_with_bkg: ScatteringCurve1D
    ):
        """Test log-log fit when background is present (expect deviation)."""
        # Kp=1, n=4, Bkg=0.1
        # The fit will be distorted by the background if not subtracted.
        # Exponent will likely be lower than 4, Kp will be affected.
        results = porod_analysis(
            porod_curve_with_bkg, q_range=(2.0, 8.0), fit_log_log=True
        )

        assert results is not None
        # Exact values are hard to predict without knowing the distortion precisely,
        # but exponent should be < 4 if background is positive.
        assert results["porod_exponent"] < 4.0
        # Kp will also be off.
        assert not np.isclose(results["porod_constant_kp"], 1.0, rtol=1e-2)
        warnings.warn(
            "Porod test with background: Note that current porod_analysis does not "
            "subtract background, so fitted parameters will deviate from ideal Kp=1, n=4.",
            UserWarning,
            stacklevel=2,
        )

    def test_insufficient_points_porod(
        self, very_few_points_curve: ScatteringCurve1D, caplog
    ):
        """Test Porod analysis with too few points."""
        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            results = porod_analysis(very_few_points_curve, min_points=5)
        assert results is None
        assert any("Insufficient data points" in r.message for r in caplog.records)

    def test_q_range_yields_too_few_points_porod(
        self, ideal_porod_curve_n4: ScatteringCurve1D, caplog
    ):
        """Test when selected q-range for Porod has too few points."""
        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            results = porod_analysis(
                ideal_porod_curve_n4, q_range=(8.0, 9.0), min_points=10
            )  # This range has <10 pts
        assert results is None
        assert any(
            "Selected q-range resulted in" in r.message
            and "less than min_points" in r.message
            for r in caplog.records
        )

    def test_no_positive_intensity_or_q_porod(self, caplog):
        """Test Porod with no valid (I>0, q>0) data."""
        q_neg = np.array([-0.1, -0.2, -0.3])
        i_pos = np.array([1.0, 0.5, 0.2])
        curve_neg_q = ScatteringCurve1D(q_neg, i_pos)

        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            results = porod_analysis(curve_neg_q)
        assert results is None
        assert any(
            "No positive intensity and q values found" in r.message
            for r in caplog.records
        )

        caplog.clear()
        q_pos = np.array([0.1, 0.2, 0.3])
        i_neg = np.array([-1.0, -0.5, -0.2])
        curve_neg_i = ScatteringCurve1D(q_pos, i_neg)
        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            results = porod_analysis(curve_neg_i)
        assert results is None
        assert any(
            "No positive intensity and q values found" in r.message
            for r in caplog.records
        )

    def test_average_kp_mode_no_expected_exponent(
        self, ideal_porod_curve_n4: ScatteringCurve1D, caplog
    ):
        """Test average Kp mode if expected_exponent is not provided."""
        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            results = porod_analysis(
                ideal_porod_curve_n4, fit_log_log=False, expected_exponent=None
            )
        assert results is None
        assert any(
            "'expected_exponent' must be provided if 'fit_log_log' is False"
            in r.message
            for r in caplog.records
        )

    def test_invalid_input_type_porod(self):
        """Test passing an invalid type for the curve argument to Porod."""
        with pytest.raises(
            AnalysisError, match="Input 'curve' must be a ScatteringCurve1D object."
        ):
            porod_analysis("not_a_curve")


# --- Test Cases for scattering_invariant ---


class TestScatteringInvariant:
    """Tests for the scattering_invariant function."""

    @pytest.fixture
    def flat_q2_curve(self) -> ScatteringCurve1D:
        """Curve with I(q) = 1 so q^2 * I = q^2; integral = q^3/3."""
        q = np.linspace(0.1, 1.0, 200)
        i = np.ones_like(q)
        return ScatteringCurve1D(q, i)

    @pytest.fixture
    def power_law_curve(self) -> ScatteringCurve1D:
        """I(q) = Kp * q^-4, Kp=1."""
        q = np.geomspace(0.1, 2.0, 300)
        i = q**-4
        return ScatteringCurve1D(q, i)

    def test_numerical_integral_matches_trapz(
        self, flat_q2_curve: ScatteringCurve1D
    ) -> None:
        result = scattering_invariant(flat_q2_curve)
        assert result is not None
        expected = float(
            np.trapezoid(flat_q2_curve.q**2 * flat_q2_curve.intensity, flat_q2_curve.q)
        )
        assert np.isclose(result["Q_star"], expected, rtol=1e-10)

    def test_result_keys_present(self, flat_q2_curve: ScatteringCurve1D) -> None:
        result = scattering_invariant(flat_q2_curve)
        assert result is not None
        for key in (
            "Q_star",
            "Q_star_low_q",
            "Q_star_high_q",
            "Q_star_total",
            "q_min",
            "q_max",
            "num_points",
            "extrapolation_method",
        ):
            assert key in result, f"Missing key: {key}"

    def test_porod_high_q_extrapolation_n4(
        self, flat_q2_curve: ScatteringCurve1D
    ) -> None:
        kp = 2.5
        n = 4.0
        q_max = float(flat_q2_curve.q.max())
        porod_result = {"porod_constant_kp": kp, "porod_exponent": n}
        result = scattering_invariant(flat_q2_curve, porod_result=porod_result)
        assert result is not None
        expected_high_q = kp * q_max ** (3.0 - n) / (n - 3.0)
        assert np.isclose(result["Q_star_high_q"], expected_high_q, rtol=1e-10)

    def test_porod_exponent_le3_gives_nan_high_q(
        self, flat_q2_curve: ScatteringCurve1D, caplog: pytest.LogCaptureFixture
    ) -> None:
        porod_result = {"porod_constant_kp": 1.0, "porod_exponent": 3.0}
        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            result = scattering_invariant(flat_q2_curve, porod_result=porod_result)
        assert result is not None
        assert np.isnan(result["Q_star_high_q"])
        assert any("diverges" in r.message for r in caplog.records)

    def test_guinier_low_q_adds_positive_contribution(
        self, flat_q2_curve: ScatteringCurve1D
    ) -> None:
        guinier_result = {"Rg": 2.0, "I0": 100.0}
        result = scattering_invariant(flat_q2_curve, guinier_result=guinier_result)
        assert result is not None
        assert result["Q_star_low_q"] > 0.0

    def test_soft_failure_on_too_few_points(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        q = np.linspace(0.1, 0.2, 3)
        i = np.ones(3)
        curve = ScatteringCurve1D(q, i)
        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            result = scattering_invariant(curve)
        assert result is None
        assert any("valid points" in r.message for r in caplog.records)

    def test_q_range_selection(self, flat_q2_curve: ScatteringCurve1D) -> None:
        q_min, q_max = 0.3, 0.7
        result = scattering_invariant(flat_q2_curve, q_range=(q_min, q_max))
        assert result is not None
        assert result["q_min"] >= q_min
        assert result["q_max"] <= q_max

    def test_invalid_curve_type_raises(self) -> None:
        with pytest.raises(AnalysisError, match="ScatteringCurve1D"):
            scattering_invariant("not_a_curve")  # type: ignore[arg-type]

    def test_total_without_extrapolations(
        self, flat_q2_curve: ScatteringCurve1D
    ) -> None:
        result = scattering_invariant(flat_q2_curve)
        assert result is not None
        # Without extrapolations, Q_star_low_q=0, Q_star_high_q=0, total = Q_star
        assert np.isclose(result["Q_star_total"], result["Q_star"], rtol=1e-10)
        assert result["extrapolation_method"] == "none"
