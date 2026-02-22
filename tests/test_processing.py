# tests/test_processing.py
"""
Unit tests for scatterbrain.processing.background.subtract_background.
"""

import pytest
import numpy as np

from scatterbrain.core import ScatteringCurve1D
from scatterbrain.processing import subtract_background, normalize
from scatterbrain.utils import ProcessingError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_curve():
    q = np.linspace(0.1, 1.0, 20)
    intensity = 100.0 * np.exp(-(q**2) * 3**2 / 3)
    error = np.sqrt(intensity) * 0.05
    return ScatteringCurve1D(q, intensity, error, q_unit="nm^-1", intensity_unit="a.u.")


@pytest.fixture
def base_curve_no_error():
    q = np.linspace(0.1, 1.0, 20)
    intensity = 100.0 * np.exp(-(q**2) * 3**2 / 3)
    return ScatteringCurve1D(q, intensity, q_unit="nm^-1", intensity_unit="a.u.")


@pytest.fixture
def matching_bg_curve(base_curve):
    """Background on same q-grid as base_curve."""
    bg_intensity = np.full_like(base_curve.intensity, 5.0)
    bg_error = np.full_like(base_curve.intensity, 0.5)
    return ScatteringCurve1D(
        base_curve.q.copy(),
        bg_intensity,
        bg_error,
        q_unit="nm^-1",
        intensity_unit="a.u.",
    )


@pytest.fixture
def coarser_bg_curve(base_curve):
    """Background on a coarser (but covering) q-grid."""
    q_bg = np.linspace(0.05, 1.2, 8)
    bg_intensity = np.full_like(q_bg, 5.0)
    return ScatteringCurve1D(q_bg, bg_intensity, q_unit="nm^-1", intensity_unit="a.u.")


# ---------------------------------------------------------------------------
# Constant background
# ---------------------------------------------------------------------------


class TestConstantBackground:
    def test_basic_subtraction(self, base_curve):
        result = subtract_background(base_curve, 5.0)
        np.testing.assert_allclose(
            result.intensity, base_curve.intensity - 5.0, rtol=1e-10
        )

    def test_q_unchanged(self, base_curve):
        result = subtract_background(base_curve, 5.0)
        np.testing.assert_array_equal(result.q, base_curve.q)

    def test_error_unchanged(self, base_curve):
        result = subtract_background(base_curve, 5.0)
        np.testing.assert_array_equal(result.error, base_curve.error)

    def test_no_error_preserved(self, base_curve_no_error):
        result = subtract_background(base_curve_no_error, 5.0)
        assert result.error is None

    def test_scale_factor(self, base_curve):
        result = subtract_background(base_curve, 2.0, scale_factor=3.0)
        np.testing.assert_allclose(
            result.intensity, base_curve.intensity - 6.0, rtol=1e-10
        )

    def test_integer_background(self, base_curve):
        result = subtract_background(base_curve, 5)
        np.testing.assert_allclose(
            result.intensity, base_curve.intensity - 5.0, rtol=1e-10
        )

    def test_does_not_modify_original(self, base_curve):
        original_intensity = base_curve.intensity.copy()
        subtract_background(base_curve, 5.0)
        np.testing.assert_array_equal(base_curve.intensity, original_intensity)

    def test_processing_history_updated(self, base_curve):
        result = subtract_background(base_curve, 5.0)
        assert any(
            "background" in entry.lower()
            for entry in result.metadata["processing_history"]
        )

    def test_units_preserved(self, base_curve):
        result = subtract_background(base_curve, 5.0)
        assert result.q_unit == base_curve.q_unit
        assert result.intensity_unit == base_curve.intensity_unit


# ---------------------------------------------------------------------------
# Curve background — matching grid
# ---------------------------------------------------------------------------


class TestCurveBackgroundMatchingGrid:
    def test_basic_subtraction(self, base_curve, matching_bg_curve):
        result = subtract_background(base_curve, matching_bg_curve)
        np.testing.assert_allclose(
            result.intensity, base_curve.intensity - 5.0, rtol=1e-10
        )

    def test_error_propagation_in_quadrature(self, base_curve, matching_bg_curve):
        result = subtract_background(base_curve, matching_bg_curve)
        expected_err = np.sqrt(base_curve.error**2 + matching_bg_curve.error**2)
        np.testing.assert_allclose(result.error, expected_err, rtol=1e-10)

    def test_scale_factor(self, base_curve, matching_bg_curve):
        result = subtract_background(base_curve, matching_bg_curve, scale_factor=0.5)
        np.testing.assert_allclose(
            result.intensity, base_curve.intensity - 2.5, rtol=1e-10
        )

    def test_signal_no_error_bg_has_error(self, base_curve_no_error, matching_bg_curve):
        result = subtract_background(base_curve_no_error, matching_bg_curve)
        # Only bg error contributes
        np.testing.assert_allclose(result.error, matching_bg_curve.error, rtol=1e-10)

    def test_both_no_error(self, base_curve_no_error):
        bg = ScatteringCurve1D(
            base_curve_no_error.q.copy(),
            np.full_like(base_curve_no_error.intensity, 3.0),
        )
        result = subtract_background(base_curve_no_error, bg)
        assert result.error is None


# ---------------------------------------------------------------------------
# Curve background — interpolation
# ---------------------------------------------------------------------------


class TestCurveBackgroundInterpolation:
    def test_interpolated_subtraction(self, base_curve, coarser_bg_curve):
        result = subtract_background(base_curve, coarser_bg_curve, interpolate=True)
        # Background is constant 5.0 everywhere, so result should be signal - 5
        np.testing.assert_allclose(
            result.intensity, base_curve.intensity - 5.0, atol=1e-10
        )

    def test_no_interpolate_raises_on_mismatched_grids(
        self, base_curve, coarser_bg_curve
    ):
        with pytest.raises(ProcessingError, match="interpolate"):
            subtract_background(base_curve, coarser_bg_curve, interpolate=False)

    def test_no_interpolate_ok_on_matching_grids(self, base_curve, matching_bg_curve):
        result = subtract_background(base_curve, matching_bg_curve, interpolate=False)
        np.testing.assert_allclose(
            result.intensity, base_curve.intensity - 5.0, rtol=1e-10
        )

    def test_warning_when_bg_doesnt_cover_signal(self, base_curve):
        # Background only covers part of signal q-range; should log but not raise.
        q_narrow = np.linspace(0.3, 0.7, 10)
        bg_narrow = ScatteringCurve1D(
            q_narrow, np.full(10, 5.0), q_unit="nm^-1", intensity_unit="a.u."
        )
        result = subtract_background(base_curve, bg_narrow, interpolate=True)
        assert result is not None


# ---------------------------------------------------------------------------
# Type errors
# ---------------------------------------------------------------------------


class TestTypeErrors:
    def test_curve_not_scatteringcurve1d(self):
        with pytest.raises(TypeError, match="ScatteringCurve1D"):
            subtract_background("not a curve", 5.0)

    def test_background_wrong_type(self, base_curve):
        with pytest.raises(TypeError, match="ScatteringCurve1D"):
            subtract_background(base_curve, [1, 2, 3])


# ---------------------------------------------------------------------------
# Tests for normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    """Tests for the normalize processing function."""

    @pytest.fixture
    def curve_with_error(self) -> ScatteringCurve1D:
        q = np.linspace(0.01, 0.5, 50)
        intensity = 1000.0 * np.exp(-(q**2) * 9.0)
        error = intensity * 0.05
        return ScatteringCurve1D(
            q, intensity, error, q_unit="nm^-1", intensity_unit="a.u."
        )

    @pytest.fixture
    def curve_no_error(self) -> ScatteringCurve1D:
        q = np.linspace(0.01, 0.5, 50)
        intensity = 500.0 * np.ones(50)
        return ScatteringCurve1D(q, intensity, q_unit="nm^-1", intensity_unit="a.u.")

    def test_intensity_divided_by_factor(
        self, curve_with_error: ScatteringCurve1D
    ) -> None:
        result = normalize(curve_with_error, 2.5)
        np.testing.assert_allclose(result.intensity, curve_with_error.intensity / 2.5)

    def test_error_divided_by_factor_without_factor_error(
        self, curve_with_error: ScatteringCurve1D
    ) -> None:
        result = normalize(curve_with_error, 4.0)
        np.testing.assert_allclose(result.error, curve_with_error.error / 4.0)

    def test_error_propagation_with_factor_error(
        self, curve_with_error: ScatteringCurve1D
    ) -> None:
        factor = 4.0
        factor_error = 0.2
        result = normalize(curve_with_error, factor, factor_error=factor_error)
        expected = np.sqrt(
            (curve_with_error.error / factor) ** 2
            + (curve_with_error.intensity * factor_error / factor**2) ** 2
        )
        np.testing.assert_allclose(result.error, expected)

    def test_no_error_on_input_gives_none_error(
        self, curve_no_error: ScatteringCurve1D
    ) -> None:
        result = normalize(curve_no_error, 3.0)
        assert result.error is None

    def test_original_curve_not_modified(
        self, curve_with_error: ScatteringCurve1D
    ) -> None:
        original_intensity = curve_with_error.intensity.copy()
        original_error = curve_with_error.error.copy()
        normalize(curve_with_error, 2.0)
        np.testing.assert_array_equal(curve_with_error.intensity, original_intensity)
        np.testing.assert_array_equal(curve_with_error.error, original_error)

    def test_processing_history_without_factor_error(
        self, curve_with_error: ScatteringCurve1D
    ) -> None:
        result = normalize(curve_with_error, 2.5)
        history = result.metadata["processing_history"]
        normalize_entries = [e for e in history if "Normalized" in e]
        assert len(normalize_entries) == 1
        assert "2.5" in normalize_entries[0]
        assert "+/-" not in normalize_entries[0]

    def test_processing_history_with_factor_error(
        self, curve_with_error: ScatteringCurve1D
    ) -> None:
        result = normalize(curve_with_error, 2.5, factor_error=0.1)
        history = result.metadata["processing_history"]
        normalize_entries = [e for e in history if "Normalized" in e]
        assert len(normalize_entries) == 1
        assert "+/-" in normalize_entries[0]

    def test_raises_on_zero_factor(self, curve_with_error: ScatteringCurve1D) -> None:
        with pytest.raises(ProcessingError, match="positive"):
            normalize(curve_with_error, 0.0)

    def test_raises_on_negative_factor(
        self, curve_with_error: ScatteringCurve1D
    ) -> None:
        with pytest.raises(ProcessingError, match="positive"):
            normalize(curve_with_error, -1.0)

    def test_raises_on_wrong_curve_type(self) -> None:
        with pytest.raises(ProcessingError, match="ScatteringCurve1D"):
            normalize("not a curve", 1.0)  # type: ignore[arg-type]

    def test_metadata_preserved(self, curve_with_error: ScatteringCurve1D) -> None:
        curve_with_error.metadata["filename"] = "test.dat"
        result = normalize(curve_with_error, 2.0)
        assert result.metadata["filename"] == "test.dat"

    def test_units_preserved(self, curve_with_error: ScatteringCurve1D) -> None:
        result = normalize(curve_with_error, 2.0)
        assert result.q_unit == curve_with_error.q_unit
        assert result.intensity_unit == curve_with_error.intensity_unit
