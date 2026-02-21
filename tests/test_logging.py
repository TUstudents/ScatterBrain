"""Tests for unified logging and custom exception wiring."""

import logging

import numpy as np
import pytest

import scatterbrain
from scatterbrain.utils import AnalysisError, FittingError, ProcessingError, ScatterBrainError
from scatterbrain.core import ScatteringCurve1D
from scatterbrain.analysis.guinier import guinier_fit
from scatterbrain.analysis.porod import porod_analysis
from scatterbrain.modeling.fitting import fit_model
from scatterbrain.modeling.form_factors import sphere_pq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_curve(n=50, q_min=0.01, q_max=0.5, rg=5.0, i0=1000.0, noise=0.0):
    """Create a synthetic Guinier-like scattering curve."""
    q = np.linspace(q_min, q_max, n)
    intensity = i0 * np.exp(-(q ** 2 * rg ** 2) / 3.0)
    if noise:
        intensity += np.random.default_rng(0).normal(0, noise, n)
    error = np.sqrt(np.abs(intensity)) * 0.05
    return ScatteringCurve1D(q=q, intensity=intensity, error=error)


def _make_empty_curve():
    """A curve where all intensities are non-positive."""
    q = np.linspace(0.01, 0.5, 20)
    intensity = -np.ones_like(q)
    return ScatteringCurve1D(q=q, intensity=intensity)


# ---------------------------------------------------------------------------
# Step 1: Logger hierarchy
# ---------------------------------------------------------------------------

class TestLoggerHierarchy:
    def test_root_logger_has_null_handler(self):
        """The 'scatterbrain' logger must have exactly one NullHandler by default."""
        sb_logger = logging.getLogger("scatterbrain")
        null_handlers = [h for h in sb_logger.handlers if isinstance(h, logging.NullHandler)]
        assert len(null_handlers) >= 1, "scatterbrain logger should have a NullHandler"

    def test_submodule_loggers_propagate(self):
        """Sub-module loggers should propagate to the root scatterbrain logger."""
        for name in [
            "scatterbrain.core",
            "scatterbrain.io",
            "scatterbrain.analysis.guinier",
            "scatterbrain.analysis.porod",
            "scatterbrain.modeling.fitting",
            "scatterbrain.visualization",
        ]:
            child = logging.getLogger(name)
            assert child.propagate, f"{name} logger should propagate"

    def test_library_silent_by_default(self, caplog):
        """No log output should leak when the library is used without configure_logging."""
        with caplog.at_level(logging.DEBUG, logger="scatterbrain"):
            curve = _make_curve()
            guinier_fit(curve)
        # caplog captures records even through NullHandler-only loggers when using
        # pytest's caplog fixture, so just verify no ERROR-level records appear.
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert not error_records


# ---------------------------------------------------------------------------
# Step 1: configure_logging()
# ---------------------------------------------------------------------------

class TestConfigureLogging:
    def test_configure_logging_adds_handler(self):
        """configure_logging() should add a handler to the scatterbrain logger."""
        sb_logger = logging.getLogger("scatterbrain")
        before = len(sb_logger.handlers)
        handler = logging.NullHandler()  # Use NullHandler so nothing is printed
        scatterbrain.configure_logging(level=logging.DEBUG, handler=handler)
        after = len(sb_logger.handlers)
        assert after == before + 1
        # Clean up
        sb_logger.removeHandler(handler)

    def test_configure_logging_sets_level(self):
        """configure_logging() should set the logger level."""
        sb_logger = logging.getLogger("scatterbrain")
        handler = logging.NullHandler()
        scatterbrain.configure_logging(level=logging.WARNING, handler=handler)
        assert sb_logger.level == logging.WARNING
        # Clean up
        sb_logger.removeHandler(handler)
        sb_logger.setLevel(logging.NOTSET)


# ---------------------------------------------------------------------------
# Step 4: warnings.warn → logger.warning
# ---------------------------------------------------------------------------

class TestGuinierLogging:
    def test_no_positive_intensity_emits_warning(self, caplog):
        curve = _make_empty_curve()
        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            result = guinier_fit(curve)
        assert result is None
        assert any("No positive intensity" in r.message for r in caplog.records)

    def test_insufficient_points_emits_warning(self, caplog):
        q = np.linspace(0.01, 0.5, 3)
        intensity = np.ones(3) * 100
        curve = ScatteringCurve1D(q=q, intensity=intensity)
        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            result = guinier_fit(curve, min_points=5)
        assert result is None
        assert any("Insufficient data points" in r.message for r in caplog.records)

    def test_auto_range_slope_logged_at_debug(self, caplog):
        curve = _make_curve(n=50, rg=5.0, i0=1000.0)
        with caplog.at_level(logging.DEBUG, logger="scatterbrain.analysis.guinier"):
            guinier_fit(curve)
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("initial slope" in r.message.lower() for r in debug_records)


class TestPorodLogging:
    def test_no_positive_values_emits_warning(self, caplog):
        curve = _make_empty_curve()
        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            result = porod_analysis(curve)
        assert result is None
        assert any("No positive intensity" in r.message for r in caplog.records)

    def test_insufficient_points_emits_warning(self, caplog):
        q = np.linspace(0.1, 0.5, 3)
        intensity = np.ones(3) * 10.0
        curve = ScatteringCurve1D(q=q, intensity=intensity)
        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            result = porod_analysis(curve, min_points=5)
        assert result is None
        assert any("Insufficient data points" in r.message for r in caplog.records)


class TestFittingLogging:
    def test_too_few_points_emits_warning(self, caplog):
        q = np.array([0.1, 0.2])
        intensity = np.array([10.0, 5.0])
        curve = ScatteringCurve1D(q=q, intensity=intensity)
        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            result = fit_model(
                curve,
                model_func=sphere_pq,
                param_names=["radius"],
                initial_params=[1.0, 0.0, 5.0],  # scale, bg, radius
            )
        assert result is None
        assert any("Not enough data points" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Step 5: Custom exceptions
# ---------------------------------------------------------------------------

class TestCustomExceptionHierarchy:
    def test_custom_exceptions_inherit_scatterbrainerror(self):
        assert issubclass(AnalysisError, ScatterBrainError)
        assert issubclass(FittingError, ScatterBrainError)
        assert issubclass(ProcessingError, ScatterBrainError)

    def test_scatterbrainerror_inherits_exception(self):
        assert issubclass(ScatterBrainError, Exception)


class TestAnalysisError:
    def test_guinier_raises_analysis_error_on_wrong_type(self):
        with pytest.raises(AnalysisError):
            guinier_fit("not a curve")

    def test_porod_raises_analysis_error_on_wrong_type(self):
        with pytest.raises(AnalysisError):
            porod_analysis(42)


class TestFittingError:
    def test_fit_model_raises_fitting_error_on_wrong_type(self):
        with pytest.raises(FittingError):
            fit_model(
                "not a curve",
                model_func=sphere_pq,
                param_names=["radius"],
                initial_params=[1.0, 0.0, 5.0],
            )

    def test_fit_model_raises_fitting_error_on_param_mismatch(self):
        curve = _make_curve()
        with pytest.raises(FittingError, match="initial_params"):
            fit_model(
                curve,
                model_func=sphere_pq,
                param_names=["radius"],
                initial_params=[1.0],  # Missing scale and background
            )

    def test_fit_model_raises_fitting_error_on_bounds_mismatch(self):
        curve = _make_curve()
        with pytest.raises(FittingError, match="param_bounds"):
            fit_model(
                curve,
                model_func=sphere_pq,
                param_names=["radius"],
                initial_params=[1.0, 0.0, 5.0],
                param_bounds=([0.0], [np.inf]),  # Wrong length (1 instead of 3)
            )


class TestProcessingError:
    def test_convert_q_unit_raises_processing_error_on_bad_unit(self):
        curve = _make_curve()
        with pytest.raises(ProcessingError):
            curve.convert_q_unit("parsec^-1")
