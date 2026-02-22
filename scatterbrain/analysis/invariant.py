# scatterbrain/analysis/invariant.py
"""
Scattering invariant Q* calculation for two-phase systems.

Q* = integral_0^inf q^2 * I(q) dq

is a fundamental quantity that depends only on contrast and volume fractions,
not on particle shape or size distribution.
"""

import logging
import math
from typing import Optional, Tuple, TypedDict

import numpy as np
from scipy.integrate import quad

from ..core import ScatteringCurve1D
from ..utils import AnalysisError
from .guinier import GuinierResult
from .porod import PorodResult

logger = logging.getLogger(__name__)

_MIN_POINTS = 5


class InvariantResult(TypedDict):
    """Return type of :func:`scattering_invariant`."""

    Q_star: float
    Q_star_low_q: float
    Q_star_high_q: float
    Q_star_total: float
    q_min: float
    q_max: float
    num_points: int
    extrapolation_method: str


def scattering_invariant(
    curve: ScatteringCurve1D,
    q_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    guinier_result: Optional[GuinierResult] = None,
    porod_result: Optional[PorodResult] = None,
) -> Optional[InvariantResult]:
    """
    Calculate the scattering invariant Q* for a 1D scattering curve.

    Q* = integral_0^inf q^2 * I(q) dq

    The measured range is integrated numerically with the trapezoidal rule.
    Guinier and Porod extrapolations are added at the low- and high-q ends
    when the corresponding fit results are provided.

    Parameters
    ----------
    curve : ScatteringCurve1D
        The scattering data.
    q_range : tuple of (float or None, float or None), optional
        ``(q_min, q_max)`` bounds for the numerical integration.  Use
        ``None`` for a boundary to take the curve's edge value.
    guinier_result : GuinierResult dict, optional
        Output of :func:`scatterbrain.analysis.guinier.guinier_fit`.
        When provided and contains finite Rg and I0, the integral
        ``int_0^{q_min} q^2 * I0 * exp(-(q*Rg)^2/3) dq`` is added.
    porod_result : PorodResult dict, optional
        Output of :func:`scatterbrain.analysis.porod.porod_analysis`.
        When provided and contains finite Kp (porod_constant_kp) and
        exponent (porod_exponent) > 3, the integral
        ``int_{q_max}^inf q^2 * Kp * q^{-n} dq = Kp * q_max^{3-n} / (n-3)``
        is added.  For n <= 3 this integral diverges; Q_star_high_q is
        set to nan and a WARNING is logged.

    Returns
    -------
    InvariantResult or None
        None with a WARNING log message if fewer than 5 valid data points
        exist in the selected q range.

    Raises
    ------
    AnalysisError
        If *curve* is not a ScatteringCurve1D.
    """
    if not isinstance(curve, ScatteringCurve1D):
        raise AnalysisError("'curve' must be a ScatteringCurve1D object.")

    # --- Select q range ---
    q_min = (
        float(curve.q.min())
        if q_range is None or q_range[0] is None
        else float(q_range[0])
    )
    q_max = (
        float(curve.q.max())
        if q_range is None or q_range[1] is None
        else float(q_range[1])
    )

    mask = (curve.q >= q_min) & (curve.q <= q_max)
    q_sel = curve.q[mask]
    i_sel = curve.intensity[mask]

    if len(q_sel) < _MIN_POINTS:
        logger.warning(
            "scattering_invariant: only %d valid points in q range [%.4g, %.4g]; "
            "need at least %d. Returning None.",
            len(q_sel),
            q_min,
            q_max,
            _MIN_POINTS,
        )
        return None

    # --- Core integral over measured range ---
    q_star = float(np.trapezoid(q_sel**2 * i_sel, q_sel))

    # --- Low-q Guinier extrapolation ---
    q_star_low_q = 0.0
    extrapolation_notes = []

    if guinier_result is not None:
        rg = float(guinier_result.get("Rg", math.nan))
        i0 = float(guinier_result.get("I0", math.nan))
        if math.isfinite(rg) and rg > 0 and math.isfinite(i0) and i0 > 0:

            def _guinier_integrand(q: float) -> float:
                return q**2 * i0 * math.exp(-((q * rg) ** 2) / 3.0)

            q_star_low_q, _ = quad(_guinier_integrand, 0.0, q_sel[0])
            extrapolation_notes.append("Guinier low-q")
        else:
            logger.warning(
                "scattering_invariant: guinier_result has non-finite or non-positive "
                "Rg/I0; skipping low-q extrapolation."
            )

    # --- High-q Porod extrapolation ---
    q_star_high_q = 0.0

    if porod_result is not None:
        kp = porod_result.get("porod_constant_kp")
        n = porod_result.get("porod_exponent")
        if (
            kp is not None
            and n is not None
            and math.isfinite(float(kp))
            and math.isfinite(float(n))
        ):
            kp = float(kp)
            n = float(n)
            q_max_val = float(q_sel[-1])
            if n <= 3.0:
                logger.warning(
                    "scattering_invariant: Porod exponent %.4g <= 3; "
                    "high-q integral diverges. Setting Q_star_high_q = nan.",
                    n,
                )
                q_star_high_q = math.nan
            else:
                # int_{q_max}^inf Kp * q^{2-n} dq = Kp * q_max^{3-n} / (n-3)
                q_star_high_q = kp * q_max_val ** (3.0 - n) / (n - 3.0)
                extrapolation_notes.append("Porod high-q")
        else:
            logger.warning(
                "scattering_invariant: porod_result has non-finite Kp or exponent; "
                "skipping high-q extrapolation."
            )

    q_star_total = q_star + q_star_low_q
    if not math.isnan(q_star_high_q):
        q_star_total += q_star_high_q
    else:
        q_star_total = math.nan

    extrapolation_method = (
        ", ".join(extrapolation_notes) if extrapolation_notes else "none"
    )

    return {
        "Q_star": q_star,
        "Q_star_low_q": q_star_low_q,
        "Q_star_high_q": q_star_high_q,
        "Q_star_total": q_star_total,
        "q_min": float(q_sel[0]),
        "q_max": float(q_sel[-1]),
        "num_points": int(len(q_sel)),
        "extrapolation_method": extrapolation_method,
    }
