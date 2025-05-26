# scatterbrain/analysis/porod.py
"""
Porod analysis functions for SAXS data.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from scipy.stats import linregress
import warnings

from ..core import ScatteringCurve1D


def porod_analysis(
    curve: ScatteringCurve1D,
    q_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    q_fraction_high: float = 0.25,
    min_points: int = 5,
    expected_exponent: Optional[float] = 4.0,
    fit_log_log: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Performs Porod analysis on a 1D scattering curve.

    This function can estimate the Porod exponent and the Porod constant.
    The Porod law states that for large q, I(q) ~ Kp * q^(-n) + Bkg,
    where Kp is the Porod constant, n is the Porod exponent (typically 4
    for smooth 3D interfaces), and Bkg is a constant background.

    If `fit_log_log` is True (default):
        log10(I(q)) = log10(Kp) - n * log10(q)  (assuming Bkg is negligible or subtracted)
        A linear fit of log10(I(q)) vs log10(q) yields:
            slope = -n
            intercept = log10(Kp) => Kp = 10^(intercept)

    Alternatively, one can analyze the Porod plot I(q)*q^n vs q^(n-m) or similar.
    For simplicity, this initial version focuses on the log-log fit or
    calculating an average Kp = I(q)q^n in the Porod region.

    Parameters
    ----------
    curve : ScatteringCurve1D
        The input scattering curve data. Assumes background has been subtracted
        if an accurate Porod constant is desired.
    q_range : Optional[Tuple[Optional[float], Optional[float]]], optional
        Manual q-range (min_q, max_q) for the Porod fit/analysis.
        If None (default), the highest `q_fraction_high` of q-points is used.
    q_fraction_high : float, optional
        Fraction of the highest q-data to use if `q_range` is None. Default is 0.25.
    min_points : int, optional
        Minimum number of data points required in the selected q-range for
        analysis. Default is 5.
    expected_exponent : Optional[float], optional
        The theoretical Porod exponent (e.g., 4.0 for smooth 3D interfaces).
        If `fit_log_log` is False, this exponent is used to calculate an
        average Porod constant Kp = mean(I(q) * q^expected_exponent).
        If `fit_log_log` is True, this is mainly for context/comparison.
        Default is 4.0.
    fit_log_log : bool, optional
        If True (default), performs a linear fit on log10(I(q)) vs log10(q)
        to determine the Porod exponent `n` and Porod constant `Kp`.
        If False, calculates an average Porod constant `Kp` using the
        `expected_exponent`.

    Returns
    -------
    Optional[Dict[str, Any]]
        A dictionary containing the analysis results:
        - 'porod_exponent' (float, only if `fit_log_log` is True): The fitted exponent `n`.
        - 'porod_exponent_err' (float, only if `fit_log_log` is True): Error on `n`.
        - 'porod_constant_kp' (float): The Porod constant Kp.
        - 'porod_constant_kp_err' (float, only if `fit_log_log` is True): Error on Kp.
        - 'log_kp_intercept' (float, only if `fit_log_log` is True): Intercept of log-log fit.
        - 'log_kp_intercept_err' (float, only if `fit_log_log` is True): Error on intercept.
        - 'r_value' (float, only if `fit_log_log` is True): Pearson correlation coefficient.
        - 'q_fit_min': Minimum q value used in the analysis.
        - 'q_fit_max': Maximum q value used in the analysis.
        - 'num_points_fit': Number of points used.
        - 'method': Description of the method used.
        Returns None if analysis cannot be performed.

    Notes
    -----
    - Accurate determination of the Porod constant requires careful background
      subtraction prior to calling this function. This function does not
      handle background fitting itself.
    - The specific surface area Sv can be calculated from Kp if the contrast
      (Δρ)^2 is known: Sv = (Kp * π) / Q_invariant, where Q_invariant for
      two-phase systems is often related to volume fractions and contrast.
      More simply, Sv = Kp / (2π * (Δρ)^2) in some conventions, but this
      requires careful unit handling and system knowledge. This function
      only returns Kp.
    """
    if not isinstance(curve, ScatteringCurve1D):
        raise TypeError("Input 'curve' must be a ScatteringCurve1D object.")

    # --- Data preparation ---
    valid_mask = (curve.intensity > 0) & (curve.q > 0) # Need q > 0 for log(q)
    if not np.any(valid_mask):
        warnings.warn("Porod analysis: No positive intensity and q values found.", UserWarning)
        return None

    q_data = curve.q[valid_mask]
    i_data = curve.intensity[valid_mask]

    if len(q_data) < min_points:
        warnings.warn(
            f"Porod analysis: Insufficient data points ({len(q_data)} after filtering) "
            f"for analysis (min_points={min_points}).", UserWarning
        )
        return None

    # --- q-range selection ---
    if q_range is not None:
        q_fit_min_val, q_fit_max_val = q_range
        q_fit_min_val = q_fit_min_val if q_fit_min_val is not None else q_data.min()
        q_fit_max_val = q_fit_max_val if q_fit_max_val is not None else q_data.max()
        method_str_range = f"Manual q_range: ({q_fit_min_val:.3g}, {q_fit_max_val:.3g})"
    else:
        num_points_to_take = max(min_points, int(len(q_data) * q_fraction_high))
        if num_points_to_take < min_points and len(q_data) >= min_points:
             num_points_to_take = min_points # Ensure at least min_points if available overall
        elif num_points_to_take < min_points: # Not enough points even overall
            warnings.warn(f"Porod analysis: Not enough points for auto q-range based on q_fraction_high.", UserWarning)
            return None

        q_fit_min_val = q_data[-num_points_to_take] # Start from high-q end
        q_fit_max_val = q_data.max()
        method_str_range = f"Automatic q_range (highest {q_fraction_high*100:.0f}% of q, min {num_points_to_take} pts)"


    fit_indices = np.where((q_data >= q_fit_min_val) & (q_data <= q_fit_max_val))[0]

    if len(fit_indices) < min_points:
        warnings.warn(
            f"Porod analysis: Selected q-range resulted in {len(fit_indices)} points, "
            f"which is less than min_points={min_points}. Range: {method_str_range}",
            UserWarning
        )
        return None

    q_fit = q_data[fit_indices]
    i_fit = i_data[fit_indices]

    results: Dict[str, Any] = {
        "q_fit_min": q_fit.min(),
        "q_fit_max": q_fit.max(),
        "num_points_fit": len(q_fit),
    }

    if fit_log_log:
        log_q_fit = np.log10(q_fit)
        log_i_fit = np.log10(i_fit)

        try:
            # Use result object for newer scipy for more detailed error info
            regression_result = linregress(log_q_fit, log_i_fit)
            slope = regression_result.slope
            intercept = regression_result.intercept
            r_value = regression_result.rvalue
            stderr_slope = regression_result.stderr
            stderr_intercept = regression_result.intercept_stderr
        except ValueError: # pragma: no cover
            warnings.warn("Porod analysis (log-log fit): ValueError during linear regression.", UserWarning)
            return None

        porod_exponent_n = -slope
        # Kp = 10^intercept
        porod_constant_kp = 10**intercept

        # Error propagation
        # n_err = |dn/dslope * stderr_slope| = |-1 * stderr_slope| = stderr_slope
        porod_exponent_n_err = stderr_slope if stderr_slope is not None else np.nan

        # Kp_err = |dKp/dintercept * stderr_intercept|
        # dKp/dintercept = d(10^intercept)/dintercept = 10^intercept * ln(10) = Kp * ln(10)
        # Kp_err = Kp * ln(10) * stderr_intercept
        porod_constant_kp_err = (
            porod_constant_kp * np.log(10) * stderr_intercept
            if stderr_intercept is not None else np.nan
        )

        results.update({
            "porod_exponent": porod_exponent_n,
            "porod_exponent_err": porod_exponent_n_err,
            "porod_constant_kp": porod_constant_kp,
            "porod_constant_kp_err": porod_constant_kp_err,
            "log_kp_intercept": intercept,
            "log_kp_intercept_err": stderr_intercept,
            "r_value": r_value,
            "method": f"Log-log fit. {method_str_range}"
        })

    else: # Calculate average Kp using expected_exponent
        if expected_exponent is None:
            warnings.warn("Porod analysis: 'expected_exponent' must be provided if 'fit_log_log' is False.", UserWarning)
            return None

        kp_values = i_fit * (q_fit ** expected_exponent)
        porod_constant_kp = np.mean(kp_values)
        # Error on mean Kp could be std(kp_values) / sqrt(N) or propagated if errors on I are known
        porod_constant_kp_err = np.std(kp_values) / np.sqrt(len(kp_values)) if len(kp_values) > 1 else np.nan


        results.update({
            "porod_exponent": expected_exponent, # This is the assumed exponent
            "porod_exponent_err": np.nan, # Not fitted
            "porod_constant_kp": porod_constant_kp,
            "porod_constant_kp_err": porod_constant_kp_err,
            "method": f"Average Kp = I*q^{expected_exponent}. {method_str_range}"
        })

    return results