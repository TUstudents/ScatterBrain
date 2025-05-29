# scatterbrain/analysis/guinier.py
"""
Guinier analysis functions for SAXS data.
"""

from typing import Optional, Tuple, Dict, Any # ... (rest of the imports from previous guinier_fit)
import numpy as np
from scipy.stats import linregress
import warnings

from ..core import ScatteringCurve1D # Note the relative import for core

#
# <<< PASTE THE ENTIRE guinier_fit FUNCTION CODE FROM THE PREVIOUS RESPONSE HERE >>>
# (The one starting with "def guinier_fit(...)")
#
def guinier_fit(
    curve: ScatteringCurve1D,
    q_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    qrg_limit_max: Optional[float] = 1.3,
    qrg_limit_min: Optional[float] = None,
    auto_q_selection_fraction: float = 0.1,
    min_points: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    Performs Guinier analysis on a 1D scattering curve to determine the
    Radius of Gyration (Rg) and forward scattering intensity (I0).

    The Guinier approximation is: I(q) = I(0) * exp(-(q*Rg)^2 / 3)
    This can be linearized to: ln(I(q)) = ln(I(0)) - (Rg^2 / 3) * q^2
    A linear fit of ln(I(q)) vs q^2 yields:
        slope = -Rg^2 / 3  => Rg = sqrt(-3 * slope)
        intercept = ln(I(0)) => I(0) = exp(intercept)

    Parameters
    ----------
    curve : ScatteringCurve1D
        The input scattering curve data.
    q_range : Optional[Tuple[Optional[float], Optional[float]]], optional
        Manual q-range (min_q, max_q) for the fit. If None (default),
        an attempt is made to automatically select a suitable range based on
        `qrg_limit_max`, `qrg_limit_min`, and `auto_q_selection_fraction`.
        Use `None` for min or max if one boundary is to be auto-selected, e.g. `(0.1, None)`.
    qrg_limit_max : Optional[float], optional
        Maximum q*Rg value for the Guinier region if `q_range` is auto-selected.
        Typically around 1.0-1.5 for globular particles. Default is 1.3.
        Set to None to disable upper q*Rg limit in auto-selection.
    qrg_limit_min : Optional[float], optional
        Minimum q*Rg value for the Guinier region if `q_range` is auto-selected.
        Useful for excluding very low q data affected by aggregation or beamstop.
        Default is None (no lower q*Rg limit).
    auto_q_selection_fraction : float, optional
        Fraction of the lowest q-data to initially consider for an Rg estimate
        when `q_range` is None. Default is 0.1 (lowest 10% of q-points).
        This initial Rg is then used with `qrg_limit_max` and `qrg_limit_min`
        to refine the q-range for the final fit.
    min_points : int, optional
        Minimum number of data points required in the selected q-range for
        a fit to be attempted. Default is 5.

    Returns
    -------
    Optional[Dict[str, Any]]
        A dictionary containing the fit results:
        - 'Rg': Radius of Gyration.
        - 'Rg_err': Estimated error on Rg (from fit uncertainty).
        - 'I0': Forward scattering intensity I(0).
        - 'I0_err': Estimated error on I(0) (from fit uncertainty).
        - 'slope': Slope of the linear fit.
        - 'intercept': Intercept of the linear fit.
        - 'r_value': Pearson correlation coefficient of the fit.
        - 'p_value': Two-sided p-value for a hypothesis test whose null
                     hypothesis is that the slope is zero.
        - 'stderr_slope': Standard error of the estimated slope.
        - 'stderr_intercept': Standard error of the estimated intercept.
        - 'q_fit_min': Minimum q value used in the fit.
        - 'q_fit_max': Maximum q value used in the fit.
        - 'num_points_fit': Number of points used in the fit.
        - 'valid_guinier_range_criteria': String describing how the range was determined.
        Returns None if a fit cannot be performed (e.g., insufficient data points,
        positive slope, q_range results in no data).

    Notes
    -----
    - Error propagation for Rg_err and I0_err is based on the standard errors
      of the slope and intercept from the linear regression.
    - Automatic q-range selection is iterative and heuristic. It may not always
      find the optimal range, especially for complex or noisy data.
      Manual specification of `q_range` is recommended for critical analysis.
    - The function assumes I(q) > 0 for ln(I(q)). Points where I(q) <= 0
      are excluded from the fit. If curve.error is available, it is used
      to weight ln(I(q)) values (currently not implemented, but planned).
    """
    if not isinstance(curve, ScatteringCurve1D):
        raise TypeError("Input 'curve' must be a ScatteringCurve1D object.")

    # --- Data preparation ---
    valid_i_mask = curve.intensity > 0
    if not np.any(valid_i_mask):
        warnings.warn("Guinier fit: No positive intensity values found in the curve.", UserWarning)
        return None

    q_data = curve.q[valid_i_mask]
    i_data = curve.intensity[valid_i_mask]

    if len(q_data) < min_points:
        warnings.warn(
            f"Guinier fit: Insufficient data points ({len(q_data)} after filtering I>0) "
            f"for analysis (min_points={min_points}).", UserWarning
        )
        return None

    ln_i_data = np.log(i_data)
    q_squared_data = q_data**2

    # --- q-range selection logic ---
    q_fit_min_val: float = 0.0
    q_fit_max_val: float = np.inf
    fit_indices: np.ndarray
    criteria_str: str = ""

    if q_range is not None: # Manual q_range provided
        manual_q_min, manual_q_max = q_range
        q_fit_min_val = manual_q_min if manual_q_min is not None else q_data.min()
        q_fit_max_val = manual_q_max if manual_q_max is not None else q_data.max()
        criteria_str = f"Manual q_range: ({q_fit_min_val:.3g}, {q_fit_max_val:.3g})"
        fit_indices = np.where((q_data >= q_fit_min_val) & (q_data <= q_fit_max_val))[0]

    else: # Automatic q-range selection
        criteria_str = "Automatic q_range selection: "
        num_initial_points = max(min_points, int(len(q_data) * auto_q_selection_fraction))
        if num_initial_points < min_points:
             warnings.warn(
                f"Guinier fit (auto-range): Not enough points ({num_initial_points}) "
                f"for initial Rg estimate. Using first {min_points} points if available.", UserWarning
            )
             num_initial_points = min(min_points, len(q_data))
             if num_initial_points < min_points:
                 return None

        q_sq_initial = q_squared_data[:num_initial_points]
        ln_i_initial = ln_i_data[:num_initial_points]

        if len(q_sq_initial) < 2:
            warnings.warn("Guinier fit (auto-range): Less than 2 points for initial Rg estimate.", UserWarning)
            return None

        try:
            regression_initial = linregress(q_sq_initial, ln_i_initial)
            slope_initial = regression_initial.slope
            print(f"Initial slope: {slope_initial:.3g} (expected negative for Guinier fit).")
        except ValueError:
            warnings.warn("Guinier fit (auto-range): ValueError during initial linear regression.", UserWarning)
            return None

        # Issue warning for positive slope before any fallback logic
        if slope_initial >= 0:
            warnings.warn("Guinier fit (auto-range): Initial fit yielded non-negative slope . ", UserWarning)
            # Fallback logic after warning
            q_fit_min_val = q_data.min()
            q_fit_max_val = q_data[max(min_points-1, int(len(q_data) * 0.15))]
            criteria_str = "Fallback q-range due to initial positive slope. "
            fit_indices = np.where((q_data >= q_fit_min_val) & (q_data <= q_fit_max_val))[0]
        else:
            rg_initial_est = np.sqrt(-3 * slope_initial)
            criteria_str += f"Initial Rg_est={rg_initial_est:.3g}. "

            q_fit_min_val = q_data.min()
            if qrg_limit_min is not None and rg_initial_est > 0:
                q_fit_min_val = max(q_fit_min_val, qrg_limit_min / rg_initial_est)
                criteria_str += f"q_min_limit based on qRg_min={qrg_limit_min} -> q_min={q_fit_min_val:.3g}. "

            q_fit_max_val = q_data.max()
            if qrg_limit_max is not None and rg_initial_est > 0:
                q_fit_max_val = min(q_fit_max_val, qrg_limit_max / rg_initial_est)
                criteria_str += f"q_max_limit based on qRg_max={qrg_limit_max} -> q_max={q_fit_max_val:.3g}."
            else:
                q_fit_max_val = min(q_fit_max_val, q_data[max(min_points-1, int(len(q_data) * 0.3))])
                criteria_str += f"Fallback q_max={q_fit_max_val:.3g} (no qRg_max or invalid Rg_est). "
            
            if q_fit_min_val >= q_fit_max_val: # Handle case where limits make range invalid
                 warnings.warn(
                    f"Guinier fit (auto-range): q_min ({q_fit_min_val:.3g}) >= q_max ({q_fit_max_val:.3g}) "
                    "after applying qRg limits. Expanding q_max.", UserWarning)
                 q_fit_max_val = q_data[min(len(q_data)-1, np.searchsorted(q_data, q_fit_min_val) + min_points)]


            fit_indices = np.where((q_data >= q_fit_min_val) & (q_data <= q_fit_max_val))[0]

    if len(fit_indices) < min_points:
        warnings.warn(
            f"Guinier fit: Selected q-range resulted in {len(fit_indices)} points, "
            f"which is less than min_points={min_points}. Criteria: {criteria_str}",
            UserWarning
        )
        return None

    q_sq_fit = q_squared_data[fit_indices]
    ln_i_fit = ln_i_data[fit_indices]
    q_fit_actual_min = q_data[fit_indices].min()
    q_fit_actual_max = q_data[fit_indices].max()

    try:
        regression_result = linregress(q_sq_fit, ln_i_fit)
        slope = regression_result.slope
        intercept = regression_result.intercept
        r_value = regression_result.rvalue
        p_value = regression_result.pvalue
        stderr_slope = regression_result.stderr
        stderr_intercept = regression_result.intercept_stderr
    except ValueError: # pragma: no cover
        warnings.warn("Guinier fit: ValueError during final linear regression. Check selected data.", UserWarning)
        return None

    if slope >= 0:
        warnings.warn(
            "Guinier fit: Resulted in a non-negative slope (Rg^2 would be negative). "
            "The Guinier approximation may not be valid for this q-range or data.",
            UserWarning
        )
        return {
            "Rg": np.nan, "Rg_err": np.nan, "I0": np.nan, "I0_err": np.nan,
            "slope": slope, "intercept": intercept, "r_value": r_value,
            "p_value": p_value, "stderr_slope": stderr_slope, "stderr_intercept": stderr_intercept,
            "q_fit_min": q_fit_actual_min, "q_fit_max": q_fit_actual_max,
            "num_points_fit": len(q_sq_fit),
            "valid_guinier_range_criteria": criteria_str + " (Warning: Positive Slope)"
        }

    rg = np.sqrt(-3 * slope)
    i0 = np.exp(intercept)

    rg_err = (1.5 / rg) * stderr_slope if rg > 0 and stderr_slope is not None else np.nan
    i0_err = i0 * stderr_intercept if stderr_intercept is not None else np.nan

    return {
        "Rg": rg,
        "Rg_err": rg_err,
        "I0": i0,
        "I0_err": i0_err,
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "stderr_slope": stderr_slope,
        "stderr_intercept": stderr_intercept,
        "q_fit_min": q_fit_actual_min,
        "q_fit_max": q_fit_actual_max,
        "num_points_fit": len(q_sq_fit),
        "valid_guinier_range_criteria": criteria_str.strip()
    }