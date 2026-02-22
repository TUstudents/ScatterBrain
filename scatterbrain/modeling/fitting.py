# scatterbrain/modeling/fitting.py
"""
Model fitting utilities for the ScatterBrain library.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core import ScatteringCurve1D
from ..utils import FittingError

logger = logging.getLogger(__name__)

try:
    import lmfit

    _LMFIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LMFIT_AVAILABLE = False
    logger.warning(
        "lmfit is not installed. fit_model will fall back to scipy.optimize.curve_fit. "
        "Install lmfit>=1.2 for confidence intervals and improved error reporting."
    )


def fit_model(
    curve: ScatteringCurve1D,
    model_func: Callable[..., np.ndarray],
    param_names: List[str],
    initial_params: Sequence[float],
    param_bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    q_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    fixed_params: Optional[Dict[str, float]] = None,
    **curve_fit_kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """
    Fit a model function to a ScatteringCurve1D object.

    The fitted model is::

        I_fit(q) = scale * model_func(q, *model_params) + background


    ``scale`` and ``background`` are always the first two parameters; they
    are prepended to *param_names* internally.

    Parameters
    ----------
    curve : ScatteringCurve1D
        Experimental scattering data to fit.
    model_func : Callable
        Theoretical model.  Signature: ``model_func(q, param1, param2, ...)``.
    param_names : list of str
        Names of the model-specific parameters (*not* including scale/background).
    initial_params : sequence of float
        Initial guesses for all *fitted* parameters in the order
        ``[scale, background, *model_params]``, excluding fixed parameters.
    param_bounds : tuple of (lower, upper), optional
        Bounds for all *fitted* parameters in the same order as
        *initial_params*.  Each bound is a sequence of floats.
    q_range : tuple of (float or None, float or None), optional
        ``(q_min, q_max)`` range for the fit.  None uses the curve edge.
    fixed_params : dict, optional
        Parameters to hold fixed.  Keys: 'scale', 'background', or any name
        in *param_names*.  Values: fixed parameter values.
    **curve_fit_kwargs
        Extra keyword arguments (passed through for compatibility; used only
        by the scipy fallback path).

    Returns
    -------
    dict or None
        On success a dict with keys:

        ``fitted_params``
            dict of fitted parameter values (all params, including fixed).
        ``fitted_params_stderr``
            dict of standard errors.
        ``covariance_matrix``
            numpy array (fitted params only).
        ``fit_curve``
            ScatteringCurve1D with model evaluated over the full q range.
        ``chi_squared_reduced``
            float; nan if errors not available.
        ``success``
            bool.
        ``message``
            str.
        ``q_fit_min``, ``q_fit_max``, ``num_points_fit``
            Fit range information.
        ``confidence_intervals``
            dict mapping param name -> (lower, upper) 1-sigma CI, or None.
        ``lmfit_result``
            raw lmfit.MinimizerResult, or None on scipy fallback.

        Returns None (with a WARNING log) if the fit fails due to a
        runtime condition (not enough points, optimiser divergence).

    Raises
    ------
    FittingError
        For programming errors: wrong input type, mismatched parameter counts.
    """
    if not isinstance(curve, ScatteringCurve1D):
        raise FittingError("Input 'curve' must be a ScatteringCurve1D object.")

    _fixed = fixed_params if fixed_params is not None else {}

    # --- Prepare data in the q range ---
    q_all = curve.q
    i_all = curve.intensity
    err_all = curve.error

    q_min_fit = (
        q_all.min() if q_range is None or q_range[0] is None else float(q_range[0])
    )
    q_max_fit = (
        q_all.max() if q_range is None or q_range[1] is None else float(q_range[1])
    )

    mask = (q_all >= q_min_fit) & (q_all <= q_max_fit)
    q_fit = q_all[mask]
    i_fit = i_all[mask]

    if len(q_fit) < len(initial_params):
        logger.warning(
            "FitModel: Not enough data points (%d) in q range to fit %d parameters.",
            len(q_fit),
            len(initial_params),
        )
        return None

    sigma_fit: Optional[np.ndarray] = None
    if err_all is not None:
        sigma_fit = err_all[mask]
        if np.any(sigma_fit <= 0):
            logger.warning(
                "FitModel: Some sigma values are non-positive; replacing with small value."
            )
            if np.all(sigma_fit <= 0):
                sigma_fit = None
            else:
                sigma_fit = sigma_fit.copy()
                sigma_fit[sigma_fit <= 0] = 1e-9

    # --- Parameter bookkeeping ---
    all_param_names = ["scale", "background"] + list(param_names)

    num_expected_fitted = sum(1 for p in all_param_names if p not in _fixed)

    if len(initial_params) != num_expected_fitted:
        raise FittingError(
            f"Length of initial_params ({len(initial_params)}) does not match "
            f"the number of parameters to be fitted ({num_expected_fitted})."
        )

    if param_bounds is not None:
        if (
            len(param_bounds[0]) != num_expected_fitted
            or len(param_bounds[1]) != num_expected_fitted
        ):
            raise FittingError(
                f"Length of param_bounds components must match the number of "
                f"parameters to be fitted ({num_expected_fitted})."
            )
        lower_b = list(param_bounds[0])
        upper_b = list(param_bounds[1])
    else:
        lower_b = [-np.inf] * num_expected_fitted
        upper_b = [np.inf] * num_expected_fitted

    # Map fitted-param index to global param name (preserving order)
    fitted_param_names: List[str] = [p for p in all_param_names if p not in _fixed]

    # --- Model wrapper ---
    def _eval_model(q_arr: np.ndarray, param_vals: Dict[str, float]) -> np.ndarray:
        model_args = [
            param_vals[p] if p not in _fixed else _fixed[p] for p in param_names
        ]
        return (
            param_vals["scale"] * model_func(q_arr, *model_args)
            + param_vals["background"]
        )

    def _build_param_vals(fitted_vals: Sequence[float]) -> Dict[str, float]:
        vals = dict(_fixed)
        for name, val in zip(fitted_param_names, fitted_vals):
            vals[name] = float(val)
        return vals

    # ------------------------------------------------------------------ lmfit
    if _LMFIT_AVAILABLE:
        params = lmfit.Parameters()
        p0_idx = 0
        for pname in all_param_names:
            if pname in _fixed:
                params.add(pname, value=_fixed[pname], vary=False)
            else:
                lo = lower_b[p0_idx]
                hi = upper_b[p0_idx]
                params.add(
                    pname,
                    value=float(initial_params[p0_idx]),
                    min=lo if np.isfinite(lo) else -np.inf,
                    max=hi if np.isfinite(hi) else np.inf,
                    vary=True,
                )
                p0_idx += 1

        def _residual(lm_params: "lmfit.Parameters") -> np.ndarray:
            pv = {n: lm_params[n].value for n in all_param_names}
            model_vals = _eval_model(q_fit, pv)
            diff = i_fit - model_vals
            if sigma_fit is not None:
                return diff / sigma_fit  # type: ignore[no-any-return]
            return diff  # type: ignore[no-any-return]

        try:
            lm_result = lmfit.minimize(_residual, params, method="leastsq")
        except Exception as exc:
            logger.warning("FitModel: lmfit.minimize failed: %s", exc)
            return None

        # Extract popt / pcov from lmfit result
        fitted_vals_lm: Dict[str, float] = {
            n: lm_result.params[n].value for n in all_param_names
        }
        fitted_stderr_lm: Dict[str, float] = {}
        for n in all_param_names:
            if n in _fixed:
                fitted_stderr_lm[n] = 0.0
            else:
                se = lm_result.params[n].stderr
                fitted_stderr_lm[n] = float(se) if se is not None else np.nan

        # Build covariance matrix for the fitted (non-fixed) params
        if lm_result.covar is not None:
            pcov = np.array(lm_result.covar)
        else:
            pcov = np.full((len(fitted_param_names), len(fitted_param_names)), np.nan)

        # Confidence intervals (1-sigma = 68.27%)
        conf_intervals: Optional[Dict[str, Tuple[float, float]]] = None
        try:
            ci = lmfit.conf_interval(lm_result, lm_result, sigmas=[1])
            conf_intervals = {}
            for pname, ci_list in ci.items():
                # ci_list: [(prob, val), ...] in ascending probability order
                # Find lower (prob~0.1573) and upper (prob~0.8427)
                lower_val = upper_val = np.nan
                for prob, val in ci_list:
                    if abs(prob - 0.1573) < 0.05:
                        lower_val = val
                    if abs(prob - 0.8427) < 0.05:
                        upper_val = val
                conf_intervals[pname] = (lower_val, upper_val)
        except Exception:
            conf_intervals = None

        fit_success = bool(
            lm_result.success
            and lm_result.covar is not None
            and np.all(np.isfinite(np.diag(pcov)))
        )
        fit_message = lm_result.message or "lmfit finished."

        i_model_full = _eval_model(q_all, fitted_vals_lm)

        chi2_red = np.nan
        if sigma_fit is not None and len(sigma_fit) > 0:
            res = i_fit - _eval_model(q_fit, fitted_vals_lm)
            if sigma_fit is not None:
                res = res / sigma_fit
            dof = len(q_fit) - len(fitted_param_names)
            if dof > 0:
                chi2_red = float(np.sum(res**2) / dof)

        fit_curve = ScatteringCurve1D(
            q=q_all,
            intensity=i_model_full,
            metadata={"source": "model_fit", "original_curve_id": id(curve)},
        )

        return {
            "fitted_params": fitted_vals_lm,
            "fitted_params_stderr": fitted_stderr_lm,
            "covariance_matrix": pcov,
            "fit_curve": fit_curve,
            "chi_squared_reduced": chi2_red,
            "success": fit_success,
            "message": fit_message,
            "q_fit_min": float(q_fit.min()),
            "q_fit_max": float(q_fit.max()),
            "num_points_fit": int(len(q_fit)),
            "confidence_intervals": conf_intervals,
            "lmfit_result": lm_result,
        }

    # -------------------------------------------------------- scipy fallback
    from scipy.optimize import OptimizeWarning, curve_fit  # noqa: PLC0415
    import warnings  # noqa: PLC0415

    def fit_wrapper_func(q_wrapper: np.ndarray, *fitted_args: float) -> np.ndarray:
        pv = _build_param_vals(fitted_args)
        for n in _fixed:
            pv[n] = _fixed[n]
        return _eval_model(q_wrapper, pv)

    p0_fit = list(initial_params)
    final_bounds = (lower_b, upper_b)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                fit_wrapper_func,
                q_fit,
                i_fit,
                p0=p0_fit,
                sigma=sigma_fit,
                absolute_sigma=(sigma_fit is not None),
                bounds=final_bounds,
                **curve_fit_kwargs,
            )
        fit_success = True
        fit_message = "Fit successful (scipy fallback)."
    except (RuntimeError, ValueError) as exc:
        logger.warning("FitModel (scipy fallback): curve_fit failed: %s", exc)
        return None

    fitted_vals_sp: Dict[str, float] = {}
    fitted_stderr_sp: Dict[str, float] = {}
    popt_idx = 0
    for pname in all_param_names:
        if pname in _fixed:
            fitted_vals_sp[pname] = _fixed[pname]
            fitted_stderr_sp[pname] = 0.0
        else:
            fitted_vals_sp[pname] = float(popt[popt_idx])
            diag_val = pcov[popt_idx, popt_idx] if pcov is not None else np.nan
            fitted_stderr_sp[pname] = (
                float(np.sqrt(diag_val))
                if np.isfinite(diag_val) and diag_val >= 0
                else np.nan
            )
            popt_idx += 1

    i_model_full = _eval_model(q_all, fitted_vals_sp)
    fit_curve = ScatteringCurve1D(
        q=q_all,
        intensity=i_model_full,
        metadata={"source": "model_fit", "original_curve_id": id(curve)},
    )

    chi2_red = np.nan
    if sigma_fit is not None and len(sigma_fit) > 0:
        res = (i_fit - _eval_model(q_fit, fitted_vals_sp)) / sigma_fit
        dof = len(q_fit) - len(popt)
        if dof > 0:
            chi2_red = float(np.sum(res**2) / dof)

    pcov_valid = pcov is not None and not np.any(np.isinf(np.diag(pcov)))
    return {
        "fitted_params": fitted_vals_sp,
        "fitted_params_stderr": fitted_stderr_sp,
        "covariance_matrix": pcov,
        "fit_curve": fit_curve,
        "chi_squared_reduced": chi2_red,
        "success": fit_success and pcov_valid,
        "message": fit_message,
        "q_fit_min": float(q_fit.min()),
        "q_fit_max": float(q_fit.max()),
        "num_points_fit": int(len(q_fit)),
        "confidence_intervals": None,
        "lmfit_result": None,
    }
