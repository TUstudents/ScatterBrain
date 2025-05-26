# scatterbrain/modeling/fitting.py
"""
Model fitting utilities for the ScatterBrain library.
"""

from typing import Callable, List, Dict, Any, Optional, Tuple, Sequence
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

from ..core import ScatteringCurve1D


def fit_model(
    curve: ScatteringCurve1D,
    model_func: Callable[..., np.ndarray],
    param_names: List[str],
    initial_params: Sequence[float],
    param_bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    q_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    fixed_params: Optional[Dict[str, float]] = None,
    **curve_fit_kwargs: Any
) -> Optional[Dict[str, Any]]:
    """
    Fits a given model function to a ScatteringCurve1D object.

    This function uses `scipy.optimize.curve_fit` for least-squares fitting.
    The model function `model_func` is expected to take `q` as its first argument,
    followed by model parameters. This fitter wraps `model_func` to include
    a scale factor and a constant background term, so the actual fitted model is:
        I_fit(q) = scale * model_func(q, *model_params) + background

    Parameters
    ----------
    curve : ScatteringCurve1D
        The experimental scattering data to fit.
    model_func : Callable[..., np.ndarray]
        The theoretical model function (e.g., a form factor P(q)).
        Its signature must be `model_func(q, param1, param2, ...)`.
    param_names : List[str]
        A list of names for the parameters `param1, param2, ...` that
        `model_func` accepts, in the same order. This list should NOT include
        'scale' or 'background', as these are handled internally.
    initial_params : Sequence[float]
        Initial guesses for the model parameters (`param1, param2, ...`),
        corresponding to `param_names`.
        It should also include initial guesses for 'scale' and 'background'
        at the beginning: [initial_scale, initial_background, initial_param1, ...].
    param_bounds : Optional[Tuple[Sequence[float], Sequence[float]]], optional
        Bounds for the parameters `(lower_bounds, upper_bounds)`.
        The bounds should correspond to ALL fitted parameters, including
        'scale' and 'background' at the beginning, then model-specific parameters.
        Format: `([scale_low, bg_low, p1_low, ...], [scale_high, bg_high, p1_high, ...])`.
        Use -np.inf or np.inf for unbounded parameters. Default is unbounded.
    q_range : Optional[Tuple[Optional[float], Optional[float]]], optional
        The q-range (min_q, max_q) over which to perform the fit.
        If None (default), the full q-range of the curve is used.
    fixed_params : Optional[Dict[str, float]], optional
        A dictionary of parameters to keep fixed during the fit.
        Keys can be 'scale', 'background', or any name from `param_names`.
        The values are the fixed parameter values.
        Note: Fixed parameters are *not* included in `initial_params` or `param_bounds`
        for the fitting process itself, but the wrapper handles this.
        The `initial_params` and `param_bounds` provided should be for the
        *parameters being fitted*.
    **curve_fit_kwargs : Any
        Additional keyword arguments to pass directly to `scipy.optimize.curve_fit`
        (e.g., `maxfev`, `ftol`).

    Returns
    -------
    Optional[Dict[str, Any]]
        A dictionary containing the fit results:
        - 'fitted_params' (Dict[str, float]): Fitted values for all parameters
          (scale, background, and model_params).
        - 'fitted_params_stderr' (Dict[str, float]): Standard errors for the
          fitted parameters.
        - 'covariance_matrix' (np.ndarray): Covariance matrix of the fit.
        - 'fit_curve' (ScatteringCurve1D): The model intensity calculated with
          the fitted parameters over the original curve's q-range.
        - 'chi_squared_reduced' (float): Reduced chi-squared value if errors
          are available in the input curve and used (sigma in curve_fit).
        - 'success' (bool): True if `curve_fit` reported success.
        - 'message' (str): Message from `curve_fit`.
        Returns None if the fit fails or an error occurs.

    Raises
    ------
    ValueError
        If `param_names` and `initial_params` (excluding scale/bg) mismatch.
        If fixed_params contains unknown parameter names.
    """
    if not isinstance(curve, ScatteringCurve1D):
        raise TypeError("Input 'curve' must be a ScatteringCurve1D object.")

    _fixed_params = fixed_params if fixed_params is not None else {}

    # --- Prepare q and I data from the curve within the specified q_range ---
    q_data_full = curve.q
    i_data_full = curve.intensity
    err_data_full = curve.error

    q_min_fit = q_data_full.min() if q_range is None or q_range[0] is None else q_range[0]
    q_max_fit = q_data_full.max() if q_range is None or q_range[1] is None else q_range[1]

    fit_mask = (q_data_full >= q_min_fit) & (q_data_full <= q_max_fit)
    q_fit = q_data_full[fit_mask]
    i_fit = i_data_full[fit_mask]

    if len(q_fit) < len(initial_params): # Need at least as many points as parameters
        warnings.warn(
            f"FitModel: Not enough data points ({len(q_fit)}) in the selected q-range "
            f"to fit {len(initial_params)} parameters. Fit aborted.", UserWarning
        )
        return None

    sigma_fit: Optional[np.ndarray] = None
    if err_data_full is not None:
        sigma_fit = err_data_full[fit_mask]
        if np.any(sigma_fit <= 0):
            warnings.warn(
                "FitModel: Some error values (sigma) are non-positive. "
                "These will cause issues with `curve_fit`. Using absolute errors, "
                "or ignoring errors if all are non-positive.", UserWarning
            )
            if np.all(sigma_fit <=0 ):
                sigma_fit = None # Ignore errors
            else:
                sigma_fit[sigma_fit <= 0] = 1e-9 # Replace non-positive with small value


    # --- Parameter handling (fixed vs. fitted) ---
    # Internal parameter order for fitting: scale, background, *model_params
    internal_param_names_all = ['scale', 'background'] + param_names
    
    # Separate initial_params and bounds for fixed vs. fitted
    p0_fit: List[float] = []
    bounds_fit_lower: List[float] = []
    bounds_fit_upper: List[float] = []
    
    current_p0_idx = 0
    current_bounds_idx = 0

    # Check initial_params length against expected number of fitted params
    num_expected_model_params = len(param_names) - sum(1 for pn in param_names if pn in _fixed_params)
    num_expected_fitted = (1 if 'scale' not in _fixed_params else 0) + \
                          (1 if 'background' not in _fixed_params else 0) + \
                          num_expected_model_params
                          
    if len(initial_params) != num_expected_fitted:
        raise ValueError(
            f"Length of initial_params ({len(initial_params)}) does not match "
            f"the number of parameters to be fitted ({num_expected_fitted}). "
            f"Ensure initial_params are provided only for non-fixed parameters."
        )

    if param_bounds is not None:
        if len(param_bounds[0]) != num_expected_fitted or len(param_bounds[1]) != num_expected_fitted:
            raise ValueError(
                f"Length of param_bounds components must match the number of "
                f"parameters to be fitted ({num_expected_fitted})."
            )
        lower_b, upper_b = param_bounds
    else:
        lower_b, upper_b = ([-np.inf] * num_expected_fitted), ([np.inf] * num_expected_fitted)


    # --- Define the wrapper function for scipy.optimize.curve_fit ---
    # It takes q, then all *fitted* parameters
    def fit_wrapper_func(q_wrapper: np.ndarray, *fitted_args: float) -> np.ndarray:
        current_fitted_arg_idx = 0
        eval_params: Dict[str, float] = {} # To hold scale, background, and model params

        # Populate scale
        if 'scale' in _fixed_params:
            eval_params['scale'] = _fixed_params['scale']
        else:
            eval_params['scale'] = fitted_args[current_fitted_arg_idx]
            current_fitted_arg_idx += 1

        # Populate background
        if 'background' in _fixed_params:
            eval_params['background'] = _fixed_params['background']
        else:
            eval_params['background'] = fitted_args[current_fitted_arg_idx]
            current_fitted_arg_idx += 1
        
        # Populate model-specific parameters
        model_specific_args = []
        for p_name in param_names:
            if p_name in _fixed_params:
                model_specific_args.append(_fixed_params[p_name])
            else:
                model_specific_args.append(fitted_args[current_fitted_arg_idx])
                current_fitted_arg_idx += 1
        
        model_eval = model_func(q_wrapper, *model_specific_args)
        return eval_params['scale'] * model_eval + eval_params['background']

    # Populate p0_fit and bounds_fit for the actual fitting process
    # (only for parameters that are NOT fixed)
    for p_name_internal in internal_param_names_all: # scale, background, then model params
        if p_name_internal not in _fixed_params:
            p0_fit.append(initial_params[current_p0_idx])
            bounds_fit_lower.append(lower_b[current_p0_idx])
            bounds_fit_upper.append(upper_b[current_p0_idx])
            current_p0_idx += 1
        elif p_name_internal not in ['scale', 'background'] and p_name_internal not in param_names:
             # This case should ideally not be hit if param_names is correct
             raise ValueError(f"Unknown parameter '{p_name_internal}' encountered in fixed_params logic.") # pragma: no cover


    final_bounds = (bounds_fit_lower, bounds_fit_upper)

    # --- Perform the fit ---
    try:
        with warnings.catch_warnings(): # Catch OptimizeWarning from curve_fit
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                fit_wrapper_func,
                q_fit,
                i_fit,
                p0=p0_fit,
                sigma=sigma_fit, # Use errors if available
                absolute_sigma=True if sigma_fit is not None else False, # Treat sigma as true std dev
                bounds=final_bounds,
                **curve_fit_kwargs
            )
            fit_success = True
            fit_message = "Fit successful."
    except RuntimeError as e: # pragma: no cover
        warnings.warn(f"FitModel: `scipy.optimize.curve_fit` failed with RuntimeError: {e}", UserWarning)
        return None
    except OptimizeWarning as ow: # pragma: no cover
        # This might be caught by the filter above, but as a fallback
        warnings.warn(f"FitModel: `scipy.optimize.curve_fit` issued an OptimizeWarning: {ow}", UserWarning)
        # Continue, but mark success potentially based on pcov
        popt, pcov = ow.args[0] if len(ow.args) > 0 and isinstance(ow.args[0], tuple) else (p0_fit, np.full((len(p0_fit), len(p0_fit)), np.inf))
        fit_success = False # Or check if pcov is valid
        fit_message = str(ow)
    except ValueError as e: # e.g. incompatible shapes if bounds are wrong
        warnings.warn(f"FitModel: `scipy.optimize.curve_fit` failed with ValueError: {e}", UserWarning)
        return None


    # --- Extract results and errors ---
    fitted_params_all: Dict[str, float] = {}
    fitted_params_stderr_all: Dict[str, float] = {}
    
    popt_idx = 0
    for p_name_internal in internal_param_names_all:
        if p_name_internal in _fixed_params:
            fitted_params_all[p_name_internal] = _fixed_params[p_name_internal]
            fitted_params_stderr_all[p_name_internal] = 0.0 # No error for fixed params
        else:
            if popt_idx < len(popt):
                fitted_params_all[p_name_internal] = popt[popt_idx]
                try:
                    # Check for inf in diagonal of pcov, which means error is undefined
                    if pcov is not None and popt_idx < pcov.shape[0] and np.isfinite(pcov[popt_idx, popt_idx]) and pcov[popt_idx, popt_idx] >=0:
                         fitted_params_stderr_all[p_name_internal] = np.sqrt(pcov[popt_idx, popt_idx])
                    else:
                        fitted_params_stderr_all[p_name_internal] = np.nan
                except (TypeError, IndexError): # pragma: no cover
                    fitted_params_stderr_all[p_name_internal] = np.nan
                popt_idx += 1
            else: # Should not happen if logic is correct
                fitted_params_stderr_all[p_name_internal] = np.nan # pragma: no cover

    # --- Calculate fitted curve and chi-squared ---
    i_model_full = fit_wrapper_func(q_data_full, *[fitted_params_all[pn] for pn in internal_param_names_all if pn not in _fixed_params])
    fit_ScatteringCurve = ScatteringCurve1D(
        q=q_data_full,
        intensity=i_model_full,
        metadata={"source": "model_fit", "original_curve_id": id(curve)}
    )

    chi_squared_reduced = np.nan
    if sigma_fit is not None and len(sigma_fit) > 0 :
        residuals = (i_fit - fit_wrapper_func(q_fit, *[fitted_params_all[pn] for pn in internal_param_names_all if pn not in _fixed_params])) / sigma_fit
        degrees_of_freedom = len(q_fit) - len(popt)
        if degrees_of_freedom > 0:
            chi_squared_reduced = np.sum(residuals**2) / degrees_of_freedom
        else: # pragma: no cover
            warnings.warn("FitModel: Degrees of freedom <= 0, cannot calculate reduced chi-squared.", UserWarning)


    return {
        "fitted_params": fitted_params_all,
        "fitted_params_stderr": fitted_params_stderr_all,
        "covariance_matrix": pcov,
        "fit_curve": fit_ScatteringCurve,
        "chi_squared_reduced": chi_squared_reduced,
        "success": fit_success and (pcov is not None and not np.any(np.isinf(np.diag(pcov)))), # More robust success check
        "message": fit_message,
        "q_fit_min": q_fit.min(),
        "q_fit_max": q_fit.max(),
        "num_points_fit": len(q_fit),
    }