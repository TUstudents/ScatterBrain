# scatterbrain/processing/background.py
"""
Background subtraction functions for 1D scattering curves.
"""

import logging
import copy
from typing import Union

import numpy as np

from ..core import ScatteringCurve1D
from ..utils import ProcessingError

logger = logging.getLogger(__name__)


def subtract_background(
    curve: ScatteringCurve1D,
    background: Union[ScatteringCurve1D, float],
    interpolate: bool = True,
    scale_factor: float = 1.0,
) -> ScatteringCurve1D:
    """
    Subtract a background from a scattering curve.

    Parameters
    ----------
    curve : ScatteringCurve1D
        The signal curve to correct.
    background : ScatteringCurve1D or float
        Background to subtract. A float is treated as a constant offset.
        A ScatteringCurve1D is interpolated (or required to match) the
        signal q-grid before subtraction.
    interpolate : bool, optional
        If *background* is a ScatteringCurve1D and q-grids differ, interpolate
        the background onto the signal q-grid using ``numpy.interp``.
        If False, raise ``ProcessingError`` when q-grids do not match exactly.
        Default is True.
    scale_factor : float, optional
        Multiply the background by this factor before subtracting.
        Useful for transmission or thickness corrections. Default is 1.0.

    Returns
    -------
    ScatteringCurve1D
        A new curve with the background subtracted.  The original objects
        are not modified.

    Raises
    ------
    TypeError
        If *background* is not a ``ScatteringCurve1D`` or a number.
    ProcessingError
        If q-grids differ and ``interpolate=False``, or if the background
        q-range does not cover the signal q-range when interpolating.
    """
    if not isinstance(curve, ScatteringCurve1D):
        raise TypeError("'curve' must be a ScatteringCurve1D object.")

    new_metadata = copy.deepcopy(curve.metadata)

    # --- Constant background ---
    if isinstance(background, (int, float)):
        bg_value = float(background) * scale_factor
        new_intensity = curve.intensity - bg_value
        new_error = np.copy(curve.error) if curve.error is not None else None
        new_metadata.setdefault("processing_history", []).append(
            f"Constant background subtracted: {bg_value:.4g} "
            f"(raw={float(background):.4g}, scale_factor={scale_factor})."
        )
        return ScatteringCurve1D(
            q=np.copy(curve.q),
            intensity=new_intensity,
            error=new_error,
            metadata=new_metadata,
            q_unit=curve.q_unit,
            intensity_unit=curve.intensity_unit,
        )

    # --- Curve background ---
    if not isinstance(background, ScatteringCurve1D):
        raise TypeError(
            "'background' must be a ScatteringCurve1D or a numeric constant. "
            f"Got {type(background)}."
        )

    # Determine background intensity on the signal q-grid
    q_sig = curve.q
    bg_i = background.intensity * scale_factor
    bg_q = background.q

    grids_match = (
        len(q_sig) == len(bg_q) and np.allclose(q_sig, bg_q, rtol=1e-6, atol=0)
    )

    if grids_match:
        bg_on_signal_grid = bg_i
        bg_err_on_grid = (
            background.error * scale_factor if background.error is not None else None
        )
    elif interpolate:
        # Warn if background doesn't fully cover signal range
        if bg_q.min() > q_sig.min() or bg_q.max() < q_sig.max():
            logger.warning(
                "subtract_background: background q-range [%.4g, %.4g] does not fully "
                "cover signal q-range [%.4g, %.4g]. Boundary values will be used for "
                "extrapolation (numpy.interp clamps to edge values).",
                bg_q.min(), bg_q.max(), q_sig.min(), q_sig.max(),
            )
        bg_on_signal_grid = np.interp(q_sig, bg_q, bg_i)
        if background.error is not None:
            bg_err_on_grid = np.interp(q_sig, bg_q, background.error * scale_factor)
        else:
            bg_err_on_grid = None
        logger.debug(
            "subtract_background: background interpolated onto signal q-grid."
        )
    else:
        raise ProcessingError(
            "subtract_background: signal and background q-grids do not match and "
            "'interpolate' is False. Pass interpolate=True or ensure matching grids."
        )

    new_intensity = curve.intensity - bg_on_signal_grid

    # Error propagation in quadrature
    if curve.error is not None and bg_err_on_grid is not None:
        new_error = np.sqrt(curve.error**2 + bg_err_on_grid**2)
    elif curve.error is not None:
        new_error = np.copy(curve.error)
    elif bg_err_on_grid is not None:
        new_error = bg_err_on_grid
    else:
        new_error = None

    new_metadata.setdefault("processing_history", []).append(
        f"Curve background subtracted (scale_factor={scale_factor}, "
        f"interpolate={interpolate})."
    )

    return ScatteringCurve1D(
        q=np.copy(q_sig),
        intensity=new_intensity,
        error=new_error,
        metadata=new_metadata,
        q_unit=curve.q_unit,
        intensity_unit=curve.intensity_unit,
    )
