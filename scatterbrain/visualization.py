# scatterbrain/visualization.py
"""
Plotting and visualization utilities for the ScatterBrain library.

This module provides functions to generate common plots used in SAXS/WAXS analysis.
"""

from typing import Optional, Tuple, Any, Union, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .core import ScatteringCurve1D
# Analysis results types might be imported if we define specific result objects later
# from ..analysis.guinier import GuinierResultType ( hypothetical )
import warnings
# Define the type for the curves parameter


def plot_iq(
    curves: Union[ScatteringCurve1D, List[ScatteringCurve1D]],
    q_scale: str = 'log',
    i_scale: str = 'log',
    labels: Optional[Union[str, List[str]]] = None,
    title: Optional[str] = "Scattering Intensity",
    xlabel: Optional[str] = None, # Auto-generated if None
    ylabel: Optional[str] = None, # Auto-generated if None
    ax: Optional[Axes] = None,
    show_legend: bool = True,
    errorbars: bool = True,
    errorbar_kwargs: Optional[Dict[str, Any]] = None,
    **plot_kwargs: Any
) -> Tuple[Figure, Axes]:
    """
    Plots I(q) vs q for one or more ScatteringCurve1D objects.

    Parameters
    ----------
    curves : Union[ScatteringCurve1D, List[ScatteringCurve1D]]
        A single ScatteringCurve1D object or a list of them to plot.
    q_scale : str, optional
        Scale for the q-axis ('linear', 'log'). Default is 'log'.
    i_scale : str, optional
        Scale for the I-axis ('linear', 'log'). Default is 'log'.
    labels : Optional[Union[str, List[str]]], optional
        Label(s) for the curve(s) in the legend. If a single curve is
        provided and label is a string, it's used. If a list of curves
        is provided, this should be a list of strings of the same length.
        If None, a default label (e.g., from metadata or index) might be used.
    title : Optional[str], optional
        Title for the plot. Default is "Scattering Intensity".
    xlabel : Optional[str], optional
        Label for the q-axis. If None, it's auto-generated using the
        q_unit from the first curve (e.g., "q (nm$^{-1}$)") .
    ylabel : Optional[str], optional
        Label for the I-axis. If None, it's auto-generated using the
        intensity_unit from the first curve (e.g., "Intensity (a.u.)").
    ax : Optional[Axes], optional
        A matplotlib Axes object to plot on. If None (default), a new
        figure and axes are created.
    show_legend : bool, optional
        Whether to display the legend. Default is True if labels are present.
    errorbars : bool, optional
        If True (default) and curve.error is available, error bars are plotted.
    errorbar_kwargs : Optional[Dict[str, Any]], optional
        Additional keyword arguments to pass to `ax.errorbar()`.
        Default includes `fmt='.'`, `capsize=3`, `alpha=0.7`.
    **plot_kwargs : Any
        Additional keyword arguments to pass to `ax.plot()` for line plotting.

    Returns
    -------
    Tuple[Figure, Axes]
        The matplotlib Figure and Axes objects containing the plot.
    """
    if isinstance(curves, ScatteringCurve1D):
        curves_list = [curves]
        if labels is not None and isinstance(labels, str):
            labels_list: Optional[List[str]] = [labels]
        elif labels is not None and isinstance(labels, list) and len(labels) == 1:
            labels_list = labels
        else:
            labels_list = None # Will use default
    elif isinstance(curves, list) and all(isinstance(c, ScatteringCurve1D) for c in curves):
        curves_list = curves
        if labels is not None and isinstance(labels, list) and len(labels) == len(curves_list):
            labels_list = labels
        elif labels is not None:
            warnings.warn("Mismatch between number of curves and labels provided. Ignoring labels.", UserWarning)
            labels_list = None
        else:
            labels_list = None
    else:
        raise TypeError("Input 'curves' must be a ScatteringCurve1D object or a list of them.")

    if not curves_list:
        warnings.warn("plot_iq: No curves provided to plot.", UserWarning)
        # Create an empty plot or raise error? For now, create empty.
        fig, current_ax = plt.subplots()
        if title: current_ax.set_title(title)
        current_ax.set_xlabel(xlabel or "q")
        current_ax.set_ylabel(ylabel or "Intensity")
        return fig, current_ax


    if ax is None:
        fig, current_ax = plt.subplots(figsize=(8,6)) # Default figure size
    else:
        current_ax = ax
        fig = current_ax.get_figure()

    # Determine axis labels from the first curve if not provided
    first_curve = curves_list[0]
    if xlabel is None:
        #q_unit_str = f" ({first_curve.q_unit})" if first_curve.q_unit else ""
        #xlabel = f"q{q_unit_str}"
        unit_text = first_curve.q_unit.replace("^", "$^{") + "}$" if first_curve.q_unit else ""
        xlabel = f"q ({unit_text})" if unit_text else "q"
    if ylabel is None:
        unit_text = first_curve.intensity_unit if first_curve.intensity_unit else "a.u."
        if "^" in unit_text:
            unit_text = unit_text.replace("^", "$^{") + "}$"
        ylabel = f"Intensity ({unit_text})" if unit_text else "Intensity"

    _errorbar_kwargs = {
        "fmt": '.',       # Plot points for error bars
        "markersize": 3,  # Size of the points
        "capsize": 2,     # Size of the error bar caps
        "alpha": 0.6,     # Transparency
        "elinewidth": 0.8, # Line width of error bars
        "ecolor": None    # Default to None to use current color
    }
    if errorbar_kwargs:
        # If ecolor is specified, ensure alpha is applied correctly
        if 'ecolor' in errorbar_kwargs and 'alpha' in errorbar_kwargs:
            color = plt.matplotlib.colors.to_rgba(errorbar_kwargs['ecolor'], 
                                               alpha=errorbar_kwargs['alpha'])
            errorbar_kwargs = dict(errorbar_kwargs)  # Make a copy
            errorbar_kwargs['ecolor'] = color
        _errorbar_kwargs.update(errorbar_kwargs)


    has_any_label = False
    for i, curve_obj in enumerate(curves_list):
        label: Optional[str] = None
        if labels_list and i < len(labels_list):
            label = labels_list[i]
        elif "filename" in curve_obj.metadata:
            label = curve_obj.metadata["filename"]
        elif len(curves_list) > 1:
            label = f"Curve {i+1}"
        
        if label:
            has_any_label = True

        # Ensure plot_kwargs doesn't contain label if we're using the label parameter
        plot_args = dict(plot_kwargs)
        if label and 'label' in plot_args:
            del plot_args['label']

        if errorbars and curve_obj.error is not None and len(curve_obj.error) == len(curve_obj.q):
            current_ax.errorbar(
                curve_obj.q, curve_obj.intensity, yerr=curve_obj.error,
                label=label if _errorbar_kwargs.get('fmt', '.') != '' else None,
                **_errorbar_kwargs
            )
            if _errorbar_kwargs.get('fmt', '.') != '':
                current_ax.plot(curve_obj.q, curve_obj.intensity, **plot_args)
        else:
            current_ax.plot(curve_obj.q, curve_obj.intensity, label=label, **plot_args)

    current_ax.set_xscale(q_scale)
    current_ax.set_yscale(i_scale)

    current_ax.set_xlabel(xlabel)
    current_ax.set_ylabel(ylabel)
    if title:
        current_ax.set_title(title)

    if show_legend and has_any_label:
        current_ax.legend()

    current_ax.grid(True, which="both", axis="both", linestyle='--', alpha=0.5)
    fig.tight_layout()

    return fig, current_ax


# Placeholder for Guinier plot
def plot_guinier(
    curve: ScatteringCurve1D,
    guinier_result: Optional[Dict[str, Any]] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = "Guinier Plot",
    **plot_kwargs: Any
) -> Tuple[Figure, Axes]:
    """
    Generates a Guinier plot (ln(I) vs q^2).
    (Placeholder - to be implemented)
    """
    raise NotImplementedError("plot_guinier is not yet implemented.")

# Placeholder for Porod plot
def plot_porod(
    curve: ScatteringCurve1D,
    porod_result: Optional[Dict[str, Any]] = None,
    plot_type: str = "Iq4_vs_q", # or "logI_vs_logq"
    ax: Optional[Axes] = None,
    title: Optional[str] = "Porod Plot",
    **plot_kwargs: Any
) -> Tuple[Figure, Axes]:
    """
    Generates a Porod plot (e.g., I*q^4 vs q or log(I) vs log(q)).
    (Placeholder - to be implemented)
    """
    raise NotImplementedError("plot_porod is not yet implemented.")


# Placeholder for Model Fit plot
def plot_fit(
    curve: ScatteringCurve1D,
    fit_result_dict: Dict[str, Any], # Expects output from fit_model
    q_scale: str = 'log',
    i_scale: str = 'log',
    plot_residuals: bool = True,
    ax_main: Optional[Axes] = None, # For main plot
    ax_res: Optional[Axes] = None,  # For residuals, if plot_residuals is True
    title: Optional[str] = "Model Fit",
    **plot_kwargs: Any
) -> Figure: # Returns Figure, Axes arrangement depends on residuals
    """
    Plots the experimental data, the fitted model, and optionally residuals.
    (Placeholder - to be implemented)
    """
    raise NotImplementedError("plot_fit is not yet implemented.")


if __name__ == "__main__": # pragma: no cover
    # --- Example Usage ---
    # Create some dummy ScatteringCurve1D objects
    q1 = np.linspace(0.01, 0.5, 100)
    i1 = 1000 * np.exp(-(q1**2 * 5**2) / 3.0) + np.random.normal(0, 5, 100) + 5
    e1 = np.sqrt(i1) * 0.1
    curve1 = ScatteringCurve1D(q1, i1, e1, metadata={"filename": "Sphere R=5nm"}, q_unit="nm^-1")

    q2 = np.linspace(0.02, 0.8, 80)
    i2 = 500 * np.exp(-(q2**2 * 10**2) / 3.0) + np.random.normal(0, 2, 80) + 2
    curve2 = ScatteringCurve1D(q2, i2, metadata={"filename": "Sphere R=10nm (no error)"}, q_unit="nm^-1")
    
    # 1. Plot a single curve
    fig1, ax1 = plot_iq(curve1, title="Single Curve Example", errorbars=True)
    plt.show()

    # 2. Plot multiple curves
    fig2, ax2 = plot_iq(
        [curve1, curve2],
        labels=["R=5nm Data", "R=10nm Data (no err)"],
        title="Multiple Curves Example",
        plot_kwargs={'linewidth': 1.5} # Pass kwargs to ax.plot
    )
    plt.show()

    # 3. Plot on existing axes and change scales
    fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_iq(curve1, ax=ax3_1, q_scale='linear', i_scale='linear', title="Linear Scale")
    plot_iq(curve2, ax=ax3_2, q_scale='log', i_scale='log', title="Log Scale (Curve 2)", errorbars=False)
    fig3.suptitle("Plotting on Existing Axes")
    plt.show()

    # 4. Test no curves
    with warnings.catch_warnings(record=True) as w_empty:
        plot_iq([])
        assert len(w_empty) == 1
        assert "No curves provided" in str(w_empty[0].message)
    plt.show() # Shows an empty plot