# scatterbrain/visualization.py
"""
Plotting and visualization utilities for the ScatterBrain library.

This module provides functions to generate common plots used in SAXS/WAXS analysis.
"""

import logging
from typing import Optional, Tuple, Any, Union, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .core import ScatteringCurve1D

logger = logging.getLogger(__name__)


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
            logger.warning("Mismatch between number of curves and labels provided. Ignoring labels.")
            labels_list = None
        else:
            labels_list = None
    else:
        raise TypeError("Input 'curves' must be a ScatteringCurve1D object or a list of them.")

    if not curves_list:
        logger.warning("plot_iq: No curves provided to plot.")
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


def plot_guinier(
    curve: ScatteringCurve1D,
    guinier_result: Optional[Dict[str, Any]] = None,
    q_range_highlight: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = "Guinier Plot",
    **plot_kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Generate a Guinier plot: ln(I(q)) vs q².

    Parameters
    ----------
    curve : ScatteringCurve1D
        The scattering data to plot.  Points with I(q) ≤ 0 are excluded.
    guinier_result : dict, optional
        Output of :func:`scatterbrain.analysis.guinier.guinier_fit`.
        When provided, the fitted line is overlaid and Rg / I(0) are
        annotated on the axes.
    q_range_highlight : tuple of float, optional
        ``(q_min, q_max)`` region to shade as the Guinier fit region.
        Overridden by values in *guinier_result* if available.
    ax : Axes, optional
        Existing matplotlib Axes to draw on.  A new figure is created
        when *ax* is None (default).
    title : str, optional
        Plot title.  Default ``"Guinier Plot"``.
    **plot_kwargs
        Passed to the data ``ax.errorbar`` / ``ax.plot`` call.

    Returns
    -------
    tuple of (Figure, Axes)
    """
    if ax is None:
        fig, current_ax = plt.subplots(figsize=(7, 5))
    else:
        current_ax = ax
        fig = current_ax.get_figure()

    # Filter to positive intensities
    mask = curve.intensity > 0
    q_pos = curve.q[mask]
    i_pos = curve.intensity[mask]
    q2 = q_pos ** 2
    ln_i = np.log(i_pos)

    # Error propagation: σ_lnI = σ_I / I
    if curve.error is not None:
        err_pos = curve.error[mask]
        ln_i_err = np.abs(err_pos / i_pos)
        current_ax.errorbar(
            q2, ln_i, yerr=ln_i_err,
            fmt=".", markersize=3, capsize=2, alpha=0.6, elinewidth=0.8,
            label=curve.metadata.get("filename", "Data"),
            **plot_kwargs,
        )
    else:
        current_ax.plot(
            q2, ln_i,
            ".", markersize=3,
            label=curve.metadata.get("filename", "Data"),
            **plot_kwargs,
        )

    # Overlay fitted line
    if guinier_result is not None and not np.isnan(guinier_result.get("Rg", np.nan)):
        slope = guinier_result["slope"]
        intercept = guinier_result["intercept"]
        q_min_fit = guinier_result.get("q_fit_min", q_pos.min())
        q_max_fit = guinier_result.get("q_fit_max", q_pos.max())

        q2_fit = np.linspace(q_min_fit ** 2, q_max_fit ** 2, 100)
        current_ax.plot(
            q2_fit, slope * q2_fit + intercept,
            "-", color="tab:red", linewidth=1.5, label="Guinier fit",
        )

        rg = guinier_result["Rg"]
        rg_err = guinier_result.get("Rg_err", np.nan)
        i0 = guinier_result["I0"]
        annotation = (
            f"$R_g$ = {rg:.3g}"
            + (f" ± {rg_err:.2g}" if not np.isnan(rg_err) else "")
            + f" {curve.q_unit}$^{{-1}}$\n"
            f"$I(0)$ = {i0:.3g} {curve.intensity_unit}"
        )
        current_ax.annotate(
            annotation,
            xy=(0.98, 0.98), xycoords="axes fraction",
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

        # Use fit q-range for highlight if not explicitly given
        if q_range_highlight is None:
            q_range_highlight = (q_min_fit, q_max_fit)

    # Shade the Guinier region
    if q_range_highlight is not None:
        current_ax.axvspan(
            q_range_highlight[0] ** 2, q_range_highlight[1] ** 2,
            alpha=0.12, color="tab:green", label="Fit region",
        )

    current_ax.set_xlabel(f"q² ({curve.q_unit})²")
    current_ax.set_ylabel("ln(I(q))")
    if title:
        current_ax.set_title(title)
    current_ax.legend(fontsize=8)
    current_ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig, current_ax


def plot_porod(
    curve: ScatteringCurve1D,
    porod_result: Optional[Dict[str, Any]] = None,
    plot_type: str = "Iq4_vs_q",
    q_range_highlight: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = "Porod Plot",
    **plot_kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Generate a Porod plot.

    Parameters
    ----------
    curve : ScatteringCurve1D
        The scattering data.  Points with I(q) ≤ 0 or q ≤ 0 are excluded.
    porod_result : dict, optional
        Output of :func:`scatterbrain.analysis.porod.porod_analysis`.
        When provided and ``plot_type="logI_vs_logq"``, the fitted line is
        overlaid and the Porod exponent / constant are annotated.
    plot_type : {"Iq4_vs_q", "logI_vs_logq"}, optional
        ``"Iq4_vs_q"`` (default): I(q)·q⁴ vs q on linear axes; a flat
        plateau signals Porod behaviour.
        ``"logI_vs_logq"``: log₁₀(I) vs log₁₀(q); slope ≈ −4 for smooth
        3-D interfaces.
    q_range_highlight : tuple of float, optional
        ``(q_min, q_max)`` region to shade.  Falls back to values in
        *porod_result* when available.
    ax : Axes, optional
        Existing axes to draw on.
    title : str, optional
        Plot title.  Default ``"Porod Plot"``.
    **plot_kwargs
        Passed to the data plot call.

    Returns
    -------
    tuple of (Figure, Axes)

    Raises
    ------
    ValueError
        If *plot_type* is not one of the accepted strings.
    """
    if plot_type not in ("Iq4_vs_q", "logI_vs_logq"):
        raise ValueError(
            f"plot_type must be 'Iq4_vs_q' or 'logI_vs_logq', got '{plot_type}'."
        )

    if ax is None:
        fig, current_ax = plt.subplots(figsize=(7, 5))
    else:
        current_ax = ax
        fig = current_ax.get_figure()

    # Filter valid points
    mask = (curve.intensity > 0) & (curve.q > 0)
    q_pos = curve.q[mask]
    i_pos = curve.intensity[mask]

    if plot_type == "Iq4_vs_q":
        x_data = q_pos
        y_data = i_pos * q_pos ** 4
        xlabel = f"q ({curve.q_unit})"
        ylabel = f"I(q)·q⁴ ({curve.intensity_unit}·{curve.q_unit}⁴)"
        current_ax.plot(x_data, y_data, ".", markersize=3,
                        label=curve.metadata.get("filename", "Data"), **plot_kwargs)
    else:  # logI_vs_logq
        x_data = np.log10(q_pos)
        y_data = np.log10(i_pos)
        xlabel = f"log₁₀(q / {curve.q_unit})"
        ylabel = f"log₁₀(I / {curve.intensity_unit})"
        current_ax.plot(x_data, y_data, ".", markersize=3,
                        label=curve.metadata.get("filename", "Data"), **plot_kwargs)

        # Overlay fitted line from porod_result
        if porod_result is not None:
            q_min_fit = porod_result.get("q_fit_min", q_pos.min())
            q_max_fit = porod_result.get("q_fit_max", q_pos.max())
            log_q_fit = np.linspace(np.log10(q_min_fit), np.log10(q_max_fit), 100)
            intercept = porod_result.get("log_kp_intercept")
            exponent = porod_result.get("porod_exponent")
            if intercept is not None and exponent is not None:
                log_i_fit = intercept - exponent * log_q_fit
                current_ax.plot(
                    log_q_fit, log_i_fit,
                    "-", color="tab:red", linewidth=1.5, label="Porod fit",
                )
            if q_range_highlight is None:
                q_range_highlight = (q_min_fit, q_max_fit)

    # Annotate Porod parameters
    if porod_result is not None:
        exp = porod_result.get("porod_exponent")
        exp_err = porod_result.get("porod_exponent_err", np.nan)
        kp = porod_result.get("porod_constant_kp")
        parts = []
        if exp is not None:
            parts.append(
                f"n = {exp:.3g}"
                + (f" ± {exp_err:.2g}" if exp_err is not None and not np.isnan(exp_err) else "")
            )
        if kp is not None:
            parts.append(f"Kp = {kp:.3g}")
        if parts:
            current_ax.annotate(
                "\n".join(parts),
                xy=(0.02, 0.02), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

    # Shade analysis region
    if q_range_highlight is not None:
        if plot_type == "Iq4_vs_q":
            current_ax.axvspan(
                q_range_highlight[0], q_range_highlight[1],
                alpha=0.12, color="tab:orange", label="Analysis region",
            )
        else:
            current_ax.axvspan(
                np.log10(q_range_highlight[0]), np.log10(q_range_highlight[1]),
                alpha=0.12, color="tab:orange", label="Analysis region",
            )

    current_ax.set_xlabel(xlabel)
    current_ax.set_ylabel(ylabel)
    if title:
        current_ax.set_title(title)
    current_ax.legend(fontsize=8)
    current_ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig, current_ax


def plot_fit(
    curve: ScatteringCurve1D,
    fit_result_dict: Dict[str, Any],
    q_scale: str = "log",
    i_scale: str = "log",
    plot_residuals: bool = True,
    ax_main: Optional[Axes] = None,
    ax_res: Optional[Axes] = None,
    title: Optional[str] = "Model Fit",
    **plot_kwargs: Any,
) -> Figure:
    """
    Plot experimental data with a fitted model overlay and optional residuals.

    Parameters
    ----------
    curve : ScatteringCurve1D
        The experimental data.
    fit_result_dict : dict
        Output of :func:`scatterbrain.modeling.fitting.fit_model`.
        Must contain ``"fit_curve"`` (a ScatteringCurve1D) and optionally
        ``"chi_squared_reduced"`` and ``"fitted_params"``.
    q_scale : {"log", "linear"}, optional
        Scale for the q-axis.  Default ``"log"``.
    i_scale : {"log", "linear"}, optional
        Scale for the intensity axis.  Default ``"log"``.
    plot_residuals : bool, optional
        When True (default) and ``curve.error`` is available, a lower
        panel showing normalised residuals ``(I_data − I_model) / σ`` is
        added.  When errors are unavailable only the main panel is shown
        even if this flag is True.
    ax_main : Axes, optional
        Axes for the main data/fit panel.  Created internally when None.
    ax_res : Axes, optional
        Axes for the residuals panel.  Created internally when None and
        residuals are requested.
    title : str, optional
        Figure title.  Default ``"Model Fit"``.
    **plot_kwargs
        Passed to the experimental data plot call.

    Returns
    -------
    Figure
    """
    fit_curve: ScatteringCurve1D = fit_result_dict["fit_curve"]
    has_errors = curve.error is not None
    show_residuals = plot_residuals and has_errors

    # --- Figure / axes setup ---
    if ax_main is not None:
        current_ax_main = ax_main
        fig = current_ax_main.get_figure()
        current_ax_res = ax_res  # may be None; residuals will be skipped
    elif show_residuals:
        fig, (current_ax_main, current_ax_res) = plt.subplots(
            2, 1, figsize=(7, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )
    else:
        fig, current_ax_main = plt.subplots(figsize=(7, 5))
        current_ax_res = None

    # --- Main panel: data + model ---
    data_label = curve.metadata.get("filename", "Data")
    if has_errors:
        current_ax_main.errorbar(
            curve.q, curve.intensity, yerr=curve.error,
            fmt=".", markersize=3, capsize=2, alpha=0.6, elinewidth=0.8,
            label=data_label, **plot_kwargs,
        )
    else:
        current_ax_main.plot(
            curve.q, curve.intensity,
            ".", markersize=3, label=data_label, **plot_kwargs,
        )

    current_ax_main.plot(
        fit_curve.q, fit_curve.intensity,
        "-", color="tab:red", linewidth=1.5, label="Model fit",
    )

    current_ax_main.set_xscale(q_scale)
    current_ax_main.set_yscale(i_scale)
    current_ax_main.set_ylabel(f"Intensity ({curve.intensity_unit})")
    if not show_residuals:
        unit_text = curve.q_unit.replace("^", "$^{") + "}$" if curve.q_unit else ""
        current_ax_main.set_xlabel(f"q ({unit_text})" if unit_text else "q")
    current_ax_main.legend(fontsize=8)
    current_ax_main.grid(True, which="both", linestyle="--", alpha=0.4)

    # Parameter annotation
    fitted = fit_result_dict.get("fitted_params", {})
    chi2 = fit_result_dict.get("chi_squared_reduced", np.nan)
    ann_parts = []
    if not np.isnan(chi2):
        ann_parts.append(f"χ²_red = {chi2:.3g}")
    for k, v in fitted.items():
        ann_parts.append(f"{k} = {v:.4g}")
    if ann_parts:
        current_ax_main.annotate(
            "\n".join(ann_parts),
            xy=(0.98, 0.98), xycoords="axes fraction",
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    if title:
        fig.suptitle(title, fontsize=11)

    # --- Residuals panel ---
    if show_residuals and current_ax_res is not None:
        # Interpolate model onto data q-grid for residual calculation
        model_i_on_data = np.interp(curve.q, fit_curve.q, fit_curve.intensity)
        residuals = (curve.intensity - model_i_on_data) / curve.error
        current_ax_res.axhline(0, color="tab:red", linewidth=1)
        current_ax_res.plot(curve.q, residuals, ".", markersize=3, alpha=0.7)
        current_ax_res.set_xscale(q_scale)
        current_ax_res.set_ylabel("(I−M) / σ")
        unit_text = curve.q_unit.replace("^", "$^{") + "}$" if curve.q_unit else ""
        current_ax_res.set_xlabel(f"q ({unit_text})" if unit_text else "q")
        current_ax_res.grid(True, which="both", linestyle="--", alpha=0.4)
        current_ax_res.axhline(1, color="gray", linewidth=0.5, linestyle=":")
        current_ax_res.axhline(-1, color="gray", linewidth=0.5, linestyle=":")

    fig.tight_layout()
    return fig


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

    # 4. Test no curves (warning now goes to logger, not warnings module)
    plot_iq([])
    plt.show() # Shows an empty plot