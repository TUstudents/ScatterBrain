# tests/test_visualization.py
"""
Unit tests for the scatterbrain.visualization module.
"""
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import warnings

from scatterbrain.core import ScatteringCurve1D
from scatterbrain.visualization import plot_iq #, plot_guinier, plot_porod, plot_fit (placeholders)

# --- Fixtures for Visualization Tests ---

@pytest.fixture(autouse=True)
def mpl_backend_config():
    """Configure matplotlib for testing."""
    backend = matplotlib.get_backend()
    if backend != 'agg':
        matplotlib.use('Agg')
    yield
    plt.close('all')

@pytest.fixture
def sample_curve1() -> ScatteringCurve1D:
    """A sample ScatteringCurve1D object for plotting."""
    q = np.linspace(0.01, 0.5, 50)
    i = 100 * np.exp(-q**2 * 4**2 / 3) + 5 + np.random.rand(50) * 2
    e = np.sqrt(i) * 0.1
    return ScatteringCurve1D(q, i, e, metadata={"filename": "sample1.dat"}, q_unit="nm^-1", intensity_unit="a.u.")

@pytest.fixture
def sample_curve2_no_error() -> ScatteringCurve1D:
    """Another sample curve, without error data."""
    q = np.linspace(0.05, 0.8, 60)
    i = 200 * np.exp(-q**2 * 8**2 / 3) + 2 + np.random.rand(60)
    return ScatteringCurve1D(q, i, metadata={"filename": "sample2.dat"}, q_unit="A^-1", intensity_unit="counts")

@pytest.fixture
def plt_close_figures():
    """Fixture to close all matplotlib figures after a test."""
    yield
    plt.close('all') # Close all figures to avoid state leakage between tests

# --- Test Cases for plot_iq ---

@pytest.mark.usefixtures("plt_close_figures") # Ensure figures are closed after each test
class TestPlotIQ:
    """Tests for the plot_iq function."""

    def test_plot_single_curve(self, sample_curve1: ScatteringCurve1D):
        """Test plotting a single curve with default settings."""
        fig, ax = plot_iq(sample_curve1)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.lines) == 1 or len(ax.containers) > 0 # Line or errorbar container
        assert ax.get_title() == "Scattering Intensity"
        assert "q (nm$^{-1}$)" in ax.get_xlabel() # Check for unit inclusion
        assert "Intensity (a.u.)" in ax.get_ylabel()
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'
        assert ax.legend_ is not None # Default label from filename

    def test_plot_multiple_curves(self, sample_curve1: ScatteringCurve1D, sample_curve2_no_error: ScatteringCurve1D):
        """Test plotting multiple curves."""
        curves = [sample_curve1, sample_curve2_no_error]
        labels = ["Curve A", "Curve B"]
        fig, ax = plot_iq(curves, labels=labels)
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        
        num_artists = 0
        if sample_curve1.error is not None: # sample_curve1 has error bars
            num_artists += 1 # errorbar container usually
            if ax.containers: # check if it plotted line on top
                 num_artists += sum(1 for line in ax.lines if line.get_label() == labels[0])

        num_artists += sum(1 for line in ax.lines if line.get_label() == labels[1])
        
        # This count can be tricky due to how errorbar + line might be plotted
        # A simpler check might be on legend items or specific line properties
        assert len(ax.lines) + len(ax.containers) >= 2 # At least two primary plot elements

        legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
        assert labels[0] in legend_texts
        assert labels[1] in legend_texts

    def test_plot_on_existing_axes(self, sample_curve1: ScatteringCurve1D):
        """Test plotting on a pre-existing Axes object."""
        fig_ext, ax_ext = plt.subplots()
        orig_title = "Original Title"
        ax_ext.set_title(orig_title)

        fig, ax = plot_iq(sample_curve1, ax=ax_ext, title="New Plot Title")
        assert fig is fig_ext # Should be the same figure
        assert ax is ax_ext   # Should be the same axes
        assert ax.get_title() == "New Plot Title" # Title should be updated by plot_iq
        assert len(ax.lines) > 0 or len(ax.containers) > 0

    def test_plot_scales_and_labels(self, sample_curve1: ScatteringCurve1D):
        """Test custom scales, title, and axis labels."""
        custom_title = "My Custom Plot"
        custom_xlabel = "Momentum Transfer q"
        custom_ylabel = "Scattered Intensity I"
        fig, ax = plot_iq(
            sample_curve1,
            q_scale='linear',
            i_scale='linear',
            title=custom_title,
            xlabel=custom_xlabel,
            ylabel=custom_ylabel
        )
        assert ax.get_xscale() == 'linear'
        assert ax.get_yscale() == 'linear'
        assert ax.get_title() == custom_title
        assert ax.get_xlabel() == custom_xlabel
        assert ax.get_ylabel() == custom_ylabel

    def test_plot_no_errorbars(self, sample_curve1: ScatteringCurve1D):
        """Test plotting without error bars even if available."""
        fig, ax = plot_iq(sample_curve1, errorbars=False)
        # Check that no errorbar containers are present
        assert not any(isinstance(child, plt.matplotlib.container.ErrorbarContainer) for child in ax.containers)
        assert len(ax.lines) == 1 # Should only be one line for the data

    def test_plot_curve_without_errors(self, sample_curve2_no_error: ScatteringCurve1D):
        """Test plotting a curve that has no error data by default."""
        fig, ax = plot_iq(sample_curve2_no_error, errorbars=True) # errorbars=True should have no effect
        assert not any(isinstance(child, plt.matplotlib.container.ErrorbarContainer) for child in ax.containers)
        assert len(ax.lines) == 1
        assert "q (A$^{-1}$)" in ax.get_xlabel() # Check different unit

    def test_plot_no_legend(self, sample_curve1: ScatteringCurve1D):
        """Test disabling the legend."""
        fig, ax = plot_iq(sample_curve1, show_legend=False)
        assert ax.legend_ is None

    def test_plot_custom_plot_kwargs(self, sample_curve1: ScatteringCurve1D):
        """Test passing custom plot_kwargs."""
        fig, ax = plot_iq(sample_curve1, errorbars=False, color='red', linestyle='--', linewidth=2)
        assert len(ax.lines) == 1
        line = ax.lines[0]
        assert line.get_color() == 'red'
        assert line.get_linestyle() == '--'
        assert line.get_linewidth() == 2

    def test_plot_custom_errorbar_kwargs(self, sample_curve1: ScatteringCurve1D):
        """Test passing custom errorbar_kwargs."""
        custom_kwargs = {'ecolor': 'green', 'capsize': 5, 'alpha': 0.6}
        fig, ax = plot_iq(sample_curve1, errorbars=True, errorbar_kwargs=custom_kwargs)
        
        # Find the errorbar container
        eb_container = None
        for container in ax.containers:
            if isinstance(container, plt.matplotlib.container.ErrorbarContainer):
                eb_container = container
                break
        assert eb_container is not None
    
        # The error bar lines are in eb_container[2]
        error_lines = eb_container[2]
        assert isinstance(error_lines, tuple)  # Error lines are stored as a tuple
        
        # Get color of first error line
        error_line_color = error_lines[0].get_color()
        if isinstance(error_line_color, np.ndarray):
            # Convert 'green' to RGBA with specified alpha
            expected_color = plt.matplotlib.colors.to_rgba('green', alpha=custom_kwargs['alpha'])
            assert np.allclose(error_line_color, expected_color)
        else:
            assert error_line_color == 'green'
            
        # Verify capsize - matplotlib multiplies capsize by 2 internally
        caps = eb_container[1]  # Get the caps
        assert len(caps) > 0
        assert caps[0].get_markersize() == 2 * custom_kwargs['capsize']  # Account for internal scaling

    def test_plot_empty_curve_list(self):
        """Test plotting an empty list of curves."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax = plot_iq([])
            assert len(w) == 1
            assert "No curves provided" in str(w[0].message)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.lines) == 0 # No lines plotted
        assert len(ax.containers) == 0

    def test_invalid_curves_input(self):
        """Test providing invalid input for the 'curves' parameter."""
        with pytest.raises(TypeError, match="Input 'curves' must be a ScatteringCurve1D object or a list of them."):
            plot_iq("not_a_curve")
        with pytest.raises(TypeError, match="Input 'curves' must be a ScatteringCurve1D object or a list of them."):
            plot_iq([1, 2, 3]) # List of non-ScatteringCurve1D objects

    def test_mismatched_labels_warning(self, sample_curve1: ScatteringCurve1D, sample_curve2_no_error: ScatteringCurve1D):
        """Test warning for mismatched number of labels and curves."""
        curves = [sample_curve1, sample_curve2_no_error]
        labels_too_few = ["Only one label"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax = plot_iq(curves, labels=labels_too_few)
            assert len(w) == 1
            assert "Mismatch between number of curves and labels" in str(w[0].message)
        # Legend should use default labels in this case
        legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
        assert "sample1.dat" in legend_texts # Default from metadata
        assert "sample2.dat" in legend_texts


# --- Placeholder tests for other plotting functions (to be implemented later) ---
@pytest.mark.usefixtures("plt_close_figures")
class TestPlotGuinierPlaceholder:
    @pytest.mark.skip(reason="plot_guinier not yet implemented")
    def test_plot_guinier_runs(self, sample_curve1: ScatteringCurve1D):
        from scatterbrain.visualization import plot_guinier # Import here to avoid error if not implemented
        # result_dummy = {"Rg": 1.0, "I0": 100.0, "q_fit_min": 0.1, "q_fit_max":0.2} # Dummy result
        # fig, ax = plot_guinier(sample_curve1, guinier_result=result_dummy)
        # assert isinstance(fig, Figure)
        # assert isinstance(ax, Axes)
        with pytest.raises(NotImplementedError):
             plot_guinier(sample_curve1)


@pytest.mark.usefixtures("plt_close_figures")
class TestPlotPorodPlaceholder:
    @pytest.mark.skip(reason="plot_porod not yet implemented")
    def test_plot_porod_runs(self, sample_curve1: ScatteringCurve1D):
        from scatterbrain.visualization import plot_porod
        with pytest.raises(NotImplementedError):
            plot_porod(sample_curve1)


@pytest.mark.usefixtures("plt_close_figures")
class TestPlotFitPlaceholder:
    @pytest.mark.skip(reason="plot_fit not yet implemented")
    def test_plot_fit_runs(self, sample_curve1: ScatteringCurve1D):
        from scatterbrain.visualization import plot_fit
        # dummy_fit_result = {
        #     "fit_curve": ScatteringCurve1D(sample_curve1.q, sample_curve1.intensity * 0.9),
        #     "fitted_params": {}, "chi_squared_reduced": 1.0
        # }
        with pytest.raises(NotImplementedError):
            plot_fit(sample_curve1, fit_result_dict={}) # Dummy dict