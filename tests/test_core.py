# tests/test_core.py
"""
Unit tests for the scatterbrain.core module, particularly the ScatteringCurve1D class.
"""

import pytest
import numpy as np
import copy
from scatterbrain.core import ScatteringCurve1D, QUnit, IntensityUnit

# --- Fixtures ---

@pytest.fixture
def valid_q_data() -> np.ndarray:
    """Returns a valid 1D numpy array for q."""
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5])

@pytest.fixture
def valid_i_data(valid_q_data: np.ndarray) -> np.ndarray:
    """Returns a valid 1D numpy array for intensity, matching q_data."""
    return np.array([100.0, 80.0, 50.0, 30.0, 10.0]) * (1 / valid_q_data**2) # Simple I ~ q^-2

@pytest.fixture
def valid_e_data(valid_i_data: np.ndarray) -> np.ndarray:
    """Returns a valid 1D numpy array for error, matching i_data."""
    return np.sqrt(valid_i_data) * 0.1 # 10% of sqrt(I) as error

@pytest.fixture
def valid_metadata() -> dict:
    """Returns a valid metadata dictionary."""
    return {"sample_name": "Test Sample A", "temperature_K": 298.15}

@pytest.fixture
def default_curve(
    valid_q_data: np.ndarray,
    valid_i_data: np.ndarray,
    valid_e_data: np.ndarray,
    valid_metadata: dict
) -> ScatteringCurve1D:
    """Returns a default ScatteringCurve1D instance with all valid data."""
    return ScatteringCurve1D(
        q=valid_q_data,
        intensity=valid_i_data,
        error=valid_e_data,
        metadata=valid_metadata,
        q_unit="nm^-1",
        intensity_unit="cm^-1"
    )

@pytest.fixture
def curve_no_error(
    valid_q_data: np.ndarray,
    valid_i_data: np.ndarray,
    valid_metadata: dict
) -> ScatteringCurve1D:
    """Returns a ScatteringCurve1D instance without error data."""
    return ScatteringCurve1D(
        q=valid_q_data,
        intensity=valid_i_data,
        error=None,
        metadata=valid_metadata,
        q_unit="A^-1"
    )

# --- Test Cases ---

class TestScatteringCurve1DInitialization:
    """Tests for ScatteringCurve1D initialization."""

    def test_valid_initialization(self, default_curve: ScatteringCurve1D, valid_q_data: np.ndarray,
                                  valid_i_data: np.ndarray, valid_e_data: np.ndarray,
                                  valid_metadata: dict):
        """Test successful initialization with all valid parameters."""
        assert np.array_equal(default_curve.q, valid_q_data)
        assert np.array_equal(default_curve.intensity, valid_i_data)
        assert default_curve.error is not None
        assert np.array_equal(default_curve.error, valid_e_data)
        assert default_curve.metadata["sample_name"] == valid_metadata["sample_name"]
        assert default_curve.q_unit == "nm^-1"
        assert default_curve.intensity_unit == "cm^-1"
        assert "ScatteringCurve1D object created." in default_curve.metadata["processing_history"]

    def test_initialization_no_error(self, curve_no_error: ScatteringCurve1D, valid_q_data: np.ndarray,
                                      valid_i_data: np.ndarray):
        """Test successful initialization without error data."""
        assert np.array_equal(curve_no_error.q, valid_q_data)
        assert np.array_equal(curve_no_error.intensity, valid_i_data)
        assert curve_no_error.error is None
        assert curve_no_error.q_unit == "A^-1"
        assert curve_no_error.intensity_unit == "a.u." # Default

    def test_initialization_no_metadata(self, valid_q_data: np.ndarray, valid_i_data: np.ndarray):
        """Test initialization with no metadata provided."""
        curve = ScatteringCurve1D(q=valid_q_data, intensity=valid_i_data)
        assert isinstance(curve.metadata, dict)
        assert "processing_history" in curve.metadata
        assert "ScatteringCurve1D object created." in curve.metadata["processing_history"]

    def test_metadata_deepcopied(self, valid_q_data, valid_i_data, valid_metadata):
        """Test that input metadata is deepcopied."""
        original_meta = copy.deepcopy(valid_metadata)
        curve = ScatteringCurve1D(q=valid_q_data, intensity=valid_i_data, metadata=original_meta)
        original_meta["new_key"] = "changed_value" # Modify original
        assert "new_key" not in curve.metadata # Curve's metadata should be independent

    # --- Invalid Input Tests ---
    @pytest.mark.parametrize("q_input", [None, [0.1, 0.2], "not_an_array"])
    def test_invalid_q_type(self, q_input, valid_i_data):
        with pytest.raises(TypeError, match="Input 'q' must be a NumPy ndarray"):
            ScatteringCurve1D(q=q_input, intensity=valid_i_data)

    @pytest.mark.parametrize("i_input", [None, [10.0, 20.0], "not_an_array"])
    def test_invalid_i_type(self, i_input, valid_q_data):
        with pytest.raises(TypeError, match="Input 'intensity' must be a NumPy ndarray"):
            ScatteringCurve1D(q=valid_q_data, intensity=i_input)

    @pytest.mark.parametrize("e_input", [[1.0, 2.0], "not_an_array"])
    def test_invalid_e_type(self, e_input, valid_q_data, valid_i_data):
        with pytest.raises(TypeError, match="Input 'error' must be a NumPy ndarray if provided"):
            ScatteringCurve1D(q=valid_q_data, intensity=valid_i_data, error=e_input)

    def test_q_not_1d(self, valid_q_data, valid_i_data):
        q_2d = valid_q_data.reshape(-1, 1)
        with pytest.raises(ValueError, match="Input 'q' must be a 1D array"):
            ScatteringCurve1D(q=q_2d, intensity=valid_i_data)

    def test_i_not_1d(self, valid_q_data, valid_i_data):
        i_2d = valid_i_data.reshape(-1, 1)
        with pytest.raises(ValueError, match="Input 'intensity' must be a 1D array"):
            ScatteringCurve1D(q=valid_q_data, intensity=i_2d)

    def test_e_not_1d(self, valid_q_data, valid_i_data, valid_e_data):
        e_2d = valid_e_data.reshape(-1, 1)
        with pytest.raises(ValueError, match="Input 'error' must be a 1D array if provided"):
            ScatteringCurve1D(q=valid_q_data, intensity=valid_i_data, error=e_2d)

    def test_shape_mismatch_q_i(self, valid_q_data, valid_i_data):
        with pytest.raises(ValueError, match="Shapes of 'q' .* and 'intensity' .* must match"):
            ScatteringCurve1D(q=valid_q_data, intensity=valid_i_data[:-1])

    def test_shape_mismatch_q_e(self, valid_q_data, valid_i_data, valid_e_data):
        with pytest.raises(ValueError, match="Shape of 'error' .* must match 'q' .* if provided"):
            ScatteringCurve1D(q=valid_q_data, intensity=valid_i_data, error=valid_e_data[:-1])

    # Optional: Test for q <= 0 if strict positivity is enforced later
    # def test_non_positive_q(self, valid_i_data):
    #     q_non_positive = np.array([-0.1, 0.0, 0.1, 0.2])
    #     with pytest.raises(ValueError, match="All q-values must be positive"):
    #         ScatteringCurve1D(q=q_non_positive, intensity=valid_i_data[:4])


class TestScatteringCurve1DMethods:
    """Tests for methods of ScatteringCurve1D."""

    def test_repr_method(self, default_curve: ScatteringCurve1D):
        rep_str = repr(default_curve)
        assert "<ScatteringCurve1D:" in rep_str
        assert f"{len(default_curve.q)} points" in rep_str
        assert f"q_range=({default_curve.q.min():.3g} - {default_curve.q.max():.3g} {default_curve.q_unit})" in rep_str
        assert "errors (shape" in rep_str

    def test_repr_method_no_error(self, curve_no_error: ScatteringCurve1D):
        rep_str = repr(curve_no_error)
        assert "no errors" in rep_str

    def test_str_method(self, default_curve: ScatteringCurve1D):
        str_out = str(default_curve)
        assert "ScatteringCurve1D Object Summary:" in str_out
        assert f"Number of data points: {len(default_curve.q)}" in str_out
        assert f"q range              : {default_curve.q.min():.4g} to {default_curve.q.max():.4g} [{default_curve.q_unit}]" in str_out
        assert "Errors available     : Yes" in str_out
        assert "sample_name" in default_curve.metadata
        default_curve.metadata["filename"] = "test.dat" # Add filename for testing this part of str
        str_out_with_filename = str(default_curve)
        assert "Source Filename      : test.dat" in str_out_with_filename


    def test_len_method(self, default_curve: ScatteringCurve1D, valid_q_data: np.ndarray):
        assert len(default_curve) == len(valid_q_data)

    def test_copy_method(self, default_curve: ScatteringCurve1D):
        copied_curve = default_curve.copy()
        assert copied_curve is not default_curve
        assert np.array_equal(copied_curve.q, default_curve.q)
        assert copied_curve.q is not default_curve.q # Ensure arrays are copied
        assert np.array_equal(copied_curve.intensity, default_curve.intensity)
        assert copied_curve.intensity is not default_curve.intensity
        if default_curve.error is not None:
            assert copied_curve.error is not None
            assert np.array_equal(copied_curve.error, default_curve.error)
            assert copied_curve.error is not default_curve.error
        assert copied_curve.metadata == default_curve.metadata
        assert copied_curve.metadata is not default_curve.metadata # Ensure dict is copied
        assert "Object deep copied." in copied_curve.metadata["processing_history"]

        # Modify original and check copy is unaffected
        default_curve.q[0] = 999.0
        default_curve.metadata["new_field"] = "original_value"
        assert copied_curve.q[0] != 999.0
        assert "new_field" not in copied_curve.metadata


    @pytest.mark.parametrize("key, expected_len", [
        (slice(1, 3), 2),               # Basic slice
        (np.array([True, False, True, False, True]), 3), # Boolean mask
        (np.array([0, 2, 4]), 3),       # Integer array indexing
        (0, 1),                         # Single integer index
        (-1, 1)                         # Negative integer index
    ])
    def test_getitem_method(self, default_curve: ScatteringCurve1D, key, expected_len):
        sliced_curve = default_curve[key]
        assert isinstance(sliced_curve, ScatteringCurve1D)
        assert len(sliced_curve) == expected_len
        assert np.array_equal(sliced_curve.q, default_curve.q[key])
        assert np.array_equal(sliced_curve.intensity, default_curve.intensity[key])
        if default_curve.error is not None:
            assert sliced_curve.error is not None
            assert np.array_equal(sliced_curve.error, default_curve.error[key])
        assert sliced_curve.metadata is not default_curve.metadata # New metadata dict
        assert default_curve.metadata["sample_name"] == sliced_curve.metadata["sample_name"]
        assert f"Sliced/indexed from original (key: {slice(key, key + 1) if isinstance(key, int) else key})." in sliced_curve.metadata["processing_history"]
        assert sliced_curve.q.ndim == 1 # Ensure output q is always 1D

    def test_getitem_out_of_bounds(self, default_curve: ScatteringCurve1D):
        with pytest.raises(IndexError):
            _ = default_curve[len(default_curve) + 10]
        with pytest.raises(IndexError): # Example with slice, depends on NumPy's behavior
            _ = default_curve[slice(len(default_curve) + 10, len(default_curve) + 12)]


    def test_to_dict_method(self, default_curve: ScatteringCurve1D):
        curve_dict = default_curve.to_dict()
        assert isinstance(curve_dict, dict)
        assert np.array_equal(np.array(curve_dict["q"]), default_curve.q)
        assert isinstance(curve_dict["q"], list) # Check for list conversion
        assert np.array_equal(np.array(curve_dict["intensity"]), default_curve.intensity)
        assert isinstance(curve_dict["intensity"], list)
        if default_curve.error is not None:
            assert curve_dict["error"] is not None
            assert np.array_equal(np.array(curve_dict["error"]), default_curve.error)
            assert isinstance(curve_dict["error"], list)
        assert curve_dict["q_unit"] == default_curve.q_unit
        assert curve_dict["intensity_unit"] == default_curve.intensity_unit
        assert curve_dict["metadata"] == default_curve.metadata # Should be a deepcopy

    def test_to_dict_no_metadata(self, default_curve: ScatteringCurve1D):
        curve_dict = default_curve.to_dict(include_metadata=False)
        assert "metadata" not in curve_dict

    def test_from_dict_method(self, default_curve: ScatteringCurve1D):
        curve_dict = default_curve.to_dict()
        reconstructed_curve = ScatteringCurve1D.from_dict(curve_dict)
        assert isinstance(reconstructed_curve, ScatteringCurve1D)
        assert np.array_equal(reconstructed_curve.q, default_curve.q)
        assert np.array_equal(reconstructed_curve.intensity, default_curve.intensity)
        if default_curve.error is not None:
            assert reconstructed_curve.error is not None
            assert np.array_equal(reconstructed_curve.error, default_curve.error)
        assert reconstructed_curve.q_unit == default_curve.q_unit
        assert reconstructed_curve.intensity_unit == default_curve.intensity_unit
        assert reconstructed_curve.metadata == default_curve.metadata
        # Check if processing history includes "ScatteringCurve1D object created."
        assert "ScatteringCurve1D object created." in reconstructed_curve.metadata["processing_history"]


    def test_from_dict_minimal(self, valid_q_data, valid_i_data):
        minimal_dict = {"q": valid_q_data.tolist(), "intensity": valid_i_data.tolist()}
        curve = ScatteringCurve1D.from_dict(minimal_dict)
        assert np.array_equal(curve.q, valid_q_data)
        assert np.array_equal(curve.intensity, valid_i_data)
        assert curve.error is None
        assert curve.q_unit == "nm^-1" # Default
        assert curve.intensity_unit == "a.u." # Default
        assert isinstance(curve.metadata, dict)

    def test_get_q_range(self, default_curve: ScatteringCurve1D):
        q_min, q_max = default_curve.get_q_range()
        assert q_min == default_curve.q.min()
        assert q_max == default_curve.q.max()

    def test_get_intensity_range(self, default_curve: ScatteringCurve1D):
        i_min, i_max = default_curve.get_intensity_range()
        assert i_min == default_curve.intensity.min()
        assert i_max == default_curve.intensity.max()

    def test_update_metadata_no_overwrite(self, default_curve: ScatteringCurve1D):
        original_sample_name = default_curve.metadata["sample_name"]
        new_data = {"new_key": "new_value", "sample_name": "SHOULD_NOT_CHANGE"}
        default_curve.update_metadata(new_data, overwrite=False)
        assert default_curve.metadata["new_key"] == "new_value"
        assert default_curve.metadata["sample_name"] == original_sample_name
        assert "Metadata updated." in default_curve.metadata["processing_history"]

    def test_update_metadata_with_overwrite(self, default_curve: ScatteringCurve1D):
        new_data = {"new_key_overwrite": "new_value_2", "sample_name": "CHANGED_SAMPLE_NAME"}
        default_curve.update_metadata(new_data, overwrite=True)
        assert default_curve.metadata["new_key_overwrite"] == "new_value_2"
        assert default_curve.metadata["sample_name"] == "CHANGED_SAMPLE_NAME"
        assert "Metadata updated." in default_curve.metadata["processing_history"]

    def test_convert_q_unit_placeholder(self, default_curve: ScatteringCurve1D):
        """Test that the q-unit conversion method is a placeholder."""
        with pytest.raises(NotImplementedError):
            default_curve.convert_q_unit("A^-1")