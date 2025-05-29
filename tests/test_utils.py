# tests/test_utils.py
"""
Unit tests for the scatterbrain.utils module.
"""
import pytest
import numpy as np
import re

from scatterbrain.utils import (
    convert_q_array,
    Q_ANGSTROM_INV,
    Q_NANOMETER_INV,
    ScatterBrainError, ProcessingError, AnalysisError, FittingError
)

class TestConvertQArray:
    """Tests for the convert_q_array utility function."""

    @pytest.fixture
    def q_values_nm_inv(self) -> np.ndarray:
        """Test values in nm^-1."""
        return np.array([0.1, 1.0, 10.0])

    @pytest.fixture
    def q_values_a_inv(self) -> np.ndarray:
        """Test values in A^-1."""
        return np.array([1.0, 10.0, 100.0]) # Corresponding values in A^-1 (0.1 nm^-1 = 1 A^-1)

    def test_nm_inv_to_a_inv(self, q_values_nm_inv: np.ndarray, q_values_a_inv: np.ndarray):
        """Test conversion from nm^-1 to A^-1."""
        converted_q = convert_q_array(q_values_nm_inv, Q_NANOMETER_INV, Q_ANGSTROM_INV)
        assert np.allclose(converted_q, q_values_a_inv)

    def test_a_inv_to_nm_inv(self, q_values_a_inv: np.ndarray, q_values_nm_inv: np.ndarray):
        """Test conversion from A^-1 to nm^-1."""
        converted_q = convert_q_array(q_values_a_inv, Q_ANGSTROM_INV, Q_NANOMETER_INV)
        assert np.allclose(converted_q, q_values_nm_inv)

    def test_same_unit_conversion(self, q_values_nm_inv: np.ndarray):
        """Test conversion when current and target units are the same."""
        converted_q_nm = convert_q_array(q_values_nm_inv, Q_NANOMETER_INV, Q_NANOMETER_INV)
        assert np.allclose(converted_q_nm, q_values_nm_inv)
        assert converted_q_nm is not q_values_nm_inv # Should return a copy

        q_values_a = q_values_nm_inv * 10.0 # Arbitrary values in A^-1
        converted_q_a = convert_q_array(q_values_a, Q_ANGSTROM_INV, Q_ANGSTROM_INV)
        assert np.allclose(converted_q_a, q_values_a)
        assert converted_q_a is not q_values_a


    @pytest.mark.parametrize("current_unit, target_unit", [
        ("nm^-1", "A^-1"),
        ("A^-1", "nm^-1"),
        ("1/nm", "1/A"),
        ("1/A", "1/nm"),
        ("nanometer^-1", "Angstrom^-1"),
        ("Angstrom^-1", "nanometer^-1"),
    ])
    def test_string_variations(self, q_values_nm_inv: np.ndarray, current_unit: str, target_unit: str):
        """Test various string representations of the units."""
        # Assume q_values_nm_inv are the base for this test.
        # If current_unit is A-like, convert input q to A-like.
        if "nanometer" in current_unit.lower() or "nm" in current_unit.lower():
            q_input = q_values_nm_inv # Already in nm^-1
        else:
            q_input = q_values_nm_inv * 10.0 # Now in A^-1 equivalent

        converted_q = convert_q_array(q_input, current_unit, target_unit)

        # Check if target is A-like or nm-like to compare against correct fixture
        if "nanometer" in target_unit.lower() or "nm" in target_unit.lower():
            expected_q = q_values_nm_inv 
        else:
            expected_q = q_values_nm_inv  * 10.0 # Expected in A^-1
        
        assert np.allclose(converted_q, expected_q), \
            f"Failed for {current_unit} to {target_unit}. Got {converted_q}, expected {expected_q}"


    def test_unsupported_current_unit(self, q_values_nm_inv: np.ndarray):
        """Test with an unsupported current_unit."""
        with pytest.raises(ValueError, match=re.escape("Unsupported current_unit 'm^-1'")):
            convert_q_array(q_values_nm_inv, "m^-1", Q_ANGSTROM_INV)

    def test_unsupported_target_unit(self, q_values_nm_inv: np.ndarray):
        """Test with an unsupported target_unit."""
        with pytest.raises(ValueError, match=re.escape("Unsupported target_unit 'km^-1'")):
            convert_q_array(q_values_nm_inv, Q_NANOMETER_INV, "km^-1")

    def test_empty_q_array(self):
        """Test with an empty q_values array."""
        empty_q = np.array([])
        converted_q = convert_q_array(empty_q, Q_NANOMETER_INV, Q_ANGSTROM_INV)
        assert converted_q.shape == (0,)


class TestCustomExceptions:
    """Tests for the custom exception classes."""

    def test_scatterbrain_error_is_exception(self):
        assert issubclass(ScatterBrainError, Exception)

    @pytest.mark.parametrize("CustomError", [
        ProcessingError, AnalysisError, FittingError
    ])
    def test_specific_errors_inherit_from_scatterbrain_error(self, CustomError):
        assert issubclass(CustomError, ScatterBrainError)

    def test_can_raise_and_catch_custom_errors(self):
        with pytest.raises(ProcessingError, match="Test processing issue"):
            raise ProcessingError("Test processing issue")
        
        with pytest.raises(AnalysisError, match="Test analysis issue"):
            raise AnalysisError("Test analysis issue")

        with pytest.raises(FittingError, match="Test fitting issue"):
            raise FittingError("Test fitting issue")