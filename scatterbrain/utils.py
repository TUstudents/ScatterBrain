# scatterbrain/utils.py
"""
Utility functions and constants for the ScatterBrain library.
"""

import numpy as np
from typing import Union

# Define common q-units for clarity, though they are just strings.
# These could be enhanced with an Enum or a more structured unit system later.
Q_ANGSTROM_INV = "A^-1"  # Angstrom inverse
Q_NANOMETER_INV = "nm^-1" # Nanometer inverse

# Conversion factors: 1 nm = 10 Angstrom
# So, 1 nm^-1 = 0.1 A^-1
# And 1 A^-1 = 10 nm^-1
_CONVERSION_FACTORS_TO_NM_INV = {
    Q_ANGSTROM_INV: 10.0,       # Multiply by 10 to convert A^-1 to nm^-1
    Q_NANOMETER_INV: 1.0,
    # Add more units as needed, e.g., "1/A", "1/nm", "Angstrom^-1", "nm^-1" variations
    "1/A": 10.0,
    "1/nm": 1.0,
    "Angstrom^-1": 10.0,
    "nanometer^-1": 1.0,
}

_CONVERSION_FACTORS_FROM_NM_INV = {
    Q_ANGSTROM_INV: 0.1,       # Multiply by 0.1 to convert nm^-1 to A^-1
    Q_NANOMETER_INV: 1.0,
    "1/A": 0.1,
    "1/nm": 1.0,
    "Angstrom^-1": 0.1,
    "nanometer^-1": 1.0,
}


def normalize_unit_string(unit: str) -> str:
    """Normalize unit string for consistent comparison."""
    unit = unit.strip().lower()
    unit = unit.replace("angstrom", "a")
    unit = unit.replace("nanometer", "nm")
    # Convert all inverse notations to ^-1
    if unit.endswith("-1"):
        unit = unit[:-2] + "^-1"
    elif unit.startswith("1/"):
        unit = unit[2:] + "^-1"
    elif not unit.endswith("^-1"):
        unit += "^-1"
    return unit


def convert_q_array(
    q_values: np.ndarray,
    current_unit: str,
    target_unit: str
) -> np.ndarray:
    """
    Converts an array of q-values from a current unit to a target unit.

    Supported units are primarily "nm^-1" (nanometer inverse) and
    "A^-1" (Angstrom inverse), including common string variations.

    Parameters
    ----------
    q_values : np.ndarray
        The array of q-values to convert.
    current_unit : str
        The current unit of the q_values (e.g., "nm^-1", "A^-1").
    target_unit : str
        The desired target unit for the q_values.

    Returns
    -------
    np.ndarray
        A new array with q-values converted to the target_unit.

    Raises
    ------
    ValueError
        If either current_unit or target_unit is not recognized/supported.
    """
    if current_unit == target_unit:
        return np.copy(q_values)

    # First try direct lookup
    if current_unit in _CONVERSION_FACTORS_TO_NM_INV and target_unit in _CONVERSION_FACTORS_FROM_NM_INV:
        factor_to_nm = _CONVERSION_FACTORS_TO_NM_INV[current_unit]
        factor_from_nm = _CONVERSION_FACTORS_FROM_NM_INV[target_unit]
        return q_values / factor_to_nm / factor_from_nm

    # Try normalized versions
    norm_current = normalize_unit_string(current_unit)
    norm_target = normalize_unit_string(target_unit)

    # Map normalized units to standard forms
    for unit in [current_unit, norm_current]:
        if unit in _CONVERSION_FACTORS_TO_NM_INV:
            current_std = unit
            break
    else:
        raise ValueError(f"Unsupported current_unit '{current_unit}'")

    for unit in [target_unit, norm_target]:
        if unit in _CONVERSION_FACTORS_FROM_NM_INV:
            target_std = unit
            break
    else:
        raise ValueError(f"Unsupported target_unit '{target_unit}'")

    factor_to_nm = _CONVERSION_FACTORS_TO_NM_INV[current_std]
    factor_from_nm = _CONVERSION_FACTORS_FROM_NM_INV[target_std]
    
    return q_values /factor_to_nm / factor_from_nm


class ScatterBrainError(Exception):
    """Base class for custom exceptions in ScatterBrain."""
    pass

class ProcessingError(ScatterBrainError):
    """Exception raised for errors during data processing."""
    pass

class AnalysisError(ScatterBrainError):
    """Exception raised for errors during data analysis."""
    pass

class FittingError(ScatterBrainError):
    """Exception raised for errors during model fitting."""
    pass

# Future: Add physical constants, more complex unit conversions, logging setup, etc.