# scatterbrain/core.py
"""
Core data structures for the ScatterBrain library.
"""

from typing import Optional, Dict, Any, Union, Tuple
import numpy as np
import copy

# Define common unit types for type hinting clarity, though these are just strings
QUnit = str  # e.g., "nm^-1", "A^-1"
IntensityUnit = str # e.g., "cm^-1", "a.u."


class ScatteringCurve1D:
    """
    Represents a 1D scattering intensity curve (I(q) vs q).

    This class is the central data object for most 1D SAXS/WAXS operations
    within the ScatterBrain library. It stores scattering vector (q),
    intensity (I), and optionally errors on intensity (err), along with
    metadata related to the experiment and processing.

    Attributes
    ----------
    q : np.ndarray
        The scattering vector values. Typically 1D.
    intensity : np.ndarray
        The scattering intensity values corresponding to `q`. Typically 1D.
        Must be the same shape as `q`.
    error : Optional[np.ndarray]
        The error (uncertainty) in the intensity values. Typically 1D.
        If provided, must be the same shape as `q` and `intensity`.
    metadata : Dict[str, Any]
        A dictionary to store metadata associated with the scattering curve.
        Examples: sample name, experimental conditions (wavelength, distance),
        processing history, user notes.
    q_unit : QUnit
        The unit of the scattering vector `q`. Default is "nm^-1".
    intensity_unit : IntensityUnit
        The unit of the intensity. Default is "a.u." (arbitrary units).

    Raises
    ------
    ValueError
        If `q` and `intensity` are not 1D arrays or have different shapes.
        If `error` is provided and has a different shape than `q`.
        If `q` contains non-positive values (can be relaxed later if needed).
    TypeError
        If `q` or `intensity` are not NumPy arrays.
    """

    def __init__(
        self,
        q: np.ndarray,
        intensity: np.ndarray,
        error: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        q_unit: QUnit = "nm^-1",
        intensity_unit: IntensityUnit = "a.u.",
    ):
        """
        Initializes the ScatteringCurve1D object.

        Parameters
        ----------
        q : np.ndarray
            Scattering vector values.
        intensity : np.ndarray
            Scattering intensity values.
        error : Optional[np.ndarray], optional
            Error in intensity values, by default None.
        metadata : Optional[Dict[str, Any]], optional
            Metadata dictionary, by default None which initializes an empty dict.
        q_unit : QUnit, optional
            Unit of q values, by default "nm^-1".
        intensity_unit : IntensityUnit, optional
            Unit of intensity values, by default "a.u.".
        """
        # Set units as attributes
        self.q_unit = q_unit
        self.intensity_unit = intensity_unit

        if not isinstance(q, np.ndarray):
            raise TypeError("Input 'q' must be a NumPy ndarray.")
        if not isinstance(intensity, np.ndarray):
            raise TypeError("Input 'intensity' must be a NumPy ndarray.")

        if q.ndim != 1:
            raise ValueError(f"Input 'q' must be a 1D array, got ndim={q.ndim}.")
        if intensity.ndim != 1:
            raise ValueError(
                f"Input 'intensity' must be a 1D array, got ndim={intensity.ndim}."
            )

        if q.shape != intensity.shape:
            raise ValueError(
                f"Shapes of 'q' {q.shape} and 'intensity' {intensity.shape} must match."
            )

        # Basic check for q-values; can be expanded (e.g., strictly positive)
        # if np.any(q <= 0):
        #     raise ValueError("All q-values must be positive.")
        # Decided to relax this for now, as some data might have q=0 or negative q for specific representations.
        # The user or specific analysis functions should handle q-range validity.

        self.q: np.ndarray = q
        self.intensity: np.ndarray = intensity

        if error is not None:
            if not isinstance(error, np.ndarray):
                raise TypeError("Input 'error' must be a NumPy ndarray if provided.")
            if error.ndim != 1:
                raise ValueError(
                    f"Input 'error' must be a 1D array if provided, got ndim={error.ndim}."
                )
            if error.shape != q.shape:
                raise ValueError(
                    f"Shape of 'error' {error.shape} must match 'q' {q.shape} if provided."
                )
            self.error: Optional[np.ndarray] = error
        else:
            self.error: Optional[np.ndarray] = None

        if metadata is not None:
            self.metadata = copy.deepcopy(metadata)
        else:
            self.metadata = {}
        if "processing_history" not in self.metadata:
            self.metadata["processing_history"] = []
            self.metadata["processing_history"].append("ScatteringCurve1D object created.")
        else:
            # Only append if not already present as last entry
            if not self.metadata["processing_history"] or self.metadata["processing_history"][-1] != "ScatteringCurve1D object created.":
                self.metadata["processing_history"].append("ScatteringCurve1D object created.")

    def __repr__(self) -> str:
        """Return a detailed string representation of the object."""
        error_info = f"errors (shape {self.error.shape})" if self.error is not None else "no errors"
        return (
            f"<ScatteringCurve1D: {len(self.q)} points, "
            f"q_range=({self.q.min():.3g} - {self.q.max():.3g} {self.q_unit}), "
            f"I_range=({self.intensity.min():.3e} - {self.intensity.max():.3e} {self.intensity_unit}), "
            f"{error_info}>"
        )

    def __str__(self) -> str:
        """Return a user-friendly string summary of the object."""
        has_error = "Yes" if self.error is not None else "No"
        summary = [
            f"ScatteringCurve1D Object Summary:",
            f"  Number of data points: {len(self.q)}",
            f"  q range              : {self.q.min():.4g} to {self.q.max():.4g} [{self.q_unit}]",
            f"  Intensity range      : {self.intensity.min():.4e} to {self.intensity.max():.4e} [{self.intensity_unit}]",
            f"  Errors available     : {has_error}",
            f"  Metadata keys        : {list(self.metadata.keys()) if self.metadata else 'None'}",
        ]
        if "filename" in self.metadata:
             summary.insert(1, f"  Source Filename      : {self.metadata['filename']}")
        return "\n".join(summary)

    def __len__(self) -> int:
        """Return the number of data points in the curve."""
        return len(self.q)

    def __getitem__(self, key):
        """Allows slicing/indexing, returns a new ScatteringCurve1D object.
        
        Parameters
        ----------
        key : Union[int, slice, np.ndarray, List]
            Index, slice, or boolean/integer array for selecting data points.
            
        Returns
        -------
        ScatteringCurve1D
            New curve object containing the selected data points.
            
        Raises
        ----
        IndexError
            If indices are out of bounds.
        TypeError
            If key is of unsupported type or contains floats.
        """
        # Convert lists to numpy arrays for consistent handling
        if isinstance(key, list):
            key = np.array(key)
            
        # Handle different types of keys
        if isinstance(key, (int, slice)):
            # Existing integer and slice handling
            if isinstance(key, int):
                if key < -len(self.q) or key >= len(self.q):
                    raise IndexError("Index out of range")
            else:  # slice
                indices = range(*key.indices(len(self.q)))
                if len(indices) == 0:
                    raise IndexError("Empty slice")
        elif isinstance(key, np.ndarray):
            if key.dtype == bool:
                # Boolean indexing
                if key.shape != self.q.shape:
                    raise IndexError(f"Boolean index did not match array shape. Got {key.shape} != {self.q.shape}")
            elif np.issubdtype(key.dtype, np.integer):
                # Integer array indexing
                if np.any((key >= len(self.q)) | (key < -len(self.q))):
                    raise IndexError("Integer indices out of bounds")
            elif np.issubdtype(key.dtype, np.floating):
                raise TypeError("Float indices not supported. Use interpolation methods instead.")
            else:
                raise TypeError(f"Unsupported index array type: {key.dtype}")
        else:
            raise TypeError(f"Invalid index type: {type(key)}")

        # Create new curve with indexed data
        new_q = np.atleast_1d(self.q[key])
        new_intensity = np.atleast_1d(self.intensity[key])
        new_error = np.atleast_1d(self.error[key]) if self.error is not None else None
        new_metadata = copy.deepcopy(self.metadata)
        new_metadata["processing_history"].append(f"Indexed with {type(key).__name__} indexer")
        
        return ScatteringCurve1D(
            q=new_q,
            intensity=new_intensity,
            error=new_error,
            metadata=new_metadata,
            q_unit=self.q_unit,
            intensity_unit=self.intensity_unit
        )

    def copy(self) -> 'ScatteringCurve1D':
        """
        Creates a deep copy of the ScatteringCurve1D object.

        Returns
        -------
        ScatteringCurve1D
            A new, independent ScatteringCurve1D object.
        """
        new_metadata = copy.deepcopy(self.metadata)
        if "processing_history" in new_metadata:
            new_metadata["processing_history"].append("Object deep copied.")
        else:
            new_metadata["processing_history"] = ["Object deep copied."]
        # Avoid duplicating "ScatteringCurve1D object created." in processing_history
        return self.__class__.__new__(self.__class__).__init_copy__(
            self.q.copy(),
            self.intensity.copy(),
            self.error.copy() if self.error is not None else None,
            new_metadata,
            self.q_unit,
            self.intensity_unit,
        )

    def __init_copy__(self, q, intensity, error, metadata, q_unit, intensity_unit):
        # Internal method to bypass __init__'s "object created" history entry
        self.q = q
        self.intensity = intensity
        self.error = error
        self.metadata = metadata
        self.q_unit = q_unit
        self.intensity_unit = intensity_unit
        return self

    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Serializes the ScatteringCurve1D object to a dictionary.

        Parameters
        ----------
        include_metadata : bool, optional
            Whether to include the metadata dictionary in the output, by default True.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the object.
            NumPy arrays are converted to lists for easier JSON serialization.
        """
        data_dict = {
            "q": self.q.tolist(),
            "intensity": self.intensity.tolist(),
            "error": self.error.tolist() if self.error is not None else None,
            "q_unit": self.q_unit,
            "intensity_unit": self.intensity_unit,
        }
        if include_metadata:
            # Attempt a deepcopy of metadata for safety, but handle potential uncopyable items
            try:
                data_dict["metadata"] = copy.deepcopy(self.metadata)
            except TypeError: # pragma: no cover
                # If deepcopy fails (e.g., complex objects in metadata), do a shallow copy
                # and warn the user or log. For basic JSON-like metadata, this is rare.
                print("Warning: Metadata could not be deep-copied during to_dict; using shallow copy.")
                data_dict["metadata"] = self.metadata.copy()
        return data_dict

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'ScatteringCurve1D':
        """
        Creates a ScatteringCurve1D object from a dictionary representation.

        Parameters
        ----------
        data_dict : Dict[str, Any]
            A dictionary, typically created by `to_dict()`.
            Must contain 'q', 'intensity' (as lists or np.ndarray).
            Optionally 'error', 'q_unit', 'intensity_unit', 'metadata'.

        Returns
        -------
        ScatteringCurve1D
            A new ScatteringCurve1D object.
        """
        q_data = np.array(data_dict["q"])
        intensity_data = np.array(data_dict["intensity"])
        error_data = np.array(data_dict.get("error")) if data_dict.get("error") is not None else None

        return cls(
            q=q_data,
            intensity=intensity_data,
            error=error_data,
            metadata=data_dict.get("metadata"), # Will be deepcopied in __init__
            q_unit=data_dict.get("q_unit", "nm^-1"),
            intensity_unit=data_dict.get("intensity_unit", "a.u."),
        )

    def get_q_range(self) -> Tuple[float, float]:
        """Returns the minimum and maximum q values."""
        return self.q.min(), self.q.max()

    def get_intensity_range(self) -> Tuple[float, float]:
        """Returns the minimum and maximum intensity values."""
        return self.intensity.min(), self.intensity.max()

    def update_metadata(self, new_metadata: Dict[str, Any], overwrite: bool = False):
        """
        Updates the metadata dictionary.

        Parameters
        ----------
        new_metadata : Dict[str, Any]
            Dictionary containing new metadata to add or update.
        overwrite : bool, optional
            If True, existing keys in `self.metadata` will be overwritten
            by values from `new_metadata`. If False (default), existing keys
            will not be changed, and only new keys will be added.
            For nested dicts, this applies at the top level of `new_metadata`.
        """
        if overwrite:
            self.metadata.update(new_metadata)
        else:
            for key, value in new_metadata.items():
                if key not in self.metadata:
                    self.metadata[key] = value
        self.metadata.setdefault("processing_history", []).append("Metadata updated.")

    # Placeholder for q-unit conversion - actual implementation in utils or processing
    def convert_q_unit(self, new_unit: QUnit):
        """
        Converts the q-values and q_unit to a new unit.
        (This is a placeholder; actual conversion logic will be in utils/processing
         and called from here or directly by processing functions.)

        Parameters
        ----------
        new_unit : QUnit
            The target unit for q (e.g., "A^-1").

        Raises
        ------
        NotImplementedError
            This method is a placeholder.
        """
        # Example of how it might work with an external function:
        # from .utils import convert_q_array # Or wherever this function lives
        # if new_unit != self.q_unit:
        #     self.q = convert_q_array(self.q, current_unit=self.q_unit, target_unit=new_unit)
        #     self.q_unit = new_unit
        #     self.metadata.setdefault("processing_history", []).append(
        #         f"q units converted from {self.q_unit_old} to {new_unit}."
        #     )
        raise NotImplementedError(
            "q-unit conversion logic to be implemented, likely in scatterbrain.utils "
            "or scatterbrain.processing."
        )

# Example Usage (for testing during development, would be removed or moved to examples/tests)
if __name__ == "__main__": # pragma: no cover
    q_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    i_vals = np.array([100.0, 80.0, 50.0, 30.0, 10.0])
    e_vals = np.array([10.0, 8.0, 5.0, 3.0, 1.0])
    meta = {"sample_name": "Test Sphere", "temperature_C": 25}

    # Create a curve
    curve1 = ScatteringCurve1D(q_vals, i_vals, e_vals, metadata=meta, q_unit="nm^-1")
    print("--- Curve 1 ---")
    print(repr(curve1))
    print(str(curve1))
    print(f"Length: {len(curve1)}")

    # Test copying
    curve2 = curve1.copy()
    curve2.intensity *= 0.5 # Modify copy
    curve2.update_metadata({"copied_from": "curve1"}, overwrite=True)
    print("\n--- Curve 2 (copied and modified) ---")
    print(str(curve2))
    print(f"Curve 1 intensity (should be unchanged): {curve1.intensity[0]}")

    # Test slicing
    curve_slice = curve1[1:3]
    print("\n--- Curve Slice (curve1[1:3]) ---")
    print(str(curve_slice))
    print(f"Slice q: {curve_slice.q}")
    print(f"Slice metadata: {curve_slice.metadata}")

    # Test single item indexing
    curve_item = curve1[0]
    print("\n--- Curve Item (curve1[0]) ---")
    print(str(curve_item))
    assert curve_item.q.ndim == 1 and len(curve_item.q) == 1

    # Test to_dict and from_dict
    curve_dict = curve1.to_dict()
    print("\n--- Curve 1 as Dictionary ---")
    # print(curve_dict) # Can be verbose
    curve_from_dict = ScatteringCurve1D.from_dict(curve_dict)
    print("\n--- Curve reconstructed from Dictionary ---")
    print(str(curve_from_dict))
    assert np.array_equal(curve1.q, curve_from_dict.q)
    assert np.array_equal(curve1.intensity, curve_from_dict.intensity)
    assert np.array_equal(curve1.error, curve_from_dict.error if curve1.error is not None else np.array([]))


    # Test initialization without error
    curve_no_err = ScatteringCurve1D(q_vals, i_vals, q_unit="1/A", intensity_unit="counts")
    print("\n--- Curve without error ---")
    print(str(curve_no_err))

    # Test error conditions
    try:
        ScatteringCurve1D(q_vals, i_vals[:4])
    except ValueError as e:
        print(f"\nCaught expected error: {e}")

    try:
        ScatteringCurve1D(q_vals, i_vals, e_vals[:4])
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        ScatteringCurve1D(q_vals.reshape(5,1), i_vals)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Test metadata update
    curve1.update_metadata({"new_key": "new_value", "sample_name": "SHOULD_NOT_OVERWRITE"})
    curve1.update_metadata({"another_key": "another_value", "sample_name": "OVERWRITTEN"}, overwrite=True)
    print("\n--- Curve 1 after metadata updates ---")
    print(f"Metadata: {curve1.metadata}")
    assert curve1.metadata["sample_name"] == "OVERWRITTEN"
    assert "new_key" in curve1.metadata
    assert "another_key" in curve1.metadata