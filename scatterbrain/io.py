# scatterbrain/io.py
"""
Input/Output operations for the ScatterBrain library.

This module provides functions for loading scattering data from various file formats
and (eventually) saving processed data or analysis results.
"""

import pathlib
import warnings
from typing import Optional, Dict, Any, Union, List, Tuple

import numpy as np
import pandas as pd # Using pandas for robust CSV/text file parsing

from .core import ScatteringCurve1D, QUnit, IntensityUnit


def load_ascii_1d(
    filepath: Union[str, pathlib.Path],
    q_col: Union[int, str] = 0,
    i_col: Union[int, str] = 1,
    err_col: Optional[Union[int, str]] = None,
    skip_header: int = 0,
    delimiter: Optional[str] = None,
    comments: Optional[str] = "#",
    use_names: Optional[List[str]] = None,
    metadata_func: Optional[callable] = None,
    encoding: str = 'utf-8',
    **kwargs: Any
) -> ScatteringCurve1D:
    """
    Loads 1D scattering data from an ASCII text file (e.g., .dat, .csv, .txt).

    This function uses pandas.read_csv for robust parsing of text files.
    It expects columns for q (scattering vector), I (intensity), and
    optionally err (error on intensity).

    Parameters
    ----------
    filepath : Union[str, pathlib.Path]
        Path to the data file.
    q_col : Union[int, str], optional
        Column index (integer) or name (string) for the q-values. Default is 0.
    i_col : Union[int, str], optional
        Column index (integer) or name (string) for the intensity values. Default is 1.
    err_col : Optional[Union[int, str]], optional
        Column index (integer) or name (string) for the error values.
        If None (default), errors are not loaded.
    skip_header : int, optional
        Number of lines to skip at the beginning of the file (header). Default is 0.
    delimiter : Optional[str], optional
        Delimiter to use. If None (default), pandas will try to infer it
        (common delimiters like comma, whitespace, tab).
        Examples: ',', '\\s+', '\\t'.
    comments : Optional[str], optional
        Character indicating the start of a comment line. Default is "#".
        Set to None to disable comment parsing.
    use_names : Optional[List[str]], optional
        A list of column names to use. If the file has no header and this is
        provided, these names will be assigned to the columns.
        If `q_col`, `i_col`, `err_col` are strings, they must be present in `use_names`
        or in the inferred header.
    metadata_func : Optional[callable], optional
        A function that takes the filepath (pathlib.Path object) and the raw
        header lines (List[str]) as input and returns a dictionary of metadata.
        This allows for custom metadata extraction from headers.
    encoding : str, optional
        Encoding to use for opening the file. Default is 'utf-8'.
    **kwargs : Any
        Additional keyword arguments to pass directly to `pandas.read_csv`.

    Returns
    -------
    ScatteringCurve1D
        An object containing the loaded q, I, error, and extracted metadata.

    Raises
    ------
    FileNotFoundError
        If the specified filepath does not exist.
    ValueError
        If specified columns (q_col, i_col, err_col) are not found,
        or if data in these columns cannot be converted to numeric types.
    TypeError
        If column identifiers are not int or str.

    Examples
    --------
    >>> # Assuming 'data.dat' has q in col 0, I in col 1, error in col 2, space-delimited
    >>> # curve = load_ascii_1d("data.dat", err_col=2, delimiter='\\s+')

    >>> # Assuming 'data.csv' has headers 'Q', 'Intensity', 'Error'
    >>> # curve = load_ascii_1d("data.csv", q_col='Q', i_col='Intensity', err_col='Error', skip_header=0)
    """
    file_path_obj = pathlib.Path(filepath)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Validate column types
    for col_name, col_val in [("q_col", q_col), ("i_col", i_col), ("err_col", err_col)]:
        if col_val is not None and not isinstance(col_val, (int, str)):
            raise TypeError(f"Argument '{col_name}' must be an int, str, or None. Got {type(col_val)}")

    # Read header lines if metadata_func is provided or for general info
    header_lines: List[str] = []
    if skip_header > 0 or metadata_func:
        try:
            with open(file_path_obj, 'r', encoding=encoding) as f:
                for i in range(skip_header if skip_header > 0 else 20): # Read up to 20 lines for metadata
                    try:
                        line = f.readline()
                        if not line:
                            break
                        header_lines.append(line.strip())
                    except UnicodeDecodeError as ude: # pragma: no cover
                        warnings.warn(f"Unicode decode error reading header line {i+1} from {filepath} with encoding {encoding}: {ude}. Skipping line for metadata.", UserWarning)
                        header_lines.append(f"UnicodeDecodeError on line {i+1}") # Placeholder
        except Exception as e: # pragma: no cover
            warnings.warn(f"Could not read header lines from {filepath} for metadata: {e}", UserWarning)


    # Prepare pandas read_csv arguments
    read_csv_kwargs = {
        "filepath_or_buffer": file_path_obj,
        "delimiter": delimiter,
        "comment": comments,
        "header": None if use_names or skip_header > 0 else 'infer', # Infer header if not skipping and not providing names
        "skiprows": skip_header if not use_names else 0, # If use_names, pandas handles names, so don't skip here.
                                                         # This logic might need refinement based on pandas behavior.
                                                         # If skip_header is used, it's absolute.
        "names": use_names,
        "encoding": encoding,
        **kwargs
    }
    if use_names and skip_header > 0:
        # If both use_names and skip_header are given, pandas' names argument applies *after* skiprows.
        # So, we just pass skip_header to skiprows.
        read_csv_kwargs["skiprows"] = skip_header


    try:
        df = pd.read_csv(**read_csv_kwargs)
    except Exception as e: # pragma: no cover
        raise ValueError(f"Pandas could not read the file {filepath}. Error: {e}")

    if df.empty:
        raise ValueError(f"No data loaded from {filepath}. The file might be empty or all lines were skipped/comments.")

    # --- Column selection logic ---
    def get_column_data(df_input: pd.DataFrame, col_id: Union[int, str], col_desc: str) -> pd.Series:
        try:
            if isinstance(col_id, str): # Column name
                if col_id not in df_input.columns:
                    raise ValueError(f"{col_desc} column '{col_id}' not found in file columns: {list(df_input.columns)}.")
                series = df_input[col_id]
            elif isinstance(col_id, int): # Column index
                if col_id >= len(df_input.columns):
                    raise ValueError(
                        f"{col_desc} column index {col_id} is out of bounds for {len(df_input.columns)} columns."
                    )
                series = df_input.iloc[:, col_id]
            else: # Should have been caught by earlier TypeError
                raise TypeError(f"Invalid column identifier type for {col_desc}: {type(col_id)}")

            # Attempt to convert to numeric, coercing errors to NaN
            numeric_series = pd.to_numeric(series, errors='coerce')
            if numeric_series.isnull().any():
                nan_rows = numeric_series[numeric_series.isnull()].index.tolist()
                # Show first few problematic rows for better error message
                problem_snippet = series.iloc[nan_rows[:min(3, len(nan_rows))]].to_string(index=False)
                warnings.warn(
                    f"Non-numeric values found in {col_desc} column ('{col_id}') and converted to NaN. "
                    f"Problematic rows (first few):\n{problem_snippet}",
                    UserWarning
                )
            return numeric_series.dropna() # Drop rows where this column became NaN

        except Exception as e: # pragma: no cover
            raise ValueError(f"Error accessing or converting {col_desc} column ('{col_id}'): {e}")

    q_series = get_column_data(df, q_col, "q")
    i_series = get_column_data(df, i_col, "intensity")

    err_data: Optional[np.ndarray] = None
    if err_col is not None:
        err_series = get_column_data(df, err_col, "error")
    else:
        err_series = None

    # --- Align data after potential NaN drops ---
    # Find common non-NaN indices across q, I (and E if present)
    common_indices = q_series.index.intersection(i_series.index)
    if err_series is not None:
        common_indices = common_indices.intersection(err_series.index)

    if len(common_indices) == 0:
        raise ValueError("No valid data rows found after attempting to convert q, I (and error) columns to numeric and removing NaNs.")
    if len(common_indices) < len(df):
        warnings.warn(
            f"{len(df) - len(common_indices)} rows were dropped due to NaNs in q, I, or error columns.",
            UserWarning
        )

    q_data = q_series.loc[common_indices].to_numpy()
    i_data = i_series.loc[common_indices].to_numpy()
    if err_series is not None:
        err_data = err_series.loc[common_indices].to_numpy()


    # --- Metadata ---
    metadata: Dict[str, Any] = {
        "source_filepath": str(file_path_obj.resolve()),
        "filename": file_path_obj.name,
        "loader_function": "load_ascii_1d",
        "loader_options": {
            "q_col": q_col, "i_col": i_col, "err_col": err_col,
            "skip_header": skip_header, "delimiter": delimiter,
            "comments": comments, "use_names": use_names,
            "encoding": encoding,
            "pandas_kwargs": kwargs
        }
    }
    if metadata_func:
        try:
            custom_meta = metadata_func(file_path_obj, header_lines[:skip_header]) # Pass only actual skipped header
            if isinstance(custom_meta, dict):
                metadata.update(custom_meta)
            else: # pragma: no cover
                warnings.warn(f"Custom metadata_func for {filepath} did not return a dictionary.", UserWarning)
        except Exception as e: # pragma: no cover
            warnings.warn(f"Error executing metadata_func for {filepath}: {e}", UserWarning)

    # Default units (can be overridden by metadata_func or user later)
    # These are placeholders; true units often need to be known from context
    # or parsed from metadata.
    q_unit: QUnit = metadata.get("q_unit", "nm^-1")
    intensity_unit: IntensityUnit = metadata.get("intensity_unit", "a.u.")


    return ScatteringCurve1D(
        q=q_data,
        intensity=i_data,
        error=err_data,
        metadata=metadata,
        q_unit=q_unit,
        intensity_unit=intensity_unit
    )

# Future: Add functions like save_ascii_1d, load_hdf5_1d, load_rigaku_ras, etc.