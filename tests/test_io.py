# tests/test_io.py
"""
Unit tests for the scatterbrain.io module.
"""
import logging
import pathlib

import numpy as np
import pytest

from scatterbrain.core import ScatteringCurve1D
from scatterbrain.io import load_ascii_1d, save_ascii_1d

# Helper to create temporary data files within the tests/test_data directory
# This ensures tests don't rely on files outside the test suite control after initial setup.
# For this exercise, we've manually defined the files above. In a real scenario,
# you might generate these on the fly in fixtures if they are very simple,
# or rely on them being present in a dedicated test_data folder.

TEST_DATA_DIR = pathlib.Path(__file__).parent / "test_data"

# --- Test Cases ---


class TestLoadAscii1D:
    """Tests for the load_ascii_1d function."""

    def test_load_simple_3col_dat(self):
        """Load a simple space-delimited file with q, I, err and header."""
        filepath = TEST_DATA_DIR / "simple_3col.dat"
        curve = load_ascii_1d(
            filepath,
            q_col=0,
            i_col=1,
            err_col=2,
            skip_header=2,  # Skip the two '#' header lines
            delimiter=r"\s+",  # Match one or more whitespace characters
        )
        assert isinstance(curve, ScatteringCurve1D)
        assert len(curve) == 5
        assert np.allclose(curve.q, [0.1, 0.2, 0.3, 0.4, 0.5])
        assert np.allclose(curve.intensity, [100.0, 80.0, 50.0, 30.0, 10.0])
        assert curve.error is not None
        assert np.allclose(curve.error, [10.0, 8.0, 5.0, 3.0, 1.0])
        assert curve.metadata["filename"] == "simple_3col.dat"
        assert curve.metadata["loader_options"]["skip_header"] == 2

    def test_load_simple_2col_csv(self):
        """Load a simple comma-delimited file with q, I, no error, and named headers."""
        filepath = TEST_DATA_DIR / "simple_2col_comma.csv"
        curve = load_ascii_1d(
            filepath,
            q_col="Q_values",  # Use column name
            i_col="Intensity_values",
            err_col=None,  # No error column
            delimiter=",",
            comments="#",  # '#' is a comment
            skip_header=0,  # Pandas infers header from first line
        )
        assert isinstance(curve, ScatteringCurve1D)
        assert len(curve) == 3
        assert np.allclose(curve.q, [0.05, 0.15, 0.25])
        assert np.allclose(curve.intensity, [1000.0, 200.0, 50.0])
        assert curve.error is None
        assert curve.metadata["filename"] == "simple_2col_comma.csv"

    def test_load_with_use_names(self):
        """Test loading when explicit column names are provided."""
        filepath = (
            TEST_DATA_DIR / "simple_3col.dat"
        )  # Re-use, but ignore its internal header
        curve = load_ascii_1d(
            filepath,
            q_col="my_q",
            i_col="my_I",
            err_col="my_err",
            skip_header=2,  # Still need to skip the file's own comment lines
            delimiter=r"\s+",
            use_names=["my_q", "my_I", "my_err", "extra_col_ignored"],  # Provide names
        )
        assert len(curve) == 5
        assert np.allclose(curve.q, [0.1, 0.2, 0.3, 0.4, 0.5])
        assert "extra_col_ignored" not in curve.metadata.get(
            "loaded_columns", []
        )  # Check it wasn't used for data

    def test_custom_metadata_func(self):
        """Test using a custom function to extract metadata from headers."""
        filepath = TEST_DATA_DIR / "header_and_mixed_delimiter.txt"

        def extract_meta(file_path_obj: pathlib.Path, header_lines: list[str]):
            meta = {}
            for line in header_lines:
                if line.lower().startswith("sample name:"):
                    meta["sample_id"] = line.split(":", 1)[1].strip()
                if line.lower().startswith("wavelength_nm:"):
                    try:
                        meta["wavelength_nm"] = float(line.split(":", 1)[1].strip())
                        meta["q_unit"] = "nm^-1"  # Assume based on wavelength
                    except ValueError:
                        pass
            return meta

        curve = load_ascii_1d(
            filepath,
            q_col=0,
            i_col=1,
            err_col=2,
            skip_header=6,  # Skip header + column names line
            delimiter=r"[,;\s]+",  # Match any combination of comma, semicolon, or whitespace
            comments="#",  # Handle '#' comments properly
            metadata_func=extract_meta,
        )
        assert (
            len(curve) == 5
        )  # Data rows with mixed delimiters will be tricky for pandas here.
        # This test expects the primary space delimiter to work for the numeric part.
        # The trailing comments might be handled by pandas or might cause issues.
        # The test is designed to see if the first 3 numeric columns are parsed.
        assert np.allclose(curve.q, [0.1, 0.2, 0.3, 0.4, 0.5])
        assert np.allclose(curve.intensity, [1.2e3, 9.5e2, 5.0e2, 2.1e2, 1.0e2])
        assert curve.error is not None
        assert np.allclose(curve.error, [12.0, 9.5, 5.0, 2.1, 1.0])
        assert curve.metadata["sample_id"] == "My Sample X"
        assert curve.metadata["wavelength_nm"] == 0.154
        assert curve.q_unit == "nm^-1"

    def test_file_not_found(self):
        """Test loading a non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_ascii_1d("non_existent_file.dat")

    def test_empty_data_file(self):
        """Test loading a file that becomes empty after skipping headers/comments."""
        filepath = TEST_DATA_DIR / "empty_file.dat"
        with pytest.raises(ValueError, match="Pandas could not read the file"):
            load_ascii_1d(filepath, skip_header=1)

    def test_only_header_file(self):
        """Test loading a file that only contains header lines and no data rows."""
        filepath = TEST_DATA_DIR / "only_header.dat"
        with pytest.raises(ValueError, match="Pandas could not read the file"):
            load_ascii_1d(filepath, skip_header=3)  # Skip all lines

    def test_malformed_data_coercion_and_warning(self, caplog):
        """Test behavior with non-numeric data in columns - should coerce and warn."""
        filepath = TEST_DATA_DIR / "malformed_data.dat"
        with caplog.at_level(logging.WARNING, logger="scatterbrain"):
            curve = load_ascii_1d(
                filepath,
                q_col=0,
                i_col=1,
                err_col=2,
                skip_header=1,  # Skip "q I err" header line
                delimiter=r"\s+",
            )
            # Expecting rows with "text_instead_of_number" or "not_a_number" to be dropped
            # or for those specific columns to be NaN and then rows dropped if q or I are NaN.
            # Row 0: err is "text..." -> q=0.1, I=100, err=NaN. If err is required, this row might be dropped.
            # Row 2: I is "not_a_number" -> q=0.3, I=NaN, err=5.0. This row will be dropped because I is NaN.

            # After NaN coercion and dropping:
            # Valid rows should be:
            # q=0.2, I=80, err=8.0
            # q=0.4, I=30, err=3.0
            # If err was text and q,I were numbers, that row would also be kept (with err=NaN)
            # if we handle err_series.dropna() separately or if err_col is None.
            # The current implementation drops rows if ANY of q, I, or err (if specified) are NaN after coercion.

        assert len(caplog.records) > 0  # At least one warning should be issued
        assert any("Non-numeric values found" in r.message for r in caplog.records)
        assert any("rows were dropped due to NaNs" in r.message for r in caplog.records)

        # The exact data depends on how pandas handles mixed types and how we drop NaNs.
        # With current implementation (dropna on each series before intersection):
        # q_series will be [0.1, 0.2, 0.3, 0.4]
        # i_series will be [100, 80, 30] (index 2 for 'not_a_number' dropped)
        # err_series will be [8.0, 3.0] (index 0 for 'text...' dropped)
        # Intersection of indices will lead to rows where q=0.2,I=80,err=8.0 and q=0.4,I=30,err=3.0
        assert len(curve) == 2
        assert np.allclose(curve.q, [0.2, 0.4])
        assert np.allclose(curve.intensity, [80.0, 30.0])
        assert curve.error is not None
        assert np.allclose(curve.error, [8.0, 3.0])

    def test_invalid_column_specifiers(self):
        """Test invalid types for column specifiers."""
        filepath = TEST_DATA_DIR / "simple_3col.dat"
        with pytest.raises(
            TypeError, match="Argument 'q_col' must be an int, str, or None"
        ):
            load_ascii_1d(filepath, q_col=[0])  # Pass a list
        with pytest.raises(
            TypeError, match="Argument 'err_col' must be an int, str, or None"
        ):
            load_ascii_1d(filepath, err_col={"col": 2})  # Pass a dict

    def test_q_col_not_found_by_name(self):
        filepath = TEST_DATA_DIR / "simple_2col_comma.csv"
        with pytest.raises(ValueError, match="q column 'NonExistentQ' not found"):
            load_ascii_1d(
                filepath, q_col="NonExistentQ", i_col="Intensity_values", delimiter=","
            )

    def test_i_col_not_found_by_index(self):
        filepath = TEST_DATA_DIR / "simple_2col_comma.csv"
        with pytest.raises(
            ValueError, match="intensity column index 10 is out of bounds"
        ):
            load_ascii_1d(filepath, q_col=0, i_col=10, delimiter=",")  # Only 2 columns

    def test_load_with_pandas_kwargs(self):
        """Test passing additional kwargs to pandas.read_csv, e.g., dtype."""
        filepath = TEST_DATA_DIR / "simple_3col.dat"
        # Force q to be loaded as string initially, then converted by our function.
        # This is mostly to check if kwargs are passed.
        curve = load_ascii_1d(
            filepath,
            skip_header=2,
            delimiter=r"\s+",
            dtype={0: str},  # Pass dtype for first column as string
        )
        assert len(curve) == 5
        assert (
            curve.q.dtype == np.float64
        )  # Should still be float after our pd.to_numeric
        assert np.allclose(curve.q, [0.1, 0.2, 0.3, 0.4, 0.5])


# ---------------------------------------------------------------------------
# Tests for save_ascii_1d
# ---------------------------------------------------------------------------


class TestSaveAscii1D:
    """Tests for save_ascii_1d."""

    @pytest.fixture
    def sample_curve(self):
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        intensity = np.array([100.0, 80.0, 50.0, 30.0, 10.0])
        error = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        return ScatteringCurve1D(
            q,
            intensity,
            error,
            metadata={"filename": "test.dat"},
            q_unit="nm^-1",
            intensity_unit="a.u.",
        )

    @pytest.fixture
    def sample_curve_no_error(self):
        q = np.array([0.1, 0.2, 0.3])
        intensity = np.array([100.0, 80.0, 50.0])
        return ScatteringCurve1D(q, intensity, q_unit="nm^-1", intensity_unit="a.u.")

    def test_round_trip_with_error(self, sample_curve, tmp_path):
        out = tmp_path / "out.dat"
        save_ascii_1d(sample_curve, out, include_error=True)
        # use_names forces header=None so pandas doesn't mistake the first data row for headers
        reloaded = load_ascii_1d(
            out,
            err_col=2,
            delimiter=r"\s+",
            use_names=["q", "intensity", "error"],
        )
        np.testing.assert_allclose(reloaded.q, sample_curve.q, rtol=1e-5)
        np.testing.assert_allclose(
            reloaded.intensity, sample_curve.intensity, rtol=1e-5
        )
        np.testing.assert_allclose(reloaded.error, sample_curve.error, rtol=1e-5)

    def test_round_trip_without_error(self, sample_curve, tmp_path):
        out = tmp_path / "out_no_err.dat"
        save_ascii_1d(sample_curve, out, include_error=False)
        reloaded = load_ascii_1d(
            out,
            delimiter=r"\s+",
            use_names=["q", "intensity"],
        )
        np.testing.assert_allclose(reloaded.q, sample_curve.q, rtol=1e-5)
        np.testing.assert_allclose(
            reloaded.intensity, sample_curve.intensity, rtol=1e-5
        )
        assert reloaded.error is None

    def test_no_error_column_when_curve_has_none(self, sample_curve_no_error, tmp_path):
        out = tmp_path / "no_err.dat"
        save_ascii_1d(sample_curve_no_error, out, include_error=True)
        # File should have only 2 data columns
        content = out.read_text()
        data_lines = [
            ln for ln in content.splitlines() if ln and not ln.startswith("#")
        ]
        assert len(data_lines[0].split()) == 2

    def test_header_written(self, sample_curve, tmp_path):
        out = tmp_path / "header.dat"
        save_ascii_1d(sample_curve, out, header="Custom header line")
        content = out.read_text()
        assert "# Saved by ScatterBrain" in content
        assert "Custom header line" in content

    def test_custom_delimiter(self, sample_curve, tmp_path):
        out = tmp_path / "comma.dat"
        save_ascii_1d(sample_curve, out, delimiter=",", include_error=False)
        content = out.read_text()
        data_lines = [
            ln for ln in content.splitlines() if ln and not ln.startswith("#")
        ]
        assert "," in data_lines[0]

    def test_missing_parent_raises(self, sample_curve, tmp_path):
        out = tmp_path / "nonexistent_dir" / "out.dat"
        with pytest.raises(ValueError, match="Parent directory"):
            save_ascii_1d(sample_curve, out)

    def test_wrong_type_raises(self, tmp_path):
        out = tmp_path / "out.dat"
        with pytest.raises(TypeError, match="ScatteringCurve1D"):
            save_ascii_1d("not a curve", out)
