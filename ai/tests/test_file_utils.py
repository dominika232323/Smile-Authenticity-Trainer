import csv
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ai.data_preprocessing.file_utils import append_row_to_csv, add_header_to_csv


@pytest.mark.parametrize(("row", "expected"), [([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"]), ([16], ["16"]), ([], [])])
def test_append_row_to_new_csv_file(row, expected):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "test_new.csv"
        append_row_to_csv(csv_file, row)

        assert csv_file.exists()

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

            assert len(rows) == 1
            assert rows[0] == expected


@pytest.mark.parametrize(("row", "expected"), [([70, 80, 90], ["70", "80", "90"]), ([16], ["16"]), ([], [])])
def test_append_row_to_existing_csv_file(row, expected):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "test_existing.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([10, 20, 30])
            writer.writerow([40, 50, 60])

        append_row_to_csv(csv_file, row)

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

            assert len(rows) == 3
            assert rows[0] == ["10", "20", "30"]
            assert rows[1] == ["40", "50", "60"]
            assert rows[2] == expected


def test_append_multiple_rows_sequentially():
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "test_multiple.csv"

        rows_to_append = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        for row in rows_to_append:
            append_row_to_csv(csv_file, row)

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            result_rows = list(reader)

            assert len(result_rows) == 3
            assert result_rows[0] == ["1", "2", "3"]
            assert result_rows[1] == ["4", "5", "6"]
            assert result_rows[2] == ["7", "8", "9"]


@pytest.mark.parametrize(("row", "expected"), [([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"]), ([16], ["16"]), ([], [])])
def test_file_path_with_nested_directories(row, expected):
    with tempfile.TemporaryDirectory() as temp_dir:
        nested_dir = Path(temp_dir) / "level1" / "level2"
        nested_dir.mkdir(parents=True, exist_ok=True)
        csv_file = nested_dir / "test_nested.csv"
        append_row_to_csv(csv_file, row)

        assert csv_file.exists()

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

            assert len(rows) == 1
            assert rows[0] == expected


@pytest.mark.parametrize("row", [([1, 2, 3, 4, 5]), ([16]), ([])])
def test_file_permissions_error(row):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "readonly.csv"

        csv_file.touch()
        csv_file.chmod(0o444)

        row = [1, 2, 3]

        try:
            with pytest.raises(PermissionError):
                append_row_to_csv(csv_file, row)
        finally:
            csv_file.chmod(0o666)


@pytest.mark.parametrize("row", [([1, 2, 3, 4, 5]), ([16]), ([])])
def test_invalid_directory_path(row):
    invalid_path = Path("/nonexistent/directory/test.csv")

    with pytest.raises(FileNotFoundError):
        append_row_to_csv(invalid_path, row)


@pytest.mark.parametrize(("header", "expected_header"), [
    (["A", "B", "C"], ["A", "B", "C"]),
    (["Name", "Age", "Score"], ["Name", "Age", "Score"]),
    (["Name & Title", "Age (years)", "Score %"], ["Name & Title", "Age (years)", "Score %"])
])
def test_add_header_to_csv_with_data(header, expected_header):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "test_header.csv"
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([1, 2, 3])
            writer.writerow([4, 5, 6])
        
        add_header_to_csv(csv_file, header)
        
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

            assert rows[0] == expected_header
            assert rows[1] == ["1", "2", "3"]
            assert rows[2] == ["4", "5", "6"]

            assert len(rows) == 3


@pytest.mark.parametrize("header", [
    (["A", "B", "C"]),
    (["Name", "Age", "Score"]),
    (["Name & Title", "Age (years)", "Score %"])
])
def test_add_header_to_csv_empty_file(header):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "test_empty.csv"
        
        csv_file.touch()

        with pytest.raises(pd.errors.EmptyDataError):
            add_header_to_csv(csv_file, header)


@pytest.mark.parametrize("header", [
    (["A", "B"]),
    (["Name", "Age"]),
    (["Name & Title", "Age (years)"])
])
def test_add_header_to_csv_mismatched_number_of_columns(header):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "test_mismatch.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([1, 2, 3])
            writer.writerow([4, 5, 6])

        with pytest.raises(ValueError):
            add_header_to_csv(csv_file, header)


def test_add_empty_header_to_csv():
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "test_empty_header.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([1, 2, 3])
            writer.writerow([4, 5, 6])

        with pytest.raises(ValueError):
            add_header_to_csv(csv_file, [])


@pytest.mark.parametrize("header", [
    (["A", "B", "C"]),
    (["Name", "Age", "Score"]),
    (["Name & Title", "Age (years)", "Score %"])
])
def test_add_header_to_csv_nonexistent_file(header):
    nonexistent_file = Path("/nonexistent/directory/test.csv")

    with pytest.raises(FileNotFoundError):
        add_header_to_csv(nonexistent_file, header)


@pytest.mark.parametrize("header", [
    (["A", "B", "C"]),
    (["Name", "Age", "Score"]),
    (["Name & Title", "Age (years)", "Score %"])
])
def test_add_header_to_csv_file_permissions(header):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "readonly.csv"
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([1, 2, 3])
        
        csv_file.chmod(0o444)
        
        try:
            with pytest.raises(PermissionError):
                add_header_to_csv(csv_file, header)
        finally:
            csv_file.chmod(0o666)
