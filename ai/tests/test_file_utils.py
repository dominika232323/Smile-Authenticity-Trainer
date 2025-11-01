import csv
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from ai.data_preprocessing.file_utils import add_header_to_csv, append_row_to_csv, save_frame


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


@pytest.mark.parametrize(
    ("header", "expected_header"),
    [
        (["A", "B", "C"], ["A", "B", "C"]),
        (["Name", "Age", "Score"], ["Name", "Age", "Score"]),
        (["Name & Title", "Age (years)", "Score %"], ["Name & Title", "Age (years)", "Score %"]),
    ],
)
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


@pytest.mark.parametrize(
    "header", [(["A", "B", "C"]), (["Name", "Age", "Score"]), (["Name & Title", "Age (years)", "Score %"])]
)
def test_add_header_to_csv_empty_file(header):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "test_empty.csv"

        csv_file.touch()

        with pytest.raises(pd.errors.EmptyDataError):
            add_header_to_csv(csv_file, header)


@pytest.mark.parametrize("header", [(["A", "B"]), (["Name", "Age"]), (["Name & Title", "Age (years)"])])
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


@pytest.mark.parametrize(
    "header", [(["A", "B", "C"]), (["Name", "Age", "Score"]), (["Name & Title", "Age (years)", "Score %"])]
)
def test_add_header_to_csv_nonexistent_file(header):
    nonexistent_file = Path("/nonexistent/directory/test.csv")

    with pytest.raises(FileNotFoundError):
        add_header_to_csv(nonexistent_file, header)


@pytest.mark.parametrize(
    "header", [(["A", "B", "C"]), (["Name", "Age", "Score"]), (["Name & Title", "Age (years)", "Score %"])]
)
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


@pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
def test_save_frame_rgb(image_format):
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_path = Path(temp_dir) / f"test_frame{image_format}"

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 255
        frame[:50, :, 1] = 255

        save_frame(frame, frame_path)

        assert frame_path.exists()

        loaded_frame = cv2.imread(str(frame_path))

        assert loaded_frame is not None
        assert loaded_frame.shape == frame.shape


@pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
def test_save_frame_grayscale(image_format):
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_path = Path(temp_dir) / f"grayscale{image_format}"

        frame = np.ones((50, 50), dtype=np.uint8) * 128

        save_frame(frame, frame_path)

        assert frame_path.exists()

        loaded_frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)

        assert loaded_frame is not None
        assert loaded_frame.shape == frame.shape


@pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
def test_save_frame_nested_directory(image_format):
    with tempfile.TemporaryDirectory() as temp_dir:
        nested_dir = Path(temp_dir) / "level1" / "level2"
        nested_dir.mkdir(parents=True, exist_ok=True)
        frame_path = nested_dir / f"nested_frame{image_format}"

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 255
        frame[:50, :, 1] = 255

        save_frame(frame, frame_path)

        assert frame_path.exists()

        loaded_frame = cv2.imread(str(frame_path))

        assert loaded_frame is not None
        assert loaded_frame.shape == frame.shape


@pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
def test_save_frame_overwrite_existing(image_format):
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_path = Path(temp_dir) / "overwrite.png"

        frame1 = np.zeros((50, 50, 3), dtype=np.uint8)
        save_frame(frame1, frame_path)

        original_size = frame_path.stat().st_size

        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        save_frame(frame2, frame_path)

        assert frame_path.exists()

        new_size = frame_path.stat().st_size

        assert new_size != original_size


@pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
def test_save_frame_uint8(image_format):
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_path_uint8 = Path(temp_dir) / f"uint8{image_format}"
        frame_uint8 = np.ones((50, 50, 3), dtype=np.uint8) * 127

        save_frame(frame_uint8, frame_path_uint8)

        assert frame_path_uint8.exists()


@pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
def test_save_frame_uint16(image_format):
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_path_uint16 = Path(temp_dir) / f"uint16{image_format}"
        frame_uint16 = np.ones((50, 50, 3), dtype=np.uint16) * 32767

        save_frame(frame_uint16, frame_path_uint16)

        assert frame_path_uint16.exists()


@pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
def test_save_frame_large_image(image_format):
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_path = Path(temp_dir) / f"large{image_format}"
        frame = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

        save_frame(frame, frame_path)

        assert frame_path.exists()
        assert frame_path.stat().st_size > 0


@pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
def test_save_frame_small_image(image_format):
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_path = Path(temp_dir) / f"small{image_format}"
        frame = np.array([[[255, 0, 0]]], dtype=np.uint8)

        save_frame(frame, frame_path)

        assert frame_path.exists()


@pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
def test_save_frame_invalid_directory(image_format):
    invalid_path = Path(f"/nonexistent/directory/frame{image_format}")
    frame = np.ones((50, 50, 3), dtype=np.uint8)

    save_frame(frame, invalid_path)

    assert not invalid_path.exists()


@pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
def test_save_frame_empty_array(image_format):
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_path = Path(temp_dir) / f"empty{image_format}"
        frame = np.array([], dtype=np.uint8)

        with pytest.raises(cv2.error):
            save_frame(frame, frame_path)


@pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
def test_save_frame_to_readonly_directory(image_format):
    with tempfile.TemporaryDirectory() as temp_dir:
        readonly_dir = Path(temp_dir) / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        frame_path = readonly_dir / f"frame{image_format}"
        frame = np.ones((50, 50, 3), dtype=np.uint8)

        try:
            save_frame(frame, frame_path)
        finally:
            readonly_dir.chmod(0o755)

            assert not frame_path.exists()


@pytest.mark.parametrize("extension", ["", ".unknown", ".xyz"])
def test_save_frame_unsupported_extensions(extension):
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_path = Path(temp_dir) / f"test{extension}"
        frame = np.ones((50, 50, 3), dtype=np.uint8)

        with pytest.raises(cv2.error):
            save_frame(frame, frame_path)


@pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
def test_save_frame_with_alpha_channel(image_format):
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_path = Path(temp_dir) / f"alpha{image_format}"

        frame = np.ones((50, 50, 4), dtype=np.uint8)
        frame[:, :, 0] = 255
        frame[:, :, 1] = 0
        frame[:, :, 2] = 0
        frame[:, :, 3] = 128

        save_frame(frame, frame_path)

        assert frame_path.exists()
