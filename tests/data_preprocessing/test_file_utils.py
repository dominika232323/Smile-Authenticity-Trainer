import csv
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from ai.data_preprocessing.file_utils import (
    add_header_to_csv,
    append_row_to_csv,
    save_frame,
    create_directories,
    create_csv_with_header,
    save_dataframe_to_csv,
    concat_csvs,
)


class TestAppendRowToCsv:
    @pytest.mark.parametrize(
        ("row", "expected"), [([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"]), ([16], ["16"]), ([], [])]
    )
    def test_append_row_to_new_csv_file(self, row, expected):
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
    def test_append_row_to_existing_csv_file(self, row, expected):
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

    def test_append_multiple_rows_sequentially(self):
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

    @pytest.mark.parametrize(
        ("row", "expected"), [([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"]), ([16], ["16"]), ([], [])]
    )
    def test_file_path_with_nested_directories(self, row, expected):
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
    def test_file_permissions_error(self, row):
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
    def test_invalid_directory_path(self, row):
        invalid_path = Path("/nonexistent/directory/test.csv")

        with pytest.raises(FileNotFoundError):
            append_row_to_csv(invalid_path, row)


class TestAddHeaderToCsv:
    @pytest.mark.parametrize(
        ("header", "expected_header"),
        [
            (["A", "B", "C"], ["A", "B", "C"]),
            (["Name", "Age", "Score"], ["Name", "Age", "Score"]),
            (["Name & Title", "Age (years)", "Score %"], ["Name & Title", "Age (years)", "Score %"]),
        ],
    )
    def test_add_header_to_csv_with_data(self, header, expected_header):
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
    def test_add_header_to_csv_empty_file(self, header):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "test_empty.csv"

            csv_file.touch()

            with pytest.raises(pd.errors.EmptyDataError):
                add_header_to_csv(csv_file, header)

    @pytest.mark.parametrize("header", [(["A", "B"]), (["Name", "Age"]), (["Name & Title", "Age (years)"])])
    def test_add_header_to_csv_mismatched_number_of_columns(self, header):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "test_mismatch.csv"

            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([1, 2, 3])
                writer.writerow([4, 5, 6])

            with pytest.raises(ValueError):
                add_header_to_csv(csv_file, header)

    def test_add_empty_header_to_csv(self):
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
    def test_add_header_to_csv_nonexistent_file(self, header):
        nonexistent_file = Path("/nonexistent/directory/test.csv")

        with pytest.raises(FileNotFoundError):
            add_header_to_csv(nonexistent_file, header)

    @pytest.mark.parametrize(
        "header", [(["A", "B", "C"]), (["Name", "Age", "Score"]), (["Name & Title", "Age (years)", "Score %"])]
    )
    def test_add_header_to_csv_file_permissions(self, header):
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


class TestCreateCsvWithHeader:
    @pytest.mark.parametrize(
        ("header", "expected_header"),
        [
            (["A", "B", "C"], ["A", "B", "C"]),
            (["Name", "Age", "Score"], ["Name", "Age", "Score"]),
            (["Name & Title", "Age (years)", "Score %"], ["Name & Title", "Age (years)", "Score %"]),
            (["ID", "Timestamp", "Value", "Status"], ["ID", "Timestamp", "Value", "Status"]),
            (["Single"], ["Single"]),
            (
                ["Column1", "Column2", "Column3", "Column4", "Column5", "Column6", "Column7", "Column8"],
                ["Column1", "Column2", "Column3", "Column4", "Column5", "Column6", "Column7", "Column8"],
            ),
            (
                ["Header with spaces", "Header_with_underscores", "Header-with-dashes"],
                ["Header with spaces", "Header_with_underscores", "Header-with-dashes"],
            ),
            (["Unicode 🎉", "Émojis", "Spëcîál Chàrs"], ["Unicode 🎉", "Émojis", "Spëcîál Chàrs"]),
            (
                ["Name, Title", 'Value "quoted"', "Description\nwith\nnewlines"],
                ["Name, Title", 'Value "quoted"', "Description\nwith\nnewlines"],
            ),
            (["1", "2.5", "3.14159", "-10", "0"], ["1", "2.5", "3.14159", "-10", "0"]),
        ],
    )
    def test_create_csv_with_header_basic(self, header, expected_header):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "test_create_header.csv"

            create_csv_with_header(csv_file, header)

            assert csv_file.exists()

            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

                assert len(rows) == 1
                assert rows[0] == expected_header

    def test_create_csv_with_header_empty_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "test_empty_header.csv"

            create_csv_with_header(csv_file, [])

            assert csv_file.exists()

            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

                assert len(rows) == 1
                assert rows[0] == []

    @pytest.mark.parametrize(
        ("header", "expected_header"),
        [
            (["A", "B", "C"], ["A", "B", "C"]),
            (["Name", "Age", "Score"], ["Name", "Age", "Score"]),
            (["Name & Title", "Age (years)", "Score %"], ["Name & Title", "Age (years)", "Score %"]),
            (["ID", "Timestamp", "Value", "Status"], ["ID", "Timestamp", "Value", "Status"]),
            (["Single"], ["Single"]),
            (
                ["Column1", "Column2", "Column3", "Column4", "Column5", "Column6", "Column7", "Column8"],
                ["Column1", "Column2", "Column3", "Column4", "Column5", "Column6", "Column7", "Column8"],
            ),
            (
                ["Header with spaces", "Header_with_underscores", "Header-with-dashes"],
                ["Header with spaces", "Header_with_underscores", "Header-with-dashes"],
            ),
            (["Unicode 🎉", "Émojis", "Spëcîál Chàrs"], ["Unicode 🎉", "Émojis", "Spëcîál Chàrs"]),
            (
                ["Name, Title", 'Value "quoted"', "Description\nwith\nnewlines"],
                ["Name, Title", 'Value "quoted"', "Description\nwith\nnewlines"],
            ),
            (["1", "2.5", "3.14159", "-10", "0"], ["1", "2.5", "3.14159", "-10", "0"]),
        ],
    )
    def test_create_csv_with_header_nested_directory(self, header, expected_header):
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "level1" / "level2" / "level3"
            csv_file = nested_dir / "test_nested.csv"

            assert not nested_dir.exists()

            create_csv_with_header(csv_file, header)

            assert csv_file.exists()

            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

                assert len(rows) == 1
                assert rows[0] == expected_header

    @pytest.mark.parametrize(
        ("header", "expected_header"),
        [
            (["A", "B", "C"], ["A", "B", "C"]),
            (["Name", "Age", "Score"], ["Name", "Age", "Score"]),
            (["Name & Title", "Age (years)", "Score %"], ["Name & Title", "Age (years)", "Score %"]),
            (["ID", "Timestamp", "Value", "Status"], ["ID", "Timestamp", "Value", "Status"]),
            (["Single"], ["Single"]),
            (
                ["Column1", "Column2", "Column3", "Column4", "Column5", "Column6", "Column7", "Column8"],
                ["Column1", "Column2", "Column3", "Column4", "Column5", "Column6", "Column7", "Column8"],
            ),
            (
                ["Header with spaces", "Header_with_underscores", "Header-with-dashes"],
                ["Header with spaces", "Header_with_underscores", "Header-with-dashes"],
            ),
            (["Unicode 🎉", "Émojis", "Spëcîál Chàrs"], ["Unicode 🎉", "Émojis", "Spëcîál Chàrs"]),
            (
                ["Name, Title", 'Value "quoted"', "Description\nwith\nnewlines"],
                ["Name, Title", 'Value "quoted"', "Description\nwith\nnewlines"],
            ),
            (["1", "2.5", "3.14159", "-10", "0"], ["1", "2.5", "3.14159", "-10", "0"]),
        ],
    )
    def test_create_csv_with_header_overwrite_existing_file(self, header, expected_header):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "test_overwrite.csv"

            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Old", "Header"])
                writer.writerow([1, 2])
                writer.writerow([3, 4])

            create_csv_with_header(csv_file, header)

            assert csv_file.exists()

            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

                assert len(rows) == 1
                assert rows[0] == expected_header

    @pytest.mark.parametrize(
        "header",
        [
            (["A", "B", "C"]),
            (["Name", "Age", "Score"]),
            (["Name & Title", "Age (years)", "Score %"]),
            (["ID", "Timestamp", "Value", "Status"]),
            (["Single"]),
            (["Column1", "Column2", "Column3", "Column4", "Column5", "Column6", "Column7", "Column8"]),
            (["Header with spaces", "Header_with_underscores", "Header-with-dashes"]),
            (["Unicode 🎉", "Émojis", "Spëcîál Chàrs"]),
            (["Name, Title", 'Value "quoted"', "Description\nwith\nnewlines"]),
            (["1", "2.5", "3.14159", "-10", "0"]),
        ],
    )
    def test_create_csv_with_header_invalid_directory_path(self, header):
        invalid_path = Path("/nonexistent/directory/test.csv")

        with pytest.raises(PermissionError):
            create_csv_with_header(invalid_path, header)

    @pytest.mark.parametrize(
        "header",
        [
            (["A", "B", "C"]),
            (["Name", "Age", "Score"]),
            (["Name & Title", "Age (years)", "Score %"]),
            (["ID", "Timestamp", "Value", "Status"]),
            (["Single"]),
            (["Column1", "Column2", "Column3", "Column4", "Column5", "Column6", "Column7", "Column8"]),
            (["Header with spaces", "Header_with_underscores", "Header-with-dashes"]),
            (["Unicode 🎉", "Émojis", "Spëcîál Chàrs"]),
            (["Name, Title", 'Value "quoted"', "Description\nwith\nnewlines"]),
            (["1", "2.5", "3.14159", "-10", "0"]),
        ],
    )
    def test_create_csv_with_header_readonly_directory(self, header):
        with tempfile.TemporaryDirectory() as temp_dir:
            readonly_dir = Path(temp_dir) / "readonly"
            readonly_dir.mkdir()
            readonly_dir.chmod(0o444)

            csv_file = readonly_dir / "test.csv"

            try:
                with pytest.raises(PermissionError):
                    create_csv_with_header(csv_file, header)
            finally:
                readonly_dir.chmod(0o755)

    def test_create_csv_with_header_large_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "test_large_header.csv"

            large_header = [f"Column_{i}" for i in range(1000)]

            create_csv_with_header(csv_file, large_header)

            assert csv_file.exists()

            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

                assert len(rows) == 1
                assert rows[0] == large_header


class TestSaveFrame:
    @pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
    def test_save_frame_rgb(self, image_format):
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
    def test_save_frame_grayscale(self, image_format):
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_path = Path(temp_dir) / f"grayscale{image_format}"

            frame = np.ones((50, 50), dtype=np.uint8) * 128

            save_frame(frame, frame_path)

            assert frame_path.exists()

            loaded_frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)

            assert loaded_frame is not None
            assert loaded_frame.shape == frame.shape

    @pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
    def test_save_frame_nested_directory(self, image_format):
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
    def test_save_frame_overwrite_existing(self, image_format):
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
    def test_save_frame_uint8(self, image_format):
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_path_uint8 = Path(temp_dir) / f"uint8{image_format}"
            frame_uint8 = np.ones((50, 50, 3), dtype=np.uint8) * 127

            save_frame(frame_uint8, frame_path_uint8)

            assert frame_path_uint8.exists()

    @pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
    def test_save_frame_uint16(self, image_format):
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_path_uint16 = Path(temp_dir) / f"uint16{image_format}"
            frame_uint16 = np.ones((50, 50, 3), dtype=np.uint16) * 32767

            save_frame(frame_uint16, frame_path_uint16)

            assert frame_path_uint16.exists()

    @pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
    def test_save_frame_large_image(self, image_format):
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_path = Path(temp_dir) / f"large{image_format}"
            frame = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

            save_frame(frame, frame_path)

            assert frame_path.exists()
            assert frame_path.stat().st_size > 0

    @pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
    def test_save_frame_small_image(self, image_format):
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_path = Path(temp_dir) / f"small{image_format}"
            frame = np.array([[[255, 0, 0]]], dtype=np.uint8)

            save_frame(frame, frame_path)

            assert frame_path.exists()

    @pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
    def test_save_frame_invalid_directory(self, image_format):
        invalid_path = Path(f"/nonexistent/directory/frame{image_format}")
        frame = np.ones((50, 50, 3), dtype=np.uint8)

        with pytest.raises(PermissionError):
            save_frame(frame, invalid_path)

    @pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
    def test_save_frame_empty_array(self, image_format):
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_path = Path(temp_dir) / f"empty{image_format}"
            frame = np.array([], dtype=np.uint8)

            with pytest.raises(cv2.error):
                save_frame(frame, frame_path)

    @pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
    def test_save_frame_to_readonly_directory(self, image_format):
        with tempfile.TemporaryDirectory() as temp_dir:
            readonly_dir = Path(temp_dir) / "readonly"
            readonly_dir.mkdir()
            readonly_dir.chmod(0o444)

            frame_path = readonly_dir / f"frame{image_format}"
            frame = np.ones((50, 50, 3), dtype=np.uint8)

            try:
                with pytest.raises(RuntimeError):
                    save_frame(frame, frame_path)
            finally:
                readonly_dir.chmod(0o755)

    @pytest.mark.parametrize("extension", ["", ".unknown", ".xyz"])
    def test_save_frame_unsupported_extensions(self, extension):
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_path = Path(temp_dir) / f"test{extension}"
            frame = np.ones((50, 50, 3), dtype=np.uint8)

            with pytest.raises(cv2.error):
                save_frame(frame, frame_path)

    @pytest.mark.parametrize("image_format", [".jpg", ".png", ".bmp", ".tiff"])
    def test_save_frame_with_alpha_channel(self, image_format):
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_path = Path(temp_dir) / f"alpha{image_format}"

            frame = np.ones((50, 50, 4), dtype=np.uint8)
            frame[:, :, 0] = 255
            frame[:, :, 1] = 0
            frame[:, :, 2] = 0
            frame[:, :, 3] = 128

            save_frame(frame, frame_path)

            assert frame_path.exists()


class TestCreateDirectories:
    def test_create_directories_single_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_directory"

            assert not new_dir.exists()

            create_directories([new_dir])

            assert new_dir.exists()
            assert new_dir.is_dir()

    def test_create_directories_multiple_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dir1 = Path(temp_dir) / "dir1"
            dir2 = Path(temp_dir) / "dir2"
            dir3 = Path(temp_dir) / "dir3"

            assert not dir1.exists()
            assert not dir2.exists()
            assert not dir3.exists()

            create_directories([dir1, dir2, dir3])

            assert dir1.exists() and dir1.is_dir()
            assert dir2.exists() and dir2.is_dir()
            assert dir3.exists() and dir3.is_dir()

    def test_create_directories_nested_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "level1" / "level2" / "level3"

            assert not nested_dir.exists()
            assert not nested_dir.parent.exists()
            assert not nested_dir.parent.parent.exists()

            create_directories([nested_dir])

            assert nested_dir.exists() and nested_dir.is_dir()
            assert nested_dir.parent.exists() and nested_dir.parent.is_dir()
            assert nested_dir.parent.parent.exists() and nested_dir.parent.parent.is_dir()

    def test_create_directories_already_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = Path(temp_dir) / "existing"
            existing_dir.mkdir()

            assert existing_dir.exists()

            create_directories([existing_dir])

            assert existing_dir.exists() and existing_dir.is_dir()

    def test_create_directories_empty_list(self):
        create_directories([])

    def test_create_directories_mixed_existing_and_new(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = Path(temp_dir) / "existing"
            new_dir = Path(temp_dir) / "new"

            existing_dir.mkdir()

            assert existing_dir.exists()
            assert not new_dir.exists()

            create_directories([existing_dir, new_dir])

            assert existing_dir.exists() and existing_dir.is_dir()
            assert new_dir.exists() and new_dir.is_dir()

    def test_create_directories_with_file_conflict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "conflicting_file"
            file_path.touch()

            assert file_path.exists() and file_path.is_file()

            with pytest.raises(FileExistsError):
                create_directories([file_path])

    def test_create_directories_permission_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            restricted_parent = Path(temp_dir) / "restricted"
            restricted_parent.mkdir()
            restricted_parent.chmod(0o444)

            restricted_child = restricted_parent / "child"

            try:
                with pytest.raises(PermissionError):
                    create_directories([restricted_child])
            finally:
                restricted_parent.chmod(0o755)

    def test_create_directories_complex_nested_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            complex_paths = [
                Path(temp_dir) / "project" / "src" / "main",
                Path(temp_dir) / "project" / "src" / "test",
                Path(temp_dir) / "project" / "docs" / "api",
                Path(temp_dir) / "project" / "build" / "output",
            ]

            for path in complex_paths:
                assert not path.exists()

            create_directories(complex_paths)

            for path in complex_paths:
                assert path.exists() and path.is_dir()


class TestSaveDataframeToCsv:
    def test_save_dataframe_to_csv_basic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "basic.csv"

            df = pd.DataFrame(
                {
                    "A": [1, 2, 3],
                    "B": ["x", "y", "z"],
                    "C": [0.1, 0.2, 0.3],
                }
            )

            save_dataframe_to_csv(df, output_path)

            assert output_path.exists()

            loaded = pd.read_csv(output_path)
            pd.testing.assert_frame_equal(loaded, df)

    def test_save_dataframe_to_csv_empty_dataframe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "empty.csv"

            df = pd.DataFrame(columns=["A", "B", "C"])

            save_dataframe_to_csv(df, output_path)

            assert output_path.exists()

            loaded = pd.read_csv(output_path)
            # Empty DataFrame, same columns
            assert list(loaded.columns) == ["A", "B", "C"]
            assert loaded.empty

    def test_save_dataframe_to_csv_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "overwrite.csv"

            df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
            df1.to_csv(output_path, index=False)

            original_size = output_path.stat().st_size

            df2 = pd.DataFrame({"A": [10, 20, 30], "B": [40, 50, 60]})
            save_dataframe_to_csv(df2, output_path)

            assert output_path.exists()

            new_size = output_path.stat().st_size

            assert new_size != original_size

            loaded = pd.read_csv(output_path)
            pd.testing.assert_frame_equal(loaded, df2)

    def test_save_dataframe_to_csv_various_dtypes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "dtypes.csv"

            df = pd.DataFrame(
                {
                    "int_col": pd.Series([1, 2, 3], dtype="int64"),
                    "float_col": pd.Series([1.1, 2.2, 3.3], dtype="float64"),
                    "bool_col": [True, False, True],
                    "str_col": ["a", "b", "c"],
                    "datetime_col": pd.date_range("2020-01-01", periods=3, freq="D"),
                }
            )

            save_dataframe_to_csv(df, output_path)

            assert output_path.exists()

            loaded = pd.read_csv(output_path, parse_dates=["datetime_col"])

            assert list(loaded.columns) == list(df.columns)
            assert loaded.shape == df.shape

    def test_save_dataframe_to_csv_nonexistent_directory(self):
        df = pd.DataFrame({"A": [1, 2, 3]})
        output_path = Path("/nonexistent/directory/df.csv")

        with pytest.raises(OSError):
            save_dataframe_to_csv(df, output_path)

    def test_save_dataframe_to_csv_readonly_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            readonly_dir = Path(temp_dir) / "readonly"
            readonly_dir.mkdir()
            readonly_dir.chmod(0o444)

            output_path = readonly_dir / "df.csv"
            df = pd.DataFrame({"A": [1, 2, 3]})

            try:
                with pytest.raises(PermissionError):
                    save_dataframe_to_csv(df, output_path)
            finally:
                readonly_dir.chmod(0o755)
            

class TestConcatCsvs:
    def test_concat_csvs_basic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)

            df1 = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
            df2 = pd.DataFrame({"id": [3, 4], "val": ["c", "d"]})

            df1.to_csv(input_dir / "001.csv", index=False)
            df2.to_csv(input_dir / "002.csv", index=False)

            result = concat_csvs(input_dir)

            result_sorted = result.sort_values("id").reset_index(drop=True)
            expected = pd.concat([df1, df2], ignore_index=True).sort_values("id").reset_index(drop=True)

            pd.testing.assert_frame_equal(result_sorted, expected)

    def test_concat_csvs_no_csv_files_empty_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)

            with pytest.raises(ValueError):
                concat_csvs(input_dir)

    def test_concat_csvs_no_csv_files_only_others(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)

            (input_dir / "data.txt").write_text("hello")
            (input_dir / "data.json").write_text("{}")

            with pytest.raises(ValueError):
                concat_csvs(input_dir)

    def test_concat_csvs_mismatched_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)

            df1 = pd.DataFrame({"row_id": [1], "A": [10], "B": [20]})
            df2 = pd.DataFrame({"row_id": [2], "B": [200], "C": [300]})

            df1.to_csv(input_dir / "part1.csv", index=False)
            df2.to_csv(input_dir / "part2.csv", index=False)

            result = concat_csvs(input_dir)
            result_sorted = result.sort_values("row_id").reset_index(drop=True)

            expected = pd.concat([df1, df2], ignore_index=True)
            expected_sorted = expected.sort_values("row_id").reset_index(drop=True)
            expected_sorted = expected_sorted[result_sorted.columns]

            pd.testing.assert_frame_equal(result_sorted, expected_sorted)

    def test_concat_csvs_ignores_non_csv_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)

            df1 = pd.DataFrame({"id": [1], "x": [0.1]})
            df2 = pd.DataFrame({"id": [2], "x": [0.2]})

            df1.to_csv(input_dir / "a.csv", index=False)
            (input_dir / "readme.txt").write_text("info")
            (input_dir / "notes.md").write_text("- note")
            df2.to_csv(input_dir / "b.csv", index=False)

            result = concat_csvs(input_dir)

            assert set(result.columns) == {"id", "x"}
            assert len(result) == 2
            assert set(result["id"]) == {1, 2}
