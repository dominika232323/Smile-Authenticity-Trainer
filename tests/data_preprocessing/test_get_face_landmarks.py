import csv
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pandas as pd
import pytest

from ai.data_preprocessing.get_face_landmarks import (
    create_facelandmarks_header,
    get_face_landmarks,
    get_face_landmark_coords,
)


def create_test_face_image():
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Face outline
    cv2.circle(image, (320, 240), 100, (255, 255, 255), -1)

    # Eyes
    cv2.circle(image, (300, 220), 10, (0, 0, 0), -1)
    cv2.circle(image, (340, 220), 10, (0, 0, 0), -1)

    # Nose
    cv2.line(image, (320, 240), (320, 260), (128, 128, 128), 2)

    # Mouth
    cv2.ellipse(image, (320, 280), (20, 10), 0, 0, 180, (0, 0, 0), 2)

    return image


def create_mock_landmark(x, y):
    landmark = Mock()

    landmark.x = x
    landmark.y = y

    return landmark


def create_mock_face_landmarks(num_landmarks=468):
    landmarks = []

    for i in range(num_landmarks):
        x = (i % 20) / 20.0
        y = (i // 20) / (num_landmarks // 20)

        landmarks.append(create_mock_landmark(x, y))

    face_landmarks = Mock()
    face_landmarks.landmark = landmarks

    return face_landmarks


@patch("ai.data_preprocessing.get_face_landmarks.mp.solutions.face_mesh.FaceMesh")
def test_get_face_landmarks_with_detected_face(mock_face_mesh_class):
    with tempfile.TemporaryDirectory() as temp_dir:
        landmarks_file = Path(temp_dir) / "landmarks.csv"
        frame = create_test_face_image()
        frame_number = 1

        mock_face_mesh = Mock()
        mock_face_mesh_class.return_value.__enter__.return_value = mock_face_mesh

        mock_results = Mock()
        mock_face_landmarks = create_mock_face_landmarks()
        mock_results.multi_face_landmarks = [mock_face_landmarks]
        mock_face_mesh.process.return_value = mock_results

        get_face_landmarks(frame, frame_number, landmarks_file)

        assert landmarks_file.exists()

        with open(landmarks_file, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert len(rows[0]) == 937
        assert rows[0][0] == str(frame_number)


@patch("ai.data_preprocessing.get_face_landmarks.mp.solutions.face_mesh.FaceMesh")
def test_get_face_landmarks_no_face_detected(mock_face_mesh_class):
    with tempfile.TemporaryDirectory() as temp_dir:
        landmarks_file = Path(temp_dir) / "landmarks.csv"
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_number = 1

        mock_face_mesh = Mock()
        mock_face_mesh_class.return_value.__enter__.return_value = mock_face_mesh

        mock_results = Mock()
        mock_results.multi_face_landmarks = None
        mock_face_mesh.process.return_value = mock_results

        get_face_landmarks(frame, frame_number, landmarks_file)

        assert not landmarks_file.exists()


@patch("ai.data_preprocessing.get_face_landmarks.mp.solutions.face_mesh.FaceMesh")
def test_get_face_landmarks_multiple_calls(mock_face_mesh_class):
    with tempfile.TemporaryDirectory() as temp_dir:
        landmarks_file = Path(temp_dir) / "landmarks.csv"
        frame = create_test_face_image()

        mock_face_mesh = Mock()
        mock_face_mesh_class.return_value.__enter__.return_value = mock_face_mesh

        mock_results = Mock()
        mock_face_landmarks = create_mock_face_landmarks()
        mock_results.multi_face_landmarks = [mock_face_landmarks]
        mock_face_mesh.process.return_value = mock_results

        get_face_landmarks(frame, 1, landmarks_file)
        get_face_landmarks(frame, 2, landmarks_file)
        get_face_landmarks(frame, 3, landmarks_file)

        with open(landmarks_file, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0][0] == "1"
        assert rows[1][0] == "2"
        assert rows[2][0] == "3"


@patch("ai.data_preprocessing.get_face_landmarks.mp.solutions.face_mesh.FaceMesh")
def test_get_face_landmarks_empty_landmarks(mock_face_mesh_class):
    with tempfile.TemporaryDirectory() as temp_dir:
        landmarks_file = Path(temp_dir) / "landmarks.csv"
        frame = create_test_face_image()
        frame_number = 1

        mock_face_mesh = Mock()
        mock_face_mesh_class.return_value.__enter__.return_value = mock_face_mesh

        mock_results = Mock()
        mock_results.multi_face_landmarks = []
        mock_face_mesh.process.return_value = mock_results

        get_face_landmarks(frame, frame_number, landmarks_file)

        assert not landmarks_file.exists()


@patch("ai.data_preprocessing.get_face_landmarks.mp.solutions.face_mesh.FaceMesh")
@patch("builtins.print")
def test_get_face_landmarks_handles_exceptions(mock_print, mock_face_mesh_class):
    with tempfile.TemporaryDirectory() as temp_dir:
        landmarks_file = Path(temp_dir) / "landmarks.csv"
        frame = create_test_face_image()
        frame_number = 1

        mock_face_mesh = Mock()
        mock_face_mesh_class.return_value.__enter__.return_value = mock_face_mesh

        mock_face_landmarks = Mock()
        mock_face_landmarks.landmark = Mock(side_effect=Exception("Test exception"))

        mock_results = Mock()
        mock_results.multi_face_landmarks = [mock_face_landmarks]
        mock_face_mesh.process.return_value = mock_results

        with pytest.raises(Exception):
            get_face_landmarks(frame, frame_number, landmarks_file)


@patch("ai.data_preprocessing.get_face_landmarks.mp.solutions.face_mesh.FaceMesh")
def test_get_face_landmarks_different_frame_types_and_sizes(mock_face_mesh_class):
    with tempfile.TemporaryDirectory() as temp_dir:
        landmarks_file = Path(temp_dir) / "landmarks.csv"

        mock_face_mesh = Mock()
        mock_face_mesh_class.return_value.__enter__.return_value = mock_face_mesh

        mock_results = Mock()
        mock_face_landmarks = create_mock_face_landmarks(468)
        mock_results.multi_face_landmarks = [mock_face_landmarks]
        mock_face_mesh.process.return_value = mock_results

        frames = [
            np.ones((100, 100, 3), dtype=np.uint8),
            np.ones((1080, 1920, 3), dtype=np.uint8),
            np.ones((480, 640, 3), dtype=np.uint16),
        ]

        for i, frame in enumerate(frames):
            get_face_landmarks(frame, i + 1, landmarks_file)

        with open(landmarks_file, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 3


@patch("ai.data_preprocessing.get_face_landmarks.mp.solutions.face_mesh.FaceMesh")
def test_get_face_landmarks_mediapipe_configuration(mock_face_mesh_class):
    with tempfile.TemporaryDirectory() as temp_dir:
        landmarks_file = Path(temp_dir) / "landmarks.csv"
        frame = create_test_face_image()

        mock_face_mesh = Mock()
        mock_face_mesh_class.return_value.__enter__.return_value = mock_face_mesh

        mock_results = Mock()
        mock_results.multi_face_landmarks = None
        mock_face_mesh.process.return_value = mock_results

        get_face_landmarks(frame, 1, landmarks_file)

        mock_face_mesh_class.assert_called_once_with(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )


@pytest.mark.parametrize(
    ("data", "expected_header"),
    [
        (
            [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            ["frame_number", "0_x", "0_y", "1_x", "1_y", "2_x", "2_y", "3_x", "3_y"],
        ),
        ([1, 0.1, 0.2, 0.3, 0.4], ["frame_number", "0_x", "0_y", "1_x", "1_y"]),
    ],
)
def test_create_facelandmarks_header_with_existing_csv(data, expected_header):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "test.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data)

        header = create_facelandmarks_header(csv_file)

        assert header == expected_header


def test_create_facelandmarks_header_empty_csv():
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "empty.csv"
        csv_file.touch()

        with pytest.raises(pd.errors.EmptyDataError):
            create_facelandmarks_header(csv_file)


def test_create_facelandmarks_header_nonexistent_file():
    nonexistent_file = Path("/nonexistent/file.csv")

    with pytest.raises(FileNotFoundError):
        create_facelandmarks_header(nonexistent_file)


@pytest.mark.parametrize("num_landmarks", [1, 10, 50, 468])
def test_create_facelandmarks_header_various_sizes(num_landmarks):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / f"test_{num_landmarks}.csv"

        row = [1] + [0.5] * (2 * num_landmarks)

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        header = create_facelandmarks_header(csv_file)

        expected_header = ["frame_number"]
        for i in range(num_landmarks):
            expected_header.extend([f"{i}_x", f"{i}_y"])

        assert header == expected_header
        assert len(header) == 1 + 2 * num_landmarks


@pytest.mark.parametrize(
    ("data", "expected_header"),
    [
        (
            [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ["frame_number", "0_x", "0_y", "1_x", "1_y", "2_x", "2_y", "3_x", "3_y"],
        ),
        ([1, 0.1, 0.2, 0.3], ["frame_number", "0_x", "0_y"]),
    ],
)
def test_create_facelandmarks_header_odd_number_coordinates(data, expected_header):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "odd.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data)

        header = create_facelandmarks_header(csv_file)

        assert header == expected_header


@pytest.mark.parametrize(
    ("frame_number", "landmark_idx", "expected_x", "expected_y"),
    [
        (1, 0, 100.5, 150.0),
        (2, 1, 210.0, 260.5),
        (3, 0, 300.5, 350.0),
    ],
)
def test_get_face_landmark_coords_valid_data(frame_number, landmark_idx, expected_x, expected_y):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "landmarks.csv"

        data = {
            "frame_number": [1, 2, 3],
            "0_x": [100.5, 200.5, 300.5],
            "0_y": [150.0, 250.0, 350.0],
            "1_x": [110.0, 210.0, 310.0],
            "1_y": [160.5, 260.5, 360.5],
        }

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        result = get_face_landmark_coords(csv_file, frame_number, landmark_idx)

        assert isinstance(result, tuple)
        assert len(result) == 2

        x, y = result

        assert x == expected_x
        assert y == expected_y


def test_get_face_landmark_coords_nonexistent_frame():
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "landmarks.csv"

        data = {
            "frame_number": [1, 2, 3],
            "0_x": [100.5, 200.5, 300.5],
            "0_y": [150.0, 250.0, 350.0],
            "1_x": [110.0, 210.0, 310.0],
            "1_y": [160.5, 260.5, 360.5],
        }

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        with pytest.raises(IndexError):
            get_face_landmark_coords(csv_file, 999, 0)


def test_get_face_landmark_coords_nonexistent_landmark():
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "landmarks.csv"

        data = {
            "frame_number": [1, 2, 3],
            "0_x": [100.5, 200.5, 300.5],
            "0_y": [150.0, 250.0, 350.0],
            "1_x": [110.0, 210.0, 310.0],
            "1_y": [160.5, 260.5, 360.5],
        }

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        with pytest.raises(KeyError):
            get_face_landmark_coords(csv_file, 1, 999)


def test_get_face_landmark_coords_nonexistent_file():
    nonexistent_file = Path("/nonexistent/landmarks.csv")

    with pytest.raises(FileNotFoundError):
        get_face_landmark_coords(nonexistent_file, 1, 0)


def test_get_face_landmark_coords_empty_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "empty.csv"
        csv_file.touch()

        with pytest.raises(pd.errors.EmptyDataError):
            get_face_landmark_coords(csv_file, 1, 0)


def test_get_face_landmark_coords_malformed_csv():
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "malformed.csv"

        with open(csv_file, "w") as f:
            f.write("1,100.0,150.0\n2,200.0,250.0\n")

        with pytest.raises((KeyError, IndexError)):
            get_face_landmark_coords(csv_file, 1, 0)


@pytest.mark.parametrize(
    ("frame_number", "landmark_idx", "expected_x", "expected_y"),
    [
        (1, 0, 100.0, 150.0),
        (2, 0, 200.0, 250.0),
        (3, 0, 300.0, 350.0),
    ],
)
def test_get_face_landmark_coords_duplicate_frames(frame_number, landmark_idx, expected_x, expected_y):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "landmarks.csv"

        data = {
            "frame_number": [1, 1, 2, 2, 2, 3, 3, 3],
            "0_x": [100.0, 111.0, 200.0, 220.0, 230.0, 300.0, 310.0, 320.0],
            "0_y": [150.0, 161.0, 250.0, 261.0, 270.0, 350.0, 360.0, 370.0],
        }

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        x, y = get_face_landmark_coords(csv_file, frame_number, landmark_idx)

        assert x == expected_x
        assert y == expected_y


def test_get_face_landmark_coords_float_coordinates():
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "landmarks.csv"

        data = {"frame_number": [1], "0_x": [123.456789], "0_y": [987.654321]}

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        x, y = get_face_landmark_coords(csv_file, 1, 0)

        assert abs(x - 123.456789) < 1e-6
        assert abs(y - 987.654321) < 1e-6
