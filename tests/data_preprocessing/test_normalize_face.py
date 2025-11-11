import csv
import tempfile
from pathlib import Path

import pytest

from ai.data_preprocessing.normalize_face import (
    calculate_distance_between_points,
    calculate_center_of_landmarks,
    calculate_left_eye_center,
    calculate_right_eye_center,
)
from unittest.mock import patch


class TestCalculateDistanceBetweenPoints:
    @pytest.mark.parametrize(
        ("point1", "point2", "expected"),
        [
            ((0.0, 0.0), (1.0, 0.0), 1.0),
            ((0.0, 0.0), (0.0, 1.0), 1.0),
            ((1.0, 1.0), (4.0, 5.0), 5.0),
            ((-1.0, -1.0), (2.0, 3.0), 5.0),
            ((1.5, 2.5), (4.5, 6.5), 5.0),
            ((1000.0, 2000.0), (1003.0, 2004.0), 5.0),
        ],
    )
    def test_distance(self, point1, point2, expected):
        result = calculate_distance_between_points(point1, point2)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "point",
        [(1.0, 1.0), (-4.2, -3.2), (-1.0, -1.5), (3.0, 4.0), (2.54, 6.0), (16.44, 33.81)],
    )
    def test_distance_between_same_points(self, point):
        result = calculate_distance_between_points(point, point)

        assert result == 0

    @pytest.mark.parametrize(
        ("point", "expected"),
        [
            ((1.0, 1.0), 1.41421356237),
            ((-4.2, -3.2), 5.28015151298),
            ((3.0, 4.0), 5.0),
            ((8.0, 6.0), 10.0),
            ((5.0, 12.0), 13.0),
        ],
    )
    def test_distance_between_origin_and_point(self, point, expected):
        origin = (0.0, 0.0)
        result = calculate_distance_between_points(point, origin)

        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("point1", "point2", "expected"),
        [
            ((1.0, 5.0), (6.0, 5.0), 5.0),
            ((0.0, 1.0), (1.0, 1.0), 1.0),
            ((1.0, 1.0), (4.0, 1.0), 3.0),
            ((-1.0, -1.0), (2.0, -1.0), 3.0),
            ((-1000.0, 2000.0), (1003.0, 2000.0), 2003.0),
        ],
    )
    def test_distance_horizontal_points(self, point1, point2, expected):
        result = calculate_distance_between_points(point1, point2)

        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("point1", "point2", "expected"),
        [
            ((3.0, 2.0), (3.0, 8.0), 6.0),
            ((0.0, 0.0), (0.0, 1.0), 1.0),
            ((4.0, 1.0), (4.0, 5.0), 4.0),
            ((-1.0, -1.0), (-1.0, 4.0), 5.0),
            ((1000.0, -2000.0), (1000.0, 2004.0), 4004.0),
        ],
    )
    def test_distance_vertical_points(self, point1, point2, expected):
        result = calculate_distance_between_points(point1, point2)

        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("point1", "point2"),
        [
            ((0.0, 0.0), (1.0, 0.0)),
            ((0.0, 0.0), (0.0, 1.0)),
            ((1.0, 1.0), (4.0, 5.0)),
            ((-1.0, -1.0), (2.0, 3.0)),
            ((1.5, 2.5), (4.5, 6.5)),
            ((1000.0, 2000.0), (1003.0, 2004.0)),
        ],
    )
    def test_distance_commutative_property(self, point1, point2):
        distance1 = calculate_distance_between_points(point1, point2)
        distance2 = calculate_distance_between_points(point2, point1)

        assert distance1 == pytest.approx(distance2)


class TestCalculateCenterOfLandmarks:
    def create_landmarks_csv(self, landmark_data):
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)

        fieldnames = ["frame_number", "0_x", "0_y", "1_x", "1_y", "2_x", "2_y", "3_x", "3_y"]
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in landmark_data:
            writer.writerow(row)

        temp_file.close()

        return Path(temp_file.name)

    @pytest.mark.parametrize(
        ("landmarks", "frame_number", "expected"),
        [
            ([0], 1, (10.0, 20.0)),
            ([0, 1], 2, (5.0, 5.0)),
            ([0, 1, 2, 3], 3, (3.0, 1.5)),
            ([0, 1], 4, (0.0, 0.0)),
            ([0, 1, 2], 5, (3.5, 4.5)),
            ([0, 1], 6, (2000.0, 3000.0)),
            ([0, 1], 7, (1.0, 1.0)),
            ([0, 1], 8, (15.0, 15.0)),
            ([0, 1, 2, 3], 9, (103.7, 150.225)),
        ],
    )
    def test_single_landmark_center(self, landmarks, frame_number, expected):
        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 10.0,
                "0_y": 20.0,
                "1_x": 0.0,
                "1_y": 0.0,
                "2_x": 0.0,
                "2_y": 0.0,
                "3_x": 0.0,
                "3_y": 0.0,
            },
            {
                "frame_number": 2,
                "0_x": 0.0,
                "0_y": 0.0,
                "1_x": 10.0,
                "1_y": 10.0,
                "2_x": 0.0,
                "2_y": 0.0,
                "3_x": 0.0,
                "3_y": 0.0,
            },
            {
                "frame_number": 3,
                "0_x": 0.0,
                "0_y": 0.0,
                "1_x": 6.0,
                "1_y": 0.0,
                "2_x": 3.0,
                "2_y": 6.0,
                "3_x": 3.0,
                "3_y": 0.0,
            },
            {
                "frame_number": 4,
                "0_x": -5.0,
                "0_y": -10.0,
                "1_x": 5.0,
                "1_y": 10.0,
                "2_x": 0.0,
                "2_y": 0.0,
                "3_x": 0.0,
                "3_y": 0.0,
            },
            {
                "frame_number": 5,
                "0_x": 1.5,
                "0_y": 2.5,
                "1_x": 3.5,
                "1_y": 4.5,
                "2_x": 5.5,
                "2_y": 6.5,
                "3_x": 0.0,
                "3_y": 0.0,
            },
            {
                "frame_number": 6,
                "0_x": 1000.0,
                "0_y": 2000.0,
                "1_x": 3000.0,
                "1_y": 4000.0,
                "2_x": 0.0,
                "2_y": 0.0,
                "3_x": 0.0,
                "3_y": 0.0,
            },
            {
                "frame_number": 7,
                "0_x": 0.0,
                "0_y": 0.0,
                "1_x": 2.0,
                "1_y": 2.0,
                "2_x": 0.0,
                "2_y": 0.0,
                "3_x": 0.0,
                "3_y": 0.0,
            },
            {
                "frame_number": 8,
                "0_x": 10.0,
                "0_y": 10.0,
                "1_x": 20.0,
                "1_y": 20.0,
                "2_x": 0.0,
                "2_y": 0.0,
                "3_x": 0.0,
                "3_y": 0.0,
            },
            {
                "frame_number": 9,
                "0_x": 100.5,
                "0_y": 150.2,
                "1_x": 110.3,
                "1_y": 148.7,
                "2_x": 105.8,
                "2_y": 152.1,
                "3_x": 98.2,
                "3_y": 149.9,
            },
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)

        try:
            result = calculate_center_of_landmarks(landmarks_file, frame_number, landmarks)

            assert result == expected
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert all(isinstance(coord, (int, float)) for coord in result)
        finally:
            landmarks_file.unlink()

    @pytest.mark.parametrize(
        ("landmarks_1", "landmarks_2", "frame_number", "expected"),
        [
            ([0, 1], [1, 0], 4, (0.0, 0.0)),
            ([0, 1, 2], [2, 0, 1], 5, (3.5, 4.5)),
            ([0, 1], [1, 0], 6, (2000.0, 3000.0)),
            ([0, 1, 2, 3], [2, 3, 0, 1], 9, (103.7, 150.225)),
        ],
    )
    def test_landmark_order_independence(self, landmarks_1, landmarks_2, frame_number, expected):
        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 10.0,
                "0_y": 20.0,
                "1_x": 0.0,
                "1_y": 0.0,
                "2_x": 0.0,
                "2_y": 0.0,
                "3_x": 0.0,
                "3_y": 0.0,
            },
            {
                "frame_number": 4,
                "0_x": -5.0,
                "0_y": -10.0,
                "1_x": 5.0,
                "1_y": 10.0,
                "2_x": 0.0,
                "2_y": 0.0,
                "3_x": 0.0,
                "3_y": 0.0,
            },
            {
                "frame_number": 5,
                "0_x": 1.5,
                "0_y": 2.5,
                "1_x": 3.5,
                "1_y": 4.5,
                "2_x": 5.5,
                "2_y": 6.5,
                "3_x": 0.0,
                "3_y": 0.0,
            },
            {
                "frame_number": 6,
                "0_x": 1000.0,
                "0_y": 2000.0,
                "1_x": 3000.0,
                "1_y": 4000.0,
                "2_x": 0.0,
                "2_y": 0.0,
                "3_x": 0.0,
                "3_y": 0.0,
            },
            {
                "frame_number": 9,
                "0_x": 100.5,
                "0_y": 150.2,
                "1_x": 110.3,
                "1_y": 148.7,
                "2_x": 105.8,
                "2_y": 152.1,
                "3_x": 98.2,
                "3_y": 149.9,
            },
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)

        try:
            result_1 = calculate_center_of_landmarks(landmarks_file, frame_number, landmarks_1)
            result_2 = calculate_center_of_landmarks(landmarks_file, frame_number, landmarks_2)

            assert result_1 == pytest.approx(result_2)
            assert result_1 == expected
            assert result_2 == expected
        finally:
            landmarks_file.unlink()

    @pytest.mark.parametrize(
        ("landmarks", "frame_number", "expected"),
        [
            ([0, 0, 1], 10, (3.333333, 5.333333)),
            ([0, 1, 2, 1], 3, (3.75, 1.5)),
            ([0, 1, 0, 2], 4, (-1.25, -2.5)),
            ([0, 1, 2, 1, 0], 5, (3.1, 4.1)),
            ([0, 1, 2, 3, 0, 1, 2, 3], 9, (103.7, 150.225)),
        ],
    )
    def test_duplicate_landmarks_handled_correctly(self, landmarks, frame_number, expected):
        landmark_data = [
            {
                "frame_number": 3,
                "0_x": 0.0,
                "0_y": 0.0,
                "1_x": 6.0,
                "1_y": 0.0,
                "2_x": 3.0,
                "2_y": 6.0,
                "3_x": 3.0,
                "3_y": 0.0,
            },
            {
                "frame_number": 4,
                "0_x": -5.0,
                "0_y": -10.0,
                "1_x": 5.0,
                "1_y": 10.0,
                "2_x": 0.0,
                "2_y": 0.0,
                "3_x": 0.0,
                "3_y": 0.0,
            },
            {
                "frame_number": 5,
                "0_x": 1.5,
                "0_y": 2.5,
                "1_x": 3.5,
                "1_y": 4.5,
                "2_x": 5.5,
                "2_y": 6.5,
                "3_x": 0.0,
                "3_y": 0.0,
            },
            {
                "frame_number": 9,
                "0_x": 100.5,
                "0_y": 150.2,
                "1_x": 110.3,
                "1_y": 148.7,
                "2_x": 105.8,
                "2_y": 152.1,
                "3_x": 98.2,
                "3_y": 149.9,
            },
            {
                "frame_number": 10,
                "0_x": 2.0,
                "0_y": 4.0,
                "1_x": 6.0,
                "1_y": 8.0,
                "2_x": 0.0,
                "2_y": 0.0,
                "3_x": 0.0,
                "3_y": 0.0,
            },
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)

        try:
            result = calculate_center_of_landmarks(landmarks_file, frame_number, landmarks)

            assert result == pytest.approx(expected)
        finally:
            landmarks_file.unlink()

    @pytest.fixture(autouse=True)
    def cleanup_temp_files(self):
        yield


class TestCalculateLeftEyeCenter:
    def create_landmarks_csv(self, landmark_data):
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)

        fieldnames = ["frame_number", "0_x", "0_y", "1_x", "1_y", "2_x", "2_y", "3_x", "3_y", "4_x", "4_y"]
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in landmark_data:
            writer.writerow(row)

        temp_file.close()

        return Path(temp_file.name)

    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
    def test_calculate_left_eye_center_single_frame(self, mock_upper, mock_lower):
        mock_lower.return_value = [0, 1]
        mock_upper.return_value = [2, 3]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 10.0,
                "0_y": 20.0,
                "1_x": 14.0,
                "1_y": 22.0,
                "2_x": 12.0,
                "2_y": 18.0,
                "3_x": 16.0,
                "3_y": 24.0,
                "4_x": 0.0,
                "4_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)

        try:
            result = calculate_left_eye_center(landmarks_file, 1)
            expected = (13.0, 21.0)

            assert result == expected
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert all(isinstance(coord, (int, float)) for coord in result)
        finally:
            landmarks_file.unlink()

    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
    def test_calculate_left_eye_center_duplicate_landmarks(self, mock_upper, mock_lower):
        mock_lower.return_value = [0, 1, 0]
        mock_upper.return_value = [2, 1]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 10.0,
                "0_y": 20.0,
                "1_x": 14.0,
                "1_y": 22.0,
                "2_x": 12.0,
                "2_y": 18.0,
                "3_x": 0.0,
                "3_y": 0.0,
                "4_x": 0.0,
                "4_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)

        try:
            result = calculate_left_eye_center(landmarks_file, 1)
            expected = (12.0, 20.0)

            assert result == expected
        finally:
            landmarks_file.unlink()

    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
    def test_calculate_right_eye_center_multiple_frames(self, mock_upper, mock_lower):
        mock_lower.return_value = [0, 1]
        mock_upper.return_value = [2]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 10.0,
                "0_y": 20.0,
                "1_x": 14.0,
                "1_y": 22.0,
                "2_x": 12.0,
                "2_y": 18.0,
                "3_x": 0.0,
                "3_y": 0.0,
                "4_x": 0.0,
                "4_y": 0.0,
            },
            {
                "frame_number": 2,
                "0_x": 100.0,
                "0_y": 200.0,
                "1_x": 140.0,
                "1_y": 220.0,
                "2_x": 120.0,
                "2_y": 180.0,
                "3_x": 0.0,
                "3_y": 0.0,
                "4_x": 0.0,
                "4_y": 0.0,
            },
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)

        try:
            result_frame_1 = calculate_left_eye_center(landmarks_file, 1)
            result_frame_2 = calculate_left_eye_center(landmarks_file, 2)

            expected_frame_1 = (12.0, 20.0)
            expected_frame_2 = (120.0, 200.0)

            assert result_frame_1 == expected_frame_1
            assert result_frame_2 == expected_frame_2
        finally:
            landmarks_file.unlink()

    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
    def test_calculate_right_eye_center_single_landmark(self, mock_upper, mock_lower):
        mock_lower.return_value = [0]
        mock_upper.return_value = []  # Empty upper landmarks

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 50.0,
                "0_y": 60.0,
                "1_x": 0.0,
                "1_y": 0.0,
                "2_x": 0.0,
                "2_y": 0.0,
                "3_x": 0.0,
                "3_y": 0.0,
                "4_x": 0.0,
                "4_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)

        try:
            result = calculate_left_eye_center(landmarks_file, 1)
            expected = (50.0, 60.0)

            assert result == expected
        finally:
            landmarks_file.unlink()

    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
    def test_calculate_left_eye_center_negative_coordinates(self, mock_upper, mock_lower):
        mock_lower.return_value = [0, 1]
        mock_upper.return_value = [2, 3]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": -10.0,
                "0_y": -5.0,
                "1_x": -8.0,
                "1_y": -3.0,
                "2_x": -12.0,
                "2_y": -7.0,
                "3_x": -6.0,
                "3_y": -1.0,
                "4_x": 0.0,
                "4_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)

        try:
            result = calculate_left_eye_center(landmarks_file, 1)
            expected = (-9.0, -4.0)

            assert result == expected
        finally:
            landmarks_file.unlink()

    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
    def test_calculate_left_eye_center_floating_point_precision(self, mock_upper, mock_lower):
        mock_lower.return_value = [0, 1]
        mock_upper.return_value = [2]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 1.333333,
                "0_y": 2.666667,
                "1_x": 4.555555,
                "1_y": 5.888889,
                "2_x": 7.777777,
                "2_y": 8.111111,
                "3_x": 0.0,
                "3_y": 0.0,
                "4_x": 0.0,
                "4_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)

        try:
            result = calculate_left_eye_center(landmarks_file, 1)
            expected = (4.555555, 5.555556)

            assert result == pytest.approx(expected, abs=1e-6)
        finally:
            landmarks_file.unlink()


class TestCalculateRightEyeCenter:
    def create_landmarks_csv(self, landmark_data):
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)

        fieldnames = ["frame_number", "0_x", "0_y", "1_x", "1_y", "2_x", "2_y", "3_x", "3_y", "4_x", "4_y"]
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in landmark_data:
            writer.writerow(row)

        temp_file.close()

        return Path(temp_file.name)

    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
    def test_calculate_right_eye_center_single_frame(self, mock_upper, mock_lower):
        mock_lower.return_value = [0, 1]
        mock_upper.return_value = [2, 3]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 30.0,
                "0_y": 40.0,
                "1_x": 34.0,
                "1_y": 42.0,
                "2_x": 32.0,
                "2_y": 38.0,
                "3_x": 36.0,
                "3_y": 44.0,
                "4_x": 0.0,
                "4_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)

        try:
            result = calculate_right_eye_center(landmarks_file, 1)
            expected = (33.0, 41.0)

            assert result == expected
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert all(isinstance(coord, (int, float)) for coord in result)
        finally:
            landmarks_file.unlink()

    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
    def test_calculate_right_eye_center_duplicate_landmarks(self, mock_upper, mock_lower):
        mock_lower.return_value = [0, 1, 0]
        mock_upper.return_value = [2, 1]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 30.0,
                "0_y": 40.0,
                "1_x": 34.0,
                "1_y": 42.0,
                "2_x": 32.0,
                "2_y": 38.0,
                "3_x": 0.0,
                "3_y": 0.0,
                "4_x": 0.0,
                "4_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)

        try:
            result = calculate_right_eye_center(landmarks_file, 1)
            expected = (32.0, 40.0)

            assert result == expected
        finally:
            landmarks_file.unlink()

    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
    def test_calculate_right_eye_center_multiple_frames(self, mock_upper, mock_lower):
        mock_lower.return_value = [0, 1]
        mock_upper.return_value = [2]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 10.0,
                "0_y": 20.0,
                "1_x": 14.0,
                "1_y": 22.0,
                "2_x": 12.0,
                "2_y": 18.0,
                "3_x": 0.0,
                "3_y": 0.0,
                "4_x": 0.0,
                "4_y": 0.0,
            },
            {
                "frame_number": 2,
                "0_x": 100.0,
                "0_y": 200.0,
                "1_x": 140.0,
                "1_y": 220.0,
                "2_x": 120.0,
                "2_y": 180.0,
                "3_x": 0.0,
                "3_y": 0.0,
                "4_x": 0.0,
                "4_y": 0.0,
            },
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)

        try:
            result_frame_1 = calculate_right_eye_center(landmarks_file, 1)
            result_frame_2 = calculate_right_eye_center(landmarks_file, 2)

            expected_frame_1 = (12.0, 20.0)
            expected_frame_2 = (120.0, 200.0)

            assert result_frame_1 == expected_frame_1
            assert result_frame_2 == expected_frame_2
        finally:
            landmarks_file.unlink()

    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
    def test_calculate_right_eye_center_single_landmark(self, mock_upper, mock_lower):
        mock_lower.return_value = [0]
        mock_upper.return_value = []  # Empty upper landmarks

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 50.0,
                "0_y": 60.0,
                "1_x": 0.0,
                "1_y": 0.0,
                "2_x": 0.0,
                "2_y": 0.0,
                "3_x": 0.0,
                "3_y": 0.0,
                "4_x": 0.0,
                "4_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)

        try:
            result = calculate_right_eye_center(landmarks_file, 1)
            expected = (50.0, 60.0)

            assert result == expected
        finally:
            landmarks_file.unlink()

    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
    def test_calculate_left_eye_center_negative_coordinates(self, mock_upper, mock_lower):
        mock_lower.return_value = [0, 1]
        mock_upper.return_value = [2, 3]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": -10.0,
                "0_y": -5.0,
                "1_x": -8.0,
                "1_y": -3.0,
                "2_x": -12.0,
                "2_y": -7.0,
                "3_x": -6.0,
                "3_y": -1.0,
                "4_x": 0.0,
                "4_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)

        try:
            result = calculate_right_eye_center(landmarks_file, 1)
            expected = (-9.0, -4.0)

            assert result == expected
        finally:
            landmarks_file.unlink()

    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("ai.data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
    def test_calculate_left_eye_center_floating_point_precision(self, mock_upper, mock_lower):
        mock_lower.return_value = [0, 1]
        mock_upper.return_value = [2]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 1.333333,
                "0_y": 2.666667,
                "1_x": 4.555555,
                "1_y": 5.888889,
                "2_x": 7.777777,
                "2_y": 8.111111,
                "3_x": 0.0,
                "3_y": 0.0,
                "4_x": 0.0,
                "4_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)

        try:
            result = calculate_right_eye_center(landmarks_file, 1)
            expected = (4.555555, 5.555556)

            assert result == pytest.approx(expected, abs=1e-6)
        finally:
            landmarks_file.unlink()

    @pytest.fixture(autouse=True)
    def cleanup_temp_files(self):
        yield
