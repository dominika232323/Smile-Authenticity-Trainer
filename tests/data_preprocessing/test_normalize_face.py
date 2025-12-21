import csv
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from data_preprocessing.normalize_face import (
    calculate_distance_between_points,
    calculate_center_of_landmarks,
    calculate_left_eye_center,
    calculate_right_eye_center,
    calculate_angle_between_points,
    calculate_center_between_points,
    calculate_image_scaling_factor,
    calculate_desired_eye_distance,
    normalize_face,
)
from unittest.mock import patch

import math


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

    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
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

    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
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

    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
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

    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
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

    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
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

    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
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

    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
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

    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
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

    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
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

    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
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

    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
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

    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
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


class TestCalculateAngleBetweenPoints:
    @pytest.mark.parametrize(
        ("right_point", "left_point", "expected"),
        [
            ((1.0, 0.0), (0.0, 0.0), -180.0),
            ((0.0, 0.0), (1.0, 0.0), 0.0),
            ((0.0, 1.0), (0.0, 0.0), -90.0),
            ((0.0, 0.0), (0.0, 1.0), -270.0),
            ((1.0, 1.0), (0.0, 0.0), -135.0),
            ((1.0, -1.0), (0.0, 0.0), -225.0),
            ((0.0, 0.0), (1.0, 1.0), -315.0),
            ((0.0, 0.0), (1.0, -1.0), -45.0),
            ((math.sqrt(3), 1.0), (0.0, 0.0), -150.0),
            ((1.0, math.sqrt(3)), (0.0, 0.0), -120.0),
            ((-1.0, math.sqrt(3)), (0.0, 0.0), -60.0),
            ((-math.sqrt(3), 1.0), (0.0, 0.0), -30.0),
            ((-math.sqrt(3), -1.0), (0.0, 0.0), -330.0),
            ((-1.0, -math.sqrt(3)), (0.0, 0.0), -300.0),
            ((1.0, -math.sqrt(3)), (0.0, 0.0), -240.0),
            ((math.sqrt(3), -1.0), (0.0, 0.0), -210.0),
        ],
    )
    def test_angles(self, right_point, left_point, expected):
        result = calculate_angle_between_points(right_point, left_point)
        assert result == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize(
        ("point1", "point2"),
        [
            ((1.0, 1.0), (0.0, 0.0)),
            ((5.0, 3.0), (2.0, 1.0)),
            ((-3.0, -4.0), (-1.0, -2.0)),
            ((100.0, 200.0), (150.0, 250.0)),
        ],
    )
    def test_angle_antisymmetry(self, point1, point2):
        angle1 = calculate_angle_between_points(point1, point2)
        angle2 = calculate_angle_between_points(point2, point1)

        diff = abs(angle1 - angle2)

        assert diff == pytest.approx(180.0, abs=1e-10) or diff == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize(
        "point",
        [([1.0, 1.0]), ([5.0, 3.0]), ([-3.0, -4.0]), ([100.0, 200.0]), ([5.0, 7.0])],
    )
    def test_same_points(self, point):
        result = calculate_angle_between_points(point, point)

        assert result == -180.0

    @pytest.mark.parametrize(
        ("right_point", "left_point", "expected"),
        [
            ((1000.0, 2000.0), (500.0, 1000.0), -116.565051177077),
            ((5000.0, 3000.0), (2000.0, 1000.0), -146.309932474020),
            ((0.001, 0.002), (0.0, 0.0), -116.565051177077),
            ((0.0001, 0.0002), (0.0, 0.0), -116.565051177077),
        ],
    )
    def test_extreme_coordinates(self, right_point, left_point, expected):
        result = calculate_angle_between_points(right_point, left_point)
        assert result == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        ("right_point", "left_point", "expected"),
        [
            ((-1.0, -1.0), (-2.0, -2.0), -135.0),
            ((-3.0, 4.0), (-1.0, 2.0), -45.0),
            ((2.0, -3.0), (4.0, -1.0), -315.0),
            ((-2.0, -3.0), (-4.0, -1.0), -225.0),
        ],
    )
    def test_negative_coordinates(self, right_point, left_point, expected):
        result = calculate_angle_between_points(right_point, left_point)
        assert result == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        ("right_point", "left_point"),
        [
            ((1.0, 0.0), (0.0, 0.0)),
            ((0.0, 1.0), (0.0, 0.0)),
            ((-1.0, 0.0), (0.0, 0.0)),
            ((0.0, -1.0), (0.0, 0.0)),
            ((5.0, 5.0), (3.0, 3.0)),
            ((-2.0, -2.0), (-4.0, -4.0)),
        ],
    )
    def test_angle_range(self, right_point, left_point):
        result = calculate_angle_between_points(right_point, left_point)

        assert -360.0 <= result <= 0.0

    def test_return_type(self):
        result = calculate_angle_between_points((1.0, 2.0), (3.0, 4.0))

        assert isinstance(result, (int, float))

    @pytest.mark.parametrize(
        ("right_point", "left_point"),
        [
            ((1.5, 2.7), (3.2, 4.8)),
            ((10.333, 20.666), (15.777, 25.111)),
            ((-5.123, -3.456), (-7.890, -1.234)),
            ((0.1, 0.2), (0.3, 0.4)),
        ],
    )
    def test_floating_point_precision(self, right_point, left_point):
        result = calculate_angle_between_points(right_point, left_point)

        assert -360.0 <= result <= 0.0
        assert isinstance(result, (int, float))
        assert not math.isnan(result)
        assert not math.isinf(result)

    @pytest.mark.parametrize(
        ("scale_factor", "expected"),
        [
            (1.0, -135.0),
            (10.0, -135.0),
            (100.0, -135.0),
            (0.1, -135.0),
            (0.01, -135.0),
        ],
    )
    def test_scale_invariance(self, scale_factor, expected):
        base_right = (1.0, 1.0)
        base_left = (0.0, 0.0)

        scaled_right = (base_right[0] * scale_factor, base_right[1] * scale_factor)
        scaled_left = (base_left[0] * scale_factor, base_left[1] * scale_factor)

        result = calculate_angle_between_points(scaled_right, scaled_left)

        assert result == pytest.approx(expected, abs=1e-10)

    def test_translation_invariance(self):
        offset = (100.0, 200.0)

        original_right = (3.0, 4.0)
        original_left = (1.0, 2.0)

        translated_right = (original_right[0] + offset[0], original_right[1] + offset[1])
        translated_left = (original_left[0] + offset[0], original_left[1] + offset[1])

        original_angle = calculate_angle_between_points(original_right, original_left)
        translated_angle = calculate_angle_between_points(translated_right, translated_left)

        assert original_angle == pytest.approx(translated_angle, abs=1e-10)


class TestCalculateCenterBetweenPoints:
    @pytest.mark.parametrize(
        ("right", "left", "expected"),
        [
            ((0.0, 0.0), (2.0, 0.0), (1.0, 0.0)),
            ((0.0, 0.0), (0.0, 2.0), (0.0, 1.0)),
            ((1.0, 1.0), (3.0, 5.0), (2.0, 3.0)),
            ((-1.0, -1.0), (1.0, 1.0), (0.0, 0.0)),
            ((-5.0, 10.0), (15.0, -10.0), (5.0, 0.0)),
            ((1000.0, 2000.0), (1002.0, 2004.0), (1001.0, 2002.0)),
            ((-1.0, -2.0), (-3.0, -4.0), (-2.0, -3.0)),
            ((-10.5, 5.5), (-2.5, -6.5), (-6.5, -0.5)),
        ],
    )
    def test_basic_midpoints(self, right, left, expected):
        result = calculate_center_between_points(right, left)

        assert result == expected

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(coord, (int, float)) for coord in result)

    @pytest.mark.parametrize(
        "point",
        [
            (0.0, 0.0),
            (1.0, 1.0),
            (-3.5, 4.2),
            (1000.0, -2000.0),
        ],
    )
    def test_same_points(self, point):
        result = calculate_center_between_points(point, point)

        assert result == point

    @pytest.mark.parametrize(
        ("right", "left"),
        [
            ((0.0, 0.0), (2.0, 4.0)),
            ((-5.0, -5.0), (5.0, 5.0)),
            ((10.0, 20.0), (30.0, 40.0)),
            ((-10.0, 20.0), (30.0, -40.0)),
        ],
    )
    def test_commutativity(self, right, left):
        center1 = calculate_center_between_points(right, left)
        center2 = calculate_center_between_points(left, right)

        assert center1 == center2

    @pytest.mark.parametrize(
        ("scale_factor",),
        [
            (0.1,),
            (1.0,),
            (10.0,),
            (100.0,),
        ],
    )
    def test_scale_behavior(self, scale_factor):
        base_right = (1.0, 2.0)
        base_left = (3.0, 4.0)

        scaled_right = (base_right[0] * scale_factor, base_right[1] * scale_factor)
        scaled_left = (base_left[0] * scale_factor, base_left[1] * scale_factor)

        base_center = calculate_center_between_points(base_right, base_left)
        scaled_center = calculate_center_between_points(scaled_right, scaled_left)

        assert scaled_center == (
            base_center[0] * scale_factor,
            base_center[1] * scale_factor,
        )


class TestCalculateImageScalingFactor:
    @pytest.mark.parametrize(
        ("right_eye_center", "left_eye_center", "eye_relative_x", "desired_image_width", "expected"),
        [
            ((1.0, 0.0), (3.0, 0.0), 0.25, 4.0, 1.0),
            ((0.0, 0.0), (10.0, 0.0), 0.25, 20.0, 1.0),
            ((0.0, 0.0), (4.0, 0.0), 0.1, 10.0, 2.0),
            ((0.0, 0.0), (8.0, 0.0), 0.1, 10.0, 1.0),
        ],
    )
    def test_basic_scaling_factor(
        self, right_eye_center, left_eye_center, eye_relative_x, desired_image_width, expected
    ):
        result = calculate_image_scaling_factor(right_eye_center, left_eye_center, eye_relative_x, desired_image_width)

        assert result == pytest.approx(expected)
        assert isinstance(result, (int, float))

    @pytest.mark.parametrize(
        ("right_eye_center", "left_eye_center"),
        [
            ((0.0, 0.0), (2.0, 0.0)),
            ((-1.0, 1.0), (3.0, 4.0)),
            ((100.0, 200.0), (150.0, 250.0)),
            ((-10.0, -20.0), (-15.0, -25.0)),
        ],
    )
    def test_symmetry_in_eye_order(self, right_eye_center, left_eye_center):
        eye_relative_x = 0.2
        desired_image_width = 100.0

        scale1 = calculate_image_scaling_factor(
            right_eye_center,
            left_eye_center,
            eye_relative_x,
            desired_image_width,
        )
        scale2 = calculate_image_scaling_factor(
            left_eye_center,
            right_eye_center,
            eye_relative_x,
            desired_image_width,
        )

        assert scale1 == pytest.approx(scale2)

    @pytest.mark.parametrize(
        ("eye_relative_x", "desired_image_width"),
        [
            (0.1, 10.0),
            (0.25, 40.0),
            (0.4, 100.0),
        ],
    )
    def test_scale_changes_with_eye_distance(self, eye_relative_x, desired_image_width):
        base_right = (0.0, 0.0)
        left_near = (2.0, 0.0)
        left_far = (4.0, 0.0)

        scale_near = calculate_image_scaling_factor(base_right, left_near, eye_relative_x, desired_image_width)
        scale_far = calculate_image_scaling_factor(base_right, left_far, eye_relative_x, desired_image_width)

        assert scale_far < scale_near

    def test_scale_inverse_proportional_to_distance(self):
        eye_relative_x = 0.2
        desired_image_width = 100.0

        right1, left1 = (0.0, 0.0), (5.0, 0.0)
        right2, left2 = (0.0, 0.0), (10.0, 0.0)

        scale1 = calculate_image_scaling_factor(right1, left1, eye_relative_x, desired_image_width)
        scale2 = calculate_image_scaling_factor(right2, left2, eye_relative_x, desired_image_width)

        assert scale2 == pytest.approx(scale1 / 2.0)

    def test_zero_distance_between_eyes_raises(self):
        right_eye_center = (1.0, 1.0)
        left_eye_center = (1.0, 1.0)

        with pytest.raises(ZeroDivisionError):
            calculate_image_scaling_factor(right_eye_center, left_eye_center, 0.25, 40.0)


class TestCalculateDesiredEyeDistance:
    @pytest.mark.parametrize(
        ("eye_relative_x", "desired_image_width", "expected"),
        [
            (0.25, 4.0, 2.0),
            (0.1, 10.0, 8.0),
            (0.2, 100.0, 60.0),
            (0.49, 200.0, 4.0),
        ],
    )
    def test_valid_values(self, eye_relative_x, desired_image_width, expected):
        result = calculate_desired_eye_distance(eye_relative_x, desired_image_width)

        assert result == pytest.approx(expected)
        assert isinstance(result, (int, float))

    @pytest.mark.parametrize("eye_relative_x", [0.0, 0.5, -0.1, 0.51, 1.0])
    def test_invalid_eye_relative_x_raises(self, eye_relative_x):
        with pytest.raises(ValueError):
            calculate_desired_eye_distance(eye_relative_x, 100.0)

    @pytest.mark.parametrize("desired_image_width", [0.0, -1.0, -100.0])
    def test_non_positive_width_raises(self, desired_image_width):
        with pytest.raises(ValueError):
            calculate_desired_eye_distance(0.25, desired_image_width)

    @pytest.mark.parametrize(
        "scale_factor",
        [0.1, 0.5, 1.0, 2.0],
    )
    def test_linear_scaling_with_width(self, scale_factor):
        eye_relative_x = 0.2
        base_width = 100.0
        base_distance = calculate_desired_eye_distance(eye_relative_x, base_width)

        scaled_distance = calculate_desired_eye_distance(eye_relative_x, base_width * scale_factor)

        assert scaled_distance == pytest.approx(base_distance * scale_factor)

    def test_small_eye_relative_x_near_zero(self):
        eye_relative_x = 1e-6
        width = 100.0

        result = calculate_desired_eye_distance(eye_relative_x, width)
        expected = (1 - 2 * eye_relative_x) * width

        assert result == pytest.approx(expected)


class TestNormalizeFace:
    def create_landmarks_csv(self, landmark_data):
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)

        fieldnames = [
            "frame_number",
            "0_x",
            "0_y",
            "1_x",
            "1_y",
            "2_x",
            "2_y",
            "3_x",
            "3_y",
            "4_x",
            "4_y",
            "5_x",
            "5_y",
            "6_x",
            "6_y",
            "7_x",
            "7_y",
        ]
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in landmark_data:
            writer.writerow(row)

        temp_file.close()

        return Path(temp_file.name)

    def create_test_frame(self, width=100, height=100):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        frame[10:20, 10:20] = [255, 0, 0]
        frame[80:90, 80:90] = [0, 255, 0]

        cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)

        return frame

    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
    def test_normalize_face_basic_functionality(
        self, mock_right_upper, mock_right_lower, mock_left_upper, mock_left_lower
    ):
        mock_left_lower.return_value = [0, 1]
        mock_left_upper.return_value = [2, 3]
        mock_right_lower.return_value = [4, 5]
        mock_right_upper.return_value = [6, 7]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 30.0,
                "0_y": 45.0,
                "1_x": 34.0,
                "1_y": 47.0,
                "2_x": 32.0,
                "2_y": 43.0,
                "3_x": 36.0,
                "3_y": 45.0,
                "4_x": 60.0,
                "4_y": 45.0,
                "5_x": 64.0,
                "5_y": 47.0,
                "6_x": 62.0,
                "6_y": 43.0,
                "7_x": 66.0,
                "7_y": 45.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)
        frame = self.create_test_frame(100, 100)

        try:
            result = normalize_face(
                frame=frame,
                landmarks_file_path=landmarks_file,
                frame_number=1,
                eye_relatives=(0.25, 0.35),
                desired_image_size=(80, 80),
            )

            assert isinstance(result, np.ndarray)
            assert result.shape == (80, 80, 3)
            assert result.dtype == np.uint8

        finally:
            landmarks_file.unlink()

    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
    def test_normalize_face_different_sizes(self, mock_right_upper, mock_right_lower, mock_left_upper, mock_left_lower):
        mock_left_lower.return_value = [0]
        mock_left_upper.return_value = [1]
        mock_right_lower.return_value = [2]
        mock_right_upper.return_value = [3]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 25.0,
                "0_y": 40.0,
                "1_x": 35.0,
                "1_y": 40.0,
                "2_x": 65.0,
                "2_y": 40.0,
                "3_x": 75.0,
                "3_y": 40.0,
                "4_x": 0.0,
                "4_y": 0.0,
                "5_x": 0.0,
                "5_y": 0.0,
                "6_x": 0.0,
                "6_y": 0.0,
                "7_x": 0.0,
                "7_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)
        frame = self.create_test_frame(100, 100)

        test_sizes = [(64, 64), (128, 128), (96, 96), (112, 112)]

        try:
            for width, height in test_sizes:
                result = normalize_face(
                    frame=frame,
                    landmarks_file_path=landmarks_file,
                    frame_number=1,
                    eye_relatives=(0.25, 0.35),
                    desired_image_size=(width, height),
                )

                assert result.shape == (height, width, 3)
                assert result.dtype == np.uint8

        finally:
            landmarks_file.unlink()

    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
    def test_normalize_face_different_eye_relatives(
        self, mock_right_upper, mock_right_lower, mock_left_upper, mock_left_lower
    ):
        mock_left_lower.return_value = [0]
        mock_left_upper.return_value = [1]
        mock_right_lower.return_value = [2]
        mock_right_upper.return_value = [3]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 30.0,
                "0_y": 40.0,
                "1_x": 30.0,
                "1_y": 40.0,
                "2_x": 70.0,
                "2_y": 40.0,
                "3_x": 70.0,
                "3_y": 40.0,
                "4_x": 0.0,
                "4_y": 0.0,
                "5_x": 0.0,
                "5_y": 0.0,
                "6_x": 0.0,
                "6_y": 0.0,
                "7_x": 0.0,
                "7_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)
        frame = self.create_test_frame(100, 100)

        eye_relatives_tests = [(0.1, 0.2), (0.25, 0.35), (0.3, 0.4), (0.4, 0.5)]

        try:
            for eye_rel_x, eye_rel_y in eye_relatives_tests:
                result = normalize_face(
                    frame=frame,
                    landmarks_file_path=landmarks_file,
                    frame_number=1,
                    eye_relatives=(eye_rel_x, eye_rel_y),
                    desired_image_size=(80, 80),
                )

                assert result.shape == (80, 80, 3)
                assert result.dtype == np.uint8

        finally:
            landmarks_file.unlink()

    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
    def test_normalize_face_rotated_eyes(self, mock_right_upper, mock_right_lower, mock_left_upper, mock_left_lower):
        mock_left_lower.return_value = [0]
        mock_left_upper.return_value = [1]
        mock_right_lower.return_value = [2]
        mock_right_upper.return_value = [3]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 30.0,
                "0_y": 50.0,
                "1_x": 30.0,
                "1_y": 50.0,
                "2_x": 70.0,
                "2_y": 30.0,
                "3_x": 70.0,
                "3_y": 30.0,
                "4_x": 0.0,
                "4_y": 0.0,
                "5_x": 0.0,
                "5_y": 0.0,
                "6_x": 0.0,
                "6_y": 0.0,
                "7_x": 0.0,
                "7_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)
        frame = self.create_test_frame(100, 100)

        try:
            result = normalize_face(
                frame=frame,
                landmarks_file_path=landmarks_file,
                frame_number=1,
                eye_relatives=(0.25, 0.35),
                desired_image_size=(80, 80),
            )

            assert result.shape == (80, 80, 3)
            assert result.dtype == np.uint8

        finally:
            landmarks_file.unlink()

    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
    def test_normalize_face_multiple_frames(self, mock_right_upper, mock_right_lower, mock_left_upper, mock_left_lower):
        mock_left_lower.return_value = [0]
        mock_left_upper.return_value = [1]
        mock_right_lower.return_value = [2]
        mock_right_upper.return_value = [3]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 30.0,
                "0_y": 40.0,
                "1_x": 30.0,
                "1_y": 40.0,
                "2_x": 70.0,
                "2_y": 40.0,
                "3_x": 70.0,
                "3_y": 40.0,
                "4_x": 0.0,
                "4_y": 0.0,
                "5_x": 0.0,
                "5_y": 0.0,
                "6_x": 0.0,
                "6_y": 0.0,
                "7_x": 0.0,
                "7_y": 0.0,
            },
            {
                "frame_number": 2,
                "0_x": 25.0,
                "0_y": 35.0,
                "1_x": 25.0,
                "1_y": 35.0,
                "2_x": 75.0,
                "2_y": 35.0,
                "3_x": 75.0,
                "3_y": 35.0,
                "4_x": 0.0,
                "4_y": 0.0,
                "5_x": 0.0,
                "5_y": 0.0,
                "6_x": 0.0,
                "6_y": 0.0,
                "7_x": 0.0,
                "7_y": 0.0,
            },
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)
        frame = self.create_test_frame(100, 100)

        try:
            result1 = normalize_face(
                frame=frame,
                landmarks_file_path=landmarks_file,
                frame_number=1,
                eye_relatives=(0.25, 0.35),
                desired_image_size=(80, 80),
            )

            result2 = normalize_face(
                frame=frame,
                landmarks_file_path=landmarks_file,
                frame_number=2,
                eye_relatives=(0.25, 0.35),
                desired_image_size=(80, 80),
            )

            assert result1.shape == (80, 80, 3)
            assert result2.shape == (80, 80, 3)
            assert result1.dtype == np.uint8
            assert result2.dtype == np.uint8

            assert not np.array_equal(result1, result2)

        finally:
            landmarks_file.unlink()

    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
    def test_normalize_face_grayscale_input(self, mock_right_upper, mock_right_lower, mock_left_upper, mock_left_lower):
        mock_left_lower.return_value = [0]
        mock_left_upper.return_value = [1]
        mock_right_lower.return_value = [2]
        mock_right_upper.return_value = [3]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 30.0,
                "0_y": 40.0,
                "1_x": 30.0,
                "1_y": 40.0,
                "2_x": 70.0,
                "2_y": 40.0,
                "3_x": 70.0,
                "3_y": 40.0,
                "4_x": 0.0,
                "4_y": 0.0,
                "5_x": 0.0,
                "5_y": 0.0,
                "6_x": 0.0,
                "6_y": 0.0,
                "7_x": 0.0,
                "7_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)

        gray_frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        try:
            result = normalize_face(
                frame=gray_frame,
                landmarks_file_path=landmarks_file,
                frame_number=1,
                eye_relatives=(0.25, 0.35),
                desired_image_size=(80, 80),
            )

            assert isinstance(result, np.ndarray)
            assert result.shape == (80, 80)
            assert result.dtype == np.uint8

        finally:
            landmarks_file.unlink()

    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
    def test_normalize_face_edge_cases_small_eye_distance(
        self, mock_right_upper, mock_right_lower, mock_left_upper, mock_left_lower
    ):
        mock_left_lower.return_value = [0]
        mock_left_upper.return_value = [1]
        mock_right_lower.return_value = [2]
        mock_right_upper.return_value = [3]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 49.0,
                "0_y": 50.0,
                "1_x": 49.0,
                "1_y": 50.0,
                "2_x": 51.0,
                "2_y": 50.0,
                "3_x": 51.0,
                "3_y": 50.0,
                "4_x": 0.0,
                "4_y": 0.0,
                "5_x": 0.0,
                "5_y": 0.0,
                "6_x": 0.0,
                "6_y": 0.0,
                "7_x": 0.0,
                "7_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)
        frame = self.create_test_frame(100, 100)

        try:
            result = normalize_face(
                frame=frame,
                landmarks_file_path=landmarks_file,
                frame_number=1,
                eye_relatives=(0.25, 0.35),
                desired_image_size=(80, 80),
            )

            assert result.shape == (80, 80, 3)
            assert result.dtype == np.uint8

        finally:
            landmarks_file.unlink()

    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
    def test_normalize_face_identical_eye_positions_raises_error(
        self, mock_right_upper, mock_right_lower, mock_left_upper, mock_left_lower
    ):
        mock_left_lower.return_value = [0]
        mock_left_upper.return_value = [1]
        mock_right_lower.return_value = [2]
        mock_right_upper.return_value = [3]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 50.0,
                "0_y": 50.0,
                "1_x": 50.0,
                "1_y": 50.0,
                "2_x": 50.0,
                "2_y": 50.0,
                "3_x": 50.0,
                "3_y": 50.0,
                "4_x": 0.0,
                "4_y": 0.0,
                "5_x": 0.0,
                "5_y": 0.0,
                "6_x": 0.0,
                "6_y": 0.0,
                "7_x": 0.0,
                "7_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)
        frame = self.create_test_frame(100, 100)

        try:
            with pytest.raises(ZeroDivisionError):
                normalize_face(
                    frame=frame,
                    landmarks_file_path=landmarks_file,
                    frame_number=1,
                    eye_relatives=(0.25, 0.35),
                    desired_image_size=(80, 80),
                )
        finally:
            landmarks_file.unlink()

    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.left_eye_upper_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_lower_0")
    @patch("data_preprocessing.normalize_face.FaceLandmarks.right_eye_upper_0")
    def test_normalize_face_extreme_coordinates(
        self, mock_right_upper, mock_right_lower, mock_left_upper, mock_left_lower
    ):
        mock_left_lower.return_value = [0]
        mock_left_upper.return_value = [1]
        mock_right_lower.return_value = [2]
        mock_right_upper.return_value = [3]

        landmark_data = [
            {
                "frame_number": 1,
                "0_x": 10.0,
                "0_y": 10.0,
                "1_x": 10.0,
                "1_y": 10.0,
                "2_x": 90.0,
                "2_y": 90.0,
                "3_x": 90.0,
                "3_y": 90.0,
                "4_x": 0.0,
                "4_y": 0.0,
                "5_x": 0.0,
                "5_y": 0.0,
                "6_x": 0.0,
                "6_y": 0.0,
                "7_x": 0.0,
                "7_y": 0.0,
            }
        ]

        landmarks_file = self.create_landmarks_csv(landmark_data)
        frame = self.create_test_frame(100, 100)

        try:
            result = normalize_face(
                frame=frame,
                landmarks_file_path=landmarks_file,
                frame_number=1,
                eye_relatives=(0.25, 0.35),
                desired_image_size=(80, 80),
            )

            assert result.shape == (80, 80, 3)
            assert result.dtype == np.uint8

        finally:
            landmarks_file.unlink()

    def test_normalize_face_input_types(self):
        frame_np = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        assert isinstance(frame_np, (np.ndarray, type(cv2.imread)))

    @pytest.fixture(autouse=True)
    def cleanup_temp_files(self):
        yield
