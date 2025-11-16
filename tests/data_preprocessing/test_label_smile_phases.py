import numpy as np
import pandas as pd
import pytest

from ai.data_preprocessing.label_smile_phases import calculate_radius, smooth_radius


class TestCalculateRadius:
    @pytest.mark.parametrize(
        ("left_x", "left_y", "right_x", "right_y", "expected"),
        [
            ([0.0, 10.0], [0.0, 10.0], [10.0, 20.0], [0.0, 10.0], [5.0, 5.0]),
            ([0.0], [0.0], [6.0], [8.0], [5.0]),
            ([5.0], [5.0], [5.0], [5.0], [0.0]),
            ([-10.0], [-5.0], [-2.0], [-5.0], [4.0]),
            ([0.0, 2.0, 4.0], [0.0, 0.0, 0.0], [8.0, 10.0, 12.0], [0.0, 0.0, 0.0], [4.0, 4.0, 4.0]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 4.0, 6.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]),
            ([1.0], [2.0], [4.0], [6.0], [2.5]),
        ],
    )
    def test_calculate_radius(self, left_x, left_y, right_x, right_y, expected):
        landmarks_df = pd.DataFrame(
            {
                "291_x": left_x,
                "291_y": left_y,
                "61_x": right_x,
                "61_y": right_y,
            }
        )

        result = calculate_radius(landmarks_df)
        expected = np.array(expected)

        np.testing.assert_array_almost_equal(result, expected)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_calculate_radius_empty_dataframe(self):
        landmarks_df = pd.DataFrame(
            {
                "291_x": [],
                "291_y": [],
                "61_x": [],
                "61_y": [],
            }
        )

        result = calculate_radius(landmarks_df)

        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_calculate_radius_precision(self):
        landmarks_df = pd.DataFrame(
            {
                "291_x": [0.123456789],
                "291_y": [0.987654321],
                "61_x": [1.234567891],
                "61_y": [1.987654321],
            }
        )

        result = calculate_radius(landmarks_df)

        midpoint_x = (0.123456789 + 1.234567891) / 2
        midpoint_y = (0.987654321 + 1.987654321) / 2
        expected_radius = np.sqrt((1.234567891 - midpoint_x) ** 2 + (1.987654321 - midpoint_y) ** 2)

        np.testing.assert_array_almost_equal(result, np.array([expected_radius]), decimal=10)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64


class TestSmoothRadius:
    @pytest.mark.parametrize(
        ("radius", "smoothing_window", "expected"),
        [
            ([1.0, 2.0, 3.0, 4.0, 5.0], 3, [1.5, 2.0, 3.0, 4.0, 4.5]),
            ([1.0, 2.0, 3.0, 4.0, 5.0], 1, [1.0, 2.0, 3.0, 4.0, 5.0]),
            ([1.0, 2.0, 3.0], 5, [2.0, 2.0, 2.0]),
            ([5.0], 3, [5.0]),
            ([0.0, 0.0, 0.0, 0.0, 0.0], 3, [0.0, 0.0, 0.0, 0.0, 0.0]),
        ],
    )
    def test_smooth_radius_basic(self, radius, smoothing_window, expected):
        radius = np.array(radius)
        result = smooth_radius(radius, smoothing_window)
        expected = np.array(expected)

        np.testing.assert_array_almost_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_smooth_radius_empty_array(self):
        radius = np.array([])
        smoothing_window = 3

        result = smooth_radius(radius, smoothing_window)

        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_smooth_radius_with_noise(self):
        radius = np.array([1.0, 10.0, 2.0, 9.0, 3.0])
        smoothing_window = 3

        result = smooth_radius(radius, smoothing_window)

        assert np.all(result >= np.min(radius))
        assert np.all(result <= np.max(radius))

        assert np.var(result) < np.var(radius)
        assert isinstance(result, np.ndarray)

    def test_smooth_radius_even_window_size(self):
        radius = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        smoothing_window = 4

        result = smooth_radius(radius, smoothing_window)

        assert len(result) == len(radius)
        assert isinstance(result, np.ndarray)

        assert np.all(result >= np.min(radius))
        assert np.all(result <= np.max(radius))

    def test_smooth_radius_preserves_length(self):
        for length in [1, 3, 5, 10, 100]:
            radius = np.random.rand(length)
            smoothing_window = 5

            result = smooth_radius(radius, smoothing_window)

            assert len(result) == length
            assert isinstance(result, np.ndarray)

    def test_smooth_radius_with_negative_values(self):
        radius = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
        smoothing_window = 3

        result = smooth_radius(radius, smoothing_window)

        assert len(result) == len(radius)
        assert isinstance(result, np.ndarray)

        assert np.all(result <= np.max(radius))
        assert np.all(result >= np.min(radius))

    def test_smooth_radius_data_types(self):
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            radius = np.array([1, 2, 3, 4, 5], dtype=dtype)
            smoothing_window = 3

            result = smooth_radius(radius, smoothing_window)

            assert isinstance(result, np.ndarray)
            assert len(result) == len(radius)
