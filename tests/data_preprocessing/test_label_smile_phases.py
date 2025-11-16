import numpy as np
import pandas as pd
import pytest

from ai.data_preprocessing.label_smile_phases import calculate_radius


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
