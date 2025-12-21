import numpy as np
import pandas as pd
import pytest

from data_preprocessing.label_smile_phases import (
    calculate_radius,
    smooth_radius,
    detect_longest_segment,
    label_phases,
    label_smile_phases,
)

from pathlib import Path
from unittest.mock import patch
import tempfile


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


class TestDetectLongestSegment:
    @pytest.mark.parametrize(
        ("delta", "start_expected", "end_expected"),
        [
            ([0.0, 1.0, 2.0, 3.0, -1.0, -2.0], 1, 3),
            ([0.0, 1.0, 2.0, -1.0, 3.0, 4.0, 5.0, 6.0, -2.0], 4, 7),
            ([0.0, 1.0, 2.0, -1.0, 3.0, 4.0, -2.0], 1, 2),
            ([0.0, 1.0, 2.0, 3.0, 4.0], 1, 4),
            ([0.0, -1.0, 1.0, 2.0, 3.0], 2, 4),
            ([5.0, 1.0, 2.0, -1.0], 1, 2),
            ([0.0, 1.0, -1.0, 2.0, 3.0, 4.0, 5.0, -2.0, 1.0, -3.0], 3, 6),
            ([0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 5.0], 4, 6),
            ([0.0, 1.0, -1.0], 1, 1),
            ([0.0, -1.0, 1.0], 2, 2),
            ([0.0, 0.0, 1.0, 0.0], 2, 2),
            ([0.0, 1.0, 0.0, 1.0, 0.0], 1, 1),
            ([0.0, 0.0, 0.0, 1.0], 3, 3),
        ],
    )
    def test_detect_longest_segment_d_greater_than_0(self, delta, start_expected, end_expected):
        delta = np.array(delta)

        def condition(d):
            return d > 0

        start, end = detect_longest_segment(delta, condition)

        assert start == start_expected
        assert end == end_expected

        assert isinstance(start, int)
        assert isinstance(end, int)

    @pytest.mark.parametrize(
        ("delta", "start_expected", "end_expected"),
        [
            ([0.0, 1.0, 2.0, -1.0, -2.0, -3.0], 3, 5),
            ([0.0, -1.0, -2.0, 1.0, -3.0, -4.0, -5.0, -6.0, 2.0], 4, 7),
            ([0.0, 1.0, -1.0, 1.0], 2, 2),
        ],
    )
    def test_detect_longest_segment_d_lesser_than_0(self, delta, start_expected, end_expected):
        delta = np.array(delta)

        def condition(d):
            return d < 0

        start, end = detect_longest_segment(delta, condition)

        assert start == start_expected
        assert end == end_expected

        assert isinstance(start, int)
        assert isinstance(end, int)

    def test_detect_longest_segment_no_matching_values(self):
        delta = np.array([0.0, -1.0, -2.0, -3.0])

        def condition(d):
            return d > 0

        start, end = detect_longest_segment(delta, condition)

        assert start is None
        assert end is None

    def test_detect_longest_segment_single_element_array(self):
        delta = np.array([1.0])

        def condition(d):
            return d > 0

        start, end = detect_longest_segment(delta, condition)

        assert start is None
        assert end is None

    def test_detect_longest_segment_empty_array(self):
        delta = np.array([])

        def condition(d):
            return d > 0

        start, end = detect_longest_segment(delta, condition)

        assert start is None
        assert end is None

    def test_detect_longest_segment_custom_condition(self):
        delta = np.array([0.0, 1.0, 3.0, 4.0, 1.0, -3.0, -4.0, -5.0, 1.0])

        def condition(d):
            return abs(d) > 2

        start, end = detect_longest_segment(delta, condition)

        assert start == 5
        assert end == 7

    def test_detect_longest_segment_equal_condition(self):
        delta = np.array([0.0, 1.0, 2.0, 2.0, 2.0, 1.0])

        def condition(d):
            return d == 2.0

        start, end = detect_longest_segment(delta, condition)

        assert start == 2
        assert end == 4

    def test_detect_longest_segment_floating_point_precision(self):
        delta = np.array([0.0, 0.1, 0.2, 0.3, -0.1, 0.4, 0.5])

        def condition(d):
            return d > 0.15

        start, end = detect_longest_segment(delta, condition)

        assert start == 2
        assert end == 3


class TestLabelPhases:
    @pytest.mark.parametrize(
        ("num_frames", "onset_start", "onset_end", "offset_start", "offset_end", "expected_phases"),
        [
            (5, 1, 1, 3, 3, ["neutral", "onset", "apex", "offset", "neutral"]),
            (4, 0, 1, 2, 3, ["onset", "onset", "offset", "offset"]),
            (3, 0, 0, 2, 2, ["onset", "apex", "offset"]),
            (6, 1, 2, None, None, ["neutral", "onset", "onset", "neutral", "neutral", "neutral"]),
            (6, None, None, 3, 4, ["neutral", "neutral", "neutral", "offset", "offset", "neutral"]),
            (5, 2, 3, 2, 3, ["neutral", "neutral", "offset", "offset", "neutral"]),
            (10, 0, 0, 9, 9, ["onset", "apex", "apex", "apex", "apex", "apex", "apex", "apex", "apex", "offset"]),
            (
                10,
                2,
                3,
                6,
                8,
                ["neutral", "neutral", "onset", "onset", "apex", "apex", "offset", "offset", "offset", "neutral"],
            ),
            (8, 2, 4, None, None, ["neutral", "neutral", "onset", "onset", "onset", "neutral", "neutral", "neutral"]),
            (
                8,
                None,
                None,
                3,
                5,
                ["neutral", "neutral", "neutral", "offset", "offset", "offset", "neutral", "neutral"],
            ),
            (1, None, None, None, None, ["neutral"]),
            (8, 1, 3, 4, 6, ["neutral", "onset", "onset", "onset", "offset", "offset", "offset", "neutral"]),
            (5, 2, 2, None, None, ["neutral", "neutral", "onset", "neutral", "neutral"]),
            (5, None, None, 3, 3, ["neutral", "neutral", "neutral", "offset", "neutral"]),
            (6, 0, 1, 4, 5, ["onset", "onset", "apex", "apex", "offset", "offset"]),
            (6, 1, 2, 4, 5, ["neutral", "onset", "onset", "apex", "offset", "offset"]),
            (7, 1, 2, 3, 5, ["neutral", "onset", "onset", "offset", "offset", "offset", "neutral"]),
        ],
    )
    def test_label_phases_parametrized(
        self, num_frames, onset_start, onset_end, offset_start, offset_end, expected_phases
    ):
        result = label_phases(num_frames, onset_start, onset_end, offset_start, offset_end)

        assert result == expected_phases
        assert len(result) == num_frames

        assert isinstance(result, list)
        assert all(isinstance(phase, str) for phase in result)

    def test_label_phases_zero_frames(self):
        num_frames = 0
        onset_start, onset_end = None, None
        offset_start, offset_end = None, None

        result = label_phases(num_frames, onset_start, onset_end, offset_start, offset_end)

        assert result == []
        assert len(result) == 0

    def test_label_phases_no_phases(self):
        num_frames = 5
        onset_start, onset_end = None, None
        offset_start, offset_end = None, None

        result = label_phases(num_frames, onset_start, onset_end, offset_start, offset_end)

        expected = ["neutral"] * 5
        assert result == expected
        assert len(result) == num_frames

    def test_label_phases_large_sequence(self):
        num_frames = 100
        onset_start, onset_end = 20, 30
        offset_start, offset_end = 70, 80

        result = label_phases(num_frames, onset_start, onset_end, offset_start, offset_end)

        assert all(phase == "neutral" for phase in result[:20])
        assert all(phase == "onset" for phase in result[20:31])
        assert all(phase == "apex" for phase in result[31:70])
        assert all(phase == "offset" for phase in result[70:81])
        assert all(phase == "neutral" for phase in result[81:])
        assert len(result) == num_frames


class TestLabelSmilePhases:
    @pytest.fixture
    def sample_landmarks_data(self):
        return pd.DataFrame(
            {
                "frame_number": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "291_x": [10.0, 10.1, 10.2, 10.3, 10.4, 10.3, 10.2, 10.1, 10.0, 9.9],
                "291_y": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                "61_x": [30.0, 30.2, 30.4, 30.6, 30.8, 30.6, 30.4, 30.2, 30.0, 29.8],
                "61_y": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            }
        )

    @pytest.fixture
    def temp_csv_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            landmarks_file = Path(temp_dir) / "landmarks.csv"
            output_file = Path(temp_dir) / "output.csv"
            yield landmarks_file, output_file

    def test_label_smile_phases_complete_workflow(self, sample_landmarks_data, temp_csv_files):
        landmarks_file, output_file = temp_csv_files
        sample_landmarks_data.to_csv(landmarks_file, index=False)
        label_smile_phases(landmarks_file, output_file, smoothing_window=3)

        assert output_file.exists()

        result_df = pd.read_csv(output_file)

        expected_columns = ["frame_number", "smile_phase", "radius", "dist_smooth", "delta"]

        assert list(result_df.columns) == expected_columns

        assert len(result_df) == len(sample_landmarks_data)
        assert list(result_df["frame_number"]) == list(sample_landmarks_data["frame_number"])

        valid_phases = {"neutral", "onset", "apex", "offset"}

        assert all(phase in valid_phases for phase in result_df["smile_phase"])
        assert all(isinstance(radius, (int, float)) and not np.isnan(radius) for radius in result_df["radius"])

    def test_label_smile_phases_with_different_smoothing_window(self, sample_landmarks_data, temp_csv_files):
        landmarks_file, output_file = temp_csv_files
        sample_landmarks_data.to_csv(landmarks_file, index=False)

        for smoothing_window in [1, 3, 5, 7]:
            if output_file.exists():
                output_file.unlink()

            label_smile_phases(landmarks_file, output_file, smoothing_window=smoothing_window)

            assert output_file.exists()

            result_df = pd.read_csv(output_file)

            assert len(result_df) == len(sample_landmarks_data)

    def test_label_smile_phases_empty_dataframe(self, temp_csv_files):
        landmarks_file, output_file = temp_csv_files

        empty_df = pd.DataFrame(
            {
                "frame_number": [],
                "291_x": [],
                "291_y": [],
                "61_x": [],
                "61_y": [],
            }
        )

        empty_df.to_csv(landmarks_file, index=False)

        with pytest.raises(ValueError):
            label_smile_phases(landmarks_file, output_file)

    def test_label_smile_phases_single_frame(self, temp_csv_files):
        landmarks_file, output_file = temp_csv_files

        single_frame_df = pd.DataFrame(
            {
                "frame_number": [1],
                "291_x": [10.0],
                "291_y": [20.0],
                "61_x": [30.0],
                "61_y": [20.0],
            }
        )

        single_frame_df.to_csv(landmarks_file, index=False)
        label_smile_phases(landmarks_file, output_file)

        assert output_file.exists()

        result_df = pd.read_csv(output_file)

        assert len(result_df) == 1
        assert result_df["smile_phase"].iloc[0] == "neutral"

    @patch("data_preprocessing.label_smile_phases.save_dataframe_to_csv")
    @patch("pandas.read_csv")
    def test_label_smile_phases_mocked_dependencies(self, mock_read_csv, mock_save_csv):
        mock_landmarks_df = pd.DataFrame(
            {
                "frame_number": [1, 2, 3, 4, 5],
                "291_x": [10.0, 11.0, 12.0, 11.5, 10.5],
                "291_y": [20.0, 20.0, 20.0, 20.0, 20.0],
                "61_x": [30.0, 31.0, 32.0, 31.5, 30.5],
                "61_y": [20.0, 20.0, 20.0, 20.0, 20.0],
            }
        )

        mock_read_csv.return_value = mock_landmarks_df

        landmarks_file = Path("test_landmarks.csv")
        output_file = Path("test_output.csv")

        label_smile_phases(landmarks_file, output_file, smoothing_window=3)

        mock_read_csv.assert_called_once_with(landmarks_file)
        mock_save_csv.assert_called_once()

        saved_df = mock_save_csv.call_args[0][0]

        assert list(saved_df.columns) == ["frame_number", "smile_phase", "radius", "dist_smooth", "delta"]
        assert len(saved_df) == 5

    def test_label_smile_phases_realistic_smile_sequence(self, temp_csv_files):
        landmarks_file, output_file = temp_csv_files

        realistic_data = pd.DataFrame(
            {
                "frame_number": list(range(1, 21)),
                "291_x": [10.0] * 5 + [9.8, 9.6, 9.4, 9.2, 9.0] + [9.0] * 5 + [9.2, 9.4, 9.6, 9.8, 10.0],
                "291_y": [20.0] * 20,
                "61_x": [30.0] * 5 + [30.2, 30.4, 30.6, 30.8, 31.0] + [31.0] * 5 + [30.8, 30.6, 30.4, 30.2, 30.0],
                "61_y": [20.0] * 20,
            }
        )

        realistic_data.to_csv(landmarks_file, index=False)
        label_smile_phases(landmarks_file, output_file, smoothing_window=3)

        assert output_file.exists()

        result_df = pd.read_csv(output_file)

        phases = result_df["smile_phase"].tolist()
        unique_phases = set(phases)

        assert "neutral" in unique_phases
        assert len(result_df) == 20

    def test_label_smile_phases_constant_values(self, temp_csv_files):
        landmarks_file, output_file = temp_csv_files

        constant_data = pd.DataFrame(
            {
                "frame_number": [1, 2, 3, 4, 5],
                "291_x": [10.0] * 5,
                "291_y": [20.0] * 5,
                "61_x": [30.0] * 5,
                "61_y": [20.0] * 5,
            }
        )

        constant_data.to_csv(landmarks_file, index=False)
        label_smile_phases(landmarks_file, output_file)

        assert output_file.exists()

        result_df = pd.read_csv(output_file)

        assert all(phase == "neutral" for phase in result_df["smile_phase"])

        radii = result_df["radius"].tolist()

        assert all(abs(r - radii[0]) < 1e-10 for r in radii)

    def test_label_smile_phases_file_paths_validation(self, sample_landmarks_data):
        with tempfile.TemporaryDirectory() as temp_dir:
            landmarks_file = Path(temp_dir) / "landmarks.csv"
            output_file = Path(temp_dir) / "output.csv"
            sample_landmarks_data.to_csv(landmarks_file, index=False)

            label_smile_phases(landmarks_file, output_file)

            assert output_file.exists()

            output_file.unlink()

            landmarks_str = str(landmarks_file)
            output_str = str(output_file)

            label_smile_phases(Path(landmarks_str), Path(output_str))

            assert Path(output_str).exists()

    def test_label_smile_phases_large_dataset(self, temp_csv_files):
        landmarks_file, output_file = temp_csv_files

        num_frames = 100

        large_data = pd.DataFrame(
            {
                "frame_number": list(range(1, num_frames + 1)),
                "291_x": [10.0 + 0.1 * np.sin(i * 0.1) for i in range(num_frames)],
                "291_y": [20.0] * num_frames,
                "61_x": [30.0 + 0.1 * np.sin(i * 0.1) for i in range(num_frames)],
                "61_y": [20.0] * num_frames,
            }
        )

        large_data.to_csv(landmarks_file, index=False)
        label_smile_phases(landmarks_file, output_file, smoothing_window=5)

        assert output_file.exists()

        result_df = pd.read_csv(output_file)

        assert len(result_df) == num_frames

    def test_label_smile_phases_default_smoothing_window(self, sample_landmarks_data, temp_csv_files):
        landmarks_file, output_file = temp_csv_files
        sample_landmarks_data.to_csv(landmarks_file, index=False)

        label_smile_phases(landmarks_file, output_file)

        assert output_file.exists()

        result_df = pd.read_csv(output_file)

        assert len(result_df) == len(sample_landmarks_data)

    @patch("data_preprocessing.label_smile_phases.calculate_radius")
    @patch("data_preprocessing.label_smile_phases.smooth_radius")
    @patch("data_preprocessing.label_smile_phases.detect_longest_segment")
    @patch("data_preprocessing.label_smile_phases.label_phases")
    @patch("pandas.read_csv")
    @patch("data_preprocessing.label_smile_phases.save_dataframe_to_csv")
    def test_label_smile_phases_function_integration(
        self,
        mock_save_csv,
        mock_read_csv,
        mock_label_phases,
        mock_detect_segment,
        mock_smooth_radius,
        mock_calculate_radius,
    ):
        mock_landmarks_df = pd.DataFrame(
            {
                "frame_number": [1, 2, 3],
                "291_x": [10.0, 11.0, 12.0],
                "291_y": [20.0, 20.0, 20.0],
                "61_x": [30.0, 31.0, 32.0],
                "61_y": [20.0, 20.0, 20.0],
            }
        )

        mock_read_csv.return_value = mock_landmarks_df
        mock_calculate_radius.return_value = np.array([10.0, 10.5, 11.0])
        mock_smooth_radius.return_value = np.array([10.0, 10.5, 11.0])
        mock_detect_segment.side_effect = [(1, 2), (None, None)]  # onset, then offset
        mock_label_phases.return_value = ["neutral", "onset", "onset"]

        landmarks_file = Path("test.csv")
        output_file = Path("output.csv")

        label_smile_phases(landmarks_file, output_file, smoothing_window=7)

        mock_read_csv.assert_called_once_with(landmarks_file)
        mock_calculate_radius.assert_called_once()
        mock_smooth_radius.assert_called_once_with(mock_calculate_radius.return_value, 7)

        assert mock_detect_segment.call_count == 2

        mock_label_phases.assert_called_once_with(3, 1, 2, None, None)
        mock_save_csv.assert_called_once()

    def test_label_smile_phases_output_dataframe_structure(self, sample_landmarks_data, temp_csv_files):
        landmarks_file, output_file = temp_csv_files
        sample_landmarks_data.to_csv(landmarks_file, index=False)

        label_smile_phases(landmarks_file, output_file)

        result_df = pd.read_csv(output_file)
        expected_columns = ["frame_number", "smile_phase", "radius", "dist_smooth", "delta"]

        assert list(result_df.columns) == expected_columns

        assert all(isinstance(frame, (int, np.integer)) for frame in result_df["frame_number"])
        assert all(isinstance(phase, str) for phase in result_df["smile_phase"])
        assert all(isinstance(radius, (int, float, np.number)) for radius in result_df["radius"])
        assert all(isinstance(radius, (int, float, np.number)) for radius in result_df["dist_smooth"])
        assert all(isinstance(radius, (int, float, np.number)) for radius in result_df["delta"])

        assert list(result_df["frame_number"]) == list(sample_landmarks_data["frame_number"])
