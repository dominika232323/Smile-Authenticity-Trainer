import numpy as np
import pandas as pd
import pytest

from ai.data_preprocessing.extract_features import (
    safe_len,
    safe_sum,
    safe_mean,
    safe_max,
    safe_std,
    join_segments,
    segment_increasing_decreasing,
    extract_features,
    extract_features_for_phase,
)


class TestSafeLen:
    def test_safe_len_1d_array(self):
        arr = np.array([1, 2, 3])
        assert safe_len(arr) == 3

    def test_safe_len_2d_array(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        assert safe_len(arr) == 3

    def test_safe_len_empty_array(self):
        arr = np.array([])
        assert safe_len(arr) == 0

    def test_safe_len_string_array(self):
        arr = np.array(["a", "b", "c", "d"])
        assert safe_len(arr) == 4

    def test_safe_len_object_array(self):
        arr = np.array([{"x": 1}, {"y": 2}])
        assert safe_len(arr) == 2

    def test_safe_len_ndarray_subclass(self):
        class MyArray(np.ndarray):
            pass

        base = np.array([1, 2, 3])
        arr = base.view(MyArray)

        assert safe_len(arr) == 3


class TestSafeSum:
    def test_safe_sum_basic(self):
        arr = np.array([1, 2, 3])
        assert safe_sum(arr) == 6.0

    def test_safe_sum_empty_array(self):
        arr = np.array([])
        assert safe_sum(arr) == 0.0

    def test_safe_sum_negative_values(self):
        arr = np.array([-5, 10, -3])
        assert safe_sum(arr) == 2.0

    def test_safe_sum_float_values(self):
        arr = np.array([0.1, 0.2, 0.3])
        assert safe_sum(arr) == pytest.approx(0.6)

    def test_safe_sum_2d_array(self):
        arr = np.array([[1, 2], [3, 4]])
        assert safe_sum(arr) == 10.0

    def test_safe_sum_object_array(self):
        arr = np.array([1, 2, 3], dtype=object)
        assert safe_sum(arr) == 6.0

    def test_safe_sum_ndarray_subclass(self):
        class MyArray(np.ndarray):
            pass

        base = np.array([10, 20])
        arr = base.view(MyArray)

        assert safe_sum(arr) == 30.0


class TestSafeMean:
    def test_safe_mean_basic(self):
        arr = np.array([1, 2, 3])
        assert safe_mean(arr) == 2.0

    def test_safe_mean_empty_array(self):
        arr = np.array([])
        assert safe_mean(arr) == 0.0

    def test_safe_mean_negative_values(self):
        arr = np.array([-5, 10, -3])
        assert safe_mean(arr) == pytest.approx(0.6666667)

    def test_safe_mean_float_values(self):
        arr = np.array([0.1, 0.2, 0.3])
        assert safe_mean(arr) == pytest.approx(0.2)

    def test_safe_mean_2d_array(self):
        arr = np.array([[1, 2], [3, 4]])
        assert safe_mean(arr) == 2.5

    def test_safe_mean_object_array(self):
        arr = np.array([1, 2, 3], dtype=object)
        assert safe_mean(arr) == 2.0

    def test_safe_mean_ndarray_subclass(self):
        class MyArray(np.ndarray):
            pass

        base = np.array([10, 20, 30])
        arr = base.view(MyArray)

        assert safe_mean(arr) == 20.0


class TestSafeMax:
    def test_safe_max_basic(self):
        arr = np.array([1, 2, 3])
        assert safe_max(arr) == 3.0

    def test_safe_max_empty_array(self):
        arr = np.array([])
        assert safe_max(arr) == 0.0

    def test_safe_max_negative_values(self):
        arr = np.array([-5, -2, -10])
        assert safe_max(arr) == -2.0

    def test_safe_max_float_values(self):
        arr = np.array([0.1, 0.2, 0.3])
        assert safe_max(arr) == pytest.approx(0.3)

    def test_safe_max_2d_array(self):
        arr = np.array([[1, 2], [3, 4]])
        assert safe_max(arr) == 4.0

    def test_safe_max_object_array(self):
        arr = np.array([1, 2, 3], dtype=object)
        assert safe_max(arr) == 3.0

    def test_safe_max_ndarray_subclass(self):
        class MyArray(np.ndarray):
            pass

        base = np.array([10, 20, 5])
        arr = base.view(MyArray)

        assert safe_max(arr) == 20.0


class TestSafeStd:
    def test_safe_std_basic(self):
        arr = np.array([1, 2, 3])
        assert safe_std(arr) == pytest.approx(np.std(arr))

    def test_safe_std_empty_array(self):
        arr = np.array([])
        assert safe_std(arr) == 0.0

    def test_safe_std_negative_values(self):
        arr = np.array([-5, 10, -3])
        assert safe_std(arr) == pytest.approx(np.std(arr))

    def test_safe_std_float_values(self):
        arr = np.array([0.1, 0.2, 0.3])
        assert safe_std(arr) == pytest.approx(np.std(arr))

    def test_safe_std_2d_array(self):
        arr = np.array([[1, 2], [3, 4]])
        assert safe_std(arr) == pytest.approx(np.std(arr))

    def test_safe_std_object_array(self):
        arr = np.array([1, 2, 3], dtype=object)
        assert safe_std(arr) == pytest.approx(np.std(arr))

    def test_safe_std_ndarray_subclass(self):
        class MyArray(np.ndarray):
            pass

        base = np.array([10, 20, 30])
        arr = base.view(MyArray)

        assert safe_std(arr) == pytest.approx(np.std(arr))


class TestJoinSegments:
    def test_empty_list_returns_empty_array(self):
        result = join_segments([])
        assert np.array_equal(result, np.array([]))

    def test_single_segment_returns_same(self):
        seg = np.array([1, 2, 3])
        result = join_segments([seg])
        assert np.array_equal(result, seg)

    def test_multiple_segments_concatenate(self):
        segs = [np.array([1, 2]), np.array([3]), np.array([4, 5])]
        result = join_segments(segs)
        assert np.array_equal(result, np.array([1, 2, 3, 4, 5]))

    def test_segments_with_empty_arrays(self):
        segs = [np.array([]), np.array([1, 2]), np.array([]), np.array([3])]
        result = join_segments(segs)
        assert np.array_equal(result, np.array([1, 2, 3]))

    def test_object_dtype_segments(self):
        segs = [np.array([1, 2], dtype=object), np.array([3], dtype=object)]
        result = join_segments(segs)
        assert result.dtype == object
        assert np.array_equal(result, np.array([1, 2, 3], dtype=object))


class TestSegmentIncreasingDecreasing:
    def test_all_increasing_single_segment(self):
        signal = np.array([1, 2, 3, 4])
        inc, dec = segment_increasing_decreasing(signal)

        assert len(inc) == 1
        assert len(dec) == 0
        assert np.array_equal(inc[0], np.array([1, 2, 3, 4]))

    def test_all_decreasing_single_segment(self):
        signal = np.array([4, 3, 2, 1])
        inc, dec = segment_increasing_decreasing(signal)

        assert len(inc) == 0
        assert len(dec) == 1
        assert np.array_equal(dec[0], np.array([4, 3, 2, 1]))

    def test_increase_then_decrease_overlap_at_turning_point(self):
        signal = np.array([1, 2, 1])
        inc, dec = segment_increasing_decreasing(signal)

        assert len(inc) == 1
        assert len(dec) == 1
        assert np.array_equal(inc[0], np.array([1, 2]))
        assert np.array_equal(dec[0], np.array([2, 1]))

    def test_plateau_splits_segments(self):
        signal = np.array([1, 2, 2, 3])
        inc, dec = segment_increasing_decreasing(signal)

        assert len(dec) == 0
        assert len(inc) == 2
        assert np.array_equal(inc[0], np.array([1, 2]))
        assert np.array_equal(inc[1], np.array([2, 3]))

    def test_plateau_after_decrease_appends_decreasing_segment(self):
        signal = np.array([3, 2, 2])
        inc, dec = segment_increasing_decreasing(signal)

        assert inc == []
        assert len(dec) == 1
        assert np.array_equal(dec[0], np.array([3, 2]))

    def test_flat_sequence_no_segments(self):
        signal = np.array([1, 1, 1, 1])
        inc, dec = segment_increasing_decreasing(signal)

        assert inc == []
        assert dec == []

    def test_zigzag_multiple_segments(self):
        signal = np.array([1, 2, 1, 2, 1])
        inc, dec = segment_increasing_decreasing(signal)

        assert len(inc) == 2
        assert len(dec) == 2
        assert np.array_equal(inc[0], np.array([1, 2]))
        assert np.array_equal(dec[0], np.array([2, 1]))
        assert np.array_equal(inc[1], np.array([1, 2]))
        assert np.array_equal(dec[1], np.array([2, 1]))

    def test_single_element_no_segments(self):
        signal = np.array([5])
        inc, dec = segment_increasing_decreasing(signal)

        assert inc == []
        assert dec == []

    def test_float_and_negative_values(self):
        signal = np.array([0.0, -1.0, -2.0, -1.5, -3.0])
        inc, dec = segment_increasing_decreasing(signal)

        assert len(dec) == 2
        assert len(inc) == 1
        assert np.array_equal(dec[0], np.array([0.0, -1.0, -2.0]))
        assert np.array_equal(inc[0], np.array([-2.0, -1.5]))
        assert np.array_equal(dec[1], np.array([-1.5, -3.0]))


class TestExtractFeaturesForPhase:
    def test_all_increasing_signals(self):
        D = np.array([1.0, 2.0, 3.0])
        V = np.array([1.0, 2.0, 3.0])
        A = np.array([1.0, 2.0, 3.0])

        df = pd.DataFrame({"D": D, "V": V, "A": A})
        omega = 2.0

        feats = extract_features_for_phase(df, omega)

        assert feats["duration_all"] == pytest.approx(3 / omega)
        assert feats["duration_plus"] == pytest.approx(3 / omega)
        assert feats["duration_minus"] == pytest.approx(0.0)
        assert feats["duration_ratio_plus"] == pytest.approx(1.0)
        assert feats["duration_ratio_minus"] == pytest.approx(0.0)

        assert feats["max_amplitude"] == pytest.approx(3.0)
        assert feats["mean_amplitude"] == pytest.approx(2.0)
        assert feats["mean_amplitude_plus"] == pytest.approx(2.0)
        assert feats["mean_amplitude_minus"] == pytest.approx(0.0)
        assert feats["std_amplitude"] == pytest.approx(np.std(D))
        assert feats["total_amplitude_plus"] == pytest.approx(6.0)
        assert feats["total_amplitude_minus"] == pytest.approx(0.0)
        assert feats["net_amplitude"] == pytest.approx(6.0)
        assert feats["amplitude_ratio_plus"] == pytest.approx(1.0)
        assert feats["amplitude_ratio_minus"] == pytest.approx(0.0)
        assert feats["net_amplitude_duration_ratio"] == pytest.approx((6.0 - 0.0) * omega / 3)

        assert feats["max_speed_plus"] == pytest.approx(3.0)
        assert feats["max_speed_minus"] == pytest.approx(0.0)
        assert feats["mean_speed_plus"] == pytest.approx(2.0)
        assert feats["mean_speed_minus"] == pytest.approx(0.0)

        assert feats["max_acceleration_plus"] == pytest.approx(3.0)
        assert feats["max_acceleration_minus"] == pytest.approx(0.0)
        assert feats["mean_acceleration_plus"] == pytest.approx(2.0)
        assert feats["mean_acceleration_minus"] == pytest.approx(0.0)

    def test_all_decreasing_amplitude_flat_speed_and_accel(self):
        D = np.array([3.0, 2.0, 1.0, 0.0])
        V = np.array([0.0, 0.0, 0.0, 0.0])
        A = np.array([0.0, 0.0, 0.0, 0.0])

        df = pd.DataFrame({"D": D, "V": V, "A": A})
        omega = 1.0

        feats = extract_features_for_phase(df, omega)

        assert feats["duration_all"] == pytest.approx(4 / omega)
        assert feats["duration_plus"] == pytest.approx(0.0)
        assert feats["duration_minus"] == pytest.approx(4 / omega)
        assert feats["duration_ratio_plus"] == pytest.approx(0.0)
        assert feats["duration_ratio_minus"] == pytest.approx(1.0)

        assert feats["max_amplitude"] == pytest.approx(3.0)
        assert feats["mean_amplitude"] == pytest.approx(np.mean(D))
        assert feats["mean_amplitude_plus"] == pytest.approx(0.0)
        assert feats["mean_amplitude_minus"] == pytest.approx(np.mean(np.abs([3.0, 2.0, 1.0, 0.0])))
        assert feats["std_amplitude"] == pytest.approx(np.std(D))
        assert feats["total_amplitude_plus"] == pytest.approx(0.0)
        assert feats["total_amplitude_minus"] == pytest.approx(6.0)
        assert feats["net_amplitude"] == pytest.approx(-6.0)
        assert feats["amplitude_ratio_plus"] == pytest.approx(0.0)
        assert feats["amplitude_ratio_minus"] == pytest.approx(1.0)
        assert feats["net_amplitude_duration_ratio"] == pytest.approx((-6.0) * omega / 4)

        assert feats["max_speed_plus"] == pytest.approx(0.0)
        assert feats["max_speed_minus"] == pytest.approx(0.0)
        assert feats["mean_speed_plus"] == pytest.approx(0.0)
        assert feats["mean_speed_minus"] == pytest.approx(0.0)
        assert feats["max_acceleration_plus"] == pytest.approx(0.0)
        assert feats["max_acceleration_minus"] == pytest.approx(0.0)
        assert feats["mean_acceleration_plus"] == pytest.approx(0.0)
        assert feats["mean_acceleration_minus"] == pytest.approx(0.0)

    def test_increase_then_decrease_overlap(self):
        D = np.array([1.0, 3.0, 2.0])
        V = np.array([1.0, 2.0, 1.0])
        A = np.array([0.0, 0.0, 0.0])

        df = pd.DataFrame({"D": D, "V": V, "A": A})
        omega = 1.0

        feats = extract_features_for_phase(df, omega)

        assert feats["duration_all"] == pytest.approx(3.0)
        assert feats["duration_plus"] == pytest.approx(2.0)
        assert feats["duration_minus"] == pytest.approx(2.0)
        assert feats["duration_ratio_plus"] == pytest.approx(2 / 3)
        assert feats["duration_ratio_minus"] == pytest.approx(2 / 3)

        assert feats["max_amplitude"] == pytest.approx(3.0)
        assert feats["mean_amplitude"] == pytest.approx(np.mean(D))
        assert feats["mean_amplitude_plus"] == pytest.approx(np.mean([1.0, 3.0]))
        assert feats["mean_amplitude_minus"] == pytest.approx(np.mean([3.0, 2.0]))
        assert feats["std_amplitude"] == pytest.approx(np.std(D))
        assert feats["total_amplitude_plus"] == pytest.approx(4.0)
        assert feats["total_amplitude_minus"] == pytest.approx(5.0)
        assert feats["net_amplitude"] == pytest.approx(-1.0)
        assert feats["amplitude_ratio_plus"] == pytest.approx(4.0 / 9.0)
        assert feats["amplitude_ratio_minus"] == pytest.approx(5.0 / 9.0)
        assert feats["net_amplitude_duration_ratio"] == pytest.approx((-1.0) * omega / 3.0)

        assert feats["max_speed_plus"] == pytest.approx(2.0)
        assert feats["max_speed_minus"] == pytest.approx(2.0)
        assert feats["mean_speed_plus"] == pytest.approx(1.5)
        assert feats["mean_speed_minus"] == pytest.approx(1.5)

        assert feats["max_acceleration_plus"] == pytest.approx(0.0)
        assert feats["max_acceleration_minus"] == pytest.approx(0.0)
        assert feats["mean_acceleration_plus"] == pytest.approx(0.0)
        assert feats["mean_acceleration_minus"] == pytest.approx(0.0)


class TestExtractFeatures:
    def test_all_phases_present_values_and_columns(self):
        df = pd.DataFrame(
            {
                "D": [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 3.0],
                "V": [1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
                "A": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "smile_phase": [
                    "onset",
                    "onset",
                    "apex",
                    "apex",
                    "apex",
                    "offset",
                    "offset",
                ],
            }
        )

        omega = 2.0
        out = extract_features(df, omega)

        assert isinstance(out, pd.DataFrame)
        assert out.shape[0] == 1

        base_keys = list(
            extract_features_for_phase(df[df["smile_phase"] == "onset"].reset_index(drop=True), omega).keys()
        )

        for phase in ["onset", "apex", "offset"]:
            df_phase = df[df["smile_phase"] == phase].reset_index(drop=True)
            phase_feats = extract_features_for_phase(df_phase, omega)

            for k in base_keys:
                col = f"{phase}_{k}"

                assert col in out.columns
                assert out.at[0, col] == pytest.approx(phase_feats[k])

        assert out.shape[1] == 3 * len(base_keys)

    def test_missing_phase_yields_zero_features(self):
        df = pd.DataFrame(
            {
                "D": [1.0, 2.0, 4.0, 3.0],
                "V": [1.0, 1.0, 1.0, 0.0],
                "A": [0.0, 0.0, 0.0, 0.0],
                "smile_phase": ["onset", "onset", "offset", "offset"],
            }
        )

        omega = 1.5
        out = extract_features(df, omega)

        base_keys = list(
            extract_features_for_phase(df[df["smile_phase"] == "onset"].reset_index(drop=True), omega).keys()
        )

        for k in base_keys:
            col = f"apex_{k}"
            assert col in out.columns
            assert out.at[0, col] == pytest.approx(0.0)

        for phase in ["onset", "offset"]:
            df_phase = df[df["smile_phase"] == phase].reset_index(drop=True)
            phase_feats = extract_features_for_phase(df_phase, omega)
            for k in base_keys:
                col = f"{phase}_{k}"
                assert out.at[0, col] == pytest.approx(phase_feats[k])

    def test_single_row_per_phase_minimal_durations(self):
        df = pd.DataFrame(
            {
                "D": [2.0, 5.0, 3.0],
                "V": [0.0, 1.0, 0.0],
                "A": [0.0, 0.0, 0.0],
                "smile_phase": ["onset", "apex", "offset"],
            }
        )

        omega = 5.0
        out = extract_features(df, omega)

        for phase in ["onset", "apex", "offset"]:
            df_phase = df[df["smile_phase"] == phase].reset_index(drop=True)
            exp = extract_features_for_phase(df_phase, omega)

            assert out.at[0, f"{phase}_duration_all"] == pytest.approx(1.0 / omega)
            assert out.at[0, f"{phase}_duration_plus"] == pytest.approx(0.0)
            assert out.at[0, f"{phase}_duration_minus"] == pytest.approx(0.0)
            assert out.at[0, f"{phase}_std_amplitude"] == pytest.approx(0.0)

            for k, v in exp.items():
                assert out.at[0, f"{phase}_{k}"] == pytest.approx(v)
