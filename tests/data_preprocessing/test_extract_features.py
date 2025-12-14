import numpy as np
import pytest

from ai.data_preprocessing.extract_features import (
    safe_len,
    safe_sum,
    safe_mean,
    safe_max,
    safe_std,
    join_segments,
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
