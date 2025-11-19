import numpy as np
import pytest

from ai.data_preprocessing.extract_features import safe_len, safe_sum


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
