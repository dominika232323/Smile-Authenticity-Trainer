import numpy as np

from ai.data_preprocessing.extract_features import safe_len


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
