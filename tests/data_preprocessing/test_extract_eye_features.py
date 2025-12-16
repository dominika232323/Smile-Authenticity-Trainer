import numpy as np

from ai.data_preprocessing.extract_eye_features import euclidean


class TestEuclidean:
    def test_basic_two_point_pairs(self):
        a = np.array([[0.0, 0.0], [1.0, 1.0]])
        b = np.array([[3.0, 4.0], [4.0, 5.0]])

        d = euclidean(a, b)

        np.testing.assert_allclose(d, np.array([5.0, 5.0]), rtol=0, atol=1e-12)

    def test_zero_distance_for_identical_points(self):
        a = np.array([[2.5, -7.0], [0.0, 0.0], [100.0, 100.0]], dtype=float)
        b = a.copy()

        d = euclidean(a, b)

        np.testing.assert_allclose(d, np.zeros(3), rtol=0, atol=0)

    def test_translation_invariance(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(10, 2))
        b = rng.normal(size=(10, 2))

        shift = np.array([123.456, -78.9])
        a_shift = a + shift
        b_shift = b + shift

        d = euclidean(a, b)
        d_shift = euclidean(a_shift, b_shift)

        np.testing.assert_allclose(d, d_shift, rtol=0, atol=1e-12)

    def test_output_shape_and_dtype(self):
        a = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=np.float64)
        b = np.array([[1.0, 0.0], [0.0, 0.0], [6.0, 8.0]], dtype=np.float64)

        d = euclidean(a, b)

        assert d.shape == (3,)
        assert np.issubdtype(d.dtype, np.floating)

    def test_handles_non_contiguous_inputs(self):
        base_a = np.vstack([np.arange(0, 20, dtype=float), np.arange(20, 40, dtype=float)]).T  # shape (20, 2)
        base_b = base_a[::-1]

        a = base_a[::2]
        b = base_b[::2]

        manual = np.linalg.norm(a - b, axis=1)
        d = euclidean(a, b)

        np.testing.assert_allclose(d, manual, rtol=0, atol=1e-12)
