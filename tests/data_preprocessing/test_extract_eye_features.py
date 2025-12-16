import numpy as np

from ai.data_preprocessing.extract_eye_features import euclidean, kappa


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


class TestKappa:
    def test_basic_signs(self):
        a = np.array([[0.0, 0.0], [1.0, 2.0], [5.0, 5.0]], dtype=float)
        b = np.array([[0.0, 1.0], [1.0, 1.0], [10.0, 5.0]], dtype=float)

        out = kappa(a, b)

        np.testing.assert_array_equal(out, np.array([-1, 1, 1], dtype=int))

    def test_equal_y_returns_positive_one(self):
        a = np.array([[0.0, 3.14], [2.0, -7.0], [9.0, 0.0]], dtype=float)
        b = np.array([[5.0, 3.14], [3.0, -7.0], [9.0, 0.0]], dtype=float)

        out = kappa(a, b)

        np.testing.assert_array_equal(out, np.ones(3, dtype=int))

    def test_translation_invariance(self):
        rng = np.random.default_rng(123)
        a = rng.normal(size=(50, 2))
        b = rng.normal(size=(50, 2))

        shift = np.array([1000.0, -2500.0])
        a_shift = a + shift
        b_shift = b + shift

        base = kappa(a, b)
        shifted = kappa(a_shift, b_shift)

        np.testing.assert_array_equal(base, shifted)

    def test_ignores_x_coordinate(self):
        a = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]], dtype=float)
        b = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 2.0]], dtype=float)

        base = kappa(a, b)

        a_changed_x = a.copy()
        b_changed_x = b.copy()
        a_changed_x[:, 0] = np.array([100.0, -50.0, 999.0])
        b_changed_x[:, 0] = np.array([-1000.0, 500.0, -999.0])

        changed = kappa(a_changed_x, b_changed_x)

        np.testing.assert_array_equal(base, changed)

    def test_output_shape_and_dtype_and_values(self):
        a = np.array([[i, i % 3 - 1] for i in range(20)], dtype=float)
        b = np.array([[i, (i % 3) - 0.5] for i in range(20)], dtype=float)

        out = kappa(a, b)

        assert out.shape == (20,)
        assert np.issubdtype(out.dtype, np.integer)
        assert set(np.unique(out)).issubset({-1, 1})

    def test_handles_non_contiguous_inputs(self):
        base = np.vstack([np.arange(0, 40, dtype=float), np.arange(40, 80, dtype=float)]).T
        a = base[:-1:3]
        b = base[1::3]

        expected = np.where(b[:, 1] > a[:, 1], -1, 1)
        out = kappa(a, b)

        np.testing.assert_array_equal(out, expected)
