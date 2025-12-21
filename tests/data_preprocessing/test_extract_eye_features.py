import numpy as np
import pandas as pd
import pytest

from data_preprocessing.extract_eye_features import (
    euclidean,
    kappa,
    tau,
    compute_eyelid_amplitude,
    extract_eye_features,
)
from data_preprocessing.face_landmarks import FaceLandmarks


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


class TestTau:
    def test_basic_values_match_kappa_times_euclidean(self):
        a = np.array([[0.0, 0.0], [1.0, 2.0], [5.0, 5.0]], dtype=float)
        b = np.array([[3.0, 4.0], [4.0, 1.0], [10.0, 10.0]], dtype=float)

        t = tau(a, b)
        k = kappa(a, b).astype(float)
        d = euclidean(a, b)

        np.testing.assert_allclose(t, k * d, rtol=0, atol=1e-12)

        expected_signs = np.array([-1.0, 1.0, -1.0])
        np.testing.assert_array_equal(np.sign(t), expected_signs)

    def test_zero_when_points_identical(self):
        a = np.array([[2.5, -7.0], [0.0, 0.0], [100.0, 100.0]], dtype=float)
        b = a.copy()

        t = tau(a, b)

        np.testing.assert_allclose(t, np.zeros(3), rtol=0, atol=0)

    def test_translation_invariance(self):
        rng = np.random.default_rng(2025)
        a = rng.normal(size=(20, 2))
        b = rng.normal(size=(20, 2))

        shift = np.array([123.456, -78.9])
        a_shift = a + shift
        b_shift = b + shift

        t = tau(a, b)
        t_shift = tau(a_shift, b_shift)

        np.testing.assert_allclose(t, t_shift, rtol=0, atol=1e-12)

    def test_abs_equals_euclidean_and_sign_equals_kappa(self):
        rng = np.random.default_rng(7)
        a = rng.uniform(-10, 10, size=(30, 2))
        b = rng.uniform(-10, 10, size=(30, 2))

        t = tau(a, b)
        d = euclidean(a, b)
        k = kappa(a, b)

        np.testing.assert_allclose(np.abs(t), d, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(np.sign(t).astype(int), k)

    def test_handles_non_contiguous_inputs(self):
        base = np.vstack([np.arange(0, 60, dtype=float), np.arange(100, 160, dtype=float)]).T  # (60, 2)
        a = base[::4]
        b = base[1::4]

        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]

        manual = kappa(a, b).astype(float) * np.linalg.norm(a - b, axis=1)
        t = tau(a, b)

        np.testing.assert_allclose(t, manual, rtol=0, atol=1e-12)

    def test_output_shape_and_dtype(self):
        a = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=np.float64)
        b = np.array([[1.0, 0.0], [0.0, 5.0], [6.0, 10.0]], dtype=np.float64)

        t = tau(a, b)

        assert t.shape == (3,)
        assert np.issubdtype(t.dtype, np.floating)


class TestComputeEyelidAmplitude:
    @pytest.fixture
    def eye_landmark_indices(self, monkeypatch):
        monkeypatch.setattr(FaceLandmarks, "right_eye_right_corner", staticmethod(lambda: [10]))
        monkeypatch.setattr(FaceLandmarks, "right_eye_upper_0_middle", staticmethod(lambda: [11]))
        monkeypatch.setattr(FaceLandmarks, "right_eye_left_corner", staticmethod(lambda: [12]))

        monkeypatch.setattr(FaceLandmarks, "left_eye_right_corner", staticmethod(lambda: [20]))
        monkeypatch.setattr(FaceLandmarks, "left_eye_upper_0_middle", staticmethod(lambda: [21]))
        monkeypatch.setattr(FaceLandmarks, "left_eye_left_corner", staticmethod(lambda: [22]))

        return {"RRC": 10, "REM": 11, "RLC": 12, "LRC": 20, "LEM": 21, "LLC": 22}

    def _df_with_eyes(self, frames, right_eye, right_mid, left_eye, left_mid, idx):
        rows = []

        for f, (rc_r, rc_l), rm, (lc_r, lc_l), lm in zip(frames, right_eye, right_mid, left_eye, left_mid):
            rows.append(
                {
                    "frame_number": f,
                    f"{idx['RRC']}_x": float(rc_r[0]),
                    f"{idx['RRC']}_y": float(rc_r[1]),
                    f"{idx['REM']}_x": float(rm[0]),
                    f"{idx['REM']}_y": float(rm[1]),
                    f"{idx['RLC']}_x": float(rc_l[0]),
                    f"{idx['RLC']}_y": float(rc_l[1]),
                    f"{idx['LRC']}_x": float(lc_r[0]),
                    f"{idx['LRC']}_y": float(lc_r[1]),
                    f"{idx['LEM']}_x": float(lm[0]),
                    f"{idx['LEM']}_y": float(lm[1]),
                    f"{idx['LLC']}_x": float(lc_l[0]),
                    f"{idx['LLC']}_y": float(lc_l[1]),
                }
            )

        return pd.DataFrame(rows)

    def test_happy_path_known_geometry(self, eye_landmark_indices):
        idx = eye_landmark_indices
        frames = [0, 1]

        right_eye = [((4, 0), (0, 0)), ((4, 0), (0, 0))]
        left_eye = [((14, 0), (10, 0)), ((14, 0), (10, 0))]

        right_mid = [(2, 0), (2, 1)]
        left_mid = [(12, 0), (12, 1)]

        df = self._df_with_eyes(frames, right_eye, right_mid, left_eye, left_mid, idx)

        out = compute_eyelid_amplitude(df)

        l1 = df[[f"{idx['RRC']}_x", f"{idx['RRC']}_y"]].values
        l3 = df[[f"{idx['RLC']}_x", f"{idx['RLC']}_y"]].values
        r_mid = (l1 + l3) / 2
        l2 = df[[f"{idx['REM']}_x", f"{idx['REM']}_y"]].values

        l4 = df[[f"{idx['LRC']}_x", f"{idx['LRC']}_y"]].values
        l6 = df[[f"{idx['LLC']}_x", f"{idx['LLC']}_y"]].values
        l_mid = (l4 + l6) / 2
        l5 = df[[f"{idx['LEM']}_x", f"{idx['LEM']}_y"]].values

        expected = (tau(r_mid, l2) + tau(l_mid, l5)) / (2 * euclidean(l1, l3))

        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)
        assert out.shape == (len(frames),)
        assert np.issubdtype(out.dtype, np.floating)

    def test_translation_invariance(self, eye_landmark_indices):
        idx = eye_landmark_indices
        rng = np.random.default_rng(123)
        frames = list(range(10))

        right_eye = []
        left_eye = []
        right_mid = []
        left_mid = []

        for _ in frames:
            r0 = rng.uniform(-5, 5, size=2)
            r1 = r0 + np.array([4.0, 0.0])

            r_m = (r0 + r1) / 2 + np.array([0.0, rng.uniform(-2, 2)])
            right_eye.append((r1, r0))
            right_mid.append(tuple(r_m))

            l0 = rng.uniform(8, 15, size=2)
            l1c = l0 + np.array([4.0, 0.0])
            l_m = (l0 + l1c) / 2 + np.array([0.0, rng.uniform(-2, 2)])
            left_eye.append((l1c, l0))
            left_mid.append(tuple(l_m))

        df = self._df_with_eyes(frames, right_eye, right_mid, left_eye, left_mid, idx)

        shift = np.array([123.456, -78.9])
        df_shift = df.copy()

        for key in ["RRC", "REM", "RLC", "LRC", "LEM", "LLC"]:
            df_shift[[f"{idx[key]}_x", f"{idx[key]}_y"]] = df_shift[[f"{idx[key]}_x", f"{idx[key]}_y"]] + shift

        base = compute_eyelid_amplitude(df)
        shifted = compute_eyelid_amplitude(df_shift)

        np.testing.assert_allclose(base, shifted, rtol=0, atol=1e-12)

    def test_scale_invariance(self, eye_landmark_indices):
        idx = eye_landmark_indices
        frames = [0, 1, 2]

        right_eye = [((4, 0), (0, 0)), ((4, 0), (0, 0)), ((4, 1), (0, 1))]
        left_eye = [((14, 0), (10, 0)), ((14, 0), (10, 0)), ((14, 1), (10, 1))]
        right_mid = [(2, 0), (2, 1), (2, 2)]
        left_mid = [(12, 0), (12, 1), (12, 2)]

        df = self._df_with_eyes(frames, right_eye, right_mid, left_eye, left_mid, idx)

        s = 3.5
        df_scaled = df.copy()
        for key in ["RRC", "REM", "RLC", "LRC", "LEM", "LLC"]:
            df_scaled[[f"{idx[key]}_x", f"{idx[key]}_y"]] = df_scaled[[f"{idx[key]}_x", f"{idx[key]}_y"]] * s

        base = compute_eyelid_amplitude(df)
        scaled = compute_eyelid_amplitude(df_scaled)

        np.testing.assert_allclose(base, scaled, rtol=0, atol=1e-12)

    def test_raises_keyerror_when_required_columns_missing(self, eye_landmark_indices):
        idx = eye_landmark_indices
        df = pd.DataFrame(
            [
                {
                    "frame_number": 0,
                    f"{idx['RRC']}_x": 4.0,
                    f"{idx['RRC']}_y": 0.0,
                    f"{idx['REM']}_x": 2.0,
                    f"{idx['REM']}_y": 0.0,
                    f"{idx['RLC']}_x": 0.0,
                    f"{idx['RLC']}_y": 0.0,
                    f"{idx['LRC']}_x": 14.0,
                    f"{idx['LRC']}_y": 0.0,
                    f"{idx['LEM']}_x": 12.0,
                    f"{idx['LEM']}_y": 0.5,
                    f"{idx['LLC']}_x": 10.0,
                    # f"{idx['LLC']}_y" missing
                }
            ]
        )

        with pytest.raises(KeyError):
            compute_eyelid_amplitude(df)

    def test_warns_path_zero_right_eye_corner_distance_results_in_inf(self, eye_landmark_indices):
        idx = eye_landmark_indices
        frames = [0, 1]

        right_eye = [((0, 0), (0, 0)), ((4, 0), (0, 0))]
        left_eye = [((14, 0), (10, 0)), ((14, 0), (10, 0))]

        right_mid = [(0, 1), (2, 1)]
        left_mid = [(12, 0), (12, 1)]

        df = self._df_with_eyes(frames, right_eye, right_mid, left_eye, left_mid, idx)

        out = compute_eyelid_amplitude(df)

        assert np.isinf(out[0])
        assert np.isfinite(out[1])


class TestExtractEyeFeatures:
    @pytest.fixture
    def eye_landmark_indices(self, monkeypatch):
        monkeypatch.setattr(FaceLandmarks, "right_eye_right_corner", staticmethod(lambda: [10]))
        monkeypatch.setattr(FaceLandmarks, "right_eye_upper_0_middle", staticmethod(lambda: [11]))
        monkeypatch.setattr(FaceLandmarks, "right_eye_left_corner", staticmethod(lambda: [12]))

        monkeypatch.setattr(FaceLandmarks, "left_eye_right_corner", staticmethod(lambda: [20]))
        monkeypatch.setattr(FaceLandmarks, "left_eye_upper_0_middle", staticmethod(lambda: [21]))
        monkeypatch.setattr(FaceLandmarks, "left_eye_left_corner", staticmethod(lambda: [22]))

        return {"RRC": 10, "REM": 11, "RLC": 12, "LRC": 20, "LEM": 21, "LLC": 22}

    def _df_with_eyes(self, frames, right_eye, right_mid, left_eye, left_mid, idx):
        rows = []

        for f, (rc_r, rc_l), rm, (lc_r, lc_l), lm in zip(frames, right_eye, right_mid, left_eye, left_mid):
            rows.append(
                {
                    "frame_number": f,
                    f"{idx['RRC']}_x": float(rc_r[0]),
                    f"{idx['RRC']}_y": float(rc_r[1]),
                    f"{idx['REM']}_x": float(rm[0]),
                    f"{idx['REM']}_y": float(rm[1]),
                    f"{idx['RLC']}_x": float(rc_l[0]),
                    f"{idx['RLC']}_y": float(rc_l[1]),
                    f"{idx['LRC']}_x": float(lc_r[0]),
                    f"{idx['LRC']}_y": float(lc_r[1]),
                    f"{idx['LEM']}_x": float(lm[0]),
                    f"{idx['LEM']}_y": float(lm[1]),
                    f"{idx['LLC']}_x": float(lc_l[0]),
                    f"{idx['LLC']}_y": float(lc_l[1]),
                }
            )

        return pd.DataFrame(rows)

    @pytest.fixture
    def setup_landmarks_and_smile(self, eye_landmark_indices):
        idx = eye_landmark_indices

        frames = [0, 1, 2, 3]
        right_eye = [((4, 0), (0, 0)), ((4, 0), (0, 0)), ((4, 1), (0, 1)), ((4, 1), (0, 1))]
        left_eye = [((14, 0), (10, 0)), ((14, 0), (10, 0)), ((14, 1), (10, 1)), ((14, 1), (10, 1))]
        right_mid = [(2, 0), (2, 1), (2, 2), (2, 3)]
        left_mid = [(12, 0), (12, 1), (12, 2), (12, 3)]

        landmarks_df = self._df_with_eyes(frames, right_eye, right_mid, left_eye, left_mid, idx)

        smile_phases_df = pd.DataFrame(
            {
                "frame_number": [0, 2, 3],
                "phase": ["onset", "apex", "offset"],
            }
        )

        return landmarks_df, smile_phases_df, idx

    def test_happy_path_calls_extract_features_with_expected_df(self, monkeypatch, setup_landmarks_and_smile):
        landmarks_df, smile_phases_df, idx = setup_landmarks_and_smile

        read_calls = {"count": 0}

        def fake_read_csv(path):
            if read_calls["count"] == 0:
                read_calls["count"] += 1
                return landmarks_df.copy()

            read_calls["count"] += 1
            return smile_phases_df.copy()

        monkeypatch.setattr(pd, "read_csv", fake_read_csv)

        captured = {}

        def fake_extract_features(df, fps):
            captured["df"] = df.copy()
            captured["fps"] = fps
            return pd.DataFrame({"some_feature": [1, 2, 3]})

        import data_preprocessing.extract_eye_features as eye_mod

        monkeypatch.setattr(eye_mod, "extract_features", fake_extract_features)

        fps = 25.0
        result = extract_eye_features(PathLikeDummy("landmarks.csv"), PathLikeDummy("smile.csv"), fps)

        pd.testing.assert_frame_equal(result, pd.DataFrame({"some_feature": [1, 2, 3]}))
        assert captured["fps"] == fps

        df_passed = captured["df"]
        expected_cols = {"frame_number", "D", "V", "A", "phase"}
        assert expected_cols.issubset(set(df_passed.columns))
        assert df_passed["frame_number"].tolist() == [0, 2, 3]

        merged_frames = df_passed["frame_number"].tolist()

        D_full = pd.Series(compute_eyelid_amplitude(landmarks_df), index=landmarks_df["frame_number"], dtype=float)
        D = D_full.loc[merged_frames].reset_index(drop=True)
        V = D.diff()
        A = V.diff()

        np.testing.assert_allclose(df_passed["D"].to_numpy(), D.to_numpy(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(df_passed["V"].to_numpy(), V.fillna(0.0).to_numpy(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(df_passed["A"].to_numpy(), A.fillna(0.0).to_numpy(), rtol=0, atol=1e-12)

        assert not df_passed.isna().any().any()

    def test_raises_when_reading_files_fails(self, monkeypatch):
        def boom(_):
            raise IOError("failed to read")

        monkeypatch.setattr(pd, "read_csv", boom)

        with pytest.raises(Exception):
            extract_eye_features(PathLikeDummy("landmarks.csv"), PathLikeDummy("smile.csv"), 30.0)


class PathLikeDummy:
    """Minimal path-like object exposing .name used by the function under test."""

    def __init__(self, name: str):
        self.name = name
