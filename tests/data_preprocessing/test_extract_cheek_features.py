import numpy as np
import pandas as pd
import pytest

from data_preprocessing.extract_cheek_features import (
    extract_cheek_features,
    normalized_amplitude_signal_of_cheeks,
)
from data_preprocessing.face_landmarks import FaceLandmarks


@pytest.fixture
def cheek_center_indices(monkeypatch):
    monkeypatch.setattr(FaceLandmarks, "left_cheek_center", staticmethod(lambda: [10]))
    monkeypatch.setattr(FaceLandmarks, "right_cheek_center", staticmethod(lambda: [20]))

    return 10, 20


def _df_with_cheeks(frames, left_points, right_points, li, ri):
    rows = []

    for f, lp, rp in zip(frames, left_points, right_points):
        rows.append(
            {
                "frame_number": f,
                f"{li}_x": float(lp[0]),
                f"{li}_y": float(lp[1]),
                f"{ri}_x": float(rp[0]),
                f"{ri}_y": float(rp[1]),
            }
        )

    return pd.DataFrame(rows)


class TestsNormalizedAmplitudeSignalOfCheeks:
    def test_normalized_amplitude_happy_path_known_values(self, cheek_center_indices):
        li, ri = cheek_center_indices

        df = _df_with_cheeks(
            frames=[0, 1],
            left_points=[(0, 0), (-1, 0)],
            right_points=[(4, 0), (5, 0)],
            li=li,
            ri=ri,
        )

        out = normalized_amplitude_signal_of_cheeks(df)

        assert list(out.columns) == ["frame_number", "normalized_amplitude_signal_of_cheeks"]
        assert out["frame_number"].tolist() == [0, 1]

        assert out.loc[out["frame_number"] == 0, "normalized_amplitude_signal_of_cheeks"].item() == pytest.approx(0.5)
        assert out.loc[out["frame_number"] == 1, "normalized_amplitude_signal_of_cheeks"].item() == pytest.approx(0.75)

    def test_normalized_amplitude_translation_invariant(self, cheek_center_indices):
        li, ri = cheek_center_indices

        base = _df_with_cheeks(
            frames=[0, 1, 2],
            left_points=[(0, 0), (-1, 0), (-2, 0)],
            right_points=[(4, 0), (5, 0), (6, 0)],
            li=li,
            ri=ri,
        )

        shift = np.array([100.0, -50.0])
        shifted = _df_with_cheeks(
            frames=[0, 1, 2],
            left_points=[tuple(np.array(p, dtype=float) + shift) for p in [(0, 0), (-1, 0), (-2, 0)]],
            right_points=[tuple(np.array(p, dtype=float) + shift) for p in [(4, 0), (5, 0), (6, 0)]],
            li=li,
            ri=ri,
        )

        out_base = normalized_amplitude_signal_of_cheeks(base)
        out_shifted = normalized_amplitude_signal_of_cheeks(shifted)

        np.testing.assert_allclose(
            out_base["normalized_amplitude_signal_of_cheeks"].to_numpy(),
            out_shifted["normalized_amplitude_signal_of_cheeks"].to_numpy(),
            rtol=0,
            atol=1e-12,
        )

    def test_normalized_amplitude_scale_invariant(self, cheek_center_indices):
        li, ri = cheek_center_indices

        base = _df_with_cheeks(
            frames=[0, 1, 2],
            left_points=[(0, 0), (-1, 0), (-2, 1)],
            right_points=[(4, 0), (5, 0), (6, 1)],
            li=li,
            ri=ri,
        )

        s = 3.5
        scaled = _df_with_cheeks(
            frames=[0, 1, 2],
            left_points=[(0 * s, 0 * s), (-1 * s, 0 * s), (-2 * s, 1 * s)],
            right_points=[(4 * s, 0 * s), (5 * s, 0 * s), (6 * s, 1 * s)],
            li=li,
            ri=ri,
        )

        out_base = normalized_amplitude_signal_of_cheeks(base)
        out_scaled = normalized_amplitude_signal_of_cheeks(scaled)

        np.testing.assert_allclose(
            out_base["normalized_amplitude_signal_of_cheeks"].to_numpy(),
            out_scaled["normalized_amplitude_signal_of_cheeks"].to_numpy(),
            rtol=0,
            atol=1e-12,
        )

    def test_swapping_left_and_right_cheeks_does_not_change_result(self, cheek_center_indices):
        li, ri = cheek_center_indices

        df = _df_with_cheeks(
            frames=[0, 1, 2],
            left_points=[(0, 0), (-1, 0), (-2, 0)],
            right_points=[(4, 0), (5, 0), (6, 0)],
            li=li,
            ri=ri,
        )
        swapped = _df_with_cheeks(
            frames=[0, 1, 2],
            left_points=[(4, 0), (5, 0), (6, 0)],
            right_points=[(0, 0), (-1, 0), (-2, 0)],
            li=li,
            ri=ri,
        )

        out = normalized_amplitude_signal_of_cheeks(df)
        out_swapped = normalized_amplitude_signal_of_cheeks(swapped)

        np.testing.assert_allclose(
            out["normalized_amplitude_signal_of_cheeks"].to_numpy(),
            out_swapped["normalized_amplitude_signal_of_cheeks"].to_numpy(),
            rtol=0,
            atol=1e-12,
        )

    def test_preserves_row_order_and_length(self, cheek_center_indices):
        li, ri = cheek_center_indices

        df = _df_with_cheeks(
            frames=[5, 3, 4],
            left_points=[(0, 0), (-1, 0), (-2, 0)],
            right_points=[(4, 0), (5, 0), (6, 0)],
            li=li,
            ri=ri,
        )
        out = normalized_amplitude_signal_of_cheeks(df)

        assert len(out) == len(df)
        assert out["frame_number"].tolist() == [5, 3, 4]

    def test_raises_when_required_columns_missing(self, cheek_center_indices):
        li, ri = cheek_center_indices

        df = pd.DataFrame(
            [
                {"frame_number": 0, f"{li}_x": 0.0, f"{li}_y": 0.0, f"{ri}_x": 4.0},  # missing f"{ri}_y"
                {"frame_number": 1, f"{li}_x": -1.0, f"{li}_y": 0.0, f"{ri}_x": 5.0},
            ]
        )

        with pytest.raises(KeyError):
            normalized_amplitude_signal_of_cheeks(df)

    def test_raises_when_reference_frame_not_found(self, cheek_center_indices):
        li, ri = cheek_center_indices

        df = _df_with_cheeks(
            frames=[0, 1],
            left_points=[(0, 0), (-1, 0)],
            right_points=[(4, 0), (5, 0)],
            li=li,
            ri=ri,
        )
        df.loc[0, "frame_number"] = np.nan

        with pytest.raises(ValueError, match="Frame 0 not found|Frame 0 not found in landmarks data"):
            normalized_amplitude_signal_of_cheeks(df)

    def test_raises_when_reference_cheek_distance_is_zero(self, cheek_center_indices):
        li, ri = cheek_center_indices

        df = _df_with_cheeks(
            frames=[0, 1],
            left_points=[(1, 1), (0, 0)],
            right_points=[(1, 1), (2, 0)],
            li=li,
            ri=ri,
        )

        with pytest.raises(ValueError, match="Invalid reference cheek positions"):
            normalized_amplitude_signal_of_cheeks(df)


class TestExtractCheekFeatures:
    @pytest.fixture
    def setup_landmarks_and_smile(self, cheek_center_indices):
        li, ri = cheek_center_indices

        landmarks_df = _df_with_cheeks(
            frames=[0, 1, 2, 3],
            left_points=[(0, 0), (-1, 0), (-2, 1), (-3, 1)],
            right_points=[(4, 0), (5, 0), (6, 1), (7, 1)],
            li=li,
            ri=ri,
        )

        smile_phases_df = pd.DataFrame(
            {
                "frame_number": [0, 2, 3],
                "phase": ["onset", "apex", "offset"],
            }
        )

        return landmarks_df, smile_phases_df

    def test_happy_path_calls_extract_features_with_expected_df(self, monkeypatch, setup_landmarks_and_smile):
        landmarks_df, smile_phases_df = setup_landmarks_and_smile

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

        import data_preprocessing.extract_cheek_features as cheek_mod

        monkeypatch.setattr(cheek_mod, "extract_features", fake_extract_features)

        fps = 25.0
        result = extract_cheek_features(PathLikeDummy("landmarks.csv"), PathLikeDummy("smile.csv"), fps)

        pd.testing.assert_frame_equal(result, pd.DataFrame({"some_feature": [1, 2, 3]}))

        assert captured["fps"] == fps

        df_passed = captured["df"]
        expected_cols = {"frame_number", "D", "V", "A", "phase"}

        assert expected_cols.issubset(set(df_passed.columns))
        assert df_passed["frame_number"].tolist() == [0, 2, 3]

        merged_frames = df_passed["frame_number"].tolist()

        D = (
            normalized_amplitude_signal_of_cheeks(landmarks_df)
            .set_index("frame_number")
            .loc[merged_frames, "normalized_amplitude_signal_of_cheeks"]
            .reset_index(drop=True)
        )
        V = D.diff().fillna(0.0)
        A = D.diff().diff().fillna(0.0)

        np.testing.assert_allclose(df_passed["D"].to_numpy(), D.to_numpy(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(df_passed["V"].to_numpy(), V.to_numpy(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(df_passed["A"].to_numpy(), A.to_numpy(), rtol=0, atol=1e-12)

        assert not df_passed.isna().any().any()

    def test_raises_when_reading_files_fails(self, monkeypatch):
        def boom(_):
            raise IOError("failed to read")

        monkeypatch.setattr(pd, "read_csv", boom)

        with pytest.raises(Exception):
            extract_cheek_features(PathLikeDummy("landmarks.csv"), PathLikeDummy("smile.csv"), 30.0)


class PathLikeDummy:
    """Minimal path-like object exposing .name used by the function under test."""

    def __init__(self, name: str):
        self.name = name
