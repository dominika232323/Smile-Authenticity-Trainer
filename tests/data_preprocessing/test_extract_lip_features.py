import numpy as np
import pandas as pd
import pytest

from ai.data_preprocessing.extract_lip_features import normalized_amplitude_signal_of_lip_corners
from ai.data_preprocessing.face_landmarks import FaceLandmarks


@pytest.fixture
def lip_corner_indices(monkeypatch):
    monkeypatch.setattr(FaceLandmarks, "left_lip_corner", staticmethod(lambda: [10]))
    monkeypatch.setattr(FaceLandmarks, "right_lip_corner", staticmethod(lambda: [20]))

    return 10, 20


def _df_with_lip_corners(frames, left_points, right_points, li, ri):
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


class TestNormalizedAmplitudeSignalOfLipCorners:
    def test_normalized_amplitude_happy_path_known_values(self, lip_corner_indices):
        li, ri = lip_corner_indices

        df = _df_with_lip_corners(
            frames=[0, 1],
            left_points=[(0, 0), (-1, 0)],
            right_points=[(4, 0), (5, 0)],
            li=li,
            ri=ri,
        )

        out = normalized_amplitude_signal_of_lip_corners(df)

        assert list(out.columns) == ["frame_number", "normalized_amplitude_signal_of_lip_corners"]
        assert out["frame_number"].tolist() == [0, 1]

        assert out.loc[out["frame_number"] == 0, "normalized_amplitude_signal_of_lip_corners"].item() == pytest.approx(
            0.5
        )
        assert out.loc[out["frame_number"] == 1, "normalized_amplitude_signal_of_lip_corners"].item() == pytest.approx(
            0.75
        )

    def test_normalized_amplitude_translation_invariant(self, lip_corner_indices):
        li, ri = lip_corner_indices

        base = _df_with_lip_corners(
            frames=[0, 1, 2],
            left_points=[(0, 0), (-1, 0), (-2, 0)],
            right_points=[(4, 0), (5, 0), (6, 0)],
            li=li,
            ri=ri,
        )

        shift = np.array([100.0, -50.0])
        shifted = _df_with_lip_corners(
            frames=[0, 1, 2],
            left_points=[tuple(np.array(p, dtype=float) + shift) for p in [(0, 0), (-1, 0), (-2, 0)]],
            right_points=[tuple(np.array(p, dtype=float) + shift) for p in [(4, 0), (5, 0), (6, 0)]],
            li=li,
            ri=ri,
        )

        out_base = normalized_amplitude_signal_of_lip_corners(base)
        out_shifted = normalized_amplitude_signal_of_lip_corners(shifted)

        np.testing.assert_allclose(
            out_base["normalized_amplitude_signal_of_lip_corners"].to_numpy(),
            out_shifted["normalized_amplitude_signal_of_lip_corners"].to_numpy(),
            rtol=0,
            atol=1e-12,
        )

    def test_normalized_amplitude_scale_invariant(self, lip_corner_indices):
        li, ri = lip_corner_indices

        base = _df_with_lip_corners(
            frames=[0, 1, 2],
            left_points=[(0, 0), (-1, 0), (-2, 1)],
            right_points=[(4, 0), (5, 0), (6, 1)],
            li=li,
            ri=ri,
        )

        s = 3.5
        scaled = _df_with_lip_corners(
            frames=[0, 1, 2],
            left_points=[(0 * s, 0 * s), (-1 * s, 0 * s), (-2 * s, 1 * s)],
            right_points=[(4 * s, 0 * s), (5 * s, 0 * s), (6 * s, 1 * s)],
            li=li,
            ri=ri,
        )

        out_base = normalized_amplitude_signal_of_lip_corners(base)
        out_scaled = normalized_amplitude_signal_of_lip_corners(scaled)

        np.testing.assert_allclose(
            out_base["normalized_amplitude_signal_of_lip_corners"].to_numpy(),
            out_scaled["normalized_amplitude_signal_of_lip_corners"].to_numpy(),
            rtol=0,
            atol=1e-12,
        )

    def test_swapping_left_and_right_corners_does_not_change_result(self, lip_corner_indices):
        li, ri = lip_corner_indices

        df = _df_with_lip_corners(
            frames=[0, 1, 2],
            left_points=[(0, 0), (-1, 0), (-2, 0)],
            right_points=[(4, 0), (5, 0), (6, 0)],
            li=li,
            ri=ri,
        )
        swapped = _df_with_lip_corners(
            frames=[0, 1, 2],
            left_points=[(4, 0), (5, 0), (6, 0)],
            right_points=[(0, 0), (-1, 0), (-2, 0)],
            li=li,
            ri=ri,
        )

        out = normalized_amplitude_signal_of_lip_corners(df)
        out_swapped = normalized_amplitude_signal_of_lip_corners(swapped)

        np.testing.assert_allclose(
            out["normalized_amplitude_signal_of_lip_corners"].to_numpy(),
            out_swapped["normalized_amplitude_signal_of_lip_corners"].to_numpy(),
            rtol=0,
            atol=1e-12,
        )

    def test_preserves_row_order_and_length(self, lip_corner_indices):
        li, ri = lip_corner_indices

        df = _df_with_lip_corners(
            frames=[5, 3, 4],
            left_points=[(0, 0), (-1, 0), (-2, 0)],
            right_points=[(4, 0), (5, 0), (6, 0)],
            li=li,
            ri=ri,
        )
        out = normalized_amplitude_signal_of_lip_corners(df)

        assert len(out) == len(df)
        assert out["frame_number"].tolist() == [5, 3, 4]

    def test_raises_when_required_columns_missing(self, lip_corner_indices):
        li, ri = lip_corner_indices

        df = pd.DataFrame(
            [
                {"frame_number": 0, f"{li}_x": 0.0, f"{li}_y": 0.0, f"{ri}_x": 4.0},  # missing f"{ri}_y"
                {"frame_number": 1, f"{li}_x": -1.0, f"{li}_y": 0.0, f"{ri}_x": 5.0},
            ]
        )

        with pytest.raises(KeyError):
            normalized_amplitude_signal_of_lip_corners(df)

    def test_raises_when_reference_frame_not_found(self, lip_corner_indices):
        li, ri = lip_corner_indices

        df = _df_with_lip_corners(
            frames=[0, 1],
            left_points=[(0, 0), (-1, 0)],
            right_points=[(4, 0), (5, 0)],
            li=li,
            ri=ri,
        )
        df.loc[0, "frame_number"] = np.nan

        with pytest.raises(ValueError, match="Frame 0 not found|Frame 0 not found in landmarks data"):
            normalized_amplitude_signal_of_lip_corners(df)

    def test_raises_when_reference_lip_width_is_zero(self, lip_corner_indices):
        li, ri = lip_corner_indices

        df = _df_with_lip_corners(
            frames=[0, 1],
            left_points=[(1, 1), (0, 0)],
            right_points=[(1, 1), (2, 0)],
            li=li,
            ri=ri,
        )

        with pytest.raises(ValueError, match="Invalid reference lip corner positions"):
            normalized_amplitude_signal_of_lip_corners(df)
