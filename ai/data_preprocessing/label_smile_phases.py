from pathlib import Path

import numpy as np
import pandas as pd

from ai.data_preprocessing.face_landmarks import FaceLandmarks
from ai.data_preprocessing.file_utils import save_dataframe_to_csv


def label_smile_phases(landmarks_file_path: Path, output_csv: Path, smoothing_window: int = 5) -> None:
    landmarks_df = pd.read_csv(landmarks_file_path)

    radius = calculate_radius(landmarks_df)
    dist_smooth = smooth_radius(radius, smoothing_window)
    delta = np.diff(dist_smooth, prepend=dist_smooth[0])

    onset_start, onset_end = detect_longest_segment(delta, condition=lambda d: d > 0)
    offset_start, offset_end = detect_longest_segment(delta, condition=lambda d: d < 0)

    phases = label_phases(len(landmarks_df), onset_start, onset_end, offset_start, offset_end)

    smile_phases_df = pd.DataFrame(
        {
            "frame_number": landmarks_df["frame_number"],
            "smile_phase": phases,
            "radius": radius,
        }
    )

    save_dataframe_to_csv(smile_phases_df, output_csv)


def calculate_radius(landmarks_df: pd.DataFrame) -> np.ndarray:
    left_lip_corner_coords_x = landmarks_df[f"{FaceLandmarks.left_lip_corner()[0]}_x"].values
    left_lip_corner_coords_y = landmarks_df[f"{FaceLandmarks.left_lip_corner()[0]}_y"].values
    right_lip_corner_coords_x = landmarks_df[f"{FaceLandmarks.right_lip_corner()[0]}_x"].values
    right_lip_corner_coords_y = landmarks_df[f"{FaceLandmarks.right_lip_corner()[0]}_y"].values

    lips_midpoint_coords_x = (left_lip_corner_coords_x + right_lip_corner_coords_x) / 2
    lips_midpoint_coords_y = (left_lip_corner_coords_y + right_lip_corner_coords_y) / 2

    radius = np.sqrt(
        (right_lip_corner_coords_x - lips_midpoint_coords_x) ** 2
        + (right_lip_corner_coords_y - lips_midpoint_coords_y) ** 2
    )

    return radius


def smooth_radius(radius: np.ndarray, smoothing_window: int) -> np.ndarray:
    return pd.Series(radius).rolling(smoothing_window, center=True, min_periods=1).mean().values


def detect_longest_segment(delta: np.ndarray, condition) -> tuple[int | None, int | None]:
    start = None
    best_length = 0
    segment_start = None
    segment_end = None

    for i in range(1, len(delta)):
        if condition(delta[i]):
            if start is None:
                start = i
        else:
            if start is not None:
                segment_length = i - start

                if segment_length > best_length:
                    best_length = segment_length
                    segment_start = start
                    segment_end = i - 1

                start = None

    if start is not None:
        segment_length = len(delta) - start

        if segment_length > best_length:
            segment_start = start
            segment_end = len(delta) - 1

    return segment_start, segment_end


def label_phases(
    num_frames: int, onset_start: int | None, onset_end: int | None, offset_start: int | None, offset_end: int | None
) -> list[str]:
    phases = ["neutral"] * num_frames

    apex_start = onset_end + 1 if onset_end is not None else None
    apex_end = offset_start - 1 if offset_start is not None else None

    if onset_start is not None and onset_end is not None:
        for i in range(onset_start, onset_end + 1):
            phases[i] = "onset"

    if apex_start is not None and apex_end is not None and apex_start <= apex_end:
        for i in range(apex_start, apex_end + 1):
            phases[i] = "apex"

    if offset_start is not None and offset_end is not None:
        for i in range(offset_start, offset_end + 1):
            phases[i] = "offset"

    if onset_start is not None:
        for i in range(0, onset_start):
            phases[i] = "neutral"

    if offset_end is not None:
        for i in range(offset_end + 1, num_frames):
            phases[i] = "neutral"

    return phases
