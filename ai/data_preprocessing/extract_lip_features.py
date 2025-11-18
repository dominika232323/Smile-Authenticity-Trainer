from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai.data_preprocessing.extract_features import extract_features
from ai.data_preprocessing.face_landmarks import FaceLandmarks


def extract_lip_features(landmarks_file_path: Path, smile_phases_file_path: Path, video_fps: float) -> pd.DataFrame:
    landmarks_df = pd.read_csv(landmarks_file_path)
    smile_phases_df = pd.read_csv(smile_phases_file_path)

    lips_features_df = normalized_amplitude_signal_of_lip_corners(landmarks_df)
    lips_features_df = pd.merge(lips_features_df, smile_phases_df, on="frame_number")

    lips_features_df["speed"] = lips_features_df["normalized_amplitude_signal_of_lip_corners"].diff()
    lips_features_df["acceleration"] = lips_features_df["speed"].diff()

    lips_features_df = lips_features_df.fillna(0)

    lips_features_df = lips_features_df.rename(
        columns={"normalized_amplitude_signal_of_lip_corners": "D", "speed": "V", "acceleration": "A"}
    )

    return extract_features(lips_features_df, video_fps)


def normalized_amplitude_signal_of_lip_corners(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    frame_0 = landmarks_df.loc[landmarks_df["frame_number"] == 0]

    left_lip_corner_landmark_index = FaceLandmarks.left_lip_corner()[0]
    right_lip_corner_landmark_index = FaceLandmarks.right_lip_corner()[0]

    left_lip_corner_ref = np.array(
        [frame_0[f"{left_lip_corner_landmark_index}_x"], frame_0[f"{left_lip_corner_landmark_index}_y"]]
    )
    right_lip_corner_ref = np.array(
        [frame_0[f"{right_lip_corner_landmark_index}_x"], frame_0[f"{right_lip_corner_landmark_index}_y"]]
    )

    lips_midpoint_ref = (right_lip_corner_ref + left_lip_corner_ref) / 2

    denominator = 2 * np.linalg.norm(right_lip_corner_ref - left_lip_corner_ref)

    def compute_D_lip(row: pd.Series) -> np.floating[Any]:
        left_lip_corner_frame_t = np.array(
            [row[f"{left_lip_corner_landmark_index}_x"], row[f"{left_lip_corner_landmark_index}_y"]]
        )
        right_lip_corner_frame_t = np.array(
            [row[f"{right_lip_corner_landmark_index}_x"], row[f"{right_lip_corner_landmark_index}_y"]]
        )

        distance_1 = np.linalg.norm(lips_midpoint_ref - right_lip_corner_frame_t)
        distance_2 = np.linalg.norm(lips_midpoint_ref - left_lip_corner_frame_t)

        return (distance_1 + distance_2) / denominator

    landmarks_df["normalized_amplitude_signal_of_lip_corners"] = landmarks_df.apply(compute_D_lip, axis=1)

    return landmarks_df[["frame_number", "normalized_amplitude_signal_of_lip_corners"]]
