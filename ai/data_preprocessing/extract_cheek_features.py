from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai.data_preprocessing.extract_features import extract_features
from ai.data_preprocessing.face_landmarks import FaceLandmarks


def extract_cheek_features(landmarks_file_path: Path, smile_phases_file_path: Path, video_fps: float) -> pd.DataFrame:
    landmarks_df = pd.read_csv(landmarks_file_path)
    smile_phases_df = pd.read_csv(smile_phases_file_path)

    cheeks_features_df = normalized_amplitude_signal_of_cheeks(landmarks_df)
    cheeks_features_df = pd.merge(cheeks_features_df, smile_phases_df, on="frame_number")

    cheeks_features_df["speed"] = cheeks_features_df["normalized_amplitude_signal_of_cheeks"].diff()
    cheeks_features_df["acceleration"] = cheeks_features_df["speed"].diff()

    cheeks_features_df = cheeks_features_df.fillna(0)

    cheeks_features_df = cheeks_features_df.rename(
        columns={"normalized_amplitude_signal_of_cheeks": "D", "speed": "V", "acceleration": "A"}
    )

    return extract_features(cheeks_features_df, video_fps)


def normalized_amplitude_signal_of_cheeks(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    frame_0 = landmarks_df.loc[landmarks_df["frame_number"] == 0]

    right_cheek_landmark_index = FaceLandmarks.right_cheek_center()[0]
    left_cheek_landmark_index = FaceLandmarks.left_cheek_center()[0]

    right_cheek_ref = np.array([frame_0[f"{right_cheek_landmark_index}_x"], frame_0[f"{right_cheek_landmark_index}_y"]])
    left_cheek_ref = np.array([frame_0[f"{left_cheek_landmark_index}_x"], frame_0[f"{left_cheek_landmark_index}_y"]])

    cheeks_midpoint_ref = (right_cheek_ref + left_cheek_ref) / 2

    denominator = 2 * np.linalg.norm(right_cheek_ref - left_cheek_ref)

    def compute_D_cheek(row: pd.Series) -> np.floating[Any]:
        right_cheek_frame_t = np.array([row[f"{right_cheek_landmark_index}_x"], row[f"{right_cheek_landmark_index}_y"]])
        left_cheek_frame_t = np.array([row[f"{left_cheek_landmark_index}_x"], row[f"{left_cheek_landmark_index}_y"]])

        distance_1 = np.linalg.norm(cheeks_midpoint_ref - right_cheek_frame_t)
        distance_2 = np.linalg.norm(cheeks_midpoint_ref - left_cheek_frame_t)

        return (distance_1 + distance_2) / denominator

    landmarks_df["normalized_amplitude_signal_of_cheeks"] = landmarks_df.apply(compute_D_cheek, axis=1)

    return landmarks_df[["frame_number", "normalized_amplitude_signal_of_cheeks"]]
