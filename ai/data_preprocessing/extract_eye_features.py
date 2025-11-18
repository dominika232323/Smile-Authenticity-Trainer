from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ai.data_preprocessing.extract_features import extract_features
from ai.data_preprocessing.face_landmarks import FaceLandmarks


def extract_eye_features(landmarks_file_path: Path, smile_phases_file_path: Path, video_fps: float) -> pd.DataFrame:
    landmarks_df = pd.read_csv(landmarks_file_path)
    smile_phases_df = pd.read_csv(smile_phases_file_path)

    eyes_features_df = pd.DataFrame(
        {
            "frame_number": landmarks_df["frame_number"],
            "normalized_amplitude_signal_of_eyelids": compute_eyelid_amplitude(landmarks_df),
        }
    )

    eyes_features_df = pd.merge(eyes_features_df, smile_phases_df, on="frame_number")

    eyes_features_df["speed"] = eyes_features_df["normalized_amplitude_signal_of_eyelids"].diff()
    eyes_features_df["acceleration"] = eyes_features_df["speed"].diff()

    eyes_features_df = eyes_features_df.fillna(0)

    eyes_features_df = eyes_features_df.rename(
        columns={"normalized_amplitude_signal_of_eyelids": "D", "speed": "V", "acceleration": "A"}
    )

    return extract_features(eyes_features_df, video_fps)


def compute_eyelid_amplitude(df: pd.DataFrame) -> np.ndarray:
    right_eye_right_corner_landmark_index = FaceLandmarks.right_eye_right_corner()[0]
    right_eyelid_middle = FaceLandmarks.right_eye_upper_0_middle()[0]
    right_eye_left_corner_landmark_index = FaceLandmarks.right_eye_left_corner()[0]

    left_eye_right_corner_landmark_index = FaceLandmarks.left_eye_right_corner()[0]
    left_eyelid_middle = FaceLandmarks.left_eye_upper_0_middle()[0]
    left_eye_left_corner_landmark_index = FaceLandmarks.left_eye_left_corner()[0]

    l1 = df[[f"{right_eye_right_corner_landmark_index}_x", f"{right_eye_right_corner_landmark_index}_y"]].values
    l2 = df[[f"{right_eyelid_middle}_x", f"{right_eyelid_middle}_y"]].values
    l3 = df[[f"{right_eye_left_corner_landmark_index}_x", f"{right_eye_left_corner_landmark_index}_y"]].values

    l4 = df[[f"{left_eye_right_corner_landmark_index}_x", f"{left_eye_right_corner_landmark_index}_y"]].values
    l5 = df[[f"{left_eyelid_middle}_x", f"{left_eyelid_middle}_y"]].values
    l6 = df[[f"{left_eye_left_corner_landmark_index}_x", f"{left_eye_left_corner_landmark_index}_y"]].values

    right_eye_midpoint = (l1 + l3) / 2
    left_eye_midpoint = (l4 + l6) / 2

    tau_1 = tau(right_eye_midpoint, l2)
    tau_2 = tau(left_eye_midpoint, l5)

    denominator = 2 * euclidean(l1, l3)

    D_eyelid = (tau_1 + tau_2) / denominator

    return D_eyelid


def euclidean(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.linalg.norm(a - b, axis=1)


def kappa(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.int_]:
    return np.where(b[:, 1] > a[:, 1], -1, 1)


def tau(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.floating]:
    return kappa(a, b).astype(float) * euclidean(a, b)
