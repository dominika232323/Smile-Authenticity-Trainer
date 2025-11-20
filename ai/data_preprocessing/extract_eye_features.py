from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray

from ai.data_preprocessing.extract_features import extract_features
from ai.data_preprocessing.face_landmarks import FaceLandmarks


def extract_eye_features(landmarks_file_path: Path, smile_phases_file_path: Path, video_fps: float) -> pd.DataFrame:
    logger.info(f"Starting eye feature extraction from {landmarks_file_path.name} with FPS={video_fps}")

    try:
        landmarks_df = pd.read_csv(landmarks_file_path)
        logger.debug(f"Loaded landmarks data: {landmarks_df.shape[0]} frames, {landmarks_df.shape[1]} columns")

        smile_phases_df = pd.read_csv(smile_phases_file_path)
        logger.debug(f"Loaded smile phases data: {smile_phases_df.shape[0]} frames")

    except Exception as e:
        logger.error(f"Failed to load input files: {e}")
        raise

    logger.debug("Computing eyelid amplitude signals")
    eyelid_amplitude = compute_eyelid_amplitude(landmarks_df)

    eyes_features_df = pd.DataFrame(
        {
            "frame_number": landmarks_df["frame_number"],
            "normalized_amplitude_signal_of_eyelids": eyelid_amplitude,
        }
    )

    logger.debug("Merging eye features with smile phases data")
    eyes_features_df = pd.merge(eyes_features_df, smile_phases_df, on="frame_number")
    logger.debug(f"Merged data shape: {eyes_features_df.shape}")

    logger.debug("Computing speed and acceleration derivatives")
    eyes_features_df["speed"] = eyes_features_df["normalized_amplitude_signal_of_eyelids"].diff()
    eyes_features_df["acceleration"] = eyes_features_df["speed"].diff()

    nan_count = eyes_features_df.isna().sum().sum()
    logger.debug(f"Filling {nan_count} NaN values with zeros")
    eyes_features_df = eyes_features_df.fillna(0)

    eyes_features_df = eyes_features_df.rename(
        columns={"normalized_amplitude_signal_of_eyelids": "D", "speed": "V", "acceleration": "A"}
    )

    logger.debug("Extracting final features using feature extraction pipeline")
    features_df = extract_features(eyes_features_df, video_fps)

    logger.info(f"Eye feature extraction completed successfully, extracted {features_df.shape[1]} features")
    return features_df


def compute_eyelid_amplitude(df: pd.DataFrame) -> np.ndarray:
    logger.debug(f"Computing eyelid amplitude for {len(df)} frames")

    right_eye_right_corner_landmark_index = FaceLandmarks.right_eye_right_corner()[0]
    right_eyelid_middle = FaceLandmarks.right_eye_upper_0_middle()[0]
    right_eye_left_corner_landmark_index = FaceLandmarks.right_eye_left_corner()[0]

    left_eye_right_corner_landmark_index = FaceLandmarks.left_eye_right_corner()[0]
    left_eyelid_middle = FaceLandmarks.left_eye_upper_0_middle()[0]
    left_eye_left_corner_landmark_index = FaceLandmarks.left_eye_left_corner()[0]

    logger.debug(
        f"Using eye landmarks - Right eye: corners={right_eye_right_corner_landmark_index},{right_eye_left_corner_landmark_index}, middle={right_eyelid_middle}"
    )
    logger.debug(
        f"Using eye landmarks - Left eye: corners={left_eye_right_corner_landmark_index},{left_eye_left_corner_landmark_index}, middle={left_eyelid_middle}"
    )

    l1 = df[[f"{right_eye_right_corner_landmark_index}_x", f"{right_eye_right_corner_landmark_index}_y"]].values
    l2 = df[[f"{right_eyelid_middle}_x", f"{right_eyelid_middle}_y"]].values
    l3 = df[[f"{right_eye_left_corner_landmark_index}_x", f"{right_eye_left_corner_landmark_index}_y"]].values

    l4 = df[[f"{left_eye_right_corner_landmark_index}_x", f"{left_eye_right_corner_landmark_index}_y"]].values
    l5 = df[[f"{left_eyelid_middle}_x", f"{left_eyelid_middle}_y"]].values
    l6 = df[[f"{left_eye_left_corner_landmark_index}_x", f"{left_eye_left_corner_landmark_index}_y"]].values

    logger.debug("Computing eye midpoints and tau values")
    right_eye_midpoint = (l1 + l3) / 2
    left_eye_midpoint = (l4 + l6) / 2

    tau_1 = tau(right_eye_midpoint, l2)
    tau_2 = tau(left_eye_midpoint, l5)

    denominator = 2 * euclidean(l1, l3)
    zero_denominators = np.sum(denominator == 0)

    if zero_denominators > 0:
        logger.warning(f"Found {zero_denominators} frames with zero eye corner distance")

    D_eyelid = (tau_1 + tau_2) / denominator

    logger.debug("Eyelid amplitude computation completed")
    return D_eyelid


def euclidean(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.floating]:
    distances = np.linalg.norm(a - b, axis=1)

    logger.debug(f"Computed euclidean distances - mean: {np.mean(distances):.2f}, max: {np.max(distances):.2f}")
    return distances


def kappa(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.int_]:
    result = np.where(b[:, 1] > a[:, 1], -1, 1)
    negative_count = np.sum(result == -1)
    positive_count = np.sum(result == 1)

    logger.debug(f"Kappa orientation - negative: {negative_count}, positive: {positive_count}")
    return result


def tau(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.floating]:
    kappa_values = kappa(a, b)
    euclidean_values = euclidean(a, b)
    tau_values = kappa_values.astype(float) * euclidean_values

    logger.debug(f"Tau values - mean: {np.mean(tau_values):.3f}, std: {np.std(tau_values):.3f}")
    return tau_values
