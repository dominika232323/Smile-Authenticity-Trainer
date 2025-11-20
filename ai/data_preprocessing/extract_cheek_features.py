from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ai.data_preprocessing.extract_features import extract_features
from ai.data_preprocessing.face_landmarks import FaceLandmarks


def extract_cheek_features(landmarks_file_path: Path, smile_phases_file_path: Path, video_fps: float) -> pd.DataFrame:
    logger.info(f"Starting cheek feature extraction from {landmarks_file_path.name} with FPS={video_fps}")

    try:
        landmarks_df = pd.read_csv(landmarks_file_path)
        logger.debug(f"Loaded landmarks data: {landmarks_df.shape[0]} frames, {landmarks_df.shape[1]} columns")

        smile_phases_df = pd.read_csv(smile_phases_file_path)
        logger.debug(f"Loaded smile phases data: {smile_phases_df.shape[0]} frames")

    except Exception as e:
        logger.error(f"Failed to load input files: {e}")
        raise

    logger.debug("Computing normalized amplitude signal of cheeks")
    cheeks_features_df = normalized_amplitude_signal_of_cheeks(landmarks_df)

    logger.debug("Merging landmarks with smile phases data")
    cheeks_features_df = pd.merge(cheeks_features_df, smile_phases_df, on="frame_number")
    logger.debug(f"Merged data shape: {cheeks_features_df.shape}")

    logger.debug("Computing speed and acceleration derivatives")
    cheeks_features_df["speed"] = cheeks_features_df["normalized_amplitude_signal_of_cheeks"].diff()
    cheeks_features_df["acceleration"] = cheeks_features_df["speed"].diff()

    nan_count = cheeks_features_df.isna().sum().sum()
    logger.debug(f"Filling {nan_count} NaN values with zeros")
    cheeks_features_df = cheeks_features_df.fillna(0)

    cheeks_features_df = cheeks_features_df.rename(
        columns={"normalized_amplitude_signal_of_cheeks": "D", "speed": "V", "acceleration": "A"}
    )

    logger.debug("Extracting final features using feature extraction pipeline")
    features_df = extract_features(cheeks_features_df, video_fps)

    logger.info(f"Cheek feature extraction completed successfully, extracted {features_df.shape[1]} features")
    return features_df


def normalized_amplitude_signal_of_cheeks(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Computing normalized amplitude signal of cheeks")

    frame_0 = landmarks_df.loc[landmarks_df["frame_number"] == 0]

    if frame_0.empty:
        logger.error("Frame 0 not found in landmarks data - cannot establish reference points")
        raise ValueError("Frame 0 not found in landmarks data")

    right_cheek_landmark_index = FaceLandmarks.right_cheek_center()[0]
    left_cheek_landmark_index = FaceLandmarks.left_cheek_center()[0]

    logger.debug(f"Using cheek landmarks: left={left_cheek_landmark_index}, right={right_cheek_landmark_index}")

    right_cheek_ref = np.array([frame_0[f"{right_cheek_landmark_index}_x"], frame_0[f"{right_cheek_landmark_index}_y"]])
    left_cheek_ref = np.array([frame_0[f"{left_cheek_landmark_index}_x"], frame_0[f"{left_cheek_landmark_index}_y"]])

    cheeks_midpoint_ref = (right_cheek_ref + left_cheek_ref) / 2
    denominator = 2 * np.linalg.norm(right_cheek_ref - left_cheek_ref)

    logger.debug(f"Reference cheek distance: {denominator / 2:.2f} pixels")
    logger.debug(
        f"Reference midpoint: ({cheeks_midpoint_ref.flatten()[0]:.1f}, {cheeks_midpoint_ref.flatten()[1]:.1f})"
    )

    if denominator == 0:
        logger.error("Reference cheek distance is zero - invalid landmark data")
        raise ValueError("Invalid reference cheek positions")

    def compute_D_cheek(row: pd.Series) -> np.floating[Any]:
        right_cheek_frame_t = np.array([row[f"{right_cheek_landmark_index}_x"], row[f"{right_cheek_landmark_index}_y"]])
        left_cheek_frame_t = np.array([row[f"{left_cheek_landmark_index}_x"], row[f"{left_cheek_landmark_index}_y"]])

        distance_1 = np.linalg.norm(cheeks_midpoint_ref - right_cheek_frame_t)
        distance_2 = np.linalg.norm(cheeks_midpoint_ref - left_cheek_frame_t)

        return (distance_1 + distance_2) / denominator

    logger.debug(f"Computing normalized amplitude for {len(landmarks_df)} frames")
    landmarks_df["normalized_amplitude_signal_of_cheeks"] = landmarks_df.apply(compute_D_cheek, axis=1)

    result_df = landmarks_df[["frame_number", "normalized_amplitude_signal_of_cheeks"]]
    logger.debug(f"Normalized amplitude signal computation completed for {len(result_df)} frames")

    return result_df
