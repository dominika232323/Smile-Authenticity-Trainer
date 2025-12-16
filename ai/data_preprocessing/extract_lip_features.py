from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ai.data_preprocessing.extract_features import extract_features
from ai.data_preprocessing.face_landmarks import FaceLandmarks


def extract_lip_features(landmarks_file_path: Path, smile_phases_file_path: Path, video_fps: float) -> pd.DataFrame:
    logger.info(f"Starting lip feature extraction from {landmarks_file_path.name} with FPS={video_fps}")

    try:
        landmarks_df = pd.read_csv(landmarks_file_path)
        logger.debug(f"Loaded landmarks data: {landmarks_df.shape[0]} frames, {landmarks_df.shape[1]} columns")

        smile_phases_df = pd.read_csv(smile_phases_file_path)
        logger.debug(f"Loaded smile phases data: {smile_phases_df.shape[0]} frames")

    except Exception as e:
        logger.error(f"Failed to load input files: {e}")
        raise

    logger.debug("Computing normalized amplitude signal of lip corners")
    lips_features_df = normalized_amplitude_signal_of_lip_corners(landmarks_df)

    logger.debug("Merging landmarks with smile phases data")
    lips_features_df = (
        pd.merge(lips_features_df, smile_phases_df, on="frame_number", how="inner")
        .sort_values("frame_number")
        .reset_index(drop=True)
    )
    logger.debug(f"Merged data shape: {lips_features_df.shape}")

    logger.debug("Computing speed and acceleration derivatives")
    D_series = lips_features_df["normalized_amplitude_signal_of_lip_corners"].astype(float)
    V_series = D_series.diff().fillna(0.0)
    A_series = V_series.diff().fillna(0.0)

    lips_features_df["speed"] = V_series
    lips_features_df["acceleration"] = A_series

    nan_count = lips_features_df.isna().sum().sum()
    logger.debug(f"Filling {nan_count} NaN values with zeros")
    lips_features_df = lips_features_df.fillna(0.0)

    lips_features_df = lips_features_df.rename(
        columns={"normalized_amplitude_signal_of_lip_corners": "D", "speed": "V", "acceleration": "A"}
    )

    logger.debug("Extracting final features using feature extraction pipeline")
    features_df = extract_features(lips_features_df, video_fps)

    logger.info(f"Lip feature extraction completed successfully, extracted {features_df.shape[1]} features")
    return features_df


def normalized_amplitude_signal_of_lip_corners(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Computing normalized amplitude signal of lip corners")

    frame_ref = landmarks_df.loc[landmarks_df["frame_number"] == landmarks_df.iloc[0]["frame_number"]]

    if frame_ref.empty:
        logger.error("Frame 0 not found in landmarks data - cannot establish reference points")
        raise ValueError("Frame 0 not found in landmarks data")

    left_lip_corner_landmark_index = FaceLandmarks.left_lip_corner()[0]
    right_lip_corner_landmark_index = FaceLandmarks.right_lip_corner()[0]

    logger.debug(
        f"Using lip corner landmarks: left={left_lip_corner_landmark_index}, right={right_lip_corner_landmark_index}"
    )

    left_lip_corner_ref = np.array(
        [
            float(frame_ref[f"{left_lip_corner_landmark_index}_x"].iloc[0]),
            float(frame_ref[f"{left_lip_corner_landmark_index}_y"].iloc[0]),
        ],
        dtype=float,
    )
    right_lip_corner_ref = np.array(
        [
            float(frame_ref[f"{right_lip_corner_landmark_index}_x"].iloc[0]),
            float(frame_ref[f"{right_lip_corner_landmark_index}_y"].iloc[0]),
        ],
        dtype=float,
    )

    lips_midpoint_ref = (right_lip_corner_ref + left_lip_corner_ref) / 2
    denominator = 2 * np.linalg.norm(right_lip_corner_ref - left_lip_corner_ref)

    logger.debug(f"Reference lip width: {denominator / 2:.2f} pixels")
    logger.debug(f"Reference midpoint: ({lips_midpoint_ref.flatten()[0]:.1f}, {lips_midpoint_ref.flatten()[1]:.1f})")

    if np.isclose(denominator, 0.0):
        logger.error("Reference lip corner distance is zero - invalid landmark data")
        raise ValueError("Invalid reference lip corner positions")

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

    logger.debug(f"Computing normalized amplitude for {len(landmarks_df)} frames")
    landmarks_df["normalized_amplitude_signal_of_lip_corners"] = landmarks_df.apply(compute_D_lip, axis=1)

    result_df = landmarks_df[["frame_number", "normalized_amplitude_signal_of_lip_corners"]]
    logger.debug(f"Normalized amplitude signal computation completed for {len(result_df)} frames")

    return result_df
