from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ai.data_preprocessing.face_landmarks import FaceLandmarks
from ai.data_preprocessing.file_utils import save_dataframe_to_csv


def label_smile_phases(landmarks_file_path: Path, output_csv: Path, smoothing_window: int = 5) -> None:
    logger.info(f"Starting smile phase labeling for file: {landmarks_file_path}")

    landmarks_df = pd.read_csv(landmarks_file_path)
    logger.info(f"Loaded landmarks dataframe with {len(landmarks_df)} rows")

    if landmarks_df.empty:
        logger.error("Landmarks dataframe is empty")
        raise ValueError("Landmarks dataframe is empty")

    radius = calculate_radius(landmarks_df)
    logger.debug(f"Calculated radius - min: {radius.min():.3f}, max: {radius.max():.3f}, mean: {radius.mean():.3f}")

    dist_smooth = smooth_radius(radius, smoothing_window)
    delta = np.diff(dist_smooth, prepend=dist_smooth[0])
    logger.debug(f"Delta values - min: {delta.min():.3f}, max: {delta.max():.3f}")

    onset_start, onset_end = detect_longest_segment(delta, condition=lambda d: d > 0)
    logger.debug(f"Onset phase detected: start={onset_start}, end={onset_end}")

    offset_start, offset_end = detect_longest_segment(delta, condition=lambda d: d < 0)
    logger.debug(f"Offset phase detected: start={offset_start}, end={offset_end}")

    phases = label_phases(len(landmarks_df), onset_start, onset_end, offset_start, offset_end)

    phase_counts = pd.Series(phases).value_counts().to_dict()
    logger.debug(f"Phase distribution: {phase_counts}")

    smile_phases_df = pd.DataFrame(
        {
            "frame_number": landmarks_df["frame_number"],
            "smile_phase": phases,
            "radius": radius,
        }
    )

    logger.debug(f"Saving results to {output_csv}")
    save_dataframe_to_csv(smile_phases_df, output_csv)
    logger.info(f"Successfully completed smile phase labeling and saved results to {output_csv}")


def calculate_radius(landmarks_df: pd.DataFrame) -> np.ndarray:
    logger.debug("Extracting lip corner coordinates")
    left_lip_corner_coords_x = landmarks_df[f"{FaceLandmarks.left_lip_corner()[0]}_x"].values
    left_lip_corner_coords_y = landmarks_df[f"{FaceLandmarks.left_lip_corner()[0]}_y"].values
    right_lip_corner_coords_x = landmarks_df[f"{FaceLandmarks.right_lip_corner()[0]}_x"].values
    right_lip_corner_coords_y = landmarks_df[f"{FaceLandmarks.right_lip_corner()[0]}_y"].values

    logger.debug("Calculating lips midpoint coordinates")
    lips_midpoint_coords_x = (left_lip_corner_coords_x + right_lip_corner_coords_x) / 2
    lips_midpoint_coords_y = (left_lip_corner_coords_y + right_lip_corner_coords_y) / 2

    logger.debug("Computing radius from midpoint to right lip corner")
    radius = np.sqrt(
        (right_lip_corner_coords_x - lips_midpoint_coords_x) ** 2
        + (right_lip_corner_coords_y - lips_midpoint_coords_y) ** 2
    )

    logger.debug(f"Radius calculation completed - {len(radius)} values computed")
    return radius


def smooth_radius(radius: np.ndarray, smoothing_window: int) -> np.ndarray:
    logger.debug(f"Applying rolling window smoothing with window size {smoothing_window}")
    smoothed = pd.Series(radius).rolling(smoothing_window, center=True, min_periods=1).mean().values

    logger.debug("Radius smoothing completed")
    return smoothed


def detect_longest_segment(delta: np.ndarray, condition) -> tuple[int | None, int | None]:
    logger.debug("Starting longest segment detection")
    start = None
    best_length = 0
    segment_start = None
    segment_end = None
    segments_found = 0

    for i in range(1, len(delta)):
        if condition(delta[i]):
            if start is None:
                start = i
        else:
            if start is not None:
                segment_length = i - start
                segments_found += 1
                logger.debug(f"Segment {segments_found}: length={segment_length}, start={start}, end={i - 1}")

                if segment_length > best_length:
                    best_length = segment_length
                    segment_start = start
                    segment_end = i - 1
                    logger.debug(f"New best segment: length={best_length}, start={segment_start}, end={segment_end}")

                start = None

    if start is not None:
        segment_length = len(delta) - start
        segments_found += 1
        logger.debug(f"Final segment {segments_found}: length={segment_length}, start={start}, end={len(delta) - 1}")

        if segment_length > best_length:
            segment_start = start
            segment_end = len(delta) - 1
            logger.debug(f"Final segment is best: length={segment_length}, start={segment_start}, end={segment_end}")

    logger.debug(f"Longest segment detection completed - found {segments_found} segments total")
    return segment_start, segment_end


def label_phases(
    num_frames: int, onset_start: int | None, onset_end: int | None, offset_start: int | None, offset_end: int | None
) -> list[str]:
    logger.debug(f"Labeling phases for {num_frames} frames")
    logger.debug(f"Phase boundaries - onset: [{onset_start}, {onset_end}], offset: [{offset_start}, {offset_end}]")

    phases = ["neutral"] * num_frames

    apex_start = onset_end + 1 if onset_end is not None else None
    apex_end = offset_start - 1 if offset_start is not None else None

    if apex_start is not None and apex_end is not None:
        logger.debug(f"Apex phase boundaries: [{apex_start}, {apex_end}]")

    if onset_start is not None and onset_end is not None:
        onset_frames = onset_end - onset_start + 1

        logger.debug(f"Labeling {onset_frames} frames as 'onset' (frames {onset_start}-{onset_end})")

        for i in range(onset_start, onset_end + 1):
            phases[i] = "onset"

    if apex_start is not None and apex_end is not None and apex_start <= apex_end:
        apex_frames = apex_end - apex_start + 1

        logger.debug(f"Labeling {apex_frames} frames as 'apex' (frames {apex_start}-{apex_end})")

        for i in range(apex_start, apex_end + 1):
            phases[i] = "apex"

    if offset_start is not None and offset_end is not None:
        offset_frames = offset_end - offset_start + 1

        logger.debug(f"Labeling {offset_frames} frames as 'offset' (frames {offset_start}-{offset_end})")

        for i in range(offset_start, offset_end + 1):
            phases[i] = "offset"

    if onset_start is not None:
        pre_onset_frames = onset_start

        logger.debug(f"Labeling {pre_onset_frames} frames as 'neutral' before onset (frames 0-{onset_start - 1})")

        for i in range(0, onset_start):
            phases[i] = "neutral"

    if offset_end is not None:
        post_offset_frames = num_frames - offset_end - 1

        logger.debug(
            f"Labeling {post_offset_frames} frames as 'neutral' after offset (frames {offset_end + 1}-{num_frames - 1})"
        )

        for i in range(offset_end + 1, num_frames):
            phases[i] = "neutral"

    logger.debug("Phase labeling completed")
    return phases
