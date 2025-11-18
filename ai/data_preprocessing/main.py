from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from ai.config import (
    ALL_EYES_FEATURES_CSV,
    ALL_LIP_FEATURES_CSV,
    EYES_FEATURES_DIR,
    LIP_FEATURES_DIR,
    ORIGINAL_FACELANDMARKS_DIR,
    ORIGINAL_FRAMES_DIR,
    PREPROCESSED_DATA_DIR,
    PREPROCESSED_FACELANDMARKS_DIR,
    PREPROCESSED_FRAMES_DIR,
    PREPROCESSED_SMILE_PHASES_DIR,
    UvA_NEMO_SMILE_DETAILS,
    UvA_NEMO_SMILE_VIDEOS_DIR,
)
from ai.data_preprocessing.extract_eye_features import extract_eye_features
from ai.data_preprocessing.extract_lip_features import extract_lip_features
from ai.data_preprocessing.file_utils import (
    create_csv_with_header,
    create_directories,
    save_dataframe_to_csv,
    save_frame,
)
from ai.data_preprocessing.get_details import get_details
from ai.data_preprocessing.get_face_landmarks import create_facelandmarks_header, get_face_landmarks
from ai.data_preprocessing.label_smile_phases import label_smile_phases
from ai.data_preprocessing.normalize_face import normalize_face
from ai.logging_config import setup_logging


@logger.catch
def preprocess_frame(
    frame: cv2.Mat | np.ndarray[Any, np.dtype[Any]],
    frame_number: int,
    video_name: str,
    original_face_landmarks_file_path: Path,
    normalized_face_landmarks_file_path: Path,
) -> None:
    logger.debug(f"Processing frame {frame_number} for video {video_name}")

    original_frame_path = ORIGINAL_FRAMES_DIR / f"{video_name}" / f"{frame_number}.jpg"
    save_frame(frame, original_frame_path)
    logger.debug(f"Saved frame to {original_frame_path}")

    get_face_landmarks(frame, frame_number, original_face_landmarks_file_path)
    logger.debug(f"Extracted face landmarks for frame {frame_number}")

    normalized_frame_path = PREPROCESSED_FRAMES_DIR / f"{video_name}" / f"{frame_number}.jpg"
    normalized_frame = normalize_face(frame, original_face_landmarks_file_path, frame_number, (0.35, 0.35), (256, 256))
    save_frame(normalized_frame, normalized_frame_path)
    logger.debug(
        f"Saved normalized frame to {normalized_frame_path} (face landmarks saved to {original_face_landmarks_file_path})"
    )

    get_face_landmarks(normalized_frame, frame_number, normalized_face_landmarks_file_path)
    logger.debug(f"Extracted normalized face landmarks for frame {frame_number}")


@logger.catch
def preprocess_video(
    video_path: Path, original_face_landmarks_file_path: Path, normalized_face_landmarks_file_path: Path
) -> float | None:
    video_name = video_path.stem
    logger.info(f"Starting preprocessing video: {video_name}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video {video_name} has {total_frames} frames")

    for frame_number in tqdm(
        range(0, total_frames), desc=f"Preprocessing frames for video {video_name}", colour="yellow"
    ):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            logger.warning(f"Failed to read frame {frame_number} from video {video_name}")

            continue

        preprocess_frame(
            frame, frame_number, video_name, original_face_landmarks_file_path, normalized_face_landmarks_file_path
        )

        if frame_number % 100 == 0:
            logger.info(f"Processed {frame_number}/{total_frames} frames for video {video_name}")

    cap.release()
    logger.info(f"Finished processing all frames for video {video_name}")

    return fps


@logger.catch
def main() -> None:
    setup_logging()
    logger.info("Starting data preprocessing pipeline")

    directories = [
        PREPROCESSED_DATA_DIR,
        ORIGINAL_FRAMES_DIR,
        ORIGINAL_FACELANDMARKS_DIR,
        PREPROCESSED_FRAMES_DIR,
        PREPROCESSED_FACELANDMARKS_DIR,
        PREPROCESSED_SMILE_PHASES_DIR,
        LIP_FEATURES_DIR,
        EYES_FEATURES_DIR,
    ]
    create_directories(directories)

    logger.info(f"Processing UvA-NEMO SMILE DATABASE details file: {UvA_NEMO_SMILE_DETAILS}")

    try:
        details_df = get_details(UvA_NEMO_SMILE_DETAILS)
        details_path = PREPROCESSED_DATA_DIR / "details.csv"
        details_df.to_csv(details_path, index=False)

        logger.info(f"Saved UvA-NEMO SMILE DATABASE details to {details_path} ({len(details_df)} records)")
    except Exception as e:
        logger.error(f"Failed to process UvA-NEMO SMILE DATABASE details file: {e}")

        return

    videos_to_process = [
        UvA_NEMO_SMILE_VIDEOS_DIR / "001_deliberate_smile_2.mp4",
        UvA_NEMO_SMILE_VIDEOS_DIR / "001_deliberate_smile_3.mp4",
    ]

    logger.info(f"Processing {len(videos_to_process)} videos")

    lip_features_list = []
    eyes_features_list = []

    for i, video_path in enumerate(videos_to_process, 1):
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            continue

        video_name = video_path.stem

        original_video_directory = ORIGINAL_FRAMES_DIR / video_name
        preprocessed_video_directory = PREPROCESSED_FRAMES_DIR / video_name
        create_directories([original_video_directory, preprocessed_video_directory])

        landmarks_header = create_facelandmarks_header()

        original_face_landmarks_file_path = ORIGINAL_FACELANDMARKS_DIR / f"{video_name}.csv"
        create_csv_with_header(original_face_landmarks_file_path, landmarks_header)
        logger.info(f"Added header to face landmarks CSV: {original_face_landmarks_file_path}")

        normalized_face_landmarks_file_path = PREPROCESSED_FACELANDMARKS_DIR / f"{video_name}.csv"
        create_csv_with_header(normalized_face_landmarks_file_path, landmarks_header)
        logger.info(f"Added header to normalized face landmarks CSV: {normalized_face_landmarks_file_path}")

        logger.info(f"Processing video {i}/{len(videos_to_process)}: {video_path.name}")
        video_fps = preprocess_video(video_path, original_face_landmarks_file_path, normalized_face_landmarks_file_path)

        if video_fps is None:
            continue

        logger.info(f"Labeling smile phases for video {video_name}")
        smile_phase_file_path = PREPROCESSED_SMILE_PHASES_DIR / f"{video_name}.csv"
        label_smile_phases(normalized_face_landmarks_file_path, smile_phase_file_path)

        logger.info(f"Extracting lip features for video {video_name}")
        video_lip_features_df = extract_lip_features(
            normalized_face_landmarks_file_path, smile_phase_file_path, video_fps
        )
        video_lip_features_df["filename"] = video_path.name

        save_dataframe_to_csv(video_lip_features_df, LIP_FEATURES_DIR / f"{video_name}.csv")

        lip_features_list.append(video_lip_features_df)

        logger.info(f"Extracting eye features for video {video_name}")
        video_eyes_features_df = extract_eye_features(
            normalized_face_landmarks_file_path, smile_phase_file_path, video_fps
        )
        video_eyes_features_df["filename"] = video_path.name

        save_dataframe_to_csv(video_eyes_features_df, EYES_FEATURES_DIR / f"{video_name}.csv")

        eyes_features_list.append(video_eyes_features_df)

    if lip_features_list:
        lip_features_df = pd.concat(lip_features_list, ignore_index=True)
        save_dataframe_to_csv(lip_features_df, ALL_LIP_FEATURES_CSV)
        logger.info(f"Saved combined lip features to {ALL_LIP_FEATURES_CSV}")
    else:
        logger.warning("No lip features were generated — nothing to save.")

    if eyes_features_list:
        eyes_features_df = pd.concat(eyes_features_list, ignore_index=True)
        save_dataframe_to_csv(eyes_features_df, ALL_EYES_FEATURES_CSV)
        logger.info(f"Saved combined eye features to {EYES_FEATURES_DIR / 'all_eyes_features.csv'}")
    else:
        logger.warning("No eye features were generated - nothing to save.")

    logger.info("Data preprocessing pipeline completed successfully")


if __name__ == "__main__":
    main()
