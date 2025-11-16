from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from ai.config import (
    ORIGINAL_FACELANDMARKS_DIR,
    ORIGINAL_FRAMES_DIR,
    PREPROCESSED_DATA_DIR,
    PREPROCESSED_FACELANDMARKS_DIR,
    PREPROCESSED_FRAMES_DIR,
    PREPROCESSED_SMILE_PHASES_DIR,
    UvA_NEMO_SMILE_DETAILS,
    UvA_NEMO_SMILE_VIDEOS_DIR,
)
from ai.data_preprocessing.file_utils import create_csv_with_header, create_directories, save_frame
from ai.data_preprocessing.get_details import get_details
from ai.data_preprocessing.get_face_landmarks import create_facelandmarks_header, get_face_landmarks
from ai.data_preprocessing.label_smile_phases import label_smile_phases
from ai.data_preprocessing.normalize_face import normalize_face
from ai.logging_config import setup_logging


@logger.catch
def preprocess_frame(frame: cv2.Mat | np.ndarray[Any, np.dtype[Any]], frame_number: int, video_name: str) -> None:
    logger.debug(f"Processing frame {frame_number} for video {video_name}")

    frame_path = ORIGINAL_FRAMES_DIR / f"{video_name}" / f"{frame_number}.jpg"
    save_frame(frame, frame_path)
    logger.debug(f"Saved frame to {frame_path}")

    face_landmarks_file_path = ORIGINAL_FACELANDMARKS_DIR / f"{video_name}.csv"
    get_face_landmarks(frame, frame_number, face_landmarks_file_path)
    logger.debug(f"Extracted face landmarks for frame {frame_number}")

    normalized_frame_path = PREPROCESSED_FRAMES_DIR / f"{video_name}" / f"{frame_number}.jpg"
    normalized_frame = normalize_face(frame, face_landmarks_file_path, frame_number, (0.35, 0.35), (256, 256))
    save_frame(normalized_frame, normalized_frame_path)
    logger.debug(
        f"Saved normalized frame to {normalized_frame_path} (face landmarks saved to {face_landmarks_file_path})"
    )

    normalized_face_landmarks_file_path = PREPROCESSED_FACELANDMARKS_DIR / f"{video_name}.csv"
    get_face_landmarks(normalized_frame, frame_number, normalized_face_landmarks_file_path)
    logger.debug(f"Extracted normalized face landmarks for frame {frame_number}")


@logger.catch
def preprocess_video(video_path: Path) -> None:
    video_name = video_path.stem
    logger.info(f"Starting preprocessing video: {video_name}")

    video_original_directory = ORIGINAL_FRAMES_DIR / video_name
    video_preprocessed_directory = PREPROCESSED_FRAMES_DIR / video_name
    create_directories([video_original_directory, video_preprocessed_directory])

    face_landmarks_file_path = ORIGINAL_FACELANDMARKS_DIR / f"{video_name}.csv"
    create_csv_with_header(face_landmarks_file_path, create_facelandmarks_header())
    logger.info(f"Added header to face landmarks CSV: {face_landmarks_file_path}")

    normalized_face_landmarks_file_path = PREPROCESSED_FACELANDMARKS_DIR / f"{video_name}.csv"
    create_csv_with_header(normalized_face_landmarks_file_path, create_facelandmarks_header())
    logger.info(f"Added header to normalized face landmarks CSV: {normalized_face_landmarks_file_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")

        return

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

        preprocess_frame(frame, frame_number, video_name)

        if frame_number % 100 == 0:
            logger.info(f"Processed {frame_number}/{total_frames} frames for video {video_name}")

    cap.release()
    logger.info(f"Finished processing all frames for video {video_name}")

    smile_phase_file_path = PREPROCESSED_SMILE_PHASES_DIR / f"{video_name}.csv"
    label_smile_phases(normalized_face_landmarks_file_path, smile_phase_file_path)


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

    for i, video_path in enumerate(videos_to_process, 1):
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            continue

        logger.info(f"Processing video {i}/{len(videos_to_process)}: {video_path.name}")
        preprocess_video(video_path)

    logger.info("Data preprocessing pipeline completed successfully")


if __name__ == "__main__":
    main()
