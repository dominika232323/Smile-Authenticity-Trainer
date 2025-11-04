from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from ai.config import (
    ORIGINAL_FACELANDMARKS_DIR,
    ORIGINAL_FRAMES_DIR,
    PREPROCESSED_DATA_DIR,
    PREPROCESSED_FRAMES_DIR,
    UvA_NEMO_SMILE_DETAILS,
    UvA_NEMO_SMILE_VIDEOS_DIR,
)
from ai.data_preprocessing.file_utils import add_header_to_csv, save_frame
from ai.data_preprocessing.get_details import get_details
from ai.data_preprocessing.get_face_landmarks import create_facelandmarks_header, get_face_landmarks


def preprocess_frame(frame: cv2.Mat | np.ndarray[Any, np.dtype[Any]], frame_number: int, video_name: str):
    frame_path = ORIGINAL_FRAMES_DIR / f"{video_name}" / f"{frame_number}.jpg"
    save_frame(frame, frame_path)

    face_landmarks_file_path = ORIGINAL_FACELANDMARKS_DIR / f"{video_name}.csv"
    get_face_landmarks(frame, frame_number, face_landmarks_file_path)


def preprocess_video(video_path: Path):
    video_name = video_path.stem

    video_original_directory = ORIGINAL_FRAMES_DIR / video_name
    video_original_directory.mkdir(parents=True, exist_ok=True)

    video_preprocessed_directory = PREPROCESSED_FRAMES_DIR / video_name
    video_preprocessed_directory.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_number in tqdm(
        range(0, total_frames), desc=f"Preprocessing frames for video {video_name}", colour="yellow"
    ):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        _, frame = cap.read()

        preprocess_frame(frame, frame_number, video_name)

    face_landmarks_file_path = ORIGINAL_FACELANDMARKS_DIR / f"{video_name}.csv"
    add_header_to_csv(face_landmarks_file_path, create_facelandmarks_header(face_landmarks_file_path))


def main():
    PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ORIGINAL_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    ORIGINAL_FACELANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

    get_details(UvA_NEMO_SMILE_DETAILS).to_csv(PREPROCESSED_DATA_DIR / "details.csv", index=False)

    videos_to_process = [
        UvA_NEMO_SMILE_VIDEOS_DIR / "001_deliberate_smile_2.mp4",
        UvA_NEMO_SMILE_VIDEOS_DIR / "001_deliberate_smile_3.mp4",
    ]

    for video_path in tqdm(videos_to_process, desc="Preprocessing videos", colour="green"):
        preprocess_video(video_path)


if __name__ == "__main__":
    main()
