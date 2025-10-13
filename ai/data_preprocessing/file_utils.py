import csv
from pathlib import Path
import numpy as np
from typing import Any

import cv2


def append_row_to_csv(landmarks_file_path: Path, row: list[int]):
    with open(landmarks_file_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_frame(frame: cv2.Mat | np.ndarray[Any, np.dtype[Any]], output_directory: Path, frame_number: int, video_name: str):
    frames_directory = output_directory / f"{video_name}"
    frames_directory.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(frames_directory / f"{frame_number}.jpg"), frame)
