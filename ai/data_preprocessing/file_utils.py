import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def append_row_to_csv(landmarks_file_path: Path, row: list[int]):
    with open(landmarks_file_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_frame(frame: cv2.Mat | np.ndarray[Any, np.dtype[Any]], frame_path: Path):
    cv2.imwrite(str(frame_path), frame)
