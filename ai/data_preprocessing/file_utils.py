import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def append_row_to_csv(landmarks_file_path: Path, row: list[int]):
    with open(landmarks_file_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def add_header_to_csv(file_path: Path, header: list[str]):
    df = pd.read_csv(file_path, header=None)
    df.columns = header
    df.to_csv(file_path, index=False)


def save_frame(frame: cv2.Mat | np.ndarray[Any, np.dtype[Any]], frame_path: Path):
    cv2.imwrite(str(frame_path), frame)
