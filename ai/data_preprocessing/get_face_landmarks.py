from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from ai.data_preprocessing.file_utils import append_row_to_csv


def get_face_landmarks(
    frame: cv2.Mat | np.ndarray[Any, np.dtype[Any]],
    frame_number: int,
    landmarks_file_path: Path,
):
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                try:
                    landmarks = face_landmarks.landmark

                    row = [frame_number]

                    for i, landmark in tqdm(
                        enumerate(landmarks),
                        total=len(landmarks),
                        desc=f"Saving landmarks for frame {frame_number}",
                        colour="blue",
                    ):
                        x = landmark.x
                        y = landmark.y

                        row.append(x)
                        row.append(y)

                    append_row_to_csv(landmarks_file_path, row)

                except Exception as e:
                    raise e


def create_facelandmarks_header(face_landmarks_file_path: Path) -> list[str]:
    header = ["frame_number"]

    df = pd.read_csv(face_landmarks_file_path, header=None)
    number_of_columns = df.shape[1]

    for i in range((number_of_columns - 1) // 2):
        header.append(f"{i}_x")
        header.append(f"{i}_y")

    return header
