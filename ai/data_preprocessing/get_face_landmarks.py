from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

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

                    for i, landmark in enumerate(landmarks):
                        x = landmark.x
                        y = landmark.y

                        row.append(x)
                        row.append(y)

                    append_row_to_csv(landmarks_file_path, row)

                except Exception as e:
                    print(e)
