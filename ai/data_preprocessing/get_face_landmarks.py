from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from loguru import logger

from ai.data_preprocessing.file_utils import append_row_to_csv


def get_face_landmarks(
    frame: cv2.Mat | np.ndarray[Any, np.dtype[Any]],
    frame_number: int,
    landmarks_file_path: Path,
) -> bool:
    logger.debug(f"Extracting face landmarks for frame {frame_number}")

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(frame)

        if results.multi_face_landmarks:
            logger.debug(f"Found face landmarks in frame {frame_number}")

            for face_landmarks in results.multi_face_landmarks:
                try:
                    landmarks = face_landmarks.landmark

                    row = [frame_number]

                    for i, landmark in enumerate(landmarks):
                        x = landmark.x * frame.shape[1]
                        y = landmark.y * frame.shape[0]

                        row.append(x)
                        row.append(y)

                    append_row_to_csv(landmarks_file_path, row)
                    logger.debug(f"Successfully saved {len(landmarks)} landmarks for frame {frame_number}")
                    return True

                except Exception as e:
                    logger.error(f"Error processing face landmarks for frame {frame_number}: {e}")
                    raise e
        else:
            logger.warning(f"No face landmarks detected in frame {frame_number}")
            return False

        return False


def create_facelandmarks_header() -> list[str]:
    logger.debug("Creating header for face landmarks csv file")

    header = ["frame_number"]
    number_of_landmarks = 478

    for i in range(number_of_landmarks):
        header.append(f"{i}_x")
        header.append(f"{i}_y")

    logger.debug(f"Created header with {number_of_landmarks} landmark pairs")

    return header


def get_face_landmark_coords(landmarks_file_path: Path, frame_number: int, landmark_index: int) -> tuple[int, int]:
    logger.debug(f"Reading face landmarks from file: {landmarks_file_path}")

    df = pd.read_csv(landmarks_file_path)

    x_coord = df.loc[df["frame_number"] == frame_number, f"{landmark_index}_x"].values[0]
    y_coord = df.loc[df["frame_number"] == frame_number, f"{landmark_index}_y"].values[0]

    logger.debug(f"Read face landmarks for frame {frame_number} at index {landmark_index}: ({x_coord}, {y_coord})")

    return x_coord, y_coord
