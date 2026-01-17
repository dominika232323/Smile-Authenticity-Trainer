from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    CHEEKS_LANDMARKS_IN_APEX_CSV,
    EYES_LANDMARKS_IN_APEX_CSV,
    LIPS_LANDMARKS_IN_APEX_CSV,
    PREPROCESSED_DATA_DIR,
    PREPROCESSED_FACELANDMARKS_DIR,
    PREPROCESSED_SMILE_PHASES_DIR,
)
from data_preprocessing.face_landmarks import FaceLandmarks


def save_landmarks_in_apex() -> None:
    details_df = pd.read_csv(PREPROCESSED_DATA_DIR / "details.csv")

    final_lips_landmarks_df = pd.DataFrame()
    final_eyes_landmarks_df = pd.DataFrame()
    final_cheeks_landmarks_df = pd.DataFrame()

    for filename, label in zip(details_df["filename"], details_df["label"]):
        filename = Path(filename).stem
        label = 0 if label == "deliberate" else 1

        smile_phases = pd.read_csv(PREPROCESSED_SMILE_PHASES_DIR / f"{filename}.csv")
        landmarks = pd.read_csv(PREPROCESSED_FACELANDMARKS_DIR / f"{filename}.csv")

        cheeks_landmarks_df, eyes_landmarks_df, lips_landmarks_df = get_lips_eyes_cheeks_landmarks_for_file(
            smile_phases, landmarks, filename, label
        )

        final_lips_landmarks_df = pd.concat([final_lips_landmarks_df, lips_landmarks_df], ignore_index=True)
        final_eyes_landmarks_df = pd.concat([final_eyes_landmarks_df, eyes_landmarks_df], ignore_index=True)
        final_cheeks_landmarks_df = pd.concat([final_cheeks_landmarks_df, cheeks_landmarks_df], ignore_index=True)

    final_lips_landmarks_df.to_csv(LIPS_LANDMARKS_IN_APEX_CSV, index=False)
    final_eyes_landmarks_df.to_csv(EYES_LANDMARKS_IN_APEX_CSV, index=False)
    final_cheeks_landmarks_df.to_csv(CHEEKS_LANDMARKS_IN_APEX_CSV, index=False)


def get_lips_eyes_cheeks_landmarks_for_file(
    smile_phases: pd.DataFrame, landmarks: pd.DataFrame, filename: str, label: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = smile_phases.merge(landmarks, on="frame_number", how="inner")
    merged["filename"] = filename

    lips_indexes = get_lips_indexes()
    eyes_indexes = get_eyes_indexes()
    cheeks_indexes = get_cheeks_indexes()

    lips_landmarks = get_list_of_coords(lips_indexes)
    eyes_landmarks = get_list_of_coords(eyes_indexes)
    cheeks_landmarks = get_list_of_coords(cheeks_indexes)

    columns_to_keep_lips = ["filename", "frame_number", "smile_phase"] + lips_landmarks
    columns_to_keep_eyes = ["filename", "frame_number", "smile_phase"] + eyes_landmarks
    columns_to_keep_cheeks = ["filename", "frame_number", "smile_phase"] + cheeks_landmarks

    filtered_lips = merged[columns_to_keep_lips]
    filtered_eyes = merged[columns_to_keep_eyes]
    filtered_cheeks = merged[columns_to_keep_cheeks]

    result_lips = filtered_lips[filtered_lips["smile_phase"] == "apex"].copy()
    result_lips["label"] = label

    result_eyes = filtered_eyes[filtered_eyes["smile_phase"] == "apex"].copy()
    result_eyes["label"] = label

    result_cheeks = filtered_cheeks[filtered_cheeks["smile_phase"] == "apex"].copy()
    result_cheeks["label"] = label

    return result_cheeks, result_eyes, result_lips


def get_list_of_coords(lips_indexes: list[int | Any]) -> list[str]:
    lips_landmarks = []

    for i in lips_indexes:
        lips_landmarks.append(f"{i}_x")
        lips_landmarks.append(f"{i}_y")

    return lips_landmarks


def get_cheeks_indexes() -> list[int]:
    cheeks_indexes = list(
        set(
            FaceLandmarks.left_cheek()
            + FaceLandmarks.right_cheek()
            + FaceLandmarks.left_cheek_center()
            + FaceLandmarks.right_cheek_center()
        )
    )
    return cheeks_indexes


def get_eyes_indexes() -> list[int]:
    eyes_indexes = list(
        set(
            FaceLandmarks.right_eye_upper_0()
            + FaceLandmarks.right_eye_lower_0()
            + FaceLandmarks.right_eye_upper_1()
            + FaceLandmarks.right_eye_lower_1()
            + FaceLandmarks.right_eye_upper_2()
            + FaceLandmarks.right_eye_lower_2()
            + FaceLandmarks.right_eye_lower_3()
            + FaceLandmarks.left_eye_upper_0()
            + FaceLandmarks.left_eye_lower_0()
            + FaceLandmarks.left_eye_upper_1()
            + FaceLandmarks.left_eye_lower_1()
            + FaceLandmarks.left_eye_upper_2()
            + FaceLandmarks.left_eye_lower_2()
            + FaceLandmarks.left_eye_lower_3()
        )
    )
    return eyes_indexes


def get_lips_indexes() -> list[int | Any]:
    lips_indexes = list(
        set(
            FaceLandmarks.lips_upper_outer()
            + FaceLandmarks.lips_lower_outer()
            + FaceLandmarks.lips_upper_inner()
            + FaceLandmarks.lips_lower_inner()
        )
    )
    return lips_indexes
