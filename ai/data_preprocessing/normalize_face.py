from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ai.data_preprocessing.face_landmarks import FaceLandmarks
from ai.data_preprocessing.get_face_landmarks import get_face_landmark_coords


def normalize_face(frame: cv2.Mat | np.ndarray[Any, np.dtype[Any]], frame_number: int):
    pass


def calculate_left_eye_center(landmarks_file_path: Path, frame_number: int) -> tuple[float, float]:
    left_eye_landmarks = list(set(FaceLandmarks.left_eye_lower_0() + FaceLandmarks.left_eye_upper_0()))
    return calculate_center_of_landmarks(landmarks_file_path, frame_number, left_eye_landmarks)


def calculate_right_eye_center(landmarks_file_path: Path, frame_number: int) -> tuple[float, float]:
    right_eye_landmarks = list(set(FaceLandmarks.right_eye_lower_0() + FaceLandmarks.right_eye_upper_0()))
    return calculate_center_of_landmarks(landmarks_file_path, frame_number, right_eye_landmarks)


def calculate_center_of_landmarks(
    landmarks_file_path: Path, frame_number: int, landmarks: list[int]
) -> tuple[float, float]:
    number_of_landmarks = len(landmarks)

    x_sum = 0
    y_sum = 0

    for landmark in landmarks:
        x, y = get_face_landmark_coords(landmarks_file_path, frame_number, landmark)

        x_sum += x
        y_sum += y

    return x_sum / number_of_landmarks, y_sum / number_of_landmarks


def calculate_distance_between_points(right: tuple[float, float], left: tuple[float, float]) -> float:
    x_right, y_right = right
    x_left, y_left = left

    return np.sqrt((x_right - x_left) ** 2 + (y_right - y_left) ** 2)


def calculate_angle_between_points(right: tuple[float, float], left: tuple[float, float]) -> float:
    x_right, y_right = right
    x_left, y_left = left

    return np.degrees(np.arctan2(y_right - y_left, x_right - x_left)) - 180


def calculate_center_between_points(right: tuple[float, float], left: tuple[float, float]) -> tuple[float, float]:
    x_right, y_right = right
    x_left, y_left = left

    return (x_right + x_left) / 2, (y_right + y_left) / 2


def calculate_image_scaling_factor(
    right_eye_center: tuple[float, float],
    left_eye_center: tuple[float, float],
    eye_relative_x: float,
    desired_image_width: float,
) -> float:
    distance_between_eyes = calculate_distance_between_points(right_eye_center, left_eye_center)
    desired_distance = calculate_desired_eye_distance(eye_relative_x, desired_image_width)

    return desired_distance / distance_between_eyes


def calculate_desired_eye_distance(eye_relative_x: float, desired_image_width: float) -> float:
    if not (0 < eye_relative_x < 0.5):
        raise ValueError("eye_relative_x should be between 0 and 0.5 for symmetric placement.")
    if desired_image_width <= 0:
        raise ValueError("desired_image_width must be positive.")

    d_desired = (1 - 2 * eye_relative_x) * desired_image_width

    return d_desired
