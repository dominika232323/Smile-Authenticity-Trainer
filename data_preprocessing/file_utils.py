import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from loguru import logger

from config import CHECKPOINT_FILE_PATH


def append_row_to_csv(file_path: Path, row: list[Any]) -> None:
    logger.debug(f"Appending row with {len(row)} values to {file_path}")

    try:
        with open(file_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        logger.error(f"Failed to append row to CSV {file_path}: {e}")
        raise


def add_header_to_csv(file_path: Path, header: list[str]) -> None:
    logger.debug(f"Adding header to CSV file: {file_path}")

    try:
        df = pd.read_csv(file_path, header=None)
        df.columns = header
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully added header with {len(header)} columns to {file_path}")
    except Exception as e:
        logger.error(f"Failed to add header to CSV {file_path}: {e}")
        raise


def create_csv_with_header(file_path: Path, header: list[str]) -> None:
    logger.debug(f"Creating CSV file with header: {file_path}")

    try:
        create_directories([file_path.parent])

        with open(file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)
    except Exception as e:
        logger.error(f"Failed to create CSV file with header {file_path}: {e}")
        raise


def save_frame(frame: cv2.Mat | np.ndarray[Any, np.dtype[Any]], frame_path: Path) -> None:
    logger.debug(f"Saving frame to {frame_path}")

    try:
        create_directories([frame_path.parent])
        success = cv2.imwrite(str(frame_path), frame)

        if not success:
            raise RuntimeError(f"cv2.imwrite failed for {frame_path}")

        logger.debug(f"Frame saved successfully to {frame_path}")
    except Exception as e:
        logger.error(f"Failed to save frame to {frame_path}: {e}")
        raise


def create_directories(directories: list[Path]) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory {directory} exists")


def save_dataframe_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    logger.debug(f"Saving dataframe to file: {output_path}")

    try:
        df.to_csv(output_path, index=False, header=True)
        logger.info(f"Dataframe saved to file: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save dataframe to file: {output_path}: {e}")
        raise


def concat_csvs(input_dir: Path) -> pd.DataFrame:
    logger.debug(f"Concatenating CSV files in directory: {input_dir}")

    csv_files = list(input_dir.glob("*.csv"))

    if not csv_files:
        logger.error(f"No CSV files found in directory: {input_dir}")
        raise ValueError(f"No CSV files found in directory: {input_dir}")

    df_list = [pd.read_csv(csv_path) for csv_path in csv_files]
    final_df = pd.concat(df_list, ignore_index=True)

    logger.debug(f"Concatenated dataframe with shape {final_df.shape}")
    return final_df


def ensure_checkpoint_file_exists() -> bool:
    file_path = Path(CHECKPOINT_FILE_PATH)

    if not file_path.exists():
        create_csv_with_header(CHECKPOINT_FILE_PATH, ["file_path", "preprocessed"])
        return False

    return True
