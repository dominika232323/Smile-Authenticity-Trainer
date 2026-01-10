import datetime
import json
from pathlib import Path

from loguru import logger
from sklearn.model_selection import ParameterGrid

from config import LIPS_LANDMARKS_RUNS_DIR
from data_preprocessing.file_utils import create_directories
from logging_config import setup_logging
from modeling_2.pipeline import pipeline


def get_timestamp() -> str:
    ct = datetime.datetime.now()
    return ct.strftime("%Y-%m-%d_%H-%M-%S")


@logger.catch
def main():
    setup_logging()
    logger.info("Starting training on lips features pipeline")

    dataset_path = Path(
        "/home/dominika/Desktop/Smile-Authenticity-Trainer/data_test/preprocessed_UvA-NEMO_SMILE_DATABASE/lips_landmarks.csv"
    )
    non_feature_cols = ["filename", "smile_phase", "frame_number"]

    param_grid = {
        "batch_size": [32],
        "dropout": [0.3],
        "epochs": [50],
        "patience": [7],
        "lr": [1e-3],
        "test_size": [0.2],
        "how_many_features": [50],
        "threshold": [0.5],
    }
    grid = ParameterGrid(param_grid)

    for params in grid:
        logger.info(f"Running with params: {params}")

        output_dir = LIPS_LANDMARKS_RUNS_DIR / get_timestamp()
        best_model_path = output_dir / "best_model.pth"
        create_directories([output_dir])

        pipeline(
            dataset_path,
            best_model_path,
            non_feature_cols,
            output_dir,
            params["batch_size"],
            params["dropout"],
            params["epochs"],
            params["patience"],
            params["lr"],
            params["test_size"],
            params["how_many_features"],
            params["threshold"],
        )

        with open(output_dir / "config.json", "w") as f:
            json.dump(params, f, indent=4)

        logger.info(f"Completed run. Saved results to {output_dir}. Config: {params}")


if __name__ == "__main__":
    main()
