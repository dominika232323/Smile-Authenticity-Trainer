from pathlib import Path

from loguru import logger

from config import LIPS_LANDMARKS_IN_APEX_CSV, LIPS_LANDMARKS_RUNS_DIR
from logging_config import setup_logging
from modeling.pipeline import hyperparameter_grid_search


@logger.catch
def main():
    setup_logging()
    logger.info("Starting training on lips landmarks pipeline")

    dataset_path = LIPS_LANDMARKS_IN_APEX_CSV
    non_feature_cols = ["filename", "smile_phase", "frame_number"]

    runs_dir, param_grid = hidden_dims_experiment()

    for i in range(3):
        hyperparameter_grid_search(dataset_path, runs_dir, param_grid, non_feature_cols)

    runs_dir, param_grid = how_many_features_experiment()

    for i in range(5):
        hyperparameter_grid_search(dataset_path, runs_dir, param_grid, non_feature_cols)

    runs_dir, param_grid = dropout_experiment()

    for i in range(5):
        hyperparameter_grid_search(dataset_path, runs_dir, param_grid, non_feature_cols)

    runs_dir, param_grid = threshold_experiment()

    for i in range(5):
        hyperparameter_grid_search(dataset_path, runs_dir, param_grid, non_feature_cols)


def hidden_dims_experiment() -> tuple[Path, dict[str, object]]:
    runs_dir = LIPS_LANDMARKS_RUNS_DIR / "hidden_dims_experiment"

    param_grid = {
        "batch_size": [32],
        "dropout": [0.3],
        "epochs": [70],
        "patience": [7],
        "lr": [1e-3],
        "test_size": [0.2],
        "how_many_features": [70],
        "threshold": [0.5],
        "hidden_dims": [
            [64, 32],
            [128, 64],
            [256, 128],
            [128, 64, 32],
            [256, 128, 64],
            [512, 256, 128],
            [256, 128, 64, 32],
            [512, 256, 128, 64],
        ],
    }

    return runs_dir, param_grid


def how_many_features_experiment() -> tuple[Path, dict[str, object]]:
    runs_dir = LIPS_LANDMARKS_RUNS_DIR / "how_many_features_experiment"

    param_grid = {
        "batch_size": [32],
        "dropout": [0.3],
        "epochs": [150],
        "patience": [7],
        "lr": [1e-3],
        "test_size": [0.2],
        "how_many_features": [20, 50, 70, 100, 200],
        "threshold": [0.5],
        "hidden_dims": [
            [128, 64],
        ],
    }

    return runs_dir, param_grid


def dropout_experiment() -> tuple[Path, dict[str, object]]:
    runs_dir = LIPS_LANDMARKS_RUNS_DIR / "dropout_experiment"

    param_grid = {
        "batch_size": [32],
        "dropout": [0.0, 0.2, 0.3, 0.4, 0.5],
        "epochs": [150],
        "patience": [7],
        "lr": [1e-3],
        "test_size": [0.2],
        "how_many_features": [70],
        "threshold": [0.5],
        "hidden_dims": [
            [128, 64],
        ],
    }

    return runs_dir, param_grid


def threshold_experiment() -> tuple[Path, dict[str, object]]:
    runs_dir = LIPS_LANDMARKS_RUNS_DIR / "threshold_experiment"

    param_grid = {
        "batch_size": [32],
        "dropout": [0.3],
        "epochs": [70],
        "patience": [7],
        "lr": [1e-3],
        "test_size": [0.2],
        "how_many_features": [70],
        "threshold": [0.2, 0.35, 0.5, 0.6, 0.7],
        "hidden_dims": [
            [128, 64],
        ],
    }

    return runs_dir, param_grid


if __name__ == "__main__":
    main()
