from loguru import logger

from config import LIPS_LANDMARKS_IN_APEX_CSV, LIPS_LANDMARKS_RUNS_DIR
from logging_config import setup_logging
from modeling.pipeline import hyperparameter_grid_search


@logger.catch
def main():
    setup_logging()
    logger.info("Starting training on lips landmarks pipeline")

    dataset_path = LIPS_LANDMARKS_IN_APEX_CSV
    runs_dir = LIPS_LANDMARKS_RUNS_DIR / "hidden_dims_experiment"
    non_feature_cols = ["filename", "smile_phase", "frame_number"]

    param_grid = {
        "batch_size": [32],
        "dropout": [0.3],
        "epochs": [150],
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

    for i in range(1):
        hyperparameter_grid_search(dataset_path, runs_dir, param_grid, non_feature_cols)


if __name__ == "__main__":
    main()
