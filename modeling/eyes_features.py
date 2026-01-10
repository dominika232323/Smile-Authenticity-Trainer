from loguru import logger

from config import ALL_EYES_FEATURES_CSV, EYES_RUNS_DIR
from logging_config import setup_logging
from modeling.pipeline import hyperparameter_grid_search


@logger.catch
def main():
    setup_logging()
    logger.info("Starting training on eyes features pipeline")

    dataset_path = ALL_EYES_FEATURES_CSV
    non_feature_cols = ["filename"]

    param_grid = {
        "batch_size": [32, 64],
        "dropout": [0.1, 0.3, 0.5],
        "epochs": [50],
        "patience": [7],
        "lr": [3e-4, 1e-3],
        "test_size": [0.2],
        "how_many_features": [30, 50, 75],
        "threshold": [0.4, 0.5, 0.6],
        "hidden_dims": [
            [],
            [64],
            [128],
            [128, 64],
            [256, 128],
        ],
    }
    hyperparameter_grid_search(dataset_path, EYES_RUNS_DIR, param_grid, non_feature_cols)


if __name__ == "__main__":
    main()
