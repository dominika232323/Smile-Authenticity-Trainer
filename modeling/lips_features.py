from loguru import logger

from config import ALL_LIP_FEATURES_CSV, LIPS_RUNS_DIR
from logging_config import setup_logging
from modeling.pipeline import hyperparameter_grid_search


@logger.catch
def main():
    setup_logging()
    logger.info("Starting training on lips features pipeline")

    dataset_path = ALL_LIP_FEATURES_CSV
    non_feature_cols = ["filename"]

    param_grid = {
        "batch_size": [8, 16, 32, 64],
        "dropout": [0.2, 0.3, 0.4, 0.5],
        "epochs": [150],
        "patience": [3, 5, 7, 10],
        "lr": [1e-2, 1e-3, 1e-4, 1e-5],
        "test_size": [0.2, 0.25, 0.3, 0.4],
        "how_many_features": [25, 40, 50, 70],
        "threshold": [0.4, 0.5, 0.6, 0.7],
    }
    hyperparameter_grid_search(dataset_path, LIPS_RUNS_DIR, param_grid, non_feature_cols)


if __name__ == "__main__":
    main()
