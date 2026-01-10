from loguru import logger

from config import ALL_CHEEKS_FEATURES_CSV, CHEEKS_RUNS_DIR
from logging_config import setup_logging
from modeling.pipeline import hyperparameter_grid_search


@logger.catch
def main():
    setup_logging()
    logger.info("Starting training on cheeks features pipeline")

    dataset_path = ALL_CHEEKS_FEATURES_CSV
    non_feature_cols = ["filename"]

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
    hyperparameter_grid_search(dataset_path, CHEEKS_RUNS_DIR, param_grid, non_feature_cols)


if __name__ == "__main__":
    main()
