from loguru import logger

from config import ALL_LIP_FEATURES_CSV, LIPS_RUNS_DIR
from logging_config import setup_logging
from modeling.pipeline import hyperparameter_grid_search


@logger.catch
def main():
    setup_logging()
    logger.info("Starting training on lips features pipeline")

    dataset_path = ALL_LIP_FEATURES_CSV
    runs_dir = LIPS_RUNS_DIR / "how_many_features_experiment"
    non_feature_cols = ["filename"]

    param_grid = {
        "batch_size": [32],
        "dropout": [0.3],
        "epochs": [70],
        "patience": [7],
        "lr": [1e-3],
        "test_size": [0.2],
        "how_many_features": [20, 50, 70],
        "threshold": [0.5],
        "hidden_dims": [
            [128, 64],
        ],
    }

    for i in range(10):
        hyperparameter_grid_search(dataset_path, runs_dir, param_grid, non_feature_cols)


if __name__ == "__main__":
    main()
