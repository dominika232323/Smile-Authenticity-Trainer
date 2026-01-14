from loguru import logger

from config import ALL_CHEEKS_FEATURES_CSV, ALL_EYES_FEATURES_CSV, ALL_FEATURES_RUNS_DIR, ALL_LIP_FEATURES_CSV
from logging_config import setup_logging
from modeling.pipeline import hyperparameter_grid_search


@logger.catch
def main():
    setup_logging()
    logger.info("Starting training on all features pipeline")

    lips_dataset_path = ALL_LIP_FEATURES_CSV
    eyes_dataset_path = ALL_EYES_FEATURES_CSV
    cheeks_dataset_path = ALL_CHEEKS_FEATURES_CSV
    dataset_path = [lips_dataset_path, eyes_dataset_path, cheeks_dataset_path]

    runs_dir = ALL_FEATURES_RUNS_DIR / "hidden_dims_experiment"
    non_feature_cols = ["filename"]

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

    for i in range(4):
        hyperparameter_grid_search(dataset_path, runs_dir, param_grid, non_feature_cols)


if __name__ == "__main__":
    main()
