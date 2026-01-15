from loguru import logger

from config import CHEEKS_LANDMARKS_IN_APEX_CSV, CHEEKS_LANDMARKS_RUNS_DIR
from logging_config import setup_logging
from modeling.pipeline import hyperparameter_grid_search


@logger.catch
def main():
    setup_logging()
    logger.info("Starting training on cheeks landmarks pipeline")

    dataset_path = CHEEKS_LANDMARKS_IN_APEX_CSV
    runs_dir = CHEEKS_LANDMARKS_RUNS_DIR / "dropout_experiment"
    non_feature_cols = ["filename", "smile_phase", "frame_number"]

    param_grid = {
        "batch_size": [32],
        "dropout": [0.0, 0.2, 0.3, 0.4, 0.5],
        "epochs": [70],
        "patience": [7],
        "lr": [1e-3],
        "test_size": [0.2],
        "how_many_features": [70],
        "threshold": [0.5],
        "hidden_dims": [
            [128, 64],
        ],
    }

    for i in range(1):
        hyperparameter_grid_search(dataset_path, runs_dir, param_grid, non_feature_cols)


if __name__ == "__main__":
    main()
