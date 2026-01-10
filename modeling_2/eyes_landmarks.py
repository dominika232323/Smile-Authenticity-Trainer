from loguru import logger

from config import EYES_LANDMARKS_IN_APEX_CSV, EYES_LANDMARKS_RUNS_DIR
from logging_config import setup_logging
from modeling_2.pipeline import pipeline_landmarks


@logger.catch
def main():
    setup_logging()
    logger.info("Starting training on eyes landmarks pipeline")

    dataset_path = EYES_LANDMARKS_IN_APEX_CSV

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
    pipeline_landmarks(dataset_path, EYES_LANDMARKS_RUNS_DIR, param_grid)


if __name__ == "__main__":
    main()
