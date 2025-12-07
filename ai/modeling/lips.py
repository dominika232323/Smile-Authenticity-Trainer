import json

from loguru import logger
from sklearn.model_selection import ParameterGrid

from ai.config import ALL_LIP_FEATURES_CSV, LIPS_RUNS_DIR, RUNS_DIR
from ai.data_preprocessing.file_utils import create_directories
from ai.logging_config import setup_logging
from ai.modeling.pipeline import get_timestamp, pipeline_mlp


@logger.catch
def main() -> None:
    setup_logging()
    logger.info("Starting training on lips features pipeline")

    param_grid = {
        "batch_size": [8, 16, 32, 64],
        "test_size": [0.2, 0.3],
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
        "epochs": [500],
        "patience": [5, 7, 10],
        "lr": [1e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6],
    }
    grid = ParameterGrid(param_grid)

    create_directories([RUNS_DIR, LIPS_RUNS_DIR])

    for params in grid:
        logger.info(f"Running with params: {params}")

        output_dir = LIPS_RUNS_DIR / get_timestamp()
        create_directories([output_dir])

        pipeline_mlp(
            ALL_LIP_FEATURES_CSV,
            output_dir,
            params["batch_size"],
            params["dropout"],
            params["epochs"],
            params["patience"],
            params["lr"],
            params["test_size"],
        )

        with open(output_dir / "config.json", "w") as f:
            json.dump(params, f, indent=4)

        logger.info(f"Completed run. Saved results to {output_dir}. Config: {params}")


if __name__ == "__main__":
    main()
