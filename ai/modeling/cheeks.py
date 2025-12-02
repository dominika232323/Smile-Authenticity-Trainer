import json

from loguru import logger

from ai.config import ALL_CHEEKS_FEATURES_CSV, CHEEKS_RUNS_DIR, RUNS_DIR
from ai.data_preprocessing.file_utils import create_directories
from ai.logging_config import setup_logging
from ai.modeling.pipeline import get_timestamp, pipeline


@logger.catch
def main() -> None:
    setup_logging()
    logger.info("Starting training on cheeks features pipeline")

    output_dir = CHEEKS_RUNS_DIR / get_timestamp()
    create_directories([RUNS_DIR, CHEEKS_RUNS_DIR, output_dir])

    batch_size = 8
    dropout = 0.3
    epochs = 500
    patience = 5
    lr = 1e-3

    pipeline(ALL_CHEEKS_FEATURES_CSV, output_dir, batch_size, dropout, epochs, patience, lr)

    config = {"batch_size": batch_size, "dropout": dropout, "epochs": epochs, "patience": patience, "lr": lr}
    json.dump(config, open(output_dir / "config.json", "w"), indent=4)

    logger.info(f"Training complete. Saved results to {output_dir}. Config: {config}")


if __name__ == "__main__":
    main()
