from loguru import logger

from ai.config import ALL_LIP_FEATURES_CSV, LIPS_RUNS_DIR, RUNS_DIR
from ai.data_preprocessing.file_utils import create_directories
from ai.logging_config import setup_logging
from ai.modeling.pipeline import get_timestamp, pipeline


@logger.catch
def main():
    setup_logging()
    logger.info("Starting training on lips features pipeline")

    output_dir = LIPS_RUNS_DIR / get_timestamp()
    create_directories([RUNS_DIR, LIPS_RUNS_DIR, output_dir])

    pipeline(ALL_LIP_FEATURES_CSV, output_dir)


if __name__ == "__main__":
    main()
