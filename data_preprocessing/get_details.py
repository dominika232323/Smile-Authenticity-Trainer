from pathlib import Path

import pandas as pd
from loguru import logger


def get_details(path_to_file: Path) -> pd.DataFrame:
    logger.info(f"Reading details from file: {path_to_file}")

    if not path_to_file.exists():
        logger.error(f"Details file not found: {path_to_file}")
        raise FileNotFoundError(f"File not found: {path_to_file}")

    try:
        df = pd.read_csv(
            path_to_file,
            sep=r"\s+",
            names=["filename", "subject_code", "gender", "age", "label"],
            skiprows=4,
        )

        logger.info(f"Successfully read {len(df)} records from details file")
        logger.debug(f"Columns: {list(df.columns)}")

        return df
    except Exception as e:
        logger.error(f"Failed to read details file {path_to_file}: {e}")
        raise
