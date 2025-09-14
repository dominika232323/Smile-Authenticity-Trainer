import pandas as pd
from pathlib import Path


def get_details(path_to_file: Path) -> pd.DataFrame:
    df = pd.read_csv(path_to_file,
                     sep=r"\s+",
                     names=["filename", "subject_code", "gender", "age", "label"],
                     skiprows=4)

    return df
