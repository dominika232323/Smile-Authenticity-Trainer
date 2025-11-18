from pathlib import Path

import pandas as pd


def assign_labels(df_features: pd.DataFrame, details_file_path: Path) -> pd.DataFrame:
    df_labels = pd.read_csv(details_file_path)

    df_merged = df_features.merge(df_labels[["filename", "label"]], on="filename", how="left")

    df_merged["label"] = df_merged["label"].map({"deliberate": 0, "spontaneous": 1})

    return df_merged
