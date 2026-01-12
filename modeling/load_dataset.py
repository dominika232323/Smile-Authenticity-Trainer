from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from modeling.smile_dataset import SmileDataset


def load_dataset(path: Path, non_feature_columns: list[str]) -> pd.DataFrame:
    logger.info(f"Reading dataset from file: {path}")
    df = pd.read_csv(path)

    logger.info(f"Dropping non-feature columns: {non_feature_columns}")
    df = df.drop(non_feature_columns, axis=1)

    logger.info(f"Dataset shape: {df.shape}")
    return df


def load_all_features(lips_dataset_path: Path, eyes_dataset_path: Path, cheeks_dataset_path: Path) -> pd.DataFrame:
    logger.info("Loading all features from datasets")

    df_lips = pd.read_csv(lips_dataset_path)
    df_cheeks = pd.read_csv(cheeks_dataset_path)
    df_eyes = pd.read_csv(eyes_dataset_path)

    df_lips = add_prefix(df_lips, "lips")
    df_cheeks = add_prefix(df_cheeks, "cheeks")
    df_eyes = add_prefix(df_eyes, "eyes")

    merged_df = df_lips.merge(df_cheeks, on="filename", how="inner", suffixes=("", "_drop")).merge(
        df_eyes, on="filename", how="inner", suffixes=("", "_drop")
    )

    label_cols = [col for col in merged_df.columns if col.startswith("label")]
    merged_df["label"] = merged_df[label_cols[0]]

    logger.info("Dropping non-feature columns")
    merged_df = merged_df.drop(columns=["filename"])

    logger.info(f"Dataset shape: {merged_df.shape}")
    return merged_df


def add_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename_dict = {col: f"{prefix}_{col}" for col in df.columns if col not in ["filename", "label"]}
    return df.rename(columns=rename_dict)


def feature_selection(
    X: pd.DataFrame, y: np.ndarray, how_many_features: int, selector_output_dir: Path
) -> pd.DataFrame:
    logger.info(f"Selecting {how_many_features} best features")
    selector = SelectKBest(score_func=f_classif, k=how_many_features)
    X_selected = selector.fit_transform(X, y)

    selector_path = selector_output_dir / "feature_selector.joblib"
    logger.info(f"Saving feature selector to {selector_path}")
    joblib.dump(selector, selector_path)

    logger.info(f"Selected features shape: {X_selected.shape}")
    return X_selected


def scale_data(X: pd.DataFrame, scaler_output_dir: Path) -> pd.DataFrame:
    logger.info("Scaling data")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaler_path = scaler_output_dir / "scaler.joblib"
    logger.info(f"Saving scaler to {scaler_path}")
    joblib.dump(scaler, scaler_path)

    logger.info(f"Scaled data shape: {X_scaled.shape}")
    return X_scaled


def split_data(
    X: pd.DataFrame, y: np.ndarray, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    logger.info(f"Splitting data into train and test sets with test size: {test_size}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    logger.info(f"Train set shape: {X_train.shape}, {y_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}, {y_test.shape}")

    return X_train, X_test, y_train, y_test


def get_dataloaders(
    X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: np.ndarray, y_val: np.ndarray, batch_size: int
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    logger.info(f"Creating dataloaders with batch size: {batch_size}")
    train_ds = SmileDataset(X_train, y_train)
    val_ds = SmileDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    logger.info(f"Number of training batches: {len(train_loader)}")
    logger.info(f"Number of validation batches: {len(val_loader)}")

    return train_loader, val_loader
