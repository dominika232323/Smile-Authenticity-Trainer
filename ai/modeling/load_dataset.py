import json
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def read_dataset(path: Path) -> pd.DataFrame:
    logger.info(f"Reading dataset from file: {path}")
    df = pd.read_csv(path)

    if "filename" in df.columns:
        logger.info("Removing filename column from dataset")
        df = df.drop(columns=["filename"])

    logger.info(f"Dataset shape: {df.shape}")
    return df


def create_data_tensors(df: pd.DataFrame, output_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    logger.info(f"Loading dataset with shape: {df.shape}")

    X: torch.Tensor
    y: torch.Tensor

    features = df.drop(columns=["label"]).values
    labels = df["label"].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    save_scaler(scaler, output_dir)

    logger.info(f"Dataset tensors shape: {X.shape}, {y.shape}")
    return X, y


def save_scaler(scaler: StandardScaler, output_dir: Path) -> None:
    logger.info(f"Saving scaler to {output_dir}")

    torch.save(scaler, output_dir / "scaler.pt")
    logger.info("Scaler saved as scaler.pt")

    scaler_data = {"mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist()}

    with open(output_dir / "scaler.json", "w") as f:
        json.dump(scaler_data, f, indent=4)

    logger.info("Scaler saved as scaler.json")


def create_dataloaders(
    X: torch.Tensor, y: torch.Tensor, batch_size: int = 32
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    logger.info(f"Creating dataloaders with batch size: {batch_size}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    logger.info(f"Number of training batches: {len(train_loader)}")
    logger.info(f"Number of validation batches: {len(val_loader)}")
    logger.info(f"Validation dataset shape: {X_val.shape}, {y_val.shape}")
    return train_loader, val_loader, X_val, y_val
