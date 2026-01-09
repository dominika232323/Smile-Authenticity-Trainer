import copy
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import joblib

from modeling_2.smile_net import SmileNet


def load_dataset(path: Path, non_feature_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(non_feature_columns, axis=1)

    return df


def feature_selection(X: pd.DataFrame, y: np.ndarray, how_many_features: int) -> pd.DataFrame:
    selector = SelectKBest(score_func=f_classif, k=how_many_features)
    X_selected = selector.fit_transform(X, y)

    joblib.dump(selector, "feature_selector.joblib")

    return X_selected


def scale_data(X: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.joblib")

    return X_scaled


def split_data(X: pd.DataFrame, y: np.ndarray, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


class SmileDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: np.ndarray, y_val: np.ndarray, batch_size: int) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_ds = SmileDataset(X_train, y_train)
    val_ds = SmileDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def calculate_pos_weight(y_train: np.ndarray) -> Tensor:
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()

    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)

    return pos_weight


def train(
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        pos_weight: Tensor,
        learning_rate: float,
        epochs: int,
        patience: int,
        threshold: float,
        device: str,
        model_path: Path
) -> dict[str, list[float]]:

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "train_balanced_acc": [],
        "val_acc": [],
        "val_balanced_acc": [],
        "lr": []
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            train_targets.append(y_batch.cpu().numpy())

        train_preds = np.concatenate(train_preds) > threshold
        train_targets = np.concatenate(train_targets)
        train_acc = accuracy_score(train_targets, train_preds)
        train_balanced_acc = balanced_accuracy_score(train_targets, train_preds)
        train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_losses.append(loss.item())
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        val_preds = np.concatenate(val_preds) > threshold
        val_targets = np.concatenate(val_targets)
        val_acc = accuracy_score(val_targets, val_preds)
        val_balanced_acc = balanced_accuracy_score(val_targets, val_preds)
        val_loss = np.mean(val_losses)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["train_balanced_acc"].append(train_balanced_acc)
        history["val_acc"].append(val_acc)
        history["val_balanced_acc"].append(val_balanced_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} Balanced Acc: {train_balanced_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Balanced Acc: {val_balanced_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return history


def draw_history(history: dict[str, list[float]], output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_plot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "accuracy_plot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_balanced_acc"], label="Train Accuracy")
    plt.plot(history["val_balanced_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Balanced Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "balanced_accuracy_plot.png", dpi=300)
    plt.close()


def load_best_model(model_path: Path, X_train: pd.DataFrame, device: str) -> nn.Module:
    model = SmileNet(input_dim=X_train.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def evaluate(model: nn.Module, test_loader: DataLoader[Any], threshold: float, device: str) -> tuple[float, float]:
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            all_logits.append(logits.cpu())
            all_labels.append(y_batch)

    # Concatenate tensors
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Use PyTorch sigmoid
    probs = torch.sigmoid(all_logits)

    # Thresholding
    preds = (probs > threshold).int()

    # Metrics (convert to numpy)
    test_acc = accuracy_score(all_labels.numpy(), preds.numpy())
    test_balanced_acc = balanced_accuracy_score(all_labels.numpy(), preds.numpy())

    report = classification_report(all_labels, preds, target_names=["No smile (0)", "Smile (1)"], digits=4)
    cm = confusion_matrix(all_labels, preds)


def main():
    DATASET_PATH = Path(
        "/home/dominika/Desktop/Smile-Authenticity-Trainer/data_test/preprocessed_UvA-NEMO_SMILE_DATABASE/lips_landmarks.csv")
    NON_FEATURE_COLS = ["filename", "smile_phase", "frame_number"]
    BEST_MODEL_PATH = Path("/home/dominika/Desktop/Smile-Authenticity-Trainer/models/lips_model.pth")
    PLOTS_DIR = Path("/home/dominika/Desktop/Smile-Authenticity-Trainer/plots/lips_plots")
    TRESHOLD = 0.5

    dataset = load_dataset(DATASET_PATH, NON_FEATURE_COLS)

    X = dataset.drop("label", axis=1)
    y = dataset["label"].values.astype(np.float32)

    X_selected = feature_selection(X, y, 50)
    X_scaled = scale_data(X_selected)

    X_train, X_test, y_train, y_test = split_data(X_scaled, y, 0.2)

    train_loader, val_loader = get_dataloaders(X_train, X_test, y_train, y_test, 32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmileNet(input_dim=X_train.shape[1]).to(device)

    pos_weight = calculate_pos_weight(y_train)
    history = train(model, train_loader, val_loader, pos_weight, 1e-3, 100, 7, TRESHOLD, device, BEST_MODEL_PATH)
    draw_history(history, PLOTS_DIR)

    evaluate(load_best_model(BEST_MODEL_PATH, X_train, device), val_loader, TRESHOLD, device)


if __name__ == "__main__":
    main()