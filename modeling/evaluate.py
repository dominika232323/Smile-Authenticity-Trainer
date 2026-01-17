import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modeling.smile_net import SmileNet


def load_best_model(
    model_path: Path, X_train: pd.DataFrame, device: str, dropout: float, hidden_dims: list[int] | None
) -> nn.Module:
    logger.info(f"Loading best model from {model_path}")

    if hidden_dims is None:
        hidden_dims = [128, 64]

    model = SmileNet(input_dim=X_train.shape[1], dropout_p=dropout, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def evaluate(
    model: nn.Module,
    test_loader: DataLoader[Any] | None,
    threshold: float,
    device: str,
    output_dir: Path,
    writer: SummaryWriter | None = None,
) -> dict[str, Any]:
    logger.info("Evaluating model...")

    if test_loader is None:
        logger.warning("No test data loader provided, skipping evaluation")
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)

            all_logits.append(logits.cpu())
            all_labels.append(y_batch)

    all_logits_tensor = torch.cat(all_logits)
    all_labels_tensor = torch.cat(all_labels)

    probs = torch.sigmoid(all_logits_tensor)
    preds = (probs > threshold).int()

    accuracy = accuracy_score(all_labels_tensor.numpy(), preds.numpy())
    balanced_accuracy = balanced_accuracy_score(all_labels_tensor.numpy(), preds.numpy())
    precision = precision_score(all_labels_tensor.numpy(), preds.numpy())
    recall = recall_score(all_labels_tensor.numpy(), preds.numpy(), zero_division=0)
    f1 = f1_score(all_labels_tensor.numpy(), preds.numpy())

    report = classification_report(all_labels_tensor, preds, target_names=["No smile (0)", "Smile (1)"], digits=4)
    save_classification_report(report, output_dir)

    cm = confusion_matrix(all_labels_tensor, preds)
    save_confusion_matrix(cm, output_dir)

    if writer is not None:
        writer.add_scalar("Evaluation/precision", precision, 0)
        writer.add_scalar("Evaluation/recall", recall, 0)
        writer.add_scalar("Evaluation/f1", f1, 0)
        writer.add_scalar("Evaluation/accuracy", accuracy, 0)

        tp = cm[1, 1]
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]

        writer.add_scalar("Evaluation/true_positives", tp, 0)
        writer.add_scalar("Evaluation/true_negatives", tn, 0)
        writer.add_scalar("Evaluation/false_positives", fp, 0)
        writer.add_scalar("Evaluation/false_negatives", fn, 0)

    logger.info("Model evaluation complete")

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    save_metrics(metrics, output_dir)

    return metrics


def predict(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
    threshold: float = 0.5,
    return_proba: bool = False,
) -> np.ndarray:
    model.eval()

    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                X = batch[0]
            else:
                X = batch

            X = X.to(device)

            logits = model(X)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())

    probs = torch.cat(all_probs).numpy().squeeze()

    if return_proba:
        return probs

    return (probs >= threshold).astype(np.int32)


def save_classification_report(report: str | dict, output_dir: Path) -> None:
    print("\n=============== Classification Report ===============")
    print(report)

    report_path = output_dir / "classification_report.txt"

    if isinstance(report, dict):
        report_str = json.dumps(report, indent=2)
    else:
        report_str = report

    with open(report_path, "w") as f:
        f.write(report_str)

    logger.info(f"Saved classification report to {report_path}")


def save_confusion_matrix(cm: Any, output_dir: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    logger.info(f"Saved confusion matrix plot to {cm_path}")


def save_metrics(metrics: dict[str, float], output_dir: Path) -> None:
    metrics_path = output_dir / "metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved metrics to {metrics_path}")
