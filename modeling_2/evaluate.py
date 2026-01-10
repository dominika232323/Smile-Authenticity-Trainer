from pathlib import Path
from typing import Any

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modeling_2.smile_net import SmileNet


def load_best_model(model_path: Path, X_train: pd.DataFrame, device: str) -> nn.Module:
    logger.info(f"Loading best model from {model_path}")

    model = SmileNet(input_dim=X_train.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def evaluate(
    model: nn.Module,
    test_loader: DataLoader[Any],
    threshold: float,
    device: str,
    output_dir: Path,
    writer: SummaryWriter | None = None,
) -> dict[str, Any]:
    logger.info("Evaluating model...")
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            all_logits.append(logits.cpu())
            all_labels.append(y_batch)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    probs = torch.sigmoid(all_logits)
    preds = (probs > threshold).int()

    accuracy = accuracy_score(all_labels.numpy(), preds.numpy())
    balanced_accuracy = balanced_accuracy_score(all_labels.numpy(), preds.numpy())
    precision = precision_score(all_labels.numpy(), preds.numpy())
    recall = recall_score(all_labels.numpy(), preds.numpy(), zero_division=0)
    f1 = f1_score(all_labels.numpy(), preds.numpy())

    report = classification_report(all_labels, preds, target_names=["No smile (0)", "Smile (1)"], digits=4)
    save_classification_report(report, output_dir)

    cm = confusion_matrix(all_labels, preds)
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

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_classification_report(report: str | dict, output_dir: Path) -> None:
    print("\n=============== Classification Report ===============")
    print(report)

    report_path = output_dir / "classification_report.txt"

    with open(report_path, "w") as f:
        f.write(report)

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
