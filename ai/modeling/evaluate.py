from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve


def evaluate_model(
    model: nn.Module, X_val: torch.Tensor, y_val: torch.Tensor, device: torch.device, output_dir: Path
) -> tuple[torch.Tensor, torch.Tensor]:
    logger.info("Evaluating model...")

    model.eval()
    X_val, y_val = X_val.to(device), y_val.to(device)

    with torch.no_grad():
        probabilities = model(X_val).cpu().numpy()
        predictions = (probabilities > 0.5).astype(int)

    y_true = y_val.cpu().numpy()

    save_classification_report(y_true, predictions, output_dir)
    save_confusion_matrix(y_true, predictions, output_dir)
    save_roc_curve_plot(y_true, probabilities, output_dir)

    logger.info("Model evaluation complete")
    return probabilities, predictions


def save_classification_report(y_true: np.ndarray, predictions: np.ndarray, output_dir: Path) -> None:
    report = classification_report(y_true, predictions)
    print("\n=============== Classification Report ===============")
    print(report)

    report_path = output_dir / "classification_report.txt"

    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Saved classification report to {report_path}")


def save_confusion_matrix(y_true: np.ndarray, predictions: np.ndarray, output_dir: Path) -> None:
    cm = confusion_matrix(y_true, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    logger.info(f"Saved confusion matrix plot to {cm_path}")


def save_roc_curve_plot(y_true: np.ndarray, probabilities: np.ndarray, output_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    logger.info(f"Saved ROC curve plot to {roc_path}")
