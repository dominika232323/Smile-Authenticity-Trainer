from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from loguru import logger
from numpy import dtype, ndarray
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.tensorboard import SummaryWriter


def evaluate_model(
    model: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    output_dir: Path,
    writer: SummaryWriter | None = None,
) -> tuple[ndarray, ndarray[tuple[int, ...], dtype[Any]]]:
    logger.info("Evaluating model...")

    model.eval()
    X_val, y_val = X_val.to(device), y_val.to(device)

    with torch.no_grad():
        outputs = model(X_val).cpu()
        probabilities = torch.sigmoid(outputs).numpy()
        predictions = (probabilities > 0.5).astype(int)

    y_true = y_val.cpu().numpy()

    if writer is not None:
        cm = confusion_matrix(y_true, predictions)
        tn, fp, fn, tp = cm.ravel()

        precision = precision_score(y_true, predictions)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions)
        accuracy = accuracy_score(y_true, predictions)

        fpr, tpr, _ = roc_curve(y_true, probabilities)
        roc_auc = auc(fpr, tpr)

        writer.add_scalar("Evaluation/precision", precision, 0)
        writer.add_scalar("Evaluation/recall", recall, 0)
        writer.add_scalar("Evaluation/f1", f1, 0)
        writer.add_scalar("Evaluation/accuracy", accuracy, 0)
        writer.add_scalar("Evaluation/auc", roc_auc, 0)

        writer.add_scalar("Evaluation/true_positives", tp, 0)
        writer.add_scalar("Evaluation/true_negatives", tn, 0)
        writer.add_scalar("Evaluation/false_positives", fp, 0)
        writer.add_scalar("Evaluation/false_negatives", fn, 0)

        writer.add_histogram("Evaluation_prediction_probabilities/prediction_probabilities", probabilities, 0)

    save_classification_report(y_true, predictions, output_dir)
    save_confusion_matrix(y_true, predictions, output_dir)
    save_roc_curve_plot(y_true, probabilities, output_dir)

    logger.info("Model evaluation complete")
    return probabilities, predictions


def evaluate_xgboost(model, X_val, y_val, output_dir: Path):
    logger.info("Evaluating XGBoost model...")

    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs > 0.5).astype(int)

    accuracy = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds, zero_division=0)
    recall = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds)
    fpr, tpr, _ = roc_curve(y_val, probs)
    roc_auc = auc(fpr, tpr)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC: {roc_auc:.4f}")

    save_classification_report(y_val, preds, output_dir)
    save_confusion_matrix(y_val, preds, output_dir)
    save_roc_curve_plot(y_val, probs, output_dir)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc,
    }


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
