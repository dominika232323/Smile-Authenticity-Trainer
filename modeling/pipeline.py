import datetime
import json
from pathlib import Path

import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from data_preprocessing.file_utils import create_directories
from modeling.evaluate import evaluate_model, evaluate_xgboost
from modeling.load_dataset import create_data_tensors, create_dataloaders, read_dataset, scale_features
from modeling.multi_layer_perceptron import MultiLayerPerceptron
from modeling.train import plot_training_curves, train_model, train_xgboost


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_timestamp() -> str:
    ct = datetime.datetime.now()
    return ct.strftime("%Y-%m-%d_%H-%M-%S")


def pipeline_mlp(
    dataset_path: Path,
    output_dir: Path,
    batch_size: int = 32,
    dropout: float = 0.3,
    epochs: int = 50,
    patience: int = 5,
    lr: float = 1e-3,
    test_size: float = 0.2,
) -> None:
    training_curves_directory = output_dir / "training_curves"
    evaluation_metrics_directory = output_dir / "evaluation_metrics"
    tensorboard_logs_directory = output_dir / "tensorboard_logs"

    create_directories([training_curves_directory, evaluation_metrics_directory, tensorboard_logs_directory])

    writer = SummaryWriter(log_dir=tensorboard_logs_directory)

    device = get_device()
    logger.info(f"Using device: {device}")

    dataset_df = read_dataset(dataset_path)
    features, labels = scale_features(dataset_df, output_dir)
    X, y = create_data_tensors(features, labels)
    train_loader, val_loader, X_val, y_val = create_dataloaders(X, y, batch_size, test_size)

    model = MultiLayerPerceptron(input_dim=X.shape[1], dropout_p=dropout).to(device)

    sample_input = torch.randn(1, X.shape[1]).to(device)
    writer.add_graph(model, sample_input)

    hparams = {
        "batch_size": batch_size,
        "test_size": test_size,
        "dropout": dropout,
        "epochs": epochs,
        "patience": patience,
        "lr": lr,
        "input_dim": X.shape[1],
    }
    writer.add_hparams(hparams, {})

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs,
        patience,
        lr,
        save_path=output_dir / "multi_layer_perceptron.pth",
        writer=writer,
    )
    plot_training_curves(history, training_curves_directory)

    probs, preds = evaluate_model(model, X_val, y_val, device, evaluation_metrics_directory, writer)

    print(f"\nExample probability of label 1: {probs[0][0] * 100:.2f}%")
    print(f"Example predicted label: {preds[0]}")

    final_train_loss = history["train_loss"][-1]
    final_val_loss = history["val_loss"][-1]
    final_train_acc = history["train_accuracy"][-1]
    final_val_acc = history["val_accuracy"][-1]

    writer.add_scalar("Final/train_loss", final_train_loss, 0)
    writer.add_scalar("Final/val_loss", final_val_loss, 0)
    writer.add_scalar("Final/train_accuracy", final_train_acc, 0)
    writer.add_scalar("Final/val_accuracy", final_val_acc, 0)

    writer.close()

    logger.info(f"TensorBoard logs saved to {tensorboard_logs_directory}")
    logger.info("To view TensorBoard, run: tensorboard --logdir=" + str(tensorboard_logs_directory))


def pipeline_xgboost(dataset_path: Path, output_dir: Path) -> None:
    dataset_df = read_dataset(dataset_path)
    features, labels = scale_features(dataset_df, output_dir)

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )

    model, params = train_xgboost(X_train, y_train, X_val, y_val)
    metrics = evaluate_xgboost(model, X_val, y_val, output_dir)

    # model.save_model(str(output_dir / "xgboost_model.json"))

    with open(output_dir / "params.json", "w") as f:
        json.dump(params, f, indent=4)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
