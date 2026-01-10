from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from data_preprocessing.file_utils import create_directories
from modeling_2.evaluate import evaluate, load_best_model
from modeling_2.load_dataset import load_dataset, feature_selection, scale_data, split_data, get_dataloaders
from modeling_2.smile_net import SmileNet
from modeling_2.train import calculate_pos_weight, train, draw_history


def pipeline(
    dataset_path: Path,
    best_model_path: Path,
    non_feature_cols: list[str],
    output_dir: Path,
    batch_size: int = 32,
    dropout: float = 0.3,
    epochs: int = 50,
    patience: int = 7,
    lr: float = 1e-3,
    test_size: float = 0.2,
    how_many_features: int = 50,
    threshold: float = 0.5,
):
    training_curves_directory = output_dir / "training_curves"
    evaluation_metrics_directory = output_dir / "evaluation_metrics"
    tensorboard_logs_directory = output_dir / "tensorboard_logs"

    create_directories([training_curves_directory, evaluation_metrics_directory, tensorboard_logs_directory])

    writer = SummaryWriter(log_dir=tensorboard_logs_directory)

    device = get_device()
    logger.info(f"Using device: {device}")

    dataset = load_dataset(dataset_path, non_feature_cols)
    X = dataset.drop("label", axis=1)
    y = dataset["label"].values.astype(np.float32)

    X_selected = feature_selection(X, y, how_many_features, output_dir)
    X_scaled = scale_data(X_selected, output_dir)

    X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size)

    train_loader, val_loader = get_dataloaders(X_train, X_test, y_train, y_test, batch_size)

    model = SmileNet(input_dim=X_train.shape[1], dropout_p=dropout).to(device)

    sample_input = torch.randn(1, X_train.shape[1]).to(device)
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

    pos_weight = calculate_pos_weight(y_train, device)
    history = train(
        model, train_loader, val_loader, pos_weight, lr, epochs, patience, threshold, device, best_model_path, writer
    )
    draw_history(history, training_curves_directory)

    evaluate(
        load_best_model(best_model_path, X_train, device),
        val_loader,
        threshold,
        device,
        evaluation_metrics_directory,
        writer,
    )

    final_train_loss = history["train_loss"][-1]
    final_val_loss = history["val_loss"][-1]
    final_train_acc = history["train_acc"][-1]
    final_train_balanced_acc = history["train_balanced_acc"][-1]
    final_val_acc = history["val_acc"][-1]
    final_val_balanced_acc = history["val_balanced_acc"][-1]

    writer.add_scalar("Final/train_loss", final_train_loss, 0)
    writer.add_scalar("Final/val_loss", final_val_loss, 0)
    writer.add_scalar("Final/train_accuracy", final_train_acc, 0)
    writer.add_scalar("Final/train_balanced_accuracy", final_train_balanced_acc, 0)
    writer.add_scalar("Final/val_accuracy", final_val_acc, 0)
    writer.add_scalar("Final/val_balanced_accuracy", final_val_balanced_acc, 0)

    writer.close()

    logger.info(f"TensorBoard logs saved to {tensorboard_logs_directory}")
    logger.info("To view TensorBoard, run: tensorboard --logdir=" + str(tensorboard_logs_directory))


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device
