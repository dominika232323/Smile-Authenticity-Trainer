import datetime
from pathlib import Path

import torch
from loguru import logger

from ai.data_preprocessing.file_utils import create_directories
from ai.modeling.evaluate import evaluate_model
from ai.modeling.load_dataset import create_data_tensors, create_dataloaders, read_dataset
from ai.modeling.simple_multi_layer_perceptron import SimpleMultiLayerPerceptron
from ai.modeling.train import plot_training_curves, train_model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_timestamp() -> str:
    ct = datetime.datetime.now()
    return ct.strftime("%Y-%m-%d_%H-%M-%S")


def pipeline(dataset_path: Path, output_dir: Path):
    training_curves_directory = output_dir / "training_curves"
    evaluation_metrics_directory = output_dir / "evaluation_metrics"

    create_directories([training_curves_directory, evaluation_metrics_directory])

    device = get_device()
    logger.info(f"Using device: {device}")

    dataset_df = read_dataset(dataset_path)

    X, y = create_data_tensors(dataset_df)
    print(X.shape, y.shape)

    train_loader, val_loader, X_val, y_val = create_dataloaders(X, y, batch_size=32)
    print(len(train_loader), len(val_loader))
    print(X_val.shape, y_val.shape)

    model = SimpleMultiLayerPerceptron(input_dim=X.shape[1])

    model, history = train_model(model, train_loader, val_loader, device)
    plot_training_curves(history, training_curves_directory)

    probs, preds = evaluate_model(model, X_val, y_val, device, evaluation_metrics_directory)

    print(f"\nExample probability of label 1: {probs[0][0] * 100:.2f}%")
    print(f"Example predicted label: {preds[0]}")
