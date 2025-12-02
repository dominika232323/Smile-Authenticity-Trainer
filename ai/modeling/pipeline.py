import datetime
from pathlib import Path

import torch
import torch.nn as nn
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


def pipeline(
    dataset_path: Path,
    output_dir: Path,
    batch_size: int = 32,
    dropout: float = 0.3,
    epochs: int = 50,
    patience: int = 5,
    lr: float = 1e-3,
) -> None:
    training_curves_directory = output_dir / "training_curves"
    evaluation_metrics_directory = output_dir / "evaluation_metrics"

    create_directories([training_curves_directory, evaluation_metrics_directory])

    device = get_device()
    logger.info(f"Using device: {device}")

    dataset_df = read_dataset(dataset_path)
    X, y = create_data_tensors(dataset_df, output_dir)
    train_loader, val_loader, X_val, y_val = create_dataloaders(X, y, batch_size)

    model = SimpleMultiLayerPerceptron(input_dim=X.shape[1], dropout_p=dropout)

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs,
        patience,
        lr,
        save_path=output_dir / "simple_multi_layer_perceptron.pth",
    )
    plot_training_curves(history, training_curves_directory)

    probs, preds = evaluate_model(model, X_val, y_val, device, evaluation_metrics_directory)

    onnx_path = output_dir / "model.onnx"
    save_model_to_onnx(model, onnx_path, input_shape=(1, X.shape[1]))

    print(f"\nExample probability of label 1: {probs[0][0] * 100:.2f}%")
    print(f"Example predicted label: {preds[0]}")


def save_model_to_onnx(model: nn.Module, output_path: Path, input_shape: tuple) -> None:
    try:
        device = next(model.parameters()).device

        dummy_input = torch.randn(input_shape, device=device)

        model.eval()

        model_cpu = model.cpu()
        dummy_input_cpu = dummy_input.cpu()

        torch.onnx.export(
            model_cpu,
            dummy_input_cpu,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )

        logger.info(f"Model successfully exported to ONNX format: {output_path}")

    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        raise
