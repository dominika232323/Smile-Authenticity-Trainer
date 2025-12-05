from pathlib import Path
from typing import Any, TypeVar

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

M = TypeVar("M", bound=nn.Module)


def train_model(
        model: M,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epochs: int = 50,
        patience: int = 5,
        lr: float = 1e-3,
        save_path: Path | None = None,
        writer: SummaryWriter | None = None,
) -> tuple[M, dict]:
    logger.info("Training model...")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_loss = float("inf")
    best_state = model.state_dict()
    early_stop_counter = 0

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "lr": [],
    }

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)

            if writer is not None:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Batch/train_loss', loss.item(), global_step)

        train_loss /= len(train_loader)
        train_accuracy = correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)

                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()

                preds = (outputs > 0.5).float()
                correct_val += (preds == y_val).sum().item()
                total_val += y_val.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct_val / total_val

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["lr"].append(current_lr)

        if writer is not None:
            writer.add_scalar('Epoch/train_loss', train_loss, epoch)
            writer.add_scalar('Epoch/val_loss', val_loss, epoch)
            writer.add_scalar('Epoch/train_accuracy', train_accuracy, epoch)
            writer.add_scalar('Epoch/val_accuracy', val_accuracy, epoch)
            writer.add_scalar('Epoch/learning_rate', current_lr, epoch)

            for name, param in model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', param, epoch)

                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            best_state = model.state_dict()

            if save_path is not None:
                logger.info(f"Saving model checkpoint to {save_path}")
                torch.save(best_state, save_path)

            if writer is not None:
                writer.add_scalar('Best/val_loss', best_loss, epoch)
                writer.add_scalar('Best/train_loss', train_loss, epoch)
        else:
            early_stop_counter += 1

            if early_stop_counter >= patience:
                logger.info("\nEarly stopping activated!")
                logger.info(f"Finished at epoch {epoch + 1}.")

                if writer is not None:
                    writer.add_scalar('Training/early_stop_epoch', epoch + 1, 0)

                break

        scheduler.step()

    model.load_state_dict(best_state)

    logger.info(f"Best validation loss: {best_loss:.4f}")
    logger.info("Training complete!")

    return model, history


def plot_training_curves(history: dict[str, Any], output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    loss_path = output_dir / "loss_plot.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()

    logger.info(f"Saved loss plot to {loss_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    acc_path = output_dir / "accuracy_plot.png"
    plt.savefig(acc_path, dpi=300)
    plt.close()

    logger.info(f"Saved accuracy plot to {acc_path}")
