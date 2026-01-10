import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch import Tensor
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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
    model_path: Path,
    writer: SummaryWriter | None = None,
) -> dict[str, list[Any]]:
    logger.info("Training model...")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "train_balanced_acc": [],
        "val_acc": [],
        "val_balanced_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
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

            if writer is not None:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Batch/train_loss", loss.item(), global_step)

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

        if writer is not None:
            writer.add_scalar("Epoch/train_loss", train_loss, epoch)
            writer.add_scalar("Epoch/val_loss", val_loss, epoch)
            writer.add_scalar("Epoch/train_accuracy", train_acc, epoch)
            writer.add_scalar("Epoch/train_balanced_accuracy", train_balanced_acc, epoch)
            writer.add_scalar("Epoch/val_accuracy", val_acc, epoch)
            writer.add_scalar("Epoch/val_balanced_accuracy", val_balanced_acc, epoch)
            writer.add_scalar("Epoch/learning_rate", current_lr, epoch)

            for name, param in model.named_parameters():
                writer.add_histogram(f"Parameters/{name}", param, epoch)

                if param.grad is not None:
                    writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} Balanced Acc: {train_balanced_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Balanced Acc: {val_balanced_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            logger.info(f"Saving model checkpoint to {model_path}")
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, model_path)

            if writer is not None:
                writer.add_scalar("Best/val_loss", best_val_loss, epoch)
                writer.add_scalar("Best/train_loss", train_loss, epoch)

            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                logger.info(f"Finished at epoch {epoch + 1}.")

                if writer is not None:
                    writer.add_scalar("Training/early_stop_epoch", epoch + 1, 0)

                break

    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("Training complete!")

    return history


def draw_history(history: dict[str, list[Any]], output_dir: Path) -> None:
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

    logger.info(f"Saved loss plot to {output_dir / 'loss_plot.png'}")

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

    logger.info(f"Saved accuracy plot to {output_dir / 'accuracy_plot.png'}")

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

    logger.info(f"Saved balanced accuracy plot to {output_dir / 'balanced_accuracy_plot.png'}")


def calculate_pos_weight(y_train: np.ndarray, device: str) -> Tensor:
    logger.info("Calculating positive weight...")
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()

    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    logger.info(f"Positive weight: {pos_weight.item():.4f}")

    return pos_weight
