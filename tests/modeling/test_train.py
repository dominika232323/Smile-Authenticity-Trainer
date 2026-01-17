import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from unittest.mock import MagicMock

from modeling.train import calculate_pos_weight, draw_history, train


class TestTrain:
    @pytest.fixture
    def simple_model(self):
        return torch.nn.Linear(10, 1)

    @pytest.fixture
    def dummy_loaders(self):
        X = torch.randn(10, 10)
        y = torch.randint(0, 2, (10, 1)).float()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        return loader, loader

    def test_train_basic(self, tmp_path, simple_model, dummy_loaders):
        train_loader, val_loader = dummy_loaders
        model_path = tmp_path / "model.pth"
        pos_weight = torch.tensor([1.0])

        history = train(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            pos_weight=pos_weight,
            learning_rate=0.001,
            epochs=2,
            patience=2,
            threshold=0.5,
            device="cpu",
            model_path=model_path,
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 2
        assert model_path.exists()

    def test_train_early_stopping(self, tmp_path, simple_model, dummy_loaders):
        train_loader, val_loader = dummy_loaders
        model_path = tmp_path / "model_es.pth"
        pos_weight = torch.tensor([1.0])

        history = train(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            pos_weight=pos_weight,
            learning_rate=0.001,
            epochs=5,
            patience=1,
            threshold=0.5,
            device="cpu",
            model_path=model_path,
        )

        assert len(history["train_loss"]) <= 5
        assert model_path.exists()

    def test_train_with_writer(self, tmp_path, simple_model, dummy_loaders):
        train_loader, val_loader = dummy_loaders
        model_path = tmp_path / "model_writer.pth"
        pos_weight = torch.tensor([1.0])
        writer = MagicMock()

        train(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            pos_weight=pos_weight,
            learning_rate=0.001,
            epochs=2,
            patience=2,
            threshold=0.5,
            device="cpu",
            model_path=model_path,
            writer=writer,
        )

        assert writer.add_scalar.called

        scalar_calls = [call.args[0] for call in writer.add_scalar.call_args_list]

        assert "Batch/train_loss" in scalar_calls
        assert "Epoch/train_loss" in scalar_calls
        assert "Epoch/val_loss" in scalar_calls

        assert writer.add_histogram.called

        histogram_calls = [call.args[0] for call in writer.add_histogram.call_args_list]

        assert any(c.startswith("Parameters/") for c in histogram_calls)
        assert any(c.startswith("Gradients/") for c in histogram_calls)

    def test_train_early_stopping_with_writer(self, tmp_path, simple_model, dummy_loaders):
        train_loader, val_loader = dummy_loaders
        model_path = tmp_path / "model_es_writer.pth"
        pos_weight = torch.tensor([1.0])
        writer = MagicMock()

        def mocked_forward(x):
            return (simple_model.weight.sum() * 0).reshape(1, 1).expand(x.size(0), 1)

        simple_model.forward = mocked_forward

        train(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            pos_weight=pos_weight,
            learning_rate=0.001,
            epochs=10,
            patience=1,
            threshold=0.5,
            device="cpu",
            model_path=model_path,
            writer=writer,
        )

        scalar_calls = [call.args[0] for call in writer.add_scalar.call_args_list]
        assert "Training/early_stop_epoch" in scalar_calls

    def test_train_no_val_loader(self, tmp_path, simple_model, dummy_loaders):
        train_loader, _ = dummy_loaders
        model_path = tmp_path / "model_no_val.pth"
        pos_weight = torch.tensor([1.0])

        history = train(
            model=simple_model,
            train_loader=train_loader,
            val_loader=None,
            pos_weight=pos_weight,
            learning_rate=0.001,
            epochs=2,
            patience=2,
            threshold=0.5,
            device="cpu",
            model_path=model_path,
        )

        assert history == {}
        assert not model_path.exists()


class TestDrawHistory:
    @pytest.fixture
    def sample_history(self) -> dict[str, list[Any]]:
        return {
            "train_loss": [1.0, 0.8, 0.6],
            "val_loss": [1.1, 0.9, 0.7],
            "train_acc": [0.6, 0.7, 0.8],
            "val_acc": [0.55, 0.65, 0.75],
            "train_balanced_acc": [0.58, 0.68, 0.78],
            "val_balanced_acc": [0.56, 0.66, 0.76],
        }

    def test_draw_history_creates_all_plots(self, tmp_path: Path, sample_history):
        draw_history(sample_history, tmp_path)

        expected_files = [
            tmp_path / "loss_plot.png",
            tmp_path / "accuracy_plot.png",
            tmp_path / "balanced_accuracy_plot.png",
        ]

        for file in expected_files:
            assert file.exists(), f"{file.name} was not created"

    def test_draw_history_plots_are_not_empty(self, tmp_path: Path, sample_history):
        draw_history(sample_history, tmp_path)

        for file in tmp_path.iterdir():
            assert file.stat().st_size > 0, f"{file.name} is empty"

    def test_draw_history_does_not_raise(self, tmp_path: Path, sample_history):
        try:
            draw_history(sample_history, tmp_path)
        except Exception as e:
            pytest.fail(f"draw_history raised an exception: {e}")

    def test_draw_history_missing_key_raises_error(self, tmp_path: Path):
        history = {
            "train_loss": [1.0],
            "val_loss": [1.0],
        }

        with pytest.raises(KeyError):
            draw_history(history, tmp_path)


class TestCalculatePosWeight:
    def test_calculate_pos_weight_balanced(self):
        y_train = np.array([0, 0, 1, 1])
        device = "cpu"
        pos_weight = calculate_pos_weight(y_train, device)

        assert isinstance(pos_weight, torch.Tensor)
        assert pos_weight.device.type == "cpu"
        assert torch.allclose(pos_weight, torch.tensor([1.0]))

    def test_calculate_pos_weight_unbalanced(self):
        y_train = np.array([0, 0, 0, 1])
        device = "cpu"
        pos_weight = calculate_pos_weight(y_train, device)

        assert torch.allclose(pos_weight, torch.tensor([3.0]))

    def test_calculate_pos_weight_more_pos(self):
        y_train = np.array([0, 1, 1, 1])
        device = "cpu"
        pos_weight = calculate_pos_weight(y_train, device)

        assert torch.allclose(pos_weight, torch.tensor([1 / 3]))
