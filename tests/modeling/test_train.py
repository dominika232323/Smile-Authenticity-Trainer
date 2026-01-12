import matplotlib

matplotlib.use("Agg")  # important for CI / headless environments

from pathlib import Path
from typing import Any

import pytest

from modeling.train import draw_history


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

    def test_draw_history_missing_key_raises_keyerror(self, tmp_path: Path):
        history = {
            "train_loss": [1.0],
            "val_loss": [1.0],
            # missing accuracy keys
        }

        with pytest.raises(KeyError):
            draw_history(history, tmp_path)
