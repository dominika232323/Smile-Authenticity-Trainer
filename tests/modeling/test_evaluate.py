from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from modeling.evaluate import (
    load_best_model,
    save_classification_report,
    save_confusion_matrix,
    save_metrics,
)


class TestLoadBestModel:
    @patch("modeling.evaluate.torch.load")
    @patch("modeling.evaluate.SmileNet")
    def test_load_best_model_defaults(self, mock_smilenet, mock_torch_load, tmp_path):
        model_path = tmp_path / "best_model.pth"
        model_path.touch()
        X_train = pd.DataFrame(torch.randn(10, 5).numpy())
        device = "cpu"
        dropout = 0.5
        hidden_dims = None
        mock_torch_load.return_value = {"key": "value"}

        mock_model = MagicMock()
        mock_smilenet.return_value = mock_model
        mock_model.to.return_value = mock_model

        model = load_best_model(model_path, X_train, device, dropout, hidden_dims)

        assert model == mock_model

        mock_smilenet.assert_called_once_with(input_dim=5, dropout_p=dropout, hidden_dims=[128, 64])
        mock_model.load_state_dict.assert_called_once_with({"key": "value"})
        mock_model.eval.assert_called_once()
        mock_torch_load.assert_called_once_with(model_path)

    @patch("modeling.evaluate.torch.load")
    @patch("modeling.evaluate.SmileNet")
    def test_load_best_model_custom_dims(self, mock_smilenet, mock_torch_load, tmp_path):
        model_path = tmp_path / "best_model.pth"
        model_path.touch()
        X_train = pd.DataFrame(torch.randn(10, 8).numpy())
        device = "cpu"
        dropout = 0.2
        hidden_dims = [32, 16]
        mock_torch_load.return_value = {"key": "value"}

        mock_model = MagicMock()
        mock_smilenet.return_value = mock_model
        mock_model.to.return_value = mock_model

        model = load_best_model(model_path, X_train, device, dropout, hidden_dims)

        assert model == mock_model

        mock_smilenet.assert_called_once_with(input_dim=8, dropout_p=dropout, hidden_dims=[32, 16])
        mock_model.load_state_dict.assert_called_once_with({"key": "value"})
        mock_model.eval.assert_called_once()


class TestSaveClassificationReport:
    def test_save_classification_report_str(self, tmp_path):
        report = "precision recall f1-score\n0.8 0.7 0.75"
        output_dir = tmp_path
        report_path = output_dir / "classification_report.txt"

        save_classification_report(report, output_dir)

        assert report_path.exists()
        with open(report_path, "r") as f:
            content = f.read()
        assert content == report

    def test_save_classification_report_dict(self, tmp_path):
        report = {"precision": 0.8, "recall": 0.7, "f1-score": 0.75}
        output_dir = tmp_path
        report_path = output_dir / "classification_report.txt"

        save_classification_report(report, output_dir)

        assert report_path.exists()
        with open(report_path, "r") as f:
            content = f.read()
        import json

        assert json.loads(content) == report


class TestSaveConfusionMatrix:
    @patch("modeling.evaluate.plt")
    @patch("modeling.evaluate.sns")
    def test_save_confusion_matrix(self, mock_sns, mock_plt, tmp_path):
        import numpy as np

        cm = np.array([[10, 2], [3, 15]])
        output_dir = tmp_path
        cm_path = output_dir / "confusion_matrix.png"

        save_confusion_matrix(cm, output_dir)

        mock_plt.figure.assert_called_once_with(figsize=(6, 5))
        mock_sns.heatmap.assert_called_once()
        args, kwargs = mock_sns.heatmap.call_args

        assert (args[0] == cm).all()
        assert kwargs["annot"] is True
        assert kwargs["fmt"] == "d"
        assert kwargs["cmap"] == "Blues"

        mock_plt.xlabel.assert_called_once_with("Predicted")
        mock_plt.ylabel.assert_called_once_with("True")
        mock_plt.title.assert_called_once_with("Confusion Matrix")
        mock_plt.savefig.assert_called_once_with(cm_path, dpi=300)
        mock_plt.close.assert_called_once()


class TestSaveMetrics:
    def test_save_metrics(self, tmp_path):
        import json

        metrics = {"accuracy": 0.95, "f1": 0.94}
        output_dir = tmp_path
        metrics_path = output_dir / "metrics.json"

        save_metrics(metrics, output_dir)

        assert metrics_path.exists()

        with open(metrics_path, "r") as f:
            content = json.load(f)

        assert content == metrics
