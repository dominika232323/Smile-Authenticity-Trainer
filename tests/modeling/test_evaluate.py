from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from modeling.evaluate import (
    evaluate,
    load_best_model,
    predict,
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


class TestEvaluate:
    @patch("modeling.evaluate.save_classification_report")
    @patch("modeling.evaluate.save_confusion_matrix")
    @patch("modeling.evaluate.save_metrics")
    def test_evaluate(self, mock_save_metrics, mock_save_cm, mock_save_report, tmp_path):
        device = "cpu"
        threshold = 0.5
        output_dir = tmp_path

        # Mock model
        mock_model = MagicMock()
        # Mock logits for 2 samples: one above threshold, one below
        # Logits are passed through sigmoid, so > 0 will be > 0.5
        mock_model.return_value = torch.tensor([1.0, -1.0])

        # Mock DataLoader
        X_batch = torch.randn(2, 5)
        y_batch = torch.tensor([1, 0])
        mock_loader = [(X_batch, y_batch)]

        # Mock SummaryWriter
        mock_writer = MagicMock()

        metrics = evaluate(
            model=mock_model,
            test_loader=mock_loader,
            threshold=threshold,
            device=device,
            output_dir=output_dir,
            writer=mock_writer,
        )

        # Assertions
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert metrics["accuracy"] == 1.0  # Since preds [1, 0] match labels [1, 0]

        mock_model.assert_called_once()
        mock_save_report.assert_called_once()
        mock_save_cm.assert_called_once()
        mock_save_metrics.assert_called_once_with(metrics, output_dir)

        # Check writer calls
        assert mock_writer.add_scalar.call_count > 0

    def test_evaluate_no_test_loader(self, tmp_path):
        mock_model = MagicMock()
        metrics = evaluate(
            model=mock_model,
            test_loader=None,
            threshold=0.5,
            device="cpu",
            output_dir=tmp_path,
        )

        expected_metrics = {"accuracy": 0.0, "balanced_accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        assert metrics == expected_metrics
        mock_model.assert_not_called()


class TestPredict:
    def test_predict_binary(self):
        # Mock model
        mock_model = MagicMock()
        # 3 samples: 0.1, 0.6, 0.9. With threshold 0.5 -> [0, 1, 1]
        # We need to return logits that after sigmoid give these probs
        # sigmoid(x) = p  => x = log(p / (1-p))
        # log(0.1/0.9) = -2.197
        # log(0.6/0.4) = 0.405
        # log(0.9/0.1) = 2.197
        logits = torch.tensor([[-2.197], [0.405], [2.197]])
        mock_model.return_value = logits

        # Mock DataLoader
        X_batch = torch.randn(3, 5)
        # Loader returning batch as a tuple (X, y)
        mock_loader = [(X_batch, torch.tensor([0, 1, 1]))]

        preds = predict(mock_model, mock_loader, device="cpu", threshold=0.5, return_proba=False)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (3,)
        np.testing.assert_array_equal(preds, [0, 1, 1])
        mock_model.eval.assert_called_once()

    def test_predict_proba(self):
        mock_model = MagicMock()
        # 2 samples
        logits = torch.tensor([[0.0], [2.197]])  # sigmoid(0)=0.5, sigmoid(2.197)=0.9
        mock_model.return_value = logits

        # Mock DataLoader returning just X (not a tuple)
        X_batch = torch.randn(2, 5)
        mock_loader = [X_batch]

        probs = predict(mock_model, mock_loader, device="cpu", return_proba=True)

        assert isinstance(probs, np.ndarray)
        assert probs.shape == (2,)
        np.testing.assert_allclose(probs, [0.5, 0.9], atol=1e-3)

    def test_predict_custom_threshold(self):
        mock_model = MagicMock()
        # Probabilities [0.4, 0.6]
        # log(0.4/0.6) = -0.405
        # log(0.6/0.4) = 0.405
        logits = torch.tensor([[-0.405], [0.405]])
        mock_model.return_value = logits

        X_batch = torch.randn(2, 5)
        mock_loader = [X_batch]

        # Threshold 0.7 -> both should be 0
        preds_high = predict(mock_model, mock_loader, device="cpu", threshold=0.7)
        np.testing.assert_array_equal(preds_high, [0, 0])

        # Threshold 0.3 -> both should be 1
        preds_low = predict(mock_model, mock_loader, device="cpu", threshold=0.3)
        np.testing.assert_array_equal(preds_low, [1, 1])
