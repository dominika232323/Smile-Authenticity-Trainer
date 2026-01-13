from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from modeling.evaluate import load_best_model


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
