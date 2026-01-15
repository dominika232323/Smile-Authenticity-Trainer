import re
import datetime
from unittest.mock import patch
from modeling.pipeline import get_timestamp, get_device, hyperparameter_grid_search


class TestGetTimestamp:
    def test_get_timestamp_format(self):
        timestamp = get_timestamp()
        pattern = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"

        assert re.match(pattern, timestamp)

    @patch("modeling.pipeline.datetime")
    def test_get_timestamp_fixed_value(self, mock_datetime_module):
        fixed_now = datetime.datetime(2023, 10, 27, 12, 30, 45)
        mock_datetime_module.datetime.now.return_value = fixed_now
        timestamp = get_timestamp()

        assert timestamp == "2023-10-27_12-30-45"


class TestGetDevice:
    @patch("modeling.pipeline.torch.cuda.is_available")
    def test_get_device_cuda(self, mock_cuda_available):
        mock_cuda_available.return_value = True
        device = get_device()

        assert device == "cuda"

    @patch("modeling.pipeline.torch.cuda.is_available")
    def test_get_device_cpu(self, mock_cuda_available):
        mock_cuda_available.return_value = False
        device = get_device()

        assert device == "cpu"


class TestHyperparameterGridSearch:
    @patch("modeling.pipeline.ParameterGrid")
    @patch("modeling.pipeline.pipeline")
    @patch("modeling.pipeline.create_directories")
    @patch("modeling.pipeline.get_timestamp")
    @patch("builtins.open")
    def test_hyperparameter_grid_search_calls(
        self,
        mock_open,
        mock_get_timestamp,
        mock_create_directories,
        mock_pipeline,
        mock_parameter_grid,
    ):
        from pathlib import Path

        # Setup
        dataset_path = Path("dummy_dataset")
        runs_dir = Path("dummy_runs")
        param_grid = {"lr": [0.01, 0.001]}
        non_feature_cols = ["id"]

        mock_parameter_grid.return_value = [{"lr": 0.01}, {"lr": 0.001}]
        mock_get_timestamp.side_effect = ["ts1", "ts2"]

        # Execute
        hyperparameter_grid_search(dataset_path, runs_dir, param_grid, non_feature_cols)

        # Assertions
        assert mock_pipeline.call_count == 2
        assert mock_create_directories.call_count == 2
        assert mock_open.call_count == 2

        # Check first call to pipeline
        mock_pipeline.assert_any_call(
            dataset_path,
            runs_dir / "ts1" / "best_model.pth",
            non_feature_cols,
            runs_dir / "ts1",
            32,  # default batch_size
            0.3,  # default dropout
            50,  # default epochs
            7,  # default patience
            0.01,  # from params
            0.2,  # default test_size
            50,  # default how_many_features
            0.5,  # default threshold
            [128, 64],  # default hidden_dims
        )
