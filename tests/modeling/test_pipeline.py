import re
import datetime
from unittest.mock import patch
from modeling.pipeline import get_timestamp, get_device, hyperparameter_grid_search, pipeline


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


class TestPipeline:
    @patch("modeling.pipeline.create_directories")
    @patch("modeling.pipeline.SummaryWriter")
    def test_pipeline_invalid_path(self, mock_summary_writer, mock_create_directories):
        from pathlib import Path
        import pytest

        with pytest.raises(ValueError, match="Invalid dataset path"):
            pipeline("not_a_path_or_list", Path("best.pth"), [], Path("out"))

    @patch("modeling.pipeline.load_all_features")
    @patch("modeling.pipeline.create_directories")
    @patch("modeling.pipeline.SummaryWriter")
    @patch("modeling.pipeline.get_device")
    @patch("modeling.pipeline.load_dataset")
    @patch("modeling.pipeline.split_data")
    @patch("modeling.pipeline.feature_selection")
    @patch("modeling.pipeline.scale_data")
    @patch("modeling.pipeline.get_dataloaders")
    @patch("modeling.pipeline.SmileNet")
    @patch("modeling.pipeline.torch.randn")
    @patch("modeling.pipeline.calculate_pos_weight")
    @patch("modeling.pipeline.train")
    @patch("modeling.pipeline.draw_history")
    @patch("modeling.pipeline.evaluate")
    @patch("modeling.pipeline.load_best_model")
    @patch("modeling.pipeline.torch.tensor")
    def test_pipeline_execution_flow_list_path(
        self,
        mock_torch_tensor,
        mock_load_best_model,
        mock_evaluate,
        mock_draw_history,
        mock_train,
        mock_calculate_pos_weight,
        mock_torch_randn,
        mock_smile_net,
        mock_get_dataloaders,
        mock_scale_data,
        mock_feature_selection,
        mock_split_data,
        mock_load_dataset,
        mock_get_device,
        mock_summary_writer,
        mock_create_directories,
        mock_load_all_features,
    ):
        from pathlib import Path
        import pandas as pd
        import numpy as np

        # Setup
        dataset_paths = [Path("d1"), Path("d2"), Path("d3")]
        best_model_path = Path("best_model.pth")
        non_feature_cols = ["id"]
        output_dir = Path("output")

        # Mocking return values
        mock_get_device.return_value = "cpu"

        mock_dataset = pd.DataFrame({"feature1": [1, 2], "label": [0, 1]})
        mock_load_all_features.return_value = mock_dataset

        mock_split_data.return_value = (
            pd.DataFrame({"feature1": [1]}),
            pd.DataFrame({"feature1": [2]}),
            np.array([0]),
            np.array([1]),
        )

        mock_feature_selection.return_value = (pd.DataFrame({"feature1": [1]}), pd.DataFrame({"feature1": [2]}))
        mock_scale_data.return_value = (pd.DataFrame({"feature1": [1]}), pd.DataFrame({"feature1": [2]}))
        mock_get_dataloaders.return_value = ("train_loader", "val_loader")

        mock_model = mock_smile_net.return_value
        mock_model.to.return_value = mock_model

        mock_train.return_value = {
            "train_loss": [0.5],
            "val_loss": [0.4],
            "train_acc": [0.8],
            "train_balanced_acc": [0.75],
            "val_acc": [0.85],
            "val_balanced_acc": [0.8],
        }

        # Execute
        pipeline(dataset_paths, best_model_path, non_feature_cols, output_dir)

        # Assertions
        mock_load_all_features.assert_called_with(dataset_paths[0], dataset_paths[1], dataset_paths[2])
        mock_load_dataset.assert_not_called()

    @patch("modeling.pipeline.create_directories")
    @patch("modeling.pipeline.SummaryWriter")
    @patch("modeling.pipeline.get_device")
    @patch("modeling.pipeline.load_dataset")
    @patch("modeling.pipeline.split_data")
    @patch("modeling.pipeline.feature_selection")
    @patch("modeling.pipeline.scale_data")
    @patch("modeling.pipeline.get_dataloaders")
    @patch("modeling.pipeline.SmileNet")
    @patch("modeling.pipeline.torch.randn")
    @patch("modeling.pipeline.calculate_pos_weight")
    @patch("modeling.pipeline.train")
    @patch("modeling.pipeline.draw_history")
    @patch("modeling.pipeline.evaluate")
    @patch("modeling.pipeline.load_best_model")
    @patch("modeling.pipeline.torch.tensor")
    def test_pipeline_execution_flow(
        self,
        mock_torch_tensor,
        mock_load_best_model,
        mock_evaluate,
        mock_draw_history,
        mock_train,
        mock_calculate_pos_weight,
        mock_torch_randn,
        mock_smile_net,
        mock_get_dataloaders,
        mock_scale_data,
        mock_feature_selection,
        mock_split_data,
        mock_load_dataset,
        mock_get_device,
        mock_summary_writer,
        mock_create_directories,
    ):
        from pathlib import Path
        import pandas as pd
        import numpy as np

        # Setup
        dataset_path = Path("dummy_dataset.csv")
        best_model_path = Path("best_model.pth")
        non_feature_cols = ["id"]
        output_dir = Path("output")

        # Mocking return values
        mock_get_device.return_value = "cpu"

        mock_dataset = pd.DataFrame({"feature1": [1, 2], "label": [0, 1]})
        mock_load_dataset.return_value = mock_dataset

        mock_split_data.return_value = (
            pd.DataFrame({"feature1": [1]}),
            pd.DataFrame({"feature1": [2]}),
            np.array([0]),
            np.array([1]),
        )

        mock_feature_selection.return_value = (pd.DataFrame({"feature1": [1]}), pd.DataFrame({"feature1": [2]}))
        mock_scale_data.return_value = (pd.DataFrame({"feature1": [1]}), pd.DataFrame({"feature1": [2]}))
        mock_get_dataloaders.return_value = ("train_loader", "val_loader")

        mock_model = mock_smile_net.return_value
        mock_model.to.return_value = mock_model

        mock_train.return_value = {
            "train_loss": [0.5],
            "val_loss": [0.4],
            "train_acc": [0.8],
            "train_balanced_acc": [0.75],
            "val_acc": [0.85],
            "val_balanced_acc": [0.8],
        }

        # Execute
        pipeline(dataset_path, best_model_path, non_feature_cols, output_dir)

        # Assertions
        mock_create_directories.assert_called()
        mock_summary_writer.assert_called_with(log_dir=output_dir / "tensorboard_logs")
        mock_load_dataset.assert_called_with(dataset_path, non_feature_cols)
        mock_split_data.assert_called()
        mock_feature_selection.assert_called()
        mock_scale_data.assert_called()
        mock_get_dataloaders.assert_called()
        mock_smile_net.assert_called()
        mock_train.assert_called()
        mock_draw_history.assert_called()
        mock_evaluate.assert_called()
        mock_summary_writer.return_value.close.assert_called_once()
