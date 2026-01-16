from pathlib import Path
from unittest.mock import patch, ANY
from modeling.eyes_features import main


class TestEyeFeaturesMain:
    @patch("modeling.eyes_features.setup_logging")
    @patch("modeling.eyes_features.hyperparameter_grid_search")
    @patch("modeling.eyes_features.ALL_EYES_FEATURES_CSV", "eyes.csv")
    @patch("modeling.eyes_features.EYES_RUNS_DIR")
    def test_main_execution(self, mock_runs_dir, mock_grid_search, mock_setup_logging):
        mock_runs_dir.__truediv__.return_value = Path("dummy_runs/threshold_experiment")

        main()

        mock_setup_logging.assert_called_once()
        assert mock_grid_search.called

        expected_dataset_path = "eyes.csv"
        expected_runs_dir = Path("dummy_runs/threshold_experiment")
        expected_non_feature_cols = ["filename"]

        mock_grid_search.assert_any_call(
            expected_dataset_path,
            expected_runs_dir,
            ANY,
            expected_non_feature_cols,
        )
