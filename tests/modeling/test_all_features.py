from unittest.mock import patch, ANY
from modeling.all_features import main


class TestAllFeaturesMain:
    @patch("modeling.all_features.setup_logging")
    @patch("modeling.all_features.hyperparameter_grid_search")
    @patch("modeling.all_features.ALL_CHEEKS_FEATURES_CSV", "cheeks.csv")
    @patch("modeling.all_features.ALL_EYES_FEATURES_CSV", "eyes.csv")
    @patch("modeling.all_features.ALL_LIP_FEATURES_CSV", "lips.csv")
    @patch("modeling.all_features.ALL_FEATURES_RUNS_DIR")
    def test_main_execution(self, mock_runs_dir, mock_grid_search, mock_setup_logging):
        from pathlib import Path

        mock_runs_dir.__truediv__.return_value = Path("dummy_runs/threshold_experiment")
        main()
        mock_setup_logging.assert_called_once()

        assert mock_grid_search.called

        expected_dataset_path = ["lips.csv", "eyes.csv", "cheeks.csv"]
        expected_runs_dir = Path("dummy_runs/threshold_experiment")
        expected_non_feature_cols = ["filename"]

        mock_grid_search.assert_any_call(
            expected_dataset_path,
            expected_runs_dir,
            ANY,  # param_grid
            expected_non_feature_cols,
        )
