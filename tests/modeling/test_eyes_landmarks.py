from pathlib import Path
from unittest.mock import patch, ANY
from modeling.eyes_landmarks import main


class TestEyeLandmarksMain:
    @patch("modeling.eyes_landmarks.setup_logging")
    @patch("modeling.eyes_landmarks.hyperparameter_grid_search")
    @patch("modeling.eyes_landmarks.EYES_LANDMARKS_IN_APEX_CSV", "eyes_landmarks.csv")
    @patch("modeling.eyes_landmarks.EYES_LANDMARKS_RUNS_DIR")
    def test_main_execution(self, mock_runs_dir, mock_grid_search, mock_setup_logging):
        mock_runs_dir.__truediv__.return_value = Path("dummy_runs/threshold_experiment")

        main()

        mock_setup_logging.assert_called_once()
        assert mock_grid_search.called

        expected_dataset_path = "eyes_landmarks.csv"
        expected_runs_dir = Path("dummy_runs/threshold_experiment")
        expected_non_feature_cols = ["filename", "smile_phase", "frame_number"]

        mock_grid_search.assert_any_call(
            expected_dataset_path,
            expected_runs_dir,
            ANY,
            expected_non_feature_cols,
        )
