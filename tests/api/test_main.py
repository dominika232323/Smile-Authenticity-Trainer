import pytest
from api.main import app
from io import BytesIO
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np


class TestProcessVideo:
    @pytest.fixture
    def client(self):
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_process_video_no_video(self, client):
        response = client.post("/process-video")
        assert response.status_code == 400
        assert response.get_json() == {"error": "No video file provided"}

    def test_process_video_empty_filename(self, client):
        data = {"video": (BytesIO(b"fake video data"), "")}
        response = client.post("/process-video", data=data, content_type="multipart/form-data")
        assert response.status_code == 400
        assert response.get_json() == {"error": "Empty filename"}

    @patch("api.main.create_directories")
    @patch("api.main.preprocess_video")
    @patch("api.main.label_smile_phases")
    @patch("api.main.pd.read_csv")
    @patch("api.main.get_lips_eyes_cheeks_landmarks_for_file")
    @patch("api.main.load_and_apply_feature_selector")
    @patch("api.main.load_and_apply_scaler")
    @patch("api.main.load_json")
    @patch("api.main.get_dataloaders")
    @patch("api.main.get_device")
    @patch("api.main.load_best_model")
    @patch("api.main.predict")
    @patch("api.main.get_tip_from_gemini")
    @patch("api.main.shutil.rmtree")
    def test_process_video_success(
        self,
        mock_rmtree,
        mock_get_tip,
        mock_predict,
        mock_load_model,
        mock_get_device,
        mock_get_dataloaders,
        mock_load_json,
        mock_scaler,
        mock_selector,
        mock_get_landmarks,
        mock_read_csv,
        mock_label_phases,
        mock_preprocess,
        mock_create_dirs,
        client,
    ):
        # Mocking return values
        mock_preprocess.return_value = 30.0  # FPS

        # Mock DataFrames
        df_mock = MagicMock(spec=pd.DataFrame)
        df_mock.drop.return_value = df_mock
        df_mock.__getitem__.return_value.values.astype.return_value = np.array([0, 1], dtype=np.float32)
        mock_read_csv.return_value = df_mock

        mock_get_landmarks.return_value = (df_mock, df_mock, df_mock)

        mock_selector.return_value = df_mock
        mock_scaler.return_value = df_mock
        mock_load_json.return_value = {"batch_size": 32, "dropout": 0.3, "hidden_dims": [128, 64], "threshold": 0.5}
        mock_get_dataloaders.return_value = (MagicMock(), None)
        mock_get_device.return_value = "cpu"
        mock_load_model.return_value = MagicMock()
        mock_predict.side_effect = [[0.8], [0.7], [0.9]]  # lips, eyes, cheeks
        mock_get_tip.return_value = "Keep smiling!"

        data = {"video": (BytesIO(b"fake video data"), "test_video.mp4")}
        response = client.post("/process-video", data=data, content_type="multipart/form-data")

        assert response.status_code == 200
        json_data = response.get_json()
        assert "score" in json_data
        assert "score_lips" in json_data
        assert "score_eyes" in json_data
        assert "score_cheeks" in json_data
        assert "tip" in json_data
        assert json_data["tip"] == "Keep smiling!"
        assert json_data["score_lips"] == 80.0
        assert json_data["score_eyes"] == 70.0
        assert json_data["score_cheeks"] == 90.0
        assert json_data["score"] == pytest.approx(80.0)

    @patch("api.main.create_directories")
    @patch("api.main.preprocess_video")
    @patch("api.main.shutil.rmtree")
    def test_process_video_failed_processing(self, mock_rmtree, mock_preprocess, mock_create_dirs, client):
        mock_preprocess.return_value = None

        data = {"video": (BytesIO(b"fake video data"), "test_video.mp4")}
        response = client.post("/process-video", data=data, content_type="multipart/form-data")

        assert response.status_code == 500
        assert response.get_json() == {"error": "Failed to process video"}

    @patch("api.main.Path.exists")
    @patch("api.main.create_directories")
    @patch("api.main.preprocess_video")
    @patch("api.main.shutil.rmtree")
    def test_process_video_cleanup(self, mock_rmtree, mock_preprocess, mock_create_dirs, mock_exists, client):
        # Mock exists to return True to enter the if blocks at 160 and 163
        mock_exists.return_value = True
        mock_preprocess.return_value = None

        data = {"video": (BytesIO(b"fake video data"), "test_video.mp4")}
        client.post("/process-video", data=data, content_type="multipart/form-data")

        # Check if shutil.rmtree was called for the directories (lines 160 and 163)
        # It's also called for temp_dir at line 157
        assert mock_rmtree.call_count == 3

    @patch("api.main.create_directories")
    def test_process_video_exception(self, mock_create_dirs, client):
        mock_create_dirs.side_effect = Exception("Unexpected error")

        data = {"video": (BytesIO(b"fake video data"), "test_video.mp4")}
        response = client.post("/process-video", data=data, content_type="multipart/form-data")

        assert response.status_code == 500
        assert response.get_json() == {"error": "Unexpected error"}
