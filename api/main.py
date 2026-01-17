from pathlib import Path
import tempfile
import shutil
import uuid

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

from api.gemini_client import get_tip_from_gemini
from data_preprocessing.label_smile_phases import label_smile_phases
from data_preprocessing.main import preprocess_video
from data_preprocessing.get_face_landmarks import create_facelandmarks_header
from data_preprocessing.file_utils import create_csv_with_header, create_directories, load_json
from config import (
    ORIGINAL_FRAMES_DIR,
    PREPROCESSED_FRAMES_DIR,
    LIPS_LANDMARKS_FEATURE_SELECTOR,
    EYES_LANDMARKS_FEATURE_SELECTOR,
    CHEEKS_LANDMARKS_FEATURE_SELECTOR,
    LIPS_LANDMARKS_SCALER,
    EYES_LANDMARKS_SCALER,
    CHEEKS_LANDMARKS_SCALER,
    LIPS_LANDMARKS_CONFIG,
    EYES_LANDMARKS_CONFIG,
    CHEEKS_LANDMARKS_CONFIG,
    LIPS_LANDMARKS_MODEL,
    EYES_LANDMARKS_MODEL,
    CHEEKS_LANDMARKS_MODEL,
)
from data_preprocessing.save_landmarks_in_apex import get_lips_eyes_cheeks_landmarks_for_file
from modeling.evaluate import load_best_model, predict
from modeling.load_dataset import get_dataloaders, load_and_apply_feature_selector, load_and_apply_scaler
from modeling.pipeline import get_device

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 500  # 500 MB


@app.route("/process-video", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]

    if video_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    request_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.mkdtemp())

    original_video_directory = ORIGINAL_FRAMES_DIR / request_id
    preprocessed_video_directory = PREPROCESSED_FRAMES_DIR / request_id

    try:
        create_directories([original_video_directory, preprocessed_video_directory])

        video_path = temp_dir / f"{request_id}.mp4"
        video_file.save(str(video_path))

        landmarks_header = create_facelandmarks_header()

        original_face_landmarks_file_path = temp_dir / f"{request_id}_original.csv"
        create_csv_with_header(original_face_landmarks_file_path, landmarks_header)

        normalized_face_landmarks_file_path = temp_dir / f"{request_id}_normalized.csv"
        create_csv_with_header(normalized_face_landmarks_file_path, landmarks_header)

        video_fps = preprocess_video(video_path, original_face_landmarks_file_path, normalized_face_landmarks_file_path)

        if video_fps is None:
            return jsonify({"error": "Failed to process video"}), 500

        smile_phase_file_path = temp_dir / f"{request_id}_smile_phases.csv"
        label_smile_phases(normalized_face_landmarks_file_path, smile_phase_file_path)

        smile_phases_df = pd.read_csv(smile_phase_file_path)
        landmarks_df = pd.read_csv(normalized_face_landmarks_file_path)

        cheeks_landmarks_df, eyes_landmarks_df, lips_landmarks_df = get_lips_eyes_cheeks_landmarks_for_file(
            smile_phases_df, landmarks_df, request_id, -1
        )

        non_features_columns = ["filename", "frame_number", "smile_phase", "label"]

        lips_X = lips_landmarks_df.drop(columns=non_features_columns, axis=1)
        eyes_X = eyes_landmarks_df.drop(columns=non_features_columns, axis=1)
        cheeks_X = cheeks_landmarks_df.drop(columns=non_features_columns, axis=1)
        y = lips_landmarks_df["label"].values.astype(np.float32)

        lips_X = load_and_apply_feature_selector(lips_X, LIPS_LANDMARKS_FEATURE_SELECTOR)
        eyes_X = load_and_apply_feature_selector(eyes_X, EYES_LANDMARKS_FEATURE_SELECTOR)
        cheeks_X = load_and_apply_feature_selector(cheeks_X, CHEEKS_LANDMARKS_FEATURE_SELECTOR)

        lips_X = load_and_apply_scaler(lips_X, LIPS_LANDMARKS_SCALER)
        eyes_X = load_and_apply_scaler(eyes_X, EYES_LANDMARKS_SCALER)
        cheeks_X = load_and_apply_scaler(cheeks_X, CHEEKS_LANDMARKS_SCALER)

        lips_config = load_json(LIPS_LANDMARKS_CONFIG)
        eyes_config = load_json(EYES_LANDMARKS_CONFIG)
        cheeks_config = load_json(CHEEKS_LANDMARKS_CONFIG)

        lips_loader, _ = get_dataloaders(lips_X, None, y, None, batch_size=lips_config.get("batch_size", 32))
        eyes_loader, _ = get_dataloaders(eyes_X, None, y, None, batch_size=eyes_config.get("batch_size", 32))
        cheeks_loader, _ = get_dataloaders(cheeks_X, None, y, None, batch_size=cheeks_config.get("batch_size", 32))

        device = get_device()

        lips_model = load_best_model(
            LIPS_LANDMARKS_MODEL,
            lips_X,
            device,
            lips_config.get("dropout", 0.3),
            lips_config.get("hidden_dims", [128, 64]),
        )
        eyes_model = load_best_model(
            EYES_LANDMARKS_MODEL,
            eyes_X,
            device,
            eyes_config.get("dropout", 0.3),
            eyes_config.get("hidden_dims", [128, 64]),
        )
        cheeks_model = load_best_model(
            CHEEKS_LANDMARKS_MODEL,
            cheeks_X,
            device,
            cheeks_config.get("dropout", 0.3),
            cheeks_config.get("hidden_dims", [128, 64]),
        )

        lips_probs = predict(lips_model, lips_loader, device, lips_config.get("threshold", 0.5), True)
        eyes_probs = predict(eyes_model, eyes_loader, device, eyes_config.get("threshold", 0.5), True)
        cheeks_probs = predict(cheeks_model, cheeks_loader, device, cheeks_config.get("threshold", 0.5), True)

        lips_score = float(np.mean(lips_probs)) * 100
        eyes_score = float(np.mean(eyes_probs)) * 100
        cheeks_score = float(np.mean(cheeks_probs)) * 100

        score = (lips_score + eyes_score + cheeks_score) / 3
        tip = get_tip_from_gemini(lips_score, eyes_score, cheeks_score)

        return jsonify(
            {
                "score": score,
                "score_lips": lips_score,
                "score_eyes": eyes_score,
                "score_cheeks": cheeks_score,
                "tip": tip,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        shutil.rmtree(temp_dir)

        if original_video_directory.exists():
            shutil.rmtree(original_video_directory)

        if preprocessed_video_directory.exists():
            shutil.rmtree(preprocessed_video_directory)


if __name__ == "__main__":
    app.run(debug=True)
