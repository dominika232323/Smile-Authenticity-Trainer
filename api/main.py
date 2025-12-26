from pathlib import Path
import tempfile
import shutil
import uuid
from flask import Flask, request, jsonify

from data_preprocessing.main import preprocess_video
from data_preprocessing.get_face_landmarks import create_facelandmarks_header
from data_preprocessing.file_utils import create_csv_with_header, create_directories
from data_preprocessing.label_smile_phases import label_smile_phases
from data_preprocessing.extract_lip_features import extract_lip_features
from data_preprocessing.extract_eye_features import extract_eye_features
from data_preprocessing.extract_cheek_features import extract_cheek_features
from config import ORIGINAL_FRAMES_DIR, PREPROCESSED_FRAMES_DIR

app = Flask(__name__)


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

        video_lip_features_df = extract_lip_features(
            normalized_face_landmarks_file_path, smile_phase_file_path, video_fps
        )
        video_eyes_features_df = extract_eye_features(
            normalized_face_landmarks_file_path, smile_phase_file_path, video_fps
        )
        video_cheeks_features_df = extract_cheek_features(
            normalized_face_landmarks_file_path, smile_phase_file_path, video_fps
        )

        return jsonify(
            {
                "lips_features": video_lip_features_df.to_dict(orient="records"),
                "eyes_features": video_eyes_features_df.to_dict(orient="records"),
                "cheeks_features": video_cheeks_features_df.to_dict(orient="records"),
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
