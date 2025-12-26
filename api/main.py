from flask import Flask, request, jsonify
import os
import cv2
import tempfile

app = Flask(__name__)

@app.route("/process-video", methods=["POST"])
def process_video():
    # 1. Check if file exists in request
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]

    if video_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # 2. Save video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        video_path = temp.name
        video_file.save(video_path)

    try:
        # 3. Open video with OpenCV
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return jsonify({"error": "Could not open video"}), 400

        # 4. Get number of frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        # 5. Return result
        return jsonify({
            "frames": frame_count
        })

    finally:
        # 6. Clean up temp file
        if os.path.exists(video_path):
            os.remove(video_path)


if __name__ == "__main__":
    app.run(debug=True)
