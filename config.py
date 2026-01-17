from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[0]

LOGS_DIR = PROJ_ROOT / "logs"
DATA_DIR = PROJ_ROOT / "data"
# DATA_DIR = PROJ_ROOT / "data_test"

UvA_NEMO_SMILE_DATABASE_DIR = DATA_DIR / "UvA-NEMO_SMILE_DATABASE"
UvA_NEMO_SMILE_DETAILS = UvA_NEMO_SMILE_DATABASE_DIR / "UvA-NEMO_Smile_Database_File_Details.txt"
UvA_NEMO_SMILE_VIDEOS_DIR = UvA_NEMO_SMILE_DATABASE_DIR / "videos"

PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed_UvA-NEMO_SMILE_DATABASE"
PREPROCESSED_FRAMES_DIR = PREPROCESSED_DATA_DIR / "frames"
PREPROCESSED_FACELANDMARKS_DIR = PREPROCESSED_DATA_DIR / "facelandmarks"
PREPROCESSED_SMILE_PHASES_DIR = PREPROCESSED_DATA_DIR / "smile_phases"

LIP_FEATURES_DIR = PREPROCESSED_DATA_DIR / "lip_features"
ALL_LIP_FEATURES_CSV = PREPROCESSED_DATA_DIR / "all_lip_features.csv"

EYES_FEATURES_DIR = PREPROCESSED_DATA_DIR / "eyes_features"
ALL_EYES_FEATURES_CSV = PREPROCESSED_DATA_DIR / "all_eyes_features.csv"

CHEEKS_FEATURES_DIR = PREPROCESSED_DATA_DIR / "cheeks_features"
ALL_CHEEKS_FEATURES_CSV = PREPROCESSED_DATA_DIR / "all_cheeks_features.csv"

LIPS_LANDMARKS_IN_APEX_CSV = PREPROCESSED_DATA_DIR / "lips_landmarks.csv"
EYES_LANDMARKS_IN_APEX_CSV = PREPROCESSED_DATA_DIR / "eyes_landmarks.csv"
CHEEKS_LANDMARKS_IN_APEX_CSV = PREPROCESSED_DATA_DIR / "cheeks_landmarks.csv"

ORIGINAL_DATA = DATA_DIR / "original_data"
ORIGINAL_FRAMES_DIR = ORIGINAL_DATA / "frames"
ORIGINAL_FACELANDMARKS_DIR = ORIGINAL_DATA / "facelandmarks"

CHECKPOINT_FILE_PATH = PREPROCESSED_DATA_DIR / "processed_files.csv"

RUNS_DIR = PROJ_ROOT / "runs"
LIPS_RUNS_DIR = RUNS_DIR / "lips_runs"
LIPS_LANDMARKS_RUNS_DIR = RUNS_DIR / "lips_landmarks_runs"
EYES_RUNS_DIR = RUNS_DIR / "eyes_runs"
EYES_LANDMARKS_RUNS_DIR = RUNS_DIR / "eyes_landmarks_runs"
CHEEKS_RUNS_DIR = RUNS_DIR / "cheek_runs"
CHEEKS_LANDMARKS_RUNS_DIR = RUNS_DIR / "cheek_landmarks_runs"
ALL_FEATURES_RUNS_DIR = RUNS_DIR / "all_features_runs"

MODELS_DIR = PROJ_ROOT / "models"

LIPS_FEATURES_MODEL = MODELS_DIR / "lips_features.pth"
LIPS_LANDMARKS_MODEL = MODELS_DIR / "lips_landmarks.pth"
EYES_FEATURES_MODEL = MODELS_DIR / "eyes_features.pth"
EYES_LANDMARKS_MODEL = MODELS_DIR / "eyes_landmarks.pth"
CHEEKS_FEATURES_MODEL = MODELS_DIR / "cheeks_features.pth"
CHEEKS_LANDMARKS_MODEL = MODELS_DIR / "cheeks_landmarks.pth"
ALL_FEATURES_MODEL = MODELS_DIR / "all_features.pth"

LIPS_FEATURES_CONFIG = MODELS_DIR / "lips_features_config.json"
LIPS_LANDMARKS_CONFIG = MODELS_DIR / "lips_landmarks_config.json"
EYES_FEATURES_CONFIG = MODELS_DIR / "eyes_features_config.json"
EYES_LANDMARKS_CONFIG = MODELS_DIR / "eyes_landmarks_config.json"
CHEEKS_FEATURES_CONFIG = MODELS_DIR / "cheeks_features_config.json"
CHEEKS_LANDMARKS_CONFIG = MODELS_DIR / "cheeks_landmarks_config.json"
ALL_FEATURES_CONFIG = MODELS_DIR / "all_features_config.json"

LIPS_FEATURES_FEATURE_SELECTOR = MODELS_DIR / "lips_features_feature_selector.joblib"
LIPS_LANDMARKS_FEATURE_SELECTOR = MODELS_DIR / "lips_landmarks_feature_selector.joblib"
EYES_FEATURES_FEATURE_SELECTOR = MODELS_DIR / "eyes_features_feature_selector.joblib"
EYES_LANDMARKS_FEATURE_SELECTOR = MODELS_DIR / "eyes_landmarks_feature_selector.joblib"
CHEEKS_FEATURES_FEATURE_SELECTOR = MODELS_DIR / "cheeks_features_feature_selector.joblib"
CHEEKS_LANDMARKS_FEATURE_SELECTOR = MODELS_DIR / "cheeks_landmarks_feature_selector.joblib"
ALL_FEATURES_FEATURE_SELECTOR = MODELS_DIR / "all_features_feature_selector.joblib"

LIPS_FEATURES_SCALER = MODELS_DIR / "lips_features_scaler.joblib"
LIPS_LANDMARKS_SCALER = MODELS_DIR / "lips_landmarks_scaler.joblib"
EYES_FEATURES_SCALER = MODELS_DIR / "eyes_features_scaler.joblib"
EYES_LANDMARKS_SCALER = MODELS_DIR / "eyes_landmarks_scaler.joblib"
CHEEKS_FEATURES_SCALER = MODELS_DIR / "cheeks_features_scaler.joblib"
CHEEKS_LANDMARKS_SCALER = MODELS_DIR / "cheeks_landmarks_scaler.joblib"
ALL_FEATURES_SCALER = MODELS_DIR / "all_features_scaler.joblib"

EYE_RELATIVE_LEFT = 0.35
EYE_RELATIVE_TOP = 0.35
EYE_RELATIVE_SIZE = (EYE_RELATIVE_LEFT, EYE_RELATIVE_TOP)

DESIRED_FRAME_WIDTH = 256
DESIRED_FRAME_HEIGHT = 256
DESIRED_FRAME_SIZE = (DESIRED_FRAME_WIDTH, DESIRED_FRAME_HEIGHT)
