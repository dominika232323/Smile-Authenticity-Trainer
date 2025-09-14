from ai.data_preprocessing.get_details import get_details
from ai.config import UvA_NEMO_SMILE_DETAILS, PREPROCESSED_UvA_NEMO_SMILE_DATABASE_DIR


if __name__ == "__main__":
    PREPROCESSED_UvA_NEMO_SMILE_DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    get_details(UvA_NEMO_SMILE_DETAILS).to_csv(PREPROCESSED_UvA_NEMO_SMILE_DATABASE_DIR / "details.csv", index=False)
