import re
from google import genai

from config import GEMINI_API_KEY


def get_tip_from_gemini(lips_score: float, eye_score: float, cheeks_score: float) -> str | None:
    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=(
            f"Write a short tip to a person whose smile is "
            f"{eye_score}% authentic in the eyes area, "
            f"{cheeks_score}% authentic in the cheeks area and "
            f"{lips_score}% authentic in the lips area about how they could improve their smile."
        ),
    )

    return clean_gemini_response(response.text)


def clean_gemini_response(text: str | None) -> str:
    if not text:
        return ""

    if "\\" in text:
        try:
            text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
        except Exception:
            pass

    text = normalize_punctuation(text)

    text = re.sub(r"[*_`#>-]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_punctuation(text: str) -> str:
    replacements = {
        "—": "-",
        "–": "-",
        "“": "'",
        "”": "'",
        '"': "'",
        "’": "'",
        "‘": "'",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    text = text.replace("\\", "")

    return text
