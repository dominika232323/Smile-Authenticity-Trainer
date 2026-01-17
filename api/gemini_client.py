from google import genai

from config import GEMINI_API_KEY


def get_tip_from_gemini(lips_score: float, eye_score: float, cheeks_score: float) -> str | None:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=f"Write a short tip to a person whose smile is {eye_score}% authentic in the eyes area, {cheeks_score}% authentic in the cheeks area and {lips_score}% authentic in the lips area about how they could improve their smile.",
    )

    return response.text
