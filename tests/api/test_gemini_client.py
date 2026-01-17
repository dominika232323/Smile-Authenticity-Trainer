from unittest.mock import MagicMock, patch
from api.gemini_client import get_tip_from_gemini


class TestGetTipFromGemini:
    @patch("api.gemini_client.genai.Client")
    def test_get_tip_from_gemini_success(self, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "Try to squint your eyes more for a more authentic smile."
        mock_client.models.generate_content.return_value = mock_response

        lips_score = 70.0
        eye_score = 50.0
        cheeks_score = 60.0

        result = get_tip_from_gemini(lips_score, eye_score, cheeks_score)

        assert result == "Try to squint your eyes more for a more authentic smile."
        mock_client.models.generate_content.assert_called_once_with(
            model="gemini-3-flash-preview",
            contents=f"Write a short tip to a person whose smile is {eye_score}% authentic in the eyes area, {cheeks_score}% authentic in the cheeks area and {lips_score}% authentic in the lips area about how they could improve their smile.",
        )

    @patch("api.gemini_client.genai.Client")
    def test_get_tip_from_gemini_none_response(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = None
        mock_client.models.generate_content.return_value = mock_response

        result = get_tip_from_gemini(80.0, 80.0, 80.0)

        assert result is None
