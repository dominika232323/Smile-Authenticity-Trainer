from unittest.mock import MagicMock, patch
from api.gemini_client import get_tip_from_gemini, clean_gemini_response


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

        assert result == ""


class TestCleanGeminiResponse:
    def test_clean_gemini_response_empty(self):
        assert clean_gemini_response("") == ""
        assert clean_gemini_response(None) == ""

    def test_clean_gemini_response_special_chars(self):
        text = "*Special* _chars_ `code` #hashtag >quote -dash"
        expected = "Special chars code hashtag quote dash"

        assert clean_gemini_response(text) == expected

    def test_clean_gemini_response_whitespace(self):
        text = "  Too    many    spaces  "
        expected = "Too many spaces"

        assert clean_gemini_response(text) == expected

    def test_clean_gemini_response_combined(self):
        text = "\n  *Tip*: Try   to _smile_ more!  #happy \n"
        expected = "Tip: Try to smile more! happy"

        assert clean_gemini_response(text) == expected

    def test_clean_gemini_response_quotes(self):
        text = '“Smart” quotes and "normal" quotes'
        expected = "'Smart' quotes and 'normal' quotes"

        assert clean_gemini_response(text) == expected

    def test_clean_gemini_response_escaped_unicode(self):
        text = "Escaped \\u201csmart\\u201d quotes"
        expected = "Escaped 'smart' quotes"
        assert clean_gemini_response(text) == expected

    def test_clean_gemini_response_backslashes(self):
        text = 'Backslash \\ and escaped quote \\"'
        expected = "Backslash and escaped quote '"

        assert clean_gemini_response(text) == expected
