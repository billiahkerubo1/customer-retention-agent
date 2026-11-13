
import os

import pytest
from dotenv import load_dotenv
from google import genai


load_dotenv()


@pytest.mark.skipif(
    os.getenv("RUN_GEMINI_INTEGRATION") != "1",
    reason="Live Gemini integration test is disabled by default.",
)
def test_gemini_connection():
    api_key = os.getenv("GEMINI_API_KEY")

    assert api_key, "GEMINI_API_KEY was not found in the .env file."

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents="Explain customer churn in one simple sentence.",
    )

    assert response.text
    assert len(response.text.strip()) > 0
