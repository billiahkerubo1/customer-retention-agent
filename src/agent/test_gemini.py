import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY was not found in the .env file.")

client = genai.Client(api_key=API_KEY)

response = client.models.generate_content(
    model="gemini-flash-latest",
    contents="Explain customer churn in one simple sentence."
)

print(response.text)