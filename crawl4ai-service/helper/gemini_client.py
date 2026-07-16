import os
from google import genai
from google.genai import types

# Initialize Gemini Client (Uses official Google GenAI SDK)
# Requires the GEMINI_API_KEY environment variable to be set
def get_gemini_client():
    GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
    if GEMINI_KEY:
        return genai.Client(api_key=GEMINI_KEY)
    else:
        return None