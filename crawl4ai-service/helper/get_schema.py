import os
import json
from fastapi import HTTPException
from pydantic import BaseModel
from helper.gemini_client import get_gemini_client

# Define what fields the frontend must send us
class ScrapeRequest(BaseModel):
    url: str
    user_query: str

def generate_dynamic_schema(user_query: str) -> dict:
    """Uses Gemini to design a custom JSON Schema based on the user's natural language request."""
    client = get_gemini_client()
    if not client:
        raise HTTPException(status_code=500, detail="Gemini API Key is missing on the server.")
    
    prompt = f"""
    Analyze the user's data extraction request: "{user_query}"
    
    Generate a JSON Schema (dictionary format) that can represent this data.
    The outer object MUST be a container/wrapper (e.g., 'items' or 'products' or 'results') 
    which holds a list of the extracted entities.
    
    Return ONLY a valid JSON object. No markdown code block wrap.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate schema via Gemini: {str(e)}")
