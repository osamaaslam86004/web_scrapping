import os
import json
import httpx
from fastapi import HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from helper.gemini_client import get_gemini_client

class ScrapeRequest(BaseModel):
    url: str
    user_query: str

def generate_dynamic_schema(user_query: str) -> dict:
    """Uses OpenRouter (or falls back to Gemini) to design a custom JSON Schema."""
    
    prompt = f"""
    Analyze the user's data extraction request: "{user_query}"
    
    Generate a JSON Schema (dictionary format) that can represent this data.
    The outer object MUST be a container/wrapper (e.g., 'items' or 'products' or 'results') 
    which holds a list of the extracted entities.
    
    Return ONLY a valid JSON object. No markdown code block wrap.
    """

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    # --- STRATEGY 1: TRY OPENROUTER FREE PLAN ---
    if openrouter_key:
        try:
            # Using httpx for synchronous/clean execution matching your current flow
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openrouter_key}",
                        "Content-Type": "application/json",
                        # OpenRouter rankings prefer a site URL/Title for free tier optimization
                        "HTTP-Referer": "http://localhost:3000", 
                        "X-Title": "AI Scraper Workspace"
                    },
                    json={
                        # Using a highly reliable, fast free model on OpenRouter
                        "model": "meta-llama/llama-3-8b-instruct:free",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        # Enforce JSON formatting at the API layer
                        "response_format": { "type": "json_object" }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    ai_text = result['choices'][0]['message']['content']
                    return json.loads(ai_text)
                
                # If OpenRouter fails with something other than 200, log it and fall through to Gemini
                print(f"[OpenRouter Error]: Status {response.status_code} - {response.text}")
        except Exception as openrouter_err:
            print(f"[OpenRouter Exception]: {str(openrouter_err)}")

    # --- STRATEGY 2: FALLBACK TO ORIGINAL GEMINI CLIENT ---
    print("Falling back to Gemini client...")
    gemini_client = get_gemini_client()
    if not gemini_client:
        raise HTTPException(
            status_code=500, 
            detail="Both OpenRouter and Gemini API configurations are unavailable."
        )
    
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as gemini_err:
        raise HTTPException(
            status_code=503, 
            detail=f"Schema generation failed across all available free tiers. \
            Gemini Error: {str(gemini_err)}"
        )