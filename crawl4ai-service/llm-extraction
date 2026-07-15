import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

app = FastAPI(title="Dynamic AI Scraper API")

# Enable CORS so your Vercel frontend can securely communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Vercel production URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client (Uses official Google GenAI SDK)
# Requires the GEMINI_API_KEY environment variable to be set
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_KEY:
    client = genai.Client(api_key=GEMINI_KEY)
else:
    client = None

# Define what fields the frontend must send us
class ScrapeRequest(BaseModel):
    url: str
    user_query: str

def generate_dynamic_schema(user_query: str) -> dict:
    """Uses Gemini to design a custom JSON Schema based on the user's natural language request."""
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

@app.post("/api/scrape")
async def scrape_endpoint(payload: ScrapeRequest):
    """
    POST endpoint called by the frontend.
    Accepts: { "url": "https://example.com", "user_query": "Extract names and prices" }
    """
    url = payload.url
    user_query = payload.user_query
    
    # 1. Generate dynamic schema
    dynamic_schema = generate_dynamic_schema(user_query)
    
    # 2. Configure Crawl4AI LLM Extraction
    llm_config = LLMConfig(
        provider="gemini/gemini-2.5-flash",
        api_token=GEMINI_KEY
    )
    
    extraction_strategy = LLMExtractionStrategy(
        llm_config=llm_config,
        schema=dynamic_schema,
        extraction_type="schema",
        instruction=f"Extract data matching the user request: {user_query}"
    )
    
    run_config = CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        cache_mode=CacheMode.BYPASS
    )
    
    # 3. Run Crawl4AI Crawler
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=run_config)
            
            if not result.success:
                raise HTTPException(status_code=500, detail=f"Crawl4AI failed: {result.error_message}")
                
            # Parse & return the structured extraction result back to the frontend
            extracted_data = json.loads(result.extracted_content)
            return {
                "success": True,
                "schema_designed": dynamic_schema,
                "data": extracted_data
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal crawler error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}