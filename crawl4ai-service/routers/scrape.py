import os
import json
from fastapi import APIRouter, HTTPException, Depends
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from dependencies import verify_api_token
from helper.get_schema import ScrapeRequest, generate_dynamic_schema

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

router = APIRouter()


@router.post("/api/scrape", dependencies=[Depends(verify_api_token)])
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
