import os
from fastapi import HTTPException, Header, Depends

CRAWL4AI_API_TOKEN = os.environ.get("CRAWL4AI_API_TOKEN")

def verify_api_token(authorization: str = Header(None)):
    """Validates that the client sent the correct Bearer token in the headers."""
    # Skip validation if you haven't set a token in the server's environment variables
    if not CRAWL4AI_API_TOKEN:
        return
        
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
        
    try:
        scheme, token = authorization.split(" ")
        if scheme.lower() != "bearer" or token != CRAWL4AI_API_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid API token")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")