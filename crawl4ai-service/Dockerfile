# Use the official crawl4ai image as it contains pre-configured Chromium and Playwright dependencies
FROM unclecode/crawl4ai:latest

# Install additional python dependencies for our FastAPI server and Google GenAI SDK
RUN pip install --no-cache-dir fastapi uvicorn google-genai

# Set the working directory
WORKDIR /app

# Copy your FastAPI app code into the container
COPY main.py /app/main.py

# Expose FastAPI's port
EXPOSE 8000

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]