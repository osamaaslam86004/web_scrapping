# import asyncio
# import os
# from urllib.parse import urlencode

# import aiohttp
# from loguru import logger
# from my_crawler.google_books import GoogleBooksCrawler

# # Setup loguru
# logger.add("error.log", level="ERROR", format="{time} [{level}] {message}")
# logger.remove()  # Remove default stderr logger
# logger.add(lambda msg: print(msg, end=""), level="INFO", format="{message}")

# # Constants
# API_KEY = open("api_key.txt").read().strip()
# QUERIES = ["python", "machine learning", "data science"]
# MAX_RESULTS = 10
# OUTPUT_DIR = "output"
# MAX_RETRIES = 3
# TIMEOUT = 10

# os.makedirs(OUTPUT_DIR, exist_ok=True)


# async def fetch_with_retries(crawler, query, attempt=1):
#     params = {
#         "q": query,
#         "startIndex": 0,
#         "maxResults": MAX_RESULTS,
#         "printType": "books",
#         "key": API_KEY,
#     }
#     url = f"https://www.googleapis.com/books/v1/volumes?{urlencode(params)}"

#     try:
#         json_str = await asyncio.wait_for(crawler.run(url=url), timeout=TIMEOUT)
#         return json_str
#     except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as e:
#         logger.error(f"Error on query '{query}' (Attempt {attempt}): {e}")
#         if attempt < MAX_RETRIES:
#             await asyncio.sleep(2**attempt)  # exponential backoff
#             return await fetch_with_retries(crawler, query, attempt + 1)
#         else:
#             logger.error(
#                 f"❌ Failed to fetch data for '{query}' after {MAX_RETRIES} attempts."
#             )
#             return None


# async def main():
#     crawler = GoogleBooksCrawler()

#     for query in QUERIES:
#         json_str = await fetch_with_retries(crawler, query)
#         if json_str:
#             filename = f"{query.replace(' ', '_')}.json"
#             filepath = os.path.join(OUTPUT_DIR, filename)
#             with open(filepath, "w", encoding="utf-8") as f:
#                 f.write(json_str)
#             logger.info(f"✅ Saved '{query}' results to {filepath}")


# if __name__ == "__main__":
#     asyncio.run(main())


import asyncio
import os
from urllib.parse import urlencode

import aiohttp
from loguru import logger
from my_crawler.google_books import GoogleBooksCrawler

# Setup loguru (Logging configuration)
logger.add("error.log", level="ERROR", format="{time} [{level}] {message}")
logger.remove()  # Remove default stderr logger to avoid duplicate output
logger.add(lambda msg: print(msg, end=""), level="INFO", format="{message}")
#  This logger setup directs INFO level messages to the console and ERROR level messages to both the console and a file named "error.log".


# Constants (Configuration variables)
API_KEY = (
    open("api_key.txt").read().strip()
)  # Reads and stores the Google Books API key from "api_key.txt". Leading/trailing whitespace is removed.
OUTPUT_DIR = "output"  # Directory where the output JSON files will be saved.
MAX_RESULTS = 10  # Number of results to fetch per query.
MAX_RETRIES = 3  # Maximum number of retry attempts for failed requests.
TIMEOUT = 10  # Timeout in seconds for each API request.
CONCURRENT_LIMIT = 10  # Limits the number of concurrent API requests to avoid rate limiting or overwhelming the server.

os.makedirs(
    OUTPUT_DIR, exist_ok=True
)  # Creates the output directory if it doesn't exist, suppressing errors if it already exists.


def load_queries(filename="queries.txt"):
    """
    Loads search queries from a text file.

    Args:
        filename (str): The name of the file containing the queries (default: "queries.txt").

    Returns:
        list: A list of strings, where each string is a search query.  Empty lines or lines with only whitespace are ignored.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


async def fetch_with_retries(crawler, query, semaphore, attempt=1):
    """
    Fetches data from the Google Books API for a given query, with retries on failure.

    Args:
        crawler (GoogleBooksCrawler): An instance of the GoogleBooksCrawler class.
        query (str): The search query string.
        semaphore (asyncio.Semaphore): A semaphore to limit concurrent requests.
        attempt (int): The current retry attempt number (default: 1).

    Returns:
        tuple: A tuple containing the query and the JSON string of the API response if successful, or (query, None) if all retries fail.
    """


async def fetch_with_retries(crawler, query, semaphore, attempt=1):
    async with semaphore:
        params = {
            "q": query,
            "startIndex": 0,
            "maxResults": MAX_RESULTS,
            "printType": "books",
            "key": API_KEY,
        }
        url = f"https://www.googleapis.com/books/v1/volumes?{urlencode(params)}"

        try:
            json_str = await asyncio.wait_for(crawler.run(url=url), timeout=TIMEOUT)
            return query, json_str
        except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as e:
            logger.error(f"Error on query '{query}' (Attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2**attempt)
                return await fetch_with_retries(crawler, query, semaphore, attempt + 1)
            else:
                logger.error(
                    f"❌ Failed to fetch data for '{query}' after {MAX_RETRIES} attempts."
                )
                return query, None


async def main():
    """
    Main asynchronous function to orchestrate the data fetching process.
    """
    crawler = GoogleBooksCrawler()
    queries = load_queries()
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)

    tasks = [fetch_with_retries(crawler, query, semaphore) for query in queries]
    results = await asyncio.gather(
        *tasks
    )  # Run all tasks concurrently and collect their results. `asyncio.gather` returns a list of results in the same order as the tasks.

    for query, json_str in results:
        if json_str:
            filename = f"{query.replace(' ', '_')}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"✅ Saved '{query}' to {filepath}")
        else:
            print(f"❌ No result for '{query}'")


if __name__ == "__main__":
    # Run the main asynchronous function.  `asyncio.run` manages the event loop for you.
    asyncio.run(main())
