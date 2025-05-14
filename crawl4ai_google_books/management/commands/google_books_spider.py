import asyncio
import os
import json
import logging
from urllib.parse import urlencode

from django.core.management.base import BaseCommand
from decouple import config
from mycrawler.google_books import GoogleBooksCrawler  # update import path

import aiohttp

# Setup logging
logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# Constants
API_KEY = config("GOOGLE_BOOKS_API_KEY")
OUTPUT_DIR = "output"
MAX_RESULTS = 10
MAX_RETRIES = 3
TIMEOUT = 10
CONCURRENT_LIMIT = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_queries(filename="queries.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


async def fetch_with_retries(crawler, query, semaphore, attempt=1):
    async with semaphore:
        params = {
            "q": query,
            "startIndex": 0,
            "maxResults": MAX_RESULTS,
            "printType": "books",
            "key": API_KEY
        }
        url = f"https://www.googleapis.com/books/v1/volumes?{urlencode(params)}"

        try:
            json_str = await asyncio.wait_for(crawler.run(url=url), timeout=TIMEOUT)
            return query, json_str
        except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as e:
            logging.error(f"Error on query '{query}' (Attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)
                return await fetch_with_retries(crawler, query, semaphore, attempt + 1)
            else:
                logging.error(f"❌ Failed to fetch data for '{query}' after {MAX_RETRIES} attempts.")
                return query, None


class Command(BaseCommand):
    help = "Scrapes Google Books API using async Crawl4AI spider and saves results as JSON."

    def handle(self, *args, **kwargs):
        asyncio.run(self.async_main())

    async def async_main(self):
        crawler = GoogleBooksCrawler()
        queries = load_queries()
        semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)

        tasks = [fetch_with_retries(crawler, query, semaphore) for query in queries]
        results = await asyncio.gather(*tasks)

        for query, json_str in results:
            if json_str:
                filename = f"{query.replace(' ', '_')}.json"
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(json_str)
                self.stdout.write(self.style.SUCCESS(f"✅ Saved '{query}' to {filepath}"))
            else:
                self.stdout.write(self.style.ERROR(f"❌ Failed: '{query}'"))
