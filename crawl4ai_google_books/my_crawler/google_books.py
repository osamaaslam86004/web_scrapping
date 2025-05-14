import json

import aiohttp
from crawl4ai.hub import BaseCrawler


class GoogleBooksCrawler(BaseCrawler):
    async def run(self, url: str = "", **kwargs) -> str:
        """
        Accepts a URL and fetches JSON data from Google Books API
        using aiohttp (as required by crawl4ai).
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                results = []

                for item in data.get("items", []):
                    results.append(
                        {
                            "title": item["volumeInfo"].get("title"),
                            "authors": item["volumeInfo"].get("authors"),
                            "publishedDate": item["volumeInfo"].get("publishedDate"),
                            "description": item["volumeInfo"].get("description"),
                            "infoLink": item["volumeInfo"].get("infoLink"),
                        }
                    )

                return json.dumps(results)
