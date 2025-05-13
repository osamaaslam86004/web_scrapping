# File: async_webcrawler_multiple_urls_example.py
import os
import sys

# append 2 parent directories to sys.path to import crawl4ai
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent_dir)

import asyncio

from crawl4ai import AsyncWebCrawler


async def main():
    # Initialize the AsyncWebCrawler
    async with AsyncWebCrawler(verbose=True) as crawler:
        # List of URLs to crawl
        urls = [
            "https://github.com/veryacademy/Django-ORM-Mastery-DJ003",
            "https://medium.com/@farad.dev/mastering-djangos-orm-advanced-queries-and-performance-tips-e8613978176d",
            "http://books.toscrape.com/catalogue/category/books/travel_2/index.html",
        ]

        # Set up crawling parameters
        result = await crawler.arun_many(
            urls=urls,
            bypass_cache=True,
            verbose=True,
        )

        for result in result:
            # Process the results
            if result.success:
                print(f"Successfully crawled: {result.url}")
                print(f"Title: {result.metadata.get('title', 'N/A')}")
                print(f"Word count: {len(result.markdown.split())}")
                print(
                    f"Number of links: {len(result.links.get('internal', [])) + len(result.links.get('external', []))}"
                )
                print(f"Number of images: {len(result.media.get('images', []))}")
                print(result.markdown)
                # print(result.cleaned_html)
            else:
                print(f"Failed to crawl: {result.url}")
                print(f"Error: {result.error_message}")
                print("---")


if __name__ == "__main__":
    asyncio.run(main())
