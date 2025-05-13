import asyncio
import base64

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

schema = {
    "name": "Books Extraction",
    "baseSelector": "article.product_pod",
    "fields": [
        {"name": "title", "selector": "h3 a", "type": "text"},
        {
            "name": "price",
            "selector": "div.product_price p.price_color",
            "type": "text",
        },
        {
            "name": "rating",
            "selector": "p.star-rating",
            "type": "attr",
            "attribute": "class",
        },
        {"name": "stock", "selector": "p.instock", "type": "text"},
    ],
}


extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)


async def main():
    """
    1. The code extract Tilte, Stock and Price of a Product
    but not 'Star Rating'

    2. Use e_commrace_selenium_json.py to extract all details including
    'Star Rating' in json format
    """

    browser_cfg = BrowserConfig(
        headless=True,
        viewport_width=1280,
        viewport_height=720,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    )

    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        exclude_external_links=True,
        extraction_strategy=extraction_strategy,
        wait_for="css:article.product_pod",  # Wait for product elements
        screenshot=True,
        scan_full_page=True,
    )

    # Travel category URL
    url = "http://books.toscrape.com/catalogue/category/books/travel_2/index.html"

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)

        if not result.success:
            print("Crawl failed:", result.error_message)
            return

        # Parse the extracted JSON
        result = result.extracted_content
        if result is not None:
            print("Raw extracted content:")
            print(result)

        # Debug: Screenshot
        screenshot_bytes = None
        if isinstance(result.screenshot, str):
            screenshot_bytes = base64.b64decode(result.screenshot)
        else:
            screenshot_bytes = result.screenshot

        if screenshot_bytes:
            with open("screenshot.png", "wb") as f:
                f.write(screenshot_bytes)
            print("Screenshot saved as screenshot.png")


if __name__ == "__main__":
    """
    1. The code extract Tilte, Stock and Price of a Product
    but not 'Star Rating'

    2. Use e_commrace_selenium_json.py to extract all details including
    'Star Rating' in json format
    """
    asyncio.run(main())
