import asyncio
import json

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

# Map rating words to numbers
RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}

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
            "name": "star_rating_class",
            "type": "attribute",
            "selector": "p.star-rating",
            "attribute": "class",
        },
        {"name": "stock", "selector": "p.instock", "type": "text"},
    ],
}


def parse_star_rating(class_list):
    # Loop over the class list and find a match
    for cls in class_list:
        if cls in RATING_MAP:
            return RATING_MAP[cls]
    return 0  # default if no match found


extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)


async def main():
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
        wait_for="css:article.product_pod",
        delay_before_return_html=2.0,
        page_timeout=60000,
        magic=True,
        simulate_user=True,
        override_navigator=True,
    )

    url = "http://books.toscrape.com/catalogue/category/books/travel_2/index.html"

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)

        if not result.success:
            print("Crawl failed:", result.error_message)
            return

        result = result.extracted_content
        if result is not None:
            print("Raw extracted content:", result)
            result = json.loads(result)

            # Convert class to numeric star rating
            for item in result:
                class_str = item.get("star_rating_class", "")
                item["star_rating"] = parse_star_rating(class_str)
                item.pop("star_rating_class", None)

            print(result)


if __name__ == "__main__":
    """
    1. The code extract Tilte, Stock and Price, Star Rating count of a Product'
    2. Use e_commrace_selenium_json.py to extract all details using selenium
    """
    asyncio.run(main())
