import asyncio
import json

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

schema = {
    "name": "Laptop extraction",
    "baseSelector": "div.card.thumbnail",
    "fields": [
        {
            "name": "image_url",
            "type": "attribute",
            "selector": "img.img-responsive",
            "attribute": "src",
        },
        {"name": "title", "type": "text", "selector": "a.title"},
        {"name": "price", "type": "text", "selector": "h4.price"},
        {"name": "description", "type": "text", "selector": "p.description"},
        {
            "name": "reviews",
            "type": "text",
            "selector": ".review-count>span:first-child",
            "regex": r"(\d+) reviews",
        },
        {
            "name": "star_rating",
            "type": "attribute",
            "selector": "p[data-rating]",
            "attribute": "data-rating",
        },
    ],
}


extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)


async def saving_results(result):
    with open("laptops.json", "w", encoding="utf-8") as f:
        f.write(result)


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
        wait_for="div.card.thumbnail",
        js_code=None,
        js_only=False,
        delay_before_return_html=2.0,
        page_timeout=60000,
        magic=True,
        simulate_user=True,
        override_navigator=True,
    )

    # Laptopcategory URL
    url = "https://webscraper.io/test-sites/e-commerce/static/computers/laptops"

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)

        if not result.success:
            print("Crawl failed:", result.error_message)
            return

        # Parse the extracted JSON
        result = result.extracted_content
        if result is not None:
            print(result)

            await saving_results(result)


if __name__ == "__main__":
    asyncio.run(main())
