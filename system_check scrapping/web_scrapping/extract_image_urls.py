import asyncio
import json
import os
import re

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode
from crawl4ai.async_configs import CrawlerRunConfig

# Target URL
url = "https://www.jetbrains.com/guide/django/tips/django-system-check-framework/"

# Crawl4AI Configuration for Extracting Images
run_config = CrawlerRunConfig(
    css_selector="//img/@src",
    verbose=True,
    cache_mode=CacheMode.ENABLED,
    exclude_external_links=True,
    remove_overlay_elements=True,
    process_iframes=False,
    wait_for="js:() => window.loaded === true",
    screenshot=False,
    pdf=False,
    excluded_tags=["nav", "footer"],
    remove_forms=True,
    exclude_social_media_links=True,
)

browser_config = BrowserConfig(
    browser_type="chromium",
    ignore_https_errors=True,
    java_script_enabled=True,
    headless=True,
    text_mode=True,
    viewport_width=1280,
    viewport_height=720,
    verbose=True,
    extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
)


async def save_image_data(image_data):
    """
    Saves extracted image data (src, alt, desc) in a prettified JSON file.
    """
    os.makedirs("output", exist_ok=True)
    output_file = os.path.join("output", "images_data.json")

    # Extract only relevant fields
    formatted_data = [
        {
            "src": img.get("src", ""),
            "alt": img.get("alt", ""),
            "desc": img.get("desc", ""),
        }
        for img in image_data
        if isinstance(img, dict)
    ]

    # Save JSON in a prettified format
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=4)

    print(f"✅ Saved extracted image data to {output_file}")


async def main():
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Run the crawler
        result = await crawler.arun(
            url=url, bypass_cache=True, CrawlerRunConfig=run_config
        )

        if result.success:
            print(f"✅ Successfully crawled: {result.url}")

            # Extract all images
            all_images = result.media.get("images", [])
            print(all_images)

            # Filter images likely containing Django code
            await save_image_data(all_images)

            print(f"📌 Total Images Found: {len(all_images)}")

        else:
            print(f"❌ Failed to crawl: {result.url}")
            print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
