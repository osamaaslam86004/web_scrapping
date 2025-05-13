import asyncio
import json
import os
from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode
from crawl4ai.async_configs import CrawlerRunConfig

""""
Method 1: Use Crawl4AI
Method 2: Use aiohttp

"""

METHOD_1 = False
METHOD_2 = True

# Path to filtered JSON file
JSON_FILE = "output/images_data.json"
IMAGE_DIR = "output/images"

# Ensure output directory exists
os.makedirs(IMAGE_DIR, exist_ok=True)

# Method 1 Here:
if METHOD_1:

    # Crawl4AI Configuration for Extracting Images
    run_config = CrawlerRunConfig(
        verbose=True,
        cache_mode=CacheMode.ENABLED,
        exclude_external_links=True,
        remove_overlay_elements=True,
        process_iframes=False,
        js_code="js:() => window.loaded === true",
        wait_for=5,
        screenshot=False,
        pdf=False,
        excluded_tags=["nav", "footer"],
        remove_forms=True,
        exclude_social_media_links=True,
    )

    browser_config = BrowserConfig(
        accept_downloads=True,
        downloads_path=IMAGE_DIR,
        browser_type="chromium",
        ignore_https_errors=True,
        java_script_enabled=True,
        text_mode=False,
        verbose=True,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )

    async def download_images():
        """
        Reads filtered image URLs from JSON and downloads them using Crawl4ai.
        """
        # Load manually filtered images
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            image_data = json.load(f)

        # Extract only the "src" values for downloading
        image_urls = [img["src"] for img in image_data if img["src"]]

        if not image_urls:
            print(
                "❌ No images found to download. Make sure the JSON file is correctly filtered."
            )
            return

        full_urls = []
        for img_url in image_urls:
            # Convert relative URLs to absolute if needed
            if img_url.startswith("/"):
                img_url = (
                    "https://www.jetbrains.com" + img_url
                )  # Modify based on source website
            full_urls.append(img_url)

        # Debugging: Print constructed URLs
        print("Constructed Image URLs:", full_urls)

        # Initialize AsyncWebCrawler for downloading
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for url in full_urls:
                # Run the crawler
                try:
                    result = await crawler.arun(
                        url=url, bypass_cache=True, CrawlerRunConfig=run_config
                    )
                except Exception as e:
                    print(f"❌ Error downloading {url}: {e}")
                    continue

                if result and result.downloaded_files:
                    # Save each downloaded file
                    for downloaded_file in result.downloaded_files:
                        filename = os.path.join(
                            IMAGE_DIR, os.path.basename(downloaded_file)
                        )

                        def download_image(filename):
                            with open(filename, "wb") as f:
                                f.write(downloaded_file)
                                print(f"✅ Saved: {filename}")

                        await download_image(filename)

        print(f"✅ All images successfully downloaded to {IMAGE_DIR}")

    if __name__ == "__main__":
        asyncio.run(download_images())

else:
    # Method 2 Here:

    import aiohttp  # Async HTTP client

    async def download_image(session, img_url):
        """Downloads a single image using aiohttp and saves it."""
        try:
            # Convert relative URLs to absolute
            if img_url.startswith("/"):
                img_url = "https://www.jetbrains.com" + img_url

            print(f"⬇ Downloading: {img_url}")

            # Get image filename
            filename = os.path.join(IMAGE_DIR, os.path.basename(img_url))

            async with session.get(img_url) as response:
                if response.status == 200:
                    with open(filename, "wb") as f:
                        f.write(await response.read())
                    print(f"✅ Saved: {filename}")
                else:
                    print(f"❌ Failed to download {img_url}: HTTP {response.status}")

        except Exception as e:
            print(f"❌ Error downloading {img_url}: {e}")

    async def download_images():
        """
        Reads filtered image URLs from JSON and downloads them concurrently using aiohttp.
        """
        # Load manually filtered images
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            image_data = json.load(f)

        # Extract image URLs
        image_urls = [img["src"] for img in image_data if img.get("src")]

        if not image_urls:
            print("❌ No images found to download. Check the JSON file.")
            return

        # Convert relative URLs to absolute
        image_urls = [
            "https://www.jetbrains.com" + url if url.startswith("/") else url
            for url in image_urls
        ]

        # Initialize aiohttp client session
        async with aiohttp.ClientSession() as session:
            # Create concurrent download tasks
            tasks = [download_image(session, img_url) for img_url in image_urls]

            # Execute all downloads concurrently
            await asyncio.gather(*tasks)

        print(f"✅ All images successfully downloaded to {IMAGE_DIR}")

    if __name__ == "__main__":
        asyncio.run(download_images())
