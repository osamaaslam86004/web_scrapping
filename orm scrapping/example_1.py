import asyncio
import os
import re

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode
from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


# Function to clean the fetched content
async def clean_article_content(raw_content):
    cleaned_lines = []
    for line in raw_content.splitlines():
        # Remove unwanted lines
        if "https://" in line:
            continue

        # Strip leading and trailing whitespace
        # line = line.strip()

        # Add the cleaned line to the list
        cleaned_lines.append(line)

    # Join the cleaned lines back into a single string
    return "\n".join(cleaned_lines)


# Function to sanitize the title for use in a filename
def sanitize_filename(title):
    return re.sub(r'[<>:"/\\|?*]', "_", title)


# Function to save content
async def saving_file(title, markdown, markdown_type):
    title = sanitize_filename(title)

    filename = os.path.join("output", f"{title}_{markdown_type}.rtf")
    os.makedirs("output", exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Saved: {filename}")


async def crawl_sequential():

    title = None
    markdown = None

    print("\n=== Sequential Crawling with Session Reuse ===")

    prune_filter = PruningContentFilter(
        threshold=0.5, threshold_type="dynamic", min_word_threshold=50  # or "dynamic"
    )

    # Example: ignore all links, don't escape HTML, and wrap text at 80 characters
    md_generator = DefaultMarkdownGenerator(
        options={
            "ignore_links": True,
            "escape_html": True,
            "body_width": 80,
            "ignore_images": True,
            "skip_internal_links": True,
        },
        content_filter=prune_filter,
    )

    crawl_config = CrawlerRunConfig(
        verbose=True,
        cache_mode=CacheMode.ENABLED,
        exclude_external_links=True,
        remove_overlay_elements=True,
        process_iframes=False,  # default = True
        markdown_generator=md_generator,
        extraction_strategy=None,
        word_count_threshold=200,
        js_code="js:() => window.loaded === true",
        wait_for="css:.main-loaded",
        screenshot=False,
        pdf=False,
        excluded_tags=["nav", "footer"],
        prettiify=True,  # beautifies final HTML (slower, purely cosmetic)
        remove_forms=True,  # If True, remove all <form> elements
        exclude_social_media_links=True,
    )

    base_browser_config = BrowserConfig(
        headless=True,
        # For better performance in Docker or low-memory environments:
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )

    # Create the crawler (opens the browser)
    crawler = AsyncWebCrawler(config=base_browser_config)
    await crawler.start()

    url = "https://medium.com/@farad.dev/mastering-djangos-orm-advanced-queries-and-performance-tips-e8613978176d"

    try:
        result = await crawler.arun(url=url, config=crawl_config)
        # Process the results
        if result.success:
            print(f"Successfully crawled: {result.url}")

            title = result.metadata.get("title", "N/A")
            print(f"Title: {result.metadata.get('title', 'N/A')}")

            markdown = result.markdown

            # Save the cleaned content to a Rich Text file
            if markdown is not None:
                print(f"Word count: {len(result.markdown.split())}")
                # cleaning content
                markdown = await clean_article_content(markdown)
                await saving_file(title, markdown, "markdown")

        else:
            print(f"Failed to crawl: {result.url}")
            print(f"Error: {result.error_message}")

    finally:
        # After all URLs are done, close the crawler (and the browser)
        await crawler.close()


async def main():

    await crawl_sequential()


if __name__ == "__main__":
    asyncio.run(main())
