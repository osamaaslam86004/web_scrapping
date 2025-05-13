import asyncio
import os
import re

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode
from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

prune_filter = PruningContentFilter(
    threshold=0.5, threshold_type="dynamic", min_word_threshold=50  # or "dynamic"
)


# Example: ignore all links, don't escape HTML, and wrap text at 80 characters
md_generator = DefaultMarkdownGenerator(
    options={
        "ignore_links": True,
        "escape_html": False,
        "body_width": 80,
        "ignore_images": True,
        "skip_internal_links": True,
    },
    content_filter=prune_filter,
)


# List of URLs to crawl
url = "https://medium.com/@farad.dev/mastering-djangos-orm-advanced-queries-and-performance-tips-e8613978176d"


run_config = CrawlerRunConfig(
    css_selector="div.fj.fk.fl.fm.fn",
    verbose=True,
    cache_mode=CacheMode.ENABLED,
    exclude_external_links=True,
    remove_overlay_elements=True,
    process_iframes=False,  # default = True
    markdown_generator=md_generator,
    extraction_strategy=None,
    # word_count_threshold=200,
    js_code=None,
    # wait_for="css:.main-loaded"
    wait_for="css:div.fj.fk.fl.fm.fn"
    or "js:() => window.loaded === true",  # Wait for fully loaded page
    screenshot=False,
    pdf=False,
    excluded_tags=["nav", "footer"],
    prettiify=True,  # beautifies final HTML (slower, purely cosmetic)
    remove_forms=True,  # If True, remove all <form> elements
    exclude_social_media_links=True,
)

browser_config = BrowserConfig(
    browser_type="chromium",
    ignore_https_errors=True,  # continues despite invalid certificates (common in dev/staging)
    java_script_enabled=True,  # Disable if you want no JS overhead, or if only static content is needed
    headless=True,
    text_mode=True,
    proxy_config=None,
    viewport_width=1280,
    viewport_height=720,
    verbose=True,
    use_persistent_context=False,
    user_data_dir=None,
    cookies=[
        {
            "name": "sessionid",
            "value": "your_session_id",  # Replace with your actual session ID
            "domain": ".medium.com",  # Domain should be a valid subdomain or main domain
            "path": "/",  # Path is required for valid cookies
        }
    ],
    headers=None,
    light_mode=False,
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.121 Safari/537.36",  # Example user-agent
    # For better performance in Docker or low-memory environments:
    extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
)


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


async def saving_file(title, markdown, result_type):
    title = sanitize_filename(title)

    filename = os.path.join("output", f"{title}_{result_type}.rtf")

    os.makedirs("output", exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Saved: {filename}")


# Main function to scrape the Medium article
async def main():

    title = None
    markdown = None
    fit_markdown = None

    # async with AsyncWebCrawler(verbose=True) as crawler:
    async with AsyncWebCrawler(config=browser_config) as crawler:

        # Set up crawling parameters
        result = await crawler.arun(
            url=url, bypass_cache=True, CrawlerRunConfig=run_config
        )

        # Process the results
        if result.success:
            print(f"Successfully crawled: {result.url}")
            print(f"Status code: {result.status_code}")

            title = result.metadata.get("title", "N/A")
            print(f"Title: {result.metadata.get('title', 'N/A')}")

            markdown = result.markdown
            fit_markdown = result.fit_markdown

            print(f"Word count: {len(result.markdown.split())}")

            print(
                f"Number of links: {len(result.links.get('internal', [])) + len(result.links.get('external', []))}"
            )
            print(f"Number of images: {len(result.media.get('images', []))}")

            # raw content
            markdown = await clean_article_content(markdown)
            # Save the cleaned content to a Rich Text file
            await saving_file(title, markdown, "markdown")
            # print(f"Raw HTML without Filtering / Prunging: {result.html}")

            # fit markdown after filtering, prunging
            fit_markdown = await clean_article_content(fit_markdown)
            # Save the cleaned content to a Rich Text file
            await saving_file(title, fit_markdown, "fit_markdown")
            # print("Fit HTML after Filtering and Prunging length:", len(result.cleaned_html))

        else:
            print(f"Failed to crawl: {result.url}")
            print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
