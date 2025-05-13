import asyncio
import os
import re

import requests  # Use requests to fetch raw content
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig

run_config = CrawlerRunConfig(
    exclude_external_links=True,
    remove_overlay_elements=True,
    process_iframes=True,
)


# Function to extract all URLs containing /tree/main/
async def extract_urls(repo_url, crawler):
    print(f"Extracting URLs from: {repo_url}")
    result = await crawler.arun(repo_url, config=run_config)
    if result.success:
        internal_links = result.links.get("internal", [])
        urls = [
            link.get("href")
            for link in internal_links
            if link.get("href", "").startswith(f"{repo_url}/tree/main/")
        ]
        print(f"Found {len(urls)} URLs in the repository.")
        return urls
    else:
        print(f"Failed to extract URLs from {repo_url}: {result.error_message}")
        return []


# Function to sanitize the title for use in a filename
def sanitize_filename(title):
    return re.sub(r'[<>:"/\\|?*]', "_", title)


# Function to fetch raw content from a URL
def fetch_raw_content(session, raw_url):
    try:
        response = session.get(raw_url)
        response.raise_for_status()  # Raise an error for bad responses
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {raw_url}: {e}")
        return None


def clean_code_data(raw_code):
    cleaned_lines = []
    for line in raw_code.splitlines():
        # Remove comments
        if line.strip().startswith("#"):
            continue

        # Remove lines that are just headers or URLs
        if line.strip().startswith("##") or "https://raw.githubusercontent.com" in line:
            continue

        # Strip leading and trailing whitespace
        # line = line.strip()

        # Skip empty lines
        # if not line:
        #     continue

        # Add the cleaned line to the list
        cleaned_lines.append(line)

    # Join the cleaned lines back into a single string
    return "\n".join(cleaned_lines)


# Main crawling function
async def main():
    repo_url = "https://github.com/veryacademy/Django-ORM-Mastery-DJ003"

    async with AsyncWebCrawler(verbose=True) as crawler:
        urls_to_crawl = await extract_urls(repo_url, crawler)
        print("Filtered URLs to crawl:", urls_to_crawl)

        with requests.Session() as session:
            for url in urls_to_crawl:
                print(f"Processing URL: {url}")
                title = url.split("/")[-1]  # Extract the last part of the URL as title
                sanitized_title = sanitize_filename(title)
                combined_markdown = f"# {title}\n\n"

                # Construct raw URLs for views.py and models.py
                raw_views_url = f"https://raw.githubusercontent.com/veryacademy/Django-ORM-Mastery-DJ003/main/{title}/student/views.py"
                raw_models_url = f"https://raw.githubusercontent.com/veryacademy/Django-ORM-Mastery-DJ003/main/{title}/student/models.py"

                # Fetch models.py
                print(f"Fetching models.py from: {raw_models_url}")
                models_content = fetch_raw_content(session, raw_models_url)
                if models_content:
                    cleaned_models = clean_code_data(models_content)
                    combined_markdown += cleaned_models + "\n\n"

                # Fetch views.py
                print(f"Fetching views.py from: {raw_views_url}")
                views_content = fetch_raw_content(session, raw_views_url)
                if views_content:
                    cleaned_views = clean_code_data(views_content)
                    combined_markdown += cleaned_views + "\n\n"

                # Save combined data to a single Rich Text file
                filename = os.path.join("output", f"{sanitized_title}.rtf")
                os.makedirs("output", exist_ok=True)
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(combined_markdown)
                print(f"Saved: {filename}")

                await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
