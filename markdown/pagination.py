import asyncio
import os

from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig
from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.cache_context import CacheMode

"""
Version 1: extract quotes for tag 'love' from 5 pages not checking if quotes exists
Version 2: extract quotes for tag 'love' from page only if quotes exists
Version 2: extract quotes for tag 'love' from pages by determining the first page, 
        and last page number if 'Next' exist in page. If only 'Next' exists in 
        pagination then assumes only 2 pages exists 
"""

VERSION = 1
run_config = CrawlerRunConfig(
    verbose=False,
    cache_mode=CacheMode.DISABLED,
    exclude_external_links=True,
    process_iframes=False,
    wait_for=None,
    magic=True,
    simulate_user=True,
    override_navigator=True,
    delay_before_return_html=2.0,  # Wait 2s before capturing final HTML
    page_timeout=60000,  # Navigation & script timeout (ms)
)

browser_config = BrowserConfig(
    browser_type="chromium",
    java_script_enabled=True,
    headless=True,
    text_mode=True,  # This enables Markdown generation
    verbose=False,
)


async def extract_quotes_markdown(html_content):
    """Extracts and formats quote markdown from HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    quotes = soup.select("div.quote")
    markdown_lines = []

    for quote in quotes:
        text = quote.select_one("span.text").get_text(strip=True)
        author = quote.select_one("small.author").get_text(strip=True)
        about_link = quote.select_one("a[href*='/author/']")["href"]
        full_about_url = f"https://quotes.toscrape.com{about_link}"

        tags = quote.select("div.tags a.tag")
        tag_line = "Tags: " + " ".join(
            f"[{tag.text}](https://quotes.toscrape.com{tag['href']})" for tag in tags
        )

        markdown_lines.append(
            f'{text} by {author} [("about")]({full_about_url})\n{tag_line}\n'
        )

    return "\n".join(markdown_lines)


async def save_markdown(filename, markdown):
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", f"{filename}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Saved: {path}")


match VERSION:
    case 1:

        async def main():
            base_url = "https://quotes.toscrape.com/tag/love/page/{}/"
            combined_markdown = ""

            async with AsyncWebCrawler(config=browser_config) as crawler:
                for page in range(1, 6):
                    url = base_url.format(page)
                    print(f"Scraping: {url}")
                    result = await crawler.arun(url=url, CrawlerRunConfig=run_config)

                    if result.success:
                        quote_markdown = await extract_quotes_markdown(
                            result.html
                        )  # Access raw HTML
                        combined_markdown += (
                            f"\n\n## Page {page}\n\n{quote_markdown.strip()}"
                        )
                    else:
                        print(f"Failed to scrape {url}: {result.error_message}")

            await save_markdown("quotes_love_all_pages", combined_markdown)

        if __name__ == "__main__":
            """
            Version 1: extract quotes for tag 'love' from 5 pages not checking if quotes exists
            """
            asyncio.run(main())

    case 2:

        async def check_for_quotes(html_content):
            """Checks if any quotes are present on the page."""
            soup = BeautifulSoup(html_content, "html.parser")
            return bool(soup.select("div.quote"))

        async def main():
            base_url = "https://quotes.toscrape.com/tag/love/page/{}/"
            combined_markdown = ""
            page_num = 1

            async with AsyncWebCrawler(config=browser_config) as crawler:
                while True:
                    url = base_url.format(page_num)
                    print(f"Scraping: {url}")
                    result = await crawler.arun(url=url, CrawlerRunConfig=run_config)

                    if result.success:
                        if not await check_for_quotes(result.html):
                            print(f"No more quotes found on page {page_num}. Stopping.")
                            break  # Stop if no quotes are found.
                        quote_markdown = await extract_quotes_markdown(
                            result.html
                        )  # Access raw HTML
                        combined_markdown += (
                            f"\n\n## Page {page_num}\n\n{quote_markdown.strip()}"
                        )
                        page_num += 1
                    else:
                        print(f"Failed to scrape {url}: {result.error_message}")
                        break  # Stop on a scraping error

            await save_markdown("quotes_love_all_pages", combined_markdown)

        if __name__ == "__main__":
            """
            Version 2: extract quotes for tag 'love' from page only if quotes exists
            """
            asyncio.run(main())

    case 3:

        async def get_total_pages(crawler, base_url):
            """Gets the total number of pages for the given tag."""
            url = base_url.format(1)  # Check the first page for pagination info
            result = await crawler.arun(url=url, CrawlerRunConfig=run_config)
            if not result.success:
                print(f"Error getting total pages: {result.error_message}")
                return 1  # Default to 1 page if error

            soup = BeautifulSoup(result.html, "html.parser")
            next_page = soup.select_one("li.next a")
            if not next_page:
                return 1  # Only one page exists

            # Find the last page number.  Assumes pagination is of the form .../page/N/
            last_page_link = (
                soup.select("li.next ~ li a")[-1]
                if soup.select("li.next ~ li a")
                else None
            )  # Correctly handle cases with no trailing numbers
            if last_page_link and last_page_link.has_attr("href"):
                href = last_page_link["href"]
                try:
                    return int(href.split("/page/")[1].split("/")[0])
                except (IndexError, ValueError):
                    print(f"Could not parse last page number from href: {href}")
                    return 1
            else:
                # If we can't find the last page number from a direct link, assume multiple pages exist if "next" exists.
                return 2 if next_page else 1

        async def main():
            base_url = "https://quotes.toscrape.com/tag/love/page/{}/"
            combined_markdown = ""

            async with AsyncWebCrawler(config=browser_config) as crawler:
                total_pages = await get_total_pages(crawler, base_url)
                print(f"Total pages found: {total_pages}")

                for page in range(1, total_pages + 1):  # Use the dynamic total_pages
                    url = base_url.format(page)
                    print(f"Scraping: {url}")
                    result = await crawler.arun(url=url, CrawlerRunConfig=run_config)

                    if result.success:
                        quote_markdown = await extract_quotes_markdown(
                            result.html
                        )  # Access raw HTML
                        combined_markdown += (
                            f"\n\n## Page {page}\n\n{quote_markdown.strip()}"
                        )
                    else:
                        print(f"Failed to scrape {url}: {result.error_message}")

            await save_markdown("quotes_love_all_pages", combined_markdown)

        if __name__ == "__main__":
            asyncio.run(main())
