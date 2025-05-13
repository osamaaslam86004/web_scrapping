# import asyncio
# import os
# import re

# from crawl4ai import AsyncWebCrawler
# from crawl4ai.async_configs import CrawlerRunConfig
# from crawl4ai.extraction_strategy import CosineStrategy

# cosine_strategy = CosineStrategy(
#     # Content Filtering
#     # semantic_filter="Django ORM",  # Topic/keyword filter
#     word_count_threshold=10,  # Minimum words per cluster
#     sim_threshold=0.3,  # Similarity threshold
#     # Clustering Parameters
#     max_dist=0.2,  # Maximum cluster distance
#     linkage_method="ward",  # Clustering method
#     top_k=3,  # Top clusters to return
#     # Model Configuration
#     model_name="sentence-transformers/all-MiniLM-L6-v2",  # Embedding model
#     verbose=False,  # Enable verbose logging
# )


# run_config = CrawlerRunConfig(
#     exclude_external_links=True,
#     remove_overlay_elements=True,
#     process_iframes=True,
#     extraction_strategy=cosine_strategy,
# )


# # Function to clean the fetched content
# async def clean_article_content(raw_content):
#     cleaned_lines = []
#     for line in raw_content.splitlines():
#         # Remove unwanted lines
#         if "https://" in line:
#             continue

#         # Add the cleaned line to the list
#         cleaned_lines.append(line)

#     # Join the cleaned lines back into a single string
#     return "\n".join(cleaned_lines)


# # Function to sanitize the title for use in a filename
# def sanitize_filename(title):
#     return re.sub(r'[<>:"/\\|?*]', "_", title)


# async def saving_file(title, markdown):
#     title = sanitize_filename(title)

#     filename = os.path.join("output", f"{title}.rtf")
#     os.makedirs("output", exist_ok=True)

#     print(f"Saving to: {filename}")  # Debugging line

#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(markdown)
#     print(f"Saved: {filename}")


# # Main function to scrape the Medium article
# async def main():
#     title = None
#     cleaned_markdown = None

#     async with AsyncWebCrawler(verbose=True) as crawler:
#         # List of URLs to crawl
#         url = "https://medium.com/@farad.dev/mastering-djangos-orm-advanced-queries-and-performance-tips-e8613978176d"

#         # Set up crawling parameters
#         result = await crawler.arun(
#             url=url, cache_mode="no-cache", verbose=True, config=run_config
#         )

#         # Process the results
#         if result.success:
#             print(f"Successfully crawled: {result.url}")

#             title = result.metadata.get("title", "N/A")
#             print(f"Title: {result.metadata.get('title', 'N/A')}")

#             cleaned_html = result.cleaned_html
#             print("Length of Cleaned Html :", len(cleaned_html))

#             print("Extracted Cleaned HTML :", cleaned_html)

#             # cleaning content
#             cleaned_html = await clean_article_content(cleaned_html)
#             # Save the cleaned content to a Rich Text file
#             await saving_file(title, cleaned_html)

#         else:
#             print(f"Failed to crawl: {result.url}")
#             print(f"Error: {result.error_message}")
#             cleaned_html = "No content extracted"


# if __name__ == "__main__":
#     asyncio.run(main())


import asyncio
import os
import re

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.extraction_strategy import CosineStrategy

# Define extraction strategy
cosine_strategy = CosineStrategy(
    semantic_filter="mastering djangos orm advanced-queries and performance tips",  # Topic/keyword filter
    word_count_threshold=10,  # Minimum words per cluster
    sim_threshold=0.3,  # Similarity threshold
    max_dist=0.2,  # Maximum cluster distance
    linkage_method="ward",  # Clustering method
    top_k=3,  # Top clusters to return
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Embedding model
    verbose=True,  # Enable verbose logging
)

# fallback
fallback_strategy = CosineStrategy(
    word_count_threshold=10,  # Minimum words per cluster
    sim_threshold=0.3,  # Similarity threshold
    max_dist=0.2,  # Maximum cluster distance
    linkage_method="ward",  # Clustering method
    top_k=3,  # Top clusters to return
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Embedding model
    verbose=True,  # Enable verbose logging
)


# Define run config
run_config = CrawlerRunConfig(
    exclude_external_links=True,
    remove_overlay_elements=True,
    process_iframes=True,
    extraction_strategy=cosine_strategy,  # Start with CosineStrategy
)

# Define run config
fallback_config = CrawlerRunConfig(
    exclude_external_links=True,
    remove_overlay_elements=True,
    process_iframes=True,
    extraction_strategy=fallback_strategy,  # Start with CosineStrategy
)


# Function to sanitize title
def sanitize_filename(title):
    return re.sub(r'[<>:"/\\|?*]', "_", title)


# Function to save content
async def saving_file(title, markdown):
    title = sanitize_filename(title)
    filename = os.path.join("output", f"{title}.rtf")
    os.makedirs("output", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Saved: {filename}")


# Main function
async def main():
    title = None
    url = "https://medium.com/@farad.dev/mastering-djangos-orm-advanced-queries-and-performance-tips-e8613978176d"

    async with AsyncWebCrawler(verbose=True) as crawler:
        # First attempt with CosineStrategy
        result = await crawler.arun(url=url, config=run_config)

        if result.success and result.markdown is None:
            print("Successfully extracted content with CosineStrategy.")

            title = result.metadata.get("title", "N/A")
            print(f"Title: {result.metadata.get('title', 'N/A')}")

            print(f"Markdown: {result.markdown}")

            await saving_file(title, result.markdown)

        else:
            result = await crawler.arun(url=url, config=fallback_config)

            if result.success and result.cleaned_html:
                print("Successfully extracted content with CosineStrategy.")

                title = result.metadata.get("title", "N/A")
                print(f"Title: {result.metadata.get('title', 'N/A')}")

                await saving_file(title, result.cleaned_html)

                print(f"Markdown: {result.cleaned_html}")
            else:
                print("Failed to extract content with Fallback Strategy.")
                print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
