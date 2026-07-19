import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig
from camoufox.constants import ARGS
from camoufox.pkgman import get_executable_path

async def main():
    # 1. Fetch the exact path to Camoufox's customized Firefox binary
    # This points to the 490+ MB package you downloaded earlier
    camoufox_executable = get_executable_path()

    print(f"🦊 Wiring Camoufox binary into Crawl4AI...")
    print(f"Path: {camoufox_executable}")

    # 2. Configure Crawl4AI's Browser profile to use Camoufox
    browser_cfg = BrowserConfig(
        browser_type="firefox",             # Camoufox is built on Firefox
        executable_path=camoufox_executable,# Force Crawl4AI to use the stealth engine
        headless=False,                     # Change to True if you want it invisible
        # Inject Camoufox's mandatory internal C++ launch arguments
        extra_args=list(ARGS), 
    )

    # 3. Standard Crawl4AI execution parameters
    run_cfg = CrawlerRunConfig(
        wait_until="domcontentloaded"
    )

    # 4. Execute the crawl pipeline
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(
            url="https://bot.sannysoft.com/",
            config=run_cfg
        )
        
        if result.success:
            print("\n✅ Crawl4AI successfully extracted the page using Camoufox!")
            # Print a snippet of the markdown result to verify
            print(result.markdown[:500])
        else:
            print(f"✗ Extraction failed: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())