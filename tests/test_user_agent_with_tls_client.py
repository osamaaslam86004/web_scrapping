import asyncio
from camoufox.async_api import AsyncCamoufox

async def test_sannysoft_with_camoufox():
    print("🚀 Initializing Camoufox Stealth Test...")
    print("--------------------------------------------------")

    async with AsyncCamoufox(headless=False, os="windows") as browser:
        page = await browser.new_page()
        
        print("🕵️‍♂️ Navigating to bot.sannysoft.com...")
        await page.goto("https://bot.sannysoft.com/", wait_until="domcontentloaded")
        await page.wait_for_timeout(5000)
        
        print("\n📊 Extracting Results from Browser DOM:")
        print("--------------------------------------------------")
        
        # 1. Grab raw HTML rows to look at the actual visual classes (passed/failed) assigned by sannysoft
        rows = await page.locator("table tr").all()
        
        # 2. Expanded target checklist to match all major indicators
        wanted = ["User Agent", "WebDriver (New)", "WebDriver Advanced", "Chrome (New)", "Browser (New)"]
        
        for row in rows:
            text = await row.inner_text()
            text_cleaned = text.replace('\n', ' ').strip()
            
            for key in wanted:
                if text_cleaned.startswith(key):
                    # Check the class attribute or the raw inner HTML for failure indicators
                    raw_html = await row.inner_html()
                    
                    if "failed" in raw_html.lower() or "fail" in text_cleaned.lower():
                        status = "✗ FAILED"
                    elif "warn" in raw_html.lower():
                        status = "⚠ WARNING"
                    else:
                        status = "✓ PASSED"
                        
                    print(f"{status}: {text_cleaned}")
                    break

if __name__ == "__main__":
    asyncio.run(test_sannysoft_with_camoufox())