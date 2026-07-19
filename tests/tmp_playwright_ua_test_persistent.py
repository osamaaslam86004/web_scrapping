import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        profile_dir = 'playwright_user_data_test'
        browser = await p.chromium.launch_persistent_context(profile_dir, headless=False, user_agent=user_agent)
        page = await browser.new_page()
        await page.goto('about:blank')
        ua = await page.evaluate('() => navigator.userAgent')
        print('PERSISTENT EVAL UA:', ua)
        await browser.close()

asyncio.run(main())
