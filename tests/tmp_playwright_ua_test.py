import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36')
        page = await context.new_page()
        await page.goto('about:blank')
        ua = await page.evaluate('() => navigator.userAgent')
        print('EVAL UA:', ua)
        await browser.close()

asyncio.run(main())
