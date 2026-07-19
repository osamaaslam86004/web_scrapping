import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            'test_profile_dir',
            headless=False,
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        )
        
        # Simple init script to override UA
        await browser.add_init_script("""
        const navigatorProto = Object.getPrototypeOf(navigator);
        try {
            Object.defineProperty(navigatorProto, 'userAgent', {
                get: () => 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                configurable: true
            });
        } catch (e) {
            console.log('Failed to override userAgent:', e);
        }
        """)
        
        page = await browser.new_page()
        await page.goto('about:blank')
        ua = await page.evaluate('() => navigator.userAgent')
        print('UA after init_script:', ua)
        await browser.close()

asyncio.run(main())
