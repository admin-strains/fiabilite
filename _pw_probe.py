"""Probe the DOM to find how to click .ds icons."""
import asyncio, os
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8000"
PROFILE = r"C:\workspace\storage\admin\Moulin_Blanc\Test_Lshape_NORMAL.ds\.playwright_profile"
OUT = r"C:\workspace\fiabilite\_probe_out"
os.makedirs(OUT, exist_ok=True)

async def main():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            PROFILE, headless=False, args=["--start-maximized", "--no-sandbox"], no_viewport=True,
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto(URL, timeout=30000, wait_until="networkidle")
        await page.wait_for_timeout(3000)

        # Click Moulin_Blanc
        await page.evaluate("""() => {
            const links = Array.from(document.querySelectorAll('a, span, td'));
            for (const el of links) if (el.textContent && el.textContent.trim() === 'Moulin_Blanc') { el.click(); return; }
        }""")
        await page.wait_for_timeout(2500)
        await page.screenshot(path=os.path.join(OUT, "moulin_blanc.png"))

        # Probe : find all elements containing "Test_Lshape" anywhere
        info = await page.evaluate("""() => {
            const all = document.querySelectorAll('*');
            const matches = [];
            for (const el of all) {
                if (el.children.length === 0) {
                    const txt = (el.textContent || '').trim();
                    if (txt.includes('Test_Lshape')) {
                        matches.push({ tag: el.tagName, text: txt.substring(0, 50), title: el.title || '', class: el.className.toString().substring(0, 80) });
                    }
                }
            }
            return matches.slice(0, 20);
        }""")
        print("Matches:", info)

        # Try clicking on the parent of a span containing Test_Lshape_NORMAL
        clicked = await page.evaluate("""() => {
            const spans = document.querySelectorAll('span, div, a, td');
            for (const s of spans) {
                if ((s.textContent || '').trim() === 'Test_Lshape_NORMAL.ds' || (s.title || '') === 'Test_Lshape_NORMAL.ds') {
                    // Find clickable ancestor (probably parent or grandparent with ng-dblclick)
                    let n = s;
                    while (n && n.parentElement) {
                        if (n.hasAttribute && (n.hasAttribute('ng-dblclick') || n.hasAttribute('ng-click'))) {
                            const evt = new MouseEvent('dblclick', {bubbles:true});
                            n.dispatchEvent(evt);
                            return 'clicked via ng-dblclick on '+n.tagName;
                        }
                        n = n.parentElement;
                    }
                    // Try dblclick on the span itself
                    const evt2 = new MouseEvent('dblclick', {bubbles:true});
                    s.dispatchEvent(evt2);
                    return 'fallback dblclick on '+s.tagName;
                }
            }
            return 'not found';
        }""")
        print("Click result:", clicked)
        await page.wait_for_timeout(8000)
        await page.screenshot(path=os.path.join(OUT, "after_click.png"))

        # Check if scope loaded
        st = await page.evaluate("""() => {
            try {
                const all = document.querySelectorAll('[ng-controller]');
                for (const el of all) {
                    const s = angular.element(el).scope();
                    if (s && s.result) return { ok: true, name: s.result.name };
                }
            } catch(e){}
            return { ok: false };
        }""")
        print("Scope:", st)

        await ctx.close()

if __name__ == "__main__":
    asyncio.run(main())
