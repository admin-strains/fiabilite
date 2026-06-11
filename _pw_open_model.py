"""Open Test_Lshape_NORMAL.ds via scope.smartClick(item) and validate."""
import asyncio, os, json, time
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8000"
PROFILE = r"C:\workspace\storage\admin\Moulin_Blanc\Test_Lshape_NORMAL.ds\.playwright_profile"
OUT = r"C:\workspace\fiabilite\_pw_open"
os.makedirs(OUT, exist_ok=True)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

async def main():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            PROFILE, headless=False, args=["--start-maximized", "--no-sandbox"], no_viewport=True,
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        page.on("console", lambda m: print(f"[con] {m.type}: {m.text[:200]}", flush=True))
        page.on("pageerror", lambda e: print(f"[ERR] {e}", flush=True))
        await page.goto(URL, timeout=30000, wait_until="networkidle")
        await page.wait_for_timeout(3000)

        # Click Moulin_Blanc folder
        log(">>> click Moulin_Blanc")
        await page.evaluate("""() => {
            const links = Array.from(document.querySelectorAll('a, span, td'));
            for (const el of links) if (el.textContent && el.textContent.trim() === 'Moulin_Blanc') { el.click(); return; }
        }""")
        await page.wait_for_timeout(3500)
        await page.screenshot(path=os.path.join(OUT, "01_moulin.png"))

        # Probe : find scope with fileNavigator and call smartClick on Test_Lshape
        log(">>> find item via scope")
        result = await page.evaluate("""() => {
            // Find any ng-controller scope that has fileNavigator
            const ctrls = document.querySelectorAll('[ng-controller]');
            for (const ctrl of ctrls) {
                const sc = angular.element(ctrl).scope();
                if (sc && sc.fileNavigator && sc.fileNavigator.fileList && sc.smartClick) {
                    const list = sc.fileNavigator.fileList;
                    const item = list.find(i => i.model && i.model.name === 'Test_Lshape_NORMAL.ds');
                    if (item) {
                        sc.smartClick(item);
                        sc.$apply();
                        return { ok: true, found: list.length, name: item.model.name };
                    }
                    return { ok: false, listLen: list.length, names: list.slice(0, 5).map(i => i.model && i.model.name) };
                }
            }
            return { ok: false, msg: 'no scope with fileNavigator' };
        }""")
        log(f"    {result}")
        await page.wait_for_timeout(15000)
        await page.screenshot(path=os.path.join(OUT, "02_after_smartclick.png"))

        # Check ResultMesh
        st = await page.evaluate("""() => {
            try {
                const ok = typeof myBABYLON !== 'undefined' && !!myBABYLON.ResultMesh;
                return { babylon: ok, rmClass: ok ? myBABYLON.ResultMesh.constructor.name : null };
            } catch(e){ return { err: e.message }; }
        }""")
        log(f"    BABYLON state: {st}")
        await ctx.close()

if __name__ == "__main__":
    asyncio.run(main())
