"""Test multi-panel scenario where ft palette showed 235 (fy value)."""
import asyncio, os, json, time
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8000"
PROFILE = r"C:\workspace\storage\admin\Moulin_Blanc\Test_Lshape_NORMAL.ds\.playwright_profile"
OUT = r"C:\workspace\fiabilite\_pw_multi"
os.makedirs(OUT, exist_ok=True)
CONSOLE, ERRORS = [], []
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

async def main():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(PROFILE, headless=False, args=["--start-maximized", "--no-sandbox"], no_viewport=True)
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        page.on("console", lambda m: CONSOLE.append(f"[{m.type}] {m.text[:300]}"))
        page.on("pageerror", lambda e: ERRORS.append(str(e)[:300]))
        await page.goto(URL, timeout=30000, wait_until="networkidle")
        await page.wait_for_timeout(3000)
        await page.evaluate("""() => { const links = Array.from(document.querySelectorAll('a, span, td')); for (const el of links) if (el.textContent && el.textContent.trim() === 'Moulin_Blanc') { el.click(); return; } }""")
        await page.wait_for_timeout(3500)
        await page.evaluate("""() => {
            const ctrls = document.querySelectorAll('[ng-controller]');
            for (const ctrl of ctrls) {
                const sc = angular.element(ctrl).scope();
                if (sc && sc.fileNavigator && sc.fileNavigator.fileList && sc.smartClick) {
                    const item = sc.fileNavigator.fileList.find(i => i.model && i.model.name === 'Test_Lshape_NORMAL.ds');
                    if (item) { sc.smartClick(item); sc.$apply(); return; }
                }
            }
        }""")
        await page.wait_for_timeout(15000)
        await page.evaluate("""() => {
            const all = document.querySelectorAll('[ng-controller]');
            for (const el of all) { const s = angular.element(el).scope(); if (s && s.result) { s.$root.selectedResult = s.result; s.$apply(); return; } }
        }""")
        await page.wait_for_timeout(4000)
        log("loaded")

        # Uncheck all
        await page.evaluate("""() => document.querySelectorAll('.dsFieldRow input[type=checkbox]').forEach(cb => { if (cb.checked) cb.click(); })""")
        await page.wait_for_timeout(1000)
        await page.evaluate("""() => document.querySelectorAll('.dsInputSubFieldRow input[type=checkbox]').forEach(cb => { if (cb.checked) cb.click(); })""")
        await page.wait_for_timeout(800)
        await page.evaluate("""() => { const exp = document.querySelector('.dsInputDataExpander'); if (exp && exp.textContent.trim().startsWith('▶')) exp.click(); }""")
        await page.wait_for_timeout(700)

        # Click fy rebar FIRST
        log("Click fy rebar")
        await page.evaluate("""() => {
            const rows = Array.from(document.querySelectorAll('.dsInputSubFieldRow'));
            const r = rows.find(r => r.textContent.toLowerCase().includes('fy rebar'));
            if (r) { const cb = r.querySelector('input[type=checkbox]'); if (cb && !cb.checked) cb.click(); }
        }""")
        await page.wait_for_timeout(6000)

        # Then click ft vol
        log("Click ft vol (with fy rebar still on)")
        ERR_BEFORE = len(ERRORS)
        await page.evaluate("""() => {
            const rows = Array.from(document.querySelectorAll('.dsInputSubFieldRow'));
            const r = rows.find(r => r.textContent.toLowerCase().includes('ft vol'));
            if (r) { const cb = r.querySelector('input[type=checkbox]'); if (cb && !cb.checked) cb.click(); }
        }""")
        await page.wait_for_timeout(7000)
        await page.screenshot(path=os.path.join(OUT, "01_both_on.png"))

        # State after both active
        st = await page.evaluate("""() => {
            const rm = myBABYLON.ResultMesh;
            return {
                cls: rm.constructor.name,
                field: rm.currentField ? rm.currentField.name : null,
                min: rm.currentField ? rm.currentField.min : null,
                max: rm.currentField ? rm.currentField.max : null,
                fieldsKeys: rm.fields ? Object.keys(rm.fields) : [],
                allPanelEls: Array.from(document.querySelectorAll('[class*="anel"]')).map(p => p.className).slice(0,15),
                sceneMeshes: scene.meshes.filter(m=>m.isVisible).map(m=>m.name).slice(0,15),
            };
        }""")
        log(f"   {st}")
        log(f"   errs_in_this_step: {len(ERRORS)-ERR_BEFORE}")
        for e in ERRORS[ERR_BEFORE:]: log(f"   ERR: {e[:200]}")

        with open(os.path.join(OUT, "console.txt"), "w", encoding="utf-8") as f: f.write("\n".join(CONSOLE[-150:]))
        with open(os.path.join(OUT, "errors.txt"), "w", encoding="utf-8") as f: f.write("\n".join(ERRORS))
        await ctx.close()

if __name__ == "__main__":
    asyncio.run(main())
