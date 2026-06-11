"""Test 3-panel scenario: fy rebar + ft vol + Displacement together."""
import asyncio, os, json, time
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8000"
PROFILE = r"C:\workspace\storage\admin\Moulin_Blanc\Test_Lshape_NORMAL.ds\.playwright_profile"
OUT = r"C:\workspace\fiabilite\_pw_3panels"
os.makedirs(OUT, exist_ok=True)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

async def main():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(PROFILE, headless=False, args=["--start-maximized", "--no-sandbox"], no_viewport=True)
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto(URL, timeout=30000, wait_until="networkidle")
        await page.wait_for_timeout(3000)
        await page.evaluate("""() => { const links = Array.from(document.querySelectorAll('a, span, td')); for (const el of links) if (el.textContent && el.textContent.trim() === 'Moulin_Blanc') { el.click(); return; } }""")
        await page.wait_for_timeout(3500)
        await page.evaluate("""() => {
            const ctrls = document.querySelectorAll('[ng-controller]');
            for (const ctrl of ctrls) { const sc = angular.element(ctrl).scope(); if (sc && sc.fileNavigator && sc.smartClick) { const item = sc.fileNavigator.fileList.find(i => i.model && i.model.name === 'Test_Lshape_NORMAL.ds'); if (item) { sc.smartClick(item); sc.$apply(); return; } } }
        }""")
        await page.wait_for_timeout(15000)
        await page.evaluate("""() => { const all = document.querySelectorAll('[ng-controller]'); for (const el of all) { const s = angular.element(el).scope(); if (s && s.result) { s.$root.selectedResult = s.result; s.$apply(); return; } } }""")
        await page.wait_for_timeout(4000)

        await page.evaluate("""() => document.querySelectorAll('.dsFieldRow input[type=checkbox]').forEach(cb => { if (cb.checked) cb.click(); })""")
        await page.wait_for_timeout(1000)
        await page.evaluate("""() => { const exp = document.querySelector('.dsInputDataExpander'); if (exp && exp.textContent.trim().startsWith('▶')) exp.click(); }""")
        await page.wait_for_timeout(700)
        await page.evaluate("""() => document.querySelectorAll('.dsInputSubFieldRow input[type=checkbox]').forEach(cb => { if (cb.checked) cb.click(); })""")
        await page.wait_for_timeout(800)

        # Activate Displacement (NORM)
        log(">>> Click Displacement")
        await page.evaluate("""() => { const r = Array.from(document.querySelectorAll('.dsFieldRow')).find(r => { const l = r.querySelector('.dsFieldLabel span'); return l && l.textContent.trim() === 'Displacement'; }); if (r) r.querySelector('input[type=checkbox]').click(); }""")
        await page.wait_for_timeout(6000)
        # Activate ft vol
        log(">>> Click ft vol")
        await page.evaluate("""() => { const r = Array.from(document.querySelectorAll('.dsInputSubFieldRow')).find(r => r.textContent.toLowerCase().includes('ft vol')); if (r) r.querySelector('input[type=checkbox]').click(); }""")
        await page.wait_for_timeout(6000)
        # Activate fy rebar
        log(">>> Click fy rebar")
        await page.evaluate("""() => { const r = Array.from(document.querySelectorAll('.dsInputSubFieldRow')).find(r => r.textContent.toLowerCase().includes('fy rebar')); if (r) r.querySelector('input[type=checkbox]').click(); }""")
        await page.wait_for_timeout(7000)
        await page.screenshot(path=os.path.join(OUT, "01_3panels.png"))

        # Inspect each panel
        result = await page.evaluate("""() => {
            const cards = Array.from(document.querySelectorAll('.result-panel-card'));
            return cards.map(c => {
                const sc = angular.element(c).scope();
                const titleM = (c.textContent || '').match(/(Input data[^\\n]*|Displacement[^\\n]*)/);
                const title = titleM ? titleM[0].substring(0, 50) : '?';
                const boundBoxes = Array.from(c.querySelectorAll('.dsBoundBox'));
                return {
                    title,
                    scopeMode: sc ? sc.mode : null,
                    scopeAuto: sc && sc.colorScaleOptions ? sc.colorScaleOptions.auto : 'NO_CSO',
                    scopeMax: sc && sc.colorScaleOptions ? sc.colorScaleOptions.max : null,
                    scopeMin: sc && sc.colorScaleOptions ? sc.colorScaleOptions.min : null,
                    bbCount: boundBoxes.length,
                    bbVisible: boundBoxes.map(b => {
                        const cs = getComputedStyle(b);
                        return { display: cs.display, ngHide: b.classList.contains('ng-hide'), op: cs.opacity };
                    }),
                };
            });
        }""")
        log(json.dumps(result, indent=2, default=str))
        await ctx.close()

if __name__ == "__main__":
    asyncio.run(main())
