"""Probe colorScaleOptions.auto for both panels (why ft has visible boxes, fy hidden)."""
import asyncio, os, json, time
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8000"
PROFILE = r"C:\workspace\storage\admin\Moulin_Blanc\Test_Lshape_NORMAL.ds\.playwright_profile"
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
        # uncheck all
        await page.evaluate("""() => document.querySelectorAll('.dsFieldRow input[type=checkbox]').forEach(cb => { if (cb.checked) cb.click(); })""")
        await page.wait_for_timeout(1000)
        await page.evaluate("""() => { const exp = document.querySelector('.dsInputDataExpander'); if (exp && exp.textContent.trim().startsWith('▶')) exp.click(); }""")
        await page.wait_for_timeout(700)
        await page.evaluate("""() => document.querySelectorAll('.dsInputSubFieldRow input[type=checkbox]').forEach(cb => { if (cb.checked) cb.click(); })""")
        await page.wait_for_timeout(800)
        # ft then fy
        await page.evaluate("""() => { const r = Array.from(document.querySelectorAll('.dsInputSubFieldRow')).find(r => r.textContent.toLowerCase().includes('ft vol')); if (r) r.querySelector('input[type=checkbox]').click(); }""")
        await page.wait_for_timeout(6000)
        await page.evaluate("""() => { const r = Array.from(document.querySelectorAll('.dsInputSubFieldRow')).find(r => r.textContent.toLowerCase().includes('fy rebar')); if (r) r.querySelector('input[type=checkbox]').click(); }""")
        await page.wait_for_timeout(7000)

        # Probe both panels
        out = await page.evaluate("""() => {
            const cards = Array.from(document.querySelectorAll('.result-panel-card'));
            return cards.map(c => {
                const sc = angular.element(c).scope();
                if (!sc) return { err: 'no scope' };
                const title = (c.textContent || '').match(/Input data[^\\n]*/) || ['?'];
                return {
                    title: title[0].substring(0, 80),
                    mode: sc.mode,
                    panelCtxOptionsPath: sc.panelCtx ? sc.panelCtx.optionsPath : null,
                    csoKeys: sc.options && sc.options.colorScaleOptions ? Object.keys(sc.options.colorScaleOptions) : null,
                    autoForMode: sc.options && sc.options.colorScaleOptions && sc.mode ? (sc.options.colorScaleOptions[sc.mode] || {}).auto : null,
                    sharedColorScaleOptions: sc.colorScaleOptions === (sc.options.colorScaleOptions[sc.mode] || null),
                    colorScaleOptionsAuto: sc.colorScaleOptions ? sc.colorScaleOptions.auto : 'missing',
                    colorScaleOptionsObj: sc.colorScaleOptions || null,
                };
            });
        }""")
        log(json.dumps(out, indent=2, default=str))
        await ctx.close()

if __name__ == "__main__":
    asyncio.run(main())
