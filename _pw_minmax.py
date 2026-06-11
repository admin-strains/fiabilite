"""Detect why MIN is missing on fy rebar panel when ft vol is also active."""
import asyncio, os, json, time
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8000"
PROFILE = r"C:\workspace\storage\admin\Moulin_Blanc\Test_Lshape_NORMAL.ds\.playwright_profile"
OUT = r"C:\workspace\fiabilite\_pw_minmax"
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

        await page.evaluate("""() => document.querySelectorAll('.dsFieldRow input[type=checkbox]').forEach(cb => { if (cb.checked) cb.click(); })""")
        await page.wait_for_timeout(1000)
        await page.evaluate("""() => { const exp = document.querySelector('.dsInputDataExpander'); if (exp && exp.textContent.trim().startsWith('▶')) exp.click(); }""")
        await page.wait_for_timeout(700)
        await page.evaluate("""() => document.querySelectorAll('.dsInputSubFieldRow input[type=checkbox]').forEach(cb => { if (cb.checked) cb.click(); })""")
        await page.wait_for_timeout(800)

        # Activate ft vol first
        log("ft vol")
        await page.evaluate("""() => { const r = Array.from(document.querySelectorAll('.dsInputSubFieldRow')).find(r => r.textContent.toLowerCase().includes('ft vol')); if (r) r.querySelector('input[type=checkbox]').click(); }""")
        await page.wait_for_timeout(6000)

        # Then fy rebar
        log("fy rebar")
        await page.evaluate("""() => { const r = Array.from(document.querySelectorAll('.dsInputSubFieldRow')).find(r => r.textContent.toLowerCase().includes('fy rebar')); if (r) r.querySelector('input[type=checkbox]').click(); }""")
        await page.wait_for_timeout(7000)
        await page.screenshot(path=os.path.join(OUT, "01_both.png"))

        # Inspect both panels' MAX/MIN visibility
        result = await page.evaluate("""() => {
            const cards = Array.from(document.querySelectorAll('.result-panel-card'));
            const out = [];
            for (const c of cards) {
                const title = c.querySelector('.result-panel-title, .dsOptionsPanelTitle, h4, .panel-title');
                const titleTxt = title ? title.textContent.trim() : (c.textContent || '').substring(0, 50);
                // find MAX label and MIN label
                const maxEls = Array.from(c.querySelectorAll('*')).filter(el => el.children.length === 0 && (el.textContent || '').trim() === 'MAX');
                const minEls = Array.from(c.querySelectorAll('*')).filter(el => el.children.length === 0 && (el.textContent || '').trim() === 'MIN');
                const maxVisible = maxEls.length > 0 && maxEls.some(el => {
                    const rect = el.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0;
                });
                const minVisible = minEls.length > 0 && minEls.some(el => {
                    const rect = el.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0;
                });
                const inputs = Array.from(c.querySelectorAll('input[type=text], input.form-control'));
                out.push({
                    title: titleTxt.substring(0, 60),
                    maxLabel: { count: maxEls.length, visible: maxVisible },
                    minLabel: { count: minEls.length, visible: minVisible },
                    inputCount: inputs.length,
                    inputValues: inputs.map(i => ({val: i.value, hidden: !i.offsetParent})).slice(0, 4),
                });
            }
            return out;
        }""")
        log(f"panels: {json.dumps(result, indent=2)}")

        # Look at the scope of the rebar panel for its options
        rebar_scope = await page.evaluate("""() => {
            const cards = Array.from(document.querySelectorAll('.result-panel-card'));
            const rebar = cards.find(c => (c.textContent || '').toLowerCase().includes('fy rebar'));
            if (!rebar) return 'no rebar card';
            const sc = angular.element(rebar).scope();
            if (!sc) return 'no scope';
            return {
                mode: sc.mode,
                hasOptions: !!sc.options,
                colorScaleOptionsKeys: sc.options && sc.options.colorScaleOptions ? Object.keys(sc.options.colorScaleOptions) : null,
                colorScaleForMode: sc.options && sc.options.colorScaleOptions && sc.mode ? sc.options.colorScaleOptions[sc.mode] : null,
                filterOptions: sc.options && sc.options.filter_options ? sc.options.filter_options : null,
                auto: sc.options && sc.options.colorScaleOptions && sc.mode ? sc.options.colorScaleOptions[sc.mode].auto : null,
            };
        }""")
        log(f"rebar scope: {json.dumps(rebar_scope, indent=2, default=str)}")

        await ctx.close()

if __name__ == "__main__":
    asyncio.run(main())
