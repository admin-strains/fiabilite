"""Full validation: fc/ft vol, fy rebar, multi-panel Min/Max edit, filtering."""
import asyncio, os, json, time
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8000"
PROFILE = r"C:\workspace\storage\admin\Moulin_Blanc\Test_Lshape_NORMAL.ds\.playwright_profile"
OUT = r"C:\workspace\fiabilite\_pw_full2"
os.makedirs(OUT, exist_ok=True)

CONSOLE, ERRORS = [], []
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

async def open_model(page):
    log(">>> Goto + Moulin_Blanc + smartClick Test_Lshape")
    await page.goto(URL, timeout=30000, wait_until="networkidle")
    await page.wait_for_timeout(3000)
    await page.evaluate("""() => {
        const links = Array.from(document.querySelectorAll('a, span, td'));
        for (const el of links) if (el.textContent && el.textContent.trim() === 'Moulin_Blanc') { el.click(); return; }
    }""")
    await page.wait_for_timeout(3500)
    r = await page.evaluate("""() => {
        const ctrls = document.querySelectorAll('[ng-controller]');
        for (const ctrl of ctrls) {
            const sc = angular.element(ctrl).scope();
            if (sc && sc.fileNavigator && sc.fileNavigator.fileList && sc.smartClick) {
                const item = sc.fileNavigator.fileList.find(i => i.model && i.model.name === 'Test_Lshape_NORMAL.ds');
                if (item) { sc.smartClick(item); sc.$apply(); return true; }
            }
        }
        return false;
    }""")
    await page.wait_for_timeout(15000)
    await page.evaluate("""() => {
        const all = document.querySelectorAll('[ng-controller]');
        for (const el of all) { const s = angular.element(el).scope(); if (s && s.result) { s.$root.selectedResult = s.result; s.$apply(); return; } }
    }""")
    await page.wait_for_timeout(4000)
    return r

async def uncheck_all(page):
    await page.evaluate("""() => document.querySelectorAll('.dsFieldRow input[type=checkbox]').forEach(cb => { if (cb.checked) cb.click(); })""")
    await page.wait_for_timeout(1000)
    await page.evaluate("""() => document.querySelectorAll('.dsInputSubFieldRow input[type=checkbox]').forEach(cb => { if (cb.checked) cb.click(); })""")
    await page.wait_for_timeout(800)

async def expand_input(page):
    await page.evaluate("""() => { const exp = document.querySelector('.dsInputDataExpander'); if (exp && exp.textContent.trim().startsWith('▶')) exp.click(); }""")
    await page.wait_for_timeout(700)

async def click_sub(page, label):
    await page.evaluate(f"""() => {{
        const rows = Array.from(document.querySelectorAll('.dsInputSubFieldRow'));
        const r = rows.find(r => r.textContent.toLowerCase().includes({json.dumps(label.lower())}));
        if (r) {{ const cb = r.querySelector('input[type=checkbox]'); if (cb) cb.click(); }}
    }}""")
    await page.wait_for_timeout(5000)

async def state(page, label):
    return await page.evaluate(f"""() => {{
        const rm = (typeof myBABYLON !== 'undefined') ? myBABYLON.ResultMesh : null;
        if (!rm) return {{ label: {json.dumps(label)}, no_rm: true }};
        const cf = rm.currentField;
        return {{
            label: {json.dumps(label)},
            cls: rm.constructor.name,
            field: cf ? cf.name : null,
            min: cf ? cf.min : null,
            max: cf ? cf.max : null,
            rmVis: rm.isVisible,
            meshVis: rm.mesh ? rm.mesh.isVisible : null,
            panels: Array.from(document.querySelectorAll('.dsOptionsPanelTitle')).map(p => p.textContent.trim()),
        }};
    }}""")

async def main():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            PROFILE, headless=False, args=["--start-maximized", "--no-sandbox"], no_viewport=True,
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        page.on("console", lambda m: CONSOLE.append(f"[{m.type}] {m.text[:300]}"))
        page.on("pageerror", lambda e: ERRORS.append(str(e)[:300]))
        cdp = await ctx.new_cdp_session(page)
        await cdp.send("Network.setCacheDisabled", {"cacheDisabled": True})
        await cdp.send("Network.clearBrowserCache")

        opened = await open_model(page)
        log(f"opened={opened}")
        await page.screenshot(path=os.path.join(OUT, "00_loaded.png"))
        if not opened:
            log("CANNOT open model")
            with open(os.path.join(OUT, "console.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(CONSOLE))
            await ctx.close()
            return

        await uncheck_all(page)
        await expand_input(page)
        await page.screenshot(path=os.path.join(OUT, "01_expanded.png"))

        # T1: fc vol
        log(">>> T1: fc vol")
        ERR_BEFORE = len(ERRORS)
        await click_sub(page, "fc vol")
        s1 = await state(page, "fc_vol")
        log(f"   {s1}")
        log(f"   errs+={len(ERRORS)-ERR_BEFORE}")
        await page.screenshot(path=os.path.join(OUT, "02_fc_vol.png"))

        # T2: switch ft vol
        log(">>> T2: ft vol (switch)")
        ERR_BEFORE = len(ERRORS)
        await click_sub(page, "ft vol")
        s2 = await state(page, "ft_vol")
        log(f"   {s2}")
        log(f"   errs+={len(ERRORS)-ERR_BEFORE}")
        await page.screenshot(path=os.path.join(OUT, "03_ft_vol.png"))

        # T3: fy rebar alone
        log(">>> T3: uncheck ft, check fy rebar")
        await click_sub(page, "ft vol")  # uncheck
        await page.wait_for_timeout(2000)
        ERR_BEFORE = len(ERRORS)
        await click_sub(page, "fy rebar")
        s3 = await state(page, "fy_rebar")
        log(f"   {s3}")
        log(f"   errs+={len(ERRORS)-ERR_BEFORE}")
        await page.screenshot(path=os.path.join(OUT, "04_fy_rebar.png"))

        # T4: fy rebar + ft vol, then edit Min in fy rebar (test TypeError fix)
        log(">>> T4: also check ft vol, then edit fy rebar Max")
        await click_sub(page, "ft vol")
        await page.wait_for_timeout(3000)
        ERR_BEFORE = len(ERRORS)
        edit_result = await page.evaluate("""() => {
            const panels = Array.from(document.querySelectorAll('.dsOptionsPanel'));
            const fyPanel = panels.find(p => {
                const t = p.querySelector('.dsOptionsPanelTitle');
                return t && t.textContent.includes('fy rebar');
            });
            if (!fyPanel) return 'no_fy_panel';
            const inputs = fyPanel.querySelectorAll('input[type=text], input[type=number]');
            if (inputs.length < 2) return 'no_inputs len='+inputs.length;
            // Try to set Max value
            const maxIn = inputs[1];
            const sc = angular.element(maxIn).scope();
            if (!sc.options || !sc.options.colorScaleOptions) return 'no_options';
            sc.options.colorScaleOptions[sc.mode].max = '300';
            sc.options.colorScaleOptions[sc.mode].auto = false;
            sc.$apply();
            if (sc.resetBound) { try { sc.resetBound(); } catch(e) { return 'resetBound_err: '+e.message; } }
            return 'set_OK';
        }""")
        log(f"   edit_result: {edit_result}")
        log(f"   errs+={len(ERRORS)-ERR_BEFORE}")
        await page.wait_for_timeout(2000)
        await page.screenshot(path=os.path.join(OUT, "05_after_edit.png"))

        # T5: try filtering on fy rebar
        log(">>> T5: enable filtering on fy rebar")
        ERR_BEFORE = len(ERRORS)
        filter_result = await page.evaluate("""() => {
            const panels = Array.from(document.querySelectorAll('.dsOptionsPanel'));
            const fyPanel = panels.find(p => {
                const t = p.querySelector('.dsOptionsPanelTitle');
                return t && t.textContent.includes('fy rebar');
            });
            if (!fyPanel) return 'no_fy_panel';
            const filterCb = fyPanel.querySelector('input[type=checkbox]');
            if (filterCb && !filterCb.checked) filterCb.click();
            const sc = angular.element(fyPanel).scope();
            if (sc && sc.options && sc.options.filter_options) {
                sc.options.filter_options.filter_min = '0';
                sc.options.filter_options.filter_max = '1';
                sc.options.filtering = true;
                sc.$apply();
                if (sc.updateFilteredMesh) sc.updateFilteredMesh();
                return 'filter_set';
            }
            return 'no_options';
        }""")
        log(f"   filter_result: {filter_result}")
        log(f"   errs+={len(ERRORS)-ERR_BEFORE}")
        await page.wait_for_timeout(3000)
        await page.screenshot(path=os.path.join(OUT, "06_filtered.png"))

        with open(os.path.join(OUT, "console.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(CONSOLE[-200:]))
        with open(os.path.join(OUT, "errors.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(ERRORS))
        with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"T1_fc": s1, "T2_ft": s2, "T3_fy": s3, "edit": edit_result, "filter": filter_result, "all_errors": ERRORS[:10]}, f, indent=2, default=str)

        log(f">>> TOTAL ERRORS: {len(ERRORS)}")
        for e in ERRORS[:5]: log(f"  {e}")

        await ctx.close()

if __name__ == "__main__":
    asyncio.run(main())
