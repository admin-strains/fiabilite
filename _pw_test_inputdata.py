"""Verify the new '+ Input data' panel renders fc/ft volumic + fy rebar correctly."""
import asyncio, os, time
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8000"
HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE = os.path.join(HERE, ".playwright_profile")
OUT = os.path.join(HERE, "_autotest_inputdata")
os.makedirs(OUT, exist_ok=True)

CONSOLE, ERRORS = [], []
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

async def main():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            PROFILE, headless=False, args=["--start-maximized", "--no-sandbox"], no_viewport=True,
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        page.on("console", lambda m: CONSOLE.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: ERRORS.append(str(e)))
        cdp = await ctx.new_cdp_session(page)
        await cdp.send("Network.setCacheDisabled", {"cacheDisabled": True})
        await cdp.send("Network.clearBrowserCache")

        log(">>> Goto " + URL)
        await page.goto(URL, timeout=30000, wait_until="networkidle")
        await page.wait_for_timeout(3000)

        st = await page.evaluate("""() => {
            const all = document.querySelectorAll('[ng-controller]');
            for (const el of all) { const s = angular.element(el).scope(); if (s && s.result) return { ok: true }; }
            return { ok: false };
        }""")
        if not st.get('ok'):
            log(">>> Navigate Moulin_Blanc / Test_Lshape_NORMAL")
            await page.evaluate("""() => {
                const links = Array.from(document.querySelectorAll('a, span, td'));
                for (const el of links) if (el.textContent && el.textContent.trim() === 'Moulin_Blanc') { el.click(); return; }
            }""")
            await page.wait_for_timeout(1500)
            try: await page.locator("text=/Test_Lshape_NORMAL/").first.dblclick(timeout=10000)
            except: pass
            await page.wait_for_timeout(12000)

        await page.evaluate("""() => {
            const all = document.querySelectorAll('[ng-controller]');
            for (const el of all) { const s = angular.element(el).scope(); if (s && s.result) { s.$root.selectedResult = s.result; s.$apply(); return; } }
        }""")
        await page.wait_for_timeout(4000)

        # Uncheck everything to start clean
        await page.evaluate("""() => document.querySelectorAll('.dsFieldRow input[type=checkbox]').forEach(cb => { if (cb.checked) cb.click(); })""")
        await page.wait_for_timeout(1500)

        # STEP 1 : verify Input data group is present
        log(">>> STEP 1 : Verify '+ Input data' group present")
        groups = await page.evaluate("""() => {
            const rows = Array.from(document.querySelectorAll('.dsFieldRow'));
            return rows.map(r => { const l = r.querySelector('.dsFieldLabel span'); return l ? l.textContent.trim() : ''; });
        }""")
        log(f"    Groups: {groups}")
        has_input = any('Input' in g for g in groups)
        log(f"    ASSERT step1 (Input data row exists): {'PASS' if has_input else 'FAIL'}")
        await page.screenshot(path=os.path.join(OUT, "01_groups.png"))

        # STEP 2 : expand Input data
        log(">>> STEP 2 : Expand Input data")
        await page.evaluate("""() => {
            const rows = Array.from(document.querySelectorAll('.dsFieldRow'));
            const t = rows.find(r => { const l = r.querySelector('.dsFieldLabel span'); return l && l.textContent.trim().toLowerCase().includes('input'); });
            if (t) {
                const expander = t.querySelector('.dsInputDataChevron, .dsInputDataToggle, [ng-click*=toggleInputData], [ng-click*=Input]');
                if (expander) { expander.click(); }
                else {
                    const l = t.querySelector('.dsFieldLabel span');
                    if (l) l.click();
                }
            }
        }""")
        await page.wait_for_timeout(1500)
        subs = await page.evaluate("""() => {
            return Array.from(document.querySelectorAll('.dsInputSubFieldRow, .dsInputSubField'))
                .map(r => r.textContent.trim()).filter(t => t);
        }""")
        log(f"    Sub-fields visible: {subs}")
        await page.screenshot(path=os.path.join(OUT, "02_expanded.png"))

        # STEP 3 : click 'fc vol'
        log(">>> STEP 3 : Click 'fc vol' sub-field")
        await page.evaluate("""() => {
            const rows = Array.from(document.querySelectorAll('.dsInputSubFieldRow, .dsInputSubField'));
            const t = rows.find(r => r.textContent.toLowerCase().includes('fc'));
            if (t) { const cb = t.querySelector('input[type=checkbox]'); if (cb && !cb.checked) cb.click(); else if (t) t.click(); }
        }""")
        await page.wait_for_timeout(5000)
        info_fc = await page.evaluate("""() => {
            const rm = myBABYLON && myBABYLON.ResultMesh;
            const cf = rm && rm.currentField;
            return {
                rmExists: !!rm,
                currentFieldName: cf ? cf.name : null,
                meshCount: typeof scene !== 'undefined' ? scene.meshes.length : 0,
                anyColored: typeof scene !== 'undefined' ? scene.meshes.filter(m => m.material && m.material.diffuseTexture).length : 0,
            };
        }""")
        log(f"    {info_fc}")
        await page.screenshot(path=os.path.join(OUT, "03_fc_vol.png"))
        ok_fc = info_fc.get('currentFieldName') and 'InputData' in (info_fc.get('currentFieldName') or '')
        log(f"    ASSERT step3 (fc vol applied): {'PASS' if ok_fc else 'FAIL'}")

        # STEP 4 : click 'ft vol' -> should swap (volumic exclusivity)
        log(">>> STEP 4 : Click 'ft vol' (volumic exclusivity)")
        await page.evaluate("""() => {
            const rows = Array.from(document.querySelectorAll('.dsInputSubFieldRow, .dsInputSubField'));
            const t = rows.find(r => r.textContent.toLowerCase().includes('ft'));
            if (t) { const cb = t.querySelector('input[type=checkbox]'); if (cb && !cb.checked) cb.click(); else if (t) t.click(); }
        }""")
        await page.wait_for_timeout(5000)
        info_ft = await page.evaluate("""() => {
            const rm = myBABYLON && myBABYLON.ResultMesh;
            const cf = rm && rm.currentField;
            return { currentFieldName: cf ? cf.name : null };
        }""")
        log(f"    {info_ft}")
        await page.screenshot(path=os.path.join(OUT, "04_ft_vol.png"))

        # STEP 5 : click 'fy rebar' -> should be independent
        log(">>> STEP 5 : Click 'fy rebar' (independent)")
        await page.evaluate("""() => {
            const rows = Array.from(document.querySelectorAll('.dsInputSubFieldRow, .dsInputSubField'));
            const t = rows.find(r => r.textContent.toLowerCase().includes('fy'));
            if (t) { const cb = t.querySelector('input[type=checkbox]'); if (cb && !cb.checked) cb.click(); else if (t) t.click(); }
        }""")
        await page.wait_for_timeout(5000)
        info_fy = await page.evaluate("""() => {
            const rm = myBABYLON && myBABYLON.ResultMesh;
            const cf = rm && rm.currentField;
            const meshes = scene.meshes.map(m => ({name: m.name, visible: m.isVisible}));
            return { currentFieldName: cf ? cf.name : null, meshCount: meshes.length, sampleMeshes: meshes.slice(0,10) };
        }""")
        log(f"    {info_fy}")
        await page.screenshot(path=os.path.join(OUT, "05_fy_rebar.png"))

        log(">>> Page errors:")
        for e in ERRORS: log(f"    {e}")

        with open(os.path.join(OUT, "console_log.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(CONSOLE))

        await page.wait_for_timeout(2000)
        await ctx.close()

if __name__ == "__main__":
    asyncio.run(main())
