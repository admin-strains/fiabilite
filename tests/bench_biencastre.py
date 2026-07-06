# -*- coding: utf-8 -*-
# =====================================================================
# BANC bi-encastre (poutre encastree AUX DEUX BOUTS) : minimum INTERIEUR
# a mi-portee -> exerce vraiment la secante (A) et le re-solve multi-
# mecanisme (B). Compare le nombre de SOCPs pour trouver s*.  (2026-07-06, MM)
#
# CORE PARTAGE. dsCad genere ici (2 BOUNDARY Fixed sur Beam:f1 et Beam:f2).
# Reutilise le parseur de mecanisme + make_W du banc cantilever.
#
#   C:\python3\python.exe tests\bench_biencastre.py probe
#   C:\python3\python.exe tests\bench_biencastre.py full
# =====================================================================
import os, sys, glob, json, time, shutil, contextlib
os.environ["PL_FRONT"] = r"C:\workspace\front"
sys.path.insert(0, r"C:\workspace\fiabilite\tests")
import numpy as np
import pl_harness                                   # setup sys.path + DLLs + APIs
from pl_harness import _init_catalog, _capture_cstdout
from bench_optA_vs_optB_cantilever import parse_top_vz, make_W, SLO, SHI, Z_TOP, FP_HALF_X
from STRAINS.rupt.APIs.CetCAD_API import *
from STRAINS.rupt.APIs import CetLOAD
from STRAINS.rupt.APIs.CetLOAD_API import *
import STRAINS.rupt.core.CetMESH as CetMESH
from STRAINS.rupt.core import CetSOLV

MESH = "0.10"
TOL_B = 0.03
TOL_A = 0.02
WBASE = r"C:\workspace\storage\admin\Moulin_Blanc\_bench_BI.ds"
NSOCP = [0]
T0 = time.time()

# ---- dsCad : poutre BA encastree AUX DEUX BOUTS + footprint 'FPL' en xc ----
_DSCAD = """\
fy_top = 550.0
fy_bot = 500.0
fc = 20.0; ft = 0.1; E = 35.0
fyd_top = fy_top; fyd_bot = fy_bot; fcd = fc
b = 0.40; h = 0.50; L = 4.0
phi = 16.0; phi_bot = 10.0; cover = 0.04; n_bars = 4
z_top = h/2 - cover - phi/1000.0/2.0
y0_top = -b/2 + cover + phi/1000.0/2.0
y1_top =  b/2 - cover - phi/1000.0/2.0
z_bot = -h/2 + cover + phi_bot/1000.0/2.0
y0_bot = -b/2 + cover + phi_bot/1000.0/2.0
y1_bot =  b/2 - cover - phi_bot/1000.0/2.0
BLOCK('Beam', L, 'RECTANGLE', {{'b':b, 'h':h}}, 0,
      POINT(0, 0, 0), Plus_XLoc, 'XAxis', Plus_ZLoc, 'ZAxis',
      COMPRESSIVE_STRENGTH=str(fcd), TENSILE_STRENGTH=str(ft),
      YOUNG_MODULUS=str(E), POISSON_RATIO='0.2', DENSITY='2.5')
BOUNDARIES(
    BOUNDARY('Fixed',  'Beam:f1', POINT(0, 0, 0), X=VECTOR(-1, 0, 0), Y=VECTOR(0, -1, 0)),
    BOUNDARY('Fixed2', 'Beam:f2', POINT(L, 0, 0), X=VECTOR( 1, 0, 0), Y=VECTOR(0, -1, 0))
)
ys_top = [y0_top + (y1_top - y0_top) * i / (n_bars - 1) for i in range(n_bars)]
REBAR('HA1', E=EDGE(PS=[POINT(0, ys_top[0], z_top), POINT(L, ys_top[0], z_top)]), DIAMETER=phi, GRADE=fyd_top, DISTANCE=0)
REBAR('HA2', E=EDGE(PS=[POINT(0, ys_top[1], z_top), POINT(L, ys_top[1], z_top)]), DIAMETER=phi, GRADE=fyd_top, DISTANCE=0)
REBAR('HA3', E=EDGE(PS=[POINT(0, ys_top[2], z_top), POINT(L, ys_top[2], z_top)]), DIAMETER=phi, GRADE=fyd_top, DISTANCE=0)
REBAR('HA4', E=EDGE(PS=[POINT(0, ys_top[3], z_top), POINT(L, ys_top[3], z_top)]), DIAMETER=phi, GRADE=fyd_top, DISTANCE=0)
ys_bot = [y0_bot + (y1_bot - y0_bot) * i / (n_bars - 1) for i in range(n_bars)]
REBAR('HA5', E=EDGE(PS=[POINT(0, ys_bot[0], z_bot), POINT(L, ys_bot[0], z_bot)]), DIAMETER=phi_bot, GRADE=fyd_bot, DISTANCE=0)
REBAR('HA6', E=EDGE(PS=[POINT(0, ys_bot[1], z_bot), POINT(L, ys_bot[1], z_bot)]), DIAMETER=phi_bot, GRADE=fyd_bot, DISTANCE=0)
REBAR('HA7', E=EDGE(PS=[POINT(0, ys_bot[2], z_bot), POINT(L, ys_bot[2], z_bot)]), DIAMETER=phi_bot, GRADE=fyd_bot, DISTANCE=0)
REBAR('HA8', E=EDGE(PS=[POINT(0, ys_bot[3], z_bot), POINT(L, ys_bot[3], z_bot)]), DIAMETER=phi_bot, GRADE=fyd_bot, DISTANCE=0)
POLYGON(NAME='FPL', POINTS=[POINT({x0:.6f},-0.15,0.25), POINT({x1:.6f},-0.15,0.25), POINT({x1:.6f},0.15,0.25), POINT({x0:.6f},0.15,0.25)])
"""

_DSLOAD = """\
q = 100.0

with LOAD_CASE('LCL'):
    SUPPORT(BOUNDARY='Fixed',  MEAN=False, RIGID=False, X=True, Y=True, Z=True)
    SUPPORT(BOUNDARY='Fixed2', MEAN=False, RIGID=False, X=True, Y=True, Z=True)
    LOAD(POLYGON='FPL', X=str(0.0), Y=str(0.0), Z=str(-0.15))
    MOVING_LOAD(path=[({xm:.6f},0,0.25),({xp:.6f},0,0.25)], position=1.0, unit='absolute')

YIELD_ANALYSIS('Yield_analysis0',
               LIVE_LOAD_CASES=[('LCL', '1')],
               MESH={{"global_physical_size": "{ms}"}})
"""
REGIONS = [{"param": "LIVE_LOAD", "load_case": "LCL"},
           {"param": "LIVE_LOAD", "axis": "position", "load_case": "LCL"}]


def solve(xc, tag=""):
    _init_catalog()
    NSOCP[0] += 1
    PATH = WBASE.replace(".ds", "_%s_%.4f.ds" % (tag, xc))
    if os.path.isdir(PATH):
        shutil.rmtree(PATH, ignore_errors=True)
    os.makedirs(PATH, exist_ok=True)
    open(os.path.join(PATH, "dsCad.txt"), "w").write(
        _DSCAD.format(x0=xc - 0.3, x1=xc + 0.3))
    open(os.path.join(PATH, "dsLoad.txt"), "w").write(
        _DSLOAD.format(xm=xc - 1, xp=xc + 1, ms=MESH))
    AN = "Yield_analysis0"
    out = {"xc": xc, "lam": None, "dlds": None, "status": None, "wd": PATH, "err": None}
    try:
        with _capture_cstdout(os.path.join(PATH, "_c.log")):
            model = MODEL(); SET_CONTEXT(model, PATH)
            exec(open(os.path.join(PATH, "dsCad.txt")).read(), globals())
            model.Save(os.path.join(PATH, AN + ".dscad"))
            with CetLOAD.LOAD_MODEL(model, PATH):
                exec(open(os.path.join(PATH, "dsLoad.txt")).read(), globals())
            Mk = {"cadSurfOptions": {"volume_gradation":1.5,"gradation":1.5,"anisotropic_ratio":10},
                  "tetraOptions": {"optimisation_level":"standard","verbose":"10"},
                  "global_physical_size": float(MESH), "max_size": float(MESH), "min_size":"-1",
                  "gradation":1.5,"volume_gradation":1.5,"optimisation_level":"standard","anisotropic_ratio":"10",
                  "geometric_approximation_min":"4","geometric_approximation_max":"25",
                  "geometric_approximation_on_edge":"false","geometric_approximation_on_face":"true",
                  "use_surface_proximity":"false","surface_proximity_ratio":0,"approach":"kinematic",
                  "write_debug_files":"false","is_iso":"true","coeff_on_error":0.01,"remesh_type":1,
                  "old_size_factor":0.0,"model_handle":model.GETHANDLEPTR()}
            CetMESH.ANISO_MESH(AN, 0, PATH, **Mk)
            kwargs = {"scaling":1,"write_debug_files":"true"}          # <- doGmsh -> PL_cin_out.msh
            exec(open(r"C:\workspace\fiabilite\InitSolver.py").read(), globals())
            kwargs.update(static_params=static_params, cinematic_params=cinematic_params,
                          MKLPardiso_params=MKLPardiso_params, MyPardiso_params=MyPardiso_params,
                          MUMPS_params=MUMPS_params, FullLorentz=False, LorentzToSdp=False, SdpToLorentz=0,
                          printIntPointSolutioEvolution=False, trace_sur_point_integration=False,
                          calculate_error="false", max_nbOfDiv=0, customized_inc=[1],
                          tetra_discontinuities=False, activated_plasticity=True, welds_throat_limit=True,
                          approach="kinematic", model_handle=model.GETHANDLEPTR())
            kwargs["sensitivity_analysis"] = "true"
            kwargs["sensitivity_regions"] = json.dumps(REGIONS)
            CetSOLV.SOLV(AN, 0, PATH, **kwargs)
        info = json.load(open(os.path.join(PATH, AN + "_0_kine.dsmetares")))["info"]
        out["status"] = info.get("solver_status")
        out["lam"] = (info.get("Primal_bound") or [None])[0]
        s = info.get("Sensitivity", {}) or {}
        out["dlds"] = s.get("LIVE_LOAD:position:LCL")
    except Exception as e:
        out["err"] = repr(e)[:200]
    print("  [SOCP %2d | %4.0fs] xc=%.4f  lam=%s  dl/ds=%s  status=%s  err=%s"
          % (NSOCP[0], time.time() - T0, xc, out["lam"], out["dlds"], out["status"], out["err"]), flush=True)
    return out


def option_A():
    print("\n===== OPTION A (bi-encastre) : balayage + secante =====", flush=True)
    n0 = NSOCP[0]
    a = solve(SLO, "A"); b = solve(SHI, "A")
    ha, hb = a["dlds"], b["dlds"]
    hist = [(SLO, ha, a["lam"]), (SHI, hb, b["lam"])]
    if ha is None or hb is None:
        return {"mode": "FAILED", "sstar": None, "nsocp": NSOCP[0] - n0, "hist": hist}
    if ha < 0 and hb < 0:
        res = {"mode": "edge_droit", "sstar": SHI, "lam": b["lam"]}
    elif ha > 0 and hb > 0:
        res = {"mode": "edge_gauche", "sstar": SLO, "lam": a["lam"]}
    elif ha > 0 and hb < 0:
        s = SLO if a["lam"] <= b["lam"] else SHI
        res = {"mode": "max_int->bord", "sstar": s, "lam": min(a["lam"], b["lam"])}
    else:                                        # min interieur -> secante
        s0, h0, s1, h1 = SLO, ha, SHI, hb; sstar = lam = None
        for _ in range(8):
            if abs(h1 - h0) < 1e-14:
                break
            s2 = min(max(s1 - h1 * (s1 - s0) / (h1 - h0), SLO), SHI)
            c = solve(s2, "A"); hist.append((s2, c["dlds"], c["lam"]))
            sstar, lam = s2, c["lam"]
            if abs(s2 - s1) < TOL_A:
                break
            s0, h0, s1, h1 = s1, h1, s2, c["dlds"]
        res = {"mode": "secante", "sstar": sstar, "lam": lam}
    res["nsocp"] = NSOCP[0] - n0; res["hist"] = hist
    return res


def option_B(s_anchor=1.2, kkt_gate=True):
    """kkt_gate=False : convergence NAIVE (lam_formule ~ lam_vrai) -> peut se PIEGER.
    kkt_gate=True  : convergence gatee par le VRAI dl/ds -> 0 (KKT), gratuit au controle."""
    print("\n===== OPTION B (bi-encastre) : formule fermee + corrector (kkt_gate=%s) =====" % kkt_gate, flush=True)
    n0 = NSOCP[0]
    grid = np.linspace(SLO, SHI, 101)
    LC = SHI - SLO
    trace = []
    r = solve(s_anchor, "B")
    for it in range(12):
        W, xs, vz = make_W(r["wd"])
        Wg = np.array([W(x) for x in grid])
        s_pred = float(grid[np.nanargmax(Wg)])
        lam_formule_pred = r["lam"] * W(r["xc"]) / W(s_pred)
        rc = solve(s_pred, "B")
        rel = abs(rc["lam"] - lam_formule_pred) / rc["lam"]
        kkt = abs(rc["dlds"]) * LC / rc["lam"]                # |dl/ds|*L/lam : ~0 au vrai min
        trace.append({"iter": it, "anchor": round(r["xc"], 4), "s_pred": s_pred,
                      "lam_formule": lam_formule_pred, "lam_vrai": rc["lam"], "rel": rel, "kkt": kkt})
        print("    iter %d : ancre=%.3f -> s_pred=%.3f  lam_f=%.4f  lam_vrai=%.4f  rel=%.2f%%  KKT=%.3f"
              % (it, r["xc"], s_pred, lam_formule_pred, rc["lam"], 100 * rel, kkt), flush=True)
        conv = (rel <= TOL_B) and (kkt <= 0.05 if kkt_gate else True)
        if conv or abs(s_pred - r["xc"]) < 1e-3:
            return {"mode": "converge", "sstar": s_pred, "lam": rc["lam"],
                    "nsocp": NSOCP[0] - n0, "trace": trace}
        r = rc
    return {"mode": "max_iter", "sstar": trace[-1]["s_pred"], "lam": trace[-1]["lam_vrai"],
            "nsocp": NSOCP[0] - n0, "trace": trace}


def probe():
    print("=== PROBE bi-encastre : 1 SOCP en xc=1.2, profil W(x) ===", flush=True)
    r = solve(1.2, "probe")
    if r["status"] != "OPTIMAL":
        print("  !! status non OPTIMAL, err=%s" % r["err"]); return
    W, xs, vz = make_W(r["wd"])
    print("  noeuds face haute : %d  (x in [%.2f, %.2f])  dl/ds=%s" % (len(xs), xs.min(), xs.max(), r["dlds"]))
    mx = max(W(x) for x in np.linspace(SLO, SHI, 13))
    for xc in np.linspace(SLO, SHI, 13):
        print("    xc=%.2f  W=%.4e  %s" % (xc, W(xc), "#" * int(60 * W(xc) / mx)))
    grid = np.linspace(SLO, SHI, 101)
    print("  -> argmax W (s* predit par B) = %.3f" % grid[np.nanargmax([W(x) for x in grid])])


def full():
    resA = option_A()
    resB_naif = option_B(s_anchor=1.2, kkt_gate=False)
    resB_kkt = option_B(s_anchor=1.2, kkt_gate=True)
    print("\n============= COMPARAISON A vs B (bi-encastre) =============", flush=True)
    print("OPTION A            : s*=%.3f  lam=%.4f  mode=%s  -> %d SOCPs"
          % (resA["sstar"] or -1, resA["lam"] or -1, resA["mode"], resA["nsocp"]))
    print("OPTION B (naif)     : s*=%.3f  lam=%.4f  mode=%s  -> %d SOCPs   [PIEGE si != A]"
          % (resB_naif["sstar"] or -1, resB_naif["lam"] or -1, resB_naif["mode"], resB_naif["nsocp"]))
    print("OPTION B (KKT gate) : s*=%.3f  lam=%.4f  mode=%s  -> %d SOCPs"
          % (resB_kkt["sstar"] or -1, resB_kkt["lam"] or -1, resB_kkt["mode"], resB_kkt["nsocp"]))
    print("total %d SOCPs   |   %.0fs" % (NSOCP[0], time.time() - T0))
    json.dump({"A": resA, "B_naif": resB_naif, "B_kkt": resB_kkt, "nsocp_total": NSOCP[0]},
              open(os.path.join(r"C:\workspace\fiabilite\_docs", "bench_biencastre.json"), "w"), indent=2)
    print("JSON -> _docs\\bench_biencastre.json")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    (probe if mode == "probe" else full)()
