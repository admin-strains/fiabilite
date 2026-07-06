# -*- coding: utf-8 -*-
# =====================================================================
# BANC : Option A (balayage + secante sur dlambda/ds) vs Option B
# (parametrique : 1 SOCP -> mecanisme u* -> formule fermee lambda(s) =
#  lam0 * W(u*,s0)/W(u*,s), glisser sans re-solver, corriger si derive).
# Compte les SOCPs de chaque methode pour trouver s* sur le cantilever.
#                                                        (2026-07-06, MM)
#
# CORE PARTAGE (C:\workspace\front). run_config -> temp dirs, PAS de
# SOCP_history (driver standalone), donc pas de souci disque.
#
#   C:\python3\python.exe tests\bench_optA_vs_optB_cantilever.py probe
#   C:\python3\python.exe tests\bench_optA_vs_optB_cantilever.py full
# =====================================================================
import os, sys, glob, json, time
os.environ["PL_FRONT"] = r"C:\workspace\front"      # <- core PARTAGE (avant import pl_harness)
sys.path.insert(0, r"C:\workspace\fiabilite\tests")
import numpy as np
from pl_harness import run_config

SLO, SHI = 0.8, 3.2          # course de la charge (bornes)
Z_TOP = 0.25                 # face haute du cantilever (h=0.5 -> +0.25)
FP_HALF_X = 0.3              # demi-longueur empreinte en x (0.6 total)
TOL_B = 0.03                 # tolerance predictor-corrector (rel.)
TOL_A = 0.02                 # arret secante |ds|
MESH = "0.10"
WBASE = r"C:\workspace\storage\admin\Moulin_Blanc\_bench_AB.ds"
NSOCP = [0]
T0 = time.time()


def make_cfg(xc, fy=550.0, p=-0.15):
    return {
        "name": "xc%.3f" % xc,
        "footprints": [{
            "name": "FPL",
            "polygon": [[xc - 0.3, -0.15, Z_TOP], [xc + 0.3, -0.15, Z_TOP],
                        [xc + 0.3, 0.15, Z_TOP], [xc - 0.3, 0.15, Z_TOP]],
            "F": [0, 0, p], "role": "live", "lc": "LCL",
            "mobile": {"path": [[xc - 1, 0, Z_TOP], [xc + 1, 0, Z_TOP]],
                       "position": 1.0, "unit": "absolute"},
        }],
        "regions": [{"param": "LIVE_LOAD", "load_case": "LCL"},
                    {"param": "LIVE_LOAD", "axis": "position", "load_case": "LCL"}],
        "mesh_size": MESH, "geom": {"fy_top": fy},
    }


def solve(xc, tag=""):
    """1 SOCP -> (lam, dlds, workdir). keep=True pour lire le mecanisme."""
    wd = WBASE.replace(".ds", "_%s_%.4f.ds" % (tag, xc))
    r = run_config(make_cfg(xc), workdir=wd, keep=True, write_debug=True)
    NSOCP[0] += 1
    lam = r["lam"]; dlds = (r["sens"] or {}).get("LIVE_LOAD:position:LCL")
    print("  [SOCP %2d | %4.0fs] xc=%.4f  lam=%s  dl/ds=%s  status=%s"
          % (NSOCP[0], time.time() - T0, xc, lam, dlds, r["status"]), flush=True)
    return {"xc": xc, "lam": lam, "dlds": dlds, "status": r["status"], "wd": wd, "err": r["err"]}


# ---------------------------------------------------------------------
# Parseur du mecanisme u* (gmsh 2.2, ElementNodeData "DISPLACEMENTS", tet10)
# -> profil W(xc) = |vz moyen sur l'empreinte| (travail de la charge sur u*)
# ---------------------------------------------------------------------
def parse_top_vz(wd):
    cands = glob.glob(os.path.join(wd, "*PL_cin_out*.msh"))
    if not cands:
        raise FileNotFoundError("PL_cin_out.msh introuvable dans %s" % wd)
    L = open(cands[0]).read().split("\n")
    i = L.index("$Nodes"); nn = int(L[i + 1]); nodes = {}
    for k in range(nn):
        q = L[i + 2 + k].split()
        nodes[int(q[0])] = (float(q[1]), float(q[2]), float(q[3]))
    j = L.index("$Elements"); ne = int(L[j + 1]); conn = {}
    for k in range(ne):
        q = L[j + 2 + k].split()
        if int(q[1]) == 11:                       # tet10
            nt = int(q[2]); conn[int(q[0])] = [int(x) for x in q[3 + nt:3 + nt + 10]]
    d = L.index("$ElementNodeData"); p = d + 1
    nst = int(L[p]); p += 1 + nst
    nrt = int(L[p]); p += 1 + nrt
    nit = int(L[p]); p += 1
    ints = [int(L[p + t]) for t in range(nit)]; p += nit
    ncomp, count = ints[1], ints[2]
    node_vz = {}
    for k in range(count):
        q = L[p + k].split()
        eid = int(q[0]); nnod = int(q[1]); vals = q[2:2 + nnod * ncomp]
        if eid not in conn:
            continue
        for li in range(nnod):
            node_vz[conn[eid][li]] = float(vals[li * ncomp + 2])   # composante z
    xs, vz = [], []
    for nid, (x, y, z) in nodes.items():
        if abs(z - Z_TOP) < 1e-3 and nid in node_vz:
            xs.append(x); vz.append(node_vz[nid])
    return np.array(xs), np.array(vz)


def make_W(wd):
    """Renvoie W(xc) = |vz moyen des noeuds de la face haute dans [xc-0.3, xc+0.3]|."""
    xs, vz = parse_top_vz(wd)
    order = np.argsort(xs); xs, vz = xs[order], vz[order]

    def W(xc):
        m = (xs >= xc - FP_HALF_X) & (xs <= xc + FP_HALF_X)
        if not m.any():
            return np.nan
        return abs(float(vz[m].mean()))
    return W, xs, vz


# ---------------------------------------------------------------------
# OPTION A : balayage court + secante sur dlambda/ds (algo Agnes)
# ---------------------------------------------------------------------
def option_A():
    print("\n===== OPTION A : balayage + secante sur dlambda/ds =====", flush=True)
    n0 = NSOCP[0]
    a = solve(SLO, "A"); b = solve(SHI, "A")
    ha, hb = a["dlds"], b["dlds"]
    hist = [(SLO, ha, a["lam"]), (SHI, hb, b["lam"])]
    if ha is None or hb is None:
        return {"mode": "FAILED", "sstar": None, "nsocp": NSOCP[0] - n0}
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


# ---------------------------------------------------------------------
# OPTION B : parametrique (formule fermee + predictor-corrector)
# ---------------------------------------------------------------------
def option_B(s_anchor=2.0):
    print("\n===== OPTION B : formule fermee lam0*W(s0)/W(s) + corrector =====", flush=True)
    n0 = NSOCP[0]
    grid = np.linspace(SLO, SHI, 101)
    trace = []
    r = solve(s_anchor, "B")                     # 1er SOCP : ancre + mecanisme
    for it in range(6):
        W, xs, vz = make_W(r["wd"])
        Wg = np.array([W(x) for x in grid])
        s_pred = float(grid[np.nanargmax(Wg)])   # argmax W = min lambda (GRATUIT, pas de SOCP)
        lam_formule_pred = r["lam"] * W(r["xc"]) / W(s_pred)
        rc = solve(s_pred, "B")                  # SOCP de controle (predictor-corrector)
        rel = abs(rc["lam"] - lam_formule_pred) / rc["lam"]
        trace.append({"iter": it, "anchor": r["xc"], "s_pred": s_pred,
                      "lam_formule": lam_formule_pred, "lam_vrai": rc["lam"], "rel": rel})
        print("    iter %d : ancre=%.3f -> s_pred=%.3f  lam_formule=%.4f  lam_vrai=%.4f  rel=%.2f%%"
              % (it, r["xc"], s_pred, lam_formule_pred, rc["lam"], 100 * rel), flush=True)
        if rel <= TOL_B or abs(s_pred - r["xc"]) < 1e-3:
            return {"mode": "converge", "sstar": s_pred, "lam": rc["lam"],
                    "nsocp": NSOCP[0] - n0, "trace": trace}
        r = rc                                   # la formule a derive -> re-ancrer sur le nouveau mecanisme
    return {"mode": "max_iter", "sstar": trace[-1]["s_pred"], "lam": trace[-1]["lam_vrai"],
            "nsocp": NSOCP[0] - n0, "trace": trace}


def probe():
    print("=== PROBE : 1 SOCP en xc=2.0, parse mecanisme, profil W(x) ===", flush=True)
    r = solve(2.0, "probe")
    W, xs, vz = make_W(r["wd"])
    print("  noeuds face haute : %d  (x in [%.2f, %.2f])" % (len(xs), xs.min(), xs.max()))
    print("  profil W(xc) sur la course :")
    for xc in np.linspace(SLO, SHI, 13):
        bar = "#" * int(60 * W(xc) / max(W(x) for x in np.linspace(SLO, SHI, 13)))
        print("    xc=%.2f  W=%.4e  %s" % (xc, W(xc), bar))
    grid = np.linspace(SLO, SHI, 101)
    s_pred = grid[np.nanargmax([W(x) for x in grid])]
    print("  -> argmax W (s* predit par B depuis ce seul SOCP) = %.3f" % s_pred)


def full():
    resA = option_A()
    resB = option_B(s_anchor=2.0)
    print("\n================= COMPARAISON A vs B =================", flush=True)
    print("OPTION A : s*=%.3f  lam=%.4f  mode=%s  -> %d SOCPs"
          % (resA["sstar"], resA["lam"], resA["mode"], resA["nsocp"]))
    print("OPTION B : s*=%.3f  lam=%.4f  mode=%s  -> %d SOCPs"
          % (resB["sstar"], resB["lam"], resB["mode"], resB["nsocp"]))
    dstar = abs(resA["sstar"] - resB["sstar"])
    print("Delta s* (A vs B) = %.4f m   |   total SOCPs = %d   |   %.0fs"
          % (dstar, NSOCP[0], time.time() - T0))
    json.dump({"A": resA, "B": resB, "delta_sstar": dstar, "nsocp_total": NSOCP[0]},
              open(os.path.join(r"C:\workspace\fiabilite\_docs", "bench_optA_vs_optB.json"), "w"), indent=2)
    print("JSON -> _docs\\bench_optA_vs_optB.json")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    (probe if mode == "probe" else full)()
