# -*- coding: utf-8 -*-
# =====================================================================
# HARNESS de validation CHARGE PONCTUELLE (2026-07-03, MM)
# =====================================================================
# Genere dsCad + dsLoad a partir d'une CONFIG (points + footprints + regions),
# lance mesh + solve en capturant le stdout C (lignes [PointLoad-CONSERV] pour
# la conservation exacte), et renvoie un dict structure :
#   status, lambda, sens{}, conserv[] (SUM(w), F, err), located[] (in/projected/ignored)
#
# Utilise des repertoires modele ISOLES (copie de Test_point_load.ds -> tmp) pour
# permettre des runs PARALLELES (agents) sans se marcher dessus.
# =====================================================================
import os, sys, json, re, shutil, tempfile, io, contextlib

FRONT = os.environ.get("PL_FRONT", r"C:\workspace\front_mohamad")   # 2026-07-06 (MM) : surchargeable pour tester le core partage
CAT = FRONT if os.path.isdir(os.path.join(FRONT, "STRAINS", "common", "Catalog")) else r"C:\workspace\front"
for d in [os.path.join(FRONT, r"STRAINS\rupt\core\bin"), os.path.join(FRONT, r"STRAINS\rupt\core"),
          os.path.join(FRONT, r"STRAINS\common\Dll"), os.path.join(FRONT, r"STRAINS\rupt\core\bin\meshgems"),
          os.path.join(FRONT, r"STRAINS\rupt\core\bin\mosek")]:
    if os.path.isdir(d): os.add_dll_directory(d)
sys.path.insert(0, FRONT); sys.path.insert(0, r'C:\workspace\fiabilite')

from STRAINS.rupt.APIs.CetCAD_API import *
from STRAINS.rupt.APIs import CetLOAD
from STRAINS.rupt.APIs.CetLOAD_API import *
import STRAINS.rupt.core.CetMESH as CetMESH
from STRAINS.rupt.core import CetSOLV

_INIT_DONE = False
def _init_catalog():
    global _INIT_DONE
    if _INIT_DONE: return
    def _f(p):
        with open(p) as f: return f.read()
    INITCATALOG(_f(os.path.join(CAT, r"STRAINS\common\Catalog\CatalogTopo.json")),
                _f(os.path.join(CAT, r"STRAINS\common\Catalog\CatalogDimensions.json")),
                _f(os.path.join(CAT, r"STRAINS\common\Catalog\CatalogBolts.json")))
    _INIT_DONE = True

# ---- generation dsCad (poutre cantilever BA + points de charge) ----
_DSCAD_TMPL = """\
fy_top = {fy_top}
fy_bot = 500.0
fc = 20.0
ft = 0.1
E  = 35.0
fyd_top = fy_top; fyd_bot = fy_bot; fcd = fc
b = {b}
h = {h}
L = {L}
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
    BOUNDARY('Fixed', 'Beam:f1', POINT(0, 0, 0),
             X=VECTOR(-1, 0, 0), Y=VECTOR(0, -1, 0))
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
{points_cad}
{polys_cad}
"""

def _gen_dscad(cfg):
    pts = cfg.get("points", [])
    points_cad = ""
    for i, p in enumerate(pts):
        x, y, z = p["xyz"]
        points_cad += "POINTS(NAME='%s', XS=['%.10f'], YS=['%.10f'], ZS=['%.10f'])\n" % (p["name"], x, y, z)
    polys_cad = ""
    for fp in cfg.get("footprints", []):
        pts3 = fp["polygon"]
        ptsrepr = ", ".join("POINT(%.6f,%.6f,%.6f)" % (q[0], q[1], q[2]) for q in pts3)
        polys_cad += "POLYGON(NAME='%s', POINTS=[%s])\n" % (fp["name"], ptsrepr)
    g = cfg.get("geom", {})
    return _DSCAD_TMPL.format(fy_top=g.get("fy_top", 550.0), b=g.get("b", 0.40),
                              h=g.get("h", 0.50), L=g.get("L", 4.0),
                              points_cad=points_cad, polys_cad=polys_cad)

def _gen_dsload(cfg):
    """Un LOAD_CASE par nom, chaque LC porte ses points + footprints (live) ; les dead
    vont dans DEAD_LOAD_CASES. MOVING_LOAD si le LC a une cle 'moving'."""
    live_lcs = {}   # name -> list of (kind, item)
    dead_lcs = {}
    moving = {}     # lc name -> moving spec
    for p in cfg.get("points", []):
        d = dead_lcs if p.get("role") == "dead" else live_lcs
        d.setdefault(p["lc"], []).append(("point", p))
        if p.get("mobile"): moving[p["lc"]] = p["mobile"]
    for fp in cfg.get("footprints", []):
        d = dead_lcs if fp.get("role") == "dead" else live_lcs
        d.setdefault(fp["lc"], []).append(("poly", fp))
        if fp.get("mobile"): moving[fp["lc"]] = fp["mobile"]

    lines = ["q = 100.0", ""]
    # supports vont dans le PREMIER live LC (il en faut au moins un)
    all_live = list(live_lcs.keys())
    all_dead = list(dead_lcs.keys())
    first_live = all_live[0] if all_live else None

    def emit_lc(name, items, is_first_live):
        lines.append("with LOAD_CASE('%s'):" % name)
        if is_first_live:
            lines.append("    SUPPORT(BOUNDARY='Fixed', MEAN=False, RIGID=False, X=True, Y=True, Z=True)")
        for kind, it in items:
            if kind == "point":
                F = it["F"]
                lines.append("    LOAD(POINT='%s', X=str(%.6f), Y=str(%.6f), Z=str(%.6f))"
                             % (it["name"] + "_0", F[0], F[1], F[2]))
            else:  # poly footprint
                F = it["F"]
                lines.append("    LOAD(POLYGON='%s', X=str(%.6f), Y=str(%.6f), Z=str(%.6f))"
                             % (it["name"], F[0], F[1], F[2]))
        if name in moving:
            mv = moving[name]
            path = mv["path"]
            # 2026-07-04 (MM) : variable= optionnelle -> groupe de position PARTAGE entre
            # plusieurs load cases (meme variable) ; absente -> groupe = nom du LC.
            _var = ", variable='%s'" % mv["variable"] if mv.get("variable") else ""
            lines.append("    MOVING_LOAD(path=[(%.6f,%.6f,%.6f),(%.6f,%.6f,%.6f)], position=%.6f, unit='%s'%s)"
                         % (path[0][0], path[0][1], path[0][2], path[1][0], path[1][1], path[1][2],
                            mv.get("position", 0.5), mv.get("unit", "absolute"), _var))
        lines.append("")

    for i, name in enumerate(all_live):
        emit_lc(name, live_lcs[name], i == 0)
    for name in all_dead:
        emit_lc(name, dead_lcs[name], False)

    live_arg = ", ".join("('%s', '1')" % n for n in all_live)
    dead_arg = ", ".join("('%s', '1')" % n for n in all_dead)
    ana = "YIELD_ANALYSIS('Yield_analysis0',\n"
    ana += "               LIVE_LOAD_CASES=[%s],\n" % live_arg
    if all_dead:
        ana += "               DEAD_LOAD_CASES=[%s],\n" % dead_arg
    ana += "               MESH={\"global_physical_size\": \"%s\"})\n" % cfg.get("mesh_size", "0.06")
    lines.append(ana)
    return "\n".join(lines)

# ---- capture stdout C (fd 1) pour parser les lignes [PointLoad*] ----
@contextlib.contextmanager
def _capture_cstdout(path):
    sys.stdout.flush()
    old = os.dup(1)
    f = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(f, 1); os.close(f)
    try:
        yield
    finally:
        sys.stdout.flush(); os.dup2(old, 1); os.close(old)

def run_config(cfg, workdir=None, keep=False, write_debug=False):
    """Execute une CONFIG. Renvoie dict {status, lam, sens, conserv, located, npts, err}.
    write_debug=True -> write_debug_files='true' au solve (doGmsh) -> ecrit PL_cin_out.msh
    (mecanisme u*) dans le workdir. 2026-07-06 (MM), additif : defaut inchange."""
    _init_catalog()
    base = r"C:\workspace\storage\admin\Moulin_Blanc\Test_point_load.ds"
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="plval_", suffix=".ds")
    PATH = workdir
    if os.path.isdir(PATH): shutil.rmtree(PATH, ignore_errors=True)
    os.makedirs(PATH, exist_ok=True)
    open(os.path.join(PATH, "dsCad.txt"), "w").write(_gen_dscad(cfg))
    open(os.path.join(PATH, "dsLoad.txt"), "w").write(_gen_dsload(cfg))
    AN = 'Yield_analysis0'
    logf = os.path.join(PATH, "_cstdout.log")
    res = {"name": cfg.get("name", "?"), "status": None, "lam": None, "sens": {},
           "conserv": [], "located": [], "err": None}
    try:
        with _capture_cstdout(logf):
            model = MODEL(); SET_CONTEXT(model, PATH)
            exec(open(os.path.join(PATH, "dsCad.txt")).read(), globals())
            model.Save(os.path.join(PATH, AN + ".dscad"))
            with CetLOAD.LOAD_MODEL(model, PATH):
                exec(open(os.path.join(PATH, "dsLoad.txt")).read(), globals())
            ms = str(cfg.get("mesh_size", "0.06"))
            Mk = {"cadSurfOptions": {"volume_gradation":1.5,"gradation":1.5,"anisotropic_ratio":10},
                  "tetraOptions": {"optimisation_level":"standard","verbose":"10"},
                  "global_physical_size": float(ms), "max_size": float(ms), "min_size":"-1",
                  "gradation":1.5,"volume_gradation":1.5,"optimisation_level":"standard","anisotropic_ratio":"10",
                  "geometric_approximation_min":"4","geometric_approximation_max":"25",
                  "geometric_approximation_on_edge":"false","geometric_approximation_on_face":"true",
                  "use_surface_proximity":"false","surface_proximity_ratio":0,"approach":"kinematic",
                  "write_debug_files":"false","is_iso":"true","coeff_on_error":0.01,"remesh_type":1,
                  "old_size_factor":0.0,"model_handle":model.GETHANDLEPTR()}
            CetMESH.ANISO_MESH(AN, 0, PATH, **Mk)
            kwargs = {"scaling":1,"write_debug_files":("true" if write_debug else "false")}
            exec(open(r"C:\workspace\fiabilite\InitSolver.py").read(), globals())
            kwargs.update(static_params=static_params, cinematic_params=cinematic_params,
                          MKLPardiso_params=MKLPardiso_params, MyPardiso_params=MyPardiso_params,
                          MUMPS_params=MUMPS_params, FullLorentz=False, LorentzToSdp=False, SdpToLorentz=0,
                          printIntPointSolutioEvolution=False, trace_sur_point_integration=False,
                          calculate_error="false", max_nbOfDiv=0, customized_inc=[1],
                          tetra_discontinuities=False, activated_plasticity=True, welds_throat_limit=True,
                          approach="kinematic")
            kwargs["sensitivity_analysis"] = "true"
            kwargs["sensitivity_regions"] = json.dumps(cfg.get("regions", []))
            kwargs["model_handle"] = model.GETHANDLEPTR()
            CetSOLV.SOLV(AN, 0, PATH, **kwargs)
        info = json.load(open(os.path.join(PATH, AN + "_0_kine.dsmetares")))["info"]
        res["status"] = info.get("solver_status")
        res["lam"] = (info.get("Primal_bound") or [None])[0]
        res["sens"] = info.get("Sensitivity", {}) or {}
    except Exception as e:
        res["err"] = repr(e)[:300]
    # parse les lignes de conservation + localisation
    try:
        txt = open(logf, encoding="latin-1").read()
    except Exception:
        txt = ""
    for m in re.finditer(r"\[PointLoad-CONSERV\] ipl=(\d+) nw=(\d+) SUM\(w\)=([\-0-9.eE+]+)\s+F=\(([^)]+)\)\s+SUM\(w\*F\)=\(([^)]+)\)", txt):
        sw = float(m.group(3))
        res["conserv"].append({"ipl": int(m.group(1)), "nw": int(m.group(2)), "sumw": sw, "err_sumw": abs(sw - 1.0)})
    for m in re.finditer(r"\[PointLoad\] p=\(([^)]+)\) (-> tetra \d+|hors maillage -> PROJETE|HORS maillage.*IGNORE|FAST-PATH)", txt):
        tag = m.group(2)
        kind = ("located" if tag.startswith("-> tetra") else
                "projected" if "PROJETE" in tag else
                "ignored" if "IGNORE" in tag else "fastpath")
        res["located"].append(kind)
    res["npts"] = len(cfg.get("points", []))
    if not keep and workdir.startswith(tempfile.gettempdir()):
        shutil.rmtree(PATH, ignore_errors=True)
    return res

if __name__ == "__main__":
    # smoke test du harness sur 1 config interieur
    cfg = {"name": "smoke_interior",
           "points": [{"name": "P1", "xyz": [2.0, 0.0, 0.2], "F": [0, 0, -100], "role": "live", "lc": "trafic"}],
           "regions": [{"param": "LIVE_LOAD", "load_case": "trafic"}],
           "mesh_size": "0.08"}
    r = run_config(cfg, workdir=r"C:\workspace\storage\admin\Moulin_Blanc\_plval_smoke.ds", keep=True)
    print(json.dumps(r, indent=2))
