# -*- coding: utf-8 -*-
# =====================================================================
# SWEEP de validation ANALYTIQUE (2026-07-03, MM) -- nouveau format MOVING_LOAD.
# =====================================================================
# Reproduit + etend la figure "VALIDATION sensibilite analytique". Sur le cantilever
# a footprint mobile (Test_moving_load.ds, sans shift), on balaie s in [0,1] et a
# CHAQUE s on demande 3 sensibilites ENSEMBLE (coexistence) :
#   (1) position  : {LIVE_LOAD, axis:position, load_case:convoi}  -> dalpha/ds
#   (2) live mag   : {LIVE_LOAD, load_case:convoi}                -> dalpha/dm_L
#   (3) dead mag   : {DEAD_LOAD, load_case:LC_poids}              -> dalpha/dm_d
# et on compare chacune a l'analytique du modele poutre 1D :
#   alpha(s) = M_net / (q*A*x_c)   (M_live = q*A*x_c, x_c = abscisse barycentre)
#   (1) dalpha/ds = -alpha * (dx_c/ds) / x_c          (loi 1/x_c^2 ; = -M_net*L'/(qA x_c^2))
#   (2) dalpha/dm_L = -alpha                           (magnitude live = tout le live -> trivial)
#   (3) dalpha/dm_d = -M_dead/(q*A*x_c) -> dalpha/dm_d * x_c = const  (loi 1/x_c)
# -> valide : position+dead ensemble, position+live ensemble, chacune vs analytique.
#
#   C:\python3\python.exe tests\sweep_validation_analytique.py
# =====================================================================
import os, sys, json, re
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

FRONT = r"C:\workspace\front_mohamad"
CATALOG_ROOT = FRONT if os.path.isdir(os.path.join(FRONT, "STRAINS", "common", "Catalog")) else r"C:\workspace\front"
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

def _f(p):
    with open(p) as f: return f.read()
INITCATALOG(_f(os.path.join(CATALOG_ROOT, r"STRAINS\common\Catalog\CatalogTopo.json")),
            _f(os.path.join(CATALOG_ROOT, r"STRAINS\common\Catalog\CatalogDimensions.json")),
            _f(os.path.join(CATALOG_ROOT, r"STRAINS\common\Catalog\CatalogBolts.json")))

PATH = r"C:\workspace\storage\admin\Moulin_Blanc\Test_moving_load.ds"
ANALYSIS = 'Yield_analysis0'
DSCAD = os.path.join(PATH, 'dsCad.txt'); DSLOAD = os.path.join(PATH, 'dsLoad.txt')

# geometrie du path (barycentre) : cf dsLoad path_convoi = [(0.3,..),(3.7,..)]
XC0, XC1 = 0.3, 3.7
def xc_of_s(s):  return XC0 + s * (XC1 - XC0)   # abscisse barycentre
DXC_DS = XC1 - XC0                               # 3.4

REGIONS = [
    {"param": "LIVE_LOAD", "axis": "position", "load_case": "convoi"},   # dalpha/ds
    {"param": "LIVE_LOAD", "load_case": "convoi"},                        # dalpha/dm_L (magnitude live)
    {"param": "DEAD_LOAD", "load_case": "LC_poids"},                      # dalpha/dm_d (magnitude dead)
]

def patch(s, fy_top=550.0):
    c = open(DSCAD).read()
    c = re.sub(r'^fy_top\s*=.*$', 'fy_top    = %.10f' % fy_top, c, count=1, flags=re.MULTILINE)
    open(DSCAD, 'w').write(c)
    d = open(DSLOAD).read()
    d = re.sub(r'^s\s*=.*$', 's = %.10f' % s, d, count=1, flags=re.MULTILINE)
    open(DSLOAD, 'w').write(d)

def run(s):
    patch(s)
    model = MODEL(); SET_CONTEXT(model, PATH)
    exec(open(DSCAD).read(), globals())
    model.Save(os.path.join(PATH, ANALYSIS + ".dscad"))
    with CetLOAD.LOAD_MODEL(model, PATH):
        exec(open(DSLOAD).read(), globals())
    Mk = {"cadSurfOptions": {"volume_gradation":1.5,"gradation":1.5,"anisotropic_ratio":10},
          "tetraOptions": {"optimisation_level":"standard","verbose":"10"},
          "global_physical_size":0.05,"max_size":0.05,"min_size":"-1","gradation":1.5,"volume_gradation":1.5,
          "optimisation_level":"standard","anisotropic_ratio":"10","geometric_approximation_min":"4",
          "geometric_approximation_max":"25","geometric_approximation_on_edge":"false",
          "geometric_approximation_on_face":"true","use_surface_proximity":"false","surface_proximity_ratio":0,
          "approach":"kinematic","write_debug_files":"false","is_iso":"true","coeff_on_error":0.01,
          "remesh_type":1,"old_size_factor":0.0,"model_handle":model.GETHANDLEPTR()}
    CetMESH.ANISO_MESH(ANALYSIS, 0, PATH, **Mk)
    kwargs = {"scaling":1,"write_debug_files":"false"}
    exec(open(r"C:\workspace\fiabilite\InitSolver.py").read(), globals())
    kwargs.update(static_params=static_params, cinematic_params=cinematic_params, MKLPardiso_params=MKLPardiso_params,
                  MyPardiso_params=MyPardiso_params, MUMPS_params=MUMPS_params, FullLorentz=False, LorentzToSdp=False,
                  SdpToLorentz=0, printIntPointSolutioEvolution=False, trace_sur_point_integration=False,
                  calculate_error="false", max_nbOfDiv=0, customized_inc=[1], tetra_discontinuities=False,
                  activated_plasticity=True, welds_throat_limit=True, approach="kinematic")
    kwargs["sensitivity_analysis"] = "true"
    kwargs["sensitivity_regions"] = json.dumps(REGIONS)
    kwargs["model_handle"] = model.GETHANDLEPTR()
    CetSOLV.SOLV(ANALYSIS, 0, PATH, **kwargs)
    info = json.load(open(os.path.join(PATH, ANALYSIS + "_0_kine.dsmetares")))['info']
    sens = info.get("Sensitivity", {}) or {}
    return {"status": info.get("solver_status"), "lam": (info.get("Primal_bound") or [None])[0],
            "ds": sens.get("LIVE_LOAD:position:convoi"), "dmL": sens.get("LIVE_LOAD:convoi"),
            "dmd": sens.get("DEAD_LOAD:LC_poids")}

S_LIST = [0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90]
rows = []
print("\n############ SWEEP validation analytique (format MOVING_LOAD) ############")
for s in S_LIST:
    r = run(s)
    r["s"] = s; r["xc"] = xc_of_s(s); rows.append(r)
    print("s=%.2f | status=%s | alpha=%.5f | ds=%s | dmL=%s | dmd=%s"
          % (s, r["status"], r["lam"] or -1, r["ds"], r["dmL"], r["dmd"]))

# --- analytique auto-calibre (M_net = median(alpha*xc), pas de fit de M_p a la main) ---
import statistics
good = [r for r in rows if r["status"] == "OPTIMAL" and r["lam"]]
C = statistics.median([r["lam"] * r["xc"] for r in good])   # = M_net/(q*A) (loi alpha=C/xc)
Cd = statistics.median([-r["dmd"] * r["xc"] for r in good if r["dmd"] is not None])  # dead : -dmd*xc const

print("\n############ COMPARAISON SOLVEUR vs ANALYTIQUE ############")
print("C (=M_net/qA, alpha=C/xc) = %.5f | Cd (=-dmd*xc const) = %.5f\n" % (C, Cd))
print(" s     xc    | alpha  ~C/xc  | ds_solv  ds_ana=-C*L'/xc^2  ecart | dmL_solv -alpha  | dmd_solv  -Cd/xc  ecart")
def pct(a,b): return abs(a-b)/max(abs(b),1e-12)*100
for r in good:
    xc = r["xc"]; a_ana = C/xc
    ds_ana = -C * DXC_DS / xc**2
    dmL_ana = -r["lam"]
    dmd_ana = -Cd / xc
    print(" %.2f  %.3f | %.4f %.4f | %+.3f  %+.3f  %5.1f%% | %+.4f %+.4f | %+.4f  %+.4f  %5.1f%%"
          % (r["s"], xc, r["lam"], a_ana, r["ds"], ds_ana, pct(r["ds"], ds_ana),
             r["dmL"], dmL_ana, r["dmd"], dmd_ana, pct(r["dmd"], dmd_ana)))

# --- FIGURE (3 panneaux : position / live mag / dead mag) ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    ss = np.linspace(max(0.03, S_LIST[0]-0.02), 0.98, 200); xcc = XC0 + ss*(XC1-XC0)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    # (1) position
    ax[0].plot(ss, -C*DXC_DS/xcc**2, 'b-', label=r"analytique $-M_{net}\,L/(q A\,x_c^2)$")
    ax[0].plot([r["s"] for r in good], [r["ds"] for r in good], 'ro', label="solveur (KKT)")
    ax[0].set_title("POSITION  d$\\alpha$/ds"); ax[0].set_xlabel("s"); ax[0].set_ylabel("d$\\alpha$/ds"); ax[0].legend()
    # (2) live magnitude = -alpha
    ax[1].plot([r["s"] for r in good], [-r["lam"] for r in good], 'b-', label=r"analytique $-\alpha$")
    ax[1].plot([r["s"] for r in good], [r["dmL"] for r in good], 'ro', label="solveur (KKT)")
    ax[1].set_title("MAGNITUDE LIVE  d$\\alpha$/dm$_L$"); ax[1].set_xlabel("s"); ax[1].legend()
    # (3) dead magnitude ~ -Cd/xc
    ax[2].plot(ss, -Cd/xcc, 'b-', label=r"analytique $-M_{dead}/(qA\,x_c)$")
    ax[2].plot([r["s"] for r in good], [r["dmd"] for r in good], 'ro', label="solveur (KKT)")
    ax[2].set_title("MAGNITUDE DEAD  d$\\alpha$/dm$_d$"); ax[2].set_xlabel("s"); ax[2].legend()
    fig.suptitle("VALIDATION sensibilite analytique -- position + magnitudes ENSEMBLE (format MOVING_LOAD, sans shift)")
    fig.tight_layout()
    out = os.path.join(PATH, "sweep_validation_analytique.png")
    fig.savefig(out, dpi=110)
    print("\nFIGURE -> %s" % out)
except Exception as e:
    print("\n(matplotlib indispo : %r -- tableau ci-dessus suffit)" % e)

print("\n############ FIN SWEEP ############")
