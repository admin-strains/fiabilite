# -*- coding: utf-8 -*-
# =====================================================================
# Test CHANTIER 3 : POSITION par load case, VRAIE version MOVING_LOAD (2026-07-03, MM)
# =====================================================================
# Nouveau format (cf Test_moving_load.ds) : empreinte STATIQUE dans dsCad (aucun
# shift()), path defini SEPAREMENT par son nom dans dsLoad, MOVING_LOAD applique au
# LOAD_CASE entier (pas de variable= -> position_group = nom du LC), charge Z=-q.
#
# On valide le routage unifie :
#   {"param":"LIVE_LOAD","axis":"position","load_case":"convoi"} -> "LIVE_LOAD:position:convoi"
# doit :
#   (a) etre NON NUL (le convoi bouge -> dAlpha/ds != 0),
#   (b) == {"param":"LOAD_POSITION","group":"convoi"} ("LOAD_POSITION:convoi"), car les
#       deux lisent le MEME vecteur FEXT_dpos_per_grp["convoi"] (coherence du routage).
#
# Core ISOLE front_mohamad (chantier 3 + API MOVING_LOAD variable-optionnel). Lancer :
#   C:\python3\python.exe tests\test_position_par_load_case.py
# =====================================================================
import os, sys, json, re
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

FRONT = r"C:\workspace\front_mohamad"
CATALOG_ROOT = FRONT if os.path.isdir(os.path.join(FRONT, "STRAINS", "common", "Catalog")) \
                     else r"C:\workspace\front"
for d in [os.path.join(FRONT, r"STRAINS\rupt\core\bin"),
          os.path.join(FRONT, r"STRAINS\rupt\core"),
          os.path.join(FRONT, r"STRAINS\common\Dll"),
          os.path.join(FRONT, r"STRAINS\rupt\core\bin\meshgems"),
          os.path.join(FRONT, r"STRAINS\rupt\core\bin\mosek")]:
    if os.path.isdir(d):
        os.add_dll_directory(d)
sys.path.insert(0, FRONT)
sys.path.insert(0, r'C:\workspace\fiabilite')
from STRAINS.rupt.APIs.CetCAD_API import *
from STRAINS.rupt.APIs import CetLOAD
from STRAINS.rupt.APIs.CetLOAD_API import *
import STRAINS.rupt.core.CetMESH as CetMESH
from STRAINS.rupt.core import CetSOLV

def _getFile(p):
    with open(p, 'r') as f:
        return f.read()
INITCATALOG(_getFile(os.path.join(CATALOG_ROOT, r"STRAINS\common\Catalog\CatalogTopo.json")),
            _getFile(os.path.join(CATALOG_ROOT, r"STRAINS\common\Catalog\CatalogDimensions.json")),
            _getFile(os.path.join(CATALOG_ROOT, r"STRAINS\common\Catalog\CatalogBolts.json")))

PATH = r"C:\workspace\storage\admin\Moulin_Blanc\Test_moving_load.ds"
ANALYSIS = 'Yield_analysis0'
DSCAD = os.path.join(PATH, 'dsCad.txt')

R_POS_LC  = {"param": "LIVE_LOAD", "axis": "position", "load_case": "convoi"}  # nouveau routage
R_POS_GRP = {"param": "LOAD_POSITION", "group": "convoi"}                      # cross-check (meme vecteur)

def patch_cad_stable():
    c = open(DSCAD).read()
    c = re.sub(r'^fy_top\s*=.*$', 'fy_top    = 550.0000000000', c, count=1, flags=re.MULTILINE)
    open(DSCAD, 'w').write(c)

def run(regions):
    patch_cad_stable()
    model = MODEL(); SET_CONTEXT(model, PATH)
    exec(open(DSCAD).read(), globals())
    model.Save(os.path.join(PATH, ANALYSIS + ".dscad"))
    with CetLOAD.LOAD_MODEL(model, PATH):
        exec(open(os.path.join(PATH, 'dsLoad.txt')).read(), globals())
    Meshkwargs = {
        "cadSurfOptions": {"volume_gradation": 1.5, "gradation": 1.5, "anisotropic_ratio": 10},
        "tetraOptions": {"optimisation_level": "standard", "verbose": "10"},
        "global_physical_size": 0.05, "max_size": 0.05, "min_size": "-1",
        "gradation": 1.5, "volume_gradation": 1.5, "optimisation_level": "standard",
        "anisotropic_ratio": "10", "geometric_approximation_min": "4",
        "geometric_approximation_max": "25", "geometric_approximation_on_edge": "false",
        "geometric_approximation_on_face": "true", "use_surface_proximity": "false",
        "surface_proximity_ratio": 0, "approach": "kinematic", "write_debug_files": "true",
        "is_iso": "true", "coeff_on_error": 0.01, "remesh_type": 1, "old_size_factor": 0.0,
        "model_handle": model.GETHANDLEPTR(),
    }
    CetMESH.ANISO_MESH(ANALYSIS, 0, PATH, **Meshkwargs)
    kwargs = {"scaling": 1, "write_debug_files": "true"}
    exec(open(r"C:\workspace\fiabilite\InitSolver.py").read(), globals())
    kwargs.update(static_params=static_params, cinematic_params=cinematic_params,
                  MKLPardiso_params=MKLPardiso_params, MyPardiso_params=MyPardiso_params,
                  MUMPS_params=MUMPS_params, FullLorentz=False, LorentzToSdp=False, SdpToLorentz=0,
                  printIntPointSolutioEvolution=False, trace_sur_point_integration=False,
                  calculate_error="false", max_nbOfDiv=0, customized_inc=[1],
                  tetra_discontinuities=False, activated_plasticity=True, welds_throat_limit=True,
                  approach="kinematic")
    kwargs["sensitivity_analysis"] = "true"
    kwargs["sensitivity_regions"] = json.dumps(regions)
    kwargs["model_handle"] = model.GETHANDLEPTR()
    CetSOLV.SOLV(ANALYSIS, 0, PATH, **kwargs)
    d = json.load(open(os.path.join(PATH, ANALYSIS + "_0_kine.dsmetares")))
    info = d['info']
    return {
        "status": info.get("solver_status"),
        "lam": (info.get("Primal_bound") or [None])[0],
        "sens": info.get("Sensitivity", {}) or {},
    }

print("\n############ TEST POSITION par load case (MOVING_LOAD, sans shift) ############")
print("Core :", FRONT, "| modele :", PATH)

res = run([R_POS_LC, R_POS_GRP])
print("\n   status =", res["status"], "| lambda =", res["lam"])
print("   sens   =", res["sens"])

pos_lc  = res["sens"].get("LIVE_LOAD:position:convoi")
pos_grp = res["sens"].get("LOAD_POSITION:convoi")
ok = True

def rel(a, b): return abs(a - b) / max(abs(b), 1e-12) * 100.0

c0 = res["status"] == "OPTIMAL" and res["lam"] is not None
print(("OK " if c0 else "!! ECHEC ") + ": run converge (status=%s)" % res["status"]); ok &= c0

c1 = pos_lc is not None
print(("OK " if c1 else "!! ECHEC ") + ": cle LIVE_LOAD:position:convoi presente (=%s)" % pos_lc); ok &= c1

if pos_lc is not None:
    c2 = abs(pos_lc) > 1e-6
    print(("OK " if c2 else "!! ECHEC ") +
          ": LIVE_LOAD:position:convoi NON NUL (le convoi bouge) = %.6e" % pos_lc); ok &= c2

if pos_lc is not None and pos_grp is not None:
    e = rel(pos_lc, pos_grp)
    c3 = e < 0.5
    print(("OK " if c3 else "!! ECHEC ") +
          ": LIVE_LOAD:position:convoi == LOAD_POSITION:convoi (%.6e vs %.6e, %.4f%%)"
          % (pos_lc, pos_grp, e)); ok &= c3

assert ok, "ECHEC : routage position par load case (MOVING_LOAD) incorrect"
print("\n############ CHANTIER 3 VALIDE : position par load case (MOVING_LOAD, sans shift) ############")
