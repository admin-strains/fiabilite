# -*- coding: utf-8 -*-
# =====================================================================
# Test CHANTIER 1 : footprint capte par LIVE_LOAD:<load_case>  (2026-07-03, MM)
# =====================================================================
# Verifie l'unification "magnitude par load case" : un footprint defini dans
# un load case LIVE (ici 'LC_trafic') doit desormais etre accumule dans
# FEXT_per_LC[iLC] -> la sensibilite LIVE_LOAD sur ce load case capte le footprint.
#
# Discriminant :
#   - AVANT le chantier 1 : FEXT_per_LC['LC_trafic'] ne contenait QUE les surfaces
#     (aucune ici) -> LIVE_LOAD:LC_trafic ~ 0 (footprint ignore).
#   - APRES : FEXT_per_LC['LC_trafic'] contient le footprint -> LIVE_LOAD:LC_trafic
#     == LOAD_MAGNITUDE:qmag (meme vecteur FEXT_fp pour ce cas ou footprint = tout
#     le live), et ~ -lambda_0 (footprint = toute la charge live).
#
# Core ISOLE front_mohamad (contient le chantier 1). Lancer :
#   C:\python3\python.exe tests\test_footprint_par_load_case.py
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

PATH = r"C:\workspace\storage\admin\Moulin_Blanc\Test_cantilever_s.ds"
ANALYSIS = 'Yield_analysis0'
DSCAD = os.path.join(PATH, 'dsCad.txt')
S_VALID = 2.0 / (4.0 - 0.6)

# footprint capte via son load case (LOAD_MAGNITUDE retire -> schema unifie LIVE_LOAD+load_case)
R_LIVE = {"param": "LIVE_LOAD", "load_case": "LC_trafic"}   # cle sortie = "LIVE_LOAD:LC_trafic"

def patch_cad_stable():
    c = open(DSCAD).read()
    c = re.sub(r'^s\s*=.*$', 's    = %.10f' % S_VALID, c, count=1, flags=re.MULTILINE)
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

print("\n############ TEST footprint capte par LIVE_LOAD:<load_case> ############")
print("Core :", FRONT)

res = run([R_LIVE])
print("\n   status =", res["status"], "| lambda =", res["lam"])
print("   sens   =", res["sens"])

live = res["sens"].get("LIVE_LOAD:LC_trafic")
lam  = res["lam"]
ok = True

def rel(a, b): return abs(a - b) / max(abs(b), 1e-12) * 100.0

c0 = res["status"] == "OPTIMAL" and lam is not None
print(("OK " if c0 else "!! ECHEC ") + ": run converge (status=%s)" % res["status"]); ok &= c0

c1 = live is not None
print(("OK " if c1 else "!! ECHEC ") + ": cle LIVE_LOAD:LC_trafic presente (=%s)" % live); ok &= c1

if live is not None and lam is not None:
    e = rel(live, -lam)
    cok = e < 1.0   # footprint = tout le live -> dLambda/dm = -lambda_0
    print(("OK " if cok else "!! ECHEC ") +
          ": LIVE_LOAD:LC_trafic ~ -lambda_0 (=%.4f, %.4f%%)" % (-lam, e)); ok &= cok

assert ok, "ECHEC : le footprint n'est pas correctement capte par LIVE_LOAD:LC_trafic"
print("\n############ CHANTIER 1 VALIDE : footprint capte par son load case ############")
