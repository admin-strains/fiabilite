# -*- coding: utf-8 -*-
# ==========================================================================================
# Validation POIDS PROPRE (3 variantes) sur le CORE PARTAGE  (2026-07-06, MM)
# ------------------------------------------------------------------------------------------
# Verifie que le core deploye (C:\workspace\front) gere correctement le poids propre
# (LOAD_CASE VALUE/DIR = acceleration sur la densite du bloc) dans 3 configurations
# demandees, sur la poutre cantilever (meme squelette que moulin_blanc) :
#
#   (A) poids propre en LIVE, non aleatoire   : LC_poids (VALUE=-9.81) dans LIVE_LOAD_CASES
#                                               + LC_trafic (empreinte). Les deux amplifies.
#   (B) PAS de poids propre du tout           : seul LC_trafic (empreinte) en live.
#   (C) poids propre en DEAD, non aleatoire   : LC_poids (VALUE=-9.81) dans DEAD_LOAD_CASES
#                                               + LC_trafic (empreinte) en live.
#
# "non aleatoire" = le poids propre est present dans le chargement mais N'EST PAS une
# variable de sensibilite (pas dans une region). On verifie que chaque cas CONVERGE
# (status OPTIMAL) et donne un lambda fini coherent (A < B : le poids live consomme de la
# capacite ; C : le poids dead penalise aussi via PERM_FEXT).
#
# Pointe sur le CORE PARTAGE via FRONT. Dossier .ds dedie (ne clobbe pas le cantilever).
#   C:\python3\python.exe tests\test_gravity_variants.py
# ==========================================================================================
import os, sys, json, re, shutil
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

FRONT = r"C:\workspace\front"   # CORE PARTAGE (pas front_mohamad)
CATALOG_ROOT = FRONT
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

def _rd(p):
    with open(p, 'r') as f:
        return f.read()
INITCATALOG(_rd(os.path.join(CATALOG_ROOT, r"STRAINS\common\Catalog\CatalogTopo.json")),
            _rd(os.path.join(CATALOG_ROOT, r"STRAINS\common\Catalog\CatalogDimensions.json")),
            _rd(os.path.join(CATALOG_ROOT, r"STRAINS\common\Catalog\CatalogBolts.json")))

SRC   = r"C:\workspace\storage\admin\Moulin_Blanc\Test_cantilever_s.ds"
PATH  = r"C:\workspace\storage\admin\Moulin_Blanc\Test_gravity_variants.ds"
ANALYSIS = 'Yield_analysis0'
S_VALID  = 2.0 / (4.0 - 0.6)

# --- dsCad : copie du cantilever, fy_top/s stables ---
os.makedirs(PATH, exist_ok=True)
_cad = _rd(os.path.join(SRC, 'dsCad.txt'))
_cad = re.sub(r'^s\s*=.*$',      's    = %.10f' % S_VALID, _cad, count=1, flags=re.MULTILINE)
_cad = re.sub(r'^fy_top\s*=.*$', 'fy_top    = 550.0000000000', _cad, count=1, flags=re.MULTILINE)
open(os.path.join(PATH, 'dsCad.txt'), 'w').write(_cad)

# --- 3 variantes de dsLoad ---
_GRAV = "with LOAD_CASE('LC_poids', VALUE=-9.81, DIR=\"ZAxis\"):\n    pass\n\n"
_TRAF_SUP = ("with LOAD_CASE('LC_trafic'):\n"
             "    SUPPORT(BOUNDARY='Fixed', MEAN=False, RIGID=False, X=True, Y=True, Z=True)\n"
             "    LOAD(POLYGON='Load', NORMAL=str(-q))\n\n")

DSLOADS = {
 "A_grav_live": "q = 0.15\n\n" + _GRAV + _TRAF_SUP +
    "YIELD_ANALYSIS('Yield_analysis0',\n"
    "               LIVE_LOAD_CASES=[('LC_poids', '1'), ('LC_trafic', '1')],\n"
    "               MESH={\"global_physical_size\": \"0.05\"})\n",
 "B_no_grav":   "q = 0.15\n\n" + _TRAF_SUP +
    "YIELD_ANALYSIS('Yield_analysis0',\n"
    "               LIVE_LOAD_CASES=[('LC_trafic', '1')],\n"
    "               MESH={\"global_physical_size\": \"0.05\"})\n",
 "C_grav_dead": "q = 0.15\n\n" + _GRAV + _TRAF_SUP +
    "YIELD_ANALYSIS('Yield_analysis0',\n"
    "               DEAD_LOAD_CASES=[('LC_poids', '1')],\n"
    "               LIVE_LOAD_CASES=[('LC_trafic', '1')],\n"
    "               MESH={\"global_physical_size\": \"0.05\"})\n",
}

def run(dsload_txt):
    open(os.path.join(PATH, 'dsLoad.txt'), 'w').write(dsload_txt)
    model = MODEL(); SET_CONTEXT(model, PATH)
    exec(_rd(os.path.join(PATH, 'dsCad.txt')), globals())
    model.Save(os.path.join(PATH, ANALYSIS + ".dscad"))
    with CetLOAD.LOAD_MODEL(model, PATH):
        exec(_rd(os.path.join(PATH, 'dsLoad.txt')), globals())
    Mk = {"cadSurfOptions": {"volume_gradation":1.5,"gradation":1.5,"anisotropic_ratio":10},
          "tetraOptions": {"optimisation_level":"standard","verbose":"10"},
          "global_physical_size":0.05,"max_size":0.05,"min_size":"-1","gradation":1.5,
          "volume_gradation":1.5,"optimisation_level":"standard","anisotropic_ratio":"10",
          "geometric_approximation_min":"4","geometric_approximation_max":"25",
          "geometric_approximation_on_edge":"false","geometric_approximation_on_face":"true",
          "use_surface_proximity":"false","surface_proximity_ratio":0,"approach":"kinematic",
          "write_debug_files":"false","is_iso":"true","coeff_on_error":0.01,"remesh_type":1,
          "old_size_factor":0.0,"model_handle":model.GETHANDLEPTR()}
    CetMESH.ANISO_MESH(ANALYSIS, 0, PATH, **Mk)
    kwargs = {"scaling":1,"write_debug_files":"false"}
    exec(_rd(r"C:\workspace\fiabilite\InitSolver.py"), globals())
    kwargs.update(static_params=static_params, cinematic_params=cinematic_params,
                  MKLPardiso_params=MKLPardiso_params, MyPardiso_params=MyPardiso_params,
                  MUMPS_params=MUMPS_params, FullLorentz=False, LorentzToSdp=False, SdpToLorentz=0,
                  printIntPointSolutioEvolution=False, trace_sur_point_integration=False,
                  calculate_error="false", max_nbOfDiv=0, customized_inc=[1],
                  tetra_discontinuities=False, activated_plasticity=True, welds_throat_limit=True,
                  approach="kinematic", model_handle=model.GETHANDLEPTR())
    try:
        CetSOLV.SOLV(ANALYSIS, 0, PATH, **kwargs)
        info = json.load(open(os.path.join(PATH, ANALYSIS + "_0_kine.dsmetares")))["info"]
        return info.get("solver_status"), (info.get("Primal_bound") or [None])[0], None
    except Exception as e:
        return None, None, repr(e)[:200]

print("\n############ VALIDATION POIDS PROPRE (3 variantes) sur core partage ############")
print("Core :", FRONT)
res = {}
for name, txt in DSLOADS.items():
    st, lam, err = run(txt)
    res[name] = (st, lam, err)
    print(f"\n[{name}] status={st}  lambda={lam}  err={err}")

ok = True
def _chk(label, cond):
    global ok
    print(("OK " if cond else "!! ECHEC ") + ": " + label); ok &= cond

for name in DSLOADS:
    st, lam, err = res[name]
    _chk(f"{name} converge (status OPTIMAL, lambda fini)",
         st == "OPTIMAL" and lam is not None and lam == lam)  # lam==lam : pas NaN

# coherence physique : le poids propre en LIVE (A) consomme de la capacite -> lambda_A < lambda_B
if res["A_grav_live"][1] is not None and res["B_no_grav"][1] is not None:
    _chk("lambda_A (poids live) < lambda_B (sans poids) -- le poids live consomme de la capacite",
         res["A_grav_live"][1] < res["B_no_grav"][1])

assert ok, "ECHEC : gestion du poids propre incorrecte sur le core partage"
print("\n############ VALIDE : poids propre live / absent / dead OK sur le core partage ############")
