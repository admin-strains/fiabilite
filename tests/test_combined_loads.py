# -*- coding: utf-8 -*-
# ==========================================================================================
# Cas COMBINES : empreinte (norme + position) comme variables + poids propre  (2026-07-06, MM)
# ------------------------------------------------------------------------------------------
# Sur Test_moving_load.ds (empreinte 'Load' MOBILE = seul live via MOVING_LOAD). CORE PARTAGE.
# On demande PLUSIEURS sensibilites dans le MEME solve et on OBSERVE :
#
#   C1 : empreinte seule live, 2 variables = norme q (magnitude) + position s ;
#        poids propre en DEAD (non aleatoire, pas de region).
#        -> cles "LIVE_LOAD:q" ET "LIVE_LOAD:s" attendues non nulles.
#   C2 : idem C1 mais poids propre en LIVE (AMPLIFIE) et TOUJOURS non aleatoire
#        (les 2 seules variables restent q + s de l'empreinte).
#   C3 : 2 variables aleatoires, LES DEUX en LIVE : norme q de l'empreinte + poids propre g
#        (comme sa PROPRE variable, LC_poids en live). -> on OBSERVE si "LIVE_LOAD:g"
#        (gravite dans un LC live) est capte, et on discute lois q vs g (voir NOTE en bas).
#
#   C:\python3\python.exe tests\test_combined_loads.py
# ==========================================================================================
import os, sys, json, re
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

FRONT = r"C:\workspace\front"   # CORE PARTAGE
CATALOG_ROOT = FRONT
for d in [os.path.join(FRONT, r"STRAINS\rupt\core\bin"), os.path.join(FRONT, r"STRAINS\rupt\core"),
          os.path.join(FRONT, r"STRAINS\common\Dll"), os.path.join(FRONT, r"STRAINS\rupt\core\bin\meshgems"),
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

PATH = r"C:\workspace\storage\admin\Moulin_Blanc\Test_moving_load.ds"
ANALYSIS = 'Yield_analysis0'
DSCAD = os.path.join(PATH, 'dsCad.txt')

_HEAD = ("q = 0.15\n"
         "path_convoi = [(0.3, 0.0, 0.25), (3.7, 0.0, 0.25)]\n"
         "s = 0.5882352941\n\n")
_CONVOI = ("with LOAD_CASE('convoi'):\n"
           "    SUPPORT(BOUNDARY='Fixed', MEAN=False, RIGID=False, X=True, Y=True, Z=True)\n"
           "    LOAD(POLYGON='Load', Z=str(-q))\n"
           "    MOVING_LOAD(path=path_convoi, position=s, unit='relative')\n\n")
_GRAV = "with LOAD_CASE('LC_poids', VALUE=-9.81, DIR=\"ZAxis\"):\n    pass\n\n"

DSLOAD = {
 "C1_grav_dead": _HEAD + _GRAV + _CONVOI +
    "YIELD_ANALYSIS('Yield_analysis0',\n"
    "               DEAD_LOAD_CASES=[('LC_poids', '1')],\n"
    "               LIVE_LOAD_CASES=[('convoi', '1')],\n"
    "               MESH={\"global_physical_size\": \"0.06\"})\n",
 "C2_grav_live": _HEAD + _GRAV + _CONVOI +
    "YIELD_ANALYSIS('Yield_analysis0',\n"
    "               LIVE_LOAD_CASES=[('LC_poids', '1'), ('convoi', '1')],\n"
    "               MESH={\"global_physical_size\": \"0.06\"})\n",
 "C3_two_live":  _HEAD + _GRAV + _CONVOI +
    "YIELD_ANALYSIS('Yield_analysis0',\n"
    "               LIVE_LOAD_CASES=[('LC_poids', '1'), ('convoi', '1')],\n"
    "               MESH={\"global_physical_size\": \"0.06\"})\n",
}
# regions de sensibilite par config
R_Q = {"param": "LIVE_LOAD", "load_case": "convoi", "region_key": "q"}                       # norme empreinte
R_S = {"param": "LIVE_LOAD", "axis": "position", "load_case": "convoi", "region_key": "s"}    # position empreinte
R_G = {"param": "LIVE_LOAD", "load_case": "LC_poids", "region_key": "g"}                      # poids propre (live)
REGIONS = {"C1_grav_dead": [R_Q, R_S], "C2_grav_live": [R_Q, R_S], "C3_two_live": [R_Q, R_G]}

def patch_cad_stable():
    c = open(DSCAD).read()
    c = re.sub(r'^fy_top\s*=.*$', 'fy_top    = 550.0000000000', c, count=1, flags=re.MULTILINE)
    open(DSCAD, 'w').write(c)

def run(dsload_txt, regions):
    patch_cad_stable()
    open(os.path.join(PATH, 'dsLoad.txt'), 'w').write(dsload_txt)
    model = MODEL(); SET_CONTEXT(model, PATH)
    exec(_rd(DSCAD), globals())
    model.Save(os.path.join(PATH, ANALYSIS + ".dscad"))
    with CetLOAD.LOAD_MODEL(model, PATH):
        exec(_rd(os.path.join(PATH, 'dsLoad.txt')), globals())
    Mk = {"cadSurfOptions": {"volume_gradation":1.5,"gradation":1.5,"anisotropic_ratio":10},
          "tetraOptions": {"optimisation_level":"standard","verbose":"10"},
          "global_physical_size":0.06,"max_size":0.06,"min_size":"-1","gradation":1.5,
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
    kwargs["sensitivity_analysis"] = "true"
    kwargs["sensitivity_regions"] = json.dumps(regions)
    try:
        CetSOLV.SOLV(ANALYSIS, 0, PATH, **kwargs)
        info = json.load(open(os.path.join(PATH, ANALYSIS + "_0_kine.dsmetares")))["info"]
        return info.get("solver_status"), (info.get("Primal_bound") or [None])[0], info.get("Sensitivity", {}) or {}, None
    except Exception as e:
        return None, None, {}, repr(e)[:200]

print("\n############ CAS COMBINES : empreinte (q + s) + poids propre ############")
print("Core :", FRONT)
res = {}
for name in DSLOAD:
    st, lam, sens, err = run(DSLOAD[name], REGIONS[name])
    res[name] = (st, lam, sens, err)
    print(f"\n[{name}] status={st}  lambda={lam}  err={err}")
    print(f"    sens = {sens}")

ok = True
def _chk(label, cond):
    global ok
    print(("OK " if cond else "!! ECHEC ") + ": " + label); ok &= cond

# C1 : les 2 variables empreinte (q magnitude + s position) sortent non nulles, converge
s1 = res["C1_grav_dead"][2]
_chk("C1 converge", res["C1_grav_dead"][0] == "OPTIMAL")
_chk("C1 LIVE_LOAD:q (norme) presente et non nulle",
     s1.get("LIVE_LOAD:q") not in (None,) and abs(s1.get("LIVE_LOAD:q") or 0) > 1e-9)
_chk("C1 LIVE_LOAD:s (position) presente et non nulle",
     s1.get("LIVE_LOAD:s") not in (None,) and abs(s1.get("LIVE_LOAD:s") or 0) > 1e-9)

# C2 : poids propre LIVE (amplifie) mais variables = q + s uniquement ; converge, q+s presentes
s2 = res["C2_grav_live"][2]
_chk("C2 converge (poids propre amplifie en live)", res["C2_grav_live"][0] == "OPTIMAL")
_chk("C2 LIVE_LOAD:q + LIVE_LOAD:s toujours presentes",
     s2.get("LIVE_LOAD:q") is not None and s2.get("LIVE_LOAD:s") is not None)
# le poids propre amplifie penalise -> lambda C2 < lambda C1 (dead non amplifie)
if res["C1_grav_dead"][1] and res["C2_grav_live"][1]:
    _chk("lambda C2 (poids live) < lambda C1 (poids dead)",
         res["C2_grav_live"][1] < res["C1_grav_dead"][1])

# C3 : 2 variables live (q empreinte + g poids propre). OBSERVATION de LIVE_LOAD:g.
s3 = res["C3_two_live"][2]
_chk("C3 converge", res["C3_two_live"][0] == "OPTIMAL")
_chk("C3 LIVE_LOAD:q presente", s3.get("LIVE_LOAD:q") is not None)
_g = s3.get("LIVE_LOAD:g")
print(f"    -> [C3] OBSERVATION LIVE_LOAD:g (poids propre en LC live) = {_g}"
      f"  ({'CAPTE non nul' if (_g not in (None,) and abs(_g or 0) > 1e-9) else 'nul/absent -> gravite live non captee comme sensibilite'})")

assert ok, "ECHEC : cas combines empreinte + poids propre"
print("\n############ VALIDE : q + s empreinte + poids propre (dead/live) ############")
print("\nNOTE lois q vs g : le poids propre et la charge de trafic ont des VARIABILITES")
print("differentes (poids propre ~ Normal COV faible 5-10%%, trafic ~ COV plus large / autre loi).")
print("=> lois DIFFERENTES, variables SEPAREES par defaut. Les regrouper sous un meme facteur")
print("(load_case=[...]) n'a de sens que si c'est la MEME source physique.")
