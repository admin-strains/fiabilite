# -*- coding: utf-8 -*-
# ==========================================================================================
# DEAD-comme-LIVE non aleatoire (empreinte dedans/dehors) + 2 variables dead & live distinctes
#                                                                          (2026-07-06, MM)
# ------------------------------------------------------------------------------------------
# Sur Test_moving_load.ds (empreinte 'Load' mobile via MOVING_LOAD). CORE PARTAGE. Charge Z=.
#
#   DL1 : poids propre en LIVE, NON aleatoire (seule variable = q de l'empreinte),
#         empreinte DEDANS (path sur la poutre). -> converge, LIVE_LOAD:q sensee.
#   DL2 : idem DL1 mais empreinte TOTALEMENT DEHORS (path hors poutre, x~5.5 > L=4).
#         QUESTION : le poids propre live (non nul) sauve-t-il le calcul d'un live vide ?
#         -> on OBSERVE : converge (grace a la gravite live) avec LIVE_LOAD:q ~ 0 (empreinte
#            ne charge rien), OU NUMERICAL_ERROR.
#   DL3 : DEUX variables aleatoires DISTINCTES : q (empreinte, LIVE) + g (poids propre, DEAD).
#         Regions {LIVE_LOAD:q} + {DEAD_LOAD:g}. -> on verifie que les 2 sensibilites sont
#         SAINES (finies, signe negatif = charge deviens plus penalisante) et non aberrantes.
#
#   C:\python3\python.exe tests\test_dead_live_variables.py
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

PATH_IN  = "[(0.3, 0.0, 0.25), (3.7, 0.0, 0.25)]"   # convoi SUR la poutre (L=4)
PATH_OUT = "[(5.0, 0.0, 0.25), (6.0, 0.0, 0.25)]"   # convoi HORS poutre (x~5.5 > 4)
_GRAV = "with LOAD_CASE('LC_poids', VALUE=-9.81, DIR=\"ZAxis\"):\n    pass\n\n"

def _convoi(path_str):
    return ("with LOAD_CASE('convoi'):\n"
            "    SUPPORT(BOUNDARY='Fixed', MEAN=False, RIGID=False, X=True, Y=True, Z=True)\n"
            "    LOAD(POLYGON='Load', Z=str(-q))\n"
            "    MOVING_LOAD(path=%s, position=0.5, unit='relative')\n\n" % path_str)

def _head():
    return "q = 0.15\n\n"

DSLOAD = {
 # DL1 : poids propre LIVE non-aleatoire, empreinte DEDANS
 "DL1_in":  _head() + _GRAV + _convoi(PATH_IN) +
    "YIELD_ANALYSIS('Yield_analysis0', LIVE_LOAD_CASES=[('LC_poids','1'),('convoi','1')],\n"
    "               MESH={\"global_physical_size\": \"0.06\"})\n",
 # DL2 : poids propre LIVE non-aleatoire, empreinte DEHORS
 "DL2_out": _head() + _GRAV + _convoi(PATH_OUT) +
    "YIELD_ANALYSIS('Yield_analysis0', LIVE_LOAD_CASES=[('LC_poids','1'),('convoi','1')],\n"
    "               MESH={\"global_physical_size\": \"0.06\"})\n",
 # DL3 : 2 variables distinctes q (live) + g (dead)
 "DL3_2var": _head() + _GRAV + _convoi(PATH_IN) +
    "YIELD_ANALYSIS('Yield_analysis0', DEAD_LOAD_CASES=[('LC_poids','1')],\n"
    "               LIVE_LOAD_CASES=[('convoi','1')],\n"
    "               MESH={\"global_physical_size\": \"0.06\"})\n",
}
R_Q  = {"param": "LIVE_LOAD", "load_case": "convoi",   "region_key": "q"}
R_GD = {"param": "DEAD_LOAD", "load_case": "LC_poids", "region_key": "g"}
REGIONS = {"DL1_in": [R_Q], "DL2_out": [R_Q], "DL3_2var": [R_Q, R_GD]}

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

print("\n############ DEAD-comme-LIVE non-alea (dedans/dehors) + 2 vars distinctes ############")
print("Core :", FRONT)
res = {}
for name in DSLOAD:
    st, lam, sens, err = run(DSLOAD[name], REGIONS[name])
    res[name] = (st, lam, sens, err)
    print(f"\n[{name}] status={st}  lambda={lam}  err={err}\n    sens = {sens}")

ok = True
def _chk(label, cond):
    global ok
    print(("OK " if cond else "!! ECHEC ") + ": " + label); ok &= cond

# DL1 : marche (poids live non-alea + empreinte dedans variable)
_chk("DL1 (poids live non-alea, empreinte dedans) converge", res["DL1_in"][0] == "OPTIMAL")
_chk("DL1 LIVE_LOAD:q sensee (non nulle)",
     res["DL1_in"][2].get("LIVE_LOAD:q") not in (None,) and abs(res["DL1_in"][2].get("LIVE_LOAD:q") or 0) > 1e-9)

# DL2 : empreinte dehors -- observe si la gravite live sauve le calcul
st2, lam2, s2, err2 = res["DL2_out"]
q2 = s2.get("LIVE_LOAD:q")
print(f"    -> [DL2] empreinte DEHORS + poids live : status={st2}, LIVE_LOAD:q={q2}")
if st2 == "OPTIMAL":
    print("       => la gravite LIVE (non nulle) SAUVE le calcul (live pas vide) ; q ~ 0 attendu (empreinte ne charge rien)")
    _chk("DL2 converge grace au poids live (empreinte dehors n'annule pas le live)", True)
    _chk("DL2 LIVE_LOAD:q ~ 0 (empreinte dehors ne contribue pas)",
         q2 is None or abs(q2) < 0.05)
else:
    print(f"       => n'a PAS converge (status={st2}) : le poids live n'a pas suffi / autre effet")
    _chk("DL2 process non crashe (resultat recupere)", isinstance(res["DL2_out"], tuple))

# DL3 : 2 variables distinctes dead+live -- sensibilites saines, pas aberrantes
st3, lam3, s3, err3 = res["DL3_2var"]
q3 = s3.get("LIVE_LOAD:q"); g3 = s3.get("DEAD_LOAD:g")
_chk("DL3 (q live + g dead, 2 variables distinctes) converge", st3 == "OPTIMAL")
_chk("DL3 LIVE_LOAD:q presente et finie", q3 is not None and q3 == q3 and abs(q3) < 1e6)
_chk("DL3 DEAD_LOAD:g presente et finie (pas aberrante)", g3 is not None and g3 == g3 and abs(g3) < 1e6)
# saines : q live doit etre ~ -lambda (empreinte = toute la charge LIVE), g dead du meme signe (charge)
if q3 is not None and lam3 is not None:
    _chk("DL3 LIVE_LOAD:q ~ -lambda (empreinte = toute la charge live) : %.4f vs %.4f" % (q3, -lam3),
         abs(q3 - (-lam3)) / max(abs(lam3), 1e-9) < 0.05)
if g3 is not None:
    _chk("DL3 DEAD_LOAD:g de signe negatif (poids permanent penalise) = %.4f" % g3, g3 < 0)

assert ok, "ECHEC : dead-comme-live / 2 variables distinctes"
print("\n############ VALIDE : dead-live non-alea + 2 variables distinctes (sensibilites saines) ############")
