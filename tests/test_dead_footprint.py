# -*- coding: utf-8 -*-
# =====================================================================
# Test CHANTIER 3b : footprint DEAD (magnitude + position) + separation live/dead (2026-07-03, MM)
# =====================================================================
# Modele Test_moving_load_dead.ds : LIVE = footprint 'LiveLoad' fixe (amplifie) ;
# DEAD = footprint 'DeadLoad' permanent MOBILE (MOVING_LOAD dans un DEAD load case 'deadconvoi').
# Valide :
#   (1) DEAD_LOAD:deadconvoi           (magnitude dead footprint)  -> NON NUL
#   (2) DEAD_LOAD:position:deadconvoi  (position dead footprint, type=3) -> NON NUL
#   (3) SEPARATION live/dead : LIVE_LOAD:position:deadconvoi ~ 0 (le footprint dead n'est PAS
#       dans les vecteurs live -> il ne "fuit" pas dans le FEXT live).
#
#   C:\python3\python.exe tests\test_dead_footprint.py
# =====================================================================
import os, sys, json, re
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
FRONT = r"C:\workspace\front_mohamad"
CAT = FRONT if os.path.isdir(os.path.join(FRONT,"STRAINS","common","Catalog")) else r"C:\workspace\front"
for d in [os.path.join(FRONT,r"STRAINS\rupt\core\bin"),os.path.join(FRONT,r"STRAINS\rupt\core"),
          os.path.join(FRONT,r"STRAINS\common\Dll"),os.path.join(FRONT,r"STRAINS\rupt\core\bin\meshgems"),
          os.path.join(FRONT,r"STRAINS\rupt\core\bin\mosek")]:
    if os.path.isdir(d): os.add_dll_directory(d)
sys.path.insert(0,FRONT); sys.path.insert(0,r'C:\workspace\fiabilite')
from STRAINS.rupt.APIs.CetCAD_API import *
from STRAINS.rupt.APIs import CetLOAD
from STRAINS.rupt.APIs.CetLOAD_API import *
import STRAINS.rupt.core.CetMESH as CetMESH
from STRAINS.rupt.core import CetSOLV
def _f(p):
    with open(p) as f: return f.read()
INITCATALOG(_f(os.path.join(CAT,r"STRAINS\common\Catalog\CatalogTopo.json")),
            _f(os.path.join(CAT,r"STRAINS\common\Catalog\CatalogDimensions.json")),
            _f(os.path.join(CAT,r"STRAINS\common\Catalog\CatalogBolts.json")))
PATH=r"C:\workspace\storage\admin\Moulin_Blanc\Test_moving_load_dead.ds"; AN='Yield_analysis0'
DSCAD=os.path.join(PATH,'dsCad.txt')

R_DEAD_MAG = {"param":"DEAD_LOAD", "load_case":"deadconvoi"}                    # -> DEAD_LOAD:deadconvoi
R_DEAD_POS = {"param":"DEAD_LOAD", "axis":"position", "load_case":"deadconvoi"} # -> DEAD_LOAD:position:deadconvoi
R_LIVE_POS = {"param":"LIVE_LOAD", "axis":"position", "load_case":"deadconvoi"} # separation -> ~0

def patch():
    c=open(DSCAD).read()
    c=re.sub(r'^fy_top\s*=.*$','fy_top    = 550.0000000000',c,count=1,flags=re.MULTILINE)
    open(DSCAD,'w').write(c)
def run(regions):
    patch()
    model=MODEL(); SET_CONTEXT(model,PATH); exec(open(DSCAD).read(),globals())
    model.Save(os.path.join(PATH,AN+".dscad"))
    with CetLOAD.LOAD_MODEL(model,PATH): exec(open(os.path.join(PATH,'dsLoad.txt')).read(),globals())
    Mk={"cadSurfOptions":{"volume_gradation":1.5,"gradation":1.5,"anisotropic_ratio":10},
        "tetraOptions":{"optimisation_level":"standard","verbose":"10"},"global_physical_size":0.05,"max_size":0.05,
        "min_size":"-1","gradation":1.5,"volume_gradation":1.5,"optimisation_level":"standard","anisotropic_ratio":"10",
        "geometric_approximation_min":"4","geometric_approximation_max":"25","geometric_approximation_on_edge":"false",
        "geometric_approximation_on_face":"true","use_surface_proximity":"false","surface_proximity_ratio":0,
        "approach":"kinematic","write_debug_files":"false","is_iso":"true","coeff_on_error":0.01,"remesh_type":1,
        "old_size_factor":0.0,"model_handle":model.GETHANDLEPTR()}
    CetMESH.ANISO_MESH(AN,0,PATH,**Mk)
    kwargs={"scaling":1,"write_debug_files":"false"}; exec(open(r"C:\workspace\fiabilite\InitSolver.py").read(),globals())
    kwargs.update(static_params=static_params,cinematic_params=cinematic_params,MKLPardiso_params=MKLPardiso_params,
                  MyPardiso_params=MyPardiso_params,MUMPS_params=MUMPS_params,FullLorentz=False,LorentzToSdp=False,
                  SdpToLorentz=0,printIntPointSolutioEvolution=False,trace_sur_point_integration=False,
                  calculate_error="false",max_nbOfDiv=0,customized_inc=[1],tetra_discontinuities=False,
                  activated_plasticity=True,welds_throat_limit=True,approach="kinematic")
    kwargs["sensitivity_analysis"]="true"; kwargs["sensitivity_regions"]=json.dumps(regions)
    kwargs["model_handle"]=model.GETHANDLEPTR()
    CetSOLV.SOLV(AN,0,PATH,**kwargs)
    info=json.load(open(os.path.join(PATH,AN+"_0_kine.dsmetares")))['info']
    return {"status":info.get("solver_status"),"lam":(info.get("Primal_bound") or [None])[0],
            "sens":info.get("Sensitivity",{}) or {}}

print("\n############ TEST footprint DEAD (magnitude + position) + separation live/dead ############")
res = run([R_DEAD_MAG, R_DEAD_POS, R_LIVE_POS])
print("   status =", res["status"], "| lambda =", res["lam"])
print("   sens   =", res["sens"])
dmag = res["sens"].get("DEAD_LOAD:deadconvoi")
dpos = res["sens"].get("DEAD_LOAD:position:deadconvoi")
lpos = res["sens"].get("LIVE_LOAD:position:deadconvoi")
ok = True
c0 = res["status"] == "OPTIMAL" and res["lam"] is not None
print(("OK " if c0 else "!! ECHEC ") + ": run converge (status=%s)" % res["status"]); ok &= c0
c1 = dmag is not None and abs(dmag) > 1e-6
print(("OK " if c1 else "!! ECHEC ") + ": DEAD_LOAD:deadconvoi (magnitude) NON NUL = %s" % dmag); ok &= c1
c2 = dpos is not None and abs(dpos) > 1e-6
print(("OK " if c2 else "!! ECHEC ") + ": DEAD_LOAD:position:deadconvoi (type=3) NON NUL = %s" % dpos); ok &= c2
# separation : le footprint dead ne doit PAS etre dans les vecteurs live -> LIVE_LOAD:position ~ 0
c3 = (lpos is None) or (abs(lpos) < 1e-6)
print(("OK " if c3 else "!! ECHEC ") + ": SEPARATION live/dead -> LIVE_LOAD:position:deadconvoi ~ 0 (=%s)" % lpos); ok &= c3
assert ok, "ECHEC : footprint dead (chantier 3b) incorrect"
print("\n############ CHANTIER 3b VALIDE : footprint DEAD magnitude+position, separe du live ############")
