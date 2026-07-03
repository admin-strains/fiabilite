# -*- coding: utf-8 -*-
# FD "MEME VARIABLE" (2026-07-03, MM) : valide la regle de chaine quand UNE variable
# pilote a la fois la position ET le multiplicateur de dead (poids propre).
#   dalpha/dtheta = dalpha/ds * (ds/dtheta) + dalpha/dm_d * (dm_d/dtheta)
# On perturbe s ET la gravite ENSEMBLE et on compare Delta_alpha mesure a la somme
# des deux contributions (partiels du solveur). Format MOVING_LOAD (Test_moving_load.ds).
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
PATH=r"C:\workspace\storage\admin\Moulin_Blanc\Test_moving_load.ds"; AN='Yield_analysis0'
DSCAD=os.path.join(PATH,'dsCad.txt'); DSLOAD=os.path.join(PATH,'dsLoad.txt')
G0=-9.81
def patch(s, g):
    d=open(DSLOAD).read()
    d=re.sub(r'^s\s*=.*$','s = %.10f'%s,d,count=1,flags=re.MULTILINE)
    d=re.sub(r"VALUE=-?[0-9.]+",'VALUE=%.6f'%g,d,count=1)
    open(DSLOAD,'w').write(d)
def run(s, g, sens):
    patch(s,g)
    model=MODEL(); SET_CONTEXT(model,PATH); exec(open(DSCAD).read(),globals())
    model.Save(os.path.join(PATH,AN+".dscad"))
    with CetLOAD.LOAD_MODEL(model,PATH): exec(open(DSLOAD).read(),globals())
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
    kwargs["sensitivity_analysis"]="true" if sens else "false"
    if sens:
        kwargs["sensitivity_regions"]=json.dumps([{"param":"LIVE_LOAD","axis":"position","load_case":"convoi"},
                                                   {"param":"DEAD_LOAD","load_case":"LC_poids"}])
    kwargs["model_handle"]=model.GETHANDLEPTR()
    CetSOLV.SOLV(AN,0,PATH,**kwargs)
    info=json.load(open(os.path.join(PATH,AN+"_0_kine.dsmetares")))['info']
    sn=info.get("Sensitivity",{}) or {}
    return (info.get("Primal_bound") or [None])[0], sn.get("LIVE_LOAD:position:convoi"), sn.get("DEAD_LOAD:LC_poids")

S0=0.45; DS=0.03; DMD=0.05      # theta bouge s de +DS ET amplifie le dead de +DMD (m_d: 1 -> 1+DMD)
print("\n########### FD 'MEME VARIABLE' : position + dead amplifies ENSEMBLE ###########")
a0, ds0, dmd0 = run(S0, G0, True)
print("base   s=%.2f m_d=1.00 : alpha=%.6f | dalpha/ds=%.4f | dalpha/dm_d=%.4f" % (S0, a0, ds0, dmd0))
aP, _, _ = run(S0+DS, G0*(1.0+DMD), False)
print("pertu  s=%.2f m_d=%.2f : alpha=%.6f" % (S0+DS, 1+DMD, aP))
da_meas = aP - a0
da_pred = ds0*DS + dmd0*DMD
print("\nDelta_alpha MESURE  (perturbe s ET m_d ensemble) = %.6f" % da_meas)
print("Delta_alpha PREDIT  (dalpha/ds*Ds + dalpha/dm_d*Dm_d) = %.6f  [%.4f + %.4f]"
      % (da_pred, ds0*DS, dmd0*DMD))
err = abs(da_meas-da_pred)/max(abs(da_meas),1e-12)*100
print("ecart = %.2f%%  (2nd ordre en (Ds,Dm) -> qq %% attendu)" % err)
print(("OK " if err < 8.0 else "!! ") + ": regle de chaine 'meme variable' validee (somme des partiels)")
print("########### FIN ###########")
