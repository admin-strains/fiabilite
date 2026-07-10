# -*- coding: utf-8 -*-
# J6 : validation par DIFFERENCES FINIES de la sensibilite analytique dalpha/ds
# (LOAD_POSITION) renvoyee par le solveur, sur le cas Test_cantilever_s.
#
# Principe : a fy fixe, alpha(s) = pObj. Le solveur renvoie dg/ds analytique a s0.
# On compare a la difference finie centree (alpha(s0+D) - alpha(s0-D)) / (2D).
# Bouger s ne change PAS le maillage (footprint = projection de charge sur maillage
# fixe) -> FD propre. A lancer via launcher (DLL paths) OU directement avec PYTHONPATH.
import os, sys, json, time

# --- setup STRAINS (comme le launcher) ---
import openturns as ot  # avant DLL STRAINS (MKL)
for d in [r'C:\workspace\front\STRAINS\rupt\core\bin', r'C:\workspace\front\STRAINS\rupt\core',
          r'C:\workspace\front\STRAINS\common\Dll', r'C:\workspace\front\STRAINS\rupt\core\bin\meshgems',
          r'C:\workspace\front\STRAINS\rupt\core\bin\mosek']:
    if os.path.isdir(d):
        os.add_dll_directory(d)
sys.path.insert(0, r'C:\workspace\front')
sys.path.insert(0, r'C:\workspace\fiabilite')

from STRAINS.rupt.APIs.CetCAD_API import *           # MODEL, SET_CONTEXT, POINT, BLOCK, REBAR, POLYGON...
from STRAINS.rupt.APIs import CetLOAD
from STRAINS.rupt.APIs.CetLOAD_API import *
import STRAINS.rupt.core.CetMESH as CetMESH
from STRAINS.rupt.core import CetSOLV
import re

# INITCATALOG (indispensable : sans ca BLOCK ne construit pas la geometrie -> maillage 0 points)
def _getFile(p):
    with open(p, 'r') as f:
        return f.read()
INITCATALOG(_getFile(r"C:\workspace\front\STRAINS\common\Catalog\CatalogTopo.json"),
            _getFile(r"C:\workspace\front\STRAINS\common\Catalog\CatalogDimensions.json"),
            _getFile(r"C:\workspace\front\STRAINS\common\Catalog\CatalogBolts.json"))

PATH = r"C:\workspace\storage\admin\Moulin_Blanc\Test_cantilever_s.ds"
ANALYSIS = 'Yield_analysis0'
GLOBAL_SIZE = 0.05      # comme l'AC (run_one_SOL)
GEO_MIN_APPROX = 4

def patch_dscad(s, fy_top):
    fpath = os.path.join(PATH, 'dsCad.txt')
    content = open(fpath).read()
    content = re.sub(r'^s\s*=.*$',      's    = %.10f' % s,      content, count=1, flags=re.MULTILINE)
    content = re.sub(r'^fy_top\s*=.*$', 'fy_top    = %.10f' % fy_top, content, count=1, flags=re.MULTILINE)
    open(fpath, 'w').write(content)

def run_strains(s, fy_top, sensitivity):
    """Patche (s, fy), maille, resout, retourne (pObj, dg_s, dg_fy)."""
    patch_dscad(s, fy_top)
    model = MODEL(); SET_CONTEXT(model, PATH)
    fileName = os.path.join(PATH, ANALYSIS + ".dscad")
    exec(open(os.path.join(PATH, 'dsCad.txt')).read(), globals())
    model.Save(fileName)
    with CetLOAD.LOAD_MODEL(model, PATH):
        exec(open(os.path.join(PATH, 'dsLoad.txt')).read(), globals())
    Meshkwargs = {
        "cadSurfOptions": {"volume_gradation": 1.5, "gradation": 1.5, "anisotropic_ratio": 10},
        "tetraOptions": {"optimisation_level": "standard", "verbose": "10"},
        "global_physical_size": GLOBAL_SIZE, "max_size": 0.05, "min_size": "-1",
        "gradation": 1.5, "volume_gradation": 1.5, "optimisation_level": "standard",
        "anisotropic_ratio": "10", "geometric_approximation_min": str(GEO_MIN_APPROX),
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
    if sensitivity:
        kwargs["sensitivity_analysis"] = "true"
        kwargs["sensitivity_regions"] = json.dumps(
            [{"param": "LOAD_POSITION"},
             {"param": "YIELD_STRENGTH", "rebars": ["HA1", "HA2", "HA3", "HA4"]}])
    kwargs["model_handle"] = model.GETHANDLEPTR()
    CetSOLV.SOLV(ANALYSIS, 0, PATH, **kwargs)
    d = json.load(open(os.path.join(PATH, ANALYSIS + "_0_kine.dsmetares")))
    pObj = d['info']['Primal_bound'][0]
    dg_s = dg_fy = None
    if sensitivity and 'Sensitivity' in d['info']:
        for k, v in d['info']['Sensitivity'].items():
            if 'LOAD_POSITION' in k:  dg_s = v
            elif 'YIELD_STRENGTH' in k: dg_fy = v
    return pObj, dg_s, dg_fy

if __name__ == '__main__':
    FY = 550.0
    # 2026-06-30 : s est desormais NORMALISE in [0,1] (archi shift/PATH). Le chemin
    # du bord gauche va de x=0 a x=L-load_len=3.4 m, donc s_norm = x_m / L_PATH.
    # Le tangent passe au solveur = vecteur COMPLET D=(3.4,0,0) -> dAlpha/ds est p/r
    # au s normalise. On garde la MEME position physique qu'avant (x=2.0 m) et les
    # memes deltas physiques (0.10, 0.05 m) convertis en normalise, pour comparabilite.
    # Attendu : dAlpha/ds_norm = dAlpha/dx_m * L_PATH ~ -0.863 * 3.4 ~ -2.93.
    L_PATH = 4.0 - 0.6
    S0 = 2.0 / L_PATH
    # Balayage de Delta (en METRES de deplacement reel de l'empreinte, convertis en
    # s normalise /L_PATH) pour etudier la convergence FD -> analytique. Petits Delta
    # = plus precis en theorie mais domines par le bruit FP du SOCP (cf. 0.01 m).
    DELTAS_M = (0.10, 0.08, 0.05, 0.01)
    for DELTA in (dm / L_PATH for dm in DELTAS_M):
        print("\n" + "=" * 70, flush=True)
        print(f"J6 FD  --  fy_top={FY}  s0={S0:.6f} (x={S0*L_PATH:.3f} m)  "
              f"Delta={DELTA:.6f} (={DELTA*L_PATH:.3f} m)  L_PATH={L_PATH}", flush=True)
        print("=" * 70, flush=True)
        a0, dg_s_ana, dg_fy = run_strains(S0,        FY, sensitivity=True)
        ap, _, _            = run_strains(S0 + DELTA, FY, sensitivity=False)
        am, _, _            = run_strains(S0 - DELTA, FY, sensitivity=False)
        dg_s_fd = (ap - am) / (2.0 * DELTA)
        err = abs(dg_s_ana - dg_s_fd) / max(abs(dg_s_fd), 1e-12) * 100.0
        print(f"\n  alpha(s0-D) = {am:.6f}", flush=True)
        print(f"  alpha(s0)   = {a0:.6f}", flush=True)
        print(f"  alpha(s0+D) = {ap:.6f}", flush=True)
        print(f"  --> dalpha/ds ANALYTIQUE (solveur) = {dg_s_ana:+.6f}", flush=True)
        print(f"  --> dalpha/ds DIFF FINIE (centree) = {dg_s_fd:+.6f}", flush=True)
        print(f"  --> ERREUR RELATIVE = {err:.3f} %   {'OK' if err < 5 else 'A VERIFIER'}", flush=True)
        print(f"  (dg/dfy analytique au passage = {dg_fy})", flush=True)
