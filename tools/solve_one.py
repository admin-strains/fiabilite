"""
Evaluation de la fonction de performance en UN point, par Digital Structure.

    python launcher.py tools/solve_one.py --model test_pure_flexion --fc 48 --fy 550

Extrait de `run_one_SOL` (AC3_pure_flexion.py l.603-730), reduit a l'essentiel :
patcher les parametres, mailler, resoudre, lire g = Primal_bound - 1.

Deux differences volontaires avec l'original :

* Le modele est **recopie** dans un dossier de travail avant d'etre modifie.
  L'original reecrit `dsCad.txt` en place, dans le `.ds` de l'utilisateur.
* Aucune variable globale : tout passe en argument. C'est la raison d'etre de
  ce fichier -- c'est le prototype de l'adaptateur `solver/` de la phase 5 du
  plan de nettoyage, celui qui doit devenir la SEULE frontiere avec Digital
  Structure.

A lancer par `launcher.py`, qui met en place les DLL et l'ordre d'import.
"""

import argparse
import json
import os
import shutil
import sys
import time

from STRAINS.rupt.APIs.CetCAD_API import *          # noqa: F401,F403
from STRAINS.rupt.APIs import CetLOAD
from STRAINS.rupt.APIs.CetLOAD_API import *         # noqa: F401,F403
import STRAINS.rupt.core.CetMESH as CetMESH
from STRAINS.rupt.core import CetSOLV

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _catalogues(ds_front):
    """INITCATALOG, comme en tete des scripts AC -- mais sans chemin en dur."""
    base = os.path.join(ds_front, "STRAINS", "common", "Catalog")
    lus = []
    for nom in ("CatalogTopo.json", "CatalogDimensions.json", "CatalogBolts.json"):
        with open(os.path.join(base, nom), "r") as fh:
            lus.append(fh.read())
    INITCATALOG(*lus)                                # noqa: F405


def patch_params(path, **params):
    """Reecrit dsCad.txt / dsLoad.txt (identique a AC3_pure_flexion.patch_params)."""
    import re
    for filename in ("dsCad.txt", "dsLoad.txt"):
        fpath = os.path.join(path, filename)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "r") as fh:
            content = fh.read()
        for name, value in params.items():
            content = re.sub(r"^" + name + r"\s*=.*$",
                             "%s    = %.10f" % (name, value),
                             content, count=1, flags=re.MULTILINE)
        with open(fpath, "w") as fh:
            fh.write(content)


def mesh_options(model, global_size, geo_min_approx):
    """Options de maillage de run_one_SOL, telles quelles."""
    return {
        "cadSurfOptions": {"volume_gradation": 1.5, "gradation": 1.5, "anisotropic_ratio": 10},
        "tetraOptions": {"optimisation_level": "standard", "verbose": "10"},
        "global_physical_size": global_size,
        "max_size": 0.05,
        "min_size": "-1",
        "gradation": 1.5,
        "volume_gradation": 1.5,
        "optimisation_level": "standard",
        "anisotropic_ratio": "10",
        "geometric_approximation_min": str(geo_min_approx),
        "geometric_approximation_max": "25",
        "geometric_approximation_on_edge": "false",
        "geometric_approximation_on_face": "true",
        "use_surface_proximity": "false",
        "surface_proximity_ratio": 0,
        "approach": "kinematic",
        "write_debug_files": "true",
        "is_iso": "true",
        "coeff_on_error": 0.01,
        "remesh_type": 1,
        "old_size_factor": 0.0,
        "model_handle": model.GETHANDLEPTR(),
    }


def solver_options(model, init_solver_py):
    """Parametres du point interieur, lus dans InitSolver.py comme le fait l'AC."""
    ns = {}
    with open(init_solver_py, "r") as fh:
        exec(fh.read(), ns)                          # noqa: S102
    kwargs = {
        "scaling": 1,
        "write_debug_files": "true",
        "static_params": ns["static_params"],
        "cinematic_params": ns["cinematic_params"],
        "MKLPardiso_params": ns["MKLPardiso_params"],
        "MyPardiso_params": ns["MyPardiso_params"],
        "MUMPS_params": ns["MUMPS_params"],
        "FullLorentz": False,
        "LorentzToSdp": False,
        "SdpToLorentz": 0,
        "printIntPointSolutioEvolution": False,
        "trace_sur_point_integration": False,
        "calculate_error": "false",
        "max_nbOfDiv": 0,
        "customized_inc": [1],
        "tetra_discontinuities": False,
        "activated_plasticity": True,
        "welds_throat_limit": True,
        "approach": "kinematic",
        "model_handle": model.GETHANDLEPTR(),
    }
    return kwargs


def evaluate(ds_dir, params, global_size=0.05, geo_min_approx=4,
             analysis="Yield_analysis0", iteration=0):
    """Renvoie (g, alpha, duree) pour un jeu de parametres. Modifie ds_dir."""
    patch_params(ds_dir, **params)

    model = MODEL()                                  # noqa: F405
    SET_CONTEXT(model, ds_dir)                       # noqa: F405

    with open(os.path.join(ds_dir, "dsCad.txt"), "r") as fh:
        exec(fh.read(), globals())                   # noqa: S102
    model.Save(os.path.join(ds_dir, analysis + ".dscad"))
    erreurs = model.GETERRORS()
    if erreurs:
        print("[solve_one] GETERRORS : %s" % erreurs, flush=True)

    with open(os.path.join(ds_dir, "dsLoad.txt"), "r") as fh:
        script_load = fh.read()
    with CetLOAD.LOAD_MODEL(model, ds_dir):
        exec(script_load, globals())                 # noqa: S102

    t0 = time.perf_counter()
    CetMESH.ANISO_MESH(analysis, iteration, ds_dir,
                       **mesh_options(model, global_size, geo_min_approx))
    t_mesh = time.perf_counter() - t0

    t0 = time.perf_counter()
    CetSOLV.SOLV(analysis, iteration, ds_dir,
                 **solver_options(model, os.path.join(REPO, "pure_flexion", "InitSolver.py")))
    t_solv = time.perf_counter() - t0

    with open(os.path.join(ds_dir, "%s_%d_kine.dsmetares" % (analysis, iteration)), "r") as fh:
        res = json.load(fh)
    alpha = res["info"]["Primal_bound"][0]
    return alpha - 1.0, alpha, t_mesh, t_solv


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", default="test_pure_flexion")
    ap.add_argument("--storage", default=r"C:\workspace\storage\admin\SF")
    ap.add_argument("--fc", type=float, required=True)
    ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--global-size", default="0.05")
    ap.add_argument("--geo-min", type=int, default=4)
    ap.add_argument("--workdir", default=None,
                    help="dossier de travail (defaut : copie temporaire, jamais le modele d'origine)")
    args = ap.parse_args()

    src = os.path.join(args.storage, args.model + ".ds")
    if not os.path.isdir(src):
        raise SystemExit("Modele introuvable : %s" % src)

    dst = args.workdir or os.path.join(args.storage, args.model + "_solveone.ds")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print("[solve_one] modele   : %s" % src, flush=True)
    print("[solve_one] copie    : %s" % dst, flush=True)
    print("[solve_one] point    : fc = %.6f   fy = %.6f" % (args.fc, args.fy), flush=True)
    print("[solve_one] maillage : global_physical_size = %s" % args.global_size, flush=True)

    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import launcher
    _catalogues(launcher.find_ds_root())

    g, alpha, t_mesh, t_solv = evaluate(
        dst, {"fc": args.fc, "fy": args.fy},
        global_size=args.global_size, geo_min_approx=args.geo_min)

    print("\n[solve_one] RESULTAT", flush=True)
    print("  alpha (Primal_bound) = %.10f" % alpha, flush=True)
    print("  g = alpha - 1        = %+.10f" % g, flush=True)
    print("  maillage %.1f s   solveur %.1f s" % (t_mesh, t_solv), flush=True)

    out = os.path.join(dst, "solve_one.json")
    with open(out, "w") as fh:
        json.dump({"fc": args.fc, "fy": args.fy, "alpha": alpha, "g": g,
                   "global_physical_size": args.global_size,
                   "t_mesh": t_mesh, "t_solv": t_solv}, fh, indent=1)
    print("  ecrit : %s" % out, flush=True)


if __name__ == "__main__":
    main()
