"""
CODE FIABILITE UTILISATEUR 
"""
import os
import json
import shutil
import re

from STRAINS.rupt.APIs.CetCAD_API import *
from STRAINS.rupt.APIs import CetLOAD
from STRAINS.rupt.APIs.CetLOAD_API import *
import STRAINS.rupt.core.CetMESH as CetMESH
from STRAINS.rupt.core import CetSOLV as CetSOLV
from STRAINS.rupt.core import CetVISU as CetVISU, CetLIST as CetLIST
from STRAINS.rupt.APIs.CetNOTE_API import *
from STRAINS.rupt.APIs import CetNOTE


def getFile(nameFile):
    f = open(nameFile, 'r')
    res = f.read()
    f.close()
    return res


catalogTopo = getFile("C:\\workspace\\front\\STRAINS\\common\\Catalog\\CatalogTopo.json")
catalogDimensions = getFile("C:\\workspace\\front\\STRAINS\\common\\Catalog\\CatalogDimensions.json")
catalogBolt = getFile("C:\\workspace\\front\\STRAINS\\common\\Catalog\\CatalogBolts.json")
INITCATALOG(catalogTopo, catalogDimensions, catalogBolt)

import openturns as ot
import numpy as np
import autograd.numpy as anp
import matplotlib
_HEADLESS = bool(os.environ.get("_IS_PARALLEL")) or bool(os.environ.get("_FIAB_LOG_REDIRECTED"))
matplotlib.use('Agg' if _HEADLESS else 'TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from smt.surrogate_models import GEKPLS
from scipy.optimize import brentq
import re
import math
from sklearn.cluster import DBSCAN
from scipy.stats import norm
from math import comb
import warnings
from datetime import datetime
from branche1 import fit_gepck, predict_gepck, predict_gradient_gepck
from _parallel_is import adaptive_is
from lois import loi_fy, loi_fc, loi_F_permanente, loi_F_exploitation, loi_F_intermittente, loi_uni_approx, SIGMA
from patch_params import patch_params
import etat
from config_utilisateur import *
from config_pardefaut import *
from form_is import _is_position_var, _find_position_var_index, projection_surrogate, FORM_all_modes, run_IS, run_IS_proj, print_results_IS
from visu_pure import _batch_mu_sigma, _eff_vectorized, print_EFF_graphs, print_Pf_evolution, print_logPf_evolution
from surrogate_pure import (build_starting_points, build_Y_aug, build_metamodel_PCE, calculate_PCE,
    build_metamodel_KRG, build_metamodel_GEK, PCKRGFunction, oldGEPCKFunction, GEPCKFunction,
    BoundSurrogateFunction, GEKPLSFunction, EFFFunction)
_IS_PARALLEL = os.environ.get("_IS_PARALLEL", "1") != "0"
_IS_K        = int(os.environ.get("_IS_K", "16"))
_IS_CHUNK    = int(os.environ.get("_IS_CHUNK", "8"))
_IS_PROBE    = int(os.environ.get("_IS_PROBE", "16"))


def _parse(text, name):
    return float(re.search(rf'(?m)^\s*{re.escape(name)}\s*=\s*([\d.]+)', text).group(1))

if __name__ == '__main__':
    print("=" * 70)
    print("CALCUL DE FIABILITE -- UTILISATEUR")
    print("=" * 70)
    # --- Variables importees depuis config_utilisateur.py et config_pardefaut.py ---
    # --- Voir ces fichiers pour modifier les parametres ---

    # --- Derivees calculees a l'execution ---
    _path_ds = storage + modelname + ".ds"
    modelname = os.environ.get("_DOE_WORKER_MODELNAME") or modelname
    with open(os.path.join(_path_ds, 'dsCad.txt'), 'r') as f:
        _cad_txt = f.read()

    # --- Lecture dsCad (voir prelim_dscad dans config_utilisateur.py) ---
    rebar_names, group1_names, group2_names = prelim_dscad(_cad_txt)
    n_rebars = len(rebar_names)
    print(f"[2-fy] groupe 1 (fyd1) : {len(group1_names)} aciers | groupe 2 (fyd2) : {len(group2_names)} aciers", flush=True)

    # --- PARAM_CONFIG : merge et validation ---
    PARAM_CONFIG = {**PARAM_CONFIG_LOAD, **PARAM_CONFIG_CAD}
    params_names = list(PARAM_CONFIG_LOAD.keys()) + list(PARAM_CONFIG_CAD.keys())
    n_var = len(params_names)
    _rk = [PARAM_CONFIG[p]['sens'].get('region_key') for p in params_names]
    assert all(_rk), f"region_key manquant dans PARAM_CONFIG : {[p for p, r in zip(params_names, _rk) if not r]}"
    assert len(set(_rk)) == len(_rk), f"region_key dupliques : {_rk}"

    # --- Flags derives ---
    print_grad_sp = False #option si on veut afficher les gradients des points de départ

    # --- Résultats fixés ---
    hf_3d_grid_fixed = None
    # do_custom_hf : True = utiliser la grille custom pour le contour HF (au lieu de linspace 7x7)
    do_custom_hf = True
    _custom_grid_file = os.path.join(path_dir, 'output', 'custom_hf_grid.json')
    if do_custom_hf and os.path.exists(_custom_grid_file):
        hf_custom_points = json.load(open(_custom_grid_file))['grid_u']
        print(f"[HF CUSTOM] grille chargee : {len(hf_custom_points)} points depuis {_custom_grid_file}", flush=True)
    else:
        hf_custom_points = None
        if do_custom_hf:
            print(f"[HF CUSTOM] fichier introuvable ({_custom_grid_file}) -> grille standard", flush=True)
            do_custom_hf = False

    # --- Résultats fixés du run HF 12/05 (gamma=1.0, F=0.74, n0=15) ---
    # Actifs uniquement en mode visu seule (tous do_* = False).
    if modele == None:
        sol_modes_fixed = None
        # guide pour hardcoder {
        #     # (sp_u1, sp_u2): (u*_u1, u*_u2)
        #     (-0.002,  1.332): (-5.306, -6.200),
        #     ( 0.610, -0.310): (-3.117, -7.349),
        #     (-0.571, -1.705): (-3.131, -7.347),
        #     ( 0.258,  0.306): (-4.655, -6.643),
        #     ( 0.121, -1.010): (-3.117, -7.345),
        #     ( 1.624, -0.531): (-3.046, -7.352),
        #     (-0.087, -0.172): (-4.721, -6.603),
        #     (-0.419,  0.597): (-5.290, -6.212),
        #     (-0.843, -0.005): (-5.341, -6.152),
        #     ( 0.868, -1.460): (-3.014, -7.363),
        #     (-1.681,  0.710): (-6.475, -4.986),
        #     ( 1.479,  0.239): (-3.098, -7.354),
        #     (-1.117, -0.696): (-5.200, -6.275),
        #     ( 0.745,  1.043): (-4.740, -6.571),
        #     (-0.694,  2.114): (-6.504, -4.966),
        #     ( 0.000,  0.000): (-4.776, -6.571),
        # }
        best_sol_modes_fixed = None
        # guide pour hardcoder {
        #     'A': {'sp': ( 0.868, -1.460), 'u*': (-3.014, -7.363)},
        #     'B': {'sp': ( 0.745,  1.043), 'u*': (-4.740, -6.571)},
        #     'C': {'sp': (-0.843, -0.005), 'u*': (-5.341, -6.152)},
        #     'D': {'sp': (-0.694,  2.114), 'u*': (-6.504, -4.966)},
        # }
        # Gradients HF aux sp (run 1305_0937, 4 appels STRAINS)
        grad_sp_fixed = None
        # guide pour hardcoder {
        #     'A': {'g': 0.550023, 'grad': [ 0.039500,  0.072312], 'neg_grad': [-0.039500, -0.072312]},
        #     'B': {'g': 0.722565, 'grad': [ 0.046148,  0.068963], 'neg_grad': [-0.046148, -0.068963]},
        #     'C': {'g': 0.573155, 'grad': [ 0.058216,  0.059250], 'neg_grad': [-0.058216, -0.059250]},
        #     'D': {'g': 0.704360, 'grad': [ 0.068462,  0.055309], 'neg_grad': [-0.068462, -0.055309]},
        # }
        # Trajectoires FORM hardcodees (run 1805_1957, do_HF=True, n0=7)
        # Une trajectoire representative par mode : points et gradients successifs AbdoRackwitz
        traj_runs_fixed = None
        # guide pour hardcoder {
        #         'A': {  # u* ~ [-3.12, -7.35]  (26 pts, sp=[0.61,-0.31])
        #             'points': [
        #                 [ 0.6102, -0.3098], [-3.8707, -6.4583], [-1.6303, -3.3841], [-0.5101, -1.8470],
        #                 [ 0.0501, -1.0784], [-3.8803, -6.5921], [-1.9151, -3.8353], [-0.9325, -2.4568],
        #                 [-0.4412, -1.7676], [-3.8389, -6.7200], [-2.1401, -4.2438], [-1.2907, -3.0057],
        #                 [-0.8659, -2.3867], [-3.7616, -6.8474], [-2.3138, -4.6171], [-1.5899, -3.5019],
        #                 [-1.2279, -2.9443], [-3.6819, -6.9516], [-2.4549, -4.9479], [-1.8414, -3.9461],
        #                 [-3.5392, -7.1070], [-2.6903, -5.5265], [-2.2659, -4.7363], [-3.3755, -7.2150],
        #                 [-2.8207, -5.9757], [-3.1170, -7.3495],
        #             ],
        #             'grads': [
        #                 [0.042205, 0.070421], [0.031332, 0.065618], [0.035786, 0.068399], [0.039346, 0.069164],
        #                 [0.041015, 0.069679], [0.029800, 0.066665], [0.034813, 0.068268], [0.037701, 0.069006],
        #                 [0.039546, 0.069225], [0.028110, 0.067965], [0.033785, 0.068334], [0.036434, 0.068843],
        #                 [0.037930, 0.069045], [0.027391, 0.068676], [0.032755, 0.068537], [0.035300, 0.068765],
        #                 [0.036516, 0.068944], [0.027059, 0.069116], [0.031878, 0.068754], [0.034250, 0.068777],
        #                 [0.026583, 0.069808], [0.030447, 0.069107], [0.032261, 0.068955], [0.025926, 0.070694],
        #                 [0.029462, 0.069467], [0.025199, 0.071834],
        #             ],
        #         },
        #         'B': {  # u* ~ [-4.66, -6.64]  (50 pts, sp=[0.26,0.31])
        #             'points': [
        #                 [ 0.2576,  0.3059], [-4.1149, -6.3330], [-1.9286, -3.0135], [-0.8355, -1.3538],
        #                 [-0.2890, -0.5239], [-4.1318, -6.4693], [-2.2104, -3.4966], [-1.2497, -2.0103],
        #                 [-0.7693, -1.2671], [-0.5291, -0.8955], [-0.4091, -0.7097], [-4.1355, -6.4913],
        #                 [-2.2723, -3.6005], [-1.3407, -2.1551], [-0.8749, -1.4324], [-0.6420, -1.0711],
        #                 [-0.5255, -0.8904], [-4.1495, -6.5088], [-2.3375, -3.6996], [-1.4315, -2.2950],
        #                 [-0.9785, -1.5927], [-0.7520, -1.2416], [-0.6388, -1.0660], [-4.1748, -6.5169],
        #                 [-2.4068, -3.7914], [-1.5228, -2.4287], [-1.0808, -1.7473], [-0.8598, -1.4067],
        #                 [-4.2403, -6.5157], [-2.5501, -3.9612], [-1.7049, -2.6839], [-1.2823, -2.0453],
        #                 [-1.0710, -1.7260], [-4.3300, -6.4854], [-2.7005, -4.1057], [-1.8858, -2.9158],
        #                 [-1.4784, -2.3209], [-4.5085, -6.4242], [-2.9935, -4.3726], [-2.2359, -3.3467],
        #                 [-1.8572, -2.8338], [-4.6009, -6.4217], [-3.2291, -4.6278], [-2.5431, -3.7308],
        #                 [-4.7317, -6.4391], [-3.6374, -5.0850], [-3.0903, -4.4079], [-4.7438, -6.4980],
        #                 [-3.9170, -5.4529], [-4.6552, -6.6434],
        #             ],
        #             'grads': [
        #                 [0.044519, 0.068517], [0.035878, 0.061726], [0.044263, 0.062955], [0.043386, 0.066384],
        #                 [0.043289, 0.067779], [0.034697, 0.062504], [0.042966, 0.062862], [0.043945, 0.065014],
        #                 [0.043028, 0.066722], [0.042947, 0.067351], [0.043059, 0.067588], [0.034527, 0.062612],
        #                 [0.042687, 0.062843], [0.043966, 0.064760], [0.043206, 0.066373], [0.042968, 0.067062],
        #                 [0.042943, 0.067361], [0.034494, 0.062596], [0.042443, 0.062793], [0.044100, 0.064442],
        #                 [0.043438, 0.065989], [0.042999, 0.066778], [0.042969, 0.067074], [0.034637, 0.062421],
        #                 [0.042247, 0.062703], [0.044270, 0.064105], [0.043695, 0.065590], [0.043215, 0.066405],
        #                 [0.035292, 0.061763], [0.042202, 0.062322], [0.044554, 0.063438], [0.044217, 0.064786],
        #                 [0.043780, 0.065573], [0.036382, 0.060718], [0.042725, 0.061569], [0.044644, 0.062870],
        #                 [0.044856, 0.063916], [0.037854, 0.059111], [0.043326, 0.060308], [0.044505, 0.061920],
        #                 [0.044972, 0.062769], [0.038234, 0.058548], [0.043058, 0.059735], [0.044734, 0.060876],
        #                 [0.038627, 0.057857], [0.042160, 0.058993], [0.043652, 0.059794], [0.038325, 0.058030],
        #                 [0.041163, 0.058744], [0.037124, 0.059166],
        #             ],
        #         },
        #         'C': {  # u* ~ [-5.31, -6.20]  (12 pts, sp=[-0.00,1.33])
        #             'points': [
        #                 [-0.0017,  1.3325], [-5.0636, -5.2359], [-2.5327, -1.9517], [-5.5307, -5.5909],
        #                 [-4.0317, -3.7713], [-3.2822, -2.8615], [-5.5130, -5.8096], [-4.3976, -4.3356],
        #                 [-3.8399, -3.5985], [-5.4580, -5.9699], [-4.6489, -4.7842], [-5.3056, -6.1998],
        #             ],
        #             'grads': [
        #                 [0.059352, 0.061371], [0.046179, 0.051721], [0.054808, 0.055405], [0.045432, 0.050529],
        #                 [0.048982, 0.053557], [0.051740, 0.054524], [0.044423, 0.051269], [0.047496, 0.053207],
        #                 [0.049358, 0.053988], [0.042249, 0.052934], [0.045724, 0.053430], [0.040682, 0.054516],
        #             ],
        #         },
        #         'D': {  # u* ~ [-6.48, -4.99]  (13 pts, sp=[-1.68,0.71])
        #             'points': [
        #                 [-1.6808,  0.7104], [-5.7529, -4.8956], [-3.7168, -2.0926], [-2.6988, -0.6911],
        #                 [-6.0548, -4.8684], [-4.3768, -2.7798], [-3.5378, -1.7354], [-6.1953, -4.9797],
        #                 [-4.8666, -3.3576], [-4.2022, -2.5465], [-6.3582, -4.9252], [-5.2802, -3.7358],
        #                 [-6.4751, -4.9861],
        #             ],
        #             'grads': [
        #                 [0.063569, 0.054096], [0.052114, 0.045396], [0.059464, 0.048913], [0.063145, 0.050772],
        #                 [0.053159, 0.043559], [0.059506, 0.046396], [0.060808, 0.048877], [0.052863, 0.043191],
        #                 [0.058439, 0.045099], [0.060250, 0.046670], [0.054448, 0.041540], [0.057304, 0.044126],
        #                 [0.054778, 0.040852],
        #             ],
        #         },
        #     }
    
    else:
        sol_modes_fixed = None
        best_sol_modes_fixed = None
        grad_sp_fixed = None
        traj_runs_fixed = None

    # --- Variables d'etat partagees : voir etat.py ---

    # --- Sortie PNG EFF ---
    etat.timestamp   = datetime.now().strftime('%d%m_%H%M')
    etat.out_dir_eff = os.path.join(path_dir, 'output', 'png EFF', f'png_EFF_{etat.timestamp}')
    os.makedirs(etat.out_dir_eff, exist_ok=True)

    do_KRG = True if modele == 'KRG' else False
    do_GEK = True if modele == 'GEK' else False #on ajoute peut etre plus de points avec GEK car plus précis donc voit plus derreur
    do_HF = True if modele == 'HF' else False # penser à jouer avec des bornes différentes
    do_PCKRG = True if modele == 'PCKRG' else False
    do_old_GEPCK = True if modele == 'old_GEPCK' else False
    do_GEPCK     = True if modele == 'GEPCK'     else False
    do_IS   = do_IS and modele != 'HF'                        # IS impraticable en HF
    do_EFF   = do_EFF and modele != 'HF'                     # EFF impraticable en HF

    # --------------------------------------------------------------------------- #
    # DEFINTION DE FONCTIONS                                                      #
    # --------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------- #
    # FONCTION D'APPEL STRAINS ET DOE                                             #
    # --- PARAM_CONFIG : catalogue des variables aleatoires ---

    
    # _is_position_var, _find_position_var_index : voir form_is.py


    # --- APPELS STRAINS ---

    def _save_socp_outputs(path, AnalysisName, prefix_tag, u1=None, u2=None, p_vals=None):
        """Copie le dsmed cinematique dans SOCP_history/.
        Un sous-dossier par appel SOCP, nomme prefix_tag + coords."""
        files_to_save = [
            f"{AnalysisName}_0_kine.dsmed",
            "dsCad.txt",
            "dsLoad.txt",
        ]
        import datetime as _dt_save
        _ts = _dt_save.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        coords_str = ""
        if p_vals is not None:
            coords_str = "_" + "_".join(f"{params_names[i]}{p_vals[i]:.6f}" for i in range(len(p_vals)))
        _socp_root = os.environ.get("_DOE_MAIN_DS") or path
        sub_dir = os.path.join(_socp_root, "SOCP_history", f"{prefix_tag}_{_ts}{coords_str}")
        os.makedirs(sub_dir, exist_ok=True)
        n_saved = 0
        total_size = 0
        for f in files_to_save:
            src = os.path.join(path, f)
            if os.path.exists(src):
                dst = os.path.join(sub_dir, f)
                try:
                    shutil.copy2(src, dst)
                    n_saved += 1
                    total_size += os.path.getsize(src)
                except Exception as e:
                    print(f"  [SOCP HISTORY] copy failed for {f} : {e}", flush=True)
        print(f"  [SOCP HISTORY] {prefix_tag}{coords_str} : {n_saved} fichiers sauves "
              f"({total_size/1024/1024:.1f} MB) dans {sub_dir}", flush=True)

    def _sens_key_to_param(k):
        """Mappe une cle de sensibilite STRAINS vers le nom de variable dans params_names.
        Correspondance EXACTE 'param:region_key'. General (tous types), robuste."""
        for p in params_names:
            sens = PARAM_CONFIG[p]['sens']
            if k == sens['param'] + ':' + sens['region_key']:
                return p
        return None

    def run_one_SOL(modelname, SOL, params_names, sensitivity=False, with_sens_dict=None):
        """Lance un calcul complet pour une valeur de FT donnee.
        Retourne la liste des solutions pour chaque jeu de variables dans SOL (liste de dictionnaire).
        Les gradients sont convertis en espace U (standard normal) via T = isoprobabilistic transform.
        SOL[i]['dg_<var>'] = gradient en U. SOL[i]['_u'] = coordonnees U du point."""
        path = storage + modelname + ".ds"
        AnalysisName = 'Yield_analysis0'
        iteration = 0
        dist_X = dist_jointe()
        T = dist_X.getIsoProbabilisticTransformation()
        T_inv = dist_X.getInverseIsoProbabilisticTransformation()
        for i in range (len(SOL)):
            patch_params(path, **{p: SOL[i][p] for p in params_names})
            model = MODEL()
            SET_CONTEXT(model, path)
            fileName = os.path.join(path, AnalysisName + ".dscad")

            cadfile = open(path + '\\dsCad.txt', 'r')
            cadscript = cadfile.read()
            exec(cadscript, globals())
            model.Save(fileName)
            print(model.GETERRORS())

            loadfile = open(path + '\\dsLoad.txt', 'r')
            loadscript = loadfile.read()
            with CetLOAD.LOAD_MODEL(model, path):
                exec(loadscript, globals())

            Meshkwargs = {
                "cadSurfOptions": {"volume_gradation": 1.5, "gradation": 1.5, "anisotropic_ratio": 10},
                "tetraOptions": {"optimisation_level": "standard", "verbose": "10"},
                "global_physical_size": 0.05,
                "max_size": 0.05,
                "min_size": "-1",
                "gradation": 1.5,
                "volume_gradation": 1.5,
                "optimisation_level": "standard",
                "anisotropic_ratio": "10",
                "geometric_approximation_min": "4",
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
            }
            Meshkwargs["model_handle"] = model.GETHANDLEPTR()
            CetMESH.ANISO_MESH(AnalysisName, iteration, path, **Meshkwargs)

            kwargs = {"scaling": 1, "write_debug_files": "true"}
            exec(open(os.path.join(path_dir, "code", "InitSolver.py")).read(), globals())
            kwargs["static_params"] = static_params
            kwargs["cinematic_params"] = cinematic_params
            kwargs["MKLPardiso_params"] = MKLPardiso_params
            kwargs["MyPardiso_params"] = MyPardiso_params
            kwargs["MUMPS_params"] = MUMPS_params
            kwargs["FullLorentz"] = False
            kwargs["LorentzToSdp"] = False
            kwargs["SdpToLorentz"] = 0
            kwargs["printIntPointSolutioEvolution"] = False
            kwargs["trace_sur_point_integration"] = False
            kwargs["calculate_error"] = "false"
            kwargs["max_nbOfDiv"] = 0
            kwargs["customized_inc"] = [1]
            kwargs["tetra_discontinuities"] = False
            kwargs["activated_plasticity"] = True
            kwargs["welds_throat_limit"] = True
            kwargs["approach"] = "kinematic"

            if sensitivity:
                kwargs["sensitivity_analysis"] = "true"
                kwargs["sensitivity_regions"] = json.dumps(
                    [PARAM_CONFIG[p]['sens'] for p in params_names]
                )

            kwargs["model_handle"] = model.GETHANDLEPTR()
            CetSOLV.SOLV(AnalysisName, iteration, path, **kwargs)

            # Lire le resultat
            metares_path = os.path.join(path, AnalysisName + "_0_kine.dsmetares")
            with open(metares_path, 'r') as f:
                d = json.load(f)
            SOL[i]['g']=d['info']['Primal_bound'][0] -1
            if save_history:
                etat._socp_call_counter[0] += 1
                _save_socp_outputs(path, AnalysisName,
                                   prefix_tag=f"SOL_{etat._socp_call_counter[0]:03d}",
                                   p_vals=[float(SOL[i][p]) for p in params_names])
            # --- Lecture sensibilites et conversion X -> U ---
            grad_X = [None] * n_var
            if sensitivity and 'Sensitivity' in d['info']:
                print(f"les sensibilites sont calculees pour les elements : {d['info']['Sensitivity'].items()}")
                for k, v in d['info']['Sensitivity'].items():
                    p = _sens_key_to_param(k)
                    if p is not None:
                        grad_X[params_names.index(p)] = v
                    if all(g is not None for g in grad_X):
                        break
            # Conversion en espace U
            x_point = ot.Point([float(SOL[i][p]) for p in params_names])
            u_point = T(x_point)
            SOL[i]['_u'] = [float(u_point[j]) for j in range(n_var)]
            if sensitivity and all(g is not None for g in grad_X):
                J_Tinv = T_inv.gradient(u_point)
                J_Tinv_T = J_Tinv.transpose()
                grad_U = J_Tinv_T * ot.Point(grad_X)
                for j, p in enumerate(params_names):
                    SOL[i][f'dg_{p}'] = float(grad_U[j])
            else:
                for p in params_names:
                    SOL[i][f'dg_{p}'] = None
            # --- Sauvegarde incrementale du cache DOE ---
            _n_done = sum(1 for s in SOL if 'g' in s)
            _save_doe_cache_incremental(SOL, _n_done)
        return SOL

    def run_HF(u):
        import psutil as _psutil_hf
        _mem_before = _psutil_hf.Process(os.getpid()).memory_info().rss / 1024 / 1024
        sensitivity = True
        n_var = len(u)
        dist_X = dist_jointe()
        T_inv = dist_X.getInverseIsoProbabilisticTransformation() 
        u_point = ot.Point(u)
        x_point = T_inv(u_point)
        path = storage + modelname + ".ds"
        AnalysisName = 'Yield_analysis0'
        iteration = 0
        params={params_names[i]: x_point[i] for i in range(n_var)}
        patch_params(path, **params) #à cette étape SOL ne contient que 'fc': ,'fy':
        model = MODEL() #ici model n'est pas encore rempli
        SET_CONTEXT(model, path)
        fileName = os.path.join(path, AnalysisName + ".dscad") #on crée le chemin du fichier disque .dscad lisible par C. C va tout faire et on renverra les info plus tard (.load)

        cadfile = open(path + '\\dsCad.txt', 'r')
        cadscript = cadfile.read() #on met dans cadscript les info de dsCad.txt
        exec(cadscript, globals()) # ici on modifie le modèle (C, cython) et donc les variables (on exécute le script de dsCad.txt ce qui modifie les variables - rien dans .dscad, tout dans var. en mémoire)
        model.Save(fileName) # ici on créé dscad et on enregistre les modifs des variables dans .dscad
        print(model.GETERRORS()) # est vide si pas de message d'erreur sur le logiciel

        loadfile = open(path + '\\dsLoad.txt', 'r')
        loadscript = loadfile.read()
        with CetLOAD.LOAD_MODEL(model, path): #par with on appelle enter et exit et on force l'enregistrement par exit meme si erreur/ bug dans bloc.
            exec(loadscript, globals()) # pareil, on execute dsLoad et on enregistre dans var. mémoire

        Meshkwargs = { #définit la mesh - pas à comprendre ici car ne sera pas modifié. 
            "cadSurfOptions": {"volume_gradation": 1.5, "gradation": 1.5, "anisotropic_ratio": 10},
            "tetraOptions": {"optimisation_level": "standard", "verbose": "10"},
            "global_physical_size": 0.05,  # mesh fin pour bonne convergence
            "max_size": 0.05,
            "min_size": "-1",
            "gradation": 1.5,
            "volume_gradation": 1.5,
            "optimisation_level": "standard",
            "anisotropic_ratio": "10",
            "geometric_approximation_min": "4",
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
        }
        Meshkwargs["model_handle"] = model.GETHANDLEPTR()
        CetMESH.ANISO_MESH(AnalysisName, iteration, path, **Meshkwargs)

        kwargs = {"scaling": 1, "write_debug_files": "true"} # ci-dessous on définit dict kwargs en entrée de SOLV.
        exec(open(os.path.join(path_dir, "code", "InitSolver.py")).read(), globals()) #question pour Agnes : je ne suis pas sure que ca marche comme ca.
        kwargs["static_params"] = static_params
        kwargs["cinematic_params"] = cinematic_params
        kwargs["MKLPardiso_params"] = MKLPardiso_params
        kwargs["MyPardiso_params"] = MyPardiso_params
        kwargs["MUMPS_params"] = MUMPS_params
        kwargs["FullLorentz"] = False
        kwargs["LorentzToSdp"] = False
        kwargs["SdpToLorentz"] = 0
        kwargs["printIntPointSolutioEvolution"] = False
        kwargs["trace_sur_point_integration"] = False
        kwargs["calculate_error"] = "false"
        kwargs["max_nbOfDiv"] = 0
        kwargs["customized_inc"] = [1]
        kwargs["tetra_discontinuities"] = False
        kwargs["activated_plasticity"] = True
        kwargs["welds_throat_limit"] = True
        kwargs["approach"] = "kinematic"

        if sensitivity:
            kwargs["sensitivity_analysis"] = "true"
            kwargs["sensitivity_regions"] = json.dumps(
                [PARAM_CONFIG[p]['sens'] for p in params_names]
            )

        kwargs["model_handle"] = model.GETHANDLEPTR()
        CetSOLV.SOLV(AnalysisName, iteration, path, **kwargs) #On relance le solveur avec le nouveau dsCad.

        # Lire le resultat
        metares_path = os.path.join(path, AnalysisName + "_0_kine.dsmetares") #on extrait l'addresse du fichier pour définir f
        with open(metares_path, 'r') as f: #f est le fichier créé par open, et on a with donc enter de fichier = donne accès au fichier (accès via f, toujours mettre as f) puis exit : ferme le fichier (qui reste lié à f)
            d = json.load(f) #chargement du fichier .dsmetares
        g_HF=d['info']['Primal_bound'][0] -1
        if save_history:
            etat._socp_call_counter[0] += 1
            _save_socp_outputs(path, AnalysisName,
                               prefix_tag=f"HF_{etat._socp_call_counter[0]:03d}",
                               u1=float(u[0]), u2=float(u[1]),
                               p_vals=[float(x_point[j]) for j in range(n_var)])
        grad_HF_X=[None]*n_var
        grad_HF_U=[None]*n_var
        if sensitivity and 'Sensitivity' in d['info']:
            print(f"les sensibilités sont calculées pour les elements : {d['info']['Sensitivity'].items()}")
            for k, v in d['info']['Sensitivity'].items():
                p = _sens_key_to_param(k)
                if p is not None:
                    grad_HF_X[params_names.index(p)] = v
                if all(grad_HF_X[i] is not None for i in range(n_var)):
                    break
            J_Tinv = T_inv.gradient(u)
            J_Tinv_T = J_Tinv.transpose()
            grad_HF_U = J_Tinv_T * ot.Point(grad_HF_X)
        if sensitivity and any(v is None for v in grad_HF_U):
            raise ValueError(f"run_HF : sensibilité demandée mais grad_HF_U contient None — vérifier que STRAINS a bien calculé les sensibilités. grad_HF_X={grad_HF_X}")
        _append_point_log(etat._point_log_phase[0], u, x_point, g_HF)
        _mem_after = _psutil_hf.Process(os.getpid()).memory_info().rss / 1024 / 1024
        etat._run_HF_count[0] += 1
        if etat._run_HF_count[0] <= 2:  # print seulement les 2 premiers appels
            print(f"[MEM run_HF #{etat._run_HF_count[0]}] avant={_mem_before:.0f} MB  apres={_mem_after:.0f} MB  delta={_mem_after-_mem_before:+.0f} MB", flush=True)
        return g_HF, grad_HF_U, grad_HF_X

    # --- DOE PARALLELE ---
    def run_DOE_parallel(base_modelname, SOL, params_names, n_workers):
        """Parallelise les SOCP du DOE via subprocesses independants.
        Chaque worker = launcher3.py relance en mode _DOE_WORKER sur une copie .ds isolee."""
        import subprocess as _sp
        base_ds = storage + base_modelname + ".ds"
        npts = len(SOL)
        n_workers = max(1, min(n_workers, npts))
        threads_per = max(1, 32 // n_workers)
        batches = [[] for _ in range(n_workers)]
        for i in range(npts):
            batches[i % n_workers].append(i)
        print(f"  [DOE PARALLELE] {npts} pts -> {n_workers} workers (MKL={threads_per} threads/worker)", flush=True)
        procs = []
        for w, idxs in enumerate(batches):
            if not idxs:
                continue
            wname = base_modelname + ".ds\\_doe_workers\\doew%d" % w
            wds = storage + wname + ".ds"
            os.makedirs(wds, exist_ok=True)
            shutil.copy2(base_ds + "\\dsCad.txt", wds + "\\dsCad.txt")
            shutil.copy2(base_ds + "\\dsLoad.txt", wds + "\\dsLoad.txt")
            task = {"points": [dict({"idx": i}, **{p: float(SOL[i][p]) for p in params_names}) for i in idxs]}
            task_file = wds + "\\_doe_task.json"
            out_file = wds + "\\_doe_out.json"
            with open(task_file, "w") as _f:
                json.dump(task, _f)
            if os.path.exists(out_file):
                os.remove(out_file)
            env = dict(os.environ,
                       _DOE_WORKER=task_file, _DOE_OUT=out_file, _DOE_WORKER_MODELNAME=wname,
                       _DOE_MAIN_DS=base_ds,
                       _FIAB_LOG_REDIRECTED="1",
                       MKL_NUM_THREADS=str(threads_per), OMP_NUM_THREADS=str(threads_per))
            wlog = open(wds + "\\_doe_worker.log", "w")
            print(f"    -> worker {w}: points {idxs}", flush=True)
            p = _sp.Popen([sys.executable, os.path.join(path_dir, "code", "launcher3.py")],
                          env=env, stdout=wlog, stderr=_sp.STDOUT, cwd=wds)
            procs.append((p, out_file, wlog, w, idxs))
        for p, out_file, wlog, w, idxs in procs:
            rc = p.wait(); wlog.close()
            print(f"    <- worker {w} fini (rc={rc})", flush=True)
        for p, out_file, wlog, w, idxs in procs:
            if not os.path.exists(out_file):
                raise RuntimeError(f"[DOE PARALLELE] worker {w} sans sortie {out_file} (voir _doe_worker.log)")
            res = json.load(open(out_file))
            for i_str, d in res.items():
                i = int(i_str)
                SOL[i]['g'] = d['g']
                for q in params_names:
                    SOL[i][f'dg_{q}'] = d.get(f'dg_{q}')
            print("    collecte worker {}: ".format(w) + ", ".join(f"pt{i} g={SOL[i]['g']:.4f}" for i in idxs), flush=True)
        return SOL

    # --- HF GRILLE PARALLELE ---
    def run_HF_grid_parallel(u_points, n_workers=3):
        """Calcule g sur une liste de points U en parallele via subprocesses.
        Meme mecanisme que run_DOE_parallel (workers _DOE_WORKER).
        Retourne une liste de g (meme ordre que u_points)."""
        import subprocess as _sp
        dist_X = dist_jointe()
        T_inv = dist_X.getInverseIsoProbabilisticTransformation()
        # Convertir U -> X pour chaque point
        SOL = []
        for u in u_points:
            x = T_inv(ot.Point(list(u)))
            SOL.append({p: float(x[j]) for j, p in enumerate(params_names)})
        npts = len(SOL)
        n_workers = max(1, min(n_workers, npts))
        threads_per = max(1, 32 // n_workers)
        base_ds = storage + modelname + ".ds"
        batches = [[] for _ in range(n_workers)]
        for i in range(npts):
            batches[i % n_workers].append(i)
        print(f"  [HF GRID PARALLELE] {npts} pts -> {n_workers} workers (MKL={threads_per})", flush=True)
        procs = []
        for w, idxs in enumerate(batches):
            if not idxs:
                continue
            wname = modelname + ".ds\\_hf_workers\\hfw%d" % w
            wds = storage + wname + ".ds"
            os.makedirs(wds, exist_ok=True)
            shutil.copy2(base_ds + "\\dsCad.txt", wds + "\\dsCad.txt")
            shutil.copy2(base_ds + "\\dsLoad.txt", wds + "\\dsLoad.txt")
            task = {"points": [dict({"idx": i}, **{p: float(SOL[i][p]) for p in params_names}) for i in idxs]}
            task_file = wds + "\\_doe_task.json"
            out_file = wds + "\\_doe_out.json"
            with open(task_file, "w") as _f:
                json.dump(task, _f)
            if os.path.exists(out_file):
                os.remove(out_file)
            env = dict(os.environ,
                       _DOE_WORKER=task_file, _DOE_OUT=out_file, _DOE_WORKER_MODELNAME=wname,
                       _DOE_MAIN_DS=base_ds,
                       _FIAB_LOG_REDIRECTED="1",
                       MKL_NUM_THREADS=str(threads_per), OMP_NUM_THREADS=str(threads_per))
            wlog = open(wds + "\\_hf_worker.log", "w")
            print(f"    -> hf_worker {w}: {len(idxs)} points", flush=True)
            p = _sp.Popen([sys.executable, os.path.join(path_dir, "code", "launcher3.py")],
                          env=env, stdout=wlog, stderr=_sp.STDOUT, cwd=wds)
            procs.append((p, out_file, wlog, w, idxs))
        g_results = [None] * npts
        for p, out_file, wlog, w, idxs in procs:
            rc = p.wait(); wlog.close()
            print(f"    <- hf_worker {w} fini (rc={rc})", flush=True)
        for p, out_file, wlog, w, idxs in procs:
            if not os.path.exists(out_file):
                raise RuntimeError(f"[HF GRID PARALLELE] worker {w} sans sortie (voir _hf_worker.log)")
            res = json.load(open(out_file))
            for i_str, d in res.items():
                g_results[int(i_str)] = d['g']
            print(f"    collecte hf_worker {w}: {len(idxs)} pts", flush=True)
        return g_results

    # --- DOE cache ---
    _DOE_CACHE_FILE = os.path.join(_path_ds, "doe_cache.json")

    def _load_doe_cache():
        if not config_is_identical:
            return None
        if not os.path.exists(_DOE_CACHE_FILE):
            print(f"[DOE CACHE] aucun cache ({_DOE_CACHE_FILE}) -> calcul DOE", flush=True)
            return None
        try:
            d = json.load(open(_DOE_CACHE_FILE))
            _n0_cache = d.get('n0', len(d.get('xt', [])))
            if _n0_cache != n0:
                print(f"[DOE CACHE] n0 different (cache={_n0_cache}, courant={n0}) -> recalcul DOE", flush=True)
                return None
            if not d.get('complet', False):
                print(f"[DOE CACHE] cache incomplet (complet=False) -> recalcul DOE", flush=True)
                return None
            print(f"[DOE CACHE] charge depuis {_DOE_CACHE_FILE} (n0={n0}, complet -> 0 SOCP DOE)", flush=True)
            return np.array(d["xt"]), np.array(d["yt"]), np.array(d["all_grad"])
        except Exception as e:
            print(f"[DOE CACHE] lecture echouee ({type(e).__name__}: {e}) -> recalcul DOE", flush=True)
        return None

    def _save_doe_cache(xt, yt, all_grad):
        try:
            json.dump({"n0": n0, "complet": True,
                       "xt": np.asarray(xt).tolist(),
                       "yt": np.asarray(yt).tolist(),
                       "all_grad": np.asarray(all_grad).tolist()},
                      open(_DOE_CACHE_FILE, "w"), indent=1)
            print(f"[DOE CACHE] sauve (complet) dans {_DOE_CACHE_FILE}", flush=True)
        except Exception as e:
            print(f"[DOE CACHE] sauvegarde echouee ({type(e).__name__}: {e})", flush=True)

    def _save_doe_cache_incremental(SOL, n_done):
        """Sauvegarde incrementale : ecrit les n_done premiers points de SOL.
        Gradients deja en U (converties par run_one_SOL). Pas de cle 'complet'."""
        try:
            _xt = [SOL[i]['_u'] for i in range(n_done)]
            _yt = [[SOL[i]['g']] for i in range(n_done)]
            _ag = [[SOL[i].get(f'dg_{p}', 0.0) for p in params_names] for i in range(n_done)]
            json.dump({"n0": n0, "complet": False, "n_completed": n_done,
                       "xt": _xt, "yt": _yt, "all_grad": _ag},
                      open(_DOE_CACHE_FILE, "w"), indent=1)
            print(f"[DOE CACHE INCR] {n_done}/{len(SOL)} pts sauves", flush=True)
        except Exception as e:
            print(f"[DOE CACHE INCR] echoue ({type(e).__name__}: {e})", flush=True)

    # --- SIGNATURE INFORMATIVE (utilisee par le dump restart, pas par le DOE cache) ---
    def _doe_cache_sig():
        return {"n0": n0, "params": list(params_names), "n_var": n_var, "modelname": modelname}
    # --- DUMP RESTART ---
    _RESTART_STATE_FILE = os.path.join(_path_ds, "restart_state.json")
    def _save_restart_state(xt, yt, all_grad, xt_eff, best_result, best_sp, modes, result_IS):
        def _u_beta(r):
            try:
                return {"u_star": [float(v) for v in np.array(r.getStandardSpaceDesignPoint())],
                        "beta": float(r.getHasoferReliabilityIndex())}
            except Exception:
                return None
        st = {}
        try:
            st["signature"] = _doe_cache_sig()
            st["modele"]    = modele
            st["timestamp"] = etat.timestamp
            try:    st["max_degree"] = int(max_degree)
            except Exception: st["max_degree"] = None
            st["xt"]       = np.asarray(xt).tolist()       if xt       is not None else None
            st["yt"]       = np.asarray(yt).tolist()       if yt       is not None else None
            st["all_grad"] = np.asarray(all_grad).tolist() if all_grad is not None else None
            st["xt_eff"]   = [np.asarray(p).tolist() for p in xt_eff] if xt_eff else []
            st["n_doe"]    = n0
            st["n_total"]  = int(len(xt)) if xt is not None else 0
            _prev_tot = sum(etat._round_sizes_prev) if etat._round_sizes_prev else 0
            if etat._enrich_round > 0:
                st["round_sizes"] = list(etat._round_sizes_prev) + [int(len(xt)) - _prev_tot]
            else:
                st["round_sizes"] = [int(len(xt))] if xt is not None else []
            st["enrich_round"]     = int(etat._enrich_round)
            st["round_boundaries"] = list(np.cumsum([0] + st["round_sizes"]).astype(int).tolist())
            st["hist_EFF"]     = [float(v) for v in etat._eff_history_EFF]
            st["hist_BB"]      = [None if v is None else float(v) for v in etat._eff_history_BB]
            st["hist_BS"]      = [None if v is None else float(v) for v in etat._eff_history_BS]
            st["hist_theta"]   = [[float(x) for x in t] for t in etat._eff_history_theta]
            st["hist_beta_IS"] = [None if v is None else float(v) for v in etat._eff_history_beta_IS]
            try:    st["hf_2d_grid"] = etat.hf_2d_grid_fixed
            except Exception: st["hf_2d_grid"] = None
            st["best_sp"]     = [float(v) for v in np.array(best_sp)] if best_sp is not None else None
            st["best_result"] = _u_beta(best_result) if best_result is not None else None
            st["modes"]       = [_u_beta(m) for m in modes] if modes else []
            try:
                st["IS"] = {"Pf": float(result_IS.getProbabilityEstimate())} if result_IS is not None else None
            except Exception:
                st["IS"] = None
            json.dump(st, open(_RESTART_STATE_FILE, "w"), indent=1)
            print(f"[RESTART DUMP] etat sauve dans {_RESTART_STATE_FILE} "
                  f"(n_total={st['n_total']}, n_eff={len(st['xt_eff'])}, "
                  f"hist_EFF={len(st['hist_EFF'])}, modes={len(st['modes'])})", flush=True)
        except Exception as e:
            print(f"[RESTART DUMP] sauvegarde echouee ({type(e).__name__}: {e})", flush=True)

    # --- LOG INCREMENTAL PAR POINT ---
    _POINT_LOG_FILE = os.path.join(_path_ds, "points_log.jsonl")
    def _append_point_log(phase, u, x, g):
        try:
            _u = list(u) if u is not None else []
            _x = list(x) if x is not None else []
            rec = {"phase": phase, "round": etat._point_log_round[0],
                   "g": None if g is None else float(g),
                   "lambda": None if g is None else float(g) + 1.0}
            for i, p in enumerate(params_names):
                rec[f"u_{p}"] = float(_u[i]) if i < len(_u) else None
                rec[f"x_{p}"] = float(_x[i]) if i < len(_x) else None
            with open(_POINT_LOG_FILE, "a") as _pf:
                _pf.write(json.dumps(rec) + "\n")
        except Exception as e:
            print(f"[POINT LOG] append echoue ({type(e).__name__}: {e})", flush=True)

    # --- DOE ---
    def build_DOE(n_doe=n0, eval_hf=True):
        if not do_HF and eval_hf:
            _cached = _load_doe_cache()
            if _cached is not None:
                return _cached
        dist_X   = dist_jointe()
        T     = dist_X.getIsoProbabilisticTransformation()
        T_inv = dist_X.getInverseIsoProbabilisticTransformation()
        dist_U = ot.JointDistribution([ot.Uniform(eff_bounds_min[i], eff_bounds_max[i]) for i in range(n_var)])
        lhs    = ot.LHSExperiment(dist_U, n_doe)
        sa     = ot.SimulatedAnnealingLHS(lhs, ot.SpaceFillingMinDist())
        U_doe  = sa.generate()
        if print_DOE:
                print("U_doe_fixed = ot.Sample([")
                for i in range(U_doe.getSize()):
                    vals = []
                    for j in range(U_doe.getDimension()):
                        v = U_doe[i][j]
                        vals.append(f' {v:.16f}' if v >= 0 else f'{v:.16f}')
                    print(f"    [{', '.join(vals)}],")
                print("])", flush=True)
        X_doe  = T_inv(U_doe)
        xt = np.array(U_doe)
        if not do_HF and eval_hf:
            SOL = [{} for _ in range(n_doe)]
            for i in range(n_doe):
                for j in range(n_var):
                    SOL[i][params_names[j]] = X_doe[i][j]
            if n_workers_DOE and n_workers_DOE > 1:
                SOL = run_DOE_parallel(modelname, SOL, params_names, n_workers_DOE)
            else:
                SOL = run_one_SOL(modelname, SOL, params_names, sensitivity=True, with_sens_dict=None)
            # run_one_SOL a deja converti les gradients en U
            yt = np.array([SOL[i]['g'] for i in range(n_doe)]).reshape(-1, 1)
            all_grad = np.array([[SOL[i].get(f'dg_{p}', 0.0) for p in params_names] for i in range(n_doe)])
            for i in range(n_doe):
                _append_point_log("DOE", list(U_doe[i]), list(X_doe[i]), SOL[i]['g'])
            if print_DOE:
                print("yt_doe = [")
                for i in range(n_doe):
                    print(f"    {yt[i][0]:.16f},")
                print("]", flush=True)
                print("all_grad_doe = [")
                for i in range(n_doe):
                    print(f"    [{all_grad[i][0]:.10f}, {all_grad[i][1]:.10f}],")
                print("]", flush=True)
            _save_doe_cache(xt, yt, all_grad)
            return xt, yt, all_grad
        return xt

    # build_starting_points, build_Y_aug : voir surrogate_pure.py

    # --------------------------------------------------------------------------- #
    # FONCTION ANALYTIQUE DE REFERENCE                                            #
    # --------------------------------------------------------------------------- #
    # FONCTION ANALYTIQUE                                                         #
    # --------------------------------------------------------------------------- #
    # FONCTIONS LIEES AU MODELE HF                                                #

    # --- Wrapper OpenTURNS avec gradients analytiques ---
    class HFFunction(ot.OpenTURNSPythonFunction):
        def __init__(self):
            super().__init__(n_var, 1)
            self._cache_u    = None
            self._cache_g    = None
            self._cache_grad = None
            self.n_hf_calls  = 0  # compteur pour vérification

        def _run_if_needed(self, u):
            u_arr = np.array(u)
            if self._cache_u is None or not np.allclose(u_arr, self._cache_u, atol=1e-12):
                g, grad_U, _ = run_HF(u)
                self._cache_u    = u_arr.copy()
                self._cache_g    = float(g)
                self._cache_grad = [float(grad_U[i]) for i in range(n_var)]
                self.n_hf_calls += 1
                print(f"[HF #{self.n_hf_calls:3d}] u=[{u_arr[0]:+.4f}, {u_arr[1]:+.4f}]  g={g:+.6f}  grad=[{float(grad_U[0]):+.6f}, {float(grad_U[1]):+.6f}]", flush=True)

        def _exec(self, u):
            self._run_if_needed(u)
            return [self._cache_g]

        def _gradient(self, u):
            self._run_if_needed(u)
            # Format OpenTURNS : (n_var, 1)
            return [[g] for g in self._cache_grad]

    # build_metamodel_PCE, calculate_PCE, PCKRGFunction, oldGEPCKFunction,
    # GEPCKFunction, BoundSurrogateFunction : voir surrogate_pure.py

    # --------------------------------------------------------------------------- #
    # projection_surrogate : voir form_is.py

    # build_metamodel_KRG, build_metamodel_GEK, GEKPLSFunction : voir surrogate_pure.py

    # --------------------------------------------------------------------------- #
    # FONCTIONS POUR FORM                                                         #
    def init_g_ot(g_ot, sigma_func, xt, yt, all_grad, fixed_fm=None):
        """
        Cette fonction génère xt, yt, all_grad si xt n'est pas vide puis
        contruit un metamodele à partir de ces points. Dans le cas HF, elle
        créé uniquement une fonction OT. Retourne g_ot, sigma_func, xt, yt, all_grad.
        Si fixed_fm est fourni (refit KB) : theta et polynomes fixes du fit precedent.
        """

        if do_KRG:
            if xt is None: xt, yt, all_grad = build_DOE()
            g_ot, result = build_metamodel_KRG(xt, yt)
            sigma_func = lambda u: float(np.sqrt(result.getConditionalMarginalVariance(ot.Point(list(u)))))

        elif do_GEK:
            if xt is None: xt, yt, all_grad = build_DOE()
            sm_GEK   = build_metamodel_GEK(xt, yt, all_grad)
            gek_impl = GEKPLSFunction(sm_GEK)          
            g_ot     = ot.Function(gek_impl)
            sigma_func = gek_impl._exec_sigma

        elif do_PCKRG:
            if xt is None: xt, y_hf, all_grad_hf = build_DOE()                                            # on fait les calculs HF sur les points du DOE
            else:          y_hf, all_grad_hf = yt, all_grad
            g_ot_PCE = build_metamodel_PCE(xt, y_hf)                                                      # on déduit le métamodèle PCE
            y_PCE, all_grad_PCE = calculate_PCE(xt, y_hf, all_grad_hf, g_ot_PCE)                              # on calcule la composante PCE à partir des valeurs hf
            yr, all_grad_r = y_hf-y_PCE, all_grad_hf-all_grad_PCE                                         # on construit le residu
            gr_ot, result_r = build_metamodel_KRG(xt, yr)                                                 # on construit le surrogate sur le residu
            sigma_func = lambda u: float(np.sqrt(result_r.getConditionalMarginalVariance(ot.Point(list(u)))))  # on calcule sigma_func (locale donc sur surrogate)
            g_ot = ot.Function(PCKRGFunction(g_ot_PCE, gr_ot))                                            # on wrappe la somme du surrogate et du PCE
            yt, all_grad = y_hf, all_grad_hf                                                              # on stocke les valeurs hf pour si warmstart

        elif do_old_GEPCK:
            if xt is None: xt, y_hf, all_grad_hf = build_DOE()
            else:          y_hf, all_grad_hf = yt, all_grad
            g_ot_PCE = build_metamodel_PCE(xt, y_hf)
            y_PCE, all_grad_PCE = calculate_PCE(xt, y_hf, all_grad_hf, g_ot_PCE) 
            yr, all_grad_r = y_hf-y_PCE, all_grad_hf-all_grad_PCE 
            smr_GEK = build_metamodel_GEK(xt, yr, all_grad_r)
            gepck_impl = oldGEPCKFunction(g_ot_PCE, smr_GEK)
            g_ot  = ot.Function(gepck_impl)
            sigma_func = gepck_impl._exec_sigma
            yt, all_grad = y_hf, all_grad_hf

        elif do_GEPCK:
            if xt is None: xt, yt, all_grad = build_DOE()
            _marginals = [{'Type': 'Gaussian', 'Parameters': [0.0, 1.0]}] * n_var
            _copula    = {'Type': 'Independent', 'Parameters': np.eye(n_var)}
            if fixed_fm is not None:
                _prev_theta = fixed_fm['Kriging'][0]['theta']
                _prev_npoly = fixed_fm['NumberOfPoly']
                _prev_idx   = fixed_fm['idxranking'][0][:_prev_npoly]
                _prev_poly  = fixed_fm['AllIndices'][0][np.array(_prev_idx), :]
                _prev_types = fixed_fm['PolyTypes']
                _opts = {'Mode': 'sequential',
                         'PolyIndices': _prev_poly,
                         'PolyTypes': _prev_types,
                         'Kriging': {'Optim': {'Method': 'none',
                                               'InitialValue': _prev_theta}}}
            else:
                _opts = {'Mode': 'optimal',
                         'PCE': {'Degree': list(range(1, max_degree + 1)), 'Method': 'LARS'}}
            _Y_aug = build_Y_aug(yt, all_grad)
            print(f"=== GEPCK fit N={len(xt)}{' [KB]' if fixed_fm else ''} ===", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _fm = fit_gepck(xt, _Y_aug, _opts, _marginals, _copula)
            print(f"  LOO={_fm['Error'][0]['LOO']:.4e}  n_poly={_fm['NumberOfPoly']}  theta={_fm['Kriging'][0]['theta']}", flush=True)
            _final_idx   = _fm['idxranking'][0][:_fm['NumberOfPoly']]
            _sel_indices = _fm['AllIndices'][0][np.array(_final_idx), :]
            _beta_pce    = np.array(_fm['Kriging'][0]['beta']).ravel()
            _terms = []
            for _mi, _coef in zip(_sel_indices, _beta_pce):
                _parts = [f"H{int(_mi[k])}(u{k+1})" for k in range(len(_mi)) if int(_mi[k]) > 0]
                _terms.append(f"{_coef:+.4f}*{'*'.join(_parts) if _parts else '1'}")
            print(f"  GEPCK PCE termes : {' '.join(_terms)}", flush=True)
            etat._gepck_pce_label = ' '.join(_terms)
            etat._gepck_loo       = _fm['Error'][0]['LOO']
            etat._eff_history_theta.append(list(_fm['Kriging'][0]['theta']))
            gepck_impl = GEPCKFunction(_fm)
            g_ot       = ot.Function(gepck_impl)
            sigma_func = gepck_impl._exec_sigma

        elif do_HF:
            if xt is None: xt = build_DOE()
            g_ot = ot.Function(HFFunction())
            yt, all_grad = None, None
        
        return g_ot, sigma_func, xt, yt, all_grad

    def init_surrogate():
        """
        Construit le DOE HF et retourne xt, yt, all_grad.
        Version allégée de init_g_ot sans wrapper OpenTURNS — pour les pipelines
        qui n'utilisent pas FORM directement (GEPCK 5 branches, etc.).
        sigma_func et event seront ajoutés ici quand FORM sera intégré.
        """
        xt, yt, all_grad = build_DOE()
        return xt, yt, all_grad

    def init_FORM(g_ot, sigma_func, xt, yt, all_grad):
        """
        Cette fonction créé un metamodele si aucun existant et calcule l'event FORM.
        Elle retourne metamodele et xt,yt,all_grad et l'event.
        """
        # --- Événement de défaillance ---
        distribution = ot.JointDistribution([ot.Normal(0, 1)] * n_var)
        X = ot.RandomVector(distribution)
        g_ot, sigma_func, xt, yt, all_grad = init_g_ot(g_ot, sigma_func, xt, yt, all_grad)
        Y = ot.CompositeRandomVector(g_ot, X) if g_ot is not None else None
        event = ot.ThresholdEvent(Y, ot.Less(), 0.0) if Y is not None else None
        return event, g_ot, sigma_func, xt, yt, all_grad

    # FORM_all_modes : voir form_is.py
    
    # --- Warm-start FORM depuis les points du DOE ---
    def FORM_warm_start(modes, best_sps, g_ot, sigma_func, xt, yt, all_grad, tol=0.2):
        """
        Cette fonction reçoit des résultats FORM et le DOE utilisé, et déclenche warm_start si besoin 
        - elle renvoie la liste de modes/ best_sps mise à jour mais ne renvoie pas le nouveau DOE pour 
        l'instant (choix facilement modifiable).
        """
        if len(modes)>0:
            u_star = modes[0].getStandardSpaceDesignPoint()
            g_val = g_ot(ot.Point(u_star))[0] if g_ot is not None else None

            if g_val is not None and abs(g_val) > tol:
                # -- on fait warm start uniquement si on est au dessus de 0.2, sinon, on accepte le résultat. --
                xt = np.vstack([xt, [np.array(u_star)]])
                yt = np.vstack([yt, [[g_val]]])
                grad_ot  = g_ot.gradient(ot.Point(u_star))
                grad_val = np.array([[grad_ot[i, 0] for i in range(n_var)]])
                all_grad = np.vstack([all_grad, grad_val])
                event, g_ot, sigma_func, xt, yt, all_grad = init_FORM(g_ot, sigma_func, xt, yt, all_grad)
                starting_points = np.vstack([xt, [[0.0] * n_var]]) if do_multistart else np.array([[0.0] * n_var])
                modes, best_sps = FORM_all_modes(starting_points, tol_all_modes, event)
        return modes, best_sps

    # EFFFunction : voir surrogate_pure.py

    def _find_batch_EFF_points(g_ot, sigma_func, xt, yt, all_grad):
        """Trouve n_batch_EFF points EFF par Kriging Believer.
        Si n_batch_EFF=1 : equivalent a une seule maximisation EFF (pas de KB).
        Si n_batch_EFF>1 : maximise EFF, impute mu comme observation fictive,
        refit le surrogate a theta fixe, re-maximise, etc.
        Retourne la liste des points (list of ndarray) et la valeur EFF du premier."""
        # --- Premier point : maximisation EFF standard ---
        f = ot.Function(EFFFunction(g_ot, sigma_func))
        bounds = ot.Interval(eff_bounds_min, eff_bounds_max)
        problem = ot.OptimizationProblem(f, ot.Function(), ot.Function(), bounds)
        problem.setMinimization(False)
        algo_opti = ot.NLopt(problem, "GN_DIRECT")
        algo_opti.setStartingPoint([0.0] * n_var)
        algo_opti.setMaximumCallsNumber(n_NLopt_EFF)
        algo_opti.run()
        u1 = np.array(algo_opti.getResult().getOptimalPoint())
        eff_val = f(ot.Point(u1.tolist()))[0]
        batch = [u1]

        if n_batch_EFF <= 1:
            return batch, eff_val

        # --- Points suivants : Kriging Believer ---
        xt_kb  = np.copy(xt)
        yt_kb  = np.copy(yt)
        ag_kb  = np.copy(all_grad)
        g_kb   = g_ot
        s_kb   = sigma_func
        # recuperer le fm du surrogate courant pour fixer theta + polynomes
        _fm_kb = getattr(getattr(s_kb, '__self__', None), 'fm', None)

        for k in range(1, n_batch_EFF):
            # impute mu comme observation fictive
            u_prev = batch[-1]
            y_fictif = float(g_kb(ot.Point(u_prev.tolist()))[0])
            xt_kb = np.vstack([xt_kb, [u_prev]])
            yt_kb = np.vstack([yt_kb, [[y_fictif]]])
            # gradient fictif = gradient du surrogate
            if do_GEPCK:
                grad_fictif = np.array([[float(g_kb.gradient(ot.Point(u_prev.tolist()))[i, 0])
                                         for i in range(n_var)]])
            else:
                grad_fictif = np.zeros((1, n_var))
            ag_kb = np.vstack([ag_kb, grad_fictif])
            # refit surrogate a theta fixe + polynomes fixes (KB)
            g_kb, s_kb, xt_kb, yt_kb, ag_kb = init_g_ot(None, None, xt_kb, yt_kb, ag_kb, fixed_fm=_fm_kb)
            # re-maximise EFF sur le surrogate believer
            f_kb = ot.Function(EFFFunction(g_kb, s_kb))
            problem_kb = ot.OptimizationProblem(f_kb, ot.Function(), ot.Function(), bounds)
            problem_kb.setMinimization(False)
            algo_kb = ot.NLopt(problem_kb, "GN_DIRECT")
            algo_kb.setStartingPoint([0.0] * n_var)
            algo_kb.setMaximumCallsNumber(n_NLopt_EFF)
            algo_kb.run()
            u_k = np.array(algo_kb.getResult().getOptimalPoint())
            batch.append(u_k)
            print(f"  [KB {k+1}/{n_batch_EFF}] u={list(np.round(u_k, 3))}  EFF={f_kb(ot.Point(u_k.tolist()))[0]:.6f}", flush=True)

        return batch, eff_val

    def run_EFF(g_ot, sigma_func, xt, yt, all_grad):
        """
        Cette fonction reçoit le métamodele et ses paramètres, et l'améliore jusqu'à vérifier le critère EFF puis
        renvoie métamodèle+ paramètres mis à jour.
        """
        # --- Si aucune branche ne tourne, on ne fait rien ---
        if g_ot is None or do_HF:
            return g_ot, sigma_func, xt, yt, all_grad, []
        etat._point_log_phase[0] = "EFF"

        xt_eff = list(etat._restart_xt_eff) if restart_enrich_only else []

        def _form_is_iter(g_ot_i, label, sign=0, fm=None):
            """FORM depuis [0,0] + IS sur le surrogate courant. Affiche une ligne résumé.
            sign/fm : si _IS_PARALLEL et fm fourni, l'IS passe par adaptive_is (sonde+ramp-up)
            au lieu d'OpenTURNS. sign = 0 (g moyen) / +1 (g+2sigma) / -1 (g-2sigma)."""
            distribution_i = ot.JointDistribution([ot.Normal(0, 1)] * n_var)
            X_i  = ot.RandomVector(distribution_i)
            Y_i  = ot.CompositeRandomVector(g_ot_i, X_i)
            ev_i = ot.ThresholdEvent(Y_i, ot.Less(), 0.0)
            try:
                solver_i = ot.AbdoRackwitz()
                solver_i.setStartingPoint([0.0] * n_var)
                solver_i.setMaximumIterationNumber(n_max_FORM)
                solver_i.setCheckStatus(False)
                solver_i.setMaximumConstraintError(tol_FORM)
                form_i = ot.FORM(solver_i, ev_i)
                form_i.run()
                r_i = form_i.getResult()
            except Exception as e:
                print(f"  [{label}] FORM echoue ({type(e).__name__})", flush=True)
                return None, None
            beta_f  = r_i.getHasoferReliabilityIndex()
            pf_f    = r_i.getEventProbability()
            # --- IS adaptatif parallelisable (sonde + ramp-up) ---
            if _IS_PARALLEL and fm is not None:
                u_star  = list(r_i.getStandardSpaceDesignPoint())
                _state  = dict(xt=xt, yt=yt, all_grad=all_grad, max_degree=max_degree)
                _cap    = int(os.environ.get("_IS_CAP", str(n_IS)))
                _r      = adaptive_is(fm, _state, u_star, sign=sign,
                                      cov_target=cov_IS, cap_blocks=_cap,
                                      K=_IS_K, chunk=_IS_CHUNK, probe_blocks=_IS_PROBE)
                pf_IS   = _r['pf']
                beta_IS = float(-ot.Normal().computeQuantile(pf_IS)[0]) if pf_IS > 0 else float('nan')
                print(f"  [{label}] beta_FORM={beta_f:.4f}  Pf_FORM={pf_f:.3e}"
                      f" | Pf_IS={pf_IS:.3e}  beta_IS={beta_IS:.4f}  COV={_r['cov']:.3f}  [PAR:{_r['mode']}]", flush=True)
                print(f"  [IS DETAIL PAR] {label} : blocs={_r['n_blocks']} evals~{_r['n_evals']:,} "
                      f"COV={_r['cov']:.4f} (cible {cov_IS})", flush=True)
                return beta_IS, pf_IS
            # --- IS OpenTURNS classique (fallback) ---
            res_IS  = run_IS([r_i], ev_i)
            pf_IS   = res_IS.getProbabilityEstimate()
            beta_IS = float(-ot.Normal().computeQuantile(pf_IS)[0])
            cov_v   = res_IS.getCoefficientOfVariation()
            print(f"  [{label}] beta_FORM={beta_f:.4f}  Pf_FORM={pf_f:.3e}"
                  f" | Pf_IS={pf_IS:.3e}  beta_IS={beta_IS:.4f}  COV={cov_v:.3f}", flush=True)
            return beta_IS, pf_IS

        def _three_form_is(g_ot_i, sigma_func_i, label, b_mid_precalc=None):
            """FORM+IS sur g, g+2sigma, g-2sigma. Affiche les 3 lignes + ratio.
            b_mid_precalc=(beta_IS, pf_IS) : reutilise le mu deja calcule (evite 1 FORM+IS redondant).
            Retourne (ratio, pf_mid, pf_sup, pf_inf) ou (None, None, None, None)."""
            g_sup_i = ot.Function(BoundSurrogateFunction(g_ot_i, sigma_func_i, +1))
            g_inf_i = ot.Function(BoundSurrogateFunction(g_ot_i, sigma_func_i, -1))
            _FM = getattr(getattr(sigma_func_i, '__self__', None), 'fm', None)
            if b_mid_precalc is not None:
                b_mid, pf_mid = b_mid_precalc
                print(f"  [{label} mu] reutilise mu conv (pas de recalcul FORM/IS redondant)", flush=True)
            else:
                b_mid, pf_mid = _form_is_iter(g_ot_i, f"{label} mu", sign=0, fm=_FM)
            b_sup, pf_sup = _form_is_iter(g_sup_i, f"{label} sup", sign=+1, fm=_FM)
            b_inf, pf_inf = _form_is_iter(g_inf_i, f"{label} inf", sign=-1, fm=_FM)
            if b_mid is not None and b_sup is not None and b_inf is not None and b_mid != 0:
                ratio = abs(b_sup - b_inf) / abs(b_mid)
                print(f"  [{label}] |beta_IS_sup - beta_IS_inf| / beta_IS = {ratio:.4f}", flush=True)
                return ratio, pf_mid, pf_sup, pf_inf
            return None, None, None, None

        # --- FORM+IS sur le DOE initial (avant EFF) ---
        count_valid_BB   = 0
        count_valid_BS   = 0
        count_valid_both = 0
        # --- On résoud u = argmax(EFF) (batch KB si n_batch_EFF > 1) ---
        _batch_pts, _eff_val_init = _find_batch_EFF_points(g_ot, sigma_func, xt, yt, all_grad)
        u_opt = ot.Point(_batch_pts[0].tolist())
        f = ot.Function(EFFFunction(g_ot, sigma_func))
        _sigG = sigma_func(u_opt)
        _muG  = g_ot(ot.Point(u_opt))[0]
        _eps  = epsilon_factor * _sigG
        print(f"  EFF debug u_opt={list(np.round(np.array(u_opt),3))} : sigmaG={_sigG:.6f}  muG={_muG:.6f}  epsilon={_eps:.6f}", flush=True)
        print(f"  EFF initial : EFF(u_opt)={_eff_val_init:.6f}, tol={tol_EFF}", flush=True)

        iter_count = 0
        if EFF_criteria == 'BB':
            _cond = lambda: len(xt_eff) < n_max_EFF_points and abs(f(u_opt)[0]) > tol_EFF and count_valid_BB < 3
        elif EFF_criteria == 'BS':
            _cond = lambda: len(xt_eff) < n_max_EFF_points and abs(f(u_opt)[0]) > tol_EFF and count_valid_BS < 3
        elif EFF_criteria == 'both':
            _cond = lambda: len(xt_eff) < n_max_EFF_points and abs(f(u_opt)[0]) > tol_EFF and count_valid_both < 2
        elif EFF_criteria == 'at_least_one':
            _cond = lambda: len(xt_eff) < n_max_EFF_points and abs(f(u_opt)[0]) > tol_EFF and not (count_valid_BB >= 3 or count_valid_BS >= 3 or count_valid_both >= 2)
        else:
            _cond = lambda: len(xt_eff) < n_max_EFF_points and abs(f(u_opt)[0]) > tol_EFF


        _b_mid, _pf_mid_conv = _form_is_iter(g_ot, f"N={len(xt)} initial mu conv")
        if restart_enrich_only:
            list_beta_IS = list(etat._eff_history_beta_IS) + ([_b_mid] if _b_mid is not None else [])
        else:
            list_beta_IS = [_b_mid] if _b_mid is not None else []
        if not restart_enrich_only:
            etat._eff_history_BB.clear()
            etat._eff_history_BS.clear()
            etat._eff_history_Pf.clear()
        list_ratio_BB = etat._eff_history_BB   # alias — même objet
        list_ratio_BS = etat._eff_history_BS
        list_Pf = etat._eff_history_Pf
        etat._eff_history_EFF.append(f(u_opt)[0])   # EFF initial (avant ajout du 1er point)

        # --- Ratio BB initial (avant tout enrichissement) ---
        if print_Pf:
            _ratio_init_bb, _pf_mid_0, _pf_sup_0, _pf_inf_0 = _three_form_is(g_ot, sigma_func, f"N={len(xt)} initial BB")
            list_Pf.append({'mid': _pf_mid_0, 'sup': _pf_sup_0, 'inf': _pf_inf_0})
        if EFF_criteria in ('BB', 'both', 'at_least_one') and print_Pf:
            list_ratio_BB.append(_ratio_init_bb)
            if EFF_criteria in ('BB', 'at_least_one') and _ratio_init_bb is not None and _ratio_init_bb < tol_BB:
                count_valid_BB = 1

        while _cond():
            _sigG = sigma_func(u_opt)
            _muG  = g_ot(ot.Point(u_opt))[0]
            print(f"  EFF={f(u_opt)[0]:.6f} > {tol_EFF} -- u_opt={list(np.round(np.array(u_opt),3))}  sigmaG={_sigG:.6f}  muG={_muG:.6f}", flush=True)
            etat._eff_history_EFF.append(f(u_opt)[0])   # EFF apres rebuild a cette iteration

            # --- Calcul HF des points du batch ---
            _n_batch_actual = min(n_batch_EFF, n_max_EFF_points - len(xt_eff))
            _batch_to_eval = _batch_pts[:_n_batch_actual]
            if len(_batch_to_eval) > 1 and n_workers_DOE > 1:
                # Parallele : construire SOL, appeler run_DOE_parallel
                dist_X_eff = dist_jointe()
                T_inv_eff = dist_X_eff.getInverseIsoProbabilisticTransformation()
                _SOL_eff = []
                for _u_pt in _batch_to_eval:
                    _x_pt = T_inv_eff(ot.Point(list(_u_pt)))
                    _SOL_eff.append({p: float(_x_pt[j]) for j, p in enumerate(params_names)})
                _SOL_eff = run_DOE_parallel(modelname, _SOL_eff, params_names, min(n_workers_DOE, len(_batch_to_eval)))
                for _k, _u_pt in enumerate(_batch_to_eval):
                    _g_k = _SOL_eff[_k]['g']
                    _grad_k = [_SOL_eff[_k].get(f'dg_{p}', 0.0) for p in params_names]
                    xt_eff.append(np.array(_u_pt))
                    xt = np.vstack([xt, [np.array(_u_pt)]])
                    yt = np.vstack([yt, [[_g_k]]])
                    all_grad = np.vstack([all_grad, [_grad_k]])
                    print(f"[EFF HF {_k+1}/{len(_batch_to_eval)}] u={list(np.round(_u_pt, 4))}  g={_g_k:.6f}  grad_U={[round(v, 6) for v in _grad_k]}", flush=True)
            else:
                # Sequentiel : un seul point (ou n_workers=1)
                for _u_pt in _batch_to_eval:
                    g_val, grad_U, _ = run_HF(np.array(_u_pt))
                    xt_eff.append(np.array(_u_pt))
                    xt = np.vstack([xt, [np.array(_u_pt)]])
                    yt = np.vstack([yt, [[g_val]]])
                    grad_val = np.array([[float(grad_U[i]) for i in range(n_var)]])
                    all_grad = np.vstack([all_grad, grad_val])
                    print(f"[EFF HF] u={list(np.round(_u_pt, 10))}  g={g_val:.10f}  grad_U={[round(float(grad_U[i]), 10) for i in range(n_var)]}", flush=True)
            g_ot, sigma_func, xt, yt, all_grad = init_g_ot(g_ot, sigma_func, xt, yt, all_grad)

            # --- Suivi convergence beta_IS ---
            iter_count += 1
            _fm_eff = getattr(getattr(sigma_func, '__self__', None), 'fm', None)
            import time as _t_mod; _t0_is = _t_mod.perf_counter()
            _b_mid, _pf_mid_conv = _form_is_iter(g_ot, f"N={len(xt)} mu conv", fm=_fm_eff)
            print(f"  [TIMING _form_is_iter] dt={_t_mod.perf_counter()-_t0_is:.2f}s (fm={'oui' if _fm_eff else 'non'})", flush=True)

            # --- FORM+IS mid/sup/inf (conditionne par print_Pf) ---
            if print_Pf:
                _ratio_bb, _pf_mid, _pf_sup, _pf_inf = _three_form_is(g_ot, sigma_func, f"N={len(xt)} iter {iter_count}", b_mid_precalc=(_b_mid, _pf_mid_conv))
                list_Pf.append({'mid': _pf_mid, 'sup': _pf_sup, 'inf': _pf_inf})

            # --- Critere BB ---
            if EFF_criteria == 'BB':
                if not print_Pf:
                    _ratio_bb, _, _, _ = _three_form_is(g_ot, sigma_func, f"N={len(xt)} iter {iter_count}", b_mid_precalc=(_b_mid, _pf_mid_conv))
                if _ratio_bb is not None and _ratio_bb < tol_BB:
                    count_valid_BB += 1
                else:
                    count_valid_BB = 0
                list_ratio_BB.append(_ratio_bb)

            # --- Critere BS ---
            if EFF_criteria == 'BS':
                if _b_mid is not None and list_beta_IS and _b_mid != 0:
                    _ratio_conv = abs(_b_mid - list_beta_IS[-1]) / abs(_b_mid)
                    print(f"  [N={len(xt)}] |beta_IS - beta_IS_prec| / beta_IS = {_ratio_conv:.4f}", flush=True)
                    if _ratio_conv < tol_BS:
                        count_valid_BS += 1
                    else:
                        count_valid_BS = 0
                    list_ratio_BS.append(_ratio_conv)
                else:
                    count_valid_BS = 0
                    list_ratio_BS.append(None)

            # --- Critere both ---
            if EFF_criteria == 'both':
                if not print_Pf:
                    _ratio_bb, _, _, _ = _three_form_is(g_ot, sigma_func, f"N={len(xt)} iter {iter_count}", b_mid_precalc=(_b_mid, _pf_mid_conv))
                if _b_mid is not None and list_beta_IS and _b_mid != 0:
                    _ratio_bs = abs(_b_mid - list_beta_IS[-1]) / abs(_b_mid)
                    print(f"  [N={len(xt)} both] |beta_IS - beta_IS_prec| / beta_IS = {_ratio_bs:.4f}", flush=True)
                else:
                    _ratio_bs = None
                if (_ratio_bb is not None and _ratio_bb < tol_BB and
                        _ratio_bs is not None and _ratio_bs < tol_BS):
                    count_valid_both += 1
                else:
                    count_valid_both = 0
                list_ratio_BB.append(_ratio_bb)
                list_ratio_BS.append(_ratio_bs)

            # --- Critere at_least_one ---
            if EFF_criteria == 'at_least_one':
                if not print_Pf:
                    _ratio_bb, _, _, _ = _three_form_is(g_ot, sigma_func, f"N={len(xt)} iter {iter_count}", b_mid_precalc=(_b_mid, _pf_mid_conv))
                if _b_mid is not None and list_beta_IS and _b_mid != 0:
                    _ratio_bs = abs(_b_mid - list_beta_IS[-1]) / abs(_b_mid)
                    print(f"  [N={len(xt)} alo] |beta_IS - beta_IS_prec| / beta_IS = {_ratio_bs:.4f}", flush=True)
                else:
                    _ratio_bs = None
                if _ratio_bb is not None and _ratio_bb < tol_BB:
                    count_valid_BB += 1
                else:
                    count_valid_BB = 0
                if _ratio_bs is not None and _ratio_bs < tol_BS:
                    count_valid_BS += 1
                else:
                    count_valid_BS = 0
                if (_ratio_bb is not None and _ratio_bb < tol_BB and
                        _ratio_bs is not None and _ratio_bs < tol_BS):
                    count_valid_both += 1
                else:
                    count_valid_both = 0
                list_ratio_BB.append(_ratio_bb)
                list_ratio_BS.append(_ratio_bs)

            if _b_mid is not None:
                list_beta_IS.append(_b_mid)

            # --- Visu intermediaire apres ajout de point ---
            if print_EFF_progres:
                print_planche_EFF(g_ot, sigma_func, xt, xt_eff)

            # --- Dump restart incremental (kill-safe) ---
            etat._eff_history_beta_IS = list(list_beta_IS)
            _save_restart_state(xt, yt, all_grad, xt_eff, None, None, [], None)

            # --- On re-résoud u = argmax(EFF) (batch KB si n_batch_EFF > 1) ---
            _batch_pts, _ = _find_batch_EFF_points(g_ot, sigma_func, xt, yt, all_grad)
            u_opt = ot.Point(_batch_pts[0].tolist())
            f = ot.Function(EFFFunction(g_ot, sigma_func))

        _sigG2 = sigma_func(u_opt)
        _muG2  = g_ot(ot.Point(u_opt))[0]
        _eps2  = epsilon_factor * _sigG2
        if _sigG2 > 0:
            t1 = -_muG2 / _sigG2
            t2 = (_eps2 + _muG2) / _sigG2
            t3 = (_eps2 - _muG2) / _sigG2
            term1 = 2 * _muG2 * norm.cdf(t1)
            term2 = -(_eps2 + _muG2) * norm.cdf(-t2)
            term3 = (_eps2 - _muG2) * norm.cdf(t3)
            term4 = _sigG2 * (norm.pdf(t2) - norm.pdf(t3))
            print(f"  EFF converge debug : u_opt={list(np.round(np.array(u_opt),4))}  sigmaG={_sigG2:.8f}  muG={_muG2:.8f}  epsilon={_eps2:.8f}", flush=True)
            print(f"    t1={t1:.6f}  t2={t2:.6f}  t3={t3:.6f}", flush=True)
            print(f"    norm.cdf(t1)={norm.cdf(t1):.8e}  norm.cdf(-t2)={norm.cdf(-t2):.8e}  norm.cdf(t3)={norm.cdf(t3):.8e}", flush=True)
            print(f"    norm.pdf(t2)={norm.pdf(t2):.8e}  norm.pdf(t3)={norm.pdf(t3):.8e}", flush=True)
            print(f"    term1={term1:.8e}  term2={term2:.8e}  term3={term3:.8e}  term4={term4:.8e}", flush=True)
            print(f"    EFF = {term1+term2+term3+term4:.8e}", flush=True)
        else:
            print(f"  EFF converge debug : sigmaG=0 (modele interpolant exact au point u_opt)", flush=True)
        _exit_eff = abs(f(u_opt)[0]) <= tol_EFF
        _exit_bb   = count_valid_BB   >= 3 and EFF_criteria in ('BB', 'at_least_one')
        _exit_bs   = count_valid_BS   >= 3 and EFF_criteria in ('BS', 'at_least_one')
        _exit_both = count_valid_both >= 2 and EFF_criteria in ('both', 'at_least_one')
        if _exit_eff:
            _reason = "EFF"
        elif _exit_bb:
            _reason = "BB (3 iter valides)"
        elif _exit_bs:
            _reason = "BS (3 iter valides)"
        elif _exit_both:
            _reason = "both (2 iter valides)"
        else:
            _reason = "?"
        print(f"  EFF converge [{_reason}] : EFF(u_opt)={f(u_opt)[0]:.4f}"
              f"  count_valid_BB={count_valid_BB}  count_valid_BS={count_valid_BS}"
              f"  count_valid_both={count_valid_both}  ({len(xt_eff)} point(s) ajoutes)", flush=True)
        _fmt = lambda lst: [round(r, 4) if r is not None else None for r in lst]
        if list_ratio_BB:
            print(f"  [historique ratio BB] {_fmt(list_ratio_BB)}  tol={tol_BB}", flush=True)
        if list_ratio_BS:
            print(f"  [historique ratio BS] {_fmt(list_ratio_BS)}  tol={tol_BS}", flush=True)

        # --- BB informatif final (1 appel _three_form_is apres la boucle) ---
        if print_Pf:
            # _ratio_bb deja calcule par le dernier _three_form_is dans la boucle
            print(f"  [BB informatif final] ratio = {_ratio_bb}  tol_BB = {tol_BB}", flush=True)
        else:
            _ratio_bb, _, _, _ = _three_form_is(g_ot, sigma_func, "BB final", b_mid_precalc=(_b_mid, _pf_mid_conv))
            print(f"  [BB informatif final] ratio = {_ratio_bb}  tol_BB = {tol_BB}", flush=True)

        etat._eff_history_beta_IS = list(list_beta_IS)
        return g_ot, sigma_func, xt, yt, all_grad, xt_eff

    # --------------------------------------------------------------------------- #
    # FONCTIONS RESULTATS/ AFFICHAGE                                              #

    # _batch_mu_sigma, _eff_vectorized : voir visu_pure.py
    # =========================================================================

    # print_EFF_graphs, print_Pf_evolution, print_logPf_evolution : voir visu_pure.py

    # --- HF GRID CACHE ---
    _HF_CACHE_FILE       = os.path.join(_path_ds, "hf_grid_cache.json")
    _HF_CACHE_FILE_FINAL = os.path.join(_path_ds, "hf_grid_cache_final.json")

    def _load_hf_cache(n_grid_hf_local, cache_file, sd):
        if not config_is_identical:
            return None
        if not os.path.exists(cache_file):
            print(f"[HF CACHE] aucun cache ({cache_file}) -> calcul grille HF", flush=True)
            return None
        try:
            d = json.load(open(cache_file))
            _sd_cache = tuple(d['slice_def'][:2]) if 'slice_def' in d else None
            _sd_now = (sd[0], sd[1]) if sd is not None else (0, 1)
            if _sd_cache != _sd_now:
                print(f"[HF CACHE] coupe differente (cache={_sd_cache}, courant={_sd_now}) -> recalcul", flush=True)
                return None
            print(f"[HF CACHE] charge depuis {cache_file} (coupe OK -> 0 SOCP grille)", flush=True)
            return np.array(d['Z'])
        except Exception as e:
            print(f"[HF CACHE] lecture echouee ({type(e).__name__}: {e}) -> recalcul", flush=True)
        return None

    def _save_hf_cache(Z, n_grid_hf_local, cache_file, sd):
        try:
            _sd = sd if sd is not None else (0, 1, {})
            json.dump({'Z': Z.tolist(), 'slice_def': [_sd[0], _sd[1], {str(k): v for k, v in _sd[2].items()}]},
                      open(cache_file, 'w'), indent=1)
            print(f"[HF CACHE] sauve dans {cache_file}", flush=True)
        except Exception as e:
            print(f"[HF CACHE] sauvegarde echouee ({type(e).__name__}: {e})", flush=True)

    def _save_hf_cache_partial(Z_flat, n_total, cache_file, sd):
        """Sauvegarde incrementale de la grille HF (Z_flat peut contenir des None)."""
        try:
            _sd = sd if sd is not None else (0, 1, {})
            json.dump({'Z_flat': Z_flat, 'n_total': n_total, 'complet': False,
                       'slice_def': [_sd[0], _sd[1], {str(k): v for k, v in _sd[2].items()}]},
                      open(cache_file + '.partial', 'w'), indent=1)
        except Exception as e:
            print(f"[HF CACHE PARTIAL] sauvegarde echouee ({type(e).__name__}: {e})", flush=True)

    def _load_hf_cache_partial(cache_file, sd, n_total):
        """Charge le cache partiel. Retourne une liste Z_flat (avec None) ou None."""
        if not config_is_identical:
            return None
        partial_file = cache_file + '.partial'
        if not os.path.exists(partial_file):
            return None
        try:
            d = json.load(open(partial_file))
            _sd_cache = tuple(d['slice_def'][:2]) if 'slice_def' in d else None
            _sd_now = (sd[0], sd[1]) if sd is not None else (0, 1)
            if _sd_cache != _sd_now or d.get('n_total') != n_total:
                return None
            z = d['Z_flat']
            n_done = sum(1 for v in z if v is not None)
            print(f"[HF CACHE PARTIAL] reprise : {n_done}/{n_total} points deja calcules", flush=True)
            return z
        except Exception:
            return None

    def _compute_hf_grid_with_progress(grid_hf, n_grid_hf_local, context="",
                                        cache_file=None, sd=None, grid_var_name='hf_2d_grid_fixed'):
        """Calcule la grille HF point par point avec progress + ETA.
        Lecture/ecriture automatique d'un cache sidecar JSON.
        Sauvegarde incrementale dans cache_file.partial (reprise apres crash).
        Retourne Z (n_grid_hf x n_grid_hf)."""

        if cache_file is None:
            cache_file = _HF_CACHE_FILE
        if sd is None:
            sd = slice_def
        cached = _load_hf_cache(n_grid_hf_local, cache_file, sd)
        if cached is not None:
            _grid_dict = {'params': {'slice_def': sd, 'n_grid_hf': n_grid_hf_local}, 'Z': cached.tolist()}
            if grid_var_name == 'hf_2d_grid_fixed_final':
                etat.hf_2d_grid_fixed_final = _grid_dict
            else:
                etat.hf_2d_grid_fixed = _grid_dict
            return cached
        import time as _time_local
        etat._point_log_phase[0] = "HF"
        n_total = len(grid_hf)
        # Charger le cache partiel (reprise apres crash)
        Z_flat = _load_hf_cache_partial(cache_file, sd, n_total)
        if Z_flat is None:
            Z_flat = [None] * n_total
        _n_skipped = sum(1 for v in Z_flat if v is not None)
        _n_to_compute = n_total - _n_skipped
        _t_start = _time_local.perf_counter()
        print(f"\n##### HF GRID START: {n_grid_hf_local}x{n_grid_hf_local} = {n_total} points STRAINS ({context})"
              f" [skip {_n_skipped}, calcul {_n_to_compute}] #####", flush=True)
        _n_computed = 0
        for i, pt in enumerate(grid_hf):
            if Z_flat[i] is not None:
                continue
            _t_pt0 = _time_local.perf_counter()
            g_val = run_HF(pt)[0]
            Z_flat[i] = g_val
            _n_computed += 1
            _save_hf_cache_partial(Z_flat, n_total, cache_file, sd)
            _t_pt = _time_local.perf_counter() - _t_pt0
            _t_elapsed = _time_local.perf_counter() - _t_start
            _t_avg = _t_elapsed / _n_computed
            _t_eta = _t_avg * (_n_to_compute - _n_computed)
            _u_str = ', '.join(f'{pt[j]:+.3f}' for j in range(len(pt)))
            print(f"  [HF GRID {_n_skipped + _n_computed:2d}/{n_total}]  u=[{_u_str}]  g={g_val:+.4f}  "
                  f"dt={_t_pt:.0f}s  elapsed={_t_elapsed/60:.1f}min  ETA={_t_eta/60:.1f}min", flush=True)
        _t_total = (_time_local.perf_counter() - _t_start) / 60
        print(f"\n##### HF GRID DONE in {_t_total:.1f} min ({_n_computed} appels STRAINS, {_n_skipped} skip) #####\n", flush=True)
        Z = np.array(Z_flat, dtype=float).reshape(n_grid_hf_local, n_grid_hf_local)
        _grid_dict = {'params': {'slice_def': sd, 'n_grid_hf': n_grid_hf_local}, 'Z': Z.tolist()}
        if grid_var_name == 'hf_2d_grid_fixed_final':
            etat.hf_2d_grid_fixed_final = _grid_dict
        else:
            etat.hf_2d_grid_fixed = _grid_dict
        _save_hf_cache(Z, n_grid_hf_local, cache_file, sd)
        # supprimer le cache partiel
        _partial_file = cache_file + '.partial'
        if os.path.exists(_partial_file):
            os.remove(_partial_file)
        return Z

    # --- HF GRILLE FULL (n_var-D) ---
    _HF_FULL_CACHE_FILE = os.path.join(_path_ds, "hf_grid_full_cache.json")

    def _load_hf_grid_full():
        if not config_is_identical:
            return None
        if not os.path.exists(_HF_FULL_CACHE_FILE):
            return None
        try:
            d = json.load(open(_HF_FULL_CACHE_FILE))
            if d.get('n_var') != n_var or d.get('n_grid') != n_grid_hf:
                print(f"[HF FULL CACHE] dimensions differentes -> recalcul", flush=True)
                return None
            print(f"[HF FULL CACHE] charge depuis {_HF_FULL_CACHE_FILE} (0 SOCP)", flush=True)
            return np.array(d['Z'])
        except Exception as e:
            print(f"[HF FULL CACHE] lecture echouee ({type(e).__name__}: {e}) -> recalcul", flush=True)
        return None

    def _save_hf_grid_full(Z_full):
        try:
            json.dump({'Z': Z_full.tolist(), 'n_var': n_var, 'n_grid': n_grid_hf},
                      open(_HF_FULL_CACHE_FILE, 'w'), indent=1)
            print(f"[HF FULL CACHE] sauve dans {_HF_FULL_CACHE_FILE}", flush=True)
        except Exception as e:
            print(f"[HF FULL CACHE] sauvegarde echouee ({type(e).__name__}: {e})", flush=True)

    def _compute_hf_grid_full():
        """Calcule la grille HF complete (n_grid_hf^n_var points STRAINS)."""
        cached = _load_hf_grid_full()
        if cached is not None:
            etat._hf_grid_full[0] = cached
            axes = [np.linspace(u1_min, u1_max, n_grid_hf) for _ in range(n_var)]
            etat._hf_grid_full_axes[0] = axes
            return cached
        import time as _time_local
        etat._point_log_phase[0] = "HF_FULL"
        axes = [np.linspace(u1_min, u1_max, n_grid_hf) for _ in range(n_var)]
        grids = np.meshgrid(*axes, indexing='ij')
        grid_flat = np.column_stack([g.ravel() for g in grids])
        n_total = len(grid_flat)
        Z_flat = []
        _t_start = _time_local.perf_counter()
        print(f"\n##### HF FULL GRID START: {n_grid_hf}^{n_var} = {n_total} points STRAINS #####", flush=True)
        for i, pt in enumerate(grid_flat):
            _t_pt0 = _time_local.perf_counter()
            g_val = run_HF(pt)[0]
            Z_flat.append(g_val)
            _t_pt = _time_local.perf_counter() - _t_pt0
            _t_elapsed = _time_local.perf_counter() - _t_start
            _t_avg = _t_elapsed / (i + 1)
            _t_eta = _t_avg * (n_total - i - 1)
            _u_str = ', '.join(f'{pt[j]:+.3f}' for j in range(n_var))
            print(f"  [HF FULL {i+1:3d}/{n_total}]  u=[{_u_str}]  g={g_val:+.4f}  "
                  f"dt={_t_pt:.0f}s  elapsed={_t_elapsed/60:.1f}min  ETA={_t_eta/60:.1f}min", flush=True)
        _t_total = (_time_local.perf_counter() - _t_start) / 60
        print(f"\n##### HF FULL GRID DONE in {_t_total:.1f} min ({n_total} appels STRAINS) #####\n", flush=True)
        Z_full = np.array(Z_flat).reshape([n_grid_hf] * n_var)
        etat._hf_grid_full[0] = Z_full
        etat._hf_grid_full_axes[0] = axes
        _save_hf_grid_full(Z_full)
        return Z_full

    def _extract_hf_slice(sd):
        """Extrait une coupe 2D (n_grid_hf x n_grid_hf) depuis la grille full par interpolation."""
        from scipy.interpolate import RegularGridInterpolator
        idx_x, idx_y, fixed = sd
        axes = etat._hf_grid_full_axes[0]
        interp = RegularGridInterpolator(axes, etat._hf_grid_full[0], method='linear')
        ux = np.linspace(u1_min, u1_max, n_grid_hf)
        uy = np.linspace(u2_min, u2_max, n_grid_hf)
        UX, UY = np.meshgrid(ux, uy)
        pts = np.zeros((n_grid_hf * n_grid_hf, n_var))
        pts[:, idx_x] = UX.ravel()
        pts[:, idx_y] = UY.ravel()
        for idx, val in fixed.items():
            pts[:, idx] = val
        Z = interp(pts).reshape(n_grid_hf, n_grid_hf)
        return Z

    _HF_CUSTOM_CACHE_FILE = os.path.join(_path_ds, "hf_custom_cache.json")
    def _hf_from_custom_points(sd):
        """Calcule g par run_HF sur les coordonnees hf_custom_points [[u_s, u_fy], ...],
        puis interpole (griddata) sur une grille reguliere.
        Cache incremental : sauve apres chaque point, reprend apres crash.
        Retourne (Z_true, UX_hf, UY_hf) ou (None, None, None) si hf_custom_points est None."""
        if hf_custom_points is None:
            return None, None, None
        if etat._hf_custom_result[0] is not None:
            return etat._hf_custom_result[0]
        from scipy.interpolate import griddata
        import time as _time_local
        idx_x, idx_y, fixed = sd
        pts = np.array(hf_custom_points)   # (N, 2) : coordonnees U
        n_total = len(pts)

        # Charger cache final complet (skip tout le calcul)
        if config_is_identical and os.path.exists(_HF_CUSTOM_CACHE_FILE):
            try:
                _dc = json.load(open(_HF_CUSTOM_CACHE_FILE))
                if _dc.get('complet') and _dc.get('n_total') == n_total:
                    print(f"[HF CUSTOM] cache complet charge ({n_total} pts) -> 0 SOCP", flush=True)
                    g_arr = np.array(_dc['g_vals'], dtype=float)
                    _margin = 0.1
                    _n_interp = 50
                    ux_hf = np.linspace(pts[:, 0].min() - _margin, pts[:, 0].max() + _margin, _n_interp)
                    uy_hf = np.linspace(pts[:, 1].min() - _margin, pts[:, 1].max() + _margin, _n_interp)
                    UX_hf, UY_hf = np.meshgrid(ux_hf, uy_hf)
                    Z_true = griddata(pts, g_arr, (UX_hf, UY_hf), method='linear')
                    etat._hf_custom_result[0] = (Z_true, UX_hf, UY_hf)
                    return Z_true, UX_hf, UY_hf
            except Exception:
                pass

        # Charger cache partiel (reprise)
        _partial_file = _HF_CUSTOM_CACHE_FILE + '.partial'
        g_vals = [None] * n_total
        if config_is_identical and os.path.exists(_partial_file):
            try:
                _d = json.load(open(_partial_file))
                if _d.get('n_total') == n_total:
                    g_vals = _d['g_vals']
            except Exception:
                pass
        _n_skipped = sum(1 for v in g_vals if v is not None)
        _n_to_compute = n_total - _n_skipped
        print(f"[HF CUSTOM] {n_total} points, {_n_skipped} deja calcules, {_n_to_compute} a faire", flush=True)

        # Calculer g sur les points manquants
        _idx_todo = [i for i in range(n_total) if g_vals[i] is None]
        if _idx_todo:
            _pts_todo = [list(pts[i]) for i in _idx_todo]
            if n_workers_DOE and n_workers_DOE > 1 and len(_pts_todo) > 1:
                _g_todo = run_HF_grid_parallel(_pts_todo, n_workers=n_workers_DOE)
            else:
                etat._point_log_phase[0] = "HF_CUSTOM"
                _g_todo = []
                _t_start = _time_local.perf_counter()
                for _k, pt in enumerate(_pts_todo):
                    g_val = run_HF(pt)[0]
                    _g_todo.append(g_val)
                    # Save incremental
                    g_vals[_idx_todo[_k]] = g_val
                    try:
                        json.dump({'n_total': n_total, 'g_vals': g_vals},
                                  open(_partial_file, 'w'), indent=1)
                    except Exception:
                        pass
                    _t_elapsed = _time_local.perf_counter() - _t_start
                    _t_avg = _t_elapsed / (_k + 1)
                    _t_eta = _t_avg * (len(_pts_todo) - _k - 1)
                    print(f"  [HF CUSTOM {_n_skipped + _k + 1}/{n_total}]  u=[{pt[0]:+.3f}, {pt[1]:+.3f}]  g={g_val:+.4f}  "
                          f"ETA={_t_eta/60:.1f}min", flush=True)
            # Remplir g_vals depuis les resultats (parallele ou sequentiel)
            for _k, i in enumerate(_idx_todo):
                if g_vals[i] is None:
                    g_vals[i] = _g_todo[_k]
            print(f"[HF CUSTOM] {len(_idx_todo)} points calcules ({_n_skipped} skip)", flush=True)

        # Supprimer le cache partiel, sauver le cache final
        g_arr = np.array(g_vals, dtype=float)
        try:
            json.dump({'n_total': n_total, 'pts': pts.tolist(), 'g_vals': g_vals, 'complet': True},
                      open(_HF_CUSTOM_CACHE_FILE, 'w'), indent=1)
            if os.path.exists(_partial_file):
                os.remove(_partial_file)
        except Exception:
            pass

        # Griddata sur grille reguliere adaptee aux bornes des custom points
        _margin = 0.1
        _n_interp = 50
        ux_hf = np.linspace(pts[:, 0].min() - _margin, pts[:, 0].max() + _margin, _n_interp)
        uy_hf = np.linspace(pts[:, 1].min() - _margin, pts[:, 1].max() + _margin, _n_interp)
        UX_hf, UY_hf = np.meshgrid(ux_hf, uy_hf)
        Z_true = griddata(pts, g_arr, (UX_hf, UY_hf), method='linear')
        etat._hf_custom_result[0] = (Z_true, UX_hf, UY_hf)
        return Z_true, UX_hf, UY_hf

    def _get_hf_slice(sd, cache_file=None, grid_var_name='hf_2d_grid_fixed'):
        """Retourne Z_true (n_grid_hf x n_grid_hf) pour une coupe sd.
        Cascade : cache 2D memoire -> cache 2D disque -> grille full -> recalcul 2D."""

        if cache_file is None:
            cache_file = _HF_CACHE_FILE
        # 1. Cache 2D memoire
        _mem = etat.hf_2d_grid_fixed_final if grid_var_name == 'hf_2d_grid_fixed_final' else etat.hf_2d_grid_fixed
        if _mem is not None:
            return np.array(_mem['Z'])
        # 2. Cache 2D disque
        Z_cached = _load_hf_cache(n_grid_hf, cache_file, sd)
        if Z_cached is not None:
            _grid_dict = {'params': {'slice_def': sd, 'n_grid_hf': n_grid_hf}, 'Z': Z_cached.tolist()}
            if grid_var_name == 'hf_2d_grid_fixed_final':
                etat.hf_2d_grid_fixed_final = _grid_dict
            else:
                etat.hf_2d_grid_fixed = _grid_dict
            return Z_cached
        # 3. Grille full
        if etat._hf_grid_full[0] is not None:
            print(f"[HF SLICE] extraction depuis grille full pour coupe ({sd[0]},{sd[1]})", flush=True)
            Z = _extract_hf_slice(sd)
            _save_hf_cache(Z, n_grid_hf, cache_file, sd)
            _grid_dict = {'params': {'slice_def': sd, 'n_grid_hf': n_grid_hf}, 'Z': Z.tolist()}
            if grid_var_name == 'hf_2d_grid_fixed_final':
                etat.hf_2d_grid_fixed_final = _grid_dict
            else:
                etat.hf_2d_grid_fixed = _grid_dict
            return Z
        # 4. Recalcul 2D (49 SOCP)
        idx_x, idx_y, fixed = sd
        ux_hf = np.linspace(u1_min, u1_max, n_grid_hf)
        uy_hf = np.linspace(u2_min, u2_max, n_grid_hf)
        UX_hf, UY_hf = np.meshgrid(ux_hf, uy_hf)
        grid_hf = np.zeros((n_grid_hf * n_grid_hf, n_var))
        grid_hf[:, idx_x] = UX_hf.ravel()
        grid_hf[:, idx_y] = UY_hf.ravel()
        for idx, val in fixed.items():
            grid_hf[:, idx] = val
        return _compute_hf_grid_with_progress(grid_hf, n_grid_hf, context="get_hf_slice",
                    cache_file=cache_file, sd=sd, grid_var_name=grid_var_name)

    def print_planche_EFF(g_ot, sigma_func, xt, xt_eff):
        """Planche 3 graphiques cote a cote : EFF, sigma, g surrogate.
        Utilise la globale slice_def pour definir la coupe 2D."""

        _sd = slice_def if slice_def is not None else (0, 1, {})
        idx_x, idx_y, fixed = _sd
        n_added = len(xt_eff)

        # --- Grille commune (coupe 2D dans l'espace n_var-D) ---
        ux = np.linspace(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1, n_grid)
        uy = np.linspace(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1, n_grid)
        UX, UY = np.meshgrid(ux, uy)
        grid = np.zeros((n_grid * n_grid, n_var))
        grid[:, idx_x] = UX.ravel()
        grid[:, idx_y] = UY.ravel()
        for idx, val in fixed.items():
            grid[:, idx] = val

        # --- Z_eff, Z_sigma, Z_g (batch vectorise via BLAS multi-thread) ---
        mu_grid, sigma_grid = _batch_mu_sigma(g_ot, sigma_func, grid)
        Z_eff   = _eff_vectorized(mu_grid, sigma_grid, epsilon_factor).reshape(n_grid, n_grid)
        Z_sigma = sigma_grid.reshape(n_grid, n_grid)
        Z_g     = mu_grid.reshape(n_grid, n_grid) if g_ot is not None else None

        # --- Contour g=0 HF ---
        Z_true, UX_hf, UY_hf = None, None, None
        _sd = slice_def if slice_def is not None else (0, 1, {})
        if hf_custom_points is not None:
            Z_true, UX_hf, UY_hf = _hf_from_custom_points(_sd)
        elif print_HF:
            ux_hf = np.linspace(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1, n_grid_hf)
            uy_hf = np.linspace(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1, n_grid_hf)
            UX_hf, UY_hf = np.meshgrid(ux_hf, uy_hf)
            Z_true = _get_hf_slice(_sd, _HF_CACHE_FILE, 'hf_2d_grid_fixed')

        # --- Figure ---
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
        _pce_line   = f'\n{etat._gepck_pce_label}' if etat._gepck_pce_label else ''
        _theta_str  = '  theta=[' + ', '.join(f'{v:.3f}' for v in etat._eff_history_theta[-1]) + ']' if etat._eff_history_theta else ''
        _loo_str    = f'  LOO={etat._gepck_loo:.3e}' if etat._gepck_loo is not None else ''
        _fixed_str  = '  ' + '  '.join(f'{params_names[k]}={v:.1f}' for k, v in fixed.items()) if fixed else ''
        fig.suptitle(f'{modele} - N={len(xt)} pts DOE  ({n_added} ajoutes par EFF){_theta_str}{_loo_str}{_fixed_str}{_pce_line}', fontsize=10)

        _xlabel = f'u_{params_names[idx_x]}'
        _ylabel = f'u_{params_names[idx_y]}'

        def _decorate(ax):
            if Z_g is not None:
                ax.contour(UX, UY, Z_g, levels=[0], colors='cyan', linewidths=2, linestyles='--')
            if Z_true is not None:
                ax.contour(UX_hf, UY_hf, Z_true, levels=[0], colors='red', linewidths=2)
            if xt is not None:
                ax.scatter(xt[:, idx_x], xt[:, idx_y], c='white', s=40, zorder=5,
                           edgecolors='black', linewidths=0.8, label='DOE')
            if n_added > 0:
                xt_eff_arr = np.array(xt_eff)
                ax.scatter(xt_eff_arr[:, idx_x], xt_eff_arr[:, idx_y], c='red', s=80, zorder=6,
                           marker='^', label=f'EFF ({n_added} pts)')
                for i, pt in enumerate(xt_eff_arr):
                    ax.annotate(str(i + 1), (pt[idx_x], pt[idx_y]), textcoords='offset points',
                                xytext=(0, 8), ha='center', fontsize=8, color='red', zorder=7)
            ax.set_xlabel(_xlabel)
            ax.set_ylabel(_ylabel)
            ax.set_xlim(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1)
            ax.set_ylim(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1)
            ax.legend(loc='best', fontsize=9)

        # --- Ax1 : EFF ---
        cf1 = ax1.contourf(UX, UY, Z_eff, levels=20, cmap='viridis', alpha=0.85)
        plt.colorbar(cf1, ax=ax1, label='EFF')
        ax1.set_title('Critere EFF')
        _decorate(ax1)

        # --- Ax2 : sigma ---
        cf2 = ax2.contourf(UX, UY, Z_sigma, levels=20, cmap='plasma', alpha=0.85)
        plt.colorbar(cf2, ax=ax2, label='sigma (ecart-type surrogate)')
        ax2.set_title('Ecart-type surrogate (sigma)')
        _decorate(ax2)

        # --- Ax3 : g surrogate (isocouleurs RdYlGn) ---
        if Z_g is not None:
            cf3 = ax3.contourf(UX, UY, Z_g, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf3, ax=ax3, label='g surrogate')
            ax3.contour(UX, UY, Z_g, levels=[0], colors='blue', linewidths=2)
        ax3.set_title('g surrogate - etat limite')
        _decorate(ax3)

        plt.tight_layout()
        fname = f'EFF_{n_added}points_{etat.timestamp}.png'
        fig.savefig(os.path.join(etat.out_dir_eff, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [EFF visu] -> {fname}", flush=True)

    def print_globalplanche_EFF(xt, yt, all_grad, xt_eff):
        """Planche globale EFF : 3 colonnes (EFF, sigma, g) x N lignes (DOE initial + chaque etape EFF).
        Refit le surrogate a chaque etape. Utilise etat.slice_def_final pour la coupe.
        Grille HF calculee une seule fois et reutilisee."""

        if etat.slice_def_final is None:
            print("[GLOBAL PLANCHE] etat.slice_def_final est None, skip", flush=True)
            return
        idx_x, idx_y, fixed = etat.slice_def_final
        n_eff = len(xt_eff)
        n_steps = n_eff + 1   # DOE initial + chaque point EFF

        # --- Grille commune ---
        ux = np.linspace(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1, n_grid)
        uy = np.linspace(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1, n_grid)
        UX, UY = np.meshgrid(ux, uy)
        grid = np.zeros((n_grid * n_grid, n_var))
        grid[:, idx_x] = UX.ravel()
        grid[:, idx_y] = UY.ravel()
        for idx, val in fixed.items():
            grid[:, idx] = val

        # --- Grille HF (une seule fois) ---
        Z_true, UX_hf, UY_hf = None, None, None
        if hf_custom_points is not None:
            Z_true, UX_hf, UY_hf = _hf_from_custom_points(etat.slice_def_final)
        elif print_HF:
            ux_hf = np.linspace(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1, n_grid_hf)
            uy_hf = np.linspace(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1, n_grid_hf)
            UX_hf, UY_hf = np.meshgrid(ux_hf, uy_hf)
            Z_true = _get_hf_slice(etat.slice_def_final, _HF_CACHE_FILE_FINAL, 'hf_2d_grid_fixed_final')

        _xlabel = f'u_{params_names[idx_x]}'
        _ylabel = f'u_{params_names[idx_y]}'
        _fixed_str = '  '.join(f'{params_names[k]}={v:.1f}' for k, v in fixed.items()) if fixed else ''

        # --- Figure ---
        fig, axes = plt.subplots(n_steps, 3, figsize=(21, 6 * n_steps))
        if n_steps == 1:
            axes = axes.reshape(1, 3)

        for step in range(n_steps):
            n_pts = n0 + step
            xt_k = xt[:n_pts]
            yt_k = yt[:n_pts]
            grad_k = all_grad[:n_pts]
            xt_eff_k = xt_eff[:step]

            # --- Refit surrogate ---
            g_ot_k, sigma_func_k, _, _, _ = init_g_ot(None, None, xt_k, yt_k, grad_k)

            # --- Evaluer sur la grille ---
            mu_grid, sigma_grid = _batch_mu_sigma(g_ot_k, sigma_func_k, grid)
            Z_eff   = _eff_vectorized(mu_grid, sigma_grid, epsilon_factor).reshape(n_grid, n_grid)
            Z_sigma = sigma_grid.reshape(n_grid, n_grid)
            Z_g     = mu_grid.reshape(n_grid, n_grid)

            ax1, ax2, ax3 = axes[step]

            def _decorate_row(ax):
                ax.contour(UX, UY, Z_g, levels=[0], colors='cyan', linewidths=2, linestyles='--')
                if Z_true is not None:
                    ax.contour(UX_hf, UY_hf, Z_true, levels=[0], colors='red', linewidths=2)
                ax.scatter(xt_k[:, idx_x], xt_k[:, idx_y], c='white', s=40, zorder=5,
                           edgecolors='black', linewidths=0.8)
                if len(xt_eff_k) > 0:
                    eff_arr = np.array(xt_eff_k)
                    ax.scatter(eff_arr[:, idx_x], eff_arr[:, idx_y], c='red', s=80, zorder=6, marker='^')
                    for i, pt in enumerate(eff_arr):
                        ax.annotate(str(i+1), (pt[idx_x], pt[idx_y]), textcoords='offset points',
                                    xytext=(0, 8), ha='center', fontsize=8, color='red', zorder=7)
                ax.set_xlabel(_xlabel)
                ax.set_ylabel(_ylabel)
                ax.set_xlim(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1)
                ax.set_ylim(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1)

            # --- EFF ---
            cf1 = ax1.contourf(UX, UY, Z_eff, levels=20, cmap='viridis', alpha=0.85)
            plt.colorbar(cf1, ax=ax1, label='EFF')
            ax1.set_title(f'EFF  N={n_pts}  ({step} pts EFF)')
            _decorate_row(ax1)

            # --- sigma ---
            cf2 = ax2.contourf(UX, UY, Z_sigma, levels=20, cmap='plasma', alpha=0.85)
            plt.colorbar(cf2, ax=ax2, label='sigma')
            ax2.set_title(f'sigma  N={n_pts}')
            _decorate_row(ax2)

            # --- g surrogate ---
            cf3 = ax3.contourf(UX, UY, Z_g, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf3, ax=ax3, label='g surrogate')
            ax3.contour(UX, UY, Z_g, levels=[0], colors='blue', linewidths=2)
            ax3.set_title(f'g surrogate  N={n_pts}')
            _decorate_row(ax3)

            print(f"  [GLOBAL PLANCHE] step {step}/{n_steps-1} (N={n_pts}) OK", flush=True)

        fig.suptitle(f'Evolution EFF - {modele} - {_fixed_str}', fontsize=14, y=1.0)
        plt.tight_layout()
        fname = f'globalplanche_EFF_{etat.timestamp}.png'
        fig.savefig(os.path.join(etat.out_dir_eff, fname), dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  [GLOBAL PLANCHE] -> {fname}", flush=True)

    def print_visu_EFF(g_ot, sigma_func, xt, xt_eff):
        """Carte 2D des valeurs du critere EFF sur la meme grille que print_visu."""

        u1 = np.linspace(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1, n_grid)
        u2 = np.linspace(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1, n_grid)
        U1, U2 = np.meshgrid(u1, u2)
        grid = np.column_stack([U1.ravel(), U2.ravel()])

        mu_grid, sigma_grid = _batch_mu_sigma(g_ot, sigma_func, grid)
        Z_eff = _eff_vectorized(mu_grid, sigma_grid, epsilon_factor).reshape(n_grid, n_grid)

        fig, ax = plt.subplots(figsize=(7, 6))
        cf = ax.contourf(U1, U2, Z_eff, levels=20, cmap='viridis', alpha=0.85)
        plt.colorbar(cf, ax=ax, label='EFF')

        # --- Contour g=0 du surrogate (reutilise mu_grid) ---
        if g_ot is not None:
            Z_g = mu_grid.reshape(n_grid, n_grid)
            ax.contour(U1, U2, Z_g, levels=[0], colors='cyan', linewidths=2, linestyles='--', label='surrogate g=0')

        # --- Contour g=0 HF ---
        if print_HF:
            u1_hf = np.linspace(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1, n_grid_hf)
            u2_hf = np.linspace(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1, n_grid_hf)
            U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
            if etat.hf_2d_grid_fixed is not None:
                Z_true = np.array(etat.hf_2d_grid_fixed['Z'])
            else:
                grid_hf = np.column_stack([U1_hf.ravel(), U2_hf.ravel()])
                Z_true = _compute_hf_grid_with_progress(grid_hf, n_grid_hf, context="print_visu_EFF")
            ax.contour(U1_hf, U2_hf, Z_true, levels=[0], colors='red', linewidths=2)

        # --- Points DOE ---
        if xt is not None:
            ax.scatter(xt[:, 0], xt[:, 1], c='white', s=40, zorder=5,
                       edgecolors='black', linewidths=0.8, label='DOE')

        # --- Points ajoutes par EFF ---
        if xt_eff is not None and len(xt_eff) > 0:
            xt_eff_arr = np.array(xt_eff)
            ax.scatter(xt_eff_arr[:, 0], xt_eff_arr[:, 1], c='red', s=80, zorder=6,
                       marker='^', label=f'EFF ({len(xt_eff)} pts)')
            for i, pt in enumerate(xt_eff_arr):
                ax.annotate(str(i + 1), (pt[0], pt[1]), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=8, color='red', zorder=7)

        ax.set_xlabel('u1')
        ax.set_ylabel('u2')
        ax.set_xlim(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1)
        ax.set_ylim(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1)
        ax.set_title('Critere EFF')
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        plt.show(block=False)

    def print_visu_sigma(g_ot, sigma_func, xt, xt_eff):
        """Carte 2D de l'ecart-type conditionnel du surrogate."""

        u1 = np.linspace(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1, n_grid)
        u2 = np.linspace(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1, n_grid)
        U1, U2 = np.meshgrid(u1, u2)
        grid = np.column_stack([U1.ravel(), U2.ravel()])

        mu_grid, sigma_grid = _batch_mu_sigma(g_ot, sigma_func, grid)
        Z_sigma = sigma_grid.reshape(n_grid, n_grid)

        fig, ax = plt.subplots(figsize=(7, 6))
        cf = ax.contourf(U1, U2, Z_sigma, levels=20, cmap='plasma', alpha=0.85)
        plt.colorbar(cf, ax=ax, label='sigma (ecart-type surrogate)')

        # --- Contour g=0 du surrogate (reutilise mu_grid) ---
        if g_ot is not None:
            Z_g = mu_grid.reshape(n_grid, n_grid)
            ax.contour(U1, U2, Z_g, levels=[0], colors='cyan', linewidths=2, linestyles='--', label='surrogate g=0')

        # --- Contour g=0 HF ---
        if print_HF:
            u1_hf = np.linspace(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1, n_grid_hf)
            u2_hf = np.linspace(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1, n_grid_hf)
            U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
            if etat.hf_2d_grid_fixed is not None:
                Z_true = np.array(etat.hf_2d_grid_fixed['Z'])
            else:
                grid_hf = np.column_stack([U1_hf.ravel(), U2_hf.ravel()])
                Z_true = _compute_hf_grid_with_progress(grid_hf, n_grid_hf, context="print_visu_sigma")
            ax.contour(U1_hf, U2_hf, Z_true, levels=[0], colors='red', linewidths=2)

        if xt is not None:
            ax.scatter(xt[:, 0], xt[:, 1], c='white', s=40, zorder=5,
                       edgecolors='black', linewidths=0.8, label='DOE')

        if xt_eff is not None and len(xt_eff) > 0:
            xt_eff_arr = np.array(xt_eff)
            ax.scatter(xt_eff_arr[:, 0], xt_eff_arr[:, 1], c='red', s=80, zorder=6,
                       marker='^', label=f'EFF ({len(xt_eff)} pts)')
            for i, pt in enumerate(xt_eff_arr):
                ax.annotate(str(i + 1), (pt[0], pt[1]), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=8, color='red', zorder=7)

        ax.set_xlabel('u1')
        ax.set_ylabel('u2')
        ax.set_xlim(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1)
        ax.set_ylim(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1)
        ax.set_title('Ecart-type surrogate (sigma)')
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        plt.show(block=False)

    def print_results(best_result, g_ot):
        etat._point_log_phase[0] = "USTAR"
        u_star = best_result.getStandardSpaceDesignPoint()
        n_iter = best_result.getOptimizationResult().getIterationNumber()
        dist_X = dist_jointe()
        T_inv  = dist_X.getInverseIsoProbabilisticTransformation()
        x_star = T_inv(u_star)

        # --- Résultats FORM ---
        print(f"n_iter FORM  = {n_iter}", flush=True)
        for i, p in enumerate(params_names):
            print(f"{p}*          = {x_star[i]:.4f}", flush=True)
        print(f"u*           = {[round(v, 4) for v in u_star]}", flush=True)
        print(f"Imp.         = {[round(v, 4) for v in best_result.getImportanceFactors()]}", flush=True)
        print(f"beta         = {best_result.getHasoferReliabilityIndex():.4f}", flush=True)
        print(f"Pf           = {best_result.getEventProbability():.4e}", flush=True)
        if g_ot is not None:
            g_sur_ustar = g_ot(ot.Point(u_star))[0]
            print(f"g_surrogate(u*) = {g_sur_ustar:.6f}", flush=True)

        # --- Erreur FOSM ---
        if g_ot is not None:
            _, grad_HF_U_star, _ = run_HF(u_star)
            for i, p in enumerate(params_names):
                print(f"dg/du_{p} en u* (HF@u*GEK) = {grad_HF_U_star[i]:.6f}", flush=True)
            u0             = ot.Point([0.0] * n_var)
            if etat._fosm_u0_cache[0] is None:
                g0_HF, grad_HF_U0, _ = run_HF(u0)
                etat._fosm_u0_cache[0] = (g0_HF, grad_HF_U0)
            else:
                g0_HF, grad_HF_U0 = etat._fosm_u0_cache[0]
                print("  [FOSM] run_HF([0,0]) reutilise du cache (pas de SOCP redondant)", flush=True)
            u_FOSM         = grad_HF_U0 * (-g0_HF / grad_HF_U0.normSquare())
            print(f"u* FOSM (HF) = {[round(v, 4) for v in u_FOSM]}", flush=True)
            print(f"Erreur FOSM  = {(u_FOSM - u_star).norm() / u_star.norm():.4f}", flush=True)


    def print_visu(best_result, best_sps, xt, g_ot, modes, xt_eff):
        global u1_min, u1_max, u2_min, u2_max
        if etat.slice_def_final is None:
            if best_result is not None:
                _imp = np.array(best_result.getImportanceFactors())
                _top2 = list(np.argsort(_imp)[::-1][:2])
                _u_star = np.array(best_result.getStandardSpaceDesignPoint())
                etat.slice_def_final = (min(_top2), max(_top2),
                                   {i: float(_u_star[i]) for i in range(n_var) if i not in _top2})
            else:
                etat.slice_def_final = slice_def
        idx_x, idx_y, fixed = etat.slice_def_final

        ux = np.linspace(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1, n_grid)
        uy = np.linspace(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1, n_grid)
        UX, UY = np.meshgrid(ux, uy)
        grid = np.zeros((n_grid * n_grid, n_var))
        grid[:, idx_x] = UX.ravel()
        grid[:, idx_y] = UY.ravel()
        for idx, val in fixed.items():
            grid[:, idx] = val

        fig, ax = plt.subplots(figsize=(7, 6))
        _xlabel = f'u_{params_names[idx_x]}'
        _ylabel = f'u_{params_names[idx_y]}'

        # --- Fond coloré : surrogate actif (pas en mode HF) ---
        if g_ot is not None and not do_HF:
            grid_ot = ot.Sample(grid.tolist())
            Z_sur = np.array(g_ot(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(UX, UY, Z_sur, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label=f'g ({modele})')
            ax.contour(UX, UY, Z_sur, levels=[0], colors='blue', linewidths=2)

        # --- Contour HF grossier ---
        if hf_custom_points is not None:
            Z_true, UX_hf, UY_hf = _hf_from_custom_points(etat.slice_def_final)
            if Z_true is not None:
                ax.contour(UX_hf, UY_hf, Z_true, levels=[0], colors='red', linewidths=2, linestyles='--')
        elif print_HF:
            ux_hf = np.linspace(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1, n_grid_hf)
            uy_hf = np.linspace(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1, n_grid_hf)
            UX_hf, UY_hf = np.meshgrid(ux_hf, uy_hf)
            Z_true = _get_hf_slice(etat.slice_def_final, _HF_CACHE_FILE_FINAL, 'hf_2d_grid_fixed_final')
            ax.contour(UX_hf, UY_hf, Z_true, levels=[0], colors='red', linewidths=2, linestyles='--')

        # --- Points ---
        if xt is not None:
            ax.scatter(xt[:, idx_x], xt[:, idx_y], c='black', s=30, zorder=5, label='DOE')

        if xt_eff is not None and len(xt_eff) > 0:
            xt_eff_arr = np.array(xt_eff)
            ax.scatter(xt_eff_arr[:, idx_x], xt_eff_arr[:, idx_y], c='red', s=60, zorder=6,
                       marker='^', label=f'EFF ({len(xt_eff)} pts)')
            for i, pt in enumerate(xt_eff_arr):
                ax.annotate(str(i + 1), (pt[idx_x], pt[idx_y]), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=8, color='red', zorder=7)

        ax.scatter(0, 0, c='orange', s=100, zorder=6, marker='P', label='[0, 0]')

        _fixed_mode_colors = ['gold', 'magenta', 'green', 'blue', 'purple']
        n_modes = max(len(modes), 1)
        mode_colors = [_fixed_mode_colors[i] if i < len(_fixed_mode_colors)
                       else plt.cm.tab10((i % 10) / 10.0) for i in range(n_modes)]

        if best_sps:
            for i, sp in enumerate(best_sps):
                ax.scatter(sp[idx_x], sp[idx_y], color=mode_colors[i], s=100, zorder=7, marker='D',
                        label=f'sp mode {i+1}')

        if best_result is not None:
            u_star = np.array(best_result.getStandardSpaceDesignPoint())
            ax.scatter(u_star[idx_x], u_star[idx_y], color=mode_colors[0], s=200, zorder=8, marker='*',
                    label=f'u*1 [{u_star[idx_x]:.2f},{u_star[idx_y]:.2f}] beta={best_result.getHasoferReliabilityIndex():.3f}')

        if len(modes) > 0:
            for k, mode in enumerate(modes[1:], start=2):
                u_m = np.array(mode.getStandardSpaceDesignPoint())
                ax.scatter(u_m[idx_x], u_m[idx_y], color=mode_colors[k-1], s=200, zorder=8, marker='*',
                        label=f'u*{k} [{u_m[idx_x]:.2f},{u_m[idx_y]:.2f}] beta={mode.getHasoferReliabilityIndex():.3f}')

        # --- Points fixes (run HF précédent) ---
        if best_sol_modes_fixed is not None:
            colors_fixed = ['blue', 'red', 'green', 'gold']
            for col, (lbl, data) in zip(colors_fixed, best_sol_modes_fixed.items()):
                ustar_f = data['u*']
                sp_f    = data['sp']
                ax.scatter(ustar_f[idx_x], ustar_f[idx_y], c=col, s=200, zorder=9, marker='*',
                           label=f'u* {lbl}')
                ax.scatter(sp_f[idx_x], sp_f[idx_y], c=col, s=100, zorder=9, marker='x',
                           linewidths=2, label=f'sp {lbl}')
                if grad_sp_fixed is not None and lbl in grad_sp_fixed:
                    ng = np.array(grad_sp_fixed[lbl]['neg_grad'])
                    ng = ng / np.linalg.norm(ng) * 1.5
                    ax.quiver(sp_f[idx_x], sp_f[idx_y], ng[0], ng[1], color=col,
                              angles='xy', scale_units='xy', scale=1.0, width=0.005)

        # --- Trajectoires FORM hardcodees ---
        if traj_runs_fixed is not None:
            colors_traj = {'A': 'blue', 'B': 'red', 'C': 'green', 'D': 'gold'}
            all_pts = []
            for lbl, traj in traj_runs_fixed.items():
                col = colors_traj.get(lbl, 'gray')
                pts  = np.array(traj['points'])
                grds = np.array(traj['grads'])
                all_pts.append(pts)
                ax.plot(pts[:, idx_x], pts[:, idx_y], '-', color=col, alpha=0.5, linewidth=1.2)
                ax.scatter(pts[1:-1, idx_x], pts[1:-1, idx_y], c=col, s=12, zorder=7, alpha=0.5)
                ax.scatter(pts[0, idx_x],  pts[0, idx_y],  c=col, s=60,  zorder=8, marker='o')
                ax.scatter(pts[-1, idx_x], pts[-1, idx_y], c=col, s=150, zorder=8, marker='*')
                for pt, g in zip(pts, grds):
                    ng = np.array(g)
                    nrm = np.linalg.norm(ng)
                    if nrm > 0:
                        ng = -ng / nrm * 0.3
                        ax.annotate('', xy=(pt[idx_x]+ng[0], pt[idx_y]+ng[1]), xytext=(pt[idx_x], pt[idx_y]),
                                    arrowprops=dict(arrowstyle='->', color=col, lw=0.7))
            all_pts_arr = np.vstack(all_pts)
            margin = 1.0
            u1_min = float(all_pts_arr[:, idx_x].min()) - margin
            u1_max = float(all_pts_arr[:, idx_x].max()) + margin
            u2_min = float(all_pts_arr[:, idx_y].min()) - margin
            u2_max = float(all_pts_arr[:, idx_y].max()) + margin

        # --- Légende contours ---
        legend_lines = []
        if g_ot is not None:
            legend_lines.append(Line2D([0], [0], color='blue',   linestyle='-',  linewidth=2, label=f'g=0 {modele}'))
        if print_HF:
            legend_lines.append(Line2D([0], [0], color='red',    linestyle='--', linewidth=2, label='g=0 HF'))

        ax.legend(handles=ax.legend().legend_handles + legend_lines)

        ax.set_xlabel(_xlabel)
        ax.set_ylabel(_ylabel)
        ax.set_xlim(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1)
        ax.set_ylim(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1)
        _fixed_str = '  ' + '  '.join(f'{params_names[k]}={v:.1f}' for k, v in fixed.items()) if fixed else ''
        ax.set_title(f'FORM et etat limite g=0{_fixed_str}')
        plt.tight_layout()
        fname = f'visu{modele}_{etat.timestamp}.png'
        fig.savefig(os.path.join(etat.out_dir_eff, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [visu] -> {fname}", flush=True)

    def print_3D_HF():
        if hf_3d_grid_fixed is not None:
            print("Cache hf_3d_grid_fixed disponible — pas d'appels STRAINS.", flush=True)
            u1_min_c, u1_max_c, u2_min_c, u2_max_c, n_c = hf_3d_grid_fixed['params']
            u1_hf = np.linspace(u1_min_c, u1_max_c, n_c)
            u2_hf = np.linspace(u2_min_c, u2_max_c, n_c)
            U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
            Z = np.array(hf_3d_grid_fixed['Z'])
        else:
            u1_hf = np.linspace(u1_min, u1_max, n_grid_hf)
            u2_hf = np.linspace(u2_min, u2_max, n_grid_hf)
            U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
            grid_hf = np.column_stack([U1_hf.ravel(), U2_hf.ravel()])
            print(f"Evaluation HF grille {n_grid_hf}x{n_grid_hf} = {n_grid_hf**2} appels STRAINS...", flush=True)
            Z_flat = [run_HF(pt)[0] for pt in grid_hf]
            Z = np.array(Z_flat).reshape(n_grid_hf, n_grid_hf)

        # --- Impression copy-pastable ---
        print(f"\nhf_3d_grid_fixed = {{", flush=True)
        print(f"    'params': ({u1_min}, {u1_max}, {u2_min}, {u2_max}, {n_grid_hf}),", flush=True)
        print(f"    'Z': [", flush=True)
        for row in Z:
            vals = ', '.join(f'{v:.6f}' for v in row)
            print(f"        [{vals}],", flush=True)
        print(f"    ]", flush=True)
        print(f"}}", flush=True)

        # --- Plot 3D ---
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(U1_hf, U2_hf, Z, color='red', alpha=0.3, label='g_HF')
        ax.contour(U1_hf, U2_hf, Z, levels=[0], colors='red', linewidths=2,
                   zdir='z', offset=float(Z.min()))
        ax.contour(U1_hf, U2_hf, Z, levels=[0], colors='darkred', linewidths=2)

        if best_sol_modes_fixed is not None:
            for col, (lbl, data) in zip(['blue', 'red', 'green', 'gold'],
                                         best_sol_modes_fixed.items()):
                u1_f, u2_f = data['u*']
                u1_s, u2_s = data['sp']
                ax.scatter(u1_f, u2_f, 0.0, c=col, s=200, marker='*', label=f'u* {lbl}')
                ax.scatter(u1_s, u2_s, 0.0, c=col, s=100, marker='x', linewidths=2, label=f'sp {lbl}')
                if grad_sp_fixed is not None:
                    ng = grad_sp_fixed[lbl]['neg_grad']
                    ax.quiver(u1_s, u2_s, 0.0, ng[0], ng[1], 0.0,
                              color=col, length=3.0, normalize=True, arrow_length_ratio=0.3)
        ax.set_xlabel(f'u1 ({params_names[0]})')
        ax.set_ylabel(f'u2 ({params_names[1]})')
        ax.set_zlabel('g_HF')
        ax.set_title(f'Surface g_HF - {n_grid_hf}x{n_grid_hf} pts HF')
        ax.legend()
        plt.tight_layout()
        plt.show()

    # run_IS, run_IS_proj, print_results_IS : voir form_is.py

    """
    DEBUT DE CODE
    """
    # --- Mode worker DOE parallele : calcule ses points et sort ---
    if os.environ.get("_DOE_WORKER"):
        _wtask = json.load(open(os.environ["_DOE_WORKER"]))
        _wSOL = [{p: float(pt[p]) for p in params_names} for pt in _wtask["points"]]
        print(f"[DOE WORKER] modelname={modelname} | {len(_wSOL)} points a calculer", flush=True)
        _wSOL = run_one_SOL(modelname, _wSOL, params_names, sensitivity=True, with_sens_dict=None)
        _wout = {}
        for _k, _pt in enumerate(_wtask["points"]):
            _wout[str(_pt["idx"])] = dict({"g": _wSOL[_k]["g"]},
                                          **{f"dg_{q}": _wSOL[_k].get(f"dg_{q}") for q in params_names})
        with open(os.environ["_DOE_OUT"], "w") as _f:
            json.dump(_wout, _f)
        print(f"[DOE WORKER] termine -> {os.environ['_DOE_OUT']}", flush=True)
        sys.exit(0)

    # --- Mode restart : charger le dump et preparer le re-enrichissement ---
    if restart_enrich_only:
        _rs = json.load(open(_RESTART_STATE_FILE))
        xt = np.array(_rs['xt'], float)
        yt = np.array(_rs['yt'], float)
        all_grad = np.array(_rs['all_grad'], float)
        etat._restart_xt_eff = [np.array(p, float) for p in _rs['xt_eff']]
        if _rs.get('max_degree') is not None:
            max_degree = int(_rs['max_degree'])
        etat.hf_2d_grid_fixed = _rs.get('hf_2d_grid')
        etat._eff_history_EFF   = list(_rs.get('hist_EFF', []))
        etat._eff_history_BB    = list(_rs.get('hist_BB', []))
        etat._eff_history_BS    = list(_rs.get('hist_BS', []))
        etat._eff_history_theta = list(_rs.get('hist_theta', []))
        etat._eff_history_beta_IS = list(_rs.get('hist_beta_IS', []))
        etat._enrich_round     = int(_rs.get('enrich_round', 0)) + 1
        etat._round_sizes_prev = list(_rs.get('round_sizes', [int(len(xt))]))
        etat._point_log_round[0] = etat._enrich_round
        with open(_POINT_LOG_FILE, "a") as _pf:
            _pf.write(json.dumps({"phase": "_RESTART", "round": etat._enrich_round,
                                  "n_total": int(len(xt)), "n_eff": len(etat._restart_xt_eff)}) + "\n")
        print(f"[RESTART] charge {len(xt)} pts (dont {len(etat._restart_xt_eff)} EFF) "
              f"depuis {_RESTART_STATE_FILE} (round {etat._enrich_round})", flush=True)

        # max_degree fixe (LARS gere P > N)
        event, g_ot, sigma_func = None, None, None
        xt_eff = None
    else:
        # --- Reset log incremental ---
        open(_POINT_LOG_FILE, "w").close()
        print(f"[POINT LOG] reset -> {_POINT_LOG_FILE}", flush=True)

        # max_degree fixe (LARS gere P > N)
        event, g_ot, sigma_func, xt, yt, all_grad = [None] * 6
        xt_eff = None

    if print_3D:
        print_3D_HF()
        sys.exit(0)

    if print_grad_sp:
        print("=== -grad(g) aux points de depart sp A/B/C/D ===", flush=True)
        for lbl, data in best_sol_modes_fixed.items():
            sp = list(data['sp'])
            g_sp, grad_sp, _ = run_HF(sp)
            neg_grad = [-v for v in grad_sp]
            print(f"Mode {lbl} : sp={sp}", flush=True)
            print(f"  g_HF(sp)  = {g_sp:.6f}", flush=True)
            print(f"  grad(sp)  = [{grad_sp[0]:.6f}, {grad_sp[1]:.6f}]", flush=True)
            print(f"  -grad(sp) = [{neg_grad[0]:.6f}, {neg_grad[1]:.6f}]", flush=True)
        sys.exit(0)

    g_ot, sigma_func, xt, yt, all_grad = init_g_ot(g_ot, sigma_func, xt, yt, all_grad)
    if print_HF and print_fullHF and n_var <= 3:
        _compute_hf_grid_full()
    if do_EFF:
        print_planche_EFF(g_ot, sigma_func, xt, [])
        g_ot, sigma_func, xt, yt, all_grad, xt_eff = run_EFF(g_ot, sigma_func, xt, yt, all_grad)
        if not print_EFF_progres:
            print_planche_EFF(g_ot, sigma_func, xt, xt_eff)
        print_EFF_graphs()
        if print_Pf:
            print_Pf_evolution()
            print_logPf_evolution()
    event, g_ot, sigma_func, xt, yt, all_grad = init_FORM(g_ot, sigma_func, xt, yt, all_grad)

    if event is None:
        if best_sol_modes_fixed is not None:
            print_visu(None, [], None, None, [], None)
            sys.exit(0)
        print('Aucune branche active', flush=True)
        sys.exit(1)

    if do_warmstart:
        starting_points = np.array([[0.0] * n_var])
        modes, best_sps = FORM_all_modes(starting_points, tol_all_modes, event) #FORM simple avec event créé
        modes, best_sps = FORM_warm_start(modes, best_sps, g_ot, sigma_func, xt, yt, all_grad) #warm_start puis FORM multistart avec event warm
    else:
        if start_from_LHS:
            starting_points = build_starting_points()
        else:
            starting_points = np.vstack([xt, [[0.0] * n_var]]) if do_multistart else np.array([[0.0] * n_var])
        modes, best_sps = FORM_all_modes(starting_points, tol_all_modes, event)

    best_result = modes[0] if modes else None
    if best_result is None:
        print('Aucun FORM ne marche.', flush=True)
        sys.exit(1)
    if len(modes)>1:
        print('On a trouvé plus de 1 mode! Les résultats du mode 2 sont:')
        print_results(modes[1], g_ot)
        print('Les résultats du mode 1 sont : ')
    print_results(best_result, g_ot)
    result_IS = None
    if do_IS and modes:
        result_IS = run_IS(modes, event)
        print_results_IS(result_IS)
    # --- IS sur surrogate projete (enveloppe position) ---
    g_proj = projection_surrogate(g_ot)
    if g_proj is not g_ot and do_IS and modes:
        idx_pos = _find_position_var_index()
        _idx_other = [_i for _i in range(n_var) if _i != idx_pos]
        n_proj = len(_idx_other)
        dist_proj = ot.JointDistribution([ot.Normal(0, 1)] * n_proj)
        X_proj = ot.RandomVector(dist_proj)
        Y_proj = ot.CompositeRandomVector(g_proj, X_proj)
        event_proj = ot.ThresholdEvent(Y_proj, ot.Less(), 0.0)
        print("=== IS sur surrogate projete (enveloppe position) ===", flush=True)
        result_IS_proj = run_IS_proj(modes, event_proj)
        print_results_IS(result_IS_proj)
    print_visu(best_result, best_sps, xt, g_ot, modes, xt_eff)
    if do_EFF and xt_eff:
        print_globalplanche_EFF(xt, yt, all_grad, xt_eff)
    _save_restart_state(xt, yt, all_grad, xt_eff, best_result, best_sps[0] if best_sps else None, modes, result_IS)
