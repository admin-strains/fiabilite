"""
CODE FIABILITE - VERSION AVEC DEFINITION DE FONCTIONS

Ce script N'IMPORTE PLUS Digital Structure (phase 5). L'evaluation de l'etat
limite passe par `solver/fabrique.py`, qui ne charge que l'implementation
demandee par le fichier d'etude.
"""
import os
import json
import shutil
import re
import sys                  # etait utilise SANS etre importe : il ne marchait
                            # que parce que le `import *` de Digital Structure
                            # le laissait fuiter dans les globales.

#: racine du depot, deduite de ce fichier -- aucun chemin absolu.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
from api import fit_gepck, predict_gepck, predict_gradient_gepck, fit_pck, predict_pck
from lois import (
    loi_fc, loi_fy, loi_F_permanente, loi_F_exploitation,
    loi_F_intermittente, loi_uni_approx,
)
import lois as _lois
import doe as _cache_doe
import hf as _cache_hf
import eff as _eff
import eff_ot as _eff_ot
import form as _form
import graphiques as _graphiques
import schema as _schema
from fabrique import solveur as _fabriquer_solveur
from _parallel_is import adaptive_is
_IS_PARALLEL = os.environ.get("_IS_PARALLEL", "1") != "0"
_IS_K        = int(os.environ.get("_IS_K", "16"))
_IS_CHUNK    = int(os.environ.get("_IS_CHUNK", "8"))
_IS_PROBE    = int(os.environ.get("_IS_PROBE", "16"))


def _parse(text, name):
    return float(re.search(rf'(?m)^\s*{re.escape(name)}\s*=\s*([\d.]+)', text).group(1))

if __name__ == '__main__':
    # ------------------------------------------------------------------------ #
    # CONFIGURATION                                                             #
    # ------------------------------------------------------------------------ #
    # Voir `studies/moulin_blanc.toml` et `_config/schema.py`. Meme dispositif
    # que pour l'etude de flexion pure : les cinquante-quatre parametres de
    # reglage quittent le script, les sept drapeaux `do_*` deviennent derives,
    # et une cle mal orthographiee est refusee au lieu d'etre ignoree.
    #
    # Le bloc de liaison ci-dessous n'est pas une destination : il existe tant
    # que les 2 800 lignes suivantes lisent des variables globales. Une
    # affectation posee APRES ce bloc l'emporte encore.
    CFG = _schema.charger(os.environ.get("FIABILITE_ETUDE")
                          or os.path.join(_REPO, "studies", "moulin_blanc.toml"))
    print(_schema.resume(CFG), flush=True)

    # Dossier de l'etude : celui de CE fichier. Ce script portait le chemin
    # absolu du poste de l'auteur en clair a quatre endroits
    # (C:\_workingDir\_SF\test flexion\Moulinblanc), sans meme la variable
    # `path_dir` que l'etude de flexion pure utilisait.
    path_dir = os.path.dirname(os.path.abspath(__file__))
    storage = CFG.storage

    # --- liaison aux noms attendus par la suite du script --------------------
    modele              = CFG.modele
    n0                  = CFG.n0
    max_degree          = CFG.max_degree
    max_of_maxdegree    = CFG.max_of_maxdegree
    q                   = CFG.q
    seuil_pce           = CFG.seuil_pce
    reduc_PLS           = CFG.reduc_PLS
    do_analytic_grad    = CFG.do_analytic_grad
    do_EFF              = CFG.eff_actif
    epsilon_factor      = CFG.epsilon_factor
    tol_EFF             = CFG.tol_EFF
    tol_BB              = CFG.tol_BB
    tol_BS              = CFG.tol_BS
    EFF_criteria        = CFG.EFF_criteria
    n_NLopt_EFF         = CFG.n_NLopt_EFF
    n_max_EFF_points    = CFG.n_max_EFF_points
    n_batch_EFF         = CFG.n_batch_EFF
    eps_taylor          = CFG.eps_taylor
    n_max_FORM          = CFG.n_max_FORM
    tol_FORM            = CFG.tol_FORM
    tol_all_modes       = CFG.tol_all_modes
    tol_warmstart       = CFG.tol_warmstart
    do_multistart       = CFG.do_multistart
    do_warmstart        = CFG.do_warmstart
    start_from_LHS      = CFG.start_from_LHS
    n_sp                = CFG.n_sp
    do_FORM_filter      = CFG.do_FORM_filter
    do_IS               = CFG.is_actif
    n_IS                = CFG.n_IS
    cov_IS              = CFG.cov_IS
    global_size         = CFG.global_size
    geo_min_approx      = CFG.geo_min_approx
    n_workers_DOE       = CFG.n_workers_DOE
    config_is_identical = CFG.config_is_identical
    restart_enrich_only = CFG.restart_enrich_only
    save_history        = CFG.save_history
    u1_min              = CFG.u1_min
    u1_max              = CFG.u1_max
    u2_min              = CFG.u2_min
    u2_max              = CFG.u2_max
    n_grid              = CFG.n_grid
    n_grid_hf           = CFG.n_grid_hf
    print_HF            = CFG.print_HF
    print_fullHF        = CFG.print_fullHF
    print_DOE           = CFG.print_DOE
    print_3D            = CFG.print_3D
    print_Pf            = CFG.print_Pf
    print_grad_sp       = CFG.print_grad_sp
    print_EFF_progres   = CFG.print_EFF_progres
    print_gepck_calls   = CFG.print_gepck_calls
    do_custom_hf        = CFG.do_custom_hf
    hf_2d_grid_fixed    = CFG.hf_2d_grid_fixed
    hf_3d_grid_fixed    = CFG.hf_3d_grid_fixed

    # Les sept drapeaux de modele : derives, donc mutuellement exclusifs par
    # construction.
    do_KRG       = CFG.do_KRG
    do_GEK       = CFG.do_GEK
    do_HF        = CFG.do_HF
    do_PCKRG     = CFG.do_PCKRG
    do_old_GEPCK = CFG.do_old_GEPCK
    do_GEPCK     = CFG.do_GEPCK
    do_PCK       = CFG.do_PCK

    # Un worker de DOE parallele travaille sur une copie isolee du modele : son
    # nom lui est impose par le processus pere, pas par le fichier d'etude.
    modelname = os.environ.get("_DOE_WORKER_MODELNAME") or CFG.modelname
    _path_ds = os.path.join(storage, modelname + ".ds")
    with open(os.path.join(_path_ds, 'dsCad.txt'), 'r') as f:
        _cad_txt = f.read()

    print("=" * 70)
    print("CALCUL DE FIABILITE -- PONT DU MOULIN BLANC -- LM1 TRAFIC")
    print("=" * 70)

    # --- Groupes d'aciers, lus dans le dsCad ---------------------------------
    # params_names et n_var sont derives de PARAM_CONFIG_CAD/LOAD (definis apres les loi_*)
    rebar_names = re.findall(r"REBAR\('([^']+)'", _cad_txt)
    n_rebars = len(rebar_names)
    group1_names = re.findall(r"REBAR\('([^']+)',[^\n]*GRADE=fyd1,", _cad_txt)
    group2_names = re.findall(r"REBAR\('([^']+)',[^\n]*GRADE=fyd2,", _cad_txt)
    print(f"[2-fy] groupe 1 (fyd1) : {len(group1_names)} aciers | groupe 2 (fyd2) : {len(group2_names)} aciers", flush=True)

    # --- Grille HF sur mesure ------------------------------------------------
    # do_custom_hf : utiliser la grille du fichier plutot qu'un linspace.
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
        # Gradients HF aux sp (run 1305_0937, 4 appels solveur)
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

    # --- Label PCE GEPCK et LOO (mis a jour par init_g_ot, lus par print_planche_EFF) ---
    _gepck_pce_label = ""
    _gepck_loo       = None

    # --- Historiques EFF (mis a jour par run_EFF et init_g_ot, lus par print_EFF_graphs) ---
    _eff_history_EFF   = []   # EFF(u_opt) avant ajout de chaque point (incl. initial)
    _eff_history_BB    = []   # ratio BB par iteration (None si FORM echoue)
    _eff_history_BS    = []   # ratio BS par iteration (None si calcul impossible)
    _eff_history_theta = []   # theta Kriging [theta_0,...,theta_{M-1}] apres chaque fit
    _eff_history_Pf    = []   # Pf_IS (mid/sup/inf) par iter, inconditionnel
    _fosm_u0_cache     = [None] # cache run_HF([0,0]) FOSM : calcule 1x, reutilise pour tous les modes
    _point_log_phase   = ["?"]  # phase courante pour le log incremental (HF/EFF/USTAR ; DOE logue a part)
    _point_log_round   = [0]    # round de re-enrichissement (0 = run initial)
    _eff_history_beta_IS = []   # snapshot de list_beta_IS (locale a run_EFF) pour le dump restart
    _enrich_round     = 0       # 0 = run initial, k = k-ieme reprise
    _round_sizes_prev = []      # taille de chaque round precedent (charge du dump)
    _restart_xt_eff   = []      # points EFF charges du dump (seeder xt_eff en reprise)

    # --- Sortie PNG EFF ---
    timestamp   = datetime.now().strftime('%d%m_%H%M')
    out_dir_eff = os.path.join(path_dir, 'output', 'png EFF', f'png_EFF_{timestamp}')
    os.makedirs(out_dir_eff, exist_ok=True)
    _schema.ecrire_trace(CFG, out_dir_eff)   # configuration effective, a cote des figures

    # --------------------------------------------------------------------------- #
    # DEFINTION DE FONCTIONS                                                      #
    # --------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------- #
    # APPEL AU SOLVEUR ET PLAN D'EXPERIENCES                                             #


    # --- DISTRIBUTIONS ---
    

    





    # --- CONFIG DES VARIABLES ALEATOIRES (dicts) : tout en derive (lois, patch, sensibilites) ---
    FY_MEAN = 235.0
    PARAM_CONFIG_CAD = {
        # transfert4 T4-1 (2026-07-06) : 'mean'/'cov' -> 'args' (tuple passe a la loi ;
        # supporte des lois a signatures differentes, ex. loi_uni_approx(a, b, alpha)).
        'fy1': {'sens': {"param": "YIELD_STRENGTH", "rebars": group1_names, "region_key": "fy1"},
                'loi': loi_fy, 'args': (FY_MEAN, None)},
        'fy2': {'sens': {"param": "YIELD_STRENGTH", "rebars": group2_names, "region_key": "fy2"},
                'loi': loi_fy, 'args': (FY_MEAN, None)},
    }

    PARAM_CONFIG_LOAD = {}
    # # --- PARAM_CONFIG : catalogue des variables aleatoires ---
    # PARAM_CONFIG_CAD = {}
    # PARAM_CONFIG_LOAD = {
    #     's_convoi': {'sens': {"param": "LIVE_LOAD", "load_case": "LC_convoi",
    #                           "axis": "position", "region_key": "s_convoi"},
    #                  'loi': loi_uni_approx, 'args': (0.0, 1.0, 0.15)},
    #     'q':        {'sens': {"param": "LIVE_LOAD", "load_case": "LC_convoi", "region_key": "q"},
    #                  'loi': loi_F_permanente, 'args': (0.2, 0.40)},
    # }
    PARAM_CONFIG = {**PARAM_CONFIG_LOAD, **PARAM_CONFIG_CAD}
    params_names = list(PARAM_CONFIG_LOAD.keys()) + list(PARAM_CONFIG_CAD.keys())
    n_var = len(params_names)
    _rk = [PARAM_CONFIG[p]['sens'].get('region_key') for p in params_names]
    assert all(_rk), f"region_key manquant dans PARAM_CONFIG : {[p for p, r in zip(params_names, _rk) if not r]}"
    assert len(set(_rk)) == len(_rk), f"region_key dupliques : {_rk}"
    slice_def = (0, 1, {i: 0.0 for i in range(n_var) if i > 1})
    slice_def_final = None

    eff_bounds_min = [-7.5, -7.5]     # bornes inf de la recherche EFF [fy1, fy2]
    eff_bounds_max = [+7.5, +7.5]     # bornes sup de la recherche EFF [fy1, fy2]
    
    def _is_position_var(sens):
        """Detecte si une region de sensibilite est une variable de position (axis='position')."""
        return sens.get('axis') == 'position'

    def _find_position_var_index():
        """Retourne l'index de la variable de position dans params_names, ou None."""
        for i, p in enumerate(params_names):
            if _is_position_var(PARAM_CONFIG[p]['sens']):
                return i
        return None

    def dist_jointe():
        """Loi jointe des variables de base.

        PHASE 3 : le corps est parti dans `_model/lois.py`. Ce delegue de deux
        lignes evite de toucher aux cinq sites d'appel -- il disparaitra quand
        le script lui-meme sera restructure.
        """
        return _lois.dist_jointe(PARAM_CONFIG, params_names)

    # --- APPELS AU SOLVEUR -----------------------------------------------
    # Toute la mecanique Digital Structure -- reecriture du dsCad, maillage,
    # SOCP, lecture du dsmetares, archivage des sorties -- vit maintenant dans
    # `solver/digital_structure.py`. Elle tenait ici en QUATRE exemplaires
    # (`run_one_SOL` et `run_HF`, dans les deux scripts AC) qui avaient
    # diverge : `run_HF` codait en dur `global_physical_size = 0.05` et
    # `geometric_approximation_min = "4"` la ou `run_one_SOL` lisait la
    # configuration -- alors que les deux alimentent le meme metamodele.
    # `tests/golden/options_ds.json` en garde la trace.
    _socp_call_counter = [0]
    _solveurs = {}

    def _solveur(nom=None):
        """Le solveur de cette etude, un par modele.

        Un worker de DOE parallele travaille sur SA copie du `.ds` : le nom du
        modele lui est impose par le processus pere. Le solveur est mis en
        cache pour que son compteur d'appels reste continu.
        """
        nom = nom or modelname
        if nom not in _solveurs:
            _solveurs[nom] = _fabriquer_solveur(
                CFG.solveur,
                chemin_ds=os.path.join(storage, nom + ".ds"),
                dossier_etude=path_dir,
                params_names=params_names,
                regions=[PARAM_CONFIG[p]['sens'] for p in params_names],
                global_size=global_size,
                geo_min_approx=geo_min_approx,
                max_size=CFG.max_size,
                solveur_lineaire=CFG.solveur_lineaire,
                archiver=save_history,
            )
        return _solveurs[nom]

    def _etiquette_socp(prefixe, p_vals, u=None):
        """Nom du sous-dossier de SOCP_history, dans la forme d'origine."""
        _socp_call_counter[0] += 1
        coords = ""
        if u is not None:
            coords = f"_u1{float(u[0]):+.3f}_u2{float(u[1]):+.3f}"
        coords += "_" + "_".join(f"{params_names[i]}{p_vals[i]:.1f}"
                                 for i in range(len(p_vals)))
        return f"{prefixe}_{_socp_call_counter[0]:03d}{coords}"

    def _grad_vers_U(grad_X, u_point, T_inv):
        """Passage du gradient de l'espace physique X a l'espace standard U.

        C'est ici, et pas dans le solveur : la transformation isoprobabiliste
        appartient a la loi jointe, pas au maillage.
        """
        J_Tinv_T = T_inv.gradient(u_point).transpose()
        return J_Tinv_T * ot.Point(list(grad_X))

    def run_one_SOL(modelname, SOL, params_names, sensitivity=False, with_sens_dict=None):
        """Lance un calcul complet pour une valeur de FT donnee.
        Retourne la liste des solutions pour chaque jeu de variables dans SOL (liste de dictionnaire).
        Les gradients sont convertis en espace U (standard normal) via T = isoprobabilistic transform.
        SOL[i]['dg_<var>'] = gradient en U. SOL[i]['_u'] = coordonnees U du point."""
        solveur = _solveur(modelname)
        dist_X = dist_jointe()
        T = dist_X.getIsoProbabilisticTransformation()
        T_inv = dist_X.getInverseIsoProbabilisticTransformation()
        for i in range(len(SOL)):
            p_vals = [float(SOL[i][p]) for p in params_names]
            etiquette = _etiquette_socp("SOL", p_vals) if save_history else None
            ev = solveur.evaluer({p: SOL[i][p] for p in params_names},
                                 sensibilite=sensitivity, etiquette=etiquette)
            if not ev.sain:
                # Les criteres de convergence rendus par Digital Structure ne
                # sont pas encore fiables (Agnes, 26/08/2026) : on SIGNALE,
                # on ne jette pas. Basculer `exclure_points_non_converges` a
                # true dans le fichier d'etude le jour ou ils le seront.
                print("  [SOLVEUR] point %s NON CONVERGE (%s), alpha=%.6f -- %s"
                      % (p_vals, ev.diagnostic.get("solver_status"), ev.alpha,
                         "EXCLU du plan d'experiences" if CFG.exclure_points_non_converges
                         else "conserve : critere DS juge non fiable"), flush=True)
                if CFG.exclure_points_non_converges:
                    ev.exige_sain("point %s du plan d'experiences" % (p_vals,))
            SOL[i]['g'] = ev.g

            # --- Conversion X -> U ---
            x_point = ot.Point(p_vals)
            u_point = T(x_point)
            SOL[i]['_u'] = [float(u_point[j]) for j in range(n_var)]
            if sensitivity and ev.gradient_complet:
                grad_U = _grad_vers_U(ev.grad_x, u_point, T_inv)
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
        """Evalue l'etat limite en UN point de l'espace standard U.

        Sert a l'enrichissement EFF, a la grille haute fidelite et au FOSM.
        Les points qu'elle produit rejoignent le plan d'experiences : elle doit
        donc mailler EXACTEMENT comme `run_one_SOL`, ce qui n'etait pas le cas
        avant la phase 5.
        """
        solveur = _solveur()
        n_var_local = len(u)
        dist_X = dist_jointe()
        T_inv = dist_X.getInverseIsoProbabilisticTransformation()
        u_point = ot.Point(list(u))
        x_point = T_inv(u_point)
        p_vals = [float(x_point[j]) for j in range(n_var_local)]

        etiquette = _etiquette_socp("HF", p_vals, u=u) if save_history else None
        ev = solveur.evaluer({params_names[i]: x_point[i] for i in range(n_var_local)},
                             sensibilite=True, etiquette=etiquette)
        if not ev.sain:
            # Meme regle qu'au plan d'experiences : ce point rejoint le
            # metamodele, il doit donc etre traite pareil.
            print("  [SOLVEUR] run_HF %s NON CONVERGE (%s), alpha=%.6f -- %s"
                  % (list(u), ev.diagnostic.get("solver_status"), ev.alpha,
                     "EXCLU" if CFG.exclure_points_non_converges
                     else "conserve : critere DS juge non fiable"), flush=True)
            if CFG.exclure_points_non_converges:
                ev.exige_sain("point d'enrichissement %s" % (list(u),))
        g_HF = ev.g

        grad_HF_X = list(ev.grad_x)
        grad_HF_U = [None] * n_var_local
        if ev.gradient_complet:
            grad_HF_U = _grad_vers_U(grad_HF_X, u, T_inv)
        if any(v is None for v in grad_HF_U):
            raise ValueError(f"run_HF : sensibilité demandée mais grad_HF_U contient None — vérifier que le solveur a bien calculé les sensibilités. grad_HF_X={grad_HF_X}")
        _append_point_log(_point_log_phase[0], u, x_point, g_HF)
        return g_HF, grad_HF_U, grad_HF_X




    _run_HF_count = [0]  # compteur pour le print memoire (temporaire)

    # --- DOE PARALLELE ---
    def run_DOE_parallel(base_modelname, SOL, params_names, n_workers):
        """Parallelise les SOCP du DOE via subprocesses independants.
        Chaque worker = launcher.py relance en mode _DOE_WORKER sur une copie .ds isolee.

        Les workers passaient jusqu'ici par `launcher3.py`, une copie du lanceur
        portant en dur les chemins du poste de l'auteur : ce chemin de code ne
        pouvait pas s'executer ailleurs."""
        import subprocess as _sp
        base_ds = os.path.join(storage, base_modelname + ".ds")
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
            wds = os.path.join(storage, wname + ".ds")
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
            p = _sp.Popen([sys.executable, os.path.join(_REPO, "launcher.py"), "--garder-cwd", __file__],
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
        base_ds = os.path.join(storage, modelname + ".ds")
        batches = [[] for _ in range(n_workers)]
        for i in range(npts):
            batches[i % n_workers].append(i)
        print(f"  [HF GRID PARALLELE] {npts} pts -> {n_workers} workers (MKL={threads_per})", flush=True)
        procs = []
        for w, idxs in enumerate(batches):
            if not idxs:
                continue
            wname = modelname + ".ds\\_hf_workers\\hfw%d" % w
            wds = os.path.join(storage, wname + ".ds")
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
            p = _sp.Popen([sys.executable, os.path.join(_REPO, "launcher.py"), "--garder-cwd", __file__],
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

    # --- Caches : la logique est dans _cache/doe.py et _cache/hf.py.
    # Ces delegues gardent les sites d'appel intacts et disparaitront
    # avec la refonte de la configuration (phase 4).
    # La signature est ce qui rend un cache reutilisable OU NON : elle porte
    # le modele, le solveur, le solveur lineaire et les tailles de maille.
    # Sans elle, basculer CuDss -> MUMPS aurait relu des points de l'autre
    # backend en silence (constat du 26/08/2026).
    _SIG_SOLVEUR = CFG.signature_solveur()

    def _load_doe_cache():
        return _cache_doe.load_doe_cache(_DOE_CACHE_FILE, n0, config_is_identical,
                                         signature=_SIG_SOLVEUR)

    def _save_doe_cache(xt, yt, all_grad):
        return _cache_doe.save_doe_cache(_DOE_CACHE_FILE, n0, xt, yt, all_grad,
                                         signature=_SIG_SOLVEUR)

    def _save_doe_cache_incremental(SOL, n_done):
        return _cache_doe.save_doe_cache_incremental(
            _DOE_CACHE_FILE, n0, params_names, SOL, n_done,
            signature=_SIG_SOLVEUR)

    # --- SIGNATURE INFORMATIVE (utilisee par le dump restart, pas par le DOE cache) ---
    def _doe_cache_sig():
        return _cache_doe.doe_cache_sig(n0, params_names, n_var, modelname)

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
            st["timestamp"] = timestamp
            try:    st["max_degree"] = int(max_degree)
            except Exception: st["max_degree"] = None
            st["xt"]       = np.asarray(xt).tolist()       if xt       is not None else None
            st["yt"]       = np.asarray(yt).tolist()       if yt       is not None else None
            st["all_grad"] = np.asarray(all_grad).tolist() if all_grad is not None else None
            st["xt_eff"]   = [np.asarray(p).tolist() for p in xt_eff] if xt_eff else []
            st["n_doe"]    = n0
            st["n_total"]  = int(len(xt)) if xt is not None else 0
            _prev_tot = sum(_round_sizes_prev) if _round_sizes_prev else 0
            if _enrich_round > 0:
                st["round_sizes"] = list(_round_sizes_prev) + [int(len(xt)) - _prev_tot]
            else:
                st["round_sizes"] = [int(len(xt))] if xt is not None else []
            st["enrich_round"]     = int(_enrich_round)
            st["round_boundaries"] = list(np.cumsum([0] + st["round_sizes"]).astype(int).tolist())
            st["hist_EFF"]     = [float(v) for v in _eff_history_EFF]
            st["hist_BB"]      = [None if v is None else float(v) for v in _eff_history_BB]
            st["hist_BS"]      = [None if v is None else float(v) for v in _eff_history_BS]
            st["hist_theta"]   = [[float(x) for x in t] for t in _eff_history_theta]
            st["hist_beta_IS"] = [None if v is None else float(v) for v in _eff_history_beta_IS]
            try:    st["hf_2d_grid"] = hf_2d_grid_fixed
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
            rec = {"phase": phase, "round": _point_log_round[0],
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
            if do_PCK and eps_taylor > 0:
                _n_real = len(xt)
                for _i_pt in range(_n_real):
                    for _i_dim in range(n_var):
                        _u_virt = xt[_i_pt] + eps_taylor * np.eye(n_var)[_i_dim]
                        _y_virt = yt[_i_pt, 0] + eps_taylor * all_grad[_i_pt, _i_dim]
                        xt = np.vstack([xt, [_u_virt]])
                        yt = np.vstack([yt, [[_y_virt]]])
                        all_grad = np.vstack([all_grad, [all_grad[_i_pt]]])
                print(f"  [Taylor DOE] {_n_real} HF + {_n_real * n_var} virtuels = {len(xt)} pts", flush=True)
            return xt, yt, all_grad
        return xt

    def build_starting_points():
        dist_U = ot.JointDistribution([ot.Uniform(eff_bounds_min[i], eff_bounds_max[i]) for i in range(n_var)])
        lhs = ot.LHSExperiment(dist_U, n_sp)
        sa = ot.SimulatedAnnealingLHS(lhs, ot.SpaceFillingMinDist())
        return np.array(sa.generate())  # shape (n_sp, n_var)

    def build_Y_aug(yt, all_grad):
        """
        Construit le vecteur gradient-enhanced y_dot (eq. 6 Zuhal et al.).
        y_dot = [y^1,...,y^n, dg/du1^1,...,dg/du1^n, ..., dg/dum^1,...,dg/dum^n]^T
        Shape : (n0*(1+n_var),)
        """
        y_flat      = yt.flatten()                                         # (n0,)
        grad_blocks = [all_grad[:, j] for j in range(all_grad.shape[1])]  # n_var blocs de (n0,)
        return np.concatenate([y_flat] + grad_blocks)                      # (n0*(1+n_var),)

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

    # --------------------------------------------------------------------------- #
    # FONCTIONS LIEES AU MODELE PCE                                               #
    # n0_min et update_degree supprimes : LARS gere P > N, max_degree fixe des le depart

    def build_metamodel_PCE(xt, y_hf):
        # 1. INITIALISATION : DOE ET DISTRIBUTION
        inputSample = ot.Sample(xt)
        outputSample = ot.Sample(y_hf)
        n0 = xt.shape[0]
        dist_X = dist_jointe()
        dist_U = dist_X.getStandardDistribution()

        # 2. BASE DE CANDIDATS : TYPE, ENUMERATION, DEGRE
        n_var = inputSample.getDimension()
        enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(n_var, q)
        basis = ot.OrthogonalProductPolynomialFactory([ot.HermiteFactory()] * n_var, enumerateFunction)
        basis_size = enumerateFunction.getBasisSizeFromTotalDegree(max_degree)
        basisStrategy = ot.FixedStrategy(basis, basis_size)

        # 3. PROPOSITION / PROJECTION / SELECTION
        selectionStrategy = ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut())
        projectionStrategy = ot.LeastSquaresStrategy(selectionStrategy) 

        # 4. RESULTAT
        algo = ot.FunctionalChaosAlgorithm(inputSample, outputSample, dist_U, basisStrategy, projectionStrategy)
        algo.run()
        result = algo.getResult()
        n_active = result.getCoefficients().getSize()
        print(f"PCE construite : basis_size={basis_size}, coefficients actifs LARS={n_active}", flush=True)
        indices = result.getIndices()
        coeffs  = result.getCoefficients()
        terms = []
        for k in range(indices.getSize()):
            mi = enumerateFunction(indices[k])
            a, b = int(mi[0]), int(mi[1])
            if   a == 0 and b == 0: label = "1"
            elif a == 0:            label = f"H{b}(u2)"
            elif b == 0:            label = f"H{a}(u1)"
            else:                   label = f"H{a}(u1)*H{b}(u2)"
            terms.append(f"{coeffs[k, 0]:+.4f}*{label}")
        print(f"  PCE termes : {' '.join(terms)}", flush=True)
        metamodel = result.getMetaModel()
        return metamodel

    def calculate_PCE(xt, y_hf, all_grad_hf, metamodel_PCE):
        U_doe = ot.Sample(xt)                               
        y_PCE = np.array(metamodel_PCE(U_doe))
        n_var = U_doe.getDimension()
        n0 = U_doe.getSize()
        dist_X = dist_jointe()
        T = dist_X.getIsoProbabilisticTransformation()
        T_inv = dist_X.getInverseIsoProbabilisticTransformation()
        all_grad_PCE = np.zeros((n0, n_var))
        # all_sensib_PCE = np.zeros((n0, n_var))
        for i in range(n0):
            grad_pce_u = metamodel_PCE.gradient(U_doe[i])       
            for j in range(n_var):
                all_grad_PCE[i, j] = grad_pce_u[j, 0]
        return y_PCE, all_grad_PCE

    class PCKRGFunction(ot.OpenTURNSPythonFunction):
        def __init__(self, g_pce, g_krg):
            super().__init__(n_var, 1)
            self.g_pce = g_pce
            self.g_krg = g_krg
        
        def _exec(self, u):
            return [self.g_pce(u)[0] + self.g_krg(u)[0]]

        def _exec_sample(self, U):
            U_ot = ot.Sample(U)
            Z_pce = np.array(self.g_pce(U_ot))[:, 0]
            Z_krg = np.array(self.g_krg(U_ot))[:, 0]
            return (Z_pce + Z_krg).reshape(-1, 1).tolist()

        def _gradient(self, u):
            u_ot     = ot.Point(list(u))
            grad_pce = self.g_pce.gradient(u_ot)   
            grad_krg = self.g_krg.gradient(u_ot)   
            return [[grad_pce[i, 0] + grad_krg[i, 0]] for i in range(n_var)]

    class oldGEPCKFunction(ot.OpenTURNSPythonFunction):
        def __init__(self, g_pce, sm_gepck):
            super().__init__(n_var, 1)
            self.g_pce  = g_pce
            self.sm     = sm_gepck

        def _exec(self, u):
            y_pce = self.g_pce(ot.Point(list(u)))[0]
            y_gek = self.sm.predict_values(np.array(u).reshape(1, -1)).item()
            return [y_pce + y_gek]

        def _exec_sample(self, U):
            U_ot = ot.Sample(U)
            Z_pce = np.array(self.g_pce(U_ot))[:, 0]
            Z_gek = self.sm.predict_values(np.array(U))[:, 0]
            return (Z_pce + Z_gek).reshape(-1, 1).tolist()

        def _exec_sigma(self, u):
            sm = self.sm
            n  = sm.nt; d = sm.X_norma.shape[1]
            W  = sm.coeff_pls; th = sm.optimal_theta
            s2 = float(sm.optimal_par['sigma2'])
            th_eff = (W**2) @ th
            x_n = (np.array(u).reshape(-1) - sm.X_offset) / sm.X_scale
            Xn  = sm.X_norma
            df  = x_n[None, :] - Xn
            kf  = np.exp(-np.dot(df**2, th_eff))
            kd  = (2.0 * kf[:, None] * df * th_eff[None, :]).reshape(-1)
            dff = Xn[:, None, :] - Xn[None, :, :]
            K_ff = np.exp(-np.einsum('ijk,k->ij', dff**2, th_eff))
            K_fd = (2.0 * K_ff[:, :, None] * dff * th_eff[None, None, :]).reshape(n, n*d)
            B_mat = dff * th_eff[None, None, :]
            term1 = 2.0 * np.diag(th_eff)
            term2 = 4.0 * np.einsum('ija,ijb->ijab', B_mat, B_mat)
            K_dd  = (K_ff[:, :, None, None] * (term1 - term2)).transpose(0,2,1,3).reshape(n*d, n*d)
            K_tot = np.block([[K_ff, K_fd], [K_fd.T, K_dd]])
            K_tot += 1e-10 * np.eye(K_tot.shape[0])
            k = np.concatenate([kf, kd])
            try:
                B = max(0.0, 1.0 - k @ np.linalg.solve(K_tot, k))
                return float(np.sqrt(s2 * B))
            except np.linalg.LinAlgError:
                return float(np.sqrt(sm.predict_variances(np.array(u).reshape(1, -1)).item()))

        def _gradient(self, u):
            u_np     = np.array(u).reshape(1, -1)
            grad_pce = self.g_pce.gradient(ot.Point(list(u)))   # OT Matrix (n_var, 1)
            return [[grad_pce[i, 0] + self.sm.predict_derivatives(u_np, i).item()]
                    for i in range(n_var)]

    # --------------------------------------------------------------------------- #
    # WRAPPER GEPCK 5 BRANCHES                                                   #
    class GEPCKFunction(ot.OpenTURNSPythonFunction):
        def __init__(self, fm):
            super().__init__(n_var, 1)
            self.fm = fm
            self.n_eval_calls = 0
            self.n_grad_calls = 0

        def _exec(self, u):
            u_np  = np.array(u).reshape(1, -1)
            g_val = float(predict_gepck(self.fm, u_np)[0, 0])
            self.n_eval_calls += 1
            if print_gepck_calls:
                print(f"[GEPCK eval #{self.n_eval_calls:3d}] u=[{float(u[0]):+.4f}, {float(u[1]):+.4f}]"
                      f"  g={g_val:+.6f}", flush=True)
            return [g_val]

        def _exec_sample(self, U):
            U_np = np.array(U)
            return predict_gepck(self.fm, U_np)[:, 0:1].tolist()

        def _exec_sigma(self, u):
            u_np = np.array(u).reshape(1, -1)
            _, YSig2 = predict_gepck(self.fm, u_np, return_var=True)
            return float(np.sqrt(max(0.0, float(YSig2[0, 0]))))

        def _gradient(self, u):
            u_np  = np.array(u).reshape(1, -1)
            G     = predict_gradient_gepck(self.fm, u_np)   # (1, Mred)
            grad  = [float(G[0, i]) for i in range(self.fm['Mred'])]
            g_val = float(predict_gepck(self.fm, u_np)[0, 0])
            self.n_grad_calls += 1
            print(f"[GEPCK grad #{self.n_grad_calls:3d}] u=[{float(u[0]):+.4f}, {float(u[1]):+.4f}]"
                  f"  g={g_val:+.6f}  grad=[{grad[0]:+.6f}, {grad[1]:+.6f}]", flush=True)
            return [[v] for v in grad]

    # --------------------------------------------------------------------------- #
    # WRAPPER PCK (sans gradient analytique — FORM utilise differences finies)    #
    class PCKFunction(ot.OpenTURNSPythonFunction):
        def __init__(self, fm):
            super().__init__(n_var, 1)
            self.fm = fm
            self.n_eval_calls = 0

        def _exec(self, u):
            u_np  = np.array(u).reshape(1, -1)
            g_val = float(predict_pck(self.fm, u_np)[0, 0])
            self.n_eval_calls += 1
            if print_gepck_calls:
                print(f"[PCK eval #{self.n_eval_calls:3d}] u=[{float(u[0]):+.4f}, {float(u[1]):+.4f}]"
                      f"  g={g_val:+.6f}", flush=True)
            return [g_val]

        def _exec_sample(self, U):
            U_np = np.array(U)
            return predict_pck(self.fm, U_np)[:, 0:1].tolist()

        def _exec_sigma(self, u):
            u_np = np.array(u).reshape(1, -1)
            _, YSig2 = predict_pck(self.fm, u_np, return_var=True)
            return float(np.sqrt(max(0.0, float(YSig2[0, 0]))))

        # pas de _gradient : OT utilise differences finies pour FORM

    # --------------------------------------------------------------------------- #
    # WRAPPER BORNES DE CONFIANCE DU SURROGATE                                   #

    # --- FORM multimodal et tirage d'importance : la logique est dans
    # _reliability/form.py, en un seul exemplaire pour les deux etudes.
    def BoundSurrogateFunction(g_ot, sigma_func, sign):
        return _form.bound_surrogate_function(
            g_ot, sigma_func, sign, n_var,
            predict_pck if do_PCK else predict_gepck)

    # Usage :
    #   g_ot_sup = ot.Function(BoundSurrogateFunction(g_ot, sigma_func, +1))
    #   g_ot_inf = ot.Function(BoundSurrogateFunction(g_ot, sigma_func, -1))

    # --------------------------------------------------------------------------- #
    # PROJECTION DU SURROGATE SUR LES VARIABLES NON-POSITION                     #

    def projection_surrogate(g_ot):
        """Si variable de position dans PARAM_CONFIG, retourne un g_ot projete
        g_proj(u_other) = min_p g_ot(u_full) sur la variable de position.
        Sinon retourne g_ot inchange."""
        from scipy.optimize import minimize_scalar

        idx_pos = _find_position_var_index()

        if idx_pos is None:
            return g_ot

        idx_other = [i for i in range(n_var) if i != idx_pos]
        n_proj = len(idx_other)

        class ProjectedSurrogateFunction(ot.OpenTURNSPythonFunction):
            def __init__(self):
                super().__init__(n_proj, 1)

            def _exec(self, u_reduced):
                def _obj(u_pos):
                    u_full = [0.0] * n_var
                    for k, idx in enumerate(idx_other):
                        u_full[idx] = float(u_reduced[k])
                    u_full[idx_pos] = u_pos
                    return float(g_ot(ot.Point(u_full))[0])
                # grille grossiere puis affinage (robuste pour W multi-creux)
                u_grid = np.linspace(-5.0, 5.0, 30)
                g_grid = [_obj(u) for u in u_grid]
                u_best = u_grid[np.argmin(g_grid)]
                res = minimize_scalar(_obj,
                                      bounds=(max(-5.0, u_best - 0.5),
                                              min(5.0, u_best + 0.5)),
                                      method='bounded',
                                      options={'xatol': 1e-4, 'maxiter': 200})
                return [res.fun]

        return ot.Function(ProjectedSurrogateFunction())

    # --------------------------------------------------------------------------- #
    # FONCTIONS LIEES AU KRG                                                      #

    def build_metamodel_KRG(xt, yt):
        n_var = xt.shape[1]
        basis = ot.ConstantBasisFactory(n_var).build()
        covarianceModel = ot.SquaredExponential([1.0] * n_var)
        algo_KRG = ot.KrigingAlgorithm(xt, yt, covarianceModel, basis)
        if do_KRG:
            algo_KRG.setOptimizationBounds(ot.Interval([0.0] * n_var, [100.0] * n_var))
        algo_KRG.run()
        result = algo_KRG.getResult()
        cov_opt = result.getCovarianceModel()
        print(f"  KRG theta={list(cov_opt.getScale())}  sigma={list(cov_opt.getAmplitude())}", flush=True)
        return result.getMetaModel(), result

    # --------------------------------------------------------------------------- #
    # FONCTIONS LIEES AU GEK                                                      #

    # --- Modèle smt  ---
    def build_metamodel_GEK(xt, yt, all_grad):
        xlimits = np.column_stack([xt.min(axis=0) - 1, xt.max(axis=0) + 1])
        if do_GEK:
            sm = GEKPLS(
                n_comp=2,
                theta0=[1e-2, 1e-2],
                theta_bounds=[1.0, 5.0],
                corr="squar_exp",
                poly="constant",
                xlimits=xlimits,
                print_global=False,
            )
        else:
            sm = GEKPLS(
                n_comp=2,
                theta0=[1e-2, 1e-2],
                theta_bounds=[1.0, 5.0],
                corr="squar_exp",
                poly="constant",
                xlimits=xlimits,
                print_global=False,
            )
        sm.set_training_values(xt, yt)
        for j in range(n_var):
            sm.set_training_derivatives(xt, all_grad[:, j].reshape(-1, 1), j)
        sm.train()
        print(f"  GEK theta={list(sm.optimal_theta)}  sigma={float(np.sqrt(sm.optimal_par['sigma2'])):.6f}", flush=True)
        return sm

    # --- Wrapper OpenTURNS avec gradients analytiques ---
    class GEKPLSFunction(ot.OpenTURNSPythonFunction):
        def __init__(self, surrogate):
            super().__init__(n_var, 1)
            self.sm = surrogate

        def _exec(self, u):
            return [self.sm.predict_values(np.array(u).reshape(1, -1)).item()]

        def _exec_sample(self, U):
            return self.sm.predict_values(np.array(U)).tolist()
        
        def _exec_sigma(self, u):
            sm = self.sm
            n  = sm.nt; d = sm.X_norma.shape[1]
            W  = sm.coeff_pls; th = sm.optimal_theta
            s2 = float(sm.optimal_par['sigma2'])
            th_eff = (W**2) @ th
            x_n = (np.array(u).reshape(-1) - sm.X_offset) / sm.X_scale
            Xn  = sm.X_norma
            df  = x_n[None, :] - Xn
            kf  = np.exp(-np.dot(df**2, th_eff))
            kd  = (2.0 * kf[:, None] * df * th_eff[None, :]).reshape(-1)
            dff = Xn[:, None, :] - Xn[None, :, :]
            K_ff = np.exp(-np.einsum('ijk,k->ij', dff**2, th_eff))
            K_fd = (2.0 * K_ff[:, :, None] * dff * th_eff[None, None, :]).reshape(n, n*d)
            B_mat = dff * th_eff[None, None, :]
            term1 = 2.0 * np.diag(th_eff)
            term2 = 4.0 * np.einsum('ija,ijb->ijab', B_mat, B_mat)
            K_dd  = (K_ff[:, :, None, None] * (term1 - term2)).transpose(0,2,1,3).reshape(n*d, n*d)
            K_tot = np.block([[K_ff, K_fd], [K_fd.T, K_dd]])
            K_tot += 1e-10 * np.eye(K_tot.shape[0])
            k = np.concatenate([kf, kd])
            try:
                B = max(0.0, 1.0 - k @ np.linalg.solve(K_tot, k))
                return float(np.sqrt(s2 * B))
            except np.linalg.LinAlgError:
                return float(np.sqrt(sm.predict_variances(np.array(u).reshape(1, -1)).item()))

        def _gradient(self, u):
            u_np = np.array(u).reshape(1, -1)
            return [[self.sm.predict_derivatives(u_np, kx).item()] for kx in range(n_var)]
    
    # --------------------------------------------------------------------------- #
    # FONCTIONS POUR FORM                                                         #
    def init_g_ot(g_ot, sigma_func, xt, yt, all_grad, fixed_fm=None):
        """
        Cette fonction génère xt, yt, all_grad si xt n'est pas vide puis
        contruit un metamodele à partir de ces points. Dans le cas HF, elle
        créé uniquement une fonction OT. Retourne g_ot, sigma_func, xt, yt, all_grad.
        Si fixed_fm est fourni (refit KB) : theta et polynomes fixes du fit precedent.
        """
        global _gepck_pce_label, _gepck_loo, _eff_history_theta
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
            _gepck_pce_label = ' '.join(_terms)
            _gepck_loo       = _fm['Error'][0]['LOO']
            _eff_history_theta.append(list(_fm['Kriging'][0]['theta']))
            gepck_impl = GEPCKFunction(_fm)
            g_ot       = ot.Function(gepck_impl)
            sigma_func = gepck_impl._exec_sigma

        elif do_PCK:
            if xt is None: xt, yt, all_grad = build_DOE()
            _marginals = [{'Type': 'Gaussian', 'Parameters': [0.0, 1.0]}] * n_var
            _copula    = {'Type': 'Independent', 'Parameters': np.eye(n_var)}
            if fixed_fm is not None:
                _prev_theta = fixed_fm['Kriging'][0]['theta']
                _prev_npoly = int(fixed_fm['NumberOfPoly'][0])
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
            print(f"=== PCK fit N={len(xt)}{' [KB]' if fixed_fm else ''} ===", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _fm = fit_pck(xt, yt.ravel(), _opts, _marginals, _copula)
            print(f"  LOO={_fm['Error'][0]['LOO']:.4e}  n_poly={_fm['NumberOfPoly'][0]}  theta={_fm['Kriging'][0]['theta']}", flush=True)
            _final_idx   = _fm['idxranking'][0][:_fm['NumberOfPoly'][0]]
            _sel_indices = _fm['AllIndices'][0][np.array(_final_idx), :]
            _beta_pce    = np.array(_fm['Kriging'][0]['beta']).ravel()
            _terms = []
            for _mi, _coef in zip(_sel_indices, _beta_pce):
                _parts = [f"H{int(_mi[k])}(u{k+1})" for k in range(len(_mi)) if int(_mi[k]) > 0]
                _terms.append(f"{_coef:+.4f}*{'*'.join(_parts) if _parts else '1'}")
            print(f"  PCK PCE termes : {' '.join(_terms)}", flush=True)
            _gepck_pce_label = ' '.join(_terms)
            _gepck_loo       = _fm['Error'][0]['LOO']
            _eff_history_theta.append(list(_fm['Kriging'][0]['theta']))
            pck_impl = PCKFunction(_fm)
            g_ot       = ot.Function(pck_impl)
            sigma_func = pck_impl._exec_sigma

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

    # --- Multi-start FORM depuis les points du DOE ---
    def FORM_all_modes(starting_points, tol_all_modes, event):
        return _form.form_all_modes(starting_points, tol_all_modes, event,
                                    n_var, n_max_FORM, tol_FORM,
                                    do_FORM_filter, eff_bounds_min, eff_bounds_max)
    
    # --- Warm-start FORM depuis les points du DOE ---
    def FORM_warm_start(modes, best_sps, g_ot, sigma_func, xt, yt, all_grad):
        """
        Cette fonction reçoit des résultats FORM et le DOE utilisé, et déclenche warm_start si besoin 
        - elle renvoie la liste de modes/ best_sps mise à jour mais ne renvoie pas le nouveau DOE pour 
        l'instant (choix facilement modifiable).
        """
        if len(modes)>0:
            u_star = modes[0].getStandardSpaceDesignPoint()
            g_val = g_ot(ot.Point(u_star))[0] if g_ot is not None else None

            if g_val is not None and abs(g_val) > tol_warmstart:
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

    # --------------------------------------------------------------------------- #
    # FONCTIONS D'ENRICHISSEMENT DU PLAN D'EXPERIENCE (EFF)                       #
    # --- Critere EFF : la formule est dans _reliability/eff.py, en un seul
    # exemplaire. Elle etait ecrite deux fois ici (vectorisee et scalaire).
    def EFFFunction(g_ot, sigma_func):
        return _eff_ot.eff_function(g_ot, sigma_func, n_var, epsilon_factor)

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
        global _eff_history_EFF, _eff_history_BB, _eff_history_BS, _eff_history_Pf, _eff_history_beta_IS
        if g_ot is None or do_HF:
            return g_ot, sigma_func, xt, yt, all_grad, []
        _point_log_phase[0] = "EFF"

        xt_eff = list(_restart_xt_eff) if restart_enrich_only else []

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
            if _IS_PARALLEL and fm is not None and not do_PCK:
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
        # Reprendre les compteurs de convergence depuis l'historique (restart)
        if restart_enrich_only and _eff_history_BS:
            for _v in reversed(_eff_history_BS):
                if _v < tol_BS:
                    count_valid_BS += 1
                else:
                    break
        if restart_enrich_only and _eff_history_BB:
            for _v in reversed(_eff_history_BB):
                if _v < tol_BB:
                    count_valid_BB += 1
                else:
                    break
        if count_valid_BS > 0 or count_valid_BB > 0:
            print(f"  [RESTART] compteurs repris : count_valid_BB={count_valid_BB}  count_valid_BS={count_valid_BS}", flush=True)
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
            list_beta_IS = list(_eff_history_beta_IS) + ([_b_mid] if _b_mid is not None else [])
        else:
            list_beta_IS = [_b_mid] if _b_mid is not None else []
        if not restart_enrich_only:
            _eff_history_BB = []
            _eff_history_BS = []
            _eff_history_Pf = []
        list_ratio_BB = _eff_history_BB   # alias — même objet
        list_ratio_BS = _eff_history_BS
        list_Pf = _eff_history_Pf
        _eff_history_EFF.append(f(u_opt)[0])   # EFF initial (avant ajout du 1er point)

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
            _eff_history_EFF.append(f(u_opt)[0])   # EFF apres rebuild a cette iteration

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
                    # --- Points virtuels Taylor ordre 1 (PCK uniquement) ---
                    if do_PCK and eps_taylor > 0 and n_batch_EFF <= 1:
                        for _i_dim in range(n_var):
                            _u_virt = np.array(_u_pt) + eps_taylor * np.eye(n_var)[_i_dim]
                            _y_virt = g_val + eps_taylor * float(grad_U[_i_dim])
                            xt = np.vstack([xt, [_u_virt]])
                            yt = np.vstack([yt, [[_y_virt]]])
                            all_grad = np.vstack([all_grad, grad_val])
                            print(f"[EFF Taylor] u={list(np.round(_u_virt, 10))}  y_taylor={_y_virt:.10f}  (eps={eps_taylor}, dim={_i_dim})", flush=True)
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
            _eff_history_beta_IS = list(list_beta_IS)
            _save_restart_state(xt, yt, all_grad, xt_eff, None, None, [], None)

            # --- On re-résoud u = argmax(EFF) (batch KB si n_batch_EFF > 1) ---
            _batch_pts, _ = _find_batch_EFF_points(g_ot, sigma_func, xt, yt, all_grad)
            u_opt = ot.Point(_batch_pts[0].tolist())
            f = ot.Function(EFFFunction(g_ot, sigma_func))

        _sigG2 = sigma_func(u_opt)
        _muG2  = g_ot(ot.Point(u_opt))[0]
        _eps2  = epsilon_factor * _sigG2
        if _sigG2 > 0:
            # Decomposition fournie par _reliability/eff.py : elle est derivee de
            # l'implementation unique, donc la ligne imprimee ne peut plus
            # diverger du critere optimise. La copie manuelle qui se trouvait
            # ici etait fausse (quatrieme terme), de 39 % a 1271 % d'ecart.
            _tt = _eff.eff_termes(_muG2, _sigG2, epsilon_factor)
            term1, term2, term3, term4 = (float(x) for x in _tt[3:])
            print(f"  EFF converge debug : u_opt={list(np.round(np.array(u_opt),4))}  sigmaG={_sigG2:.8f}  muG={_muG2:.8f}  epsilon={_eps2:.8f}", flush=True)
            print(f"    t1={float(_tt[0]):.6f}  t2={float(_tt[1]):.6f}  t3={float(_tt[2]):.6f}", flush=True)
            print(f"    cdf(t1)={norm.cdf(float(_tt[0])):.8e}  cdf(-t2)={norm.cdf(-float(_tt[1])):.8e}  cdf(t3)={norm.cdf(float(_tt[2])):.8e}", flush=True)
            print(f"    pdf(t2)={norm.pdf(float(_tt[1])):.8e}  pdf(t3)={norm.pdf(float(_tt[2])):.8e}", flush=True)
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

        _eff_history_beta_IS = list(list_beta_IS)
        return g_ot, sigma_func, xt, yt, all_grad, xt_eff

    # --------------------------------------------------------------------------- #
    # FONCTIONS RESULTATS/ AFFICHAGE                                              #

    # =========================================================================
    # HELPERS VECTORISES (batch eval surrogate sur grille - BLAS multi-thread)
    # =========================================================================
    def _batch_mu_sigma(g_ot, sigma_func, grid):
        return _eff_ot.batch_mu_sigma(g_ot, sigma_func, grid,
                                   predict_pck if do_PCK else predict_gepck)

    def _eff_vectorized(mu, sigma, eps_factor):
        return _eff.eff(mu, sigma, eps_factor)
    # =========================================================================

    # --- Graphiques de suivi : _reliability/graphiques.py. Les courbes Pf
    # lineaire et log y sont une seule fonction, l'echelle etant un parametre.
    def print_EFF_graphs():
        return _graphiques.tracer_convergence_eff(
            _eff_history_EFF, _eff_history_BB, _eff_history_BS,
            _eff_history_theta, params_names, tol_EFF, tol_BB, tol_BS,
            out_dir_eff, timestamp)

    def print_Pf_evolution():
        return _graphiques.tracer_pf_evolution(
            _eff_history_Pf, modele, EFF_criteria, out_dir_eff, timestamp,
            'lineaire')

    def print_logPf_evolution():
        return _graphiques.tracer_pf_evolution(
            _eff_history_Pf, modele, EFF_criteria, out_dir_eff, timestamp, 'log')

    # --- HF GRID CACHE ---
    _HF_CACHE_FILE       = os.path.join(_path_ds, "hf_grid_cache.json")
    _HF_CACHE_FILE_FINAL = os.path.join(_path_ds, "hf_grid_cache_final.json")
    hf_2d_grid_fixed_final = None

    def _load_hf_cache(n_grid_hf_local, cache_file, sd):
        return _cache_hf.load_hf_cache(n_grid_hf_local, cache_file, sd, config_is_identical)

    def _save_hf_cache(Z, n_grid_hf_local, cache_file, sd):
        return _cache_hf.save_hf_cache(Z, n_grid_hf_local, cache_file, sd)

    def _save_hf_cache_partial(Z_flat, n_total, cache_file, sd):
        return _cache_hf.save_hf_cache_partial(Z_flat, n_total, cache_file, sd)

    def _load_hf_cache_partial(cache_file, sd, n_total):
        return _cache_hf.load_hf_cache_partial(cache_file, sd, n_total, config_is_identical)

    def _compute_hf_grid_with_progress(grid_hf, n_grid_hf_local, context="",
                                        cache_file=None, sd=None, grid_var_name='hf_2d_grid_fixed'):
        """Calcule la grille HF point par point avec progress + ETA.
        Lecture/ecriture automatique d'un cache sidecar JSON.
        Sauvegarde incrementale dans cache_file.partial (reprise apres crash).
        Retourne Z (n_grid_hf x n_grid_hf)."""
        global hf_2d_grid_fixed, hf_2d_grid_fixed_final
        if cache_file is None:
            cache_file = _HF_CACHE_FILE
        if sd is None:
            sd = slice_def
        cached = _load_hf_cache(n_grid_hf_local, cache_file, sd)
        if cached is not None:
            _grid_dict = {'params': {'slice_def': sd, 'n_grid_hf': n_grid_hf_local}, 'Z': cached.tolist()}
            if grid_var_name == 'hf_2d_grid_fixed_final':
                hf_2d_grid_fixed_final = _grid_dict
            else:
                hf_2d_grid_fixed = _grid_dict
            return cached
        import time as _time_local
        _point_log_phase[0] = "HF"
        n_total = len(grid_hf)
        # Charger le cache partiel (reprise apres crash)
        Z_flat = _load_hf_cache_partial(cache_file, sd, n_total)
        if Z_flat is None:
            Z_flat = [None] * n_total
        _n_skipped = sum(1 for v in Z_flat if v is not None)
        _n_to_compute = n_total - _n_skipped
        _t_start = _time_local.perf_counter()
        print(f"\n##### HF GRID START: {n_grid_hf_local}x{n_grid_hf_local} = {n_total} points solveur ({context})"
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
        print(f"\n##### HF GRID DONE in {_t_total:.1f} min ({_n_computed} appels solveur, {_n_skipped} skip) #####\n", flush=True)
        Z = np.array(Z_flat, dtype=float).reshape(n_grid_hf_local, n_grid_hf_local)
        _grid_dict = {'params': {'slice_def': sd, 'n_grid_hf': n_grid_hf_local}, 'Z': Z.tolist()}
        if grid_var_name == 'hf_2d_grid_fixed_final':
            hf_2d_grid_fixed_final = _grid_dict
        else:
            hf_2d_grid_fixed = _grid_dict
        _save_hf_cache(Z, n_grid_hf_local, cache_file, sd)
        # supprimer le cache partiel
        _partial_file = cache_file + '.partial'
        if os.path.exists(_partial_file):
            os.remove(_partial_file)
        return Z

    # --- HF GRILLE FULL (n_var-D) ---
    _HF_FULL_CACHE_FILE = os.path.join(_path_ds, "hf_grid_full_cache.json")
    _hf_grid_full = [None]   # [Z_full] en memoire (array n_var-D), liste pour mutabilite dans closures
    _hf_grid_full_axes = [None]  # axes de la grille (liste de 1D arrays)

    def _load_hf_grid_full():
        return _cache_hf.load_hf_grid_full(
            _HF_FULL_CACHE_FILE, n_var, n_grid_hf, config_is_identical)

    def _save_hf_grid_full(Z_full):
        return _cache_hf.save_hf_grid_full(_HF_FULL_CACHE_FILE, Z_full, n_var, n_grid_hf)

    def _compute_hf_grid_full():
        """Calcule la grille HF complete (n_grid_hf^n_var points solveur)."""
        cached = _load_hf_grid_full()
        if cached is not None:
            _hf_grid_full[0] = cached
            axes = [np.linspace(u1_min, u1_max, n_grid_hf) for _ in range(n_var)]
            _hf_grid_full_axes[0] = axes
            return cached
        import time as _time_local
        _point_log_phase[0] = "HF_FULL"
        axes = [np.linspace(u1_min, u1_max, n_grid_hf) for _ in range(n_var)]
        grids = np.meshgrid(*axes, indexing='ij')
        grid_flat = np.column_stack([g.ravel() for g in grids])
        n_total = len(grid_flat)
        Z_flat = []
        _t_start = _time_local.perf_counter()
        print(f"\n##### HF FULL GRID START: {n_grid_hf}^{n_var} = {n_total} points solveur #####", flush=True)
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
        print(f"\n##### HF FULL GRID DONE in {_t_total:.1f} min ({n_total} appels solveur) #####\n", flush=True)
        Z_full = np.array(Z_flat).reshape([n_grid_hf] * n_var)
        _hf_grid_full[0] = Z_full
        _hf_grid_full_axes[0] = axes
        _save_hf_grid_full(Z_full)
        return Z_full

    def _extract_hf_slice(sd):
        """Extrait une coupe 2D (n_grid_hf x n_grid_hf) depuis la grille full par interpolation."""
        from scipy.interpolate import RegularGridInterpolator
        idx_x, idx_y, fixed = sd
        axes = _hf_grid_full_axes[0]
        interp = RegularGridInterpolator(axes, _hf_grid_full[0], method='linear')
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
    _hf_custom_result = [None]   # cache memoire (evite relecture JSON a chaque print)

    def _hf_from_custom_points(sd):
        """Calcule g par run_HF sur les coordonnees hf_custom_points [[u_s, u_fy], ...],
        puis interpole (griddata) sur une grille reguliere.
        Cache incremental : sauve apres chaque point, reprend apres crash.
        Retourne (Z_true, UX_hf, UY_hf) ou (None, None, None) si hf_custom_points est None."""
        if hf_custom_points is None:
            return None, None, None
        if _hf_custom_result[0] is not None:
            return _hf_custom_result[0]
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
                    _hf_custom_result[0] = (Z_true, UX_hf, UY_hf)
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
                _point_log_phase[0] = "HF_CUSTOM"
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
        _hf_custom_result[0] = (Z_true, UX_hf, UY_hf)
        return Z_true, UX_hf, UY_hf

    def _get_hf_slice(sd, cache_file=None, grid_var_name='hf_2d_grid_fixed'):
        """Retourne Z_true (n_grid_hf x n_grid_hf) pour une coupe sd.
        Cascade : cache 2D memoire -> cache 2D disque -> grille full -> recalcul 2D."""
        global hf_2d_grid_fixed, hf_2d_grid_fixed_final
        if cache_file is None:
            cache_file = _HF_CACHE_FILE
        # 1. Cache 2D memoire
        _mem = hf_2d_grid_fixed_final if grid_var_name == 'hf_2d_grid_fixed_final' else hf_2d_grid_fixed
        if _mem is not None:
            return np.array(_mem['Z'])
        # 2. Cache 2D disque
        Z_cached = _load_hf_cache(n_grid_hf, cache_file, sd)
        if Z_cached is not None:
            _grid_dict = {'params': {'slice_def': sd, 'n_grid_hf': n_grid_hf}, 'Z': Z_cached.tolist()}
            if grid_var_name == 'hf_2d_grid_fixed_final':
                hf_2d_grid_fixed_final = _grid_dict
            else:
                hf_2d_grid_fixed = _grid_dict
            return Z_cached
        # 3. Grille full
        if _hf_grid_full[0] is not None:
            print(f"[HF SLICE] extraction depuis grille full pour coupe ({sd[0]},{sd[1]})", flush=True)
            Z = _extract_hf_slice(sd)
            _save_hf_cache(Z, n_grid_hf, cache_file, sd)
            _grid_dict = {'params': {'slice_def': sd, 'n_grid_hf': n_grid_hf}, 'Z': Z.tolist()}
            if grid_var_name == 'hf_2d_grid_fixed_final':
                hf_2d_grid_fixed_final = _grid_dict
            else:
                hf_2d_grid_fixed = _grid_dict
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
        global hf_2d_grid_fixed
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
        _pce_line   = f'\n{_gepck_pce_label}' if _gepck_pce_label else ''
        _theta_str  = '  theta=[' + ', '.join(f'{v:.3f}' for v in _eff_history_theta[-1]) + ']' if _eff_history_theta else ''
        _loo_str    = f'  LOO={_gepck_loo:.3e}' if _gepck_loo is not None else ''
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
        fname = f'EFF_{n_added}points_{timestamp}.png'
        fig.savefig(os.path.join(out_dir_eff, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [EFF visu] -> {fname}", flush=True)

    def print_globalplanche_EFF(xt, yt, all_grad, xt_eff):
        """Planche globale EFF : 3 colonnes (EFF, sigma, g) x N lignes (DOE initial + chaque etape EFF).
        Refit le surrogate a chaque etape. Utilise slice_def_final pour la coupe.
        Grille HF calculee une seule fois et reutilisee."""
        global hf_2d_grid_fixed_final
        if slice_def_final is None:
            print("[GLOBAL PLANCHE] slice_def_final est None, skip", flush=True)
            return
        idx_x, idx_y, fixed = slice_def_final
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
            Z_true, UX_hf, UY_hf = _hf_from_custom_points(slice_def_final)
        elif print_HF:
            ux_hf = np.linspace(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1, n_grid_hf)
            uy_hf = np.linspace(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1, n_grid_hf)
            UX_hf, UY_hf = np.meshgrid(ux_hf, uy_hf)
            if slice_def_final == slice_def:
                Z_true = _get_hf_slice(slice_def, _HF_CACHE_FILE, 'hf_2d_grid_fixed')
            else:
                Z_true = _get_hf_slice(slice_def_final, _HF_CACHE_FILE_FINAL, 'hf_2d_grid_fixed_final')

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
        fname = f'globalplanche_EFF_{timestamp}.png'
        fig.savefig(os.path.join(out_dir_eff, fname), dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  [GLOBAL PLANCHE] -> {fname}", flush=True)

    def print_visu_EFF(g_ot, sigma_func, xt, xt_eff):
        """Carte 2D des valeurs du critere EFF sur la meme grille que print_visu."""
        global hf_2d_grid_fixed
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
            if hf_2d_grid_fixed is not None:
                Z_true = np.array(hf_2d_grid_fixed['Z'])
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
        global hf_2d_grid_fixed
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
            if hf_2d_grid_fixed is not None:
                Z_true = np.array(hf_2d_grid_fixed['Z'])
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
        _point_log_phase[0] = "USTAR"
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
            if _fosm_u0_cache[0] is None:
                g0_HF, grad_HF_U0, _ = run_HF(u0)
                _fosm_u0_cache[0] = (g0_HF, grad_HF_U0)
            else:
                g0_HF, grad_HF_U0 = _fosm_u0_cache[0]
                print("  [FOSM] run_HF([0,0]) reutilise du cache (pas de SOCP redondant)", flush=True)
            u_FOSM         = grad_HF_U0 * (-g0_HF / grad_HF_U0.normSquare())
            print(f"u* FOSM (HF) = {[round(v, 4) for v in u_FOSM]}", flush=True)
            print(f"Erreur FOSM  = {(u_FOSM - u_star).norm() / u_star.norm():.4f}", flush=True)


    def print_visu(best_result, best_sps, xt, g_ot, modes, xt_eff):
        global u1_min, u1_max, u2_min, u2_max, hf_2d_grid_fixed, slice_def_final
        if slice_def_final is None:
            if best_result is not None:
                _imp = np.array(best_result.getImportanceFactors())
                _top2 = list(np.argsort(_imp)[::-1][:2])
                _u_star = np.array(best_result.getStandardSpaceDesignPoint())
                slice_def_final = (min(_top2), max(_top2),
                                   {i: float(_u_star[i]) for i in range(n_var) if i not in _top2})
            else:
                slice_def_final = slice_def
        idx_x, idx_y, fixed = slice_def_final

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
            Z_true, UX_hf, UY_hf = _hf_from_custom_points(slice_def_final)
            if Z_true is not None:
                ax.contour(UX_hf, UY_hf, Z_true, levels=[0], colors='red', linewidths=2, linestyles='--')
        elif print_HF:
            ux_hf = np.linspace(eff_bounds_min[0] - 1, eff_bounds_max[0] + 1, n_grid_hf)
            uy_hf = np.linspace(eff_bounds_min[1] - 1, eff_bounds_max[1] + 1, n_grid_hf)
            UX_hf, UY_hf = np.meshgrid(ux_hf, uy_hf)
            if slice_def_final == slice_def:
                Z_true = _get_hf_slice(slice_def, _HF_CACHE_FILE, 'hf_2d_grid_fixed')
            else:
                Z_true = _get_hf_slice(slice_def_final, _HF_CACHE_FILE_FINAL, 'hf_2d_grid_fixed_final')
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
        fname = f'visu{modele}_{timestamp}.png'
        fig.savefig(os.path.join(out_dir_eff, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [visu] -> {fname}", flush=True)

    def print_3D_HF():
        if hf_3d_grid_fixed is not None:
            print("Cache hf_3d_grid_fixed disponible — pas d'appels solveur.", flush=True)
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
            print(f"Evaluation HF grille {n_grid_hf}x{n_grid_hf} = {n_grid_hf**2} appels solveur...", flush=True)
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

    # --------------------------------------------------------------------------- #
    # FONCTION IS POST-FORM                                                       #
    def run_IS(modes, event):
        return _form.run_IS(modes, event, n_var, n_IS, cov_IS)

    def run_IS_proj(modes, event_proj):
        return _form.run_IS_proj(modes, event_proj, n_var, n_IS, cov_IS,
                                 _find_position_var_index())

    def print_results_IS(result_IS):
        return _form.print_results_IS(result_IS)

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
        # Sans ce controle, l'absence du dump donnait un FileNotFoundError brut
        # APRES plusieurs minutes de construction du modele CAD. Le cas se
        # produit des qu'on reprend une etude sur un autre poste : le dump vit
        # dans le .ds du modele, il n'est pas dans le depot.
        if not os.path.isfile(_RESTART_STATE_FILE):
            raise SystemExit(
                "restart_enrich_only = true, mais aucun etat a reprendre :\n"
                "  %s\n\n"
                "Ce fichier est produit par un run precedent, dans le .ds du\n"
                "modele. Pour repartir de zero, mettre\n"
                "  restart_enrich_only = false\n"
                "dans le fichier d'etude." % _RESTART_STATE_FILE)
        _rs = json.load(open(_RESTART_STATE_FILE))
        xt = np.array(_rs['xt'], float)
        yt = np.array(_rs['yt'], float)
        all_grad = np.array(_rs['all_grad'], float)
        _restart_xt_eff = [np.array(p, float) for p in _rs['xt_eff']]
        if _rs.get('max_degree') is not None:
            max_degree = int(_rs['max_degree'])
        hf_2d_grid_fixed = _rs.get('hf_2d_grid')
        _eff_history_EFF   = list(_rs.get('hist_EFF', []))
        _eff_history_BB    = list(_rs.get('hist_BB', []))
        _eff_history_BS    = list(_rs.get('hist_BS', []))
        _eff_history_theta = list(_rs.get('hist_theta', []))
        _eff_history_beta_IS = list(_rs.get('hist_beta_IS', []))
        _enrich_round     = int(_rs.get('enrich_round', 0)) + 1
        _round_sizes_prev = list(_rs.get('round_sizes', [int(len(xt))]))
        _point_log_round[0] = _enrich_round
        with open(_POINT_LOG_FILE, "a") as _pf:
            _pf.write(json.dumps({"phase": "_RESTART", "round": _enrich_round,
                                  "n_total": int(len(xt)), "n_eff": len(_restart_xt_eff)}) + "\n")
        print(f"[RESTART] charge {len(xt)} pts (dont {len(_restart_xt_eff)} EFF) "
              f"depuis {_RESTART_STATE_FILE} (round {_enrich_round})", flush=True)

        # max_degree fixe (LARS gere P > N)
        event, g_ot, sigma_func = None, None, None
        xt_eff = list(_restart_xt_eff)
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
