"""
CODE FIABILITE - VERSION AVEC DEFINITION DE FONCTIONS

Ce script N'IMPORTE PLUS Digital Structure (phase 5). L'evaluation de l'etat
limite passe par `solver/fabrique.py`, qui ne charge que l'implementation
demandee par le fichier d'etude. Les 2 700 lignes qui suivent -- plan
d'experiences, metamodele, enrichissement EFF, FORM multimodal, tirage
d'importance -- ne dependent donc plus d'une licence.
"""
import os
import json
import re
import sys                  # etait utilise 8 fois SANS etre importe : il ne
                            # marchait que parce que le `import *` de Digital
                            # Structure le laissait fuiter dans les globales.

#: racine du depot, deduite de ce fichier -- aucun chemin absolu.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import openturns as ot
import numpy as np
import matplotlib
_HEADLESS = bool(os.environ.get("_IS_PARALLEL")) or bool(os.environ.get("_FIAB_LOG_REDIRECTED"))
matplotlib.use('Agg' if _HEADLESS else 'TkAgg')
# `pyplot` n'est plus importe ici : l'etude ne dessine plus rien.
# Le choix du backend, lui, doit rester AVANT tout import de figure.
import re
import math
from datetime import datetime
# fit_gepck/fit_pck sont partis avec le dispatch dans
# `_surrogate/ajuster.py` ; l'etude ne fait plus que PREDIRE.
from api import predict_gepck, predict_pck
# `loi_F_permanente` ne survivait que par une ligne de PARAM_CONFIG mise
# en commentaire.
from lois import loi_fc, loi_fy
import lois as _lois
import doe as _cache_doe
import journal_points as _journal_points
import reprise as _reprise
import eff_ot as _eff_ot
import form as _form
import controle as _controle
import enrichissement as _enrichissement
import graphiques as _graphiques
import figurer as _figurer
# NOM : `ajuster`, pas `fit` -- `_lib/fit.py` (le clone UQLab) porte
# deja ce nom sur le chemin d'import, et l'aurait eclipse.
import ajuster as _fit
import projection as _projection
import grille as _grille
import evaluation as _evaluation
import plan as _plan
import parallele as _parallele
import schema as _schema
from fabrique import solveur as _fabriquer_solveur


def _parse(text, name):
    return float(re.search(rf'(?m)^\s*{re.escape(name)}\s*=\s*([\d.]+)', text).group(1))

if __name__ == '__main__':
    # ------------------------------------------------------------------------ #
    # Les cinquante-trois parametres de reglage de ce script tenaient ici, en
    # affectations litterales melees aux commentaires. Ils sont desormais dans
    # `studies/pure_flexion.toml`, valides par `_config/schema.py`.
    #
    # Ce que le fichier d'etude change, des maintenant :
    #   * une cle mal orthographiee est REFUSEE. `tol_FROM = 0.05` etait jusqu'ici
    #     accepte en silence et le parametre reel gardait son defaut ;
    #   * un `modele` inconnu est refuse. Il mettait les sept drapeaux `do_*` a
    #     False et le calcul partait sans metamodele, sans un mot ;
    #   * les sept `do_*`, et les corrections `do_IS = do_IS and modele != 'HF'`
    #     posees cinquante lignes plus bas, deviennent des valeurs DERIVEES :
    #     elles ne peuvent plus contredire le choix de l'utilisateur ;
    #   * la configuration effective est imprimee en tete de journal et deposee
    #     en JSON a cote des sorties. La mesure du 25/08/2026 a montre qu'une
    #     comparaison de deux runs ne vaut rien tant qu'on ne peut pas prouver
    #     qu'ils partageaient la meme configuration.
    #
    # Le bloc de liaison ci-dessous n'est pas une destination : il existe tant
    # que les 2 700 lignes suivantes lisent des variables globales. La phase 5
    # passera `CFG` en argument. Une affectation posee APRES ce bloc l'emporte
    # encore, le temps de la transition -- c'est ce dont se sert
    # `tools/run_comparatif.py` pour imposer une configuration d'essai.
    CFG = _schema.charger(os.environ.get("FIABILITE_ETUDE")
                          or os.path.join(_REPO, "studies", "pure_flexion.toml"))
    print(_schema.resume(CFG), flush=True)

    # Dossier de l'etude : celui de CE fichier. Il portait le chemin absolu du
    # poste de l'auteur (C:\_workingDir\_SF\test flexion\pure_flexion), qui
    # n'existe nulle part ailleurs : `tools/run_comparatif.py` devait le
    # reecrire pour qu'un run soit seulement possible ici.
    path_dir = os.path.dirname(os.path.abspath(__file__))
    storage = CFG.storage

    # --- liaison aux noms attendus par la suite du script --------------------
    modele              = CFG.modele
    n0                  = CFG.n0
    max_degree          = CFG.max_degree
    do_EFF              = CFG.eff_actif
    epsilon_factor      = CFG.epsilon_factor
    tol_EFF             = CFG.tol_EFF
    tol_BB              = CFG.tol_BB
    tol_BS              = CFG.tol_BS
    EFF_criteria        = CFG.EFF_criteria
    n_NLopt_EFF         = CFG.n_NLopt_EFF
    n_batch_EFF         = CFG.n_batch_EFF
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
    print_3D            = CFG.print_3D
    print_ana           = CFG.print_ana
    print_Pf            = CFG.print_Pf
    print_grad_sp       = CFG.print_grad_sp
    print_EFF_progres   = CFG.print_EFF_progres
    do_custom_hf        = CFG.do_custom_hf
    hf_2d_grid_fixed    = CFG.hf_2d_grid_fixed
    hf_3d_grid_fixed    = CFG.hf_3d_grid_fixed

    # Les drapeaux de modele sont DERIVES, donc mutuellement exclusifs par
    # construction. Ils etaient sept lignes ecrites a la main, qui pouvaient
    # toutes valoir False sur une faute de frappe. Trois seulement sont lus
    # ici ; les quatre autres restent disponibles sur `CFG`.
    do_HF        = CFG.do_HF
    do_GEPCK     = CFG.do_GEPCK
    do_PCK       = CFG.do_PCK

    # Un worker de DOE parallele travaille sur une copie isolee du modele : son
    # nom lui est impose par le processus pere, pas par le fichier d'etude.
    modelname = os.environ.get("_DOE_WORKER_MODELNAME") or CFG.modelname
    _path_ds = os.path.join(storage, modelname + ".ds")
    with open(os.path.join(_path_ds, 'dsCad.txt'), 'r') as f:
        _cad_txt = f.read()

    print("=" * 70)
    print("CALCUL DE FIABILITE -- FLEXION PURE BETON")
    print("=" * 70)

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
    # Ce que le dernier ajustement dit de lui-meme : `_surrogate/ajuster.py`.
    _DIAG = _fit.Diagnostics()

    # --- Historiques EFF (mis a jour par run_EFF et init_g_ot, lus par print_EFF_graphs) ---
    _eff_history_EFF   = []   # EFF(u_opt) avant ajout de chaque point (incl. initial)
    _eff_history_BB    = []   # ratio BB par iteration (None si FORM echoue)
    _eff_history_BS    = []   # ratio BS par iteration (None si calcul impossible)
    _eff_history_Pf    = []   # Pf_IS (mid/sup/inf) par iter, inconditionnel
    _fosm = [None]     # l'objet ErreurFOSM, construit au premier usage
    _eff_history_beta_IS = []   # snapshot de list_beta_IS (locale a run_EFF) pour le dump restart
    _enrich_round     = 0       # 0 = run initial, k = k-ieme reprise
    _round_sizes_prev = []      # taille de chaque round precedent (charge du dump)
    _restart_xt_eff   = []      # points EFF charges du dump (seeder xt_eff en reprise)

    # --- Sortie PNG EFF ---
    timestamp   = datetime.now().strftime('%d%m_%H%M')
    out_dir_eff = CFG.dossier_png_eff(timestamp, os.path.join(path_dir, 'output'))
    os.makedirs(out_dir_eff, exist_ok=True)
    _schema.ecrire_trace(CFG, out_dir_eff)   # configuration effective, a cote des figures

    # --------------------------------------------------------------------------- #
    # APPEL AU SOLVEUR ET PLAN D'EXPERIENCES                                             #


    # --- DISTRIBUTIONS ---
    

    # --- PARAM_CONFIG : catalogue des variables aleatoires ---
    n_rebars = len(re.findall(r'REBAR\(', _cad_txt))
    rebar_names = [f"HA{i+1}" for i in range(n_rebars)]
    PARAM_CONFIG_CAD = {
        'fc': {'sens': {"param": "COMPRESSIVE_STRENGTH", "solids": ["Block1"], "region_key": "fc"},
               'loi': loi_fc, 'args': (48, 0.12)},
        'fy': {'sens': {"param": "YIELD_STRENGTH", "rebars": rebar_names, "region_key": "fy"},
               'loi': loi_fy, 'args': (550, None)},
    }
    PARAM_CONFIG_LOAD = {
        # 'F':  {'sens': {"param": "LIVE_LOAD", "load_case": "Load_case0", "region_key": "F"},
        #        'loi': loi_F_permanente, 'args': (1.0, 0.05)},
    }
    PARAM_CONFIG = {**PARAM_CONFIG_LOAD, **PARAM_CONFIG_CAD}
    params_names = list(PARAM_CONFIG_LOAD.keys()) + list(PARAM_CONFIG_CAD.keys())
    n_var = len(params_names)
    _rk = [PARAM_CONFIG[p]['sens'].get('region_key') for p in params_names]
    assert all(_rk), f"region_key manquant dans PARAM_CONFIG : {[p for p, r in zip(params_names, _rk) if not r]}"
    assert len(set(_rk)) == len(_rk), f"region_key dupliques : {_rk}"
    if not set(params_names) <= set(PARAM_CONFIG_CAD.keys()):
        print_ana = False
    slice_def = (0, 1, {i: 0.0 for i in range(n_var) if i > 1})
    slice_def_final = (0, 1, {})           # 2 variables : pas de coupe, plan complet fy vs fc

    # Bornes du domaine de recherche. Codees en dur a +/- 7,5 jusqu'au
    # 26/08/2026 (oubli de la phase 4b), elles viennent maintenant du fichier
    # d'etude. L'espace standard n'a pas de sens physique par lui-meme : ce
    # sont les LOIS qui le lui donnent, et une loi non bornee ne borne rien.
    # Voir `_config/schema.py:eff_bound_min` -- un point a fy = 8,9 MPa a tue
    # le solveur.
    eff_bounds_min = [CFG.eff_bound_min] * n_var
    eff_bounds_max = [CFG.eff_bound_max] * n_var


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

    # Appele ICI, et pas au moment ou les bornes sont posees : la trace a
    # besoin de `dist_jointe`, qui n'est definie qu'au-dessus.
    # Ce que les bornes valent EN UNITES PHYSIQUES. Un run de deux heures est
    # mort le 26/08/2026 parce que `[-7.5, +7.5]` ne se lit pas comme
    # « rapport 52 entre les deux nappes ».
    _figurer.tracer_domaine_physique(
        dist_jointe(), params_names, CFG.eff_bound_min, CFG.eff_bound_max)

    # --- APPELS AU SOLVEUR -----------------------------------------------
    # Toute la mecanique Digital Structure -- reecriture du dsCad, maillage,
    # SOCP, lecture du dsmetares, archivage des sorties -- vit maintenant dans
    # `solver/digital_structure.py`. Elle tenait ici en QUATRE exemplaires
    # (`run_one_SOL` et `run_HF`, dans les deux scripts AC) qui avaient
    # diverge : `run_HF` codait en dur `global_physical_size = 0.05` et
    # `geometric_approximation_min = "4"` la ou `run_one_SOL` lisait la
    # configuration -- alors que les deux alimentent le meme metamodele.
    # `tests/golden/options_ds.json` en garde la trace.
    #
    # Un solveur par modele, memoise -- il porte un compteur d'appels que
    # reconstruire remettrait a zero. Le cache vit dans `_doe/parallele.py`.
    _solveur = _parallele.fabrique_memoisee(
        lambda **kw: _fabriquer_solveur(CFG.solveur, **kw),
        modelname, lambda nom: os.path.join(storage, nom + ".ds"),
        dossier_etude=path_dir, params_names=params_names,
        regions=[PARAM_CONFIG[p]['sens'] for p in params_names],
        global_size=global_size, geo_min_approx=geo_min_approx,
        max_size=CFG.max_size, solveur_lineaire=CFG.solveur_lineaire,
        archiver=save_history)

    # L'unique passage vers le solveur est dans `_doe/evaluation.py`. Il est
    # construit PARESSEUSEMENT : il a besoin du journal de points et du cache
    # incremental, definis plus bas dans ce fichier.
    _evaluateur = [None]

    def _obtenir_evaluateur():
        if _evaluateur[0] is None:
            _evaluateur[0] = _evaluation.Evaluateur(
                solveur_pour=_solveur,
                dist=dist_jointe(),
                params_names=params_names,
                exclure_non_converges=CFG.exclure_points_non_converges,
                archiver=save_history,
                journaliser=_JOURNAL.enregistrer,
                sauver_partiel=_save_doe_cache_incremental)
        return _evaluateur[0]

    def run_one_SOL(modelname, SOL, params_names, sensitivity=False,
                    with_sens_dict=None):
        # `with_sens_dict` n'a JAMAIS ete lu ; il reste dans la signature le
        # temps que les deux sites d'appel soient repris.
        return _obtenir_evaluateur().evaluer_plan(
            SOL, modelname, sensibilite=sensitivity)

    def run_HF(u):
        """L'etat limite en un point de l'espace standard. UN appel solveur."""
        return _obtenir_evaluateur().evaluer_en_U(u)


    # --- DOE PARALLELE ---
    def run_DOE_parallel(base_modelname, SOL, params_names, n_workers):
        """Le plan d'experiences reparti. Tout est dans `_doe/parallele.py`."""
        return _parallele.evaluer_plan_en_parallele(
            SOL, params_names, storage, base_modelname, n_workers,
            script_etude=__file__, repo=_REPO)

    def run_HF_grid_parallel(u_points, n_workers=3):
        """Les points d'une grille, repartis. Dans `_doe/parallele.py`."""
        return _parallele.evaluer_points_en_parallele(
            u_points, dist_jointe(), params_names, storage, modelname,
            n_workers, script_etude=__file__, repo=_REPO)


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

    # Le chargement et la sauvegarde du cache complet sont partis avec
    # `build_DOE` : son seul appelant. Ne reste que la sauvegarde
    # INCREMENTALE, qui appartient a l'evaluateur -- elle s'ecrit pendant le
    # plan, point par point, pour qu'une interruption ne coute pas tout.
    def _save_doe_cache_incremental(SOL, n_done):
        return _cache_doe.save_doe_cache_incremental(
            _DOE_CACHE_FILE, n0, params_names, SOL, n_done,
            signature=_SIG_SOLVEUR)

    # --- SIGNATURE INFORMATIVE (utilisee par le dump restart, pas par le DOE cache) ---
    def _doe_cache_sig():
        return _cache_doe.doe_cache_sig(n0, params_names, n_var, modelname)

    # --- DUMP RESTART ---
    _RESTART_STATE_FILE = _reprise.fichier_de(_path_ds)
    def _save_restart_state(xt, yt, all_grad, xt_eff, best_result, best_sp, modes, result_IS):
        """Le dump de reprise -- jusqu'a 90 heures de calcul dans un fichier.

        La serialisation est dans `_cache/reprise.py`. Ne reste ici que ce
        qui appartient a l'etude : ou vivent les historiques et la coupe.
        """
        # Les deux appels -- fin de round dans `run_EFF`, et dump final --
        # sont posterieurs a la construction de `_GRILLE` : le `try/except`
        # d'origine autour de cette lecture ne protegeait rien. L'ordre est
        # verifie par `test_105_reprise`. Les deux clefs de `coupes` existent
        # des la construction, `.get` suffit.
        return _reprise.enregistrer(
            _RESTART_STATE_FILE,
            signature=_doe_cache_sig(),
            signature_solveur=CFG.signature_solveur(),
            modele=modele, timestamp=timestamp, max_degree=max_degree, n0=n0,
            xt=xt, yt=yt, all_grad=all_grad, xt_eff=xt_eff,
            enrich_round=_enrich_round, round_sizes_prev=_round_sizes_prev,
            historiques={"EFF": _eff_history_EFF, "BB": _eff_history_BB,
                         "BS": _eff_history_BS, "theta": _DIAG.theta,
                         "beta_IS": _eff_history_beta_IS, "Pf": _eff_history_Pf},
            coupe_hf=_GRILLE.coupes.get("courante"),
            best_result=best_result, best_sp=best_sp, modes=modes,
            result_IS=result_IS)

    # --- LOG INCREMENTAL PAR POINT ---
    # Une ligne JSON par appel solveur : `_cache/journal_points.py`.
    _JOURNAL = _journal_points.JournalDesPoints(
        _journal_points.fichier_de(_path_ds), params_names)

    # --- DOE ---
    def build_DOE():
        """Le plan d'experiences initial : n0 appels solveur, ou zero.

        L'enchainement est dans `_doe/plan.py`, en un seul exemplaire pour les
        deux etudes. Ne restent ici que le nom du modele et la facon d'appeler
        le solveur -- les deux seules choses qui different d'une etude a
        l'autre.
        """
        return _plan.construire_plan_initial(
            CFG, n0, dist_jointe=dist_jointe, params_names=params_names,
            bornes_min=eff_bounds_min, bornes_max=eff_bounds_max,
            fichier_cache=_DOE_CACHE_FILE, signature=_SIG_SOLVEUR,
            executer_plan=lambda SOL: run_one_SOL(
                modelname, SOL, params_names, sensitivity=True),
            executer_en_parallele=lambda SOL, nw: run_DOE_parallel(
                modelname, SOL, params_names, nw),
            journaliser=_JOURNAL.enregistrer)

    def build_starting_points():
        """Les points de depart du FORM multimodal. ZERO appel solveur."""
        return _plan.tirer_points_de_depart(n_sp, eff_bounds_min, eff_bounds_max)


    # --------------------------------------------------------------------------- #
    # FONCTION ANALYTIQUE                                                         #
    # --- Paramètres du modèle analytique ---
    Es = 200000
    ecu = 0.0035
    eud = 0.045
    gamma_c_fic = _parse(_cad_txt, 'gamma_c') # fixé à 1.0
    gamma_s_fic = _parse(_cad_txt, 'gamma_s') # fixé à 1.0
    
    # --- Fonction analytique ---
    class flexion_claude:
        def __init__(self):

            # --- Lecture du dsCad et dsLoad ---
            path = os.path.join(storage, modelname + '.ds')
            with open(os.path.join(path, 'dsCad.txt'), 'r') as f:
                _cad = f.read()
            with open(os.path.join(path, 'dsLoad.txt'), 'r') as f:
                _load = f.read()

            b   = _parse(_cad, 'b')
            h   = _parse(_cad, 'h')
            L   = _parse(_cad, 'L')
            phi = _parse(_cad, 'phi')

            n_bars = len(re.findall(r'REBAR\(', _cad))
            As = n_bars * math.pi * (phi / 2e3) ** 2

            z_rebar = [float(v) for v in re.findall(
                r'pts\d+\.append\(POINT\([^,]+,\s*[^,]+,\s*([\d.]+)\)', _cad)]
            d = h/2 + sum(z_rebar) / len(z_rebar)

            F = abs(float(re.search(r"Z='(-?[\d.]+)'", _load).group(1)))
            Med = F * L

            # --- Définition de la transformation isoprobabiliste ---
            self.fym = PARAM_CONFIG['fy']['args'][0] if 'fy' in PARAM_CONFIG else 550
            dist_X     = dist_jointe()
            self.T_inv = dist_X.getInverseIsoProbabilisticTransformation()
            self.T     = dist_X.getIsoProbabilisticTransformation()

            # --- Définiton des constantes pour le cas aciers plastifiés ---
            self.A  = As * d / gamma_s_fic
            self.B  = - As**2 * gamma_c_fic / (2 * b * gamma_s_fic**2)
            self.C  = -Med

            # --- Calcul de la limite plastique ---
            self.Ap = 0.8*d*b / (As*gamma_c_fic*Es*ecu)
            self.Bp = 0.8*b*d**2 / gamma_c_fic
            self.Cp = 2*self.Ap*self.C/self.Bp 
            ap = 1
            bp = self.Cp - 0.8
            cp = self.Cp - 0.2
            Delta_p = bp**2 - 4*ap*cp
            sol1_s = (-bp + Delta_p**0.5) / (2*ap)
            sol1_x1 = (sol1_s**2 - 1) / (4*self.Ap)
            self.u1_lim_plast = self.T(ot.Point([sol1_x1, 0.0]))[0]

            # --- Limite de plasticité ---
            self.A1 = As*gamma_c_fic*Es*ecu/(0.8*b*d)
            self.A2 = Es*ecu*gamma_s_fic

        def u2p_LS(self, u1):
            x_point = self.T_inv(ot.Point([u1, 0.0]))
            x1 = x_point[0]
            a  = self.B
            b  = self.A * x1
            c  = self.C * x1
            Delta = b**2 - 4 * a * c
            fy = (-b + Delta**0.5) / (2 * a)
            return self.T(ot.Point([0.0, fy]))[1]

        def g(self, u1, u2):
            x_point = self.T_inv(ot.Point([u1, u2]))
            x1 = x_point[0]
            x2 = x_point[1]
            x1_lim_plast_x2 = self.A1*x2*(self.A2+x2)/self.A2**2
            if x1 > x1_lim_plast_x2:
                return (self.A*x2+self.B*x2**2/x1+self.C)/(-self.C)
            else :
                s = (1 + 4*self.Ap*x1)**0.5
                return -1 - (s-1)/self.Cp + 0.8*(s-1)/(self.Cp*(s+1))
    
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
        """Enveloppe du surrogate sur la variable de position, s'il y en a une.

        Les 35 lignes de minimisation sont dans `_surrogate/projection.py` :
        elles ne dependent que de `n_var` et de l'indice de la variable.
        """
        return _projection.projeter_surrogate(
            g_ot, n_var, _find_position_var_index())


    # --------------------------------------------------------------------------- #
    # FONCTIONS POUR FORM                                                         #
    def init_g_ot(g_ot, sigma_func, xt, yt, all_grad, fixed_fm=None):
        """Ajuste le metamodele sur le plan courant.

        Tout est dans `_surrogate/ajuster.py`. `g_ot` et `sigma_func` restent
        en entree parce que cinq sites d'appel les passent ; ils n'ont jamais
        ete lus.

        `max_degree` est passe A L'APPEL : une reprise l'ecrase.
        """
        return _fit.ajuster_sur_le_plan(
            CFG, xt, yt, all_grad, max_degree=max_degree,
            dist_X=dist_jointe(), diagnostics=_DIAG,
            evaluer_hf=run_HF, fixed_fm=fixed_fm)


    def init_FORM(g_ot, sigma_func, xt, yt, all_grad):
        """Le metamodele, puis l'evenement `g < 0` qu'il definit.

        L'evenement est dans `_reliability/form.py` : sa loi normale centree
        reduite est la definition de l'espace standard, pas un choix
        d'etude.
        """
        g_ot, sigma_func, xt, yt, all_grad = init_g_ot(g_ot, sigma_func, xt, yt, all_grad)
        return (_form.evenement_de_defaillance(g_ot, n_var),
                g_ot, sigma_func, xt, yt, all_grad)

    # --- Multi-start FORM depuis les points du DOE ---
    def FORM_all_modes(starting_points, tol_all_modes, event):
        return _form.form_all_modes(starting_points, tol_all_modes, event,
                                    n_var, n_max_FORM, tol_FORM,
                                    do_FORM_filter, eff_bounds_min, eff_bounds_max)
    
    # --- Warm-start FORM depuis les points du DOE ---
    def FORM_warm_start(modes, best_sps, g_ot, sigma_func, xt, yt, all_grad):
        """Relance FORM si le mode dominant ne tombe pas sur `g = 0`.

        Le mecanisme est dans `_reliability/form.py`, avec ce qu'il coute et
        ce qu'il n'apporte pas. Ne restent ici que les deux rappels qui
        appartiennent a l'etude : ajuster le metamodele, et chercher les
        modes.
        """
        def _reajuster_et_evenement(xt_k, yt_k, ag_k):
            evenement, _, _, xt_k, _, _ = init_FORM(g_ot, sigma_func,
                                                    xt_k, yt_k, ag_k)
            return evenement, xt_k

        return _form.warm_start(
            modes, best_sps, g_ot, xt, yt, all_grad, n_var=n_var,
            tolerance=tol_warmstart, multistart=do_multistart,
            tol_all_modes=tol_all_modes,
            reajuster_et_evenement=_reajuster_et_evenement,
            rechercher_modes=FORM_all_modes)

    # --------------------------------------------------------------------------- #
    # FONCTIONS D'ENRICHISSEMENT DU PLAN D'EXPERIENCE (EFF)                       #
    # --- Critere EFF : la formule est dans _reliability/eff.py, en un seul
    # exemplaire. Elle etait ecrite deux fois ici (vectorisee et scalaire).
    def EFFFunction(g_ot, sigma_func):
        return _eff_ot.eff_function(g_ot, sigma_func, n_var, epsilon_factor)

    def _find_batch_EFF_points(g_ot, sigma_func, xt, yt, all_grad):
        """Les points d'enrichissement du prochain tour.

        L'algorithme -- maximisation globale du critere, puis Kriging
        Believer pour en obtenir plusieurs sans appeler le solveur -- est
        dans `_reliability/eff_ot.py`. Ne restent ici que les reglages de
        l'etude et le reajustement du metamodele, qui lui appartient.
        """
        return _eff_ot.batch_kriging_believer(
            g_ot, sigma_func, xt, yt, all_grad,
            n_batch=n_batch_EFF, bornes_min=eff_bounds_min,
            bornes_max=eff_bounds_max, n_var=n_var, n_appels=n_NLopt_EFF,
            epsilon_factor=epsilon_factor, reajuster=init_g_ot,
            gradient_du_surrogate=do_GEPCK)

    def run_EFF(g_ot, sigma_func, xt, yt, all_grad):
        """Ameliore le metamodele jusqu'au critere d'arret, et le renvoie.

        Les 136 lignes de la boucle sont dans `_reliability/enrichissement.py`,
        en un seul exemplaire pour les deux etudes. Ne restent ici que les
        collaborateurs -- ce que CETTE etude sait faire et que la boucle
        ignore.

        `max_degree` est passe A L'APPEL : une reprise le relit dans le dump
        et ecrase celui du fichier d'etude, donc le figer dans l'objet
        rendrait la boucle sourde a la reprise.
        """
        return _enrichissement.BoucleEFF(
            CFG, n_var, journal=_JOURNAL,
            historiques={"EFF": _eff_history_EFF, "BB": _eff_history_BB,
                         "BS": _eff_history_BS, "Pf": _eff_history_Pf,
                         "beta_IS": _eff_history_beta_IS},
            points_EFF=_find_batch_EFF_points, fonction_EFF=EFFFunction,
            ajuster=init_g_ot, bornes_surrogate=BoundSurrogateFunction,
            executer_is=run_IS, evaluer_un_point=run_HF,
            executer_en_parallele=lambda SOL, nw: run_DOE_parallel(
                modelname, SOL, params_names, nw),
            dist_jointe=dist_jointe, params_names=params_names,
            figure=lambda g, s, x, xe: print_planche_EFF(
                g, s, x, xe, fond_hf=fond_hf_pour_figures()),
            sauver=lambda x, y, a, xe: _save_restart_state(
                x, y, a, xe, None, None, [], None),
        ).enrichir(g_ot, sigma_func, xt, yt, all_grad,
                   max_degree=max_degree, xt_eff_initial=_restart_xt_eff)

    # --------------------------------------------------------------------------- #
    # FONCTIONS RESULTATS/ AFFICHAGE                                              #

    # Le predicteur en lot du metamodele courant. Les trois champs d'une
    # coupe sont dans `_reliability/eff_ot.champs_sur_coupe`.
    _PREDICT = predict_pck if do_PCK else predict_gepck

    # --- Graphiques de suivi : _reliability/graphiques.py. Les courbes Pf
    # lineaire et log y sont une seule fonction, l'echelle etant un parametre.
    # --- HF GRID CACHE ---
    # Cadrage des figures. Il DIVERGEAIT entre les deux etudes -- la flexion
    # pure cadrait sur les bornes de la grille HF, le Moulin Blanc sur les
    # bornes de recherche elargies -- et l'ecart etait recopie dans quatre
    # fonctions de trace, sans qu'un seul commentaire ne le dise. C'est
    # maintenant un reglage du fichier d'etude, calcule ici, une fois.
    _CX0, _CX1, _CY0, _CY1 = _figurer.cadre_des_figures(
        CFG.cadre_figures, (u1_min, u1_max, u2_min, u2_max),
        eff_bounds_min, eff_bounds_max, CFG.cadre_marge)

    _HF_CACHE_FILE       = os.path.join(_path_ds, "hf_grid_cache.json")

    # Le decor commun a toutes les figures : cadre, resolution, noms, dossier.
    # Ces sept valeurs etaient capturees par fermeture dans CHAQUE fonction de
    # trace -- vingt a vingt-cinq variables libres par fonction.
    _DECOR = _figurer.Decor((_CX0, _CX1, _CY0, _CY1), n_grid, params_names,
                            modele, out_dir_eff, timestamp)

    _HF_CUSTOM_CACHE_FILE = os.path.join(_path_ds, "hf_custom_cache.json")
    _HF_CACHE_FILE_FINAL = os.path.join(_path_ds, "hf_grid_cache_final.json")


    _HF_FULL_CACHE_FILE = os.path.join(_path_ds, "hf_grid_full_cache.json")

    # La grille haute fidelite -- geometrie, cache signe, reprise apres
    # interruption -- est dans `_etapes/grille.py`. Ne reste ici que le
    # rangement du resultat dans l'etat de CETTE etude.
    _GRILLE = _grille.Grille(
        evaluer=lambda u: _obtenir_evaluateur().evaluer_g_en_U(u),
        n_var=n_var, cote=n_grid_hf,
        bornes=(u1_min, u1_max, u2_min, u2_max),
        fichier_cache=_HF_CACHE_FILE, fichier_cache_complet=_HF_FULL_CACHE_FILE,
        fichier_cache_points=_HF_CUSTOM_CACHE_FILE,
        coupe_initiale=hf_2d_grid_fixed,
        coupe_courante=slice_def, points_libres=hf_custom_points,
        active=print_HF,
        evaluer_lot=(lambda pts: run_HF_grid_parallel(pts, n_workers=n_workers_DOE))
                    if n_workers_DOE and n_workers_DOE > 1 else None,
        signature=CFG.signature_grille_hf(),
        config_identique=config_is_identical,
        marquer_phase=_JOURNAL.marquer)


    # --- HF GRILLE FULL (n_var-D) ---


    def fond_hf_pour_figures(sd=None, cache=None, finale=False):
        """Le fond de contour haute fidelite. Tout est dans `_etapes/grille.py`,
        avec ce qu'il coute et la garde qui evite de le payer deux fois."""
        return _GRILLE.fond_de_figure(sd, fichier=cache, finale=finale)

    def print_planche_EFF(g_ot, sigma_func, xt, xt_eff, fond_hf=None):
        """Planche 3 vues : critere EFF, ecart-type, etat limite du surrogate.

        Le DESSIN est dans `_etapes/figurer.py` ; ce qui reste ici, c'est
        l'evaluation du metamodele sur la coupe -- du calcul, pas du trace.

        `fond_hf` est recu deja calcule : cette fonction l'obtenait
        elle-meme, ce qui lui faisait declencher jusqu'a 225 appels au
        solveur sous un nom qui dit « imprime ».
        """
        _sd = slice_def if slice_def is not None else (0, 1, {})
        grille = _DECOR.grille_de_coupe(_sd)
        Z_eff, Z_sigma, Z_g = _eff_ot.champs_sur_coupe(
            g_ot, sigma_func, grille, n_grid, epsilon_factor, _PREDICT)

        _, _, _figees = _DECOR.etiquettes(_sd)
        return _figurer.planche_EFF(_DECOR, _sd, xt, xt_eff, Z_eff, Z_sigma, Z_g,
                                    fond_hf=fond_hf,
                                    sous_titre=_DIAG.sous_titre(_figees))


    def print_globalplanche_EFF(xt, yt, all_grad, xt_eff, fond_hf=None, fond_hf_final=None):
        """L'enrichissement etape par etape : une ligne par point ajoute.

        Le metamodele est REAJUSTE a chaque etape sur le plan tel qu'il etait
        alors -- c'est du calcul, il reste ici. Le dessin est dans
        `_etapes/figurer.py`.

        Ne coute AUCUN appel solveur : les points sont deja connus, seul le
        metamodele est refait.
        """
        if slice_def_final is None:
            print("[GLOBAL PLANCHE] slice_def_final est None, skip", flush=True)
            return
        grille = _DECOR.grille_de_coupe(slice_def_final)
        etapes = _fit.rejouer_l_enrichissement(
            xt, yt, all_grad, xt_eff, n0,
            reajuster=lambda x, y, g: init_g_ot(None, None, x, y, g)[:2],
            champs=lambda g_ot_k, sigma_k: _eff_ot.champs_sur_coupe(
                g_ot_k, sigma_k, grille, n_grid, epsilon_factor, _PREDICT))

        _, _, _figees = _DECOR.etiquettes(slice_def_final)
        return _figurer.planche_globale(_DECOR, slice_def_final, etapes,
                                        fond_hf=fond_hf_final,
                                        sous_titre=_figees.strip())


    def erreur_FOSM(best_result, g_ot):
        """L'ecart FOSM. 27 lignes sont dans `_reliability/controle.py`.

        COUT : 2 appels solveur au premier mode, 1 aux suivants (le gradient
        a l'origine est mis en cache). `CFG.erreur_fosm = false` la desactive.
        """
        if g_ot is None:
            return None
        _JOURNAL.marquer("USTAR")
        if _fosm[0] is None:
            _fosm[0] = _controle.ErreurFOSM(run_HF, params_names)
        return _fosm[0].mesurer(best_result)


    def print_visu(best_result, best_sps, xt, g_ot, modes, xt_eff, coupe,
                   fond_hf=None, fond_hf_final=None):
        """La figure de synthese : ou passe l'etat limite, ou FORM a cherche.

        Le DESSIN est dans `_etapes/figurer.py`. Ne reste ici que l'evaluation
        du metamodele sur la coupe -- du calcul.

        La coupe est RECUE, plus choisie ici : elle l'etait par effet de bord
        sur une globale, alors que les fonds de contour passes en argument
        sont evalues AVANT l'appel.
        """
        grille = _DECOR.grille_de_coupe(coupe)

        Z_sur = None
        if g_ot is not None and not do_HF:
            Z_sur = np.array(g_ot(ot.Sample(grille.tolist())))[:, 0].reshape(n_grid, n_grid)

        legende = []
        if g_ot is not None:
            legende.append(dict(color='blue', linestyle='-', linewidth=2,
                                label=f'g=0 {modele}'))
        if print_HF:
            legende.append(dict(color='red', linestyle='--', linewidth=2,
                                label='g=0 HF'))
        if print_ana:
            legende.append(dict(color='green', linestyle='-.', linewidth=2,
                                label='g=0 ana'))

        return _figurer.visu_FORM(
            _DECOR, coupe, Z_surrogate=Z_sur, fond_hf=fond_hf_final,
            xt=xt, xt_eff=xt_eff, points_de_depart=best_sps,
            modes=modes if modes else ([best_result] if best_result is not None else []),
            modes_figes=best_sol_modes_fixed, gradients_figes=grad_sp_fixed,
            trajectoires=traj_runs_fixed, surcouche=_surcouche_analytique if print_ana else None,
            legende=legende)


    def _surcouche_analytique(ax):
        """La solution analytique de reference, superposee.

        Elle n'existe que pour la flexion pure : c'est ce qui permet de dire
        si l'ecart au calcul a la rupture vient du metamodele ou du modele
        mecanique. Le Moulin Blanc n'a pas d'equivalent -- c'est la DERNIERE
        difference entre les deux etudes.
        """
        calc = flexion_claude()
        u1_lim = calc.u1_lim_plast
        u2_lim = calc.u2p_LS(u1_lim)
        u1_g = np.linspace(u1_lim, u1_max, n_grid)
        u2_g = np.array([calc.u2p_LS(u) for u in u1_g])
        ax.plot(u1_g, u2_g, color='green', linestyle='-.', linewidth=2)
        ax.plot([u1_lim, u1_lim], [u2_lim, u2_max], color='green',
                linestyle='-.', linewidth=2)
        ax.plot(u1_lim, u2_lim, 'ko', ms=6, zorder=6)


    def grille_3D():
        """ACTION `grille` : la surface g_HF, pour un trace en relief.

        COUT : n_grid_hf^2 appels solveur si `hf_3d_grid_fixed` est vide.
        Les 35 lignes sont dans `_etapes/grille.py`.
        """
        return _GRILLE.surface_3d(hf_3d_grid_fixed)


    def print_3D_HF(U1_hf, U2_hf, Z):
        """FIGURE : la surface deja calculee, en relief. ZERO appel solveur."""
        return _figurer.relief(_DECOR, U1_hf, U2_hf, Z, n_grid_hf,
                               modes_figes=best_sol_modes_fixed,
                               gradients_figes=grad_sp_fixed,
                               surcouche=_relief_analytique if print_ana else None)

    def _relief_analytique(ax, plancher):
        """La surface analytique de reference, superposee en relief.

        L'ecart entre les deux surfaces se lit alors en volume : la ou elles
        se separent, la formule de section cesse de representer le calcul
        tridimensionnel. N'existe que pour la flexion pure.
        """
        calc = flexion_claude()
        U1_a, U2_a = np.meshgrid(np.linspace(u1_min, u1_max, n_grid),
                                 np.linspace(u2_min, u2_max, n_grid))
        Z_ana = np.array([calc.g(a, b) for a, b in zip(U1_a.ravel(), U2_a.ravel())]
                         ).reshape(n_grid, n_grid)
        ax.plot_surface(U1_a, U2_a, Z_ana, color='blue', alpha=0.3, label='g_ana')
        ax.contour(U1_a, U2_a, Z_ana, levels=[0], colors='green', linewidths=2,
                   zdir='z', offset=plancher)
        ax.contour(U1_a, U2_a, Z_ana, levels=[0], colors='green', linewidths=2)


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
        # Les deux controles -- le dump existe, et il a ete produit sous la
        # configuration courante -- sont dans `_cache/reprise.py`, avec le
        # detail des deux defauts qu'ils ferment.
        _rs = _reprise.charger(_RESTART_STATE_FILE, CFG.signature_solveur())

        xt = np.array(_rs['xt'], float)
        yt = np.array(_rs['yt'], float)
        all_grad = np.array(_rs['all_grad'], float)
        _restart_xt_eff = [np.array(p, float) for p in _rs['xt_eff']]
        if _rs.get('max_degree') is not None:
            max_degree = int(_rs['max_degree'])
        _GRILLE.coupes['courante'] = _rs.get('hf_2d_grid')
        _h = _reprise.historiques_de(_rs)
        _eff_history_EFF, _eff_history_BB    = _h["EFF"], _h["BB"]
        _eff_history_BS, _DIAG.theta = _h["BS"], _h["theta"]
        _eff_history_beta_IS, _eff_history_Pf = _h["beta_IS"], _h["Pf"]
        _enrich_round     = int(_rs.get('enrich_round', 0)) + 1
        _round_sizes_prev = list(_rs.get('round_sizes', [int(len(xt))]))
        _JOURNAL.marquer_reprise(_enrich_round, len(xt), len(_restart_xt_eff))
        print(f"[RESTART] charge {len(xt)} pts (dont {len(_restart_xt_eff)} EFF) "
              f"depuis {_RESTART_STATE_FILE} (round {_enrich_round})", flush=True)

        # max_degree fixe (LARS gere P > N)
        event, g_ot, sigma_func = None, None, None
        xt_eff = list(_restart_xt_eff)   # survit sans enrichissement : test_111
    else:
        # --- Reset log incremental ---
        _JOURNAL.reinitialiser()

        # max_degree fixe (LARS gere P > N)
        event, g_ot, sigma_func, xt, yt, all_grad = [None] * 6
        xt_eff = None


    if print_3D:
        print_3D_HF(*grille_3D())
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

    # --- Le PLAN, puis le SURROGATE : deux actions, deux lignes.
    # Cette ligne etait cachee sept fois dans `init_g_ot`, une par branche de
    # surrogate. C'est le seul endroit du programme ou le plan initial doit
    # reellement etre construit.
    if xt is None:
        # UNE arite, plus deux. `build_DOE` rendait tantot un triplet, tantot
        # le seul `xt`, et la branche haute fidelite devait le savoir -- elle
        # ne l'a pas toujours su : « xt devenait un tuple, silencieusement ».
        # En HF pur, `yt` et `all_grad` sont deux `None`, pas une signature.
        xt, yt, all_grad = build_DOE()
        print(f"[PLAN] plan initial construit : {len(xt)} points", flush=True)
    g_ot, sigma_func, xt, yt, all_grad = init_g_ot(g_ot, sigma_func, xt, yt, all_grad)
    if print_HF and print_fullHF and n_var <= 3:
        _GRILLE.calculer_complete()
    if do_EFF:
        print_planche_EFF(g_ot, sigma_func, xt, [],
                          fond_hf=fond_hf_pour_figures())
        g_ot, sigma_func, xt, yt, all_grad, xt_eff = run_EFF(g_ot, sigma_func, xt, yt, all_grad)
        if not print_EFF_progres:
            print_planche_EFF(g_ot, sigma_func, xt, xt_eff,
                              fond_hf=fond_hf_pour_figures())
        _graphiques.tracer_convergence_eff(
            _eff_history_EFF, _eff_history_BB, _eff_history_BS,
            _DIAG.theta, params_names, tol_EFF, tol_BB, tol_BS,
            out_dir_eff, timestamp)
        if print_Pf:
            _graphiques.tracer_pf_evolution(
                _eff_history_Pf, modele, EFF_criteria, out_dir_eff,
                timestamp, 'lineaire')
            _graphiques.tracer_pf_evolution(
                _eff_history_Pf, modele, EFF_criteria, out_dir_eff,
                timestamp, 'log')
    event, g_ot, sigma_func, xt, yt, all_grad = init_FORM(g_ot, sigma_func, xt, yt, all_grad)

    if event is None:
        if best_sol_modes_fixed is not None:
            print_visu(None, [], None, None, [], None,
                       slice_def_final or slice_def)
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
            starting_points = _form.points_de_depart(xt, n_var, do_multistart)
        modes, best_sps = FORM_all_modes(starting_points, tol_all_modes, event)

    best_result = modes[0] if modes else None
    if best_result is None:
        print('Aucun FORM ne marche.', flush=True)
        sys.exit(1)
    if len(modes)>1:
        print('On a trouvé plus de 1 mode! Les résultats du mode 2 sont:')
        _JOURNAL.marquer("USTAR")
        _figurer.resume_FORM(modes[1], dist_jointe(), params_names)
        if CFG.erreur_fosm:
            erreur_FOSM(modes[1], g_ot)
        print('Les résultats du mode 1 sont : ')
    _JOURNAL.marquer("USTAR")
    _figurer.resume_FORM(best_result, dist_jointe(), params_names)
    if CFG.erreur_fosm:
        erreur_FOSM(best_result, g_ot)
    result_IS = None
    if do_IS and modes:
        result_IS = run_IS(modes, event)
        print_results_IS(result_IS)
    # --- IS sur surrogate projete (enveloppe position) ---
    g_proj = projection_surrogate(g_ot)
    if g_proj is not g_ot and do_IS and modes:
        idx_pos = _find_position_var_index()
        _idx_other = [_i for _i in range(n_var) if _i != idx_pos]
        # Le MEME evenement `g < 0`, en dimension reduite : il etait
        # reconstruit a la main ici, alors que `init_FORM` le construisait
        # deja quatre lignes plus haut dans le fichier.
        event_proj = _form.evenement_de_defaillance(g_proj, len(_idx_other))
        print("=== IS sur surrogate projete (enveloppe position) ===", flush=True)
        result_IS_proj = run_IS_proj(modes, event_proj)
        print_results_IS(result_IS_proj)
    # La coupe finale se decide AVANT les figures, pas pendant : leurs fonds
    # de contour sont evalues a l'appel. Le detail est dans `test_112`.
    if slice_def_final is None:
        slice_def_final = _form.coupe_la_plus_parlante(best_result, n_var, slice_def)
    print_visu(best_result, best_sps, xt, g_ot, modes, xt_eff,
               slice_def_final,
               fond_hf=fond_hf_pour_figures(slice_def, _HF_CACHE_FILE),
               fond_hf_final=fond_hf_pour_figures(
                   slice_def_final, _HF_CACHE_FILE_FINAL, finale=True))
    if do_EFF and xt_eff:
        print_globalplanche_EFF(
            xt, yt, all_grad, xt_eff,
            fond_hf=fond_hf_pour_figures(slice_def, _HF_CACHE_FILE),
            fond_hf_final=fond_hf_pour_figures(
                slice_def_final, _HF_CACHE_FILE_FINAL, finale=True))
    _save_restart_state(xt, yt, all_grad, xt_eff, best_result, best_sps[0] if best_sps else None, modes, result_IS)
