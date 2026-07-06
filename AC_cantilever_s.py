"""
CODE FIABILITE - VERSION AVEC DEFINITION DE FONCTIONS
"""
import os
import sys
import json
import shutil
import re
import time

# ============================================================================
# TIMING HELPERS (instrumentation detaillee pour identifier bottlenecks)
# ============================================================================
_T_START = time.perf_counter()

def _t_now():
    return time.perf_counter()

def _t_log(label, t0=None):
    """Log un timing. Si t0 fourni: affiche le delta. Sinon: marqueur de temps absolu."""
    total = time.perf_counter() - _T_START
    if t0 is not None:
        dt = time.perf_counter() - t0
        print(f"[TIMING t={total:7.1f}s dt={dt:7.2f}s] {label}", flush=True)
    else:
        print(f"[TIMING t={total:7.1f}s          ] {label}", flush=True)
    return time.perf_counter()
# ============================================================================

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
# Backend conditionnel : headless 'Agg' en mode batch/parallele (run via launcher ou _IS_PARALLEL),
# 'TkAgg' interactif sinon. Evite la race Tcl_AsyncDelete (Tk + pool multiprocessing au teardown)
# qui peut faire planter le ramp-up de l'IS parallele. Agg supporte savefig (PNG) sans Tk.
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
import sys; sys.path.insert(0, r"C:\workspace\fiabilite\_lib")  # branche* deplaces dans _lib
from branche1 import fit_gepck, predict_gepck, predict_gradient_gepck
import os
from _parallel_is import adaptive_is                              # IS adaptatif parallelisable (sonde + ramp-up), _lib deja dans sys.path
_IS_PARALLEL = os.environ.get("_IS_PARALLEL", "1") != "0"         # flag GLOBAL : IS parallele (sonde+ramp-up) ON par defaut ; _IS_PARALLEL=0 -> fallback OpenTURNS sequentiel
_IS_K        = int(os.environ.get("_IS_K", "16"))                 # nb de workers du ramp-up
_IS_CHUNK    = int(os.environ.get("_IS_CHUNK", "8"))              # nb de blocs par worker par ronde
_IS_PROBE    = int(os.environ.get("_IS_PROBE", "16"))            # nb de blocs de la sonde sequentielle


def _parse(text, name):
    return float(re.search(rf'(?m)^\s*{re.escape(name)}\s*=\s*([\d.]+)', text).group(1))

if __name__ == '__main__':
    # 2026-06-17 : fiabilite a 2 variables fy (1 par groupe d'acier), fc fixe.
    # Cas 1 (membrure inf dans tablier) : Calcul_fiabilite_13k_2fy_membrure_inf_tablier
    # Cas 2 (membrure inf dans diagonal) : Calcul_fiabilite_13k_2fy_membrure_inf_diagonal
    modelname = "Test_cantilever_s"   # cas test sensibilite position de charge (cantilever : fy_top + s)
    modelname = os.environ.get("_DOE_WORKER_MODELNAME") or modelname   # override sur la copie .ds en mode worker DOE (ligne ci-dessus = source pour le regex du launcher)
    _path_ds = "C:\\workspace\\storage\\admin\\Moulin_Blanc\\" + modelname + ".ds"
    with open(os.path.join(_path_ds, 'dsCad.txt'), 'r') as f:
        _cad_txt = f.read()

    print("=" * 70)
    print("CALCUL DE FIABILITE -- PONT DU MOULIN BLANC -- LM1 TRAFIC (PRESSURE)")
    print("=" * 70)
    # --------------------------------------------------------------------------- #
    # OPTIONS UTILISATEUR                                                         #
    # --------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------- #
    # DEFINITION DU MODELE                                                        #
    # 2026-06-11 : 1er run HF valide (pObj=3.92 converge). Bascule sur config complete
    # comme AC3_pure_flexion d'origine : GEPCK + EFF + IS + n0=5.
    modele = 'GEPCK'                 #options: 'GEPCK', 'PCKRG', 'KRG', 'GEK', 'HF'
    do_EFF = True                              #si on veut enrichir progressivement
    do_IS   = True                            #si on veut calculer la proba globale

    n0 = 8                      # nombre de points du plan d'experience initial (DOE) (aligné flexion : 5 -> 8)
    config_is_identical = False  # False = recalcule le DOE ; True = reutilise doe_cache.json (mettre True apres 1er run OK avec ces params)
    # 2026-06-18 : nb de calculs DOE lances EN PARALLELE (multiprocessing, copies .ds isolees).
    #   1 = sequentiel (comportement d'origine). >1 = N workers concurrents, MKL epingle a 32//N
    #   threads/worker. RAM ~22 Go/worker (machine 256 Go). Avec n0=5 : 3 -> 2 vagues, 5 -> 1 vague.
    n_workers_DOE = 1           # cantilever test : DOE sequentiel (evite le spawn du launcher 2fy)
    # 2026-06-17 : 2 variables fy (1 par groupe d'acier), fc fixe (dans le dsCad).
    # params_names / n_var sont derives de PARAM_CONFIG (defini plus bas, apres loi_* et group1/2_names).

    # --------------------------------------------------------------------------- #G
    # CARACTERISTIQUES DU MODELE                                                  #

    # --- Paramètres variables ---
    # Valeurs caractéristiques du pont du Moulin Blanc (cf dsCad : COMPRESSIVE_STRENGTH=20 MPa, GRADE=235 MPa)
    FY_MEAN = 550.0              # MPa : moyenne acier HA16 haut (comme flexion ; loi_fy Normale, SIGMA~30)


    # Cantilever test : aciers HAUT (GRADE=fyd_top) = variable fy_top ; aciers BAS (fyd_bot) = fixes.
    rebar_names = re.findall(r"REBAR\('([^']+)'", _cad_txt)
    n_rebars = len(rebar_names)
    top_names = re.findall(r"REBAR\('([^']+)',[^\n]*GRADE=fyd_top,", _cad_txt)
    print(f"[cantilever-s] aciers HAUT variables (fy_top) : {len(top_names)} -> {top_names}", flush=True)
    assert top_names, "dsCad doit contenir des REBAR avec GRADE=fyd_top"
    _top_sentinel = top_names[0]

    def _sens_key_to_param(k):
        """Mappe une cle de sensibilite STRAINS -> nom de params_names.
        UNIQUE mecanisme : correspondance EXACTE 'param:region_key'. General
        (tous types : YIELD_STRENGTH, LIVE_LOAD, DEAD_LOAD, LOAD_POSITION),
        robuste (aucun raisonnement par sous-chaine / nom d'entite). region_key
        OBLIGATOIRE pour chaque variable (verifie a la construction de PARAM_CONFIG)."""
        for p in params_names:
            sens = PARAM_CONFIG[p]['sens']
            if k == sens['param'] + ':' + sens['region_key']:
                return p
        return None


    # --------------------------------------------------------------------------- #
    # PARAMETRES FORM                                                             #
    # 2026-06-11 : config complete (mirror AC3_pure_flexion)
    n_max_FORM = 50
    do_multistart = True #multistart : FORM depuis n0 points + [0,0]
    do_warmstart = False #warmstart : si FORM ne converge pas, on repart du best pt

    tol_FORM = 0.002                 # précision acceptée par FORM pour l'état limite (aligné flexion : 1.0 -> 0.002)
    tol_all_modes = 0.9                           #distance DBSCAN entre deux modes (Semia flexion: 0.9)
    tol_warmstart = 0.2 # fixe la nécessité de faire le warm_start si do_warm_start

    # --------------------------------------------------------------------------- #
    # PARAMETRES IS                                                               #
    n_IS    = 10000                                       # taille échantillon IS
    cov_IS  = 0.05                                             # critère d'arrêt COV

    # --------------------------------------------------------------------------- #
    # PARAMETRES MESH                                                             #
    global_size     = 0.05   # global_physical_size (0.05 = rapide FORM, 0.007 = très fin)
    geo_min_approx  = 4      # geometric_approximation_min (4 = fin, 35 = grossier)

    # --------------------------------------------------------------------------- #
    # PARAMETRES GEK/ PCE/ EFF                                                    #
    # 1. GEK
    do_analytic_grad = False
    reduc_PLS = 0

    # 2. PCE
    seuil_pce = 0.90                              # seuil de validation de l'erreur
    q = 0.75                                              # tri base poly candidats
    max_degree = 0     # (init 0, varie en fonction de n0) degre max base candidats
    max_of_maxdegree = 2                                # (fixe) degre max autorisé

    # 3. EFF
    epsilon_factor = 2                               # eps = epsilon_factor * sigma
    tol_EFF = 1e-3                                            # 2026-06-17 : remis a 1e-3 (valeur flexion d'origine ; avait ete teste a 1e-5)
    tol_BB       = 0.05         # critere BB : |beta_IS_sup - beta_IS_inf| / beta_IS (aligné flexion : 0.01 -> 0.05)
    tol_BS       = 0.01         # critere BS : |beta_IS - beta_IS_prec| / beta_IS (Semia flexion: 0.005 -> 0.01)
    EFF_criteria = 'BS'           # critere d'arret EFF : 'BB' | 'BS' | 'both' | 'at_least_one' | 'n_points' (BS recommande, cf. guide)
                                  #   'n_points' = on s'arrete apres un NOMBRE FIXE de points EFF (n_eff_target)
                                  #   au lieu d'un critere de convergence (utile si le bootstrap ne converge pas).
    n_eff_target = 10             # nb de points EFF a ajouter quand EFF_criteria=='n_points' (run initial)
    EFF_criteria = os.environ.get("_EFF_CRITERIA") or EFF_criteria          # override env optionnel
    if os.environ.get("_N_EFF_TARGET"): n_eff_target = int(os.environ["_N_EFF_TARGET"])
    u1_eff_min, u1_eff_max = -7.5, 7.5    # Semia flexion: -10/10 -> -7.5/7.5 (zone realiste pour EFF)
    u2_eff_min, u2_eff_max = -7.5, 7.5    # NB: resserrer u_s (test 82ac916 -> +-3) CASSE FORM
                                          #   (critique au bord s=1 -> besoin u_s>3). Garder +-7.5.
    def _eff_bounds():
        # Bornes de recherche EFF PAR AXE (axe 0 = s -> u1_eff, axe 1 = fy -> u2_eff, sinon u2_eff).
        # Corrige l'ancien [u1_eff_min]*n_var qui appliquait les bornes de s a TOUS les axes (fy inclus)
        # -> empechait un box asymetrique u_s/u_fy.
        lo = [u1_eff_min if k == 0 else u2_eff_min for k in range(n_var)]
        hi = [u1_eff_max if k == 0 else u2_eff_max for k in range(n_var)]
        return ot.Interval(lo, hi)
    n_NLopt_EFF = 30      # 2026-06-17 : 200 -> 30 (avec 200 l'EFF traque sans fin des points inutiles sur la crete u1 inerte ; 30 limite cette poursuite)
    n_max_EFF_points = 30   # plafond DUR du nombre de points EFF ajoutes (securite anti-poursuite infinie)
    print_EFF_progres = True                  # PNG par iter EFF (comme Semia) - inactif si do_EFF=False

    # --------------------------------------------------------------------------- #
    # PARAMETRES ET OPTIONS DE PRINT                                              #
    
    # Paramètres de print ---
    u1_max = 7.5    # Semia flexion: -10/10 -> -7.5/7.5 (bornes visu coherentes avec EFF)
    u2_max = 7.5
    u1_min = -7.5
    u2_min = -7.5
    n_grid = 300
    n_grid_hf = 7
    def _ax_bounds_hf(k):
        # Bornes par axe pour la grille HF : axe 0 = s (u1), axe 1 = fy (u2), sinon u1.
        # Permet un box ASYMETRIQUE (u_s resserre, u_fy large) sans que fy herite des bornes de s.
        return (u1_min, u1_max) if k == 0 else ((u2_min, u2_max) if k == 1 else (u1_min, u1_max))

    # --- Options de print ---
    print_HF = True     # run complet : courbe rouge ON (grille HF 7x7 = 49 SOCP)
    print_fullHF = False  # transfert3 T3-5 : True = grille HF complete n_grid_hf^n_var (n_var<=3, ~30min en 3D) ; False = coupe 2D directe (identique en n_var=2)
    save_history = True  # True = sauve les fichiers SOCP dans SOCP_history/ (1 sous-dossier par appel) ; False = off
    print_Pf = False     # True = calcule Pf_IS mid/sup/inf a chaque iter EFF (3 FORM+IS) ; False = 1 seul FORM+IS (mu)
    print_DOE = True
    print_3D = False

    # --- Print facultatifs, par défaut à False ---
    print_ana = False   # PAS d'analytique pour Moulin Blanc (flexion_claude = uniquement flexion pure)
    print_grad_sp = False #option si on veut afficher les gradients des points de départ

    # --- MODE REPRISE / RE-ENRICHISSEMENT (2026-06-19) ---
    # True = ne fait QUE re-enrichir : lit restart_state.json (xt/yt/all_grad + degre + histos
    # + courbe rouge), refit le surrogate, continue l'enrichissement selon le critere EFF courant
    # (budget = n_max_EFF_points ; pour "+N relatif a la reprise" : EFF_criteria='n_points' + n_eff_target),
    # puis re-ecrit le MEME dump (round incremente) + append au point-log. Chainable plusieurs fois.
    # Skip DOE et courbe rouge (reutilises du dump) -> pas de recalcul SOCP inutile.
    restart_enrich_only = False
    # override par env : _RESTART_ENRICH=1
    restart_enrich_only = (os.environ.get("_RESTART_ENRICH") == "1") or restart_enrich_only


    

    # --- Résultats fixés ---
    # 2026-06-17 : caches HF mis a None pour 2-fy (les anciennes grilles etaient fc x fy,
    # invalides pour fy1 x fy2). Recalcul automatique si besoin (print_HF / print_3D).
    hf_3d_grid_fixed = None   # utilise seulement par print_3D_HF (print_3D=False)
    hf_2d_grid_fixed = None   # utilise par la courbe rouge HF (print_HF) -> recalcul fy1 x fy2


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
    

    # --- Label PCE GEPCK (mis a jour par init_g_ot, lu par print_planche_EFF) ---
    _gepck_pce_label = ""

    # --- Historiques EFF (mis a jour par run_EFF et init_g_ot, lus par print_EFF_graphs) ---
    _eff_history_EFF   = []   # EFF(u_opt) avant ajout de chaque point (incl. initial)
    _eff_history_BB    = []   # ratio BB par iteration (None si FORM echoue)
    _eff_history_BS    = []   # ratio BS par iteration (None si calcul impossible)
    _eff_history_theta = []   # theta Kriging [theta_0,...,theta_{M-1}] apres chaque fit
    _eff_history_beta_IS = [] # historique beta_IS (snapshot de list_beta_IS en fin de run_EFF, pour le dump restart)
    _point_log_phase = ["?"]  # phase courante pour le log incremental par point (pose par chaque etape : HF/EFF/USTAR ; DOE logue a part)
    _point_log_round = [0]    # round de re-enrichissement (0 = run initial), tague chaque ligne du point-log
    _enrich_round     = 0     # round courant (0 = run initial, k = k-ieme re-enrichissement)
    _round_sizes_prev = []    # round_sizes charges du dump (vide au run initial)
    _restart_xt_eff   = []    # xt_eff charge du dump (pour reseeder les triangles EFF du plot en reprise)
    _fosm_u0_cache    = [None] # cache du run_HF([0,0]) (FOSM) : calcule 1x, reutilise pour tous les modes (evite 1 SOCP redondant/run)

    # --- Sortie PNG EFF ---
    timestamp   = datetime.now().strftime('%d%m_%H%M')
    out_dir_eff = os.path.join(r'C:\workspace\fiabilite\output\png_EFF_moulin_blanc', f'png_EFF_{timestamp}')
    os.makedirs(out_dir_eff, exist_ok=True)

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

    # --- DSCAD ET DSLOAD ---
    def patch_params(path, **params):
        """Reecrit dsCad.txt ET dsLoad.txt avec de nouvelles valeurs de parametres.
        Materiaux (fy*, fc) -> matchent dans dsCad ; charges -> matchent dans dsLoad.
        Pour 2-fy (que des aciers) le regex ne matche rien dans dsLoad -> reecrit identique (no-op)."""
        for filename in ('dsCad.txt', 'dsLoad.txt'):
            fpath = os.path.join(path, filename)
            with open(fpath, 'r') as f:
                content = f.read()
            for name, value in params.items():   # on remplace variable par variable
                content = re.sub(r'^' + name + r'\s*=.*$', f'{name}    = {value:.10f}', content, count=1, flags=re.MULTILINE)
            with open(fpath, 'w') as f:
                f.write(content)

    # --- DISTRIBUTIONS ---
    
    SIGMA_11, SIGMA_12, SIGMA_13 =  19.0, 22.0, 8.0 
    SIGMA = np.sqrt(SIGMA_11**2 + SIGMA_12**2 + SIGMA_13**2)  # ~30 MPa
    
    def loi_fy(fym, cov=None):
        if cov is not None:
            sig_ec = cov * fym
        else:
            sig_ec = SIGMA

        dist = ot.Normal(fym, sig_ec)
        return dist
    
    def loi_fc(fcm, cov=None):
        COV_TABLE = {"C15": 0.14, "C25": 0.12, "C35": 0.09, "C45": 0.07}
        fck_eq = fcm - 8.0
        classe = min(COV_TABLE, key=lambda c: abs(int(c[1:]) - fck_eq))
        v = cov if cov is not None else COV_TABLE[classe]

        sigma_ln = np.sqrt(np.log(1 + v**2))
        mu_ln    = np.log(fcm) - 0.5 * sigma_ln**2

        dist = ot.LogNormal(mu_ln, sigma_ln, 0.0)
        return dist

    # --- Loi de POSITION (charge mobile s) : uniforme sur [S_MIN, S_MAX] ---
    # 2026-06-30 : s est l'abscisse curviligne NORMALISEE in [0,1] le long du PATH
    # (archi shift/PATH dans dsCad). La longueur reelle du chemin (L-load_len=3.4 m)
    # est portee par le PATH ; l'AC reste generique avec Uniform(0,1). Le solveur
    # renvoie dAlpha/ds p/r a ce s normalise (tangent = vecteur complet D).
    # loi_position ignore (mean,cov) -> Uniforme fixe.
    S_MIN, S_MAX = 0.0, 1.0
    def loi_position(*_a):
        return ot.Uniform(float(S_MIN), float(S_MAX))

    # transfert4 T4-5 (2026-07-06, guide Semia du 02/07) : loi uniforme approchee (fenetre de
    # Tukey normalisee) pour les variables de POSITION. Copie conforme de AC3_pure_flexion.py
    # L433-519. alpha=0 -> uniforme exacte, alpha=1 -> Hann. NOTE : 's' reste en Uniform(0,1)
    # (loi_position ci-dessus) tant que la bascule Tukey n'est pas decidee.
    def loi_uni_approx(a, b, alpha=0.5):
        """Loi uniforme approchee (fenetre de Tukey normalisee).
        Support [a, b]. alpha=0 -> uniforme exacte, alpha=1 -> Hann."""

        class TukeyDistribution(ot.PythonDistribution):
            def __init__(self, a, b, alpha):
                super().__init__(1)
                self.a = float(a)
                self.b = float(b)
                self.alpha = float(alpha)
                self.L = self.b - self.a
                self.C = 1.0 - self.alpha / 2.0  # integrale de w sur [0,1]

            def getRange(self):
                return ot.Interval([self.a], [self.b])

            def computePDF(self, X):
                x = X[0]
                if x < self.a or x > self.b:
                    return 0.0
                t = (x - self.a) / self.L          # t in [0, 1]
                al = self.alpha
                if al <= 0.0:
                    w = 1.0
                elif t < al / 2.0:
                    w = 0.5 * (1.0 + np.cos(2.0 * np.pi / al * (t - al / 2.0)))
                elif t > 1.0 - al / 2.0:
                    w = 0.5 * (1.0 + np.cos(2.0 * np.pi / al * (t - 1.0 + al / 2.0)))
                else:
                    w = 1.0
                return w / (self.L * self.C)

            def computeCDF(self, X):
                x = X[0]
                if x <= self.a:
                    return 0.0
                if x >= self.b:
                    return 1.0
                t = (x - self.a) / self.L
                al = self.alpha
                if al <= 0.0:
                    return t
                if t <= al / 2.0:
                    F = t / 2.0 + al / (4.0 * np.pi) * np.sin(2.0 * np.pi / al * (t - al / 2.0))
                elif t <= 1.0 - al / 2.0:
                    F = t - al / 4.0
                else:
                    F = (1.0 - 3.0 * al / 4.0
                         + (t - 1.0 + al / 2.0) / 2.0
                         + al / (4.0 * np.pi) * np.sin(2.0 * np.pi / al * (t - 1.0 + al / 2.0)))
                return F / self.C

            def getMean(self):
                return [(self.a + self.b) / 2.0]

            def computeScalarQuantile(self, p, tail=False):
                if tail:
                    p = 1.0 - p
                al = self.alpha
                if al <= 0.0:
                    return self.a + p * self.L
                F_left  = (al / 4.0) / self.C
                F_right = (1.0 - 3.0 * al / 4.0) / self.C
                if p <= F_left:
                    t = al / 4.0
                    for _ in range(20):
                        F = t / 2.0 + al / (4.0 * np.pi) * np.sin(2.0 * np.pi / al * (t - al / 2.0))
                        f = 0.5 * (1.0 + np.cos(2.0 * np.pi / al * (t - al / 2.0)))
                        t -= (F / self.C - p) / (f / self.C)
                        t = max(0.0, min(t, al / 2.0))
                elif p <= F_right:
                    t = p * self.C + al / 4.0
                else:
                    t = 1.0 - al / 4.0
                    for _ in range(20):
                        F = (1.0 - 3.0 * al / 4.0
                             + (t - 1.0 + al / 2.0) / 2.0
                             + al / (4.0 * np.pi) * np.sin(2.0 * np.pi / al * (t - 1.0 + al / 2.0)))
                        f = 0.5 * (1.0 + np.cos(2.0 * np.pi / al * (t - 1.0 + al / 2.0)))
                        t -= (F / self.C - p) / (f / self.C)
                        t = max(1.0 - al / 2.0, min(t, 1.0))
                return self.a + t * self.L

            def isContinuous(self):
                return True

        return ot.Distribution(TukeyDistribution(a, b, alpha))

    # --- CONFIG DES VARIABLES ALEATOIRES (dicts) : tout en derive (lois, patch, sensibilites) ---
    # transfert4 T4-1 (2026-07-06) : 'mean'/'cov' -> 'args' (tuple passe a la loi ;
    # supporte des lois a signatures differentes, ex. loi_uni_approx(a, b, alpha)).
    PARAM_CONFIG_CAD = {
        'fy_top': {'sens': {"param": "YIELD_STRENGTH", "rebars": top_names, "region_key": "fy_top"},
                   'loi': loi_fy, 'args': (FY_MEAN, None)},
    }
    PARAM_CONFIG_LOAD = {
        's': {'sens': {"param": "LOAD_POSITION", "region_key": "s"},   # region_key -> cle courte "LOAD_POSITION:s"
              'loi': loi_position, 'args': ()},   # loi_position ignore ses args (Uniform fixe)
    }
    PARAM_CONFIG = {**PARAM_CONFIG_LOAD, **PARAM_CONFIG_CAD}
    params_names = list(PARAM_CONFIG_LOAD.keys()) + list(PARAM_CONFIG_CAD.keys())
    n_var = len(params_names)
    # region_key OBLIGATOIRE + UNIQUE (matching exact _sens_key_to_param ; echoue clair si oubli).
    _rk = [PARAM_CONFIG[p]['sens'].get('region_key') for p in params_names]
    assert all(_rk), f"region_key manquant dans PARAM_CONFIG : {[p for p, r in zip(params_names, _rk) if not r]}"
    assert len(set(_rk)) == len(_rk), f"region_key dupliques : {_rk}"
    # transfert3 T3-1 (28481bd guide): coupe 2D par defaut pour les planches EFF (2 premieres vars, reste a 0).
    # slice_def_final=None -> print_visu la calcule depuis les importances FORM. Pour n_var=2 : (0,1,{}).
    slice_def = (0, 1, {i: 0.0 for i in range(n_var) if i > 1})
    slice_def_final = None
    # garde : pas d'analytique si une variable n'est pas un materiau CAD (ex. variable de charge)
    if not set(params_names) <= set(PARAM_CONFIG_CAD.keys()):
        print_ana = False

    def dist_jointe():
        """Loi jointe des variables aleatoires, construite depuis PARAM_CONFIG."""
        return ot.JointDistribution([PARAM_CONFIG[p]['loi'](*PARAM_CONFIG[p]['args'])
                                     for p in params_names])

    # ===================== VALIDATION CONFIG 2-FY (logs detailles) =====================
    print("\n" + "=" * 78, flush=True)
    print("VALIDATION CONFIG 2-FY  (modele : " + modelname + ")", flush=True)
    print("=" * 78, flush=True)
    print(f"  params_names = {params_names}   (n_var = {n_var})", flush=True)
    _distX_chk = dist_jointe()
    for _i, _p in enumerate(params_names):
        _mg = _distX_chk.getMarginal(_i)
        print(f"  loi {_p} : {_mg.getClassName()}  mean={float(_mg.getMean()[0]):.3f} MPa  "
              f"std={float(_mg.getStandardDeviation()[0]):.4f} MPa", flush=True)
    print(f"  SIGMA acier = sqrt(19^2+22^2+8^2) = {SIGMA:.4f} MPa (JCSS)", flush=True)
    print(f"  fc : FIXE (COMPRESSIVE_STRENGTH='20.0' dans dsCad, plus une variable de fiabilite)", flush=True)
    print(f"  aciers HAUT variables (fy_top) : {len(top_names)} aciers {top_names}", flush=True)
    print(f"  position s : Uniforme [{S_MIN}, {S_MAX}] NORMALISE [0,1] le long du PATH (x reel 0..3.4 m)  |  sentinelle YIELD : '{_top_sentinel}'", flush=True)
    try:
        _dsl = open(os.path.join(_path_ds, 'dsLoad.txt'), encoding='utf-8', errors='replace').read()
        _dead_vide = 'DEAD_LOAD_CASES=[]' in _dsl
        _live_2 = "LIVE_LOAD_CASES=[('LC_poids', '1'), ('LC_LM1_trafic', '1')]" in _dsl
        print(f"  CHARGEMENT : DEAD vide={_dead_vide}  |  LIVE=poids+trafic amplifies={_live_2}", flush=True)
    except Exception as _e:
        print(f"  CHARGEMENT : verif dsLoad echouee ({_e})", flush=True)
    print(f"  print_HF = {print_HF}  (courbe rouge HF {'ON' if print_HF else 'OFF -> pas de 49 SOCP'})", flush=True)
    print("=" * 78 + "\n", flush=True)
    # ===================================================================================

    # --- APPELS STRAINS ---
    # Counter pour identifier chaque appel SOCP (run_one_SOL + run_HF)
    _socp_call_counter = [0]  # liste pour eviter scope issues

    def _save_socp_outputs(path, AnalysisName, prefix_tag, u_vals=None, p_vals=None):
        """Copie les fichiers de sortie SOCP dans un sous-dossier dedie (eviter ecrasement).

        prefix_tag : ex `SOL_001` ou `HF_006`
        u_vals/p_vals : listes de coords (u espace standard / valeurs physiques params_names) pour le nom du sous-dossier

        Fichiers sauves : PL_cin_out.msh, kine.dsmed, kine.dslog, kine.dsmetares, stat.dsmed.
        """
        import shutil
        import os
        files_to_save = [
            f"{AnalysisName}_0_PL_cin_out.msh",
            f"{AnalysisName}_0_kine.dsmed",
            f"{AnalysisName}_0_kine.dslog",
            f"{AnalysisName}_0_kine.dsmetares",
            f"{AnalysisName}_0_stat.dsmed",
        ]
        # Format du suffix : coords u (espace standard) + valeurs physiques (params_names), generique
        coords_str = ""
        if u_vals is not None:
            coords_str += "".join(f"_u{i+1}{u_vals[i]:+.3f}" for i in range(len(u_vals)))
        if p_vals is not None:
            coords_str += "".join(f"_{params_names[i]}{p_vals[i]:.1f}" for i in range(len(p_vals)))
        # En mode worker DOE : SOCP_history TOUJOURS a la racine du projet principal (pas dans
        # les copies _doe_workers) -> un seul SOCP_history qui contient tout. _DOE_MAIN_DS absent
        # en run normal -> save_dir = path (comportement d'origine).
        _socp_root = os.environ.get("_DOE_MAIN_DS") or path
        sub_dir = os.path.join(_socp_root, "SOCP_history", f"{prefix_tag}{coords_str}")
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

    def run_one_SOL(modelname, SOL, params_names, sensitivity=False, with_sens_dict=None):
        """Lance un calcul complet pour une valeur de FT donnee.
        Retourne la liste des solutions pour chaque jeu de variables dans SOL (liste de dictionnaire)"""
        path = "C:\\workspace\\storage\\admin\\Moulin_Blanc\\" + modelname + ".ds"
        AnalysisName = 'Yield_analysis0'
        iteration = 0
        #MODIF 1 10/04 - on doit tout mettre dans params in SOL. TOUT.
        for i in range (len(SOL)):
            _t_iter = _t_log(f"=== run_one_SOL iter {i+1}/{len(SOL)} START params={SOL[i]} ===")
            _t0 = time.perf_counter()
            patch_params(path, **SOL[i]) #à cette étape SOL ne contient que 'fc': ,'fy':
            _t_log(f"  patch_params (dsCad.txt write)", _t0)

            _t0 = time.perf_counter()
            model = MODEL() #ici model n'est pas encore rempli
            SET_CONTEXT(model, path)
            fileName = os.path.join(path, AnalysisName + ".dscad") #on crée le chemin du fichier disque .dscad lisible par C. C va tout faire et on renverra les info plus tard (.load)
            _t_log(f"  MODEL() + SET_CONTEXT", _t0)

            _t0 = time.perf_counter()
            cadfile = open(path + '\\dsCad.txt', 'r')
            cadscript = cadfile.read() #on met dans cadscript les info de dsCad.txt
            exec(cadscript, globals()) # ici on modifie le modèle (C, cython) et donc les variables (on exécute le script de dsCad.txt ce qui modifie les variables - rien dans .dscad, tout dans var. en mémoire)
            _t_log(f"  exec(dsCad.txt) - lit dsCad + remplit model OCC", _t0)

            _t0 = time.perf_counter()
            model.Save(fileName) # ici on créé dscad et on enregistre les modifs des variables dans .dscad
            _t_log(f"  model.Save(.dscad) - serialise OCC binaire", _t0)
            print(model.GETERRORS()) # est vide si pas de message d'erreur sur le logiciel

            _t0 = time.perf_counter()
            loadfile = open(path + '\\dsLoad.txt', 'r')
            # OPTIM 2026-06-12 : SKIP model.Load(fileName) (~274s redondant sur dscad 144MB)
            # Le model est deja construit en memoire par exec(cadscript) ci-dessus avec
            # les NEW fc/fy de cette iteration. Le Save() a ecrit le .dscad sur disque
            # pour les autres outils. Re-Load() serait juste deserialiser ce qu'on
            # vient de serialiser -> redondant. LOAD_MODEL accepte le model in-memory.
            loadscript = loadfile.read()
            with CetLOAD.LOAD_MODEL(model, path):
                exec(loadscript, globals()) # pareil, on execute dsLoad et on enregistre dans var. mémoire
            _t_log(f"  exec(dsLoad.txt) - lit dsLoad + LOAD_MODEL (model deja en memoire, skip Load 274s)", _t0)

            Meshkwargs = { #définit la mesh - pas à comprendre ici car ne sera pas modifié. 
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
            }
            # OPTIM 2026-05-29 (be9c7e485+630d96ccf) : skip ~270s relecture .dscad par CetMESH
            Meshkwargs["model_handle"] = model.GETHANDLEPTR()
            _t0 = time.perf_counter()
            CetMESH.ANISO_MESH(AnalysisName, iteration, path, **Meshkwargs)
            _t_log(f"  CetMESH.ANISO_MESH (avec model_handle: skip ~270s relecture)", _t0)

            kwargs = {"scaling": 1, "write_debug_files": "true"} # ci-dessous on définit dict kwargs en entrée de SOLV.
            exec(open(r"C:\workspace\fiabilite\InitSolver.py").read(), globals())
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
                )   # generique : JSON identique au 2-fy (fy1->groupe1, fy2->groupe2)

            # OPTIM 2026-05-29 (630d96ccf) : skip ~240s relecture .dscad par CetSOLV
            kwargs["model_handle"] = model.GETHANDLEPTR()
            _t0 = time.perf_counter()
            CetSOLV.SOLV(AnalysisName, iteration, path, **kwargs) #On relance le solveur avec le nouveau dsCad.
            _t_log(f"  CetSOLV.SOLV (avec model_handle: skip ~240s relecture)", _t0)

            # Lire le resultat
            _t0 = time.perf_counter()
            metares_path = os.path.join(path, AnalysisName + "_0_kine.dsmetares") #on extrait l'addresse du fichier pour définir f
            with open(metares_path, 'r') as f: #f est le fichier créé par open, et on a with donc enter de fichier = donne accès au fichier (accès via f, toujours mettre as f) puis exit : ferme le fichier (qui reste lié à f)
                d = json.load(f) #chargement du fichier .dsmetares
            SOL[i]['g']=d['info']['Primal_bound'][0] -1
            # 2026-06-16 : sauvegarde des fichiers SOCP avec prefix avant ecrasement par next iter
            _socp_call_counter[0] += 1
            if save_history:
                _save_socp_outputs(path, AnalysisName,
                                   prefix_tag=f"SOL_{_socp_call_counter[0]:03d}",
                                   p_vals=[float(SOL[i][p]) for p in params_names])
            for p in params_names:
                SOL[i][f'dg_{p}'] = None
            if sensitivity and 'Sensitivity' in d['info']:
                print(f"les sensibilités calculées (cles) : {list(d['info']['Sensitivity'].keys())[:2]}...", flush=True)
                for k, v in d['info']['Sensitivity'].items():
                    p = _sens_key_to_param(k)   # 'fy1' / 'fy2' / None
                    if p is not None:
                        SOL[i][f'dg_{p}'] = v
                    if all(SOL[i].get(f'dg_{q}') is not None for q in params_names):
                        break
                _dgs = "  ".join(f"dg/d{q}={SOL[i].get(f'dg_{q}')}" for q in params_names)
                print(f"  [SENSIBILITES 2-fy] pObj={SOL[i]['g']+1:.5f}  g={SOL[i]['g']:+.5f}  |  {_dgs}", flush=True)
            _t_log(f"  read .dsmetares + sensibilites (g={SOL[i]['g']:.4f})", _t0)
            _t_log(f"=== run_one_SOL iter {i+1}/{len(SOL)} END (g={SOL[i]['g']:.4f}) ===", _t_iter)
        return SOL

    def run_HF(u):
        _t_hf = _t_log(f"=== run_HF START u={list(np.round(np.array(u),4))} ===")
        sensitivity = True
        n_var = len(u)
        dist_X = dist_jointe()
        T_inv = dist_X.getInverseIsoProbabilisticTransformation()
        u_point = ot.Point(u)
        x_point = T_inv(u_point)
        path = "C:\\workspace\\storage\\admin\\Moulin_Blanc\\" + modelname + ".ds"
        AnalysisName = 'Yield_analysis0'
        iteration = 0
        params={params_names[i]: x_point[i] for i in range(n_var)}
        _t0 = time.perf_counter()
        patch_params(path, **params) #à cette étape SOL ne contient que 'fc': ,'fy':
        _t_log(f"  patch_params (dsCad.txt write) params={params}", _t0)

        _t0 = time.perf_counter()
        model = MODEL() #ici model n'est pas encore rempli
        SET_CONTEXT(model, path)
        fileName = os.path.join(path, AnalysisName + ".dscad") #on crée le chemin du fichier disque .dscad lisible par C. C va tout faire et on renverra les info plus tard (.load)
        _t_log(f"  MODEL() + SET_CONTEXT", _t0)

        _t0 = time.perf_counter()
        cadfile = open(path + '\\dsCad.txt', 'r')
        cadscript = cadfile.read() #on met dans cadscript les info de dsCad.txt
        exec(cadscript, globals()) # ici on modifie le modèle (C, cython) et donc les variables (on exécute le script de dsCad.txt ce qui modifie les variables - rien dans .dscad, tout dans var. en mémoire)
        _t_log(f"  exec(dsCad.txt) - lit dsCad + remplit model OCC", _t0)

        _t0 = time.perf_counter()
        model.Save(fileName) # ici on créé dscad et on enregistre les modifs des variables dans .dscad
        _t_log(f"  model.Save(.dscad) - serialise OCC binaire", _t0)
        print(model.GETERRORS()) # est vide si pas de message d'erreur sur le logiciel

        _t0 = time.perf_counter()
        loadfile = open(path + '\\dsLoad.txt', 'r')
        # OPTIM 2026-06-12 : SKIP model.Load(fileName) (~274s redondant sur dscad 144MB)
        # idem run_one_SOL : model deja en memoire apres exec(cadscript)
        loadscript = loadfile.read()
        with CetLOAD.LOAD_MODEL(model, path):
            exec(loadscript, globals()) # pareil, on execute dsLoad et on enregistre dans var. mémoire
        _t_log(f"  exec(dsLoad.txt) - lit dsLoad + LOAD_MODEL (model deja en memoire, skip Load 274s)", _t0)

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
        # OPTIM 2026-05-29 (630d96ccf) : skip ~270s relecture .dscad par CetMESH
        Meshkwargs["model_handle"] = model.GETHANDLEPTR()
        _t0 = time.perf_counter()
        CetMESH.ANISO_MESH(AnalysisName, iteration, path, **Meshkwargs)
        _t_log(f"  CetMESH.ANISO_MESH (avec model_handle: skip ~270s relecture)", _t0)

        kwargs = {"scaling": 1, "write_debug_files": "true"} # ci-dessous on définit dict kwargs en entrée de SOLV.
        exec(open(r"C:\workspace\fiabilite\InitSolver.py").read(), globals())
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
            )   # generique : JSON identique au 2-fy (fy1->groupe1, fy2->groupe2)

        # OPTIM 2026-05-29 (630d96ccf) : skip ~240s relecture .dscad par CetSOLV
        kwargs["model_handle"] = model.GETHANDLEPTR()
        _t0 = time.perf_counter()
        CetSOLV.SOLV(AnalysisName, iteration, path, **kwargs) #On relance le solveur avec le nouveau dsCad.
        _t_log(f"  CetSOLV.SOLV (avec model_handle: skip ~240s relecture)", _t0)

        # Lire le resultat
        _t0 = time.perf_counter()
        metares_path = os.path.join(path, AnalysisName + "_0_kine.dsmetares") #on extrait l'addresse du fichier pour définir f
        with open(metares_path, 'r') as f: #f est le fichier créé par open, et on a with donc enter de fichier = donne accès au fichier (accès via f, toujours mettre as f) puis exit : ferme le fichier (qui reste lié à f)
            d = json.load(f) #chargement du fichier .dsmetares
        g_HF=d['info']['Primal_bound'][0] -1
        # 2026-06-16 : sauvegarde des fichiers SOCP avec prefix avant ecrasement par next call
        _socp_call_counter[0] += 1
        if save_history:
            _save_socp_outputs(path, AnalysisName,
                               prefix_tag=f"HF_{_socp_call_counter[0]:03d}",
                               u_vals=[float(v) for v in u],
                               p_vals=[float(x_point[i]) for i in range(n_var)])
        grad_HF_X=[None]*n_var
        grad_HF_U=[None]*n_var
        if sensitivity and 'Sensitivity' in d['info']:
            print(f"les sensibilités calculées (cles) : {list(d['info']['Sensitivity'].keys())[:2]}...", flush=True)
            for k, v in d['info']['Sensitivity'].items():
                p = _sens_key_to_param(k)   # 'fy1' / 'fy2' / None
                if p is not None:
                    grad_HF_X[params_names.index(p)] = v
                if all(grad_HF_X[i] is not None for i in range(n_var)):
                    break
            J_Tinv = T_inv.gradient(u)
            J_Tinv_T = J_Tinv.transpose()
            grad_HF_U = J_Tinv_T * ot.Point(grad_HF_X)
        if sensitivity and any(v is None for v in grad_HF_U):
            raise ValueError(f"run_HF : sensibilité demandée mais grad_HF_U contient None — vérifier que STRAINS a bien calculé les sensibilités. grad_HF_X={grad_HF_X}")
        _t_log(f"  read .dsmetares + sensibilites (g_HF={g_HF:.4f})", _t0)
        _append_point_log(_point_log_phase[0], u, x_point, g_HF)   # log incremental (phase HF/EFF/USTAR posee par l'appelant)
        _t_log(f"=== run_HF END (g_HF={g_HF:.4f}) ===", _t_hf)
        return g_HF, grad_HF_U, grad_HF_X

    # --- DOE parallele (multiprocessing) ---
    def run_DOE_parallel(base_modelname, SOL, params_names, n_workers):
        """Parallelise les SOCP du DOE (independants) via multiprocessing.
        Chaque worker = le MEME AC relance (launcher) en mode _DOE_WORKER sur une COPIE
        isolee du .ds (dsCad.txt + dsLoad.txt copies ; STP partage en lecture via chemin
        absolu dans le dsCad). MKL epingle a 32//n_workers threads/worker (anti sur-souscription).
        Reutilise run_one_SOL tel quel cote worker -> zero duplication de logique STRAINS."""
        import subprocess as _sp
        storage = "C:\\workspace\\storage\\admin\\Moulin_Blanc\\"
        base_ds = storage + base_modelname + ".ds"
        npts = len(SOL)
        n_workers = max(1, min(n_workers, npts))
        threads_per = max(1, 32 // n_workers)
        batches = [[] for _ in range(n_workers)]
        for i in range(npts):
            batches[i % n_workers].append(i)   # round-robin
        _t_log(f"  [DOE PARALLELE] {npts} pts -> {n_workers} workers (MKL={threads_per} threads/worker)")
        procs = []
        for w, idxs in enumerate(batches):
            if not idxs:
                continue
            # Workers ranges DANS le .ds du projet (sous-dossier _doe_workers), pas disperses
            # dans Moulin_Blanc. run_one_SOL fait path = Moulin_Blanc\<modelname>.ds, donc le
            # modelname porte le sous-chemin relatif -> wds = <base>.ds\_doe_workers\doewN.ds
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
                       _DOE_MAIN_DS=base_ds,                       # SOCP_history -> racine du projet principal
                       _FIAB_LOG_REDIRECTED="1",
                       MKL_NUM_THREADS=str(threads_per), OMP_NUM_THREADS=str(threads_per))
            wlog = open(wds + "\\_doe_worker.log", "w")
            print(f"    -> worker {w}: points {idxs}", flush=True)
            # cwd = la copie .ds du worker : isole le fichier temporaire $$$.dscad que STRAINS
            # (CetCAD.RELOAD_MODEL) ecrit dans le repertoire courant -> sinon collision entre workers.
            p = _sp.Popen([sys.executable, r"C:\workspace\fiabilite\launcher_cantilever_s.py"],
                          env=env, stdout=wlog, stderr=_sp.STDOUT, cwd=wds)
            procs.append((p, out_file, wlog, w, idxs))
        for p, out_file, wlog, w, idxs in procs:
            rc = p.wait(); wlog.close()
            _t_log(f"    <- worker {w} fini (rc={rc})")
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

    # --- DOE ---
    # --- Cache DOE (sidecar JSON, comme le cache HF) : lu s'il existe, sinon calcule + sauve ---
    _DOE_CACHE_FILE = os.path.join(_path_ds, "doe_cache.json")
    def _doe_cache_sig():
        return {"n0": n0, "params": list(params_names), "n_var": n_var, "modelname": modelname}
    def _load_doe_cache():
        if not config_is_identical:
            print("[DOE CACHE] config_is_identical=False -> recalcul DOE", flush=True)
            return None
        if not os.path.exists(_DOE_CACHE_FILE):
            print(f"[DOE CACHE] aucun cache ({_DOE_CACHE_FILE}) -> calcul DOE", flush=True)
            return None
        try:
            d = json.load(open(_DOE_CACHE_FILE))
            _n0_cache = d.get('n0', len(d.get('xt', [])))   # transfert3 T3-7 : fallback ancien format
            if _n0_cache != n0:
                print(f"[DOE CACHE] n0 different (cache={_n0_cache}, courant={n0}) -> recalcul DOE", flush=True)
                return None
            print(f"[DOE CACHE] charge depuis {_DOE_CACHE_FILE} (config_is_identical -> 0 SOCP DOE)", flush=True)
            return np.array(d["xt"]), np.array(d["yt"]), np.array(d["all_grad"])
        except Exception as e:
            print(f"[DOE CACHE] lecture echouee ({type(e).__name__}: {e}) -> recalcul DOE", flush=True)
        return None
    def _save_doe_cache(xt, yt, all_grad):
        try:
            json.dump({"n0": n0,
                       "xt": np.asarray(xt).tolist(),
                       "yt": np.asarray(yt).tolist(),
                       "all_grad": np.asarray(all_grad).tolist()},
                      open(_DOE_CACHE_FILE, "w"), indent=1)
            print(f"[DOE CACHE] sauve dans {_DOE_CACHE_FILE} (reutilisable si config_is_identical=True)", flush=True)
        except Exception as e:
            print(f"[DOE CACHE] sauvegarde echouee ({type(e).__name__}: {e})", flush=True)

    # --- DUMP RESTART COMPLET (2026-06-18) -----------------------------------
    # Ecrit TOUT ce qu'il faut pour ULTERIEUREMENT (partie a coder ensuite) :
    #   (a) relire le set d'entrainement ENRICHI complet (DOE + points EFF) et refit le surrogate
    #       sans recalculer de SOCP,
    #   (b) regenerer TOUS les graphes pour TOUS les points -- en particulier les 2 derniers :
    #       le graphe tolerance/convergence (print_EFF_graphs, qui consomme les historiques EFF/BB/BS/theta)
    #       et le recap final (print_visu),
    #   (c) reprendre l'enrichissement et rajouter des points (xt/yt/all_grad + u* de depart).
    # Pour l'instant : DUMP UNIQUEMENT. La relecture/reenrichissement sera codee dans un 2e temps.
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
            # (a) set d'entrainement enrichi complet (DOE + EFF)
            st["xt"]       = np.asarray(xt).tolist()       if xt       is not None else None
            st["yt"]       = np.asarray(yt).tolist()       if yt       is not None else None
            st["all_grad"] = np.asarray(all_grad).tolist() if all_grad is not None else None
            st["xt_eff"]   = [np.asarray(p).tolist() for p in xt_eff] if xt_eff else []
            st["n_doe"]    = n0
            st["n_total"]  = int(len(xt)) if xt is not None else 0
            # --- metadata round (re-enrichissements consecutifs coherents) ---
            #   enrich_round : 0 = run initial, k = k-ieme re-enrichissement
            #   round_sizes  : nb de points par round, ex [25, 2, 2] (25 = DOE+EFF initial)
            #   round_boundaries : indices cumulés dans xt -> quel point vient de quel round
            _prev_tot = sum(_round_sizes_prev) if _round_sizes_prev else 0
            if _enrich_round > 0:
                st["round_sizes"] = list(_round_sizes_prev) + [int(len(xt)) - _prev_tot]
            else:
                st["round_sizes"] = [int(len(xt))] if xt is not None else []
            st["enrich_round"]     = int(_enrich_round)
            st["round_boundaries"] = list(np.cumsum([0] + st["round_sizes"]).astype(int).tolist())
            # (b) historiques de convergence (pour le graphe tolerance)
            st["hist_EFF"]     = [float(v) for v in _eff_history_EFF]
            st["hist_BB"]      = [None if v is None else float(v) for v in _eff_history_BB]
            st["hist_BS"]      = [None if v is None else float(v) for v in _eff_history_BS]
            st["hist_theta"]   = [[float(x) for x in t] for t in _eff_history_theta]
            st["hist_beta_IS"] = [None if v is None else float(v) for v in _eff_history_beta_IS]
            # (b-bis) courbe rouge HF (hf_2d_grid_fixed) : indispensable pour redessiner
            # print_planche_EFF + print_visu sans recalculer la grille SOCP. Sinon elle vit
            # uniquement dans hf_grid_cache.json (sidecar separe). On la copie ici pour avoir
            # un dump AUTO-SUFFISANT (params bornes u1/u2 + n_grid_hf + Z aplati).
            try:    st["hf_2d_grid"] = hf_2d_grid_fixed
            except Exception: st["hf_2d_grid"] = None
            # (c) resultats FORM (u*, beta) par mode + point de depart + IS
            st["best_sp"]     = [float(v) for v in np.array(best_sp)] if best_sp is not None else None
            st["best_result"] = _u_beta(best_result) if best_result is not None else None
            st["modes"]       = [_u_beta(m) for m in modes] if modes else []
            try:
                st["IS"] = {"Pf": float(result_IS.getProbabilityEstimate())} if result_IS is not None else None
            except Exception:
                st["IS"] = None
            json.dump(st, open(_RESTART_STATE_FILE, "w"), indent=1)
            print(f"[RESTART DUMP] etat complet sauve dans {_RESTART_STATE_FILE} "
                  f"(n_total={st['n_total']}, n_eff={len(st['xt_eff'])}, "
                  f"hist_EFF={len(st['hist_EFF'])}, modes={len(st['modes'])})", flush=True)
        except Exception as e:
            print(f"[RESTART DUMP] sauvegarde echouee ({type(e).__name__}: {e})", flush=True)

    # --- LOG INCREMENTAL PAR POINT (2026-06-18) ---------------------------------
    # Fichier JSONL enrichi AU FIL du calcul : 1 ligne par point SOCP calcule, avec
    # (phase, u1, u2, fy1, fy2, g, lambda=pObj). phase in {DOE, HF, EFF, USTAR} -> permet
    # de separer/filtrer DOE vs courbe rouge HF vs enrichissement EFF vs verif au point u*.
    # Append-only (pas de reecriture -> robuste), tronque une seule fois en debut de run.
    _POINT_LOG_FILE = os.path.join(_path_ds, "points_log.jsonl")
    def _append_point_log(phase, u, x, g):
        try:
            _u = list(u) if u is not None else []
            _x = list(x) if x is not None else []
            rec = {"phase": phase, "round": _point_log_round[0]}
            for i, p in enumerate(params_names):
                rec[f"u_{p}"] = float(_u[i]) if i < len(_u) else None
                rec[f"x_{p}"] = float(_x[i]) if i < len(_x) else None
            rec["g"]      = None if g is None else float(g)
            rec["lambda"] = None if g is None else float(g) + 1.0  # lambda = pObj = g+1
            with open(_POINT_LOG_FILE, "a") as _pf:
                _pf.write(json.dumps(rec) + "\n")
        except Exception as e:
            print(f"[POINT LOG] append echoue ({type(e).__name__}: {e})", flush=True)

    def build_DOE():
        if not do_HF:
            _cached = _load_doe_cache()
            if _cached is not None:
                return _cached
        dist_X   = dist_jointe()
        T     = dist_X.getIsoProbabilisticTransformation() # on interroge dist_X et trouve la transfo n�cessaire puis l'applique ici
        T_inv = dist_X.getInverseIsoProbabilisticTransformation()
        dist_U = dist_X.getStandardDistribution()
        lhs    = ot.LHSExperiment(dist_U, n0)
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
        if not do_HF:
            SOL = [{} for _ in range(n0)] 
            for i in range(n0):
                for j in range(n_var):
                    SOL[i][params_names[j]] = X_doe[i][j]
            if n_workers_DOE and n_workers_DOE > 1:
                SOL = run_DOE_parallel(modelname, SOL, params_names, n_workers_DOE)
            else:
                SOL = run_one_SOL(modelname, SOL, params_names, sensitivity=True, with_sens_dict=None)
            yt = np.array([SOL[i]['g'] for i in range(n0)]).reshape(-1, 1)
            all_grad = np.zeros((n0, n_var))
            for i in range (n0):
                J_Tinv = T_inv.gradient(U_doe[i])
                J_Tinv_T = J_Tinv.transpose()
                grad_X_g = ot.Point([SOL[i][f'dg_{p}'] for p in params_names])
                grad_U_g = J_Tinv_T * grad_X_g
                for j in range (n_var):
                    all_grad[i][j]= grad_U_g[j]
                    SOL[i][f'dg_u{j+1}'] = grad_U_g[j]
            for i in range(n0):   # log incremental des points DOE (u-space U_doe, x-space X_doe, g)
                _append_point_log("DOE", list(U_doe[i]), list(X_doe[i]), SOL[i]['g'])
            if print_DOE:
                print("yt_doe = [")
                for i in range(n0):
                    print(f"    {yt[i][0]:.16f},")
                print("]", flush=True)
            _save_doe_cache(xt, yt, all_grad)
            return xt, yt, all_grad
        return xt

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
    # --- Paramètres du modèle analytique ---
    Es = 200000
    ecu = 0.0035
    eud = 0.045
    gamma_c_fic = 1.0  # 2-fy : gamma_c retire du dsCad (beton fixe COMPRESSIVE_STRENGTH='20.0'), reste 1.0
    gamma_s_fic = _parse(_cad_txt, 'gamma_s') # toujours dans le dsCad (fyd1/fyd2 = fy/gamma_s)
    
    # --- Fonction analytique ---
    class flexion_claude:
        def __init__(self):

            # --- Lecture du dsCad et dsLoad ---
            path = os.path.join(r'C:\workspace\storage\admin\Moulin_Blanc', modelname + '.ds')
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
            self.fym = FY_MEAN
            dist_X     = dist_jointe()   # 2-fy : lois Normales acier (legacy analytique, non utilise)
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
    
    def print_visu_ana():
        calc = flexion_claude()

        u1_lim = calc.u1_lim_plast
        u2_lim = calc.u2p_LS(u1_lim)

        # Branche plastifiée
        u1_grid = np.linspace(u1_lim, u1_max, n_grid)
        u2_grid = np.array([calc.u2p_LS(u) for u in u1_grid])

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(u1_grid, u2_grid, 'b-', lw=2,
                label=r'$u_2 = u_{2p,LS}(u_1)$  (aciers plastifiés)')
        ax.plot([u1_lim, u1_lim], [u2_lim, u2_max], 'r-', lw=2,
                label=r'$u_1 = u_{1,lim\,plast}$  (aciers non plastifiés)')
        ax.plot(u1_lim, u2_lim, 'ko', ms=6, zorder=5,
                label=f'Raccord ({u1_lim:.3f}, {u2_lim:.3f})')
        ax.plot(0, 0, 'g+', ms=12, mew=2, label='Origine')

        ax.axhline(0, color='gray', lw=0.4)
        ax.axvline(0, color='gray', lw=0.4)
        ax.set_xlabel(r'$u_1$  (espace standard, $f_c$)')
        ax.set_ylabel(r'$u_2$  (espace standard, $f_y$)')
        ax.set_title("Surface d'état-limite — flexion pivot B")
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(u1_min, u1_max)
        ax.set_ylim(u2_min, u2_max)
        plt.tight_layout()
        plt.show()
        return fig, ax

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
    def n0_min(n_var, p):
        """Taille minimale du DOE pour PCE de degre p en n_var variables.
        n0_min = C(n_var+p, p) + 1 = (n_var+p)! / (n_var! * p!) + 1"""
        return comb(n_var + p, p) + 1
    
    def update_degree(new_n0):
        global max_degree
        while new_n0 >= n0_min(n_var, max_degree+1) and max_degree+1 <= max_of_maxdegree:
            max_degree += 1
        print(f'On passe à max_degree = {max_degree} car le DOE est de {new_n0} >= n0_min({max_degree}) = {n0_min(n_var, max_degree)}')

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

        def _exec(self, u):
            u_np = np.array(u).reshape(1, -1)
            return [float(predict_gepck(self.fm, u_np)[0, 0])]

        def _exec_sample(self, U):
            U_np = np.array(U)
            return predict_gepck(self.fm, U_np)[:, 0:1].tolist()

        def _exec_sigma(self, u):
            u_np = np.array(u).reshape(1, -1)
            _, YSig2 = predict_gepck(self.fm, u_np, return_var=True)
            return float(np.sqrt(max(0.0, float(YSig2[0, 0]))))

        def _gradient(self, u):
            u_np = np.array(u).reshape(1, -1)
            G = predict_gradient_gepck(self.fm, u_np)   # (1, Mred)
            return [[float(G[0, i])] for i in range(self.fm['Mred'])]

    # --------------------------------------------------------------------------- #
    # WRAPPER BORNES DE CONFIANCE DU SURROGATE                                   #

    class BoundSurrogateFunction(ot.OpenTURNSPythonFunction):
        """
        g_bound(u) = g_ot(u) + sign * 2 * sigma_func(u)
        sign = +1  →  borne supérieure  g_sup = μ̂g + 2σ̂g
        sign = -1  →  borne inférieure  g_inf = μ̂g − 2σ̂g

        Wrappeur externe : ne modifie aucune classe existante.
        Compatible avec tous les modèles (GEPCK, KRG, GEK, PCKRG).
        Pas de _gradient défini → OT utilise différences finies si FORM est appelé.
        """
        def __init__(self, g_ot, sigma_func, sign):
            super().__init__(n_var, 1)
            self._g_ot       = g_ot
            self._sigma_func = sigma_func
            self._sign       = sign   # +1 ou -1

        def _exec(self, u):
            u_pt  = ot.Point(list(u))
            mu    = self._g_ot(u_pt)[0]
            sigma = self._sigma_func(u_pt)
            return [mu + self._sign * 2.0 * sigma]

        def _exec_sample(self, U):
            # 2026-06-18 : version BATCHEE -> l'IS des bornes sup/inf n'est plus evaluee
            # point-par-point par OpenTURNS (90000+ appels _exec). mu+var en 1 appel
            # predict_gepck via sigma_func.__self__.fm (meme technique que _batch_mu_sigma).
            # Valeurs identiques a _exec, juste batchees. IS sup/inf ~17s -> ~3s.
            _fm = getattr(getattr(self._sigma_func, '__self__', None), 'fm', None)
            if _fm is not None:
                U_np = np.array(U)
                mu_arr, var_arr = predict_gepck(_fm, U_np, return_var=True)
                mu    = mu_arr[:, 0]
                sigma = np.sqrt(np.maximum(0.0, var_arr[:, 0]))
                return (mu + self._sign * 2.0 * sigma).reshape(-1, 1).tolist()
            # fallback generique point-par-point (modeles non-GEPCK)
            out = []
            for u in U:
                u_pt  = ot.Point(list(u))
                out.append([self._g_ot(u_pt)[0] + self._sign * 2.0 * self._sigma_func(u_pt)])
            return out

    # Usage :
    #   g_ot_sup = ot.Function(BoundSurrogateFunction(g_ot, sigma_func, +1))
    #   g_ot_inf = ot.Function(BoundSurrogateFunction(g_ot, sigma_func, -1))

    # def eval_PCE():
    #     return do_pce
    # --------------------------------------------------------------------------- #
    # FONCTIONS LIEES AU KRG                                                      #

    def build_metamodel_KRG(xt, yt):
        n_var = xt.shape[1]
        basis = ot.ConstantBasisFactory(n_var).build()
        covarianceModel = ot.SquaredExponential([1.0] * n_var)
        algo_KRG = ot.KrigingAlgorithm(xt, yt, covarianceModel, basis)
        if do_KRG:
            algo_KRG.setOptimizationBounds(ot.Interval([1.0] * n_var, [5.0] * n_var))
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
    def init_g_ot(g_ot, sigma_func, xt, yt, all_grad):
        """
        Cette fonction génère xt, yt, all_grad si xt n'est pas vide puis
        contruit un metamodele à partir de ces points. Dans le cas HF, elle 
        créé uniquement une fonction OT. Retourne g_ot, sigma_func, xt, yt, all_grad.
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
            _marginals = [{'Type': 'Gaussian', 'Parameters': [0.0, 1.0]} for _ in range(n_var)]
            _copula    = {'Type': 'Independent', 'Parameters': np.eye(n_var)}
            _opts      = {'Mode': 'optimal',
                          'PCE': {'Degree': list(range(1, max_degree + 1)), 'Method': 'LARS'}}
            _Y_aug = build_Y_aug(yt, all_grad)
            print(f"=== GEPCK fit N={len(xt)} ===", flush=True)
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
            global _gepck_pce_label, _eff_history_theta
            _gepck_pce_label = ' '.join(_terms)
            _eff_history_theta.append(list(_fm['Kriging'][0]['theta']))
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

    # --- Multi-start FORM depuis les points du DOE ---
    def FORM_all_modes(starting_points, tol_all_modes, event):
        """
        Multi-start FORM + DBSCAN pour identifier les modes de défaillance.
        - Chaque cluster DBSCAN = un mode distinct.
        - u* isolés (label -1) = descentes mal convergées, ignorées.
        """
        all_u_star   = []   # u* de chaque run réussi
        all_results  = []   # FORMResult correspondant
        all_sp       = []   # point de départ correspondant
        n_total = len(starting_points)
        _t_fam_start = time.perf_counter()

        for k, sp in enumerate(starting_points):
            print(f"  FORM {k+1}/{n_total}...", flush=True)
            try:
                solver = ot.AbdoRackwitz()
                solver.setStartingPoint(sp.tolist())
                solver.setMaximumIterationNumber(n_max_FORM)
                solver.setCheckStatus(False)
                solver.setMaximumConstraintError(tol_FORM)
                form_i = ot.FORM(solver, event)
                _t_fm = time.perf_counter()
                form_i.run()
                _dt_fm = time.perf_counter() - _t_fm
                r_i    = form_i.getResult()
                u_star = np.array(r_i.getStandardSpaceDesignPoint())
                all_u_star.append(u_star)
                all_results.append(r_i)
                all_sp.append(sp)
                print(f"  [sp={[round(v,3) for v in np.array(sp)]}, "
                    f"u*={[round(v,3) for v in u_star]}, "
                    f"beta={r_i.getHasoferReliabilityIndex():.4f}]", flush=True)
                print(f"  [TIMING FORM_all_modes] start {k+1}/{n_total} : FORM={_dt_fm:.3f}s", flush=True)
            except Exception as e:
                print(f"  [sp={[round(v,3) for v in np.array(sp)]}, "
                    f"ECHEC ({type(e).__name__})]", flush=True)

        _t_log(f"  [TIMING FORM_all_modes] TOTAL {n_total} starts (sequentiel)", _t_fam_start)
        if not all_u_star:
            return [], []

        # --- Cas 1 point : pas de DBSCAN ---
        if len(all_u_star) == 1:
            print(f"\n1 mode(s) distinct(s) (1 seul point de depart, pas de DBSCAN) :", flush=True)
            u = [round(v, 3) for v in all_results[0].getStandardSpaceDesignPoint()]
            print(f"  mode 1 : beta={all_results[0].getHasoferReliabilityIndex():.4f}  "
                  f"Pf={all_results[0].getEventProbability():.3e}  u*={u}", flush=True)
            return [all_results[0]], [all_sp[0]]

        # --- DBSCAN ---
        U_all  = np.array(all_u_star)          # shape (n_runs_ok, n_var)
        db     = DBSCAN(eps=tol_all_modes, min_samples=2).fit(U_all)
        labels = db.labels_

        n_noise = np.sum(labels == -1)
        if n_noise > 0:
            print(f"  {n_noise} descente(s) mal convergée(s) ignorée(s) (bruit DBSCAN)", flush=True)

        # --- Un mode par cluster : FORMResult avec beta minimal ---
        modes     = []
        best_sps  = []
        for lbl in sorted(set(labels) - {-1}):
            idx_cluster = [i for i, l in enumerate(labels) if l == lbl]
            best_i = min(idx_cluster,
                        key=lambda i: all_results[i].getHasoferReliabilityIndex())
            modes.append(all_results[best_i])
            best_sps.append(all_sp[best_i])

        order = sorted(range(len(modes)), key=lambda i: modes[i].getHasoferReliabilityIndex())
        modes    = [modes[i]    for i in order]
        best_sps = [best_sps[i] for i in order]

        print(f"\n{len(modes)} mode(s) distinct(s) "
            f"(DBSCAN eps={tol_all_modes}, min_samples=2) :", flush=True)
        for i, m in enumerate(modes):
            u = [round(v, 3) for v in m.getStandardSpaceDesignPoint()]
            print(f"  mode {i+1} : beta={m.getHasoferReliabilityIndex():.4f}  "
                f"Pf={m.getEventProbability():.3e}  u*={u}", flush=True)

        return modes, best_sps
    
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
    class EFFFunction(ot.OpenTURNSPythonFunction):
        def __init__(self, g_ot, sigma_func):
            super().__init__(n_var, 1)
            self.g_ot = g_ot
            self.sigma_func = sigma_func

        def _exec(self, u):
            u = ot.Point(u)
            sigmaG  = self.sigma_func(u)
            if sigmaG <= 0.0:
                return [0.0]
            muG     = self.g_ot(u)[0]
            epsilon = epsilon_factor * sigmaG
            t1 = -muG / sigmaG
            t2 = (epsilon + muG) / sigmaG
            t3 = (epsilon - muG) / sigmaG
            return [2*muG*norm.cdf(t1) - (epsilon+muG)*norm.cdf(-t2) + (epsilon-muG)*norm.cdf(t3) + sigmaG*(-2*norm.pdf(t1) + norm.pdf(t2) + norm.pdf(t3))]

    def run_EFF(g_ot, sigma_func, xt, yt, all_grad):
        """
        Cette fonction reçoit le métamodele et ses paramètres, et l'améliore jusqu'à vérifier le critère EFF puis
        renvoie métamodèle+ paramètres mis à jour.
        """
        # --- Si aucune branche ne tourne, on ne fait rien ---
        if g_ot is None or do_HF:
            return g_ot, sigma_func, xt, yt, all_grad, []

        _point_log_phase[0] = "EFF"   # les run_HF d'enrichissement sont taggues EFF
        # En reprise : on reseed xt_eff avec les EFF deja calcules (pour que TOUS les triangles
        # rouges s'affichent dans les plots) ; l'enrichissement continue selon le critere EFF courant.
        xt_eff = list(_restart_xt_eff) if restart_enrich_only else []
        _n_eff_start = len(xt_eff)

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
                _t_form = time.perf_counter()
                form_i.run()
                _dt_form = time.perf_counter() - _t_form
                r_i = form_i.getResult()
            except Exception as e:
                print(f"  [{label}] FORM echoue ({type(e).__name__})", flush=True)
                return None, None
            beta_f  = r_i.getHasoferReliabilityIndex()
            pf_f    = r_i.getEventProbability()
            # --- NOUVEAU CHEMIN : IS adaptatif parallelisable (sonde + ramp-up), si flag + fm ---
            if _IS_PARALLEL and fm is not None:
                u_star  = list(r_i.getStandardSpaceDesignPoint())
                _state  = dict(xt=xt, yt=yt, all_grad=all_grad, max_degree=max_degree)
                _cap    = int(os.environ.get("_IS_CAP", str(n_IS)))
                _t_is   = time.perf_counter()
                _r      = adaptive_is(fm, _state, u_star, sign=sign,
                                      cov_target=cov_IS, cap_blocks=_cap,
                                      K=_IS_K, chunk=_IS_CHUNK, probe_blocks=_IS_PROBE)
                _dt_is  = time.perf_counter() - _t_is
                pf_IS   = _r['pf']
                beta_IS = float(-ot.Normal().computeQuantile(pf_IS)[0]) if pf_IS > 0 else float('nan')
                print(f"  [{label}] beta_FORM={beta_f:.4f}  Pf_FORM={pf_f:.3e}"
                      f" | Pf_IS={pf_IS:.3e}  beta_IS={beta_IS:.4f}  COV={_r['cov']:.3f}  [PAR:{_r['mode']}]", flush=True)
                print(f"  [IS DETAIL PAR] {label} : blocs={_r['n_blocks']} evals~{_r['n_evals']:,} rondes_par={_r.get('n_rounds',0)} "
                      f"COV={_r['cov']:.4f} (cible {cov_IS}) t_IS={_r['t']:.2f}s debit={_r['n_evals']/max(_r['t'],1e-9):,.0f} evals/s", flush=True)
                print(f"  [TIMING FORM/IS] {label} : FORM={_dt_form:.2f}s  IS={_dt_is:.2f}s (parallel)  (total={_dt_form+_dt_is:.2f}s)", flush=True)
                return beta_IS, pf_IS
            # --- CHEMIN D'ORIGINE : IS OpenTURNS (inchange, fallback) ---
            _t_is = time.perf_counter()
            res_IS  = run_IS([r_i], ev_i)
            _dt_is = time.perf_counter() - _t_is
            pf_IS   = res_IS.getProbabilityEstimate()
            beta_IS = float(-ot.Normal().computeQuantile(pf_IS)[0])
            cov_v   = res_IS.getCoefficientOfVariation()
            print(f"  [{label}] beta_FORM={beta_f:.4f}  Pf_FORM={pf_f:.3e}"
                  f" | Pf_IS={pf_IS:.3e}  beta_IS={beta_IS:.4f}  COV={cov_v:.3f}", flush=True)
            print(f"  [TIMING FORM/IS] {label} : FORM={_dt_form:.2f}s  IS={_dt_is:.2f}s  (total={_dt_form+_dt_is:.2f}s)", flush=True)
            return beta_IS, pf_IS

        def _three_form_is(g_ot_i, sigma_func_i, label, b_mid_precalc=None):
            """FORM+IS sur g, g+2sigma, g-2sigma. Affiche les 3 lignes + ratio.
            Retourne (ratio, pf_mid, pf_sup, pf_inf) ou (None, None, None, None).
            2026-06-18 : b_mid_precalc=(beta_IS, pf_IS) reutilise le mu deja calcule pour eviter
            un FORM+IS redondant sur g (le FORM est deterministe donc identique ; on economise
            ~1/4 du bloc FORM/IS). Si None -> calcul interne (comportement d'origine)."""
            g_sup_i = ot.Function(BoundSurrogateFunction(g_ot_i, sigma_func_i, +1))
            g_inf_i = ot.Function(BoundSurrogateFunction(g_ot_i, sigma_func_i, -1))
            _FM = getattr(getattr(sigma_func_i, '__self__', None), 'fm', None)   # surrogate courant (None si non-GEPCK)
            if b_mid_precalc is not None:
                b_mid, pf_mid = b_mid_precalc
                print(f"  [{label} μ] reutilise mu conv (pas de recalcul FORM/IS redondant)", flush=True)
            else:
                b_mid, pf_mid = _form_is_iter(g_ot_i, f"{label} μ", sign=0, fm=_FM)
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
        # --- On résoud u = argmax(EFF) ---
        f = ot.Function(EFFFunction(g_ot, sigma_func))
        bounds = _eff_bounds()
        problem = ot.OptimizationProblem(f, ot.Function(), ot.Function(), bounds)
        problem.setMinimization(False)
        algo_opti = ot.NLopt(problem, "GN_DIRECT")
        algo_opti.setStartingPoint([0.0] * n_var)
        algo_opti.setMaximumCallsNumber(n_NLopt_EFF)
        _t0 = time.perf_counter()
        algo_opti.run()
        _t_log(f"  [recherche EFF initiale NLopt GN_DIRECT n_max={n_NLopt_EFF}]", _t0)
        u_opt = algo_opti.getResult().getOptimalPoint()
        _sigG = sigma_func(u_opt)
        _muG  = g_ot(ot.Point(u_opt))[0]
        _eps  = epsilon_factor * _sigG
        print(f"  EFF debug u_opt={list(np.round(np.array(u_opt),3))} : sigmaG={_sigG:.6f}  muG={_muG:.6f}  epsilon={_eps:.6f}", flush=True)
        print(f"  EFF initial : EFF(u_opt)={f(u_opt)[0]:.6f}, tol={tol_EFF}", flush=True)

        iter_count = 0
        if EFF_criteria == 'BB':
            _cond = lambda: len(xt_eff) < n_max_EFF_points and abs(f(u_opt)[0]) > tol_EFF and count_valid_BB < 3
        elif EFF_criteria == 'BS':
            _cond = lambda: len(xt_eff) < n_max_EFF_points and abs(f(u_opt)[0]) > tol_EFF and count_valid_BS < 3
        elif EFF_criteria == 'both':
            _cond = lambda: len(xt_eff) < n_max_EFF_points and abs(f(u_opt)[0]) > tol_EFF and count_valid_both < 2
        elif EFF_criteria == 'at_least_one':
            _cond = lambda: len(xt_eff) < n_max_EFF_points and abs(f(u_opt)[0]) > tol_EFF and not (count_valid_BB >= 3 or count_valid_BS >= 3 or count_valid_both >= 2)
        elif EFF_criteria == 'n_points':   # critere = nombre fixe de points EFF (pas de convergence)
            _cond = lambda: (len(xt_eff) - _n_eff_start) < n_eff_target
        else:
            _cond = lambda: len(xt_eff) < n_max_EFF_points and abs(f(u_opt)[0]) > tol_EFF

        _beta_IS_0, _pf_IS_0 = _form_is_iter(g_ot, f"N={len(xt)} initial μ conv")
        global _eff_history_EFF, _eff_history_BB, _eff_history_BS, _eff_history_beta_IS
        # En reprise : on ETEND les historiques charges du dump (pas de reset) -> les graphes
        # (tolerance) couvriront ancien + nouveau. Sinon (run initial) : reset comme avant.
        if restart_enrich_only:
            list_beta_IS = list(_eff_history_beta_IS) + ([_beta_IS_0] if _beta_IS_0 is not None else [])
        else:
            list_beta_IS = [_beta_IS_0] if _beta_IS_0 is not None else []
            _eff_history_BB = []
            _eff_history_BS = []
        list_ratio_BB = _eff_history_BB   # alias — même objet (charge en reprise, vide sinon)
        list_ratio_BS = _eff_history_BS
        _eff_history_EFF.append(f(u_opt)[0])   # EFF initial (avant ajout du 1er point)

        # init _b_mid/_pf_mid_conv (mu conv initial) -> BB informatif final dispose d'une valeur meme si 0 iter
        _b_mid, _pf_mid_conv = _beta_IS_0, _pf_IS_0
        # --- Ratio BB initial (avant tout enrichissement, pour criteres qui tracent BB) ---
        # 'n_points' inclus : on calcule BB/BS pour les GRAPHES de tolerance meme si l'arret = compte.
        if EFF_criteria in ('BB', 'both', 'at_least_one', 'n_points'):
            _ratio_init_bb, _, _, _ = _three_form_is(g_ot, sigma_func, f"N={len(xt)} initial BB")
            list_ratio_BB.append(_ratio_init_bb)
            if EFF_criteria in ('BB', 'at_least_one') and _ratio_init_bb is not None and _ratio_init_bb < tol_BB:
                count_valid_BB = 1

        while _cond():
            _sigG = sigma_func(u_opt)
            _muG  = g_ot(ot.Point(u_opt))[0]
            print(f"  EFF={f(u_opt)[0]:.6f} > {tol_EFF} -- u_opt={list(np.round(np.array(u_opt),3))}  sigmaG={_sigG:.6f}  muG={_muG:.6f}", flush=True)
            _eff_history_EFF.append(f(u_opt)[0])   # EFF apres rebuild a cette iteration
            xt_eff.append(np.array(u_opt))
            # --- On reconstruit le modèle ---
            g_val, grad_U, _ = run_HF(np.array(u_opt))
            xt = np.vstack([xt, [np.array(u_opt)]])
            yt = np.vstack([yt, [[g_val]]])
            grad_val = np.array([[float(grad_U[i]) for i in range(n_var)]])
            all_grad = np.vstack([all_grad, grad_val])
            # --- Upgrade max_degree si assez de points ---
            _degree_avant = max_degree
            update_degree(len(xt))
            degree_upgraded = (max_degree != _degree_avant)
            _t0 = time.perf_counter()
            g_ot, sigma_func, xt, yt, all_grad = init_g_ot(g_ot, sigma_func, xt, yt, all_grad)
            _t_log(f"  [refit GEPCK surrogate N={len(xt)}]", _t0)

            # --- FORM+IS sur le surrogate mis à jour (BB uniquement) ---
            if EFF_criteria == 'BB':
                iter_count += 1
                _ratio, _, _, _ = _three_form_is(g_ot, sigma_func, f"N={len(xt)} EFF iter {iter_count}")
                if _ratio is not None and _ratio < tol_BB:
                    count_valid_BB += 1
                else:
                    count_valid_BB = 0
                list_ratio_BB.append(_ratio)

            # --- Suivi convergence beta_IS ---
            _b_mid, _pf_mid_conv = _form_is_iter(g_ot, f"N={len(xt)} μ conv")
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
            # --- Critere both : BB et BS simultanement ---
            if EFF_criteria == 'both':
                iter_count += 1
                _ratio_bb, _, _, _ = _three_form_is(g_ot, sigma_func, f"N={len(xt)} both iter {iter_count}")
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
            # --- Critere at_least_one (et n_points : meme calcul BB/BS pour les GRAPHES) ---
            # En 'n_points' on calcule tout comme at_least_one (BB/BS/compteurs remplis -> graphe
            # tolerance complet) mais l'ARRET reste le compte (cond override len(xt_eff)).
            if EFF_criteria in ('at_least_one', 'n_points'):
                iter_count += 1
                _ratio_bb, _, _, _ = _three_form_is(g_ot, sigma_func, f"N={len(xt)} alo iter {iter_count}", b_mid_precalc=(_b_mid, _pf_mid_conv))
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
            _about_to_upgrade = (len(xt) + 1 > n0_min(n_var, max_degree + 1) and max_degree + 1 <= max_of_maxdegree)
            if print_EFF_progres or degree_upgraded or _about_to_upgrade:
                _t0 = time.perf_counter()
                print_planche_EFF(g_ot, sigma_func, xt, xt_eff)
                _t_log(f"  [PNG print_planche_EFF N={len(xt)}]", _t0)

            # --- DUMP INCREMENTAL (kill-safe) : reecrit restart_state APRES CHAQUE point EFF ---
            # Un arret en cours laisse un dump rechargeable jusqu'au dernier point fait.
            # best_result/modes/IS=None mid-run (calcules seulement a la fin via FORM+IS) -> sans
            # importance : la reprise n'a besoin que de xt/yt/all_grad+degre+hf_grid+histos ; les
            # modes se recalculent par FORM. Cout ~ms (petit JSON).
            _eff_history_beta_IS = list(list_beta_IS)
            _save_restart_state(xt, yt, all_grad, xt_eff, None, None, [], None)

            # --- On re-résoud u = argmax(EFF) ---
            f = ot.Function(EFFFunction(g_ot, sigma_func))
            bounds = _eff_bounds()
            problem = ot.OptimizationProblem(f, ot.Function(), ot.Function(), bounds)
            problem.setMinimization(False)
            algo_opti = ot.NLopt(problem, "GN_DIRECT")
            algo_opti.setStartingPoint([0.0] * n_var)
            algo_opti.setMaximumCallsNumber(n_NLopt_EFF)
            _t0 = time.perf_counter()
            algo_opti.run()
            _t_log(f"  [recherche EFF NLopt GN_DIRECT n_max={n_NLopt_EFF}]", _t0)
            u_opt = algo_opti.getResult().getOptimalPoint()

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

        # --- BB informatif final : ratio enveloppe surrogate (sup/inf) a convergence (1 _three_form_is) ---
        _ratio_bb_final, _, _, _ = _three_form_is(g_ot, sigma_func, "BB final", b_mid_precalc=(_b_mid, _pf_mid_conv))
        print(f"  [BB informatif final] ratio = {_ratio_bb_final}  tol_BB = {tol_BB}", flush=True)

        _eff_history_beta_IS = list(list_beta_IS)   # snapshot pour le dump restart (list_beta_IS est local)
        return g_ot, sigma_func, xt, yt, all_grad, xt_eff

    # --------------------------------------------------------------------------- #
    # FONCTIONS RESULTATS/ AFFICHAGE                                              #

    # ============================================================================
    # HELPERS VECTORISES (batch eval surrogate sur grille - parallelise via BLAS)
    # ============================================================================
    def _batch_mu_sigma(g_ot, sigma_func, grid):
        """Calcule (mu, sigma) en batch sur une grille de points.
        - GEPCK : 1 appel predict_gepck(return_var=True) -> mu + var en 1 fois (BLAS multi-thread)
        - Autres modeles : batch via ot.Sample pour mu, loop fallback pour sigma
        Retourne (mu_1D, sigma_1D) en np.array de shape (len(grid),).
        """
        # 2026-06-18 : chemin rapide GEPCK robuste — on recupere le modele fm via le bound method
        # sigma_func (= gepck_impl._exec_sigma), car g_ot.getImplementation() n'expose PAS .fm
        # (OpenTURNS enveloppe la PythonFunction). Variance batchee en 1 appel au lieu de 90000
        # appels point-par-point -> PNG print_planche_EFF ~80s -> ~1s.
        _fm = getattr(getattr(sigma_func, '__self__', None), 'fm', None)
        if _fm is not None:
            mu_arr, sig2_arr = predict_gepck(_fm, grid, return_var=True)
            mu = mu_arr[:, 0]
            sigma = np.sqrt(np.maximum(0.0, sig2_arr[:, 0]))
            return mu, sigma
        impl = g_ot.getImplementation()
        impl_name = type(impl).__name__
        if hasattr(impl, 'fm') and 'GEPCK' in impl_name:
            # GEPCK fast path : batch direct avec variance
            mu_arr, sig2_arr = predict_gepck(impl.fm, grid, return_var=True)
            mu = mu_arr[:, 0]
            sigma = np.sqrt(np.maximum(0.0, sig2_arr[:, 0]))
            return mu, sigma
        else:
            # Fallback generique : batch pour mu, loop pour sigma
            grid_ot = ot.Sample(grid.tolist())
            mu = np.array(g_ot(grid_ot))[:, 0]
            sigma = np.array([sigma_func(pt) for pt in grid])
            return mu, sigma

    def _eff_vectorized(mu, sigma, eps_factor):
        """Calcul vectorise du critere EFF (Expected Feasibility Function).
        Equivalent a EFFFunction._exec mais sur arrays numpy."""
        eps = eps_factor * sigma
        safe_sigma = np.where(sigma > 0, sigma, 1.0)
        t1 = -mu / safe_sigma
        t2 = (eps + mu) / safe_sigma
        t3 = (eps - mu) / safe_sigma
        eff_vals = (2*mu*norm.cdf(t1) - (eps+mu)*norm.cdf(-t2) + (eps-mu)*norm.cdf(t3)
                    + sigma*(-2*norm.pdf(t1) + norm.pdf(t2) + norm.pdf(t3)))
        return np.where(sigma > 0, eff_vals, 0.0)
    # ============================================================================

    def print_EFF_graphs():
        """Planche 3 subplots : historique EFF, criteres BB/BS, theta Kriging.
        Lit les globaux _eff_history_*. Sauvegarde en PNG dans out_dir_eff."""
        if not (_eff_history_EFF or _eff_history_BB or _eff_history_theta):
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        _clip = 1e-12   # evite log(0)

        # --- Subplot 1 : EFF vs iterations ---
        ax = axes[0]
        x_eff = list(range(len(_eff_history_EFF)))
        vals_eff = [max(abs(v), _clip) for v in _eff_history_EFF]
        ax.semilogy(x_eff, vals_eff, 'b-o', ms=4, lw=1.2, label='EFF(u_opt)')
        ax.axhline(tol_EFF, color='orange', ls='--', lw=1, label=f'tol_EFF={tol_EFF:.1e}')
        ax.set_xlabel('Iteration EFF')
        ax.set_ylabel('EFF (echelle log)')
        ax.set_title('Convergence EFF')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.4)

        # --- Subplot 2 : BB / BS vs iterations ---
        ax = axes[1]
        if _eff_history_BB:
            x_bb = list(range(1, len(_eff_history_BB) + 1))
            vals_bb = [max(v, _clip) if v is not None else np.nan for v in _eff_history_BB]
            ax.semilogy(x_bb, vals_bb, 'g-o', ms=4, lw=1.2, label='BB')
            ax.axhline(tol_BB, color='g', ls='--', lw=0.8, label=f'tol_BB={tol_BB:.1e}')
        if _eff_history_BS:
            x_bs = list(range(1, len(_eff_history_BS) + 1))
            vals_bs = [max(v, _clip) if v is not None else np.nan for v in _eff_history_BS]
            ax.semilogy(x_bs, vals_bs, 'r-s', ms=4, lw=1.2, label='BS')
            ax.axhline(tol_BS, color='r', ls='--', lw=0.8, label=f'tol_BS={tol_BS:.1e}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Ratio (echelle log)')
        ax.set_title('Criteres BB / BS')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.4)

        # --- Subplot 3 : theta Kriging vs iterations ---
        ax = axes[2]
        if _eff_history_theta:
            thetas = np.array(_eff_history_theta)   # shape (n_fits, n_var)
            x_th = list(range(len(_eff_history_theta)))
            for k in range(thetas.shape[1]):
                lbl = params_names[k] if k < len(params_names) else f'dim{k}'
                ax.semilogy(x_th, np.maximum(thetas[:, k], _clip), '-o', ms=4, lw=1.2, label=f'theta_{lbl}')
            norms_th = np.maximum(np.linalg.norm(thetas, axis=1), _clip)
            ax.semilogy(x_th, norms_th, 'k--', ms=3, lw=1, label='||theta||')
        ax.set_xlabel('Iteration (fit)')
        ax.set_ylabel('theta (echelle log)')
        ax.set_title('Evolution theta Kriging')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.4)

        fig.tight_layout()
        fname = f'EFF_graphs_{timestamp}.png'
        fig.savefig(os.path.join(out_dir_eff, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [EFF_graphs] -> {fname}", flush=True)

    # --- Cache sidecar JSON de la grille HF (evite de recalculer les 49 SOCP / ~2h30) ---
    # Fichier dans le dossier .ds du projet -> 1 cache par projet (tablier / diagonal).
    _HF_CACHE_FILE       = os.path.join(_path_ds, "hf_grid_cache.json")
    _HF_CACHE_FILE_FINAL = os.path.join(_path_ds, "hf_grid_cache_final.json")   # transfert3 T3-4 : coupe slice_def_final
    hf_2d_grid_fixed_final = None

    def _load_hf_cache(n_grid_hf_local, cache_file, sd):
        import json as _j
        if not config_is_identical:
            print("[HF CACHE] config_is_identical=False -> recalcul grille HF", flush=True)
            return None
        if not os.path.exists(cache_file):
            print(f"[HF CACHE] aucun cache ({cache_file}) -> calcul grille HF", flush=True)
            return None
        try:
            d = _j.load(open(cache_file))
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
        import json as _j
        try:
            _sd = sd if sd is not None else (0, 1, {})
            _j.dump({'Z': Z.tolist(), 'slice_def': [_sd[0], _sd[1], {str(k): v for k, v in _sd[2].items()}]},
                    open(cache_file, 'w'), indent=1)
            print(f"[HF CACHE] sauve dans {cache_file}", flush=True)
        except Exception as e:
            print(f"[HF CACHE] sauvegarde echouee ({type(e).__name__}: {e})", flush=True)

    def _compute_hf_grid_with_progress(grid_hf, n_grid_hf_local, context="",
                                        cache_file=None, sd=None, grid_var_name='hf_2d_grid_fixed'):
        """Calcule la grille HF point par point avec impression progress + ETA.
        Lecture/ecriture AUTOMATIQUE d'un cache sidecar JSON (plus de copier-coller).
        Valeurs g BRUTES (= run_HF), AUCUN traitement de divergence (comme AC3 flexion).
        cache_file/sd/grid_var_name : defauts _HF_CACHE_FILE / slice_def / 'hf_2d_grid_fixed'."""
        global hf_2d_grid_fixed, hf_2d_grid_fixed_final
        import time as _time_local
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
        _point_log_phase[0] = "HF"   # les run_HF de la grille courbe rouge sont taggues HF
        n_total = len(grid_hf)
        Z_flat = []
        _t_start = _time_local.perf_counter()
        print(f"\n##### HF GRID START: {n_grid_hf_local}x{n_grid_hf_local} = {n_total} points STRAINS ({context}) #####", flush=True)
        print(f"##### Estimation : ~{n_total * 3:.0f} min total #####\n", flush=True)
        for i, pt in enumerate(grid_hf):
            _t_pt0 = _time_local.perf_counter()
            g_val = run_HF(pt)[0]
            Z_flat.append(g_val)
            _t_pt = _time_local.perf_counter() - _t_pt0
            _t_elapsed = _time_local.perf_counter() - _t_start
            _t_avg = _t_elapsed / (i + 1)
            _t_eta = _t_avg * (n_total - i - 1)
            _u_str = ', '.join(f'{pt[j]:+.3f}' for j in range(len(pt)))
            print(f"  [HF GRID {i+1:2d}/{n_total}]  u=[{_u_str}]  g={g_val:+.4f}  "
                  f"dt={_t_pt:.0f}s  elapsed={_t_elapsed/60:.1f}min  "
                  f"ETA={_t_eta/60:.1f}min", flush=True)
        _t_total = (_time_local.perf_counter() - _t_start) / 60
        print(f"\n##### HF GRID DONE in {_t_total:.1f} min ({n_total} appels STRAINS) #####\n", flush=True)
        Z = np.array(Z_flat).reshape(n_grid_hf_local, n_grid_hf_local)
        _grid_dict = {'params': {'slice_def': sd, 'n_grid_hf': n_grid_hf_local}, 'Z': Z.tolist()}
        if grid_var_name == 'hf_2d_grid_fixed_final':
            hf_2d_grid_fixed_final = _grid_dict
        else:
            hf_2d_grid_fixed = _grid_dict
        _save_hf_cache(Z, n_grid_hf_local, cache_file, sd)
        return Z

    # --- HF GRILLE FULL (n_var-D) — transfert3 T3-5 ---
    _HF_FULL_CACHE_FILE = os.path.join(_path_ds, "hf_grid_full_cache.json")
    _hf_grid_full = [None]        # [Z_full] en memoire (array n_var-D), liste pour mutabilite dans closures
    _hf_grid_full_axes = [None]   # axes de la grille (liste de 1D arrays)

    def _load_hf_grid_full():
        import json as _j
        if not config_is_identical:
            return None
        if not os.path.exists(_HF_FULL_CACHE_FILE):
            return None
        try:
            d = _j.load(open(_HF_FULL_CACHE_FILE))
            if d.get('n_var') != n_var or d.get('n_grid') != n_grid_hf:
                print(f"[HF FULL CACHE] dimensions differentes -> recalcul", flush=True)
                return None
            print(f"[HF FULL CACHE] charge depuis {_HF_FULL_CACHE_FILE} (0 SOCP)", flush=True)
            return np.array(d['Z'])
        except Exception as e:
            print(f"[HF FULL CACHE] lecture echouee ({type(e).__name__}: {e}) -> recalcul", flush=True)
        return None

    def _save_hf_grid_full(Z_full):
        import json as _j
        try:
            _j.dump({'Z': Z_full.tolist(), 'n_var': n_var, 'n_grid': n_grid_hf},
                    open(_HF_FULL_CACHE_FILE, 'w'), indent=1)
            print(f"[HF FULL CACHE] sauve dans {_HF_FULL_CACHE_FILE}", flush=True)
        except Exception as e:
            print(f"[HF FULL CACHE] sauvegarde echouee ({type(e).__name__}: {e})", flush=True)

    def _compute_hf_grid_full():
        """Calcule la grille HF complete (n_grid_hf^n_var points STRAINS)."""
        cached = _load_hf_grid_full()
        if cached is not None:
            _hf_grid_full[0] = cached
            _hf_grid_full_axes[0] = [np.linspace(*_ax_bounds_hf(k), n_grid_hf) for k in range(n_var)]
            return cached
        import time as _time_local
        _point_log_phase[0] = "HF_FULL"
        axes = [np.linspace(*_ax_bounds_hf(k), n_grid_hf) for k in range(n_var)]
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
        _hf_grid_full[0] = Z_full
        _hf_grid_full_axes[0] = axes
        _save_hf_grid_full(Z_full)
        return Z_full

    def _extract_hf_slice(sd):
        """Extrait une coupe 2D (n_grid_hf x n_grid_hf) depuis la grille full par interpolation lineaire."""
        from scipy.interpolate import RegularGridInterpolator
        idx_x, idx_y, fixed = sd
        interp = RegularGridInterpolator(_hf_grid_full_axes[0], _hf_grid_full[0], method='linear')
        ux = np.linspace(u1_min, u1_max, n_grid_hf)
        uy = np.linspace(u2_min, u2_max, n_grid_hf)
        UX, UY = np.meshgrid(ux, uy)
        pts = np.zeros((n_grid_hf * n_grid_hf, n_var))
        pts[:, idx_x] = UX.ravel()
        pts[:, idx_y] = UY.ravel()
        for idx, val in fixed.items():
            pts[:, idx] = val
        return interp(pts).reshape(n_grid_hf, n_grid_hf)

    def _get_hf_slice(sd, cache_file=None, grid_var_name='hf_2d_grid_fixed'):
        """Retourne Z_true (n_grid_hf x n_grid_hf) pour une coupe sd.
        Cascade : cache 2D memoire -> cache 2D disque -> grille full -> recalcul 2D (49 SOCP)."""
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
        # 3. Grille full (interpolation de coupe)
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
        # 4. Recalcul 2D direct (49 SOCP)
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

    def _draw_red_curve(ax, hf_cache, linestyles='-'):
        """Trace la courbe rouge g=0 HF : contour direct sur la grille HF 7x7 brute.

        Comportement d'origine (AC_moulin_blanc.py) : AUCUN filtrage, AUCUN lissage.
        Les points divergents du cache (sentinelle -1.0 ou NaN) sont traces tels quels
        -> marching squares natif de matplotlib (NaN masques automatiquement).
        Centralise le trace pour les 3 panneaux (EFF, sigma, comparaison surrogates).
        """
        if hf_cache is None or 'Z' not in hf_cache:
            return
        Z_raw = np.array(hf_cache['Z'], dtype=float)
        u1_hf = np.linspace(u1_min, u1_max, n_grid_hf)
        u2_hf = np.linspace(u2_min, u2_max, n_grid_hf)
        U1_hf_loc, U2_hf_loc = np.meshgrid(u1_hf, u2_hf)
        ax.contour(U1_hf_loc, U2_hf_loc, Z_raw, levels=[0],
                   colors='red', linewidths=2, linestyles=linestyles)

    def print_planche_EFF(g_ot, sigma_func, xt, xt_eff):
        """Planche 2 graphiques cote a cote : critere EFF (gauche) et sigma surrogate (droite).
        Sauvegarde en PNG dans out_dir_eff sans afficher de fenetre."""
        global hf_2d_grid_fixed
        n_added = len(xt_eff)
        # transfert3 T3-2 : coupe 2D generique (slice_def). n_var=2 -> idx_x=0, idx_y=1, fixed={} : identique.
        idx_x, idx_y, fixed = slice_def if slice_def is not None else (0, 1, {})

        # --- Grille commune (calculee une seule fois) ---
        u1 = np.linspace(u1_min, u1_max, n_grid)
        u2 = np.linspace(u2_min, u2_max, n_grid)
        U1, U2 = np.meshgrid(u1, u2)
        grid = np.zeros((n_grid * n_grid, n_var))
        grid[:, idx_x] = U1.ravel()
        grid[:, idx_y] = U2.ravel()
        for _jf, _vf in fixed.items():
            grid[:, _jf] = _vf

        # --- Z_eff, Z_sigma, Z_g (batch vectorise via BLAS multi-thread) ---
        # OPTIM 2026-06-12 : 1 appel batch GEPCK vs 90000 appels sequentiels (gain ~50-200x)
        mu_grid, sigma_grid = _batch_mu_sigma(g_ot, sigma_func, grid)
        Z_eff = _eff_vectorized(mu_grid, sigma_grid, epsilon_factor).reshape(n_grid, n_grid)
        Z_sigma = sigma_grid.reshape(n_grid, n_grid)
        Z_g = mu_grid.reshape(n_grid, n_grid) if g_ot is not None else None

        # --- Contour g=0 HF (transfert3 T3-5b : cascade _get_hf_slice cache 2D -> full -> recalcul) ---
        Z_true = None
        if print_HF:
            Z_true = _get_hf_slice(slice_def, _HF_CACHE_FILE, 'hf_2d_grid_fixed')

        # --- Figure ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        _pce_line = f'\n{_gepck_pce_label}' if _gepck_pce_label else ''
        _fixed_str = ('  [fixe: ' + ', '.join(f'u_{params_names[j]}={v:+.2f}' for j, v in fixed.items()) + ']') if fixed else ''
        fig.suptitle(f'{modele} — N={len(xt)} pts DOE  ({n_added} ajoutes par EFF){_pce_line}{_fixed_str}', fontsize=10)

        def _decorate(ax):
            if Z_g is not None:
                ax.contour(U1, U2, Z_g, levels=[0], colors='cyan', linewidths=2, linestyles='--')
            if Z_true is not None:
                _draw_red_curve(ax, hf_2d_grid_fixed)
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
            ax.set_xlabel(f'u_{params_names[idx_x]}')
            ax.set_ylabel(f'u_{params_names[idx_y]}')
            ax.set_xlim(u1_min, u1_max)
            ax.set_ylim(u2_min, u2_max)
            ax.legend(loc='best', fontsize=9)

        # --- Ax1 : EFF ---
        cf1 = ax1.contourf(U1, U2, Z_eff, levels=20, cmap='viridis', alpha=0.85)
        plt.colorbar(cf1, ax=ax1, label='EFF')
        ax1.set_title('Critere EFF')
        _decorate(ax1)

        # --- Ax2 : sigma ---
        cf2 = ax2.contourf(U1, U2, Z_sigma, levels=20, cmap='plasma', alpha=0.85)
        plt.colorbar(cf2, ax=ax2, label='sigma (ecart-type surrogate)')
        ax2.set_title('Ecart-type surrogate (sigma)')
        _decorate(ax2)

        plt.tight_layout()
        fname = f'EFF_{n_added}points_{timestamp}.png'
        fig.savefig(os.path.join(out_dir_eff, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [EFF visu] -> {fname}", flush=True)

    def print_globalplanche_EFF(xt, yt, all_grad, xt_eff):
        """transfert3 T3-6 : planche globale EFF — 3 colonnes (EFF, sigma, g) x N lignes
        (DOE initial + chaque etape EFF). Refit le surrogate a chaque etape. Coupe slice_def_final.
        Grille HF calculee une seule fois et reutilisee."""
        global hf_2d_grid_fixed_final
        if slice_def_final is None:
            print("[GLOBAL PLANCHE] slice_def_final est None, skip", flush=True)
            return
        idx_x, idx_y, fixed = slice_def_final
        n_eff = len(xt_eff)
        n_steps = n_eff + 1   # DOE initial + chaque point EFF

        # --- Grille commune ---
        ux = np.linspace(u1_min, u1_max, n_grid)
        uy = np.linspace(u2_min, u2_max, n_grid)
        UX, UY = np.meshgrid(ux, uy)
        grid = np.zeros((n_grid * n_grid, n_var))
        grid[:, idx_x] = UX.ravel()
        grid[:, idx_y] = UY.ravel()
        for idx, val in fixed.items():
            grid[:, idx] = val

        # --- Grille HF (une seule fois) ---
        Z_true, UX_hf, UY_hf = None, None, None
        if print_HF:
            ux_hf = np.linspace(u1_min, u1_max, n_grid_hf)
            uy_hf = np.linspace(u2_min, u2_max, n_grid_hf)
            UX_hf, UY_hf = np.meshgrid(ux_hf, uy_hf)
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
                ax.set_xlim(u1_min, u1_max)
                ax.set_ylim(u2_min, u2_max)

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
        idx_x, idx_y, fixed = slice_def if slice_def is not None else (0, 1, {})   # transfert3 T3-3a
        u1 = np.linspace(u1_min, u1_max, n_grid)
        u2 = np.linspace(u2_min, u2_max, n_grid)
        U1, U2 = np.meshgrid(u1, u2)
        grid = np.zeros((n_grid * n_grid, n_var))
        grid[:, idx_x] = U1.ravel()
        grid[:, idx_y] = U2.ravel()
        for _jf, _vf in fixed.items():
            grid[:, _jf] = _vf

        # OPTIM 2026-06-12 : batch vectorise (BLAS multi-thread)
        mu_grid, sigma_grid = _batch_mu_sigma(g_ot, sigma_func, grid)
        Z_eff = _eff_vectorized(mu_grid, sigma_grid, epsilon_factor).reshape(n_grid, n_grid)

        fig, ax = plt.subplots(figsize=(7, 6))
        cf = ax.contourf(U1, U2, Z_eff, levels=20, cmap='viridis', alpha=0.85)
        plt.colorbar(cf, ax=ax, label='EFF')

        # --- Contour g=0 du surrogate (reutilise mu_grid) ---
        if g_ot is not None:
            Z_g = mu_grid.reshape(n_grid, n_grid)
            ax.contour(U1, U2, Z_g, levels=[0], colors='cyan', linewidths=2, linestyles='--', label='surrogate g=0')

        # --- Contour g=0 HF (transfert3 T3-5b : cascade _get_hf_slice) ---
        if print_HF:
            _get_hf_slice(slice_def, _HF_CACHE_FILE, 'hf_2d_grid_fixed')
            _draw_red_curve(ax, hf_2d_grid_fixed)

        # --- Points DOE ---
        if xt is not None:
            ax.scatter(xt[:, idx_x], xt[:, idx_y], c='white', s=40, zorder=5,
                       edgecolors='black', linewidths=0.8, label='DOE')

        # --- Points ajoutes par EFF ---
        if xt_eff is not None and len(xt_eff) > 0:
            xt_eff_arr = np.array(xt_eff)
            ax.scatter(xt_eff_arr[:, idx_x], xt_eff_arr[:, idx_y], c='red', s=80, zorder=6,
                       marker='^', label=f'EFF ({len(xt_eff)} pts)')
            for i, pt in enumerate(xt_eff_arr):
                ax.annotate(str(i + 1), (pt[idx_x], pt[idx_y]), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=8, color='red', zorder=7)

        ax.set_xlabel(f'u_{params_names[idx_x]}')
        ax.set_ylabel(f'u_{params_names[idx_y]}')
        ax.set_xlim(u1_min, u1_max)
        ax.set_ylim(u2_min, u2_max)
        ax.set_title('Critere EFF')
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        plt.show(block=False)

    def print_visu_sigma(g_ot, sigma_func, xt, xt_eff):
        """Carte 2D de l'ecart-type conditionnel du surrogate."""
        global hf_2d_grid_fixed
        idx_x, idx_y, fixed = slice_def if slice_def is not None else (0, 1, {})   # transfert3 T3-3a
        u1 = np.linspace(u1_min, u1_max, n_grid)
        u2 = np.linspace(u2_min, u2_max, n_grid)
        U1, U2 = np.meshgrid(u1, u2)
        grid = np.zeros((n_grid * n_grid, n_var))
        grid[:, idx_x] = U1.ravel()
        grid[:, idx_y] = U2.ravel()
        for _jf, _vf in fixed.items():
            grid[:, _jf] = _vf

        # OPTIM 2026-06-12 : batch vectorise (BLAS multi-thread)
        mu_grid, sigma_grid = _batch_mu_sigma(g_ot, sigma_func, grid)
        Z_sigma = sigma_grid.reshape(n_grid, n_grid)

        fig, ax = plt.subplots(figsize=(7, 6))
        cf = ax.contourf(U1, U2, Z_sigma, levels=20, cmap='plasma', alpha=0.85)
        plt.colorbar(cf, ax=ax, label='sigma (ecart-type surrogate)')

        # --- Contour g=0 du surrogate (reutilise mu_grid) ---
        if g_ot is not None:
            Z_g = mu_grid.reshape(n_grid, n_grid)
            ax.contour(U1, U2, Z_g, levels=[0], colors='cyan', linewidths=2, linestyles='--', label='surrogate g=0')

        # --- Contour g=0 HF (transfert3 T3-5b : cascade _get_hf_slice) ---
        if print_HF:
            _get_hf_slice(slice_def, _HF_CACHE_FILE, 'hf_2d_grid_fixed')
            _draw_red_curve(ax, hf_2d_grid_fixed)

        if xt is not None:
            ax.scatter(xt[:, idx_x], xt[:, idx_y], c='white', s=40, zorder=5,
                       edgecolors='black', linewidths=0.8, label='DOE')

        if xt_eff is not None and len(xt_eff) > 0:
            xt_eff_arr = np.array(xt_eff)
            ax.scatter(xt_eff_arr[:, idx_x], xt_eff_arr[:, idx_y], c='red', s=80, zorder=6,
                       marker='^', label=f'EFF ({len(xt_eff)} pts)')
            for i, pt in enumerate(xt_eff_arr):
                ax.annotate(str(i + 1), (pt[idx_x], pt[idx_y]), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=8, color='red', zorder=7)

        ax.set_xlabel(f'u_{params_names[idx_x]}')
        ax.set_ylabel(f'u_{params_names[idx_y]}')
        ax.set_xlim(u1_min, u1_max)
        ax.set_ylim(u2_min, u2_max)
        ax.set_title('Ecart-type surrogate (sigma)')
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        plt.show(block=False)

    def print_results(best_result, g_ot):
        _point_log_phase[0] = "USTAR"   # les run_HF de verif au point de conception sont taggues USTAR
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

        # --- Erreur FOSM ---
        if g_ot is not None:
            _, grad_HF_U_star, _ = run_HF(u_star)
            for i, p in enumerate(params_names):
                print(f"dg/du_{p} en u* (HF@u*GEK) = {grad_HF_U_star[i]:.6f}", flush=True)
            u0             = ot.Point([0.0] * n_var)
            # run_HF([0,0]) est INDEPENDANT du mode -> on le calcule 1 seule fois et on le
            # reutilise pour le FOSM de tous les modes (evite 1 SOCP complet redondant/run).
            if _fosm_u0_cache[0] is None:
                g0_HF, grad_HF_U0, _ = run_HF(u0)
                _fosm_u0_cache[0] = (g0_HF, grad_HF_U0)
            else:
                g0_HF, grad_HF_U0 = _fosm_u0_cache[0]
                print("  [FOSM] run_HF([0,0]) reutilise du cache (pas de SOCP redondant)", flush=True)
            u_FOSM         = grad_HF_U0 * (-g0_HF / grad_HF_U0.normSquare())
            print(f"u* FOSM (HF) = {[round(v, 4) for v in u_FOSM]}", flush=True)
            print(f"Erreur FOSM  = {(u_FOSM - u_star).norm() / u_star.norm():.4f}", flush=True)


    def print_visu(best_result, best_sp, xt, g_ot, modes, xt_eff):
        global u1_min, u1_max, u2_min, u2_max, hf_2d_grid_fixed, slice_def_final
        # transfert3 T3-3b : coupe finale via slice_def_final (auto-calculee depuis les importances FORM si None)
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
        u1 = np.linspace(u1_min, u1_max, n_grid)
        u2 = np.linspace(u2_min, u2_max, n_grid)
        U1, U2 = np.meshgrid(u1, u2)
        grid = np.zeros((n_grid * n_grid, n_var))
        grid[:, idx_x] = U1.ravel()
        grid[:, idx_y] = U2.ravel()
        for _jf, _vf in fixed.items():
            grid[:, _jf] = _vf

        fig, ax = plt.subplots(figsize=(7, 6))

        # --- Fond coloré : GEK en priorité, sinon KRG ---
        if do_GEK:
            grid_ot = ot.Sample(grid.tolist())
            Z_gek = np.array(g_ot(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_gek, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (GEK)')
            ax.contour(U1, U2, Z_gek, levels=[0], colors='blue', linewidths=2)
        elif do_KRG:
            grid_ot = ot.Sample(grid.tolist())
            Z_krg = np.array(g_ot(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_krg, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (KRG)')
        elif do_PCKRG:
            grid_ot = ot.Sample(grid.tolist())
            Z_pckrg = np.array(g_ot(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_pckrg, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (PCKRG)')
            ax.contour(U1, U2, Z_pckrg, levels=[0], colors='blue', linewidths=2)
        elif do_old_GEPCK:
            grid_ot = ot.Sample(grid.tolist())
            Z_gepck = np.array(g_ot(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_gepck, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (old GEPCK)')
            ax.contour(U1, U2, Z_gepck, levels=[0], colors='blue', linewidths=2)
        elif do_GEPCK:
            grid_ot = ot.Sample(grid.tolist())
            Z_gepck = np.array(g_ot(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_gepck, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (GEPCK)')
            ax.contour(U1, U2, Z_gepck, levels=[0], colors='blue', linewidths=2)

        # --- Contour KRG ---
        if do_KRG:
            grid_ot = ot.Sample(grid.tolist())
            Z_krg = np.array(g_ot(grid_ot))[:, 0].reshape(n_grid, n_grid)
            ax.contour(U1, U2, Z_krg, levels=[0], colors='purple', linewidths=2, linestyles=':')

        # --- Contour HF grossier (transfert3 T3-5b : coupe FINALE via _get_hf_slice) ---
        if print_HF:
            _get_hf_slice(slice_def_final, _HF_CACHE_FILE_FINAL, 'hf_2d_grid_fixed_final')
            _draw_red_curve(ax, hf_2d_grid_fixed_final, linestyles='--')

        # --- LS analytique (depuis flexion_claude) ---
        if print_ana:
            calc = flexion_claude()
            u1_lim_a = calc.u1_lim_plast
            u2_lim_a = calc.u2p_LS(u1_lim_a)
            u1_g_a = np.linspace(u1_lim_a, u1_max, n_grid)
            u2_g_a = np.array([calc.u2p_LS(u) for u in u1_g_a])
            ax.plot(u1_g_a, u2_g_a, color='green', linestyle='-.', linewidth=2)
            ax.plot([u1_lim_a, u1_lim_a], [u2_lim_a, u2_max],
                    color='green', linestyle='-.', linewidth=2)
            ax.plot(u1_lim_a, u2_lim_a, 'ko', ms=6, zorder=6)

        # --- Points ---
        if xt is not None:
            ax.scatter(xt[:, idx_x], xt[:, idx_y], c='black', s=30, zorder=5, label='DOE')

        if xt_eff is not None and len(xt_eff) > 0:
            xt_eff_arr = np.array(xt_eff)
            ax.scatter(xt_eff_arr[:, idx_x], xt_eff_arr[:, idx_y], c='red', s=60, zorder=6,
                       marker='^', label=f'EFF ({len(xt_eff)} pts)')

        ax.scatter(0, 0, c='orange', s=100, zorder=6, marker='P', label='[0, 0]')

        if best_sp is not None:
            ax.scatter(best_sp[idx_x], best_sp[idx_y], c='cyan', s=100, zorder=7, marker='D',
                    label='point de depart best')

        if best_result is not None:
            u_star = np.array(best_result.getStandardSpaceDesignPoint())
            ax.scatter(u_star[idx_x], u_star[idx_y], c='gold', s=200, zorder=8, marker='*',
                    label=f'u*1 [{u_star[idx_x]:.2f},{u_star[idx_y]:.2f}] beta={best_result.getHasoferReliabilityIndex():.3f}')

        if len(modes) > 0:
            for k, mode in enumerate(modes[1:], start=2):
                u_m = np.array(mode.getStandardSpaceDesignPoint())
                ax.scatter(u_m[idx_x], u_m[idx_y], c='magenta', s=200, zorder=8, marker='*',
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
                    ax.quiver(sp_f[idx_x], sp_f[idx_y], ng[idx_x], ng[idx_y], color=col,
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
                # polyligne
                ax.plot(pts[:, idx_x], pts[:, idx_y], '-', color=col, alpha=0.5, linewidth=1.2)
                # points intermediaires
                ax.scatter(pts[1:-1, idx_x], pts[1:-1, idx_y], c=col, s=12, zorder=7, alpha=0.5)
                # depart et arrivee
                ax.scatter(pts[0, idx_x],  pts[0, idx_y],  c=col, s=60,  zorder=8, marker='o')
                ax.scatter(pts[-1, idx_x], pts[-1, idx_y], c=col, s=150, zorder=8, marker='*')
                # petites fleches -grad normalise (longueur 0.3)
                for pt, g in zip(pts, grds):
                    ng = np.array(g)
                    nrm = np.linalg.norm(ng)
                    if nrm > 0:
                        ng = -ng / nrm * 0.3
                        ax.annotate('', xy=(pt[idx_x]+ng[idx_x], pt[idx_y]+ng[idx_y]), xytext=(pt[idx_x], pt[idx_y]),
                                    arrowprops=dict(arrowstyle='->', color=col, lw=0.7))
            # zoom sur les trajectoires : mise a jour des variables globales
            all_pts_arr = np.vstack(all_pts)
            margin = 1.0
            u1_min = float(all_pts_arr[:, idx_x].min()) - margin
            u1_max = float(all_pts_arr[:, idx_x].max()) + margin
            u2_min = float(all_pts_arr[:, idx_y].min()) - margin
            u2_max = float(all_pts_arr[:, idx_y].max()) + margin

        # --- Légende contours ---
        legend_lines = []
        
        if do_KRG:
            legend_lines.append(Line2D([0], [0], color='purple', linestyle=':',  linewidth=2, label='g=0 KRG'))
        if do_GEK:
            legend_lines.append(Line2D([0], [0], color='blue',   linestyle='-',  linewidth=2, label='g=0 GEKPLS'))
        if do_PCKRG:
            legend_lines.append(Line2D([0], [0], color='blue',   linestyle='-',  linewidth=2, label='g=0 PCKRG'))
        if do_old_GEPCK:
            legend_lines.append(Line2D([0], [0], color='blue',   linestyle='-',  linewidth=2, label='g=0 old GEPCK'))
        if do_GEPCK:
            legend_lines.append(Line2D([0], [0], color='blue',   linestyle='-',  linewidth=2, label='g=0 GEPCK'))
        if print_HF:
            legend_lines.append(Line2D([0], [0], color='red',    linestyle='--', linewidth=2, label='g=0 HF'))
        if print_ana:
            legend_lines.append(Line2D([0], [0], color='green',  linestyle='-.', linewidth=2, label='g=0 ana'))

        ax.legend(handles=ax.legend().legend_handles + legend_lines)

        ax.set_xlabel(f'u_{params_names[idx_x]}')
        ax.set_ylabel(f'u_{params_names[idx_y]}')
        ax.set_xlim(u1_min, u1_max)
        ax.set_ylim(u2_min, u2_max)
        ax.set_title('FORM et etat limite g=0')
        plt.tight_layout()
        fname = f'visu{modele}_{timestamp}.png'
        fig.savefig(os.path.join(out_dir_eff, fname), dpi=150, bbox_inches='tight')
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

        # --- Surface analytique g_ana (flexion_claude) ---
        if print_ana:
            calc = flexion_claude()
            u1_a = np.linspace(u1_min, u1_max, n_grid)
            u2_a = np.linspace(u2_min, u2_max, n_grid)
            U1_a, U2_a = np.meshgrid(u1_a, u2_a)
            Z_ana = np.array([calc.g(u1, u2)
                              for u1, u2 in zip(U1_a.ravel(), U2_a.ravel())]
                             ).reshape(n_grid, n_grid)
            ax.plot_surface(U1_a, U2_a, Z_ana, color='blue', alpha=0.3, label='g_ana')
            ax.contour(U1_a, U2_a, Z_ana, levels=[0], colors='green', linewidths=2,
                       zdir='z', offset=float(Z.min()))
            ax.contour(U1_a, U2_a, Z_ana, levels=[0], colors='green', linewidths=2)

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
        ax.set_xlabel('u1 (fc)')
        ax.set_ylabel('u2 (fy)')
        ax.set_zlabel('g_HF')
        ax.set_title(f'Surface g_HF — {n_grid_hf}x{n_grid_hf} pts HF')
        ax.legend()
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------------------- #
    # FONCTION IS POST-FORM                                                       #
    def run_IS(modes, event):
        """
        Importance Sampling post-FORM sur le surrogate.
        Distribution instrumentale : mixture de N(u*_i) pondérée par les Pf_FORM_i.
        Mono-modal : N simple centré sur u*.
        Retourne un ProbabilitySimulationResult.
        """
        if len(modes) == 1:
            g_imp = ot.Normal(n_var)
            g_imp.setMu(modes[0].getStandardSpaceDesignPoint())
            importance_dist = g_imp
        else:
            gaussians  = []
            pf_weights = []
            for m in modes:
                g_i = ot.Normal(n_var)
                g_i.setMu(m.getStandardSpaceDesignPoint())
                gaussians.append(g_i)
                pf_weights.append(m.getEventProbability())
            importance_dist = ot.Mixture(gaussians, pf_weights)

        experiment = ot.ImportanceSamplingExperiment(importance_dist, n_IS)
        std_event  = ot.StandardEvent(event)
        algo = ot.ProbabilitySimulationAlgorithm(std_event, experiment)
        algo.setMaximumCoefficientOfVariation(cov_IS)
        algo.setMaximumOuterSampling(n_IS)
        algo.run()
        return algo.getResult()

    def print_results_IS(result_IS):
        pf   = result_IS.getProbabilityEstimate()
        cov  = result_IS.getCoefficientOfVariation()
        ci   = result_IS.getConfidenceLength(0.95)
        beta = float(-ot.Normal().computeQuantile(pf)[0])
        print(f"=== Importance Sampling ===", flush=True)
        print(f"  Pf_IS   = {pf:.4e}", flush=True)
        print(f"  beta_IS = {beta:.4f}", flush=True)
        print(f"  COV     = {cov:.4f}", flush=True)
        print(f"  IC 95%  = [{pf - ci/2:.4e}, {pf + ci/2:.4e}]", flush=True)
        print(f"  N_IS    = {result_IS.getOuterSampling()}", flush=True)

    """
    DEBUT DE CODE
    """
    update_degree(n0)
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

    # ===================== MODE WORKER DOE (multiprocessing) =====================
    # Lance par run_DOE_parallel : ne calcule QUE ses points DOE sur sa copie .ds isolee
    # (modelname pointe deja sur la copie via _DOE_WORKER_MODELNAME), ecrit g + sensibilites
    # dans le JSON de sortie, puis sort. Reutilise run_one_SOL tel quel.
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
    # =============================================================================

    if restart_enrich_only:
        # === MODE REPRISE : charger le dump, injecter, preparer le re-enrichissement ===
        # (xt/yt/all_grad non-None -> init_g_ot saute build_DOE et refit le surrogate ; la courbe
        #  rouge et le degre viennent du dump -> aucun SOCP DOE ni grille HF recalcules.)
        _rs = json.load(open(_RESTART_STATE_FILE))
        xt = np.array(_rs['xt'], float); yt = np.array(_rs['yt'], float); all_grad = np.array(_rs['all_grad'], float)
        _restart_xt_eff = [np.array(p, float) for p in _rs['xt_eff']]
        if _rs.get('max_degree') is not None:
            max_degree = int(_rs['max_degree'])
        hf_2d_grid_fixed = _rs.get('hf_2d_grid')           # courbe rouge reutilisee -> 0 SOCP HF
        _eff_history_EFF     = list(_rs['hist_EFF'])
        _eff_history_BB      = list(_rs['hist_BB'])
        _eff_history_BS      = list(_rs['hist_BS'])
        _eff_history_theta   = [list(t) for t in _rs['hist_theta']]
        _eff_history_beta_IS = list(_rs['hist_beta_IS'])
        _enrich_round     = int(_rs.get('enrich_round', 0)) + 1
        _round_sizes_prev = list(_rs.get('round_sizes', [int(len(xt))]))
        _point_log_round[0] = _enrich_round
        with open(_POINT_LOG_FILE, "a") as _pf:            # APPEND + ligne-marqueur (pas de reset)
            _pf.write(json.dumps({"phase": "_RESTART", "round": _enrich_round,
                                  "loaded": int(len(xt)), "n_eff_loaded": len(_restart_xt_eff)}) + "\n")
        print(f"[RESTART] round {_enrich_round} : charge {len(xt)} pts ({len(_restart_xt_eff)} EFF), "
              f"max_degree={max_degree}, courbe_rouge_reutilisee={hf_2d_grid_fixed is not None} "
              f"-> enrichissement selon critere EFF (cap n_max_EFF_points)", flush=True)
    else:
        try:   # reset du log incremental par point (1 seule fois, process principal)
            open(_POINT_LOG_FILE, "w").close()
            print(f"[POINT LOG] reset -> {_POINT_LOG_FILE} (DOE/HF/EFF/USTAR appendus au fil du calcul)", flush=True)
        except Exception as _e:
            print(f"[POINT LOG] reset echoue ({type(_e).__name__}: {_e})", flush=True)

    _t_log("##### PHASE: init_g_ot (DOE + GEPCK fit) START #####")
    _t0_phase = time.perf_counter()
    g_ot, sigma_func, xt, yt, all_grad = init_g_ot(g_ot, sigma_func, xt, yt, all_grad)
    _t_log("##### PHASE: init_g_ot END #####", _t0_phase)
    if print_HF and print_fullHF and n_var <= 3:   # transfert3 T3-5 : grille HF full (n_var-D) si demandee
        _compute_hf_grid_full()
    if do_EFF:
        _t_log("##### PHASE: print_planche_EFF (avant EFF) START #####")
        _t0_phase = time.perf_counter()
        print_planche_EFF(g_ot, sigma_func, xt, [])
        _t_log("##### PHASE: print_planche_EFF (avant EFF) END #####", _t0_phase)

        _t_log("##### PHASE: run_EFF (enrichissement adaptatif) START #####")
        _t0_phase = time.perf_counter()
        g_ot, sigma_func, xt, yt, all_grad, xt_eff = run_EFF(g_ot, sigma_func, xt, yt, all_grad)
        _t_log(f"##### PHASE: run_EFF END (n_added={len(xt_eff)}) #####", _t0_phase)

        if not print_EFF_progres:
            # OPTIM 2026-06-12 : skip redondant si print_EFF_progres=True
            # car run_EFF a deja sauve EFF_Npoints_*.png pour le N final
            _t_log("##### PHASE: print_planche_EFF (apres EFF) START #####")
            _t0_phase = time.perf_counter()
            print_planche_EFF(g_ot, sigma_func, xt, xt_eff)
            _t_log("##### PHASE: print_planche_EFF (apres EFF) END #####", _t0_phase)
        else:
            _t_log("##### PHASE: print_planche_EFF (apres EFF) SKIPPED (deja sauve par run_EFF) #####")

        _t_log("##### PHASE: print_EFF_graphs START #####")
        _t0_phase = time.perf_counter()
        print_EFF_graphs()
        _t_log("##### PHASE: print_EFF_graphs END #####", _t0_phase)

    _t_log("##### PHASE: init_FORM START #####")
    _t0_phase = time.perf_counter()
    event, g_ot, sigma_func, xt, yt, all_grad = init_FORM(g_ot, sigma_func, xt, yt, all_grad)
    _t_log("##### PHASE: init_FORM END #####", _t0_phase)

    if event is None:
        if best_sol_modes_fixed is not None:
            print_visu(None, None, None, None, [], None)
            sys.exit(0)
        print('Aucune branche active', flush=True)
        sys.exit(1)

    _t_log("##### PHASE: FORM_all_modes (multistart) START #####")
    _t0_phase = time.perf_counter()
    if do_warmstart:
        starting_points = np.array([[0.0] * n_var])
        modes, best_sps = FORM_all_modes(starting_points, tol_all_modes, event) #FORM simple avec event créé
        modes, best_sps = FORM_warm_start(modes, best_sps, g_ot, sigma_func, xt, yt, all_grad) #warm_start puis FORM multistart avec event warm
    else:
        starting_points = np.vstack([xt, [[0.0] * n_var]]) if do_multistart else np.array([[0.0] * n_var])
        modes, best_sps = FORM_all_modes(starting_points, tol_all_modes, event)
    _t_log(f"##### PHASE: FORM_all_modes END (n_modes={len(modes)}, n_starts={len(starting_points)}) #####", _t0_phase)

    best_result = modes[0] if modes else None
    best_sp     = best_sps[0] if best_sps else None
    if best_result is None:
        print('Aucun FORM ne marche.', flush=True)
        sys.exit(1)
    if len(modes)>1:
        print('On a trouvé plus de 1 mode! Les résultats du mode 2 sont:')
        print_results(modes[1], g_ot)
        print('Les résultats du mode 1 sont : ')
    _t_log("##### PHASE: print_results (HF gradient au u*) START #####")
    _t0_phase = time.perf_counter()
    print_results(best_result, g_ot)
    _t_log("##### PHASE: print_results END #####", _t0_phase)

    result_IS = None
    if do_IS and modes:
        _t_log("##### PHASE: run_IS START #####")
        _t0_phase = time.perf_counter()
        result_IS = run_IS(modes, event)
        _t_log("##### PHASE: run_IS END #####", _t0_phase)
        print_results_IS(result_IS)

    _t_log("##### PHASE: print_visu (final PNG) START #####")
    _t0_phase = time.perf_counter()
    print_visu(best_result, best_sp, xt, g_ot, modes, xt_eff)
    _t_log("##### PHASE: print_visu END #####", _t0_phase)

    if do_EFF and xt_eff:   # transfert3 T3-6 : planche globale evolution EFF
        _t_log("##### PHASE: print_globalplanche_EFF START #####")
        _t0_phase = time.perf_counter()
        print_globalplanche_EFF(xt, yt, all_grad, xt_eff)
        _t_log("##### PHASE: print_globalplanche_EFF END #####", _t0_phase)

    # --- DUMP RESTART : tout sauver pour reprise/regeneration ulterieure (relecture a coder ensuite) ---
    _t_log("##### PHASE: _save_restart_state (dump complet) START #####")
    _t0_phase = time.perf_counter()
    _save_restart_state(xt, yt, all_grad, xt_eff, best_result, best_sp, modes, result_IS)
    _t_log("##### PHASE: _save_restart_state END #####", _t0_phase)
    _t_log("########## TOTAL CALC TIME ##########")
