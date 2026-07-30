"""
Configuration utilisateur du code de fiabilite.
Modifier les variables ci-dessous pour adapter le code a votre modele.
Voir le guide d'utilisation pour plus de details.
"""
from lois import loi_fy, loi_fc, loi_F_permanente, loi_F_exploitation, loi_F_intermittente, loi_uni_approx

# ======================================================================
# 1. Chemins et nom du modele
# ======================================================================
modelname = "Calcul_fiabilite_G+LM1_13k_2fy_membrure_inf_diagonal"
storage = "C:\\workspace\\storage\\admin\\Moulin_Blanc\\"
path_dir = r"C:\_workingDir\dir_fiabilite"

# ======================================================================
# 2. Variables globales principales
# ======================================================================
modele = 'GEPCK'                    # type de modele - options: 'GEPCK', 'PCKRG', 'KRG', 'GEK', 'HF'
n0 = 5                              # nombre de points du DOE initial
tol_FORM = 0.05                     # precision acceptee par FORM
tol_all_modes = 0.9                 # distance DBSCAN entre deux modes
n_workers_DOE = 6                   # nb de SOCP DOE en parallele

# ======================================================================
# 3. PARAM_CONFIG : catalogue des variables aleatoires
# ======================================================================
PARAM_CONFIG_CAD = {}
PARAM_CONFIG_LOAD = {
    's_convoi': {'sens': {"param": "LIVE_LOAD", "load_case": "LC_convoi",
                          "axis": "position", "region_key": "s_convoi"},
                 'loi': loi_uni_approx, 'args': (0.0, 1.0, 0.15)},
    'q':        {'sens': {"param": "LIVE_LOAD", "load_case": "LC_convoi", "region_key": "q"},
                 'loi': loi_F_permanente, 'args': (0.1, 0.30)},
}

# --- Derivees de PARAM_CONFIG (pour slice_def) ---
params_names = list(PARAM_CONFIG_LOAD.keys()) + list(PARAM_CONFIG_CAD.keys())
n_var = len(params_names)

# ======================================================================
# 4. Affichage : coupes 2D et bornes EFF
# ======================================================================
slice_def = (0, 1, {i: 0.0 for i in range(n_var) if i > 1})
slice_def_final = None

eff_bounds_min = [-2.0, -3.32]      # bornes inf [s_convoi, fy1]
eff_bounds_max = [+2.0, +7.5]       # bornes sup [s_convoi, fy1]

# ======================================================================
# 5. Affichage grille HF
# ======================================================================
print_HF = True
print_fullHF = False                # Tres deconseille : grille en R^d
n_grid_hf = 7                       # nombre de points par axe

# ======================================================================
# 6. Caches et reprise
# ======================================================================
config_is_identical = True           # True = reutiliser les caches si presents
restart_enrich_only = False          # True = charger restart_state.json et continuer
