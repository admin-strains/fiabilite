"""
Parametres par defaut du code de fiabilite.
Ne pas modifier sauf cas particulier (voir guide d'utilisation).
"""
from config_utilisateur import n_workers_DOE

# ======================================================================
# Options generales
# ======================================================================
do_EFF = True                        # enrichissement progressif
do_IS = True                         # calcul de la proba globale

# ======================================================================
# FORM
# ======================================================================
n_max_FORM = 50
do_multistart = True                 # FORM depuis n0 points + [0,0]
do_warmstart = False                 # si FORM ne converge pas, repart du best pt
start_from_LHS = False               # FORM depuis un LHS frais de n_sp points
n_sp = 200                           # taille du LHS si start_from_LHS=True
do_FORM_filter = True                # rejeter les u* FORM hors eff_bounds avant DBSCAN

# ======================================================================
# IS
# ======================================================================
n_IS = 10000                         # taille echantillon IS
cov_IS = 0.05                        # critere d'arret COV

# ======================================================================
# PCE
# ======================================================================
q = 0.75                             # tri base poly candidats
max_degree = 2                       # degre max base candidats

# ======================================================================
# EFF
# ======================================================================
epsilon_factor = 2                   # eps = epsilon_factor * sigma
n_NLopt_EFF = 30                     # budget evaluations NLopt GN_DIRECT
n_batch_EFF = n_workers_DOE          # points EFF par iteration (1 = sequentiel)
n_max_EFF_points = 200               # plafond de points EFF
EFF_criteria = 'BS'                  # critere : 'BB' | 'BS' | 'both' | 'at_least_one'
tol_EFF = 0.002                      # arret par max(EFF) < tol_EFF
tol_BB = 0.05                        # arret BB
tol_BS = 0.01                        # arret BS

# ======================================================================
# Affichage
# ======================================================================
u1_max = 7.5
u2_max = 7.5
u1_min = -7.5
u2_min = -7.5
n_grid = 300
print_EFF_progres = True             # prints debug EFF a chaque iter
print_gepck_calls = False            # log chaque appel _exec GEPCK
print_Pf = False                     # calcule Pf_IS a chaque iter EFF
print_DOE = True
print_3D = False
print_grad_sp = False

# ======================================================================
# Historique
# ======================================================================
save_history = False                 # copie le dsmed dans SOCP_history/
