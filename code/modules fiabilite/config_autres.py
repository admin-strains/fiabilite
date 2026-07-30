"""
Variables derivees et valeurs par defaut internes.
L'utilisateur n'a pas besoin de modifier ce fichier.
"""
from config_utilisateur import modele
from config_pardefaut import do_EFF, do_IS

# --- Flags derives du type de modele ---
do_KRG       = (modele == 'KRG')
do_GEK       = (modele == 'GEK')
do_HF        = (modele == 'HF')
do_PCKRG     = (modele == 'PCKRG')
do_old_GEPCK = (modele == 'old_GEPCK')
do_GEPCK     = (modele == 'GEPCK')
do_IS        = do_IS and modele != 'HF'       # IS impraticable en HF
do_EFF       = do_EFF and modele != 'HF'      # EFF impraticable en HF

# --- Resultats fixes (pour visu seule sans recalcul) ---
hf_3d_grid_fixed = None
do_custom_hf = True
hf_custom_points = None
sol_modes_fixed = None
best_sol_modes_fixed = None
grad_sp_fixed = None
traj_runs_fixed = None
