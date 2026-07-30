"""
Variables d'etat partagees entre modules.
Ces variables sont modifiees pendant l'execution du code.

Pour les immuables (str, int, None) : utiliser `import etat` puis `etat.var = ...`
Pour les mutables (listes) : `from etat import var` puis `var.append(...)` ou `var[0] = ...`
"""

# --- Labels et LOO du surrogate (modifies par init_g_ot, lus par print_planche_EFF) ---
_gepck_pce_label = ""
_gepck_loo       = None

# --- Historiques EFF (modifies par run_EFF/init_g_ot, lus par print_*/save_restart) ---
_eff_history_EFF      = []
_eff_history_BB       = []
_eff_history_BS       = []
_eff_history_theta    = []
_eff_history_Pf       = []
_eff_history_beta_IS  = []

# --- Caches et compteurs FOSM / point log ---
_fosm_u0_cache   = [None]
_point_log_phase  = ["?"]
_point_log_round  = [0]

# --- Restart / enrichissement ---
_enrich_round     = 0
_round_sizes_prev = []
_restart_xt_eff   = []

# --- Compteurs STRAINS ---
_socp_call_counter = [0]
_run_HF_count      = [0]

# --- Grilles HF en memoire ---
hf_2d_grid_fixed       = None
hf_2d_grid_fixed_final = None
_hf_grid_full          = [None]
_hf_grid_full_axes     = [None]
_hf_custom_result      = [None]

# --- Coupe finale (modifiee par print_visu) ---
slice_def_final_state = None    # renomme pour eviter conflit avec slice_def_final de config_utilisateur

# --- Constantes d'execution (fixees au demarrage par AC3, lues par visu) ---
timestamp   = ""
out_dir_eff = ""