# -*- coding: utf-8 -*-
# PREUVE DE RELECTURE : regenere le DERNIER graphe (print_visu, recap global)
# UNIQUEMENT a partir de restart_state.json (aucun calcul SOCP, aucune relance).
# Replique fidelement la branche GEPCK de print_visu de l'AC.
import os, sys, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, r'C:\workspace\fiabilite')
import sys; sys.path.insert(0, r"C:\workspaceiabilite\_lib")  # branche* deplaces dans _lib
from branche1 import fit_gepck, predict_gepck

DS   = r'C:\workspace\storage\admin\Moulin_Blanc\Calcul_fiabilite_13k_2fy_membrure_inf_tablier.ds'
OUT  = r'C:\workspace\fiabilite\output\png_EFF_moulin_blanc\png_EFF_1806_2224'
os.makedirs(OUT, exist_ok=True)

# --- constantes identiques a l'AC ---
u1_min, u1_max, u2_min, u2_max, n_grid, n_grid_hf, n_var = -7.5, 7.5, -7.5, 7.5, 300, 7, 2

# --- 1) charge le dump ---
d = json.load(open(os.path.join(DS, 'restart_state.json')))
xt = np.array(d['xt'], float); yt = np.array(d['yt'], float); ag = np.array(d['all_grad'], float)
xt_eff = np.array(d['xt_eff'], float)
best_sp = np.array(d['best_sp'], float) if d['best_sp'] is not None else None
modes   = d['modes']
hf_cache = d['hf_2d_grid']

# --- 2) refit GEPCK (comme init_g_ot) ---
Y_aug = np.concatenate([yt.flatten()] + [ag[:, j] for j in range(n_var)])
marginals = [{'Type':'Gaussian','Parameters':[0.0,1.0]}, {'Type':'Gaussian','Parameters':[0.0,1.0]}]
copula    = {'Type':'Independent','Parameters':np.eye(n_var)}
opts      = {'Mode':'optimal','PCE':{'Degree':list(range(1, d['max_degree']+1)),'Method':'LARS'}}
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fm = fit_gepck(xt, Y_aug, opts, marginals, copula)
print(f"refit OK : theta={[round(t,4) for t in fm['Kriging'][0]['theta']]}  (dump: {[round(t,4) for t in d['hist_theta'][-1]]})")

# --- 3) grille + evaluation surrogate (batch, comme _batch_mu_sigma) ---
u1 = np.linspace(u1_min, u1_max, n_grid); u2 = np.linspace(u2_min, u2_max, n_grid)
U1, U2 = np.meshgrid(u1, u2)
grid = np.column_stack([U1.ravel(), U2.ravel()])
Z_g = predict_gepck(fm, grid)[:, 0].reshape(n_grid, n_grid)
print(f"surrogate evalue sur grille {n_grid}x{n_grid}  (g range [{Z_g.min():+.2f},{Z_g.max():+.2f}])")

# --- 4) figure : replique exacte de la branche do_GEPCK de print_visu ---
fig, ax = plt.subplots(figsize=(7, 6))
cf = ax.contourf(U1, U2, Z_g, levels=20, cmap='RdYlGn', alpha=0.6)
plt.colorbar(cf, ax=ax, label='g (GEPCK)')
ax.contour(U1, U2, Z_g, levels=[0], colors='blue', linewidths=2)            # surrogate g=0

# courbe rouge HF (replique _draw_red_curve) depuis le dump
if hf_cache is not None and 'Z' in hf_cache:
    Z_raw = np.array(hf_cache['Z'], float)
    u1h = np.linspace(u1_min, u1_max, n_grid_hf); u2h = np.linspace(u2_min, u2_max, n_grid_hf)
    U1h, U2h = np.meshgrid(u1h, u2h)
    ax.contour(U1h, U2h, Z_raw, levels=[0], colors='red', linewidths=2, linestyles='--')  # courbe rouge HF

ax.scatter(xt[:, 0], xt[:, 1], c='black', s=30, zorder=5, label='DOE')
if len(xt_eff) > 0:
    ax.scatter(xt_eff[:, 0], xt_eff[:, 1], c='red', s=60, zorder=6, marker='^', label=f'EFF ({len(xt_eff)} pts)')
ax.scatter(0, 0, c='orange', s=100, zorder=6, marker='P', label='[0, 0]')
if best_sp is not None:
    ax.scatter(best_sp[0], best_sp[1], c='cyan', s=100, zorder=7, marker='D', label='point de depart best')
# u*1 (mode dominant) en or, u*k (k>=2) en magenta -- depuis le dump
u1s = modes[0]['u_star']
ax.scatter(u1s[0], u1s[1], c='gold', s=200, zorder=8, marker='*',
           label=f"u*1 [{u1s[0]:.2f},{u1s[1]:.2f}] beta={modes[0]['beta']:.3f}")
for k, m in enumerate(modes[1:], start=2):
    um = m['u_star']
    ax.scatter(um[0], um[1], c='magenta', s=200, zorder=8, marker='*',
               label=f"u*{k} [{um[0]:.2f},{um[1]:.2f}] beta={m['beta']:.3f}")

from scipy.stats import norm as _norm
_Pf = d['IS']['Pf'] if d['IS'] else None
_beta = -_norm.ppf(_Pf) if _Pf else float('nan')
ax.set_xlabel('u1'); ax.set_ylabel('u2'); ax.set_xlim(u1_min, u1_max); ax.set_ylim(u2_min, u2_max)
ax.legend(loc='best', fontsize=8)
ax.set_title(f"GEPCK recap (REGENERE depuis restart_state.json)\nbeta_IS={_beta:.3f}  Pf={_Pf:.2e}", fontsize=9)

fname = os.path.join(OUT, 'visuGEPCK_RELECTURE_from_dump.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nPNG REGENERE -> {fname}")
print(f"taille: {os.path.getsize(fname)} octets")
