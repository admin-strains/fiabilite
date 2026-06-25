# -*- coding: utf-8 -*-
# PREUVE RELECTURE (2e graphe final) : regenere le graphe TOLERANCE/CONVERGENCE
# (print_EFF_graphs : EFF vs iter, criteres BB/BS, theta Kriging) UNIQUEMENT depuis
# restart_state.json (historiques dumpes). Replique fidelement print_EFF_graphs.
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DS  = r'C:\workspace\storage\admin\Moulin_Blanc\Calcul_fiabilite_13k_2fy_membrure_inf_tablier.ds'
OUT = r'C:\workspace\fiabilite\output\png_EFF_moulin_blanc\png_EFF_1806_2224'
tol_EFF, tol_BB, tol_BS = 1e-3, 0.01, 0.01
params_names = ['fy1', 'fy2']

d = json.load(open(os.path.join(DS, 'restart_state.json')))
hist_EFF   = d['hist_EFF']
hist_BB    = d['hist_BB']
hist_BS    = d['hist_BS']
hist_theta = d['hist_theta']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
_clip = 1e-12

# --- Subplot 1 : EFF vs iterations ---
ax = axes[0]
x_eff = list(range(len(hist_EFF)))
vals_eff = [max(abs(v), _clip) for v in hist_EFF]
ax.semilogy(x_eff, vals_eff, 'b-o', ms=4, lw=1.2, label='EFF(u_opt)')
ax.axhline(tol_EFF, color='orange', ls='--', lw=1, label=f'tol_EFF={tol_EFF:.1e}')
ax.set_xlabel('Iteration EFF'); ax.set_ylabel('EFF (echelle log)')
ax.set_title('Convergence EFF'); ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.4)

# --- Subplot 2 : BB / BS vs iterations ---
ax = axes[1]
if hist_BB:
    x_bb = list(range(1, len(hist_BB) + 1))
    vals_bb = [max(v, _clip) if v is not None else np.nan for v in hist_BB]
    ax.semilogy(x_bb, vals_bb, 'g-o', ms=4, lw=1.2, label='BB')
    ax.axhline(tol_BB, color='g', ls='--', lw=0.8, label=f'tol_BB={tol_BB:.1e}')
if hist_BS:
    x_bs = list(range(1, len(hist_BS) + 1))
    vals_bs = [max(v, _clip) if v is not None else np.nan for v in hist_BS]
    ax.semilogy(x_bs, vals_bs, 'r-s', ms=4, lw=1.2, label='BS')
    ax.axhline(tol_BS, color='r', ls='--', lw=0.8, label=f'tol_BS={tol_BS:.1e}')
ax.set_xlabel('Iteration'); ax.set_ylabel('Ratio (echelle log)')
ax.set_title('Criteres BB / BS'); ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.4)

# --- Subplot 3 : theta Kriging vs iterations ---
ax = axes[2]
if hist_theta:
    thetas = np.array(hist_theta); x_th = list(range(len(hist_theta)))
    for k in range(thetas.shape[1]):
        lbl = params_names[k] if k < len(params_names) else f'dim{k}'
        ax.semilogy(x_th, np.maximum(thetas[:, k], _clip), '-o', ms=4, lw=1.2, label=f'theta_{lbl}')
    norms_th = np.maximum(np.linalg.norm(thetas, axis=1), _clip)
    ax.semilogy(x_th, norms_th, 'k--', ms=3, lw=1, label='||theta||')
ax.set_xlabel('Iteration (fit)'); ax.set_ylabel('theta (echelle log)')
ax.set_title('Evolution theta Kriging'); ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.4)

fig.suptitle('EFF_graphs (REGENERE depuis restart_state.json)', fontsize=10)
fig.tight_layout()
fname = os.path.join(OUT, 'EFF_graphs_RELECTURE_from_dump.png')
fig.savefig(fname, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f"PNG REGENERE -> {fname} ({os.path.getsize(fname)} octets)")
