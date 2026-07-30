"""
Fonctions de visualisation independantes de run_HF.
Graphiques EFF, Pf, et utilitaires batch mu/sigma et EFF vectorise.
"""
import os
import numpy as np
import openturns as ot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from branche1 import predict_gepck
from config_utilisateur import params_names, modele
from config_pardefaut import tol_EFF, tol_BB, tol_BS, EFF_criteria
import etat


def _batch_mu_sigma(g_ot, sigma_func, grid):
    """Calcule (mu, sigma) en batch sur une grille de points.
    - GEPCK : 1 appel predict_gepck(return_var=True) -> mu + var en 1 fois (BLAS multi-thread)
    - Autres modeles : batch via ot.Sample pour mu, loop fallback pour sigma
    """
    _fm = getattr(getattr(sigma_func, '__self__', None), 'fm', None)
    if _fm is not None:
        mu_arr, sig2_arr = predict_gepck(_fm, grid, return_var=True)
        mu    = mu_arr[:, 0]
        sigma = np.sqrt(np.maximum(0.0, sig2_arr[:, 0]))
        return mu, sigma
    grid_ot = ot.Sample(grid.tolist())
    mu    = np.array(g_ot(grid_ot))[:, 0]
    sigma = np.array([sigma_func(pt) for pt in grid])
    return mu, sigma

def _eff_vectorized(mu, sigma, eps_factor):
    """Calcul vectorise du critere EFF (Expected Feasibility Function)."""
    eps        = eps_factor * sigma
    safe_sigma = np.where(sigma > 0, sigma, 1.0)
    t1 = -mu / safe_sigma
    t2 = (eps + mu) / safe_sigma
    t3 = (eps - mu) / safe_sigma
    eff_vals = (2*mu*norm.cdf(t1) - (eps+mu)*norm.cdf(-t2) + (eps-mu)*norm.cdf(t3)
                + sigma*(-2*norm.pdf(t1) + norm.pdf(t2) + norm.pdf(t3)))
    return np.where(sigma > 0, eff_vals, 0.0)

def print_EFF_graphs():
    """Planche 3 subplots : historique EFF, criteres BB/BS, theta Kriging.
    Lit les globaux _eff_history_*. Sauvegarde en PNG dans etat.out_dir_eff."""
    if not (etat._eff_history_EFF or etat._eff_history_BB or etat._eff_history_theta):
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _clip = 1e-12   # evite log(0)

    # --- Subplot 1 : EFF vs iterations ---
    ax = axes[0]
    x_eff = list(range(len(etat._eff_history_EFF)))
    vals_eff = [max(abs(v), _clip) for v in etat._eff_history_EFF]
    ax.semilogy(x_eff, vals_eff, 'b-o', ms=4, lw=1.2, label='EFF(u_opt)')
    ax.axhline(tol_EFF, color='orange', ls='--', lw=1, label=f'tol_EFF={tol_EFF:.1e}')
    ax.set_xlabel('Iteration EFF')
    ax.set_ylabel('EFF (echelle log)')
    ax.set_title('Convergence EFF')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.4)

    # --- Subplot 2 : BB / BS vs iterations ---
    ax = axes[1]
    if etat._eff_history_BB:
        x_bb = list(range(1, len(etat._eff_history_BB) + 1))
        vals_bb = [max(v, _clip) if v is not None else np.nan for v in etat._eff_history_BB]
        ax.semilogy(x_bb, vals_bb, 'g-o', ms=4, lw=1.2, label='BB')
        ax.axhline(tol_BB, color='g', ls='--', lw=0.8, label=f'tol_BB={tol_BB:.1e}')
    if etat._eff_history_BS:
        x_bs = list(range(1, len(etat._eff_history_BS) + 1))
        vals_bs = [max(v, _clip) if v is not None else np.nan for v in etat._eff_history_BS]
        ax.semilogy(x_bs, vals_bs, 'r-s', ms=4, lw=1.2, label='BS')
        ax.axhline(tol_BS, color='r', ls='--', lw=0.8, label=f'tol_BS={tol_BS:.1e}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Ratio (echelle log)')
    ax.set_title('Criteres BB / BS')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.4)

    # --- Subplot 3 : theta Kriging vs iterations ---
    ax = axes[2]
    if etat._eff_history_theta:
        thetas = np.array(etat._eff_history_theta)   # shape (n_fits, n_var)
        x_th = list(range(len(etat._eff_history_theta)))
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
    fname = f'EFF_graphs_{etat.timestamp}.png'
    fig.savefig(os.path.join(etat.out_dir_eff, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [EFF_graphs] -> {fname}", flush=True)

def print_Pf_evolution():
    """PNG Pf_IS (mu, mu+2sigma, mu-2sigma) au fil des iterations EFF.
    Lit etat._eff_history_Pf. Sauvegarde dans etat.out_dir_eff."""
    if not etat._eff_history_Pf:
        return
    pf_mid = [d['mid'] if d['mid'] is not None else np.nan for d in etat._eff_history_Pf]
    pf_sup = [d['sup'] if d['sup'] is not None else np.nan for d in etat._eff_history_Pf]
    pf_inf = [d['inf'] if d['inf'] is not None else np.nan for d in etat._eff_history_Pf]
    x = list(range(len(etat._eff_history_Pf)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, pf_mid, 'r-o', ms=5, lw=1.5, label='Pf_IS (mu)')
    ax.plot(x, pf_sup, 'b--^', ms=4, lw=1.0, label='Pf_IS (mu+2sigma)')
    ax.plot(x, pf_inf, 'b--v', ms=4, lw=1.0, label='Pf_IS (mu-2sigma)')
    ax.set_xlabel('Iteration EFF')
    ax.set_ylabel('Pf_IS')
    ax.set_title(f'Evolution Pf_IS - {modele} - EFF_criteria={EFF_criteria}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f'Pf_evolution_{etat.timestamp}.png'
    fig.savefig(os.path.join(etat.out_dir_eff, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Pf_evolution] -> {fname}", flush=True)

def print_logPf_evolution():
    """PNG Pf_IS (mu, mu+2sigma, mu-2sigma) en echelle log au fil des iterations EFF.
    Lit etat._eff_history_Pf. Sauvegarde dans etat.out_dir_eff."""
    if not etat._eff_history_Pf:
        return
    pf_mid = [d['mid'] if d['mid'] is not None else np.nan for d in etat._eff_history_Pf]
    pf_sup = [d['sup'] if d['sup'] is not None else np.nan for d in etat._eff_history_Pf]
    pf_inf = [d['inf'] if d['inf'] is not None else np.nan for d in etat._eff_history_Pf]
    x = list(range(len(etat._eff_history_Pf)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(x, pf_mid, 'r-o', ms=5, lw=1.5, label='Pf_IS (mu)')
    ax.semilogy(x, pf_sup, 'b--^', ms=4, lw=1.0, label='Pf_IS (mu+2sigma)')
    ax.semilogy(x, pf_inf, 'b--v', ms=4, lw=1.0, label='Pf_IS (mu-2sigma)')
    ax.set_ylim(bottom=1e-4)
    ax.set_xlabel('Iteration EFF')
    ax.set_ylabel('Pf_IS (echelle log)')
    ax.set_title(f'Evolution Pf_IS log - {modele} - EFF_criteria={EFF_criteria}')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fname = f'logPf_evolution_{etat.timestamp}.png'
    fig.savefig(os.path.join(etat.out_dir_eff, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [logPf_evolution] -> {fname}", flush=True)
