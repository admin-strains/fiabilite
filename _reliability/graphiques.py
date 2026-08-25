"""
Graphiques de suivi de l'enrichissement EFF.

Extrait de `AC3_pure_flexion.py` / `AC3_moulinblanc.py`, ou ces fonctions
etaient definies dans `if __name__ == '__main__':` et lisaient directement les
accumulateurs globaux `_eff_history_*`. PHASE 3 du plan de nettoyage.

DEUX FONCTIONS N'EN FONT PLUS QU'UNE

`print_Pf_evolution` et `print_logPf_evolution` etaient identiques a 54 %.
Leur difference tenait entierement a l'echelle de l'axe des ordonnees --
`plot` contre `semilogy`, plus un `ylim`, des libelles et un nom de fichier.
Quatre copies dans le depot pour une seule courbe. `tracer_pf_evolution` prend
desormais un parametre `echelle` : la difference etait un reglage, pas une
logique.

Ce module ne demande que **matplotlib et numpy**. Ni OpenTURNS ni Digital
Structure : les graphiques de convergence doivent pouvoir se relire n'importe
ou, y compris sur un poste qui ne fait pas tourner d'etude.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

#: plancher applique avant passage en echelle logarithmique, pour eviter log(0)
CLIP = 1e-12


def tracer_convergence_eff(historique_eff, historique_bb, historique_bs,
                           historique_theta, params_names,
                           tol_EFF, tol_BB, tol_BS, out_dir, timestamp):
    """Planche de trois graphiques : critere EFF, criteres BB/BS, theta du krigeage.

    Les quatre historiques etaient les globaux `_eff_history_*` de `main`.
    Renvoie le nom du fichier ecrit, ou None si rien n'a ete trace.
    """
    if not (historique_eff or historique_bb or historique_theta):
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- 1 : critere EFF au fil des iterations ---
    ax = axes[0]
    x_eff = list(range(len(historique_eff)))
    vals_eff = [max(abs(v), CLIP) for v in historique_eff]
    ax.semilogy(x_eff, vals_eff, 'b-o', ms=4, lw=1.2, label='EFF(u_opt)')
    ax.axhline(tol_EFF, color='orange', ls='--', lw=1, label=f'tol_EFF={tol_EFF:.1e}')
    ax.set_xlabel('Iteration EFF')
    ax.set_ylabel('EFF (echelle log)')
    ax.set_title('Convergence EFF')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.4)

    # --- 2 : criteres d'arret BB et BS ---
    ax = axes[1]
    if historique_bb:
        x_bb = list(range(1, len(historique_bb) + 1))
        vals_bb = [max(v, CLIP) if v is not None else np.nan for v in historique_bb]
        ax.semilogy(x_bb, vals_bb, 'g-o', ms=4, lw=1.2, label='BB')
        ax.axhline(tol_BB, color='g', ls='--', lw=0.8, label=f'tol_BB={tol_BB:.1e}')
    if historique_bs:
        x_bs = list(range(1, len(historique_bs) + 1))
        vals_bs = [max(v, CLIP) if v is not None else np.nan for v in historique_bs]
        ax.semilogy(x_bs, vals_bs, 'r-s', ms=4, lw=1.2, label='BS')
        ax.axhline(tol_BS, color='r', ls='--', lw=0.8, label=f'tol_BS={tol_BS:.1e}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Ratio (echelle log)')
    ax.set_title('Criteres BB / BS')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.4)

    # --- 3 : longueurs de correlation du krigeage ---
    ax = axes[2]
    if historique_theta:
        thetas = np.array(historique_theta)          # (n_ajustements, n_var)
        x_th = list(range(len(historique_theta)))
        for k in range(thetas.shape[1]):
            lbl = params_names[k] if k < len(params_names) else f'dim{k}'
            ax.semilogy(x_th, np.maximum(thetas[:, k], CLIP), '-o', ms=4, lw=1.2,
                        label=f'theta_{lbl}')
        norms_th = np.maximum(np.linalg.norm(thetas, axis=1), CLIP)
        ax.semilogy(x_th, norms_th, 'k--', ms=3, lw=1, label='||theta||')
    ax.set_xlabel('Iteration (fit)')
    ax.set_ylabel('theta (echelle log)')
    ax.set_title('Evolution theta Kriging')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.4)

    fig.tight_layout()
    fname = f'EFF_graphs_{timestamp}.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [EFF_graphs] -> {fname}", flush=True)
    return fname


def tracer_pf_evolution(historique_pf, modele, EFF_criteria, out_dir, timestamp,
                        echelle='lineaire'):
    """Probabilite de defaillance et son encadrement, au fil des iterations EFF.

    `echelle` vaut 'lineaire' ou 'log'. C'etait la seule difference entre
    `print_Pf_evolution` et `print_logPf_evolution`, qui existaient en quatre
    exemplaires dans le depot.

    Renvoie le nom du fichier ecrit, ou None si l'historique est vide.
    """
    if echelle not in ('lineaire', 'log'):
        raise ValueError("echelle doit valoir 'lineaire' ou 'log', pas %r" % (echelle,))
    if not historique_pf:
        return None

    log = echelle == 'log'
    pf_mid = [d['mid'] if d['mid'] is not None else np.nan for d in historique_pf]
    pf_sup = [d['sup'] if d['sup'] is not None else np.nan for d in historique_pf]
    pf_inf = [d['inf'] if d['inf'] is not None else np.nan for d in historique_pf]
    x = list(range(len(historique_pf)))

    fig, ax = plt.subplots(figsize=(8, 5))
    tracer = ax.semilogy if log else ax.plot
    tracer(x, pf_mid, 'r-o', ms=5, lw=1.5, label='Pf_IS (mu)')
    tracer(x, pf_sup, 'b--^', ms=4, lw=1.0, label='Pf_IS (mu+2sigma)')
    tracer(x, pf_inf, 'b--v', ms=4, lw=1.0, label='Pf_IS (mu-2sigma)')
    if log:
        ax.set_ylim(bottom=1e-4)
    ax.set_xlabel('Iteration EFF')
    ax.set_ylabel('Pf_IS (echelle log)' if log else 'Pf_IS')
    ax.set_title(f'Evolution Pf_IS{" log" if log else ""} - {modele}'
                 f' - EFF_criteria={EFF_criteria}')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3) if log else ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f'{"logPf" if log else "Pf"}_evolution_{timestamp}.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [{"logPf" if log else "Pf"}_evolution] -> {fname}', flush=True)
    return fname
