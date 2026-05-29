"""
plot_deriv_gepck.py
Trace ∂ŷ/∂x_0 (analytique) vs FD et la vraie dérivée sur une grille 1D.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import warnings
import matplotlib.pyplot as plt

from branche1 import fit_gepck, predict_gepck, predict_deriv_gepck

# ---------------------------------------------------------------------------
# Données d'entraînement
# ---------------------------------------------------------------------------
rng = np.random.default_rng(13)
N   = 12
M   = 2

marg = [{'Type': 'Uniform', 'Parameters': [-1.0, 1.0]} for _ in range(M)]
cop  = {'Type': 'Independent', 'Parameters': np.eye(M)}

X_tr = rng.uniform(-1, 1, (N, M))

def f(X): return X[:, 0]**3 + X[:, 0] * X[:, 1] + 0.5 * X[:, 1]**2
def df0(X): return 3 * X[:, 0]**2 + X[:, 1]   # ∂f/∂x_0
def df1(X): return X[:, 0] + X[:, 1]           # ∂f/∂x_1

Y_tr  = f(X_tr)
G_tr  = np.column_stack([df0(X_tr), df1(X_tr)])
Y_aug = np.concatenate([Y_tr, G_tr[:, 0], G_tr[:, 1]])

opts = {
    'Mode': 'optimal',
    'PCE':  {'Degree': [1, 2, 3], 'Method': 'LARS'},
    'Kriging': {'Corr': {'Family': 'gaussian'}},
}

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fm = fit_gepck(X_tr, Y_aug, opts, marg, cop)

# ---------------------------------------------------------------------------
# Grille 1D : x_0 varie sur [-1,1], x_1 fixé à 0.3
# ---------------------------------------------------------------------------
Ng   = 200
x0   = np.linspace(-1, 1, Ng)
x1   = 0.3
# Pour le FD on évite les bords ±1 (hors support Legendre quand on décale par eps)
x0_fd = np.linspace(-0.95, 0.95, Ng)
X_grid = np.column_stack([x0, np.full(Ng, x1)])

# Dérivée analytique (GEPCK)
dy_analytic = predict_deriv_gepck(fm, X_grid, der_var=0)[:, 0]

# Dérivée FD (sur predict_gepck) — grille intérieure pour éviter bord Legendre
eps      = 1e-4
X_grid_fd = np.column_stack([x0_fd, np.full(Ng, x1)])
X_p   = X_grid_fd.copy(); X_p[:, 0] += eps
X_m   = X_grid_fd.copy(); X_m[:, 0] -= eps
dy_fd = (predict_gepck(fm, X_p)[:, 0] - predict_gepck(fm, X_m)[:, 0]) / (2 * eps)
dy_analytic_fd = predict_deriv_gepck(fm, X_grid_fd, der_var=0)[:, 0]  # même grille pour résidu

# Vraie dérivée
dy_true = df0(X_grid)

# ---------------------------------------------------------------------------
# Tracé
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
fig.suptitle(r'$\partial\hat{y}/\partial x_0$  —  $x_1 = 0.3$', fontsize=13)

dy_true_fd = df0(X_grid_fd)

for ax, title, x_arr, dy_approx, c in [
    (axes[0], 'Analytique  (predict_deriv_gepck)', x0,    dy_analytic,    'tab:blue'),
    (axes[1], 'Différences finies  (FD predict_gepck)',  x0_fd, dy_fd,           'tab:orange'),
]:
    dy_ref = df0(np.column_stack([x_arr, np.full(Ng, x1)]))
    ax.plot(x_arr, dy_ref,    'k--',  lw=1.5, label='vraie dérivée', zorder=3)
    ax.plot(x_arr, dy_approx,  color=c, lw=2,  label=title.split('(')[0].strip(), zorder=2)
    ax.set_xlabel('$x_0$', fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Panneau droit : ajoute aussi la courbe analytique en fond
axes[1].plot(x0_fd, dy_analytic_fd, color='tab:blue', lw=1, alpha=0.4, ls=':', label='analytique')
axes[1].legend(fontsize=9)

# Résidu FD - analytique (même grille)
ax_res = fig.add_axes([0.35, 0.12, 0.28, 0.20])
ax_res.semilogy(x0_fd, np.abs(dy_fd - dy_analytic_fd) + 1e-16, color='tab:red', lw=1)
ax_res.set_title('|FD − analytique|', fontsize=8)
ax_res.set_xlabel('$x_0$', fontsize=8)
ax_res.tick_params(labelsize=7)
ax_res.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'plot_deriv_gepck.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f'Figure sauvée : {out}')
plt.show()
