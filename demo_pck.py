"""
demo_pck.py -- Démonstration PCK

SECTION 1D : f(x) = sin(3*pi*x) * exp(-x^2)   sur [-2, 2]
SECTION 2D : g(u1,u2) = sin(u1+u2) * exp(-(u1^2+u2^2)/8)
             marginales Gaussian(0,1) x Gaussian(0,1) (meme espace que flexion)
             Affiche : vraie fonction vs PCK (contourf) + courbe g=0 superposée
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import warnings

from branche1 import fit_pck, predict_pck, generate_doe

# ===========================================================================
# SECTION 1D
# ===========================================================================
def f_true_1d(x):
    return np.sin(3 * np.pi * x) * np.exp(-x**2)

N1D   = 20
SEED  = 42
X_LO, X_HI = -2.0, 2.0

marg1d = [{'Type': 'Uniform', 'Parameters': [X_LO, X_HI]}]
cop1d  = {'Type': 'Independent', 'Parameters': np.eye(1)}
opts1d = {'Mode': 'sequential', 'PCE': {'Degree': [1, 2, 3, 4], 'Method': 'LARS'}}

X_doe1d = generate_doe(N1D, marg1d, method='lhs', seed=SEED)
Y_doe1d = f_true_1d(X_doe1d.ravel())

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fm1d = fit_pck(X_doe1d, Y_doe1d, opts1d, marg1d, cop1d)

loo1d = fm1d['Error'][0]['LOO']
print(f'[1D] N={N1D}  LOO={loo1d:.4e}  n_poly={fm1d["NumberOfPoly"][0]}')

X_grid1d = np.linspace(X_LO, X_HI, 500).reshape(-1, 1)
Y_true1d = f_true_1d(X_grid1d.ravel())
YMu1d, YSig1d = predict_pck(fm1d, X_grid1d, return_var=True)
YMu1d  = YMu1d[:, 0]
YSig1d = np.sqrt(np.maximum(YSig1d[:, 0], 0))

fig1, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(X_grid1d, Y_true1d, 'k-',  lw=1.5, label='Réelle')
ax1.plot(X_grid1d, YMu1d,   'b-',  lw=2.0, label='PCK mean')
ax1.fill_between(X_grid1d.ravel(),
                 YMu1d - 1.96*YSig1d, YMu1d + 1.96*YSig1d,
                 color='steelblue', alpha=0.20, label='IC 95%')
ax1.plot(X_doe1d, Y_doe1d, 'k+', ms=10, mew=2, label=f'DOE (N={N1D})')
ax1.set_title(f'1D — $f(x)=\\sin(3\\pi x)\\,e^{{-x^2}}$  (LOO={loo1d:.2e})')
ax1.legend(); ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'demo_pck_1d.png'), dpi=150)
print('Figure 1D sauvegardee : demo_pck_1d.png')

# ===========================================================================
# SECTION 2D  —  marginales Gaussian(0,1) comme dans flexion
# ===========================================================================
def f_true_2d(u1, u2):
    """g(u1,u2) = (u2 - u1^2/2)^2 + (u1-1)^2/4 - 1  (etat-limite banane, non periodique)"""
    return (u2 - u1**2 / 2.0)**2 + (u1 - 1.0)**2 / 4.0 - 1.0

N2D  = 15
U_LO, U_HI = -4.0, 4.0   # fenetre de visu (±4 sigma)

marg2d = [{'Type': 'Gaussian', 'Parameters': [0.0, 1.0]}] * 2
cop2d  = {'Type': 'Independent', 'Parameters': np.eye(2)}
opts2d = {'Mode': 'optimal', 'PCE': {'Degree': [1, 2], 'Method': 'LARS'}}

# --- DOE et évaluation ---
X_doe2d = generate_doe(N2D, marg2d, method='lhs', seed=SEED)
Y_doe2d = np.array([f_true_2d(u[0], u[1]) for u in X_doe2d])

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fm2d = fit_pck(X_doe2d, Y_doe2d, opts2d, marg2d, cop2d)

loo2d = fm2d['Error'][0]['LOO']
print(f'[2D] N={N2D}  LOO={loo2d:.4e}  n_poly={fm2d["NumberOfPoly"][0]}')

# --- Grille fine ---
ng = 150
u1g = np.linspace(U_LO, U_HI, ng)
u2g = np.linspace(U_LO, U_HI, ng)
U1, U2 = np.meshgrid(u1g, u2g)
X_grid2d = np.column_stack([U1.ravel(), U2.ravel()])

Z_true = f_true_2d(U1, U2)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    YMu2d = predict_pck(fm2d, X_grid2d)[:, 0].reshape(ng, ng)

# --- Figure 2D : un panneau, style print_visu ---
from matplotlib.lines import Line2D

fig2, ax2 = plt.subplots(figsize=(7, 6))

# Fond : PCK mean, RdYlGn comme print_visu
cf = ax2.contourf(U1, U2, YMu2d, levels=20, cmap='RdYlGn', alpha=0.6)
plt.colorbar(cf, ax=ax2, label='g (PCK)')

# g=0 PCK — bleu continu (comme metamodele dans print_visu)
ax2.contour(U1, U2, YMu2d,  levels=[0], colors='blue',  linewidths=2)
# g=0 vraie — vert dash-dot (comme print_ana dans print_visu)
ax2.contour(U1, U2, Z_true, levels=[0], colors='green', linewidths=2, linestyles='-.')

# Points DOE — scatter noir (comme xt dans print_visu)
ax2.scatter(X_doe2d[:, 0], X_doe2d[:, 1], c='black', s=30, zorder=5, label=f'DOE (N={N2D})')
# Origine — orange P (comme print_visu)
ax2.scatter(0, 0, c='orange', s=100, zorder=6, marker='P', label='[0, 0]')

# Legende contours via Line2D (comme print_visu)
legend_lines = [
    Line2D([0], [0], color='blue',  linestyle='-',  linewidth=2, label='g=0 PCK'),
    Line2D([0], [0], color='green', linestyle='-.', linewidth=2, label='g=0 vraie'),
]
handles, _ = ax2.get_legend_handles_labels()
ax2.legend(handles=handles + legend_lines, fontsize=9)

ax2.set_xlabel('u1')
ax2.set_ylabel('u2')
ax2.set_xlim(U_LO, U_HI)
ax2.set_ylim(U_LO, U_HI)
ax2.set_title(f'PCK et etat limite g=0\n'
              f'g=(u2-u1²/2)²+(u1-1)²/4-1  N={N2D}, LOO={loo2d:.2e}')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'demo_pck_2d.png'), dpi=150)
print('Figure 2D sauvegardee : demo_pck_2d.png')

plt.show()
