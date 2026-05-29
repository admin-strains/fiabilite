"""
demo_gepck.py -- Démonstration GEPCK (Gradient-Enhanced PCK, Zuhal 2021)

Inspiré de demo_pck.py. Teste la pipeline complète :
  fit_gepck -> predict_gepck

SECTION 1D : f(x) = sin(3*pi*x) * exp(-x^2)   sur [-2, 2]  Uniform[-2,2]
SECTION 2D : g(u1,u2) = (u2 - u1^2/2)^2 + (u1-1)^2/4 - 1  Gaussian(0,1)^2

assemble_Y_aug : helper local (pas encore dans branche3.py)
  Assemble Y_aug = [Y ; dY/dU_0 ; ... ; dY/dU_{M-1}] depuis Y et dY/dX
  en appliquant le jacobien isoprobabiliste :
    Uniform[a,b]     -> dX/dU = (b-a)/2
    Gaussian(mu,sig) -> dX/dU = sig
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import warnings

from branche1 import fit_gepck, predict_gepck, generate_doe


# ===========================================================================
# Helper : assemble_Y_aug
# ===========================================================================
def assemble_Y_aug(Y, dYdX, marginals):
    """
    Convertit les gradients physiques en espace isoprobabiliste et assemble Y_aug.

    Ordre dimension-major (Zuhal 2021) :
      Y_aug = [y(x^1..n) ; dy/du_0(x^1..n) ; ... ; dy/du_{M-1}(x^1..n)]

    Parameters
    ----------
    Y        : (N,)
    dYdX     : (N, M)  gradients dans l'espace physique X
    marginals: list of M dicts

    Returns
    -------
    Y_aug : (N*(M+1),)
    """
    dYdX = np.atleast_2d(dYdX)
    N, M = dYdX.shape
    dYdU = np.zeros_like(dYdX)
    for l, marg in enumerate(marginals):
        mtype = marg['Type'].lower()
        p = marg['Parameters']
        if mtype == 'uniform':
            jac = (p[1] - p[0]) / 2.0          # dX/dU = (b-a)/2
        elif mtype in ('gaussian', 'normal'):
            jac = p[1]                           # dX/dU = sigma
        else:
            raise ValueError(f'assemble_Y_aug: type non supporté "{mtype}"')
        dYdU[:, l] = dYdX[:, l] * jac
    return np.concatenate([Y.ravel()] + [dYdU[:, l] for l in range(M)])


# ===========================================================================
# SECTION 1D
# ===========================================================================
def f1d(x):
    return np.sin(3 * np.pi * x) * np.exp(-x**2)

def df1d(x):
    """df/dx analytique"""
    return (3 * np.pi * np.cos(3 * np.pi * x) - 2 * x * np.sin(3 * np.pi * x)) * np.exp(-x**2)

N1D    = 12
SEED   = 42
X_LO, X_HI = -2.0, 2.0

marg1d = [{'Type': 'Uniform', 'Parameters': [X_LO, X_HI]}]
cop1d  = {'Type': 'Independent', 'Parameters': np.eye(1)}
opts1d = {'Mode': 'sequential', 'PCE': {'Degree': [1, 2, 3, 4], 'Method': 'LARS'}}

X_doe1d = generate_doe(N1D, marg1d, method='lhs', seed=SEED)
Y_doe1d = f1d(X_doe1d.ravel())
G_doe1d = df1d(X_doe1d.ravel()).reshape(-1, 1)   # (N, 1) gradient

Y_aug1d = assemble_Y_aug(Y_doe1d, G_doe1d, marg1d)

print(f'[1D] Y_aug shape = {Y_aug1d.shape}  (attendu {N1D*(1+1)},)')

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fm1d = fit_gepck(X_doe1d, Y_aug1d, opts1d, marg1d, cop1d)

loo1d    = fm1d['Error'][0]['LOO']
npoly1d  = fm1d['NumberOfPoly']
theta1d  = fm1d['Kriging'][0]['theta']
beta1d   = fm1d['Kriging'][0]['beta']
print(f'[1D] LOO={loo1d:.4e}  n_poly={npoly1d}  theta={theta1d}  beta={beta1d}')

X_grid1d = np.linspace(X_LO, X_HI, 500).reshape(-1, 1)
Y_true1d = f1d(X_grid1d.ravel())

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    YMu1d, YSig1d = predict_gepck(fm1d, X_grid1d, return_var=True)

YMu1d  = YMu1d[:, 0]
YSig1d = np.sqrt(np.maximum(YSig1d[:, 0], 0))

fig1, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(X_grid1d, Y_true1d, 'k-',  lw=1.5, label='Réelle')
ax1.plot(X_grid1d, YMu1d,   'b-',  lw=2.0, label='GEPCK mean')
ax1.fill_between(X_grid1d.ravel(),
                 YMu1d - 1.96*YSig1d, YMu1d + 1.96*YSig1d,
                 color='steelblue', alpha=0.20, label='IC 95%')
ax1.plot(X_doe1d, Y_doe1d, 'k+', ms=10, mew=2, label=f'DOE (N={N1D})')
# gradient au DOE : petites barres tangentes
scale = 0.08
for xi, yi, gi in zip(X_doe1d.ravel(), Y_doe1d, G_doe1d.ravel()):
    ax1.annotate('', xy=(xi + scale, yi + gi * scale),
                 xytext=(xi - scale, yi - gi * scale),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

ax1.set_title(f'1D GEPCK — $f(x)=\\sin(3\\pi x)\\,e^{{-x^2}}$  (LOO={loo1d:.2e}, n_poly={npoly1d})')
ax1.legend(); ax1.grid(True, alpha=0.3)
plt.tight_layout()
out1 = os.path.join(os.path.dirname(__file__), 'demo_gepck_1d.png')
plt.savefig(out1, dpi=150)
print(f'Figure 1D sauvegardée : demo_gepck_1d.png')


# ===========================================================================
# SECTION 2D  — marginales Gaussian(0,1)
# ===========================================================================
def g2d(u1, u2):
    return (u2 - u1**2 / 2.0)**2 + (u1 - 1.0)**2 / 4.0 - 1.0

def dg2d(X):
    """Gradients analytiques, X shape (N,2)"""
    u1, u2 = X[:, 0], X[:, 1]
    dg_u1 = 2 * (u2 - u1**2 / 2.0) * (-u1) + (u1 - 1.0) / 2.0
    dg_u2 = 2 * (u2 - u1**2 / 2.0)
    return np.column_stack([dg_u1, dg_u2])   # (N, 2)

N2D  = 15
U_LO, U_HI = -4.0, 4.0

marg2d = [{'Type': 'Gaussian', 'Parameters': [0.0, 1.0]}] * 2
cop2d  = {'Type': 'Independent', 'Parameters': np.eye(2)}
opts2d = {'Mode': 'optimal', 'PCE': {'Degree': [1, 2], 'Method': 'LARS'}}

X_doe2d = generate_doe(N2D, marg2d, method='lhs', seed=SEED)
Y_doe2d = np.array([g2d(u[0], u[1]) for u in X_doe2d])
G_doe2d = dg2d(X_doe2d)   # (N, 2)

Y_aug2d = assemble_Y_aug(Y_doe2d, G_doe2d, marg2d)

print(f'[2D] Y_aug shape = {Y_aug2d.shape}  (attendu {N2D*(2+1)},)')

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fm2d = fit_gepck(X_doe2d, Y_aug2d, opts2d, marg2d, cop2d)

loo2d   = fm2d['Error'][0]['LOO']
npoly2d = fm2d['NumberOfPoly']
theta2d = fm2d['Kriging'][0]['theta']
beta2d  = fm2d['Kriging'][0]['beta']
print(f'[2D] LOO={loo2d:.4e}  n_poly={npoly2d}  theta={theta2d}  beta={beta2d}')

# --- Grille fine ---
ng = 150
u1g = np.linspace(U_LO, U_HI, ng)
u2g = np.linspace(U_LO, U_HI, ng)
U1, U2 = np.meshgrid(u1g, u2g)
X_grid2d = np.column_stack([U1.ravel(), U2.ravel()])

Z_true = g2d(U1, U2)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    YMu2d = predict_gepck(fm2d, X_grid2d)[:, 0].reshape(ng, ng)

# --- Figure 2D ---
from matplotlib.lines import Line2D

fig2, ax2 = plt.subplots(figsize=(7, 6))
cf = ax2.contourf(U1, U2, YMu2d, levels=20, cmap='RdYlGn', alpha=0.6)
plt.colorbar(cf, ax=ax2, label='g (GEPCK)')
ax2.contour(U1, U2, YMu2d,  levels=[0], colors='blue',  linewidths=2)
ax2.contour(U1, U2, Z_true, levels=[0], colors='green', linewidths=2, linestyles='-.')
ax2.scatter(X_doe2d[:, 0], X_doe2d[:, 1], c='black', s=30, zorder=5, label=f'DOE (N={N2D})')
ax2.scatter(0, 0, c='orange', s=100, zorder=6, marker='P', label='[0, 0]')
# quiver gradients au DOE
scale_q = 0.15
ax2.quiver(X_doe2d[:, 0], X_doe2d[:, 1],
           G_doe2d[:, 0], G_doe2d[:, 1],
           color='red', alpha=0.6, scale=20, width=0.004,
           label='gradient DOE')

legend_lines = [
    Line2D([0], [0], color='blue',  linestyle='-',  linewidth=2, label='g=0 GEPCK'),
    Line2D([0], [0], color='green', linestyle='-.', linewidth=2, label='g=0 vraie'),
]
handles, _ = ax2.get_legend_handles_labels()
ax2.legend(handles=handles + legend_lines, fontsize=9)
ax2.set_xlabel('u1'); ax2.set_ylabel('u2')
ax2.set_xlim(U_LO, U_HI); ax2.set_ylim(U_LO, U_HI)
ax2.set_title(f'GEPCK état-limite g=0\n'
              f'N={N2D}, LOO={loo2d:.2e}, n_poly={npoly2d}')
plt.tight_layout()
out2 = os.path.join(os.path.dirname(__file__), 'demo_gepck_2d.png')
plt.savefig(out2, dpi=150)
print(f'Figure 2D sauvegardée : demo_gepck_2d.png')

plt.show()
