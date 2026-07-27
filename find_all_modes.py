"""
Trouve TOUS les modes de ruine (design points) sur g=0
en utilisant le Projected Polyhedron sur le GEPCK directement.

Systeme resolu :
  f1(u1, u2) = g(u1, u2) = 0              (frontiere de defaillance)
  f2(u1, u2) = u1*dg/du2 - u2*dg/du1 = 0  (grad g parallele a u)

Les solutions sont les points ou ||u|| est extremal sur g=0 -> modes FORM.

Pipeline (sans grille tensorielle) :
  GEPCK complet -> Bernstein direct via gepck_to_bernstein_patches
  Evite la grille N_GRID^n : cout k_breaks^2 * (degree+1)^2 evals GEPCK
  Les patches incluent le Kriging (GEPCK complet, pas PCE seul).
"""

import json, sys, warnings, time
import numpy as np

sys.path.insert(0, r'C:\_workingDir\_SF\test flexion')
sys.path.insert(0, r'C:\_workingDir\_SF\test flexion\_lib')
from branche1 import fit_gepck, predict_gepck, predict_gradient_gepck

from projected_polyhedron import pp_solve
from gepck_to_bernstein import (gepck_to_bernstein_patches,
                                f2_gepck_to_bernstein_patches)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Parametres
# ============================================================

# Domaine physique (espace gaussien standard)
U1_LIM = (-3.0,  3.0)
U2_LIM = (-3.5,  7.5)

# Bernstein : nombre de sous-boites par dimension et degre
# k_breaks=20, degree=3 -> 20^2*4^2 = 6400 evals (equivalent grille 80x80)
K_BREAKS    = 20
BERN_DEGREE = 3


# ============================================================
# Classification min/max via critere SOSC (differences finies)
# Ref : Bertsekas Prop 3.2.1, applique dans Nie 2015 Sect 4.3
# ============================================================

def classify_mode(u, predict_fn, beta):
    """
    Classifie un point critique u* de ||u|| sur g=0.
    Retourne (type_str, kappa) avec type_str in {'min','max','degenere'}.
    Differences finies centrees sur predict_fn (9 evaluations GEPCK).
    """
    u1, u2 = float(u[0]), float(u[1])
    h = 1e-4
    def ev(a, b):
        return float(predict_fn(np.array([[a, b]]))[0])
    g0  = ev(u1,   u2  )
    gp1 = ev(u1+h, u2  );  gm1 = ev(u1-h, u2  )
    gp2 = ev(u1,   u2+h);  gm2 = ev(u1,   u2-h)
    gpp = ev(u1+h, u2+h);  gpm = ev(u1+h, u2-h)
    gmp = ev(u1-h, u2+h);  gmm = ev(u1-h, u2-h)
    g1  = (gp1 - gm1) / (2*h)
    g2  = (gp2 - gm2) / (2*h)
    g11 = (gp1 - 2*g0 + gm1) / h**2
    g22 = (gp2 - 2*g0 + gm2) / h**2
    g12 = (gpp - gpm - gmp + gmm) / (4*h**2)
    ng = np.hypot(g1, g2)
    if ng < 1e-12:
        return 'degenere', np.nan
    A     = g11*g2**2 - 2*g12*g1*g2 + g22*g1**2
    kappa = -A / ng**3
    dot   = u1*g1 + u2*g2
    lhs   = dot * A
    rhs   = ng**4
    tol   = 1e-8 * max(abs(lhs), abs(rhs), 1e-20)
    if   lhs < rhs - tol:  return 'min',      kappa
    elif lhs > rhs + tol:  return 'max',      kappa
    else:                   return 'degenere', kappa


# ============================================================
# 1. Refit GEPCK depuis restart_state
# ============================================================

RESTART = (r'C:\workspace\storage\admin\Moulin_Blanc'
           r'\Calcul_fiabilite_G+LM1_13k_2fy_membrure_inf_diagonal.ds'
           r'\restart_state.json')

_t_total = time.time()

print("Chargement du restart state...")
d = json.load(open(RESTART))
xt = np.array(d['xt'], float)
yt = np.array(d['yt'], float)
all_grad = np.array(d['all_grad'], float)
max_degree = int(d.get('max_degree', 2))
n_var = xt.shape[1]
N = xt.shape[0]

Y_aug = np.concatenate([yt.flatten()] + [all_grad[:, j] for j in range(n_var)])
marginals = [{'Type': 'Gaussian', 'Parameters': [0.0, 1.0]}] * n_var
copula = {'Type': 'Independent', 'Parameters': np.eye(n_var)}
opts = {'Mode': 'optimal',
        'PCE': {'Degree': list(range(1, max_degree + 1)), 'Method': 'LARS'}}

_t = time.time()
print("Fitting GEPCK...")
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fm = fit_gepck(xt, Y_aug, opts, marginals, copula)
print(f"  LOO={fm['Error'][0]['LOO']:.4e}  n_poly={fm['NumberOfPoly']}  [{time.time()-_t:.1f}s]")

# ============================================================
# 2. Evaluateurs GEPCK (pas de grille, pas de spline)
# ============================================================

def predict_g(pts):
    """GEPCK complet : (N, 2) -> (N,)"""
    v = predict_gepck(fm, pts)
    return v.flatten() if v.ndim > 1 else v

def predict_grad(pts):
    """Gradient GEPCK : (N, 2) -> (N, 2)"""
    return predict_gradient_gepck(fm, pts)

def g_eval(u1, u2):
    return predict_g(np.array([[u1, u2]]))[0]

def f2_eval(u1, u2):
    pts = np.array([[u1, u2]])
    G = predict_grad(pts)
    return float(pts[0, 0] * G[0, 1] - pts[0, 1] * G[0, 0])

# ============================================================
# 3. Conversion Bernstein directe + resolution PP
# ============================================================

domain = (U1_LIM, U2_LIM)

_t = time.time()
print(f"Conversion Bernstein directe ({K_BREAKS}x{K_BREAKS} boites, degre {BERN_DEGREE})...")
patches_g  = gepck_to_bernstein_patches(predict_g,   domain, K_BREAKS, BERN_DEGREE)
patches_f2 = f2_gepck_to_bernstein_patches(predict_grad, domain, K_BREAKS, BERN_DEGREE)
print(f"  {len(patches_g)} patches g, {len(patches_f2)} patches f2  [{time.time()-_t:.1f}s]")

_t = time.time()
print("Resolution Projected Polyhedron...")
raw_roots = pp_solve(patches_g, patches_f2, g_eval, f2_eval,
                     epsilon=1e-6, verbose=True)

# ============================================================
# 4. Filtrer et trier les modes
# ============================================================

print(f"  PP termine [{time.time()-_t:.1f}s]")
_t = time.time()
print("\n=== MODES TROUVES ===")
modes_found = []
for r in raw_roots:
    beta  = np.linalg.norm(r)
    g_val = g_eval(r[0], r[1])
    if abs(g_val) < 0.05:
        modes_found.append({'u': r, 'beta': beta, 'g': g_val})

modes_found.sort(key=lambda m: m['beta'])

# Classification min/max via SOSC (differences finies sur GEPCK)
for m in modes_found:
    typ, kappa = classify_mode(m['u'], predict_g, m['beta'])
    m['type']  = typ
    m['kappa'] = kappa

print(f"\n{'#':>3}  {'u1':>10}  {'u2':>10}  {'beta':>8}  {'g':>12}  {'type':>10}  {'kappa':>10}")
print("-" * 75)
for i, m in enumerate(modes_found):
    kv = m.get('kappa', float('nan'))
    ks = f"{kv:+10.5f}" if not np.isnan(kv) else "       nan"
    print(f"{i+1:3d}  {m['u'][0]:+10.5f}  {m['u'][1]:+10.5f}  "
          f"{m['beta']:8.4f}  {m['g']:12.4e}  {m['type']:>10}  {ks}")

print(f"  Classification [{time.time()-_t:.1f}s]")

# Modes FORM du restart pour comparaison
modes_form = d.get('modes', [])
print(f"\n=== MODES FORM (restart) ===")
for i, m in enumerate(modes_form):
    us = np.array(m['u_star'])
    print(f"  Mode {i}: u=({us[0]:+.5f}, {us[1]:+.5f})  beta={m['beta']:.4f}")

# ============================================================
# 5. Figure
# ============================================================

OUTDIR = r'C:\_workingDir\_SF\test flexion\Moulinblanc\output\image spline'

N_PLOT = 150
u1_p = np.linspace(U1_LIM[0], U1_LIM[1], N_PLOT)
u2_p = np.linspace(U2_LIM[0], U2_LIM[1], N_PLOT)
U1p, U2p = np.meshgrid(u1_p, u2_p, indexing='ij')
pts_p  = np.column_stack([U1p.ravel(), U2p.ravel()])
g_plot = predict_g(pts_p).reshape(N_PLOT, N_PLOT)

fig, ax = plt.subplots(figsize=(10, 8))
levels = np.linspace(g_plot.min(), g_plot.max(), 40)
cf = ax.contourf(U1p, U2p, g_plot, levels=levels, cmap='viridis')
plt.colorbar(cf, ax=ax, label='g (GEPCK)')

# Contour g=0 GEPCK
ax.contour(U1p, U2p, g_plot, levels=[0], colors='blue', linewidths=2, zorder=6)

# DOE + EFF
ax.plot(xt[:, 0], xt[:, 1], 'k.', ms=3, zorder=5)

# Modes PP trouves — colores par type (min / max / degenere)
_colors = {'min': 'limegreen', 'max': 'darkorange', 'degenere': 'gray'}
for m in modes_found:
    col = _colors.get(m.get('type', 'degenere'), 'gray')
    ax.plot(m['u'][0], m['u'][1], '*', color=col, ms=16, zorder=10,
            markeredgecolor='white', markeredgewidth=0.5)
    circle = plt.Circle((0, 0), m['beta'], fill=False, color=col,
                         ls='--', lw=1, zorder=4)
    ax.add_patch(circle)

# Modes FORM du restart
for m in modes_form:
    us = np.array(m['u_star'])
    ax.plot(us[0], us[1], 'r*', ms=14, zorder=10,
            markeredgecolor='white', markeredgewidth=0.5)

handles = [
    Line2D([0], [0], color='blue', lw=2,            label='g=0 GEPCK complet'),
    Line2D([0], [0], color='limegreen', marker='*', ls='', ms=14,
           label='Modes PP min local'),
    Line2D([0], [0], color='darkorange', marker='*', ls='', ms=14,
           label='Modes PP max local'),
    Line2D([0], [0], color='gray', marker='*', ls='', ms=14,
           label='Modes PP degenere'),
    Line2D([0], [0], color='r', marker='*', ls='', ms=14,
           label='Modes FORM (restart)'),
    Line2D([0], [0], color='limegreen', ls='--', lw=1,
           label='Cercles ||u||=beta'),
    Line2D([0], [0], color='k', marker='.', ls='', ms=6,
           label='DOE+EFF (%d pts)' % N),
]
ax.legend(handles=handles, loc='upper left', fontsize=9)
ax.set_xlabel('u_s_convoi')
ax.set_ylabel('u_q')
ax.set_title('Tous les modes de ruine (Projected Polyhedron)')
ax.set_xlim(U1_LIM[0], U1_LIM[1])
ax.set_ylim(U2_LIM[0], U2_LIM[1])
fig.tight_layout()
path = OUTDIR + '/all_modes_pp.png'
fig.savefig(path, dpi=150)
print(f"\nFigure sauvegardee : {path}")
print(f"\nTemps total : {time.time()-_t_total:.1f}s")
plt.close(fig)

print("\nTermine.")