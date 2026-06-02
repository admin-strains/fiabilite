"""
demo_notre_gepck.py -- Test GEPCK (5 branches) sur le modèle flexion_claude

Objectif : remplacer la partie 'modèle' de AC_pure_flexion.py (SMT/OpenTURNS)
par notre pipeline fit_gepck / predict_gepck (branche1-5).

Modèle physique : flexion_claude.g(u1, u2)
  - u1, u2 : espace standard normal N(0,1)  (transform. isoproba OpenTURNS)
  - paramètres : fc (LogNormal) et fy (Normal)
  - fonction d'état-limite analytique (pivot B, béton armé)

Pipeline GEPCK :
  1. DOE en espace N(0,1)²
  2. Évaluation de g au DOE
  3. Gradients ∂g/∂u par différences finies (g est analytique mais non linéaire)
  4. assemble_Y_aug  →  Y_aug = [g ; ∂g/∂u1 ; ∂g/∂u2]
  5. fit_gepck(U_doe, Y_aug, opts, marginals, copula)
  6. predict_gepck sur grille fine  →  visualisation
"""

import os
import sys
import re
import math
import warnings

import numpy as np
import openturns as ot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))
from branche1 import fit_gepck, predict_gepck, generate_doe


# ===========================================================================
# Variables globales et fonctions copiées de AC_pure_flexion.py
# (dépendances de flexion_claude)
# ===========================================================================

modelname    = "test_pure_flexion"
params_names = ['fc', 'fy']
n_var        = len(params_names)

fcm, fym        = 48, 550          # MPa
cov_fc, cov_fy  = 0.12, None
fc_otparams     = (fcm, cov_fc)
fy_otparams     = (fym, cov_fy)

Es   = 200000
ecu  = 0.0035
eud  = 0.045

# --- lecture de gamma depuis dsCad.txt ---
_path_ds = rf"C:\workspace\storage\admin\SF\{modelname}.ds"
with open(os.path.join(_path_ds, 'dsCad.txt'), 'r') as _f:
    _cad_txt = _f.read()


def _parse(text, name):
    return float(re.search(rf'(?m)^\s*{re.escape(name)}\s*=\s*([\d.]+)', text).group(1))


gamma_c_fic = _parse(_cad_txt, 'gamma_c')   # 1.0
gamma_s_fic = _parse(_cad_txt, 'gamma_s')   # 1.0

SIGMA_11, SIGMA_12, SIGMA_13 = 19.0, 22.0, 8.0
SIGMA = np.sqrt(SIGMA_11**2 + SIGMA_12**2 + SIGMA_13**2)


def loi_fy(fym, cov=None):
    sig_ec = cov * fym if cov is not None else SIGMA
    return ot.Normal(fym, sig_ec)


def loi_fc(fcm, cov=None):
    COV_TABLE = {"C15": 0.14, "C25": 0.12, "C35": 0.09, "C45": 0.07}
    fck_eq = fcm - 8.0
    classe = min(COV_TABLE, key=lambda c: abs(int(c[1:]) - fck_eq))
    v = cov if cov is not None else COV_TABLE[classe]
    sigma_ln = np.sqrt(np.log(1 + v**2))
    mu_ln    = np.log(fcm) - 0.5 * sigma_ln**2
    return ot.LogNormal(mu_ln, sigma_ln, 0.0)


# ===========================================================================
# Classe flexion_claude — copiée intégralement de AC_pure_flexion.py
# ===========================================================================
class flexion_claude:
    def __init__(self):
        path = os.path.join(r'C:\workspace\storage\admin\SF', modelname + '.ds')
        with open(os.path.join(path, 'dsCad.txt'), 'r') as f:
            _cad = f.read()
        with open(os.path.join(path, 'dsLoad.txt'), 'r') as f:
            _load = f.read()

        b   = _parse(_cad, 'b')
        h   = _parse(_cad, 'h')
        L   = _parse(_cad, 'L')
        phi = _parse(_cad, 'phi')

        n_bars = len(re.findall(r'REBAR\(', _cad))
        As = n_bars * math.pi * (phi / 2e3) ** 2

        z_rebar = [float(v) for v in re.findall(
            r'pts\d+\.append\(POINT\([^,]+,\s*[^,]+,\s*([\d.]+)\)', _cad)]
        d = h / 2 + sum(z_rebar) / len(z_rebar)

        F   = abs(float(re.search(r"Z='(-?[\d.]+)'", _load).group(1)))
        Med = F * L

        self.fym = fy_otparams[0]
        dist = []
        if 'fc' in params_names:
            dist.append(loi_fc(*fc_otparams))
        if 'fy' in params_names:
            dist.append(loi_fy(*fy_otparams))
        dist_X     = ot.JointDistribution(dist)
        self.T_inv = dist_X.getInverseIsoProbabilisticTransformation()
        self.T     = dist_X.getIsoProbabilisticTransformation()

        self.A = As * d / gamma_s_fic
        self.B = -As**2 * gamma_c_fic / (2 * b * gamma_s_fic**2)
        self.C = -Med

        self.Ap = 0.8 * d * b / (As * gamma_c_fic * Es * ecu)
        self.Bp = 0.8 * b * d**2 / gamma_c_fic
        self.Cp = 2 * self.Ap * self.C / self.Bp
        ap = 1
        bp = self.Cp - 0.8
        cp = self.Cp - 0.2
        Delta_p    = bp**2 - 4 * ap * cp
        sol1_s     = (-bp + Delta_p**0.5) / (2 * ap)
        sol1_x1    = (sol1_s**2 - 1) / (4 * self.Ap)
        self.u1_lim_plast = self.T(ot.Point([sol1_x1, 0.0]))[0]

        self.A1 = As * gamma_c_fic * Es * ecu / (0.8 * b * d)
        self.A2 = Es * ecu * gamma_s_fic

    def u2p_LS(self, u1):
        x_point = self.T_inv(ot.Point([u1, 0.0]))
        x1  = x_point[0]
        a   = self.B
        b   = self.A * x1
        c   = self.C * x1
        Delta = b**2 - 4 * a * c
        fy  = (-b + Delta**0.5) / (2 * a)
        return self.T(ot.Point([0.0, fy]))[1]

    def g(self, u1, u2):
        x_point = self.T_inv(ot.Point([u1, u2]))
        x1 = x_point[0]
        x2 = x_point[1]
        x1_lim_plast_x2 = self.A1 * x2 * (self.A2 + x2) / self.A2**2
        if x1 > x1_lim_plast_x2:
            return (self.A * x2 + self.B * x2**2 / x1 + self.C) / (-self.C)
        else:
            s = (1 + 4 * self.Ap * x1)**0.5
            return -1 - (s - 1) / self.Cp + 0.8 * (s - 1) / (self.Cp * (s + 1))


# ===========================================================================
# Helper : assemble_Y_aug  (même convention que demo_gepck.py)
# Pour Gaussian(0,1) : Jacobien dX/dU = sigma = 1 → dYdU = dYdX
# ===========================================================================
def assemble_Y_aug(Y, dYdX, marginals):
    dYdX = np.atleast_2d(dYdX)
    N, M = dYdX.shape
    dYdU = np.zeros_like(dYdX)
    for l, marg in enumerate(marginals):
        mtype = marg['Type'].lower()
        p     = marg['Parameters']
        if mtype == 'uniform':
            jac = (p[1] - p[0]) / 2.0
        elif mtype in ('gaussian', 'normal'):
            jac = p[1]                    # sigma
        else:
            raise ValueError(f'assemble_Y_aug : type non supporté "{mtype}"')
        dYdU[:, l] = dYdX[:, l] * jac
    return np.concatenate([Y.ravel()] + [dYdU[:, l] for l in range(M)])


# ===========================================================================
# Helper : gradients ∂g/∂u par différences finies centrées
# g est analytique → FD précises et peu coûteuses
# ===========================================================================
def compute_gradients_FD(calc, U_doe, h=1e-5):
    N, M = U_doe.shape
    dYdU = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            u_p    = U_doe[i].copy(); u_p[j] += h
            u_m    = U_doe[i].copy(); u_m[j] -= h
            dYdU[i, j] = (calc.g(u_p[0], u_p[1]) - calc.g(u_m[0], u_m[1])) / (2 * h)
    return dYdU


# ===========================================================================
# Paramètres du demo
# ===========================================================================
SEED   = 42
N_DOE  = 15

# Marginales en espace standard normal N(0,1)²
# flexion_claude.g(u1, u2) prend déjà des arguments en espace standard normal
# (T_inv convertit en interne vers l'espace physique)
marginals = [
    {'Type': 'Gaussian', 'Parameters': [0.0, 1.0]},
    {'Type': 'Gaussian', 'Parameters': [0.0, 1.0]},
]
copula = {'Type': 'Independent', 'Parameters': np.eye(2)}
opts   = {'Mode': 'optimal',
          'PCE':  {'Degree': [1, 2, 3], 'Method': 'LARS'}}

# Bornes de visualisation
u_lo, u_hi = -8.0, 4.0   # domaine pertinent : région de défaillance autour de (-4, -7)

# ===========================================================================
# 1. DOE
# ===========================================================================
print("=" * 60)
print("DEMO GEPCK — modèle flexion_claude")
print("=" * 60)

print(f"\n[1] Génération du DOE  (N={N_DOE}, LHS, N(0,1)²) ...")
calc  = flexion_claude()
print(f"    u1_lim_plast = {calc.u1_lim_plast:.4f}")

U_doe = generate_doe(N_DOE, marginals, method='lhs', seed=SEED)
print(f"    U_doe shape  = {U_doe.shape}")

# ===========================================================================
# 2. Évaluation de g au DOE
# ===========================================================================
print(f"\n[2] Évaluation de g au DOE ...")
Y_doe = np.array([calc.g(u[0], u[1]) for u in U_doe])
print(f"    Y_doe  = {Y_doe}")
print(f"    range  = [{Y_doe.min():.4f}, {Y_doe.max():.4f}]")

# ===========================================================================
# 3. Gradients par FD
# ===========================================================================
print(f"\n[3] Calcul des gradients dg/du (FD centrees, h=1e-5) ...")
dYdU_doe = compute_gradients_FD(calc, U_doe)
print(f"    dYdU range = [{dYdU_doe.min():.4f}, {dYdU_doe.max():.4f}]")
print(f"    dYdU :\n{dYdU_doe}")

# ===========================================================================
# 4. Assemblage Y_aug
# ===========================================================================
Y_aug = assemble_Y_aug(Y_doe, dYdU_doe, marginals)
print(f"\n[4] Y_aug shape = {Y_aug.shape}  (attendu {N_DOE * (n_var + 1)},)")

# ===========================================================================
# 5. Fit GEPCK
# ===========================================================================
print(f"\n[5] Fit GEPCK (mode={opts['Mode']}, degrés={opts['PCE']['Degree']}) ...")
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fm = fit_gepck(U_doe, Y_aug, opts, marginals, copula)

loo   = fm['Error'][0]['LOO']
npoly = fm['NumberOfPoly']
theta = fm['Kriging'][0]['theta']
beta  = fm['Kriging'][0]['beta']
print(f"    LOO     = {loo:.4e}")
print(f"    n_poly  = {npoly}")
print(f"    theta   = {theta}")
print(f"    beta    = {beta}")

# ===========================================================================
# 6. Prédiction sur grille
# ===========================================================================
print(f"\n[6] Prédiction GEPCK sur grille ({u_lo},{u_hi})² ...")
ng   = 120
u1g  = np.linspace(u_lo, u_hi, ng)
u2g  = np.linspace(u_lo, u_hi, ng)
U1, U2 = np.meshgrid(u1g, u2g)
X_grid = np.column_stack([U1.ravel(), U2.ravel()])

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    YMu_grid, YSig_grid = predict_gepck(fm, X_grid, return_var=True)

Z_gepck = YMu_grid[:, 0].reshape(ng, ng)
Z_sigma = np.sqrt(np.maximum(YSig_grid[:, 0], 0)).reshape(ng, ng)

# ===========================================================================
# 7. Référence analytique sur grille
# ===========================================================================
print(f"[7] Calcul de la référence analytique sur la grille ...")
ng_ref = 80
u1r    = np.linspace(u_lo, u_hi, ng_ref)
u2r    = np.linspace(u_lo, u_hi, ng_ref)
U1r, U2r = np.meshgrid(u1r, u2r)
Z_ana  = np.array([calc.g(U1r.ravel()[i], U2r.ravel()[i])
                   for i in range(ng_ref * ng_ref)]).reshape(ng_ref, ng_ref)
print(f"    Z_ana range = [{Z_ana.min():.4f}, {Z_ana.max():.4f}]")

# ===========================================================================
# 8. Figures
# ===========================================================================
print(f"\n[8] Tracé des figures ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Figure gauche : GEPCK vs analytique ──────────────────────────────────
ax = axes[0]
levels = np.linspace(-1.5, 3.0, 25)
cf = ax.contourf(U1, U2, Z_gepck, levels=levels, cmap='RdYlGn', alpha=0.65, extend='both')
plt.colorbar(cf, ax=ax, label='g GEPCK')

# g=0 GEPCK
ax.contour(U1, U2, Z_gepck, levels=[0], colors='blue',  linewidths=2.0)
# g=0 analytique
ax.contour(U1r, U2r, Z_ana, levels=[0], colors='green', linewidths=2.0, linestyles='--')

# DOE
ax.scatter(U_doe[:, 0], U_doe[:, 1], c='black', s=50, zorder=6, label=f'DOE (N={N_DOE})')

# Gradients au DOE (flèches normalisées)
scale_q = np.percentile(np.linalg.norm(dYdU_doe, axis=1), 75)
ax.quiver(U_doe[:, 0], U_doe[:, 1],
          dYdU_doe[:, 0], dYdU_doe[:, 1],
          color='red', alpha=0.7, scale=scale_q * 12, width=0.004,
          label='∂g/∂u (FD)')

# Limite de plasticité (trait vertical)
ax.axvline(calc.u1_lim_plast, color='orange', lw=1.2, ls=':', label=f'u1_lim_plast={calc.u1_lim_plast:.2f}')

legend_els = [
    Line2D([0], [0], color='blue',  lw=2,       label='g=0  GEPCK'),
    Line2D([0], [0], color='green', lw=2, ls='--', label='g=0  analytique'),
]
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles + legend_els, fontsize=8, loc='upper right')
ax.set_xlabel(r'$u_1$  (fc, espace standard $\mathcal{N}(0,1)$)')
ax.set_ylabel(r'$u_2$  (fy, espace standard $\mathcal{N}(0,1)$)')
ax.set_xlim(u_lo, u_hi)
ax.set_ylim(u_lo, u_hi)
ax.set_title(f'GEPCK — flexion_claude\n'
             f'N={N_DOE}, LOO={loo:.2e}, n_poly={npoly}')
ax.grid(True, alpha=0.25)

# ── Figure droite : incertitude GEPCK ────────────────────────────────────
ax2 = axes[1]
cf2 = ax2.contourf(U1, U2, Z_sigma, levels=20, cmap='Blues')
plt.colorbar(cf2, ax=ax2, label='σ GEPCK')
ax2.contour(U1,  U2,  Z_gepck, levels=[0], colors='blue',  linewidths=2.0)
ax2.contour(U1r, U2r, Z_ana,   levels=[0], colors='green', linewidths=2.0, linestyles='--')
ax2.scatter(U_doe[:, 0], U_doe[:, 1], c='red', s=50, zorder=6, label=f'DOE (N={N_DOE})')
ax2.legend(fontsize=8)
ax2.set_xlabel(r'$u_1$')
ax2.set_ylabel(r'$u_2$')
ax2.set_xlim(u_lo, u_hi)
ax2.set_ylim(u_lo, u_hi)
ax2.set_title('Incertitude GEPCK (σ)')
ax2.grid(True, alpha=0.25)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'demo_notre_gepck.png')
plt.savefig(out, dpi=150)
print(f"Figure sauvegardée : demo_notre_gepck.png")
plt.show()
