"""
test_f_deriv_handles.py
=======================
Vérifie que les closures F_deriv_handles[der] stockées dans
fitted_kriging par uq_GEPCK_calculate_coefficients sont bien
∂Ψ/∂u_{der} — comparaison par différences finies sur la matrice
de trend non-augmentée F_global_handle(U)[:N, :].

Tests :
  T1 — F_deriv_handles est bien stocké dans fm['Kriging'][0]
  T2 — shape cohérente : (N_test, P_sel)
  T3 — FD vs analytique : erreur < 1e-7  (mode sequential, Legendre)
  T4 — FD vs analytique : erreur < 1e-7  (mode sequential, Hermite)
  T5 — FD vs analytique : erreur < 1e-7  (mode optimal, Legendre)
  T6 — Ψ != ∂Ψ  (les handles ne sont pas identiques)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import warnings
from branche1 import fit_gepck

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = 0
FAIL = 0
EPS  = 1e-5


def ok(label, condition, info=''):
    global PASS, FAIL
    if condition:
        print(f'  PASS  {label}')
        PASS += 1
    else:
        print(f'  FAIL  {label}' + (f'  [{info}]' if info else ''))
        FAIL += 1


def fd_psi(F_handle_block0, U_test, der, eps=EPS):
    """Dérivée numérique de Ψ(U) par rapport à U[:,der]."""
    U_plus  = U_test.copy(); U_plus[:,  der] += eps
    U_minus = U_test.copy(); U_minus[:, der] -= eps
    return (F_handle_block0(U_plus) - F_handle_block0(U_minus)) / (2 * eps)


def fit_silent(X, Y_aug, options, marginals, copula):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fit_gepck(X, Y_aug, options, marginals, copula)


# ---------------------------------------------------------------------------
# Données d'entraînement  M=2, N=12
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
N   = 12
M   = 2

# --- Cas Legendre : marginals Uniform(-1, 1) → espace U = Legendre
marg_legendre = [{'Type': 'Uniform', 'Parameters': [-1.0, 1.0]} for _ in range(M)]
cop = {'Type': 'Independent', 'Parameters': np.eye(M)}

X_L = rng.uniform(-1, 1, (N, M))

def f_true(X):
    return X[:, 0]**3 + X[:, 0] * X[:, 1]

def grad_true(X):
    G = np.zeros_like(X)
    G[:, 0] = 3 * X[:, 0]**2 + X[:, 1]
    G[:, 1] = X[:, 0]
    return G

Y_L    = f_true(X_L)
G_L    = grad_true(X_L)
# Y_aug pour Uniform(-1,1) : transform U=X (identité), donc ∂y/∂u_i = ∂y/∂x_i
Y_aug_L = np.concatenate([Y_L, G_L[:, 0], G_L[:, 1]])  # (N*(M+1),)

# --- Cas Hermite : marginals Gaussian(0,1) → espace U = Hermite
marg_hermite = [{'Type': 'Gaussian', 'Parameters': [0.0, 1.0]} for _ in range(M)]
X_H     = rng.standard_normal((N, M))
Y_H     = f_true(X_H)
G_H     = grad_true(X_H)
# Pour Gaussian(0,1), u = x (transform identité dans l'espace de Hermite)
Y_aug_H = np.concatenate([Y_H, G_H[:, 0], G_H[:, 1]])

opts_seq  = {
    'Mode': 'sequential',
    'PCE':  {'Degree': [1, 2], 'Method': 'LARS'},
    'Kriging': {'Corr': {'Family': 'gaussian'}},
}
opts_opt  = {
    'Mode': 'optimal',
    'PCE':  {'Degree': [1, 2], 'Method': 'LARS'},
    'Kriging': {'Corr': {'Family': 'matern-5_2'}},
}


# ===========================================================================
print('\n=== T1 : Présence de F_deriv_handles ===')
# ===========================================================================
fm_L_seq = fit_silent(X_L, Y_aug_L, opts_seq, marg_legendre, cop)
gepck_oo = fm_L_seq['Kriging'][0]
ok('T1a  F_deriv_handles présent',
   'F_deriv_handles' in gepck_oo)
ok('T1b  longueur == Mred',
   len(gepck_oo.get('F_deriv_handles', [])) == fm_L_seq['Mred'],
   f"len={len(gepck_oo.get('F_deriv_handles', []))}  Mred={fm_L_seq['Mred']}")
ok('T1c  handles sont callables',
   all(callable(h) for h in gepck_oo.get('F_deriv_handles', [])))


# ===========================================================================
print('\n=== T2 : Shape des handles ===')
# ===========================================================================
U_test = rng.uniform(-1, 1, (5, fm_L_seq['Mred']))

for der in range(fm_L_seq['Mred']):
    dPsi = gepck_oo['F_deriv_handles'][der](U_test)          # (N_test, P)
    Psi  = gepck_oo['F_global_handle'](U_test)[:5, :]        # (N_test, P) — bloc 0
    ok(f'T2  der={der}  shape dPsi={dPsi.shape} == Psi.shape={Psi.shape}',
       dPsi.shape == Psi.shape,
       f'dPsi={dPsi.shape}  Psi={Psi.shape}')


# ===========================================================================
print('\n=== T3 : FD vs analytique — mode sequential, Legendre ===')
# ===========================================================================
N_test = 8
U_tst  = rng.uniform(-1, 1, (N_test, fm_L_seq['Mred']))

# block0 : Ψ(U) = F_global_handle(U)[:N_test, :]
block0 = lambda U: gepck_oo['F_global_handle'](U)[:N_test, :]

all_pass_T3 = True
for der in range(fm_L_seq['Mred']):
    analytic = gepck_oo['F_deriv_handles'][der](U_tst)
    numerical = fd_psi(block0, U_tst, der)
    err = np.abs(analytic - numerical).max()
    ok(f'T3  der={der}  err_FD={err:.2e} < 1e-7',
       err < 1e-7, f'err={err:.2e}')
    all_pass_T3 &= (err < 1e-7)


# ===========================================================================
print('\n=== T4 : FD vs analytique — mode sequential, Hermite ===')
# ===========================================================================
fm_H_seq = fit_silent(X_H, Y_aug_H, opts_seq, marg_hermite, cop)
gepck_H  = fm_H_seq['Kriging'][0]
Mred_H   = fm_H_seq['Mred']

U_tst_H  = rng.standard_normal((N_test, Mred_H))
block0_H = lambda U: gepck_H['F_global_handle'](U)[:N_test, :]

for der in range(Mred_H):
    analytic  = gepck_H['F_deriv_handles'][der](U_tst_H)
    numerical = fd_psi(block0_H, U_tst_H, der)
    err = np.abs(analytic - numerical).max()
    ok(f'T4  der={der}  err_FD={err:.2e} < 1e-7',
       err < 1e-7, f'err={err:.2e}')


# ===========================================================================
print('\n=== T5 : FD vs analytique — mode optimal, Matern-5/2 ===')
# ===========================================================================
fm_L_opt = fit_silent(X_L, Y_aug_L, opts_opt, marg_legendre, cop)
gepck_opt = fm_L_opt['Kriging'][0]
Mred_opt  = fm_L_opt['Mred']

U_tst_opt  = rng.uniform(-1, 1, (N_test, Mred_opt))
block0_opt = lambda U: gepck_opt['F_global_handle'](U)[:N_test, :]

for der in range(Mred_opt):
    analytic  = gepck_opt['F_deriv_handles'][der](U_tst_opt)
    numerical = fd_psi(block0_opt, U_tst_opt, der)
    err = np.abs(analytic - numerical).max()
    ok(f'T5  der={der}  err_FD={err:.2e} < 1e-7',
       err < 1e-7, f'err={err:.2e}')


# ===========================================================================
print('\n=== T6 : dPsi != Psi (cohérence : la dérivée change quelque chose) ===')
# ===========================================================================
Psi  = block0(U_tst)
for der in range(fm_L_seq['Mred']):
    dPsi = gepck_oo['F_deriv_handles'][der](U_tst)
    ok(f'T6  der={der}  dPsi != Psi',
       not np.allclose(dPsi, Psi, atol=1e-10))


# ---------------------------------------------------------------------------
print()
print(f'=== BILAN : {PASS} PASS  {FAIL} FAIL ===')
