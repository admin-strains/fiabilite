"""
test_predict_deriv_gepck.py
===========================
Teste uq_GEPCK_eval_one_output_deriv, uq_GEPCK_eval_deriv,
predict_deriv_gepck, predict_gradient_gepck par différences finies.

Stratégie : marginals Uniform(-1,1) → transform isoprobabiliste = identité
→ ∂ŷ/∂u_i = ∂ŷ/∂x_i  →  FD en X-space == gradient analytique en U-space.

Tests :
  T1 — predict_deriv_gepck  vs FD(predict_gepck)  [Legendre, mode sequential]
  T2 — predict_deriv_gepck  vs FD(predict_gepck)  [Legendre, mode optimal]
  T3 — predict_gradient_gepck : colonnes == predict_deriv_gepck der=0..Mred-1
  T4 — predict_gradient_gepck shape = (N_test, Mred)
  T5 — uq_GEPCK_eval_one_output_deriv == predict_deriv_gepck (cohérence pipeline)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import warnings

from branche1 import fit_gepck, predict_gepck, predict_deriv_gepck, predict_gradient_gepck
from branche4 import uq_GEPCK_eval_one_output_deriv

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


def fit_silent(X, Y_aug, opts, marg, cop):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fit_gepck(X, Y_aug, opts, marg, cop)


def fd_predict(fm, X_test, der, eps=EPS):
    """FD de predict_gepck en X-space dans la direction der."""
    X_plus  = X_test.copy(); X_plus[:,  der] += eps
    X_minus = X_test.copy(); X_minus[:, der] -= eps
    y_plus  = predict_gepck(fm, X_plus)[:, 0]
    y_minus = predict_gepck(fm, X_minus)[:, 0]
    return (y_plus - y_minus) / (2 * eps)


# ---------------------------------------------------------------------------
# Données — M=2, Uniform(-1,1)  →  u_i = x_i  (Legendre, identité)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(13)
N   = 12
M   = 2

marg = [{'Type': 'Uniform', 'Parameters': [-1.0, 1.0]} for _ in range(M)]
cop  = {'Type': 'Independent', 'Parameters': np.eye(M)}

X_tr = rng.uniform(-1, 1, (N, M))

def f(X): return X[:, 0]**3 + X[:, 0] * X[:, 1] + 0.5 * X[:, 1]**2

def gf(X):
    G = np.zeros_like(X)
    G[:, 0] = 3 * X[:, 0]**2 + X[:, 1]
    G[:, 1] = X[:, 0] + X[:, 1]
    return G

Y_tr    = f(X_tr)
G_tr    = gf(X_tr)
Y_aug   = np.concatenate([Y_tr, G_tr[:, 0], G_tr[:, 1]])  # (N*(M+1),)

opts_seq = {
    'Mode': 'sequential',
    'PCE':  {'Degree': [1, 2, 3], 'Method': 'LARS'},
    'Kriging': {'Corr': {'Family': 'gaussian'}},
}
opts_opt = {
    'Mode': 'optimal',
    'PCE':  {'Degree': [1, 2], 'Method': 'LARS'},
    'Kriging': {'Corr': {'Family': 'matern-5_2'}},
}

X_test = rng.uniform(-1, 1, (6, M))

# ===========================================================================
print('\n=== T1 : predict_deriv_gepck vs FD — mode sequential, Legendre ===')
# ===========================================================================
fm_seq = fit_silent(X_tr, Y_aug, opts_seq, marg, cop)
Mred   = fm_seq['Mred']

for der in range(Mred):
    analytic  = predict_deriv_gepck(fm_seq, X_test, der)[:, 0]
    numerical = fd_predict(fm_seq, X_test, der)
    err = np.abs(analytic - numerical).max()
    ok(f'T1  der={der}  err_FD={err:.2e} < 1e-6',
       err < 1e-6, f'err={err:.2e}')


# ===========================================================================
print('\n=== T2 : predict_deriv_gepck vs FD — mode optimal, Matern-5/2 ===')
# ===========================================================================
fm_opt = fit_silent(X_tr, Y_aug, opts_opt, marg, cop)
Mred_o = fm_opt['Mred']

for der in range(Mred_o):
    analytic  = predict_deriv_gepck(fm_opt, X_test, der)[:, 0]
    numerical = fd_predict(fm_opt, X_test, der)
    err = np.abs(analytic - numerical).max()
    ok(f'T2  der={der}  err_FD={err:.2e} < 1e-6',
       err < 1e-6, f'err={err:.2e}')


# ===========================================================================
print('\n=== T3 : predict_gradient_gepck colonnes == predict_deriv_gepck ===')
# ===========================================================================
G = predict_gradient_gepck(fm_seq, X_test)   # (N_test, Mred)
for der in range(Mred):
    col      = G[:, der]
    expected = predict_deriv_gepck(fm_seq, X_test, der)[:, 0]
    err = np.abs(col - expected).max()
    ok(f'T3  col[{der}] == predict_deriv(der={der})',
       err < 1e-14, f'err={err:.2e}')


# ===========================================================================
print('\n=== T4 : predict_gradient_gepck shape ===')
# ===========================================================================
ok('T4  shape (N_test, Mred)',
   G.shape == (X_test.shape[0], Mred),
   f'shape={G.shape}  attendu=({X_test.shape[0]}, {Mred})')


# ===========================================================================
print('\n=== T5 : uq_GEPCK_eval_one_output_deriv == predict_deriv_gepck ===')
# Cohérence pipeline : la fonction bas niveau donne le même résultat
# qu'après la transformation X→U (ici identité pour Uniform(-1,1)).
# ===========================================================================
from branche5 import uq_GeneralIsopTransform

gepck_oo = fm_seq['Kriging'][0]
U_train  = fm_seq['ExpDesign']['U']
Y_aug_fm = fm_seq['ExpDesign']['Y_aug']
F_tilde  = gepck_oo['F_tilde']
CorrOpts = fm_seq['CorrOptions']
Mred_fm  = fm_seq['Mred']
nonConst = fm_seq['nonConst']

red_marg = fm_seq['RedMarginals']
aux_marg = fm_seq['AuxSpace']['Marginals']
aux_cop  = fm_seq['AuxSpace']['Copula']
red_cop  = {'Type': 'Independent', 'Parameters': np.eye(Mred_fm)}
U_test   = uq_GeneralIsopTransform(
    X_test[:, nonConst], red_marg, red_cop, aux_marg, aux_cop)

for der in range(Mred_fm):
    low_level  = uq_GEPCK_eval_one_output_deriv(
        gepck_oo, U_test, U_train, Y_aug_fm, F_tilde, CorrOpts, der)
    high_level = predict_deriv_gepck(fm_seq, X_test, der)[:, 0]
    err = np.abs(low_level - high_level).max()
    ok(f'T5  der={der}  low_level == high_level  err={err:.2e}',
       err < 1e-14, f'err={err:.2e}')


# ---------------------------------------------------------------------------
print()
print(f'=== BILAN : {PASS} PASS  {FAIL} FAIL ===')
