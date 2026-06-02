"""
test_branche3.py — Tests for branche3.py (Kriging + PCK module)

Covers:
  1. uq_Kriging_calc_DiagOfCongruent
  2. uq_Kriging_calc_auxMatrices  (3 runCases)
  3. uq_Kriging_calc_beta         (QR, standard)
  4. uq_Kriging_calc_sigmaSq      (ml_chol, ml_bypass_chol, ml_nochol)
  5. uq_Kriging_calc_KFold        (LOO, K-fold)
  6. uq_Kriging_eval_J_of_theta_ML
  7. kriging_optimize_theta
  8. PCE utilities  (pce_multi_indices, pce_eval_design_matrix)
  9. fit_kriging_pck
 10. uq_PCK_calculate_coefficients  sequential
 11. uq_PCK_calculate_coefficients  optimal
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import warnings

from branche3 import (
    uq_Kriging_calc_DiagOfCongruent,
    uq_Kriging_calc_auxMatrices,
    uq_Kriging_calc_beta,
    uq_Kriging_calc_sigmaSq,
    uq_Kriging_calc_KFold,
    uq_Kriging_helper_create_randIdx,
    uq_Kriging_eval_J_of_theta_ML,
    kriging_optimize_theta,
    fit_kriging_pck,
    pce_multi_indices,
    pce_eval_design_matrix,
    uq_PCK_calculate_coefficients,
    poly_type_from_marginal,
    aux_marginal_from_poly_type,
)
from branche5 import uq_eval_Kernel

PASS = 0
FAIL = 0

def ok(label, condition, info=''):
    global PASS, FAIL
    if condition:
        print(f'  PASS  {label}')
        PASS += 1
    else:
        print(f'  FAIL  {label}' + (f'  [{info}]' if info else ''))
        FAIL += 1


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)

def make_spd(n, rng_):
    """Generate a symmetric positive definite n×n matrix."""
    A = rng_.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)

def default_corr_opts(M):
    return {
        'Handle'    : uq_eval_Kernel,
        'Family'    : 'matern-5_2',
        'Type'      : 'separable',
        'Isotropic' : False,
        'Nugget'    : 0.0,
    }

def make_kriging_data(N, M, rng_, noise=0.0):
    """N training points in [-1,1]^M, output = sum(x^2) + noise."""
    X = rng_.uniform(-1, 1, (N, M))
    Y = np.sum(X**2, axis=1) + noise * rng_.standard_normal(N)
    return X, Y

def make_F(U, degree=1):
    """Trend matrix: constant + linear terms (degree=1) or constant only (0)."""
    N = U.shape[0]
    if degree == 0:
        return np.ones((N, 1))
    return np.column_stack([np.ones(N), U])   # (N, M+1) for M=1 input


# ===========================================================================
# SECTION 1 — uq_Kriging_calc_DiagOfCongruent
# ===========================================================================
print('\n=== Section 1 : uq_Kriging_calc_DiagOfCongruent ===')

B3 = make_spd(3, rng)
A3 = rng.standard_normal((4, 3))    # A: (N2=4, N1=3)

# Reference: diag(A B^{-1} A^T) computed directly
diag_ref = np.diag(A3 @ np.linalg.inv(B3) @ A3.T)
diag_got = uq_Kriging_calc_DiagOfCongruent(A3, B3)

ok('1a shape is (N2,)', diag_got.shape == (4,))
ok('1b values match direct computation', np.allclose(diag_ref, diag_got, atol=1e-10))
ok('1c all non-negative (A B^{-1} A^T is PSD)', np.all(diag_got >= -1e-12))

# 1d. Square case A == sqrt of B (diag should equal 1)
Bsq = make_spd(3, rng)
L_sq = np.linalg.cholesky(Bsq)     # L @ L.T = B, so L.T @ L.T^{-T} @ ... = I?
# diag(L^T B^{-1} L) = diag(I) = 1 for A = L.T
diag_sq = uq_Kriging_calc_DiagOfCongruent(L_sq.T, Bsq)
ok('1d A = L^T : diag = 1', np.allclose(diag_sq, np.ones(3), atol=1e-10))


# ===========================================================================
# SECTION 2 — uq_Kriging_calc_auxMatrices
# ===========================================================================
print('\n=== Section 2 : uq_Kriging_calc_auxMatrices ===')

N2, M2 = 15, 2
X2, Y2 = make_kriging_data(N2, M2, rng)
corr2   = default_corr_opts(M2)
theta2  = np.array([1.0, 1.0])
R2  = uq_eval_Kernel(X2, X2, theta2, corr2)
F2  = make_F(X2, degree=1)          # (N, M+1) = (15, 3)
Yc2 = Y2.copy()                     # column-shaped OK as 1D

# 'default' runCase
am_def = uq_Kriging_calc_auxMatrices(R2, F2, Yc2, 'default')
ok('2a default: cholR present',      am_def['cholR'] is not None)
ok('2b default: FTRinv shape',       am_def['FTRinv'].shape == (F2.shape[1], N2))
ok('2c default: FTRinvF = FTRinv@F', np.allclose(am_def['FTRinvF'], am_def['FTRinv'] @ F2, atol=1e-12))
ok('2d default: no Ytilde',          'Ytilde' not in am_def)

# 'ml_optimization' runCase
am_opt = uq_Kriging_calc_auxMatrices(R2, F2, Yc2, 'ml_optimization')
ok('2e ml_opt: Ytilde present',      am_opt.get('Ytilde') is not None)
ok('2f ml_opt: Q1 shape (N, P)',     am_opt['Q1'].shape == (N2, F2.shape[1]))
ok('2g ml_opt: no FTRinv',           'FTRinv' not in am_opt)

# 'ml_estimation' runCase
am_est = uq_Kriging_calc_auxMatrices(R2, F2, Yc2, 'ml_estimation')
ok('2h ml_est: has all keys',
   all(k in am_est for k in ('cholR','Rinv','FTRinv','FTRinvF','Ytilde','Q1','G')))

# 2i. Rinv field: non-None only when Cholesky fails (not here)
ok('2i Rinv is None when chol ok',   am_def['Rinv'] is None)

# 2j. Q1^T Q1 ~= I (orthogonality)
ok('2j Q1 orthogonal',               np.allclose(am_est['Q1'].T @ am_est['Q1'],
                                                  np.eye(F2.shape[1]), atol=1e-12))


# ===========================================================================
# SECTION 3 — uq_Kriging_calc_beta
# ===========================================================================
print('\n=== Section 3 : uq_Kriging_calc_beta ===')

# 3a. QR method: result should satisfy F^T R^{-1} F beta = F^T R^{-1} Y (GLS normal eqs)
beta_qr  = uq_Kriging_calc_beta(F2, 'ordinary', Yc2, 'qr', am_est)
lhs_qr   = am_est['FTRinvF'] @ beta_qr
rhs_qr   = am_est['FTRinv'] @ Yc2
ok('3a QR: GLS normal equations satisfied',
   np.allclose(lhs_qr, rhs_qr, atol=1e-8))

# 3b. Standard method: same result as QR
beta_std = uq_Kriging_calc_beta(F2, 'ordinary', Yc2, 'standard', am_est)
ok('3b standard: close to QR result', np.allclose(beta_qr, beta_std, atol=1e-8))

# 3c. Simple Kriging: beta = ones
beta_sk  = uq_Kriging_calc_beta(F2, 'simple', Yc2, 'qr', am_est)
ok('3c simple kriging: beta = 1', np.allclose(beta_sk, np.ones(F2.shape[1]), atol=1e-14))

# 3d. QR fallback to standard when Q1 not available
am_noQ1 = {k: v for k, v in am_def.items()}   # am_def has no Ytilde/Q1
beta_fb  = uq_Kriging_calc_beta(F2, 'ordinary', Yc2, 'qr', am_noQ1)
ok('3d fallback to standard: close to QR', np.allclose(beta_qr, beta_fb, atol=1e-8))


# ===========================================================================
# SECTION 4 — uq_Kriging_calc_sigmaSq
# ===========================================================================
print('\n=== Section 4 : uq_Kriging_calc_sigmaSq ===')

beta4 = beta_qr

# ml_chol
kp_chol = {
    'N'     : N2,
    'beta'  : beta4,
    'Ytilde': am_est['Ytilde'],
    'Ftilde': am_est['Ftilde'],
}
sigma_chol = uq_Kriging_calc_sigmaSq(kp_chol, 'ml_chol')
ok('4a ml_chol: positive', sigma_chol > 0)
ok('4b ml_chol: finite',   np.isfinite(sigma_chol))

# ml_bypass_chol (same formula but bypasses beta)
kp_bypass = {
    'N'     : N2,
    'Q1'    : am_est['Q1'],
    'Ytilde': am_est['Ytilde'],
}
sigma_bypass = uq_Kriging_calc_sigmaSq(kp_bypass, 'ml_bypass_chol')
ok('4c ml_bypass_chol: positive', sigma_bypass > 0)
ok('4d ml_bypass_chol ~= ml_chol', np.isclose(sigma_chol, sigma_bypass, rtol=1e-6))

# ml_nochol (needs Rinv)
Rinv2    = np.linalg.inv(R2)
kp_nochol = {
    'N'   : N2,
    'Y'   : Yc2,
    'F'   : F2,
    'Rinv': Rinv2,
    'beta': beta4,
}
sigma_nochol = uq_Kriging_calc_sigmaSq(kp_nochol, 'ml_nochol')
ok('4e ml_nochol: finite', np.isfinite(sigma_nochol))
ok('4f ml_nochol ~= ml_chol (same math)', np.isclose(sigma_chol, sigma_nochol, rtol=1e-6))


# ===========================================================================
# SECTION 5 — uq_Kriging_calc_KFold
# ===========================================================================
print('\n=== Section 5 : uq_Kriging_calc_KFold ===')

# LOO case (K = 1 per fold)
randIdx_loo = uq_Kriging_helper_create_randIdx(1, N2)
ok('5a randIdx_loo has N folds', len(randIdx_loo) == N2)

CVE_loo, CVS_loo = uq_Kriging_calc_KFold(randIdx_loo, Yc2, F2, am_def)
ok('5b LOO errors: N values',       len(CVE_loo) == N2)
ok('5c LOO errors: all finite',     all(np.isfinite(e) for e in CVE_loo))
ok('5d LOO variances: all positive', all(v > 0 for v in CVS_loo))

# LOO relative error (normalized by varY)
varY2 = float(np.var(Yc2, ddof=0))
LOO2  = float(np.mean(np.array(CVE_loo))) / varY2
ok('5e LOO in (0, 1)', 0.0 < LOO2 < 1.0, f'LOO={LOO2:.4f}')

# K-fold (K = 3)
randIdx_k3 = uq_Kriging_helper_create_randIdx(3, N2)
ok('5f K=3: 3 folds', len(randIdx_k3) == 3)
CVE_k3, CVS_k3 = uq_Kriging_calc_KFold(randIdx_k3, Yc2, F2, am_def)
ok('5g K=3 errors finite', all(np.all(np.isfinite(e)) for e in CVE_k3))


# ===========================================================================
# SECTION 6 — uq_Kriging_eval_J_of_theta_ML
# ===========================================================================
print('\n=== Section 6 : uq_Kriging_eval_J_of_theta_ML ===')

kmp = {
    'X'           : X2,
    'Y'           : Yc2,
    'N'           : N2,
    'F'           : F2,
    'CorrOptions' : corr2,
    'trend_type'  : 'ordinary',
    'IsRegression': False,
    'EstimNoise'  : False,
}
J_at_1 = uq_Kriging_eval_J_of_theta_ML(theta2, kmp)
ok('6a J finite at theta=(1,1)',   np.isfinite(J_at_1))
ok('6b J is a real number',          np.isfinite(J_at_1) and not np.isnan(J_at_1))

J_at_bad = uq_Kriging_eval_J_of_theta_ML(np.array([1e-15, 1e-15]), kmp)
ok('6c J finite at near-zero theta (catches exception)', np.isfinite(J_at_bad))

# J should decrease (or at least be different) when theta changes
J_at_10 = uq_Kriging_eval_J_of_theta_ML(np.array([10.0, 10.0]), kmp)
ok('6d J(theta=10) is finite', np.isfinite(J_at_10))
ok('6e J(theta=1) != J(theta=10)', J_at_1 != J_at_10)


# ===========================================================================
# SECTION 7 — kriging_optimize_theta
# ===========================================================================
print('\n=== Section 7 : kriging_optimize_theta ===')

bounds7 = np.array([[0.05, 0.05], [20.0, 20.0]])

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    theta7, J7, flag7 = kriging_optimize_theta(kmp, theta2.copy(), bounds7,
                                                method='gradbased')

ok('7a theta_opt shape', theta7.shape == (2,))
ok('7b J_opt finite',    np.isfinite(J7))
ok('7c J_opt <= J_init', J7 <= J_at_1 + 1e-6)   # optimizer should not worsen J
ok('7d theta in bounds', np.all(theta7 >= bounds7[0] - 1e-6) and
                          np.all(theta7 <= bounds7[1] + 1e-6))


# ===========================================================================
# SECTION 8 — PCE utilities
# ===========================================================================
print('\n=== Section 8 : PCE utilities ===')

# pce_multi_indices
idx_1d_3 = pce_multi_indices(1, 3)
ok('8a M=1, D=3: shape (4,1)',   idx_1d_3.shape == (4, 1))
ok('8b M=1, D=3: values 0..3',  np.array_equal(idx_1d_3.ravel(), [0, 1, 2, 3]))

idx_2d_2 = pce_multi_indices(2, 2)
ok('8c M=2, D=2: 6 terms',      idx_2d_2.shape[0] == 6)
ok('8d M=2, D=2: total <= 2',   np.all(idx_2d_2.sum(axis=1) <= 2))
ok('8e M=2, D=2: first is (0,0)', np.array_equal(idx_2d_2[0], [0, 0]))

idx_3d_4 = pce_multi_indices(3, 4)
# count = C(3+4, 3) = C(7,3) = 35
ok('8f M=3, D=4: 35 terms',     idx_3d_4.shape[0] == 35)

# poly_type_from_marginal
ok('8g Uniform -> Legendre',   poly_type_from_marginal('Uniform') == 'Legendre')
ok('8h Gaussian -> Hermite',   poly_type_from_marginal('Gaussian') == 'Hermite')

# aux_marginal_from_poly_type
am_l = aux_marginal_from_poly_type('Legendre')
ok('8i Legendre -> Uniform(-1,1)',
   am_l['Type'] == 'Uniform' and am_l['Parameters'][0] == -1.0)

# pce_eval_design_matrix: 1D Uniform(-1,1), degree 2
N8   = 10
X8   = rng.uniform(-1, 1, (N8, 1))
idx8 = pce_multi_indices(1, 2)
marg8   = [{'Type': 'Uniform', 'Parameters': [-1.0, 1.0]}]
cop8    = {'Type': 'Independent', 'Parameters': np.eye(1)}
auxm8   = [{'Type': 'Uniform', 'Parameters': [-1.0, 1.0]}]
Psi8, U8 = pce_eval_design_matrix(X8, idx8, ['Legendre'], marg8, cop8, auxm8)
ok('8j Psi shape (N, 3)',       Psi8.shape == (N8, 3))
ok('8k first col = P0 = 1',     np.allclose(Psi8[:, 0], np.ones(N8), atol=1e-12))
ok('8l U = X when aux == orig', np.allclose(U8, X8, atol=1e-12))


# ===========================================================================
# SECTION 9 — fit_kriging_pck
# ===========================================================================
print('\n=== Section 9 : fit_kriging_pck ===')

rng9 = np.random.default_rng(42)
N9   = 25
# 1D problem: f(u) = u^2 on [-1, 1]  (pure quadratic, no noise)
U9   = np.sort(rng9.uniform(-1, 1, (N9, 1)), axis=0)
Y9   = U9.ravel()**2

# Trend: constant + linear + quadratic  (F = [1, u, u^2])
def F9_handle(U):
    u = U.ravel()
    return np.column_stack([np.ones(len(u)), u, u**2])

corr9 = default_corr_opts(1)
theta9_bounds = np.array([[0.05], [10.0]])
theta9_0      = np.array([1.0])

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fit9 = fit_kriging_pck(U9, Y9, F9_handle, corr9,
                           theta9_bounds, theta9_0,
                           estim_method='ml', optim_method='gradbased')

ok('9a theta shape',    fit9['theta'].shape == (1,))
ok('9b sigmaSQ > 0',   fit9['sigmaSQ'] > 0)
ok('9c LOO finite',    np.isfinite(fit9['LOO']))
ok('9d LOO < 0.5',     fit9['LOO'] < 0.5, f"LOO={fit9['LOO']:.4f}")
ok('9e beta finite',   np.all(np.isfinite(fit9['beta'])))
ok('9f F shape (N,3)', fit9['F'].shape == (N9, 3))

# Prediction at U9 itself: trend part should be close to Y9
beta9  = fit9['beta']
F_pred = F9_handle(U9)
Y_trend = F_pred @ beta9
ok('9g trend prediction close to Y', np.allclose(Y_trend, Y9, atol=0.5))


# ===========================================================================
# SECTION 10 — uq_PCK_calculate_coefficients : sequential mode
# ===========================================================================
print('\n=== Section 10 : uq_PCK_calculate_coefficients sequential ===')

rng10 = np.random.default_rng(7)
N10, M10 = 40, 2
# f(x) = x1^2 + 0.5*x2 + 0.1*noise, inputs ~ Uniform(-1,1)
X10 = rng10.uniform(-1, 1, (N10, M10))
Y10 = X10[:, 0]**2 + 0.5 * X10[:, 1] + 0.05 * rng10.standard_normal(N10)

marg10 = [{'Type': 'Uniform', 'Parameters': [-1.0, 1.0]} for _ in range(M10)]
cop10  = {'Type': 'Independent', 'Parameters': np.eye(M10)}

pck_cfg10 = {
    'Mode'       : 'sequential',
    'TrendMethod': 'pce',
    'PCE'        : {'Degree': [1, 2, 3], 'Method': 'LARS'},
}

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fm10 = uq_PCK_calculate_coefficients(
        X10, Y10, pck_cfg10, marg10, cop10,
        optim_method='gradbased', estim_method='ml')

ok('10a Nout = 1',          fm10['Nout'] == 1)
ok('10b Mred = M',          fm10['Mred'] == M10)
ok('10c LOO finite',        np.isfinite(fm10['Error'][0]['LOO']))
ok('10d LOO in [0, 1]',     0.0 <= fm10['Error'][0]['LOO'] <= 1.0,
   f"LOO={fm10['Error'][0]['LOO']:.4f}")
ok('10e NumberOfPoly >= 1', fm10['NumberOfPoly'][0] >= 1)
ok('10f idxranking is list', isinstance(fm10['idxranking'][0], list))
ok('10g Kriging result has theta',
   'theta' in fm10['Kriging'][0])
ok('10h theta length = Mred',
   len(fm10['Kriging'][0]['theta']) == M10)


# ===========================================================================
# SECTION 11 — uq_PCK_calculate_coefficients : optimal mode
# ===========================================================================
print('\n=== Section 11 : uq_PCK_calculate_coefficients optimal ===')

pck_cfg11 = {
    'Mode'       : 'optimal',
    'TrendMethod': 'pce',
    'CombCrit'   : 'rel_loo',
    'PCE'        : {'Degree': [1, 2], 'Method': 'LARS'},   # smaller degree for speed
}

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fm11 = uq_PCK_calculate_coefficients(
        X10, Y10, pck_cfg11, marg10, cop10,
        optim_method='gradbased', estim_method='ml')

ok('11a LOO finite',      np.isfinite(fm11['Error'][0]['LOO']))
ok('11b LOO in [0, 1]',   0.0 <= fm11['Error'][0]['LOO'] <= 1.0,
   f"LOO={fm11['Error'][0]['LOO']:.4f}")
ok('11c NumberOfPoly >= 1', fm11['NumberOfPoly'][0] >= 1)


# ===========================================================================
# SECTION 12 — uq_PCK_calculate_coefficients : multi-output
# ===========================================================================
print('\n=== Section 12 : uq_PCK_calculate_coefficients multi-output ===')

rng12 = np.random.default_rng(99)
N12, M12 = 30, 1
X12 = rng12.uniform(-1, 1, (N12, M12))
Y12 = np.column_stack([X12.ravel()**2, X12.ravel()])   # 2 outputs

marg12 = [{'Type': 'Uniform', 'Parameters': [-1.0, 1.0]}]
cop12  = {'Type': 'Independent', 'Parameters': np.eye(1)}
pck_cfg12 = {
    'Mode'       : 'sequential',
    'TrendMethod': 'pce',
    'PCE'        : {'Degree': [1, 2], 'Method': 'LARS'},
}

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fm12 = uq_PCK_calculate_coefficients(
        X12, Y12, pck_cfg12, marg12, cop12,
        optim_method='gradbased', estim_method='ml')

ok('12a Nout = 2',         fm12['Nout'] == 2)
ok('12b 2 Kriging models', len(fm12['Kriging']) == 2)
ok('12c 2 Error entries',  len(fm12['Error']) == 2)
ok('12d LOO[0] finite',    np.isfinite(fm12['Error'][0]['LOO']))
ok('12e LOO[1] finite',    np.isfinite(fm12['Error'][1]['LOO']))


# ===========================================================================
# SUMMARY
# ===========================================================================
print(f'\n{"="*50}')
print(f'Results : {PASS} PASS  |  {FAIL} FAIL')
if FAIL == 0:
    print('All tests PASSED.')
else:
    print('Some tests FAILED -- see above.')
