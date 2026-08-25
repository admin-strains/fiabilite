"""
test_branche_lars.py — Tests for branche_lars.py (LARS module)
Covers uq_PCE_loo_error, uq_blockwise_inverse, uq_PCE_OLS_regression, uq_lar.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import warnings
from branche_lars import (
    uq_PCE_loo_error, uq_blockwise_inverse,
    uq_PCE_OLS_regression, uq_lar, uq_PCE_lars
)

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


# ===========================================================================
# SECTION 1 — uq_blockwise_inverse
# ===========================================================================
print('\n=== Section 1 : uq_blockwise_inverse ===')

# 1a. 1x1 + scalar expansion -> 2x2, verify with numpy inv
# NOTE: uq_blockwise_inverse takes Ainv = inv(A), not A itself
A      = np.array([[4.0]])
Ainv_a = np.linalg.inv(A)          # [[0.25]]
B      = np.array([[2.0]])
C      = B.T
D      = np.array([[5.0]])
M_full     = np.block([[A, B], [C, D]])
M_inv_ref  = np.linalg.inv(M_full)
M_inv_got  = uq_blockwise_inverse(Ainv_a, B, C, D)
ok('1a 2x2 blockwise vs direct inv', np.allclose(M_inv_ref, M_inv_got, atol=1e-12))

# 1b. 2x2 + 1 scalar expansion -> 3x3
A2 = M_inv_got                 # already the 2x2 inverse
B2 = np.array([[3.0], [1.0]])
C2 = B2.T
D2 = np.array([[7.0]])
M_full3 = np.block([
    [np.linalg.inv(A2), B2],
    [C2,                D2],
])
M_inv3_ref = np.linalg.inv(M_full3)
M_inv3_got = uq_blockwise_inverse(A2, B2, C2, D2)
ok('1b 3x3 blockwise vs direct inv', np.allclose(M_inv3_ref, M_inv3_got, atol=1e-12))

# 1c. symmetry of output
ok('1c output is symmetric', np.allclose(M_inv_got, M_inv_got.T, atol=1e-14))


# ===========================================================================
# SECTION 2 — uq_PCE_loo_error
# ===========================================================================
print('\n=== Section 2 : uq_PCE_loo_error ===')

rng = np.random.default_rng(42)
N, P2 = 30, 4
Psi2 = rng.standard_normal((N, P2))
coeff_true = np.array([2.0, -1.0, 0.5, 3.0])
Y2 = Psi2 @ coeff_true

PsiTPsi2 = Psi2.T @ Psi2
M2 = np.linalg.solve(PsiTPsi2, np.eye(P2))
c2 = np.linalg.solve(PsiTPsi2, Psi2.T @ Y2)

# 2a. with true coefficients -> residual near 0
loo_a, emp_a, opt_a = uq_PCE_loo_error(Psi2, M2, Y2, c2, modified_flag=False)
ok('2a near-zero residual -> small LOO', loo_a < 1e-20, f'loo={loo_a:.2e}')

# 2b. empty coeff -> recomputed by OLS -> same as direct
loo_b, emp_b, opt_b = uq_PCE_loo_error(Psi2, M2, Y2, None, modified_flag=False)
ok('2b None coeff same as direct', abs(loo_a - loo_b) < 1e-12)

# 2c. modified_flag=True multiplies by T (> 1 when N > P)
loo_mod, _, opt_mod = uq_PCE_loo_error(Psi2, M2, Y2, c2, modified_flag=True)
T_expected = opt_mod['T']
ok('2c T > 1 when N>>P', T_expected > 1.0)
ok('2c loo_mod = T * loo_unmod', abs(loo_mod - T_expected * loo_a) < 1e-10)

# 2d. zero-variance Y -> loo = normEmpErr = 0
Y_const = np.ones(N) * 5.0
loo_d, emp_d, _ = uq_PCE_loo_error(Psi2, M2, Y_const, c2, modified_flag=False)
ok('2d zero-var Y -> loo=0', loo_d == 0.0)
ok('2d zero-var Y -> normEmpErr=0', emp_d == 0.0)

# 2e. modi_diag shifts h
modi = np.full(N, 1.0 / N)
loo_e, _, _ = uq_PCE_loo_error(Psi2, M2, Y2, c2, modified_flag=False,
                                 modi_diag=modi)
# LOO with modi_diag != LOO without (in general)
ok('2e modi_diag changes loo', loo_e != loo_a)

# 2f. opt_results contains required keys
required_keys = {'T', 'loo', 'ModifiedLoo', 'normEmpErr', 'LooPred'}
ok('2f opt_results has required keys', required_keys.issubset(opt_a.keys()))


# ===========================================================================
# SECTION 3 — uq_PCE_OLS_regression
# ===========================================================================
print('\n=== Section 3 : uq_PCE_OLS_regression ===')

rng = np.random.default_rng(7)
N3, P3 = 40, 5
Psi3 = rng.standard_normal((N3, P3))
c_true3 = np.array([1.0, 2.0, -1.0, 0.5, -0.3])
Y3 = Psi3 @ c_true3 + 0.01 * rng.standard_normal(N3)

res3 = uq_PCE_OLS_regression(Psi3, Y3)
ok('3a coefficients close to truth', np.allclose(res3['coefficients'], c_true3, atol=0.1))
ok('3b LOO is scalar in (0,1]', 0.0 <= res3['LOO'] <= 1.0)
ok('3c normEmpErr is non-negative', res3['normEmpErr'] >= 0.0)
ok('3d optErrorParams is dict', isinstance(res3['optErrorParams'], dict))

# 3e. ill-conditioned matrix -> pinv path
Psi3b = np.column_stack([Psi3, Psi3[:, 0]])  # duplicate column -> singular
res3b = uq_PCE_OLS_regression(Psi3b, Y3)
ok('3e ill-conditioned: no crash', True)
ok('3f LOO finite', np.isfinite(res3b['LOO']))


# ===========================================================================
# SECTION 4 — uq_lar  trivial cases
# ===========================================================================
print('\n=== Section 4 : uq_lar P==1 ===')

rng = np.random.default_rng(3)
N4 = 20
Psi4 = rng.standard_normal((N4, 1))
c4 = np.array([3.0])
Y4 = Psi4.ravel() * c4[0] + 0.01 * rng.standard_normal(N4)

r4 = uq_lar(Psi4, Y4)
ok('4a coeff[0] close to 3', abs(r4['coefficients'][0] - 3.0) < 0.5)
ok('4b nz_idx all True', r4['nz_idx'][0])
ok('4c lars_idx = [0]', r4['lars_idx'] == [0])
ok('4d best_basis_index = 0', r4['best_basis_index'] == 0)


# ===========================================================================
# SECTION 5 — uq_lar  known sparse signal recovery
# ===========================================================================
print('\n=== Section 5 : uq_lar sparse recovery ===')

rng  = np.random.default_rng(123)
N5   = 80
P5   = 20
# generate orthonormal-ish Psi
Psi5 = rng.standard_normal((N5, P5))
# true signal uses only columns 0, 5, 12
true_cols   = [0, 5, 12]
true_coeffs = [4.0, -2.0, 1.5]
Y5 = Psi5[:, true_cols] @ np.array(true_coeffs) + 0.01 * rng.standard_normal(N5)

r5 = uq_lar(Psi5, Y5, {'normalize': True, 'hybrid_lars': True,
                         'loo_modified': True, 'loo_hybrid': True,
                         'early_stop': False})

ok('5a all 3 true cols are selected',
   all(r5['nz_idx'][c] for c in true_cols),
   f"nz_idx={np.where(r5['nz_idx'])[0].tolist()}")
ok('5b coefficients close to truth (col 0)',
   abs(r5['coefficients'][0] - 4.0) < 0.5)
ok('5c LOO finite and in (0,1)',
   0.0 <= r5['LOO'] <= 1.0, f"LOO={r5['LOO']:.4f}")
ok('5d LOO_lars = 1 - max_score',
   abs(r5['LOO_lars'] - (1.0 - r5['max_score'])) < 1e-12)
ok('5e a_scores size = nvars+1',
   len(r5['a_scores']) == min(N5 - 2, P5) + 1)
ok('5f coeff_array first row all zero',
   np.all(r5['coeff_array'][0, :] == 0.0))
ok('5g nz_idx matches nonzero in best row of coeff_array',
   np.array_equal(
       np.abs(r5['coeff_array'][r5['best_basis_index'], :]) > 0,
       r5['nz_idx']))


# ===========================================================================
# SECTION 6 — uq_lar  with constant regressor (centering path)
# ===========================================================================
print('\n=== Section 6 : uq_lar with constant regressor ===')

rng = np.random.default_rng(55)
N6, P6 = 50, 8
Psi6 = rng.standard_normal((N6, P6 - 1))
Psi6 = np.column_stack([np.ones(N6), Psi6])  # constant col first
true_c6 = np.array([2.0, 1.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
Y6 = Psi6 @ true_c6 + 0.01 * rng.standard_normal(N6)

r6 = uq_lar(Psi6, Y6, {'normalize': True, 'hybrid_lars': True,
                         'loo_modified': True, 'loo_hybrid': True,
                         'early_stop': False})

ok('6a constant col (0) in nz_idx', bool(r6['nz_idx'][0]))
ok('6b constant col (0) in lars_idx', 0 in r6['lars_idx'])
ok('6c intercept coeff ~2', abs(r6['coefficients'][0] - 2.0) < 0.3,
   f"coeff[0]={r6['coefficients'][0]:.3f}")
ok('6d col 1 coeff ~1', abs(r6['coefficients'][1] - 1.0) < 0.3,
   f"coeff[1]={r6['coefficients'][1]:.3f}")
ok('6e LOO finite', np.isfinite(r6['LOO']))


# ===========================================================================
# SECTION 7 — uq_lar  no normalization
# ===========================================================================
print('\n=== Section 7 : uq_lar normalize=False ===')

r7 = uq_lar(Psi5, Y5, {'normalize': False, 'hybrid_lars': True,
                          'loo_modified': True, 'loo_hybrid': True,
                          'early_stop': False})
ok('7a no crash', True)
ok('7b true cols selected without normalize',
   all(r7['nz_idx'][c] for c in true_cols),
   f"nz_idx={np.where(r7['nz_idx'])[0].tolist()}")


# ===========================================================================
# SECTION 8 — uq_lar  early stop
# ===========================================================================
print('\n=== Section 8 : uq_lar early stop ===')

rng8 = np.random.default_rng(999)
N8, P8 = 60, 30
Psi8 = rng8.standard_normal((N8, P8))
Y8   = Psi8[:, 0] * 3 + rng8.standard_normal(N8) * 0.01

r8_stop   = uq_lar(Psi8, Y8, {'early_stop': True,  'normalize': True})
r8_nostop = uq_lar(Psi8, Y8, {'early_stop': False, 'normalize': True})
ok('8a early stop <= no stop iterations',
   r8_stop['best_basis_index'] <= r8_nostop['best_basis_index'])
ok('8b LOO finite with early stop', np.isfinite(r8_stop['LOO']))


# ===========================================================================
# SECTION 9 — uq_PCE_lars wrapper
# ===========================================================================
print('\n=== Section 9 : uq_PCE_lars wrapper ===')

r9 = uq_PCE_lars(Psi5, Y5)
ok('9a wrapper produces same nz_idx as uq_lar',
   np.array_equal(r9['nz_idx'], r5['nz_idx']))
ok('9b lars_idx is list', isinstance(r9['lars_idx'], list))


# ===========================================================================
# SECTION 10 — hybrid_lars=False path
# ===========================================================================
print('\n=== Section 10 : hybrid_lars=False ===')

r10 = uq_lar(Psi5, Y5, {'hybrid_lars': False, 'normalize': True,
                          'loo_modified': True, 'loo_hybrid': True,
                          'early_stop': False})
ok('10a no crash hybrid=False', True)
ok('10b LOO finite', np.isfinite(r10['LOO']))
ok('10c true cols still selected',
   all(r10['nz_idx'][c] for c in true_cols),
   f"nz_idx={np.where(r10['nz_idx'])[0].tolist()}")


# ===========================================================================
# SUMMARY
# ===========================================================================
print(f'\n{"="*50}')
print(f'Results : {PASS} PASS  |  {FAIL} FAIL')
if FAIL == 0:
    print('All tests PASSED.')
else:
    print('Some tests FAILED -- see above.')
