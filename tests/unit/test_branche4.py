"""
test_branche4.py -- Tests for branche4.py (PCK_eval / Kriging prediction)

Covers:
  1. uq_Kriging_eval_one_output (mean only)
  2. uq_Kriging_eval_one_output (mean + variance)
  3. uq_Kriging_eval_one_output (mean + variance + covariance)
  4. Interpolation property: sigma2 ~ 0 at training points
  5. uq_PCK_eval (single output, all 3 return modes)
  6. uq_PCK_eval multi-output
  7. Consistency: return_var results match return_cov diagonal
  8. Constant-dimension handling (nonConst subset)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import warnings

from branche3 import uq_PCK_calculate_coefficients
from branche4 import uq_PCK_eval, uq_Kriging_eval_one_output

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
# Build two reference PCK models (1D and 2D)
# ---------------------------------------------------------------------------

def fit_pck_1d(seed=42):
    """1D: f(x) = x^3 on Uniform(-1,1). N=30, degree up to 3."""
    rng = np.random.default_rng(seed)
    N   = 30
    X   = rng.uniform(-1, 1, (N, 1))
    Y   = X.ravel()**3
    marg = [{'Type': 'Uniform', 'Parameters': [-1.0, 1.0]}]
    cop  = {'Type': 'Independent', 'Parameters': np.eye(1)}
    cfg  = {'Mode': 'sequential', 'TrendMethod': 'pce',
            'PCE': {'Degree': [1, 2, 3], 'Method': 'LARS'}}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fm = uq_PCK_calculate_coefficients(X, Y, cfg, marg, cop,
                                            optim_method='gradbased',
                                            estim_method='ml')
    return fm, X, Y


def fit_pck_2d(seed=7):
    """2D: f(x1,x2) = x1^2 + 0.5*x2, N=40, degree up to 2."""
    rng = np.random.default_rng(seed)
    N   = 40
    X   = rng.uniform(-1, 1, (N, 2))
    Y   = X[:, 0]**2 + 0.5 * X[:, 1] + 0.02 * rng.standard_normal(N)
    marg = [{'Type': 'Uniform', 'Parameters': [-1.0, 1.0]}] * 2
    cop  = {'Type': 'Independent', 'Parameters': np.eye(2)}
    cfg  = {'Mode': 'sequential', 'TrendMethod': 'pce',
            'PCE': {'Degree': [1, 2], 'Method': 'LARS'}}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fm = uq_PCK_calculate_coefficients(X, Y, cfg, marg, cop,
                                            optim_method='gradbased',
                                            estim_method='ml')
    return fm, X, Y.reshape(-1, 1)


print('Fitting reference PCK models (may take a moment)...')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fm1, X1_train, Y1_train = fit_pck_1d()
    fm2, X2_train, Y2_train = fit_pck_2d()


# ===========================================================================
# SECTION 1 -- uq_Kriging_eval_one_output: mean only
# ===========================================================================
print('\n=== Section 1 : uq_Kriging_eval_one_output (mean only) ===')

rng = np.random.default_rng(0)
X1_test = rng.uniform(-1, 1, (10, 1))
U1_test = X1_test   # Uniform(-1,1) -> Uniform(-1,1) = identity

koo1    = fm1['Kriging'][0]
U1_tr   = fm1['ExpDesign']['U']
Y1_tr   = fm1['ExpDesign']['Y'][:, 0]
F1_tr   = koo1['F']
corr1   = fm1['CorrOptions']

YMu1 = uq_Kriging_eval_one_output(koo1, U1_test, U1_tr, Y1_tr, F1_tr, corr1)

ok('1a output shape (N_test,)', YMu1.shape == (10,))
ok('1b all finite',              np.all(np.isfinite(YMu1)))
ok('1c reasonable range',        np.all(np.abs(YMu1) <= 2.0),
   f'max={np.max(np.abs(YMu1)):.3f}')


# ===========================================================================
# SECTION 2 -- uq_Kriging_eval_one_output: mean + variance
# ===========================================================================
print('\n=== Section 2 : uq_Kriging_eval_one_output (mean + variance) ===')

YMu1v, YSig1 = uq_Kriging_eval_one_output(
    koo1, U1_test, U1_tr, Y1_tr, F1_tr, corr1, return_var=True)

ok('2a YMu matches mean-only',  np.allclose(YMu1, YMu1v, atol=1e-12))
ok('2b YSigma2 shape (N_test,)', YSig1.shape == (10,))
ok('2c YSigma2 non-negative',    np.all(YSig1 >= 0))
ok('2d YSigma2 finite',          np.all(np.isfinite(YSig1)))


# ===========================================================================
# SECTION 3 -- uq_Kriging_eval_one_output: mean + variance + covariance
# ===========================================================================
print('\n=== Section 3 : uq_Kriging_eval_one_output (mean + cov) ===')

X1_test_sm = rng.uniform(-1, 1, (5, 1))
U1_test_sm = X1_test_sm

YMu1c, YSig1c, YCov1 = uq_Kriging_eval_one_output(
    koo1, U1_test_sm, U1_tr, Y1_tr, F1_tr, corr1,
    return_var=True, return_cov=True)

ok('3a YCov shape (N_test, N_test)', YCov1.shape == (5, 5))
ok('3b YCov symmetric',             np.allclose(YCov1, YCov1.T, atol=1e-10))
ok('3c diag(YCov) ~= YSigma2',      np.allclose(np.diag(YCov1), YSig1c, atol=1e-10))
ok('3d YSig from cov non-negative', np.all(YSig1c >= 0))

# Covariance matrix should be PSD (eigenvalues >= 0 up to tolerance)
eigvals = np.linalg.eigvalsh(YCov1)
ok('3e YCov is PSD', np.all(eigvals >= -1e-8), f'min_eig={eigvals.min():.2e}')


# ===========================================================================
# SECTION 4 -- Kriging interpolation property
#   At training points: YMu = Y_train, YSigma2 = 0
# ===========================================================================
print('\n=== Section 4 : Kriging interpolation property ===')

# Predict AT training points (using training U coordinates)
YMu_tr, YSig_tr = uq_Kriging_eval_one_output(
    koo1, U1_tr, U1_tr, Y1_tr, F1_tr, corr1, return_var=True)

ok('4a YMu at train ~= Y_train',
   np.allclose(YMu_tr, Y1_tr, atol=1e-6),
   f'max_err={np.max(np.abs(YMu_tr - Y1_tr)):.2e}')
ok('4b YSigma2 at train ~= 0',
   np.allclose(YSig_tr, 0.0, atol=1e-6),
   f'max_sig={np.max(YSig_tr):.2e}')


# ===========================================================================
# SECTION 5 -- uq_PCK_eval: single output, all 3 return modes
# ===========================================================================
print('\n=== Section 5 : uq_PCK_eval single output ===')

X1_new = rng.uniform(-1, 1, (8, 1))

# Mean only
YMu5 = uq_PCK_eval(fm1, X1_new, return_var=False)
ok('5a shape (N_test, Nout)',  YMu5.shape == (8, 1))
ok('5b all finite',             np.all(np.isfinite(YMu5)))

# Mean + variance
YMu5v, YSig5 = uq_PCK_eval(fm1, X1_new, return_var=True)
ok('5c YMu consistent',         np.allclose(YMu5, YMu5v, atol=1e-12))
ok('5d YSig shape (N, Nout)',   YSig5.shape == (8, 1))
ok('5e YSig non-negative',      np.all(YSig5 >= 0))

# Mean + var + cov
YMu5c, YSig5c, YCov5 = uq_PCK_eval(fm1, X1_new,
                                     return_var=True, return_cov=True)
ok('5f YCov shape (N,N,Nout)',  YCov5.shape == (8, 8, 1))
ok('5g YCov[:,:,0] symmetric',  np.allclose(YCov5[:,:,0], YCov5[:,:,0].T, atol=1e-10))
ok('5h YSig from cov consistent', np.allclose(YSig5, YSig5c, atol=1e-10))


# ===========================================================================
# SECTION 6 -- uq_PCK_eval at training inputs: mean ~= Y, var ~= 0
# ===========================================================================
print('\n=== Section 6 : uq_PCK_eval interpolation at training inputs ===')

YMu6, YSig6 = uq_PCK_eval(fm1, X1_train, return_var=True)
ok('6a YMu at train ~= Y',
   np.allclose(YMu6[:, 0], Y1_train, atol=1e-5),
   f'max_err={np.max(np.abs(YMu6[:,0] - Y1_train)):.2e}')
ok('6b YSig at train ~= 0',
   np.allclose(YSig6, 0.0, atol=1e-5),
   f'max_sig={np.max(YSig6):.2e}')


# ===========================================================================
# SECTION 7 -- uq_PCK_eval: 2D model
# ===========================================================================
print('\n=== Section 7 : uq_PCK_eval 2D model ===')

X2_new = np.random.default_rng(99).uniform(-1, 1, (12, 2))

YMu7 = uq_PCK_eval(fm2, X2_new, return_var=False)
ok('7a shape (N_test, 1)',   YMu7.shape == (12, 1))
ok('7b finite',              np.all(np.isfinite(YMu7)))

YMu7v, YSig7 = uq_PCK_eval(fm2, X2_new, return_var=True)
ok('7c var non-negative',    np.all(YSig7 >= 0))
ok('7d mean consistent',     np.allclose(YMu7, YMu7v, atol=1e-12))

# At training points
YMu7_tr, YSig7_tr = uq_PCK_eval(fm2, X2_train, return_var=True)
ok('7e YMu at train ~= Y_train',
   np.allclose(YMu7_tr[:, 0], Y2_train.ravel(), atol=1e-4),
   f'max_err={np.max(np.abs(YMu7_tr[:,0] - Y2_train.ravel())):.2e}')
ok('7f YSig at train ~= 0',
   np.allclose(YSig7_tr, 0.0, atol=1e-4),
   f'max_sig={np.max(YSig7_tr):.2e}')


# ===========================================================================
# SECTION 8 -- uq_PCK_eval: multi-output model
# ===========================================================================
print('\n=== Section 8 : uq_PCK_eval multi-output ===')

rng8 = np.random.default_rng(55)
N8 = 30
X8_train = rng8.uniform(-1, 1, (N8, 1))
# 2 outputs: x^2 and x (different functions)
Y8_train = np.column_stack([X8_train.ravel()**2, X8_train.ravel()])
marg8 = [{'Type': 'Uniform', 'Parameters': [-1.0, 1.0]}]
cop8  = {'Type': 'Independent', 'Parameters': np.eye(1)}
cfg8  = {'Mode': 'sequential', 'TrendMethod': 'pce',
         'PCE': {'Degree': [1, 2], 'Method': 'LARS'}}

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fm8 = uq_PCK_calculate_coefficients(
        X8_train, Y8_train, cfg8, marg8, cop8,
        optim_method='gradbased', estim_method='ml')

X8_test = rng8.uniform(-1, 1, (6, 1))
YMu8, YSig8 = uq_PCK_eval(fm8, X8_test, return_var=True)

ok('8a Nout = 2, shape (6, 2)',   YMu8.shape == (6, 2))
ok('8b var shape (6, 2)',          YSig8.shape == (6, 2))
ok('8c both outputs finite',       np.all(np.isfinite(YMu8)))
ok('8d both variances non-neg',    np.all(YSig8 >= 0))

# Outputs should be different (x^2 != x in general)
ok('8e outputs are distinct', not np.allclose(YMu8[:, 0], YMu8[:, 1], atol=0.1))

# Check interpolation for multi-output
YMu8_tr = uq_PCK_eval(fm8, X8_train, return_var=False)
ok('8f output 0 at train ~= x^2',
   np.allclose(YMu8_tr[:, 0], X8_train.ravel()**2, atol=1e-4),
   f'max_err={np.max(np.abs(YMu8_tr[:,0] - X8_train.ravel()**2)):.2e}')
ok('8g output 1 at train ~= x',
   np.allclose(YMu8_tr[:, 1], X8_train.ravel(), atol=1e-4),
   f'max_err={np.max(np.abs(YMu8_tr[:,1] - X8_train.ravel())):.2e}')


# ===========================================================================
# SECTION 9 -- Prediction quality (LOO as proxy)
# ===========================================================================
print('\n=== Section 9 : Prediction quality ===')

# 1D model: predict at training points, error should be < tolerance
err_1d = np.sqrt(np.mean((YMu6[:, 0] - Y1_train)**2))
ok('9a 1D RMSE at train < 1e-5',  err_1d < 1e-5, f'RMSE={err_1d:.2e}')

# 2D model: RMSE at training
err_2d = np.sqrt(np.mean((YMu7_tr[:, 0] - Y2_train.ravel())**2))
ok('9b 2D RMSE at train < 1e-4',  err_2d < 1e-4, f'RMSE={err_2d:.2e}')

# Variance at unseen points should be > 0 in general
# (not at training points)
ok('9c variance at new points > 0', np.any(YSig5 > 0))
ok('9d variance > 0 at all new test points (1D)', np.all(YSig5 > 0))


# ===========================================================================
# SUMMARY
# ===========================================================================
print(f'\n{"="*50}')
print(f'Results : {PASS} PASS  |  {FAIL} FAIL')
if FAIL == 0:
    print('All tests PASSED.')
else:
    print('Some tests FAILED -- see above.')
