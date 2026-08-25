"""
test_branche1.py -- Integration tests for branche1.py (fit_pck / predict_pck)

Covers:
  1. Default options (Mode=sequential, LARS)    -- 1D
  2. Mode=optimal                               -- 1D
  3. Custom PCE degree                          -- 1D
  4. Custom Kriging correlation (Gaussian)      -- 1D
  5. TrendMethod='user' (manual PolyIndices)    -- 1D
  6. IgnoreDependence flag                      -- 1D
  7. Multi-output                               -- 1D, Nout=2
  8. 2D input                                   -- 2D
  9. predict_pck wrapper (mean + var + cov)     -- 1D
  10. Interpolation at training points           -- 1D
  11. predict_pck vs direct uq_PCK_eval          -- 1D
  12. Error cases (invalid Mode, both PCE+PolyIndices)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import warnings

from branche1 import fit_pck, predict_pck
from branche4 import uq_PCK_eval

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
# Shared data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
N1  = 25
X1  = rng.uniform(-1, 1, (N1, 1))
Y1  = X1.ravel()**3

marg1 = [{'Type': 'Uniform', 'Parameters': [-1.0, 1.0]}]
cop1  = {'Type': 'Independent', 'Parameters': np.eye(1)}


def _fit(options, X=X1, Y=Y1, marg=marg1, cop=cop1):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fit_pck(X, Y, options, marg, cop)


# ===========================================================================
# SECTION 1 -- Default options (Mode=sequential, LARS, degree 1-2-3)
# ===========================================================================
print('\n=== Section 1 : Default options ===')

opts1 = {'Mode': 'sequential', 'PCE': {'Degree': [1, 2, 3], 'Method': 'LARS'}}
fm1 = _fit(opts1)

ok('1a fitted_model is dict',        isinstance(fm1, dict))
ok('1b Nout = 1',                    fm1['Nout'] == 1)
ok('1c Mred >= 1',                   fm1['Mred'] >= 1)
ok('1d Kriging entry exists',        isinstance(fm1['Kriging'], list) and len(fm1['Kriging']) == 1)
ok('1e LOO in [0, 1]',               0 <= fm1['Error'][0]['LOO'] <= 1,
   f'LOO={fm1["Error"][0]["LOO"]:.4f}')

YMu1 = predict_pck(fm1, X1)
ok('1f predict shape (N, 1)',        YMu1.shape == (N1, 1))
ok('1g predict finite',              np.all(np.isfinite(YMu1)))


# ===========================================================================
# SECTION 2 -- Mode=optimal
# ===========================================================================
print('\n=== Section 2 : Mode=optimal ===')

opts2 = {'Mode': 'optimal', 'PCE': {'Degree': [1, 2, 3], 'Method': 'LARS'}}
fm2 = _fit(opts2)

ok('2a Nout = 1',                    fm2['Nout'] == 1)
ok('2b LOO in [0, 1]',               0 <= fm2['Error'][0]['LOO'] <= 1,
   f'LOO={fm2["Error"][0]["LOO"]:.4f}')
YMu2 = predict_pck(fm2, rng.uniform(-1, 1, (5, 1)))
ok('2c predict finite',              np.all(np.isfinite(YMu2)))


# ===========================================================================
# SECTION 3 -- Custom PCE degree (degree 1-2 only)
# ===========================================================================
print('\n=== Section 3 : Custom PCE degree ===')

opts3 = {'Mode': 'sequential', 'PCE': {'Degree': [1, 2], 'Method': 'LARS'}}
fm3 = _fit(opts3)

ok('3a fitted', fm3['Nout'] == 1)
ok('3b LOO in [0, 1]',              0 <= fm3['Error'][0]['LOO'] <= 1)
ok('3c Kriging key present',        'Kriging' in fm3)


# ===========================================================================
# SECTION 4 -- Custom Kriging correlation (Gaussian)
# ===========================================================================
print('\n=== Section 4 : Custom Kriging correlation ===')

opts4 = {
    'Mode': 'sequential',
    'PCE':  {'Degree': [1, 2, 3], 'Method': 'LARS'},
    'Kriging': {
        'Corr': {'Family': 'gaussian', 'Type': 'separable', 'Isotropic': False},
    },
}
fm4 = _fit(opts4)

ok('4a fitted',           fm4['Nout'] == 1)
ok('4b Family stored',    fm4['CorrOptions']['Family'] == 'gaussian',
   f"Family={fm4['CorrOptions']['Family']}")
ok('4c LOO in [0, 1]',    0 <= fm4['Error'][0]['LOO'] <= 1)
YMu4 = predict_pck(fm4, rng.uniform(-1, 1, (5, 1)))
ok('4d predict finite',   np.all(np.isfinite(YMu4)))


# ===========================================================================
# SECTION 5 -- TrendMethod='user' (manual PolyIndices)
# ===========================================================================
print('\n=== Section 5 : TrendMethod=user ===')

# Polynomial basis: 1, x, x^2, x^3 for Uniform(-1,1) -> Legendre degree 0-3
PolyIdx = np.array([[0], [1], [2], [3]])   # (4, 1) multi-indices
opts5 = {
    'PolyIndices': PolyIdx,
    'PolyTypes':   ['Legendre'],
}
fm5 = _fit(opts5)

ok('5a TrendMethod=user',  fm5['pck_config']['TrendMethod'] == 'user')
ok('5b Nout = 1',          fm5['Nout'] == 1)
ok('5c LOO in [0, 1]',     0 <= fm5['Error'][0]['LOO'] <= 1)
YMu5 = predict_pck(fm5, rng.uniform(-1, 1, (5, 1)))
ok('5d predict finite',    np.all(np.isfinite(YMu5)))


# ===========================================================================
# SECTION 6 -- IgnoreDependence flag (independent copula substitution)
# ===========================================================================
print('\n=== Section 6 : IgnoreDependence flag ===')

# Use a 2D input with a non-independent copula to verify IgnoreDependence
rng6 = np.random.default_rng(10)
N6   = 30
X6   = rng6.uniform(-1, 1, (N6, 2))
Y6   = X6[:, 0]**2 + 0.5 * X6[:, 1]

marg6 = [{'Type': 'Uniform', 'Parameters': [-1.0, 1.0]}] * 2
cop6_dep = {
    'Type': 'Gaussian',
    'Parameters': np.array([[1.0, 0.8], [0.8, 1.0]]),
}

opts6_id = {
    'Mode': 'sequential',
    'PCE':  {'Degree': [1, 2], 'Method': 'LARS'},
    'IgnoreDependence': True,
}
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fm6 = fit_pck(X6, Y6, opts6_id, marg6, cop6_dep)

ok('6a fitted',           fm6['Nout'] == 1)
ok('6b LOO in [0, 1]',    0 <= fm6['Error'][0]['LOO'] <= 1)
YMu6 = predict_pck(fm6, rng6.uniform(-1, 1, (4, 2)))
ok('6c predict finite',   np.all(np.isfinite(YMu6)))


# ===========================================================================
# SECTION 7 -- Multi-output
# ===========================================================================
print('\n=== Section 7 : Multi-output ===')

rng7 = np.random.default_rng(77)
N7   = 30
X7   = rng7.uniform(-1, 1, (N7, 1))
Y7   = np.column_stack([X7.ravel()**2, X7.ravel()])   # Nout=2

opts7 = {'Mode': 'sequential', 'PCE': {'Degree': [1, 2], 'Method': 'LARS'}}
fm7   = _fit(opts7, X=X7, Y=Y7)

ok('7a Nout=2',            fm7['Nout'] == 2)
ok('7b Kriging list len',  len(fm7['Kriging']) == 2)
ok('7c LOO[0] in [0,1]',   0 <= fm7['Error'][0]['LOO'] <= 1)
ok('7d LOO[1] in [0,1]',   0 <= fm7['Error'][1]['LOO'] <= 1)

YMu7, YSig7 = predict_pck(fm7, rng7.uniform(-1, 1, (6, 1)), return_var=True)
ok('7e predict shape (6,2)',   YMu7.shape == (6, 2))
ok('7f var shape (6,2)',       YSig7.shape == (6, 2))
ok('7g both finite',           np.all(np.isfinite(YMu7)))
ok('7h outputs distinct',      not np.allclose(YMu7[:, 0], YMu7[:, 1], atol=0.1))


# ===========================================================================
# SECTION 8 -- 2D input
# ===========================================================================
print('\n=== Section 8 : 2D input ===')

rng8 = np.random.default_rng(88)
N8   = 40
X8   = rng8.uniform(-1, 1, (N8, 2))
Y8   = X8[:, 0]**2 + 0.5 * X8[:, 1] + 0.02 * rng8.standard_normal(N8)

marg8 = [{'Type': 'Uniform', 'Parameters': [-1.0, 1.0]}] * 2
cop8  = {'Type': 'Independent', 'Parameters': np.eye(2)}

opts8 = {'Mode': 'sequential', 'PCE': {'Degree': [1, 2], 'Method': 'LARS'}}
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fm8 = fit_pck(X8, Y8, opts8, marg8, cop8)

ok('8a fitted Mred=2',    fm8['Mred'] == 2)
ok('8b LOO in [0,1]',     0 <= fm8['Error'][0]['LOO'] <= 1)
YMu8 = predict_pck(fm8, rng8.uniform(-1, 1, (10, 2)))
ok('8c predict shape (10,1)', YMu8.shape == (10, 1))
ok('8d predict finite',       np.all(np.isfinite(YMu8)))


# ===========================================================================
# SECTION 9 -- predict_pck: all 3 return modes
# ===========================================================================
print('\n=== Section 9 : predict_pck return modes ===')

X9_test = rng.uniform(-1, 1, (7, 1))

YMu9  = predict_pck(fm1, X9_test)
ok('9a mean shape (7,1)',   YMu9.shape == (7, 1))

YMu9v, YSig9 = predict_pck(fm1, X9_test, return_var=True)
ok('9b var shape (7,1)',    YSig9.shape == (7, 1))
ok('9c mean consistent',   np.allclose(YMu9, YMu9v, atol=1e-12))
ok('9d var non-neg',        np.all(YSig9 >= 0))

YMu9c, YSig9c, YCov9 = predict_pck(fm1, X9_test, return_var=True, return_cov=True)
ok('9e cov shape (7,7,1)', YCov9.shape == (7, 7, 1))
ok('9f cov symmetric',      np.allclose(YCov9[:,:,0], YCov9[:,:,0].T, atol=1e-10))
ok('9g var consistent',    np.allclose(YSig9, YSig9c, atol=1e-10))


# ===========================================================================
# SECTION 10 -- Interpolation property at training points
# ===========================================================================
print('\n=== Section 10 : Interpolation at training points ===')

YMu10, YSig10 = predict_pck(fm1, X1, return_var=True)
ok('10a YMu at train ~= Y',
   np.allclose(YMu10[:, 0], Y1, atol=1e-5),
   f'max_err={np.max(np.abs(YMu10[:,0] - Y1)):.2e}')
ok('10b YSig at train ~= 0',
   np.allclose(YSig10, 0.0, atol=1e-5),
   f'max_sig={np.max(YSig10):.2e}')


# ===========================================================================
# SECTION 11 -- predict_pck vs direct uq_PCK_eval
# ===========================================================================
print('\n=== Section 11 : predict_pck == uq_PCK_eval ===')

X11 = rng.uniform(-1, 1, (8, 1))

a_mu, a_var = predict_pck(fm1, X11, return_var=True)
b_mu, b_var = uq_PCK_eval(fm1, X11, return_var=True)

ok('11a YMu identical',   np.allclose(a_mu, b_mu, atol=1e-15))
ok('11b YSig identical',  np.allclose(a_var, b_var, atol=1e-15))


# ===========================================================================
# SECTION 12 -- Error cases
# ===========================================================================
print('\n=== Section 12 : Error cases ===')

# 12a: invalid Mode
try:
    _fit({'Mode': 'invalid'})
    ok('12a invalid Mode raises', False)
except (ValueError, Exception):
    ok('12a invalid Mode raises', True)

# 12b: both PCE and PolyIndices
try:
    _fit({'PCE': {'Degree': [1,2], 'Method': 'LARS'},
          'PolyIndices': np.array([[0],[1]]),
          'PolyTypes': ['Legendre']})
    ok('12b PCE+PolyIndices raises', False)
except (ValueError, Exception):
    ok('12b PCE+PolyIndices raises', True)

# 12c: PolyIndices without PolyTypes
try:
    _fit({'PolyIndices': np.array([[0],[1]])})
    ok('12c PolyIndices without PolyTypes raises', False)
except (ValueError, Exception):
    ok('12c PolyIndices without PolyTypes raises', True)

# 12d: Mode=optimal with CombCrit override
opts12d = {'Mode': 'optimal',
           'PCE': {'Degree': [1, 2], 'Method': 'LARS'},
           'CombCrit': 'rel_loo'}
fm12d = _fit(opts12d)
ok('12d optimal + CombCrit override works', fm12d['Nout'] == 1)


# ===========================================================================
# SUMMARY
# ===========================================================================
print(f'\n{"="*50}')
print(f'Results : {PASS} PASS  |  {FAIL} FAIL')
if FAIL == 0:
    print('All tests PASSED.')
else:
    print('Some tests FAILED -- see above.')
