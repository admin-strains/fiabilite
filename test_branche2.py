"""
test_branche2.py — Tests for branche2.py (B2 : uq_PCK_initialize)
Covers every branch in uq_PCK_initialize and uq_process_option.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import copy
import numpy as np
import traceback

from branche2 import uq_PCK_initialize, uq_process_option, make_model

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

def raises(label, fn, exc_type=Exception, substr=None):
    global PASS, FAIL
    try:
        fn()
        print(f'  FAIL  {label}  [no exception raised]')
        FAIL += 1
    except exc_type as e:
        if substr and substr.lower() not in str(e).lower():
            print(f'  FAIL  {label}  [wrong message: {e}]')
            FAIL += 1
        else:
            print(f'  PASS  {label}')
            PASS += 1
    except Exception as e:
        print(f'  FAIL  {label}  [wrong exception type: {type(e).__name__}: {e}]')
        FAIL += 1


# ===========================================================================
# Minimal Input object used in tests
# ===========================================================================
def make_input(copula_type='independent', n=2):
    return {
        'Marginals': [{'Type': 'Uniform', 'Parameters': [0, 1]}] * n,
        'Copula':    {'Type': copula_type, 'Parameters': np.eye(n)},
        'nonConst':  list(range(n)),
    }


# ===========================================================================
# SECTION 1 — uq_process_option  (standalone tests)
# ===========================================================================
print('\n=== Section 1 : uq_process_option ===')

# 1a. Option not in AllOptions → Missing, Value=Default
opt, rem = uq_process_option({'A': 1}, 'B', 'hello', 'char')
ok('1a missing flag', opt['Missing'] is True)
ok('1a default value', opt['Value'] == 'hello')
ok('1a AllOptions unchanged', 'A' in rem)

# 1b. Option found, correct type
opt, rem = uq_process_option({'Mode': 'optimal'}, 'Mode', 'sequential', 'char')
ok('1b value', opt['Value'] == 'optimal')
ok('1b not missing', not opt['Missing'])
ok('1b key removed', 'Mode' not in rem)

# 1c. Case-insensitive key lookup
opt, rem = uq_process_option({'mode': 'optimal'}, 'Mode', 'sequential', 'char')
ok('1c case-insensitive', opt['Value'] == 'optimal')

# 1d. Wrong type → Invalid, Value=Default
opt, rem = uq_process_option({'Mode': 42}, 'Mode', 'sequential', 'char')
ok('1d invalid flag', opt['Invalid'] is True)
ok('1d default on invalid', opt['Value'] == 'sequential')
ok('1d key removed on invalid', 'Mode' not in rem)

# 1e. Struct merging — user supplies subset of fields
default_pce = {'MetaType': 'PCE', 'Degree': [1,2,3], 'Method': 'LARS'}
user_pce    = {'Method': 'OMP'}
opt, rem = uq_process_option({'PCE': user_pce}, 'PCE', copy.copy(default_pce), 'struct')
ok('1e method overridden',  opt['Value']['Method'] == 'OMP')
ok('1e degree kept',        opt['Value']['Degree'] == [1,2,3])
ok('1e metatype kept',      opt['Value']['MetaType'] == 'PCE')

# 1f. Multiple keys matching same option → warning + first kept
opt, rem = uq_process_option({'Mode': 'optimal', 'mode': 'sequential'}, 'Mode', 'x', 'char')
ok('1f first match used', opt['Value'] in ('optimal', 'sequential'))
ok('1f duplicate removed', 'mode' not in rem or 'Mode' not in rem)

# 1g. EmptyAsMissing
opt, rem = uq_process_option({'X': None}, 'X', 'def', 'char', EmptyAsMissing=True)
ok('1g EmptyAsMissing triggers Missing', opt['Missing'] is True)

# 1h. logical/double list for AllowedClasses
opt, rem = uq_process_option({'IgnDep': True}, 'IgnDep', False, ['logical', 'double'])
ok('1h bool accepted', opt['Value'] is True)
opt2, _ = uq_process_option({'IgnDep': 1.0}, 'IgnDep', False, ['logical', 'double'])
ok('1h float accepted', opt2['Value'] == 1.0)

# 1i. uq_input type (dict accepted)
inp = make_input()
opt, rem = uq_process_option({'Input': inp}, 'Input', None, 'uq_input')
ok('1i uq_input accepted', opt['Value'] is inp)


# ===========================================================================
# SECTION 2 — uq_PCK_initialize : defaults
# ===========================================================================
print('\n=== Section 2 : defaults (no PCE, no PolyIndices, mode not set) ===')

inp = make_input()
m   = make_model({}, M=2, global_input=inp)
uq_PCK_initialize(m, global_input=inp)

ok('2a Mode=sequential', m['Internal']['Mode'].lower() == 'sequential')
ok('2b TrendMethod=pce', m['Internal']['TrendMethod'] == 'pce')
ok('2c PCE.Method=LARS', m['Internal']['PCE']['Method'] == 'LARS')
ok('2d PCE.Degree=[1,2,3]', m['Internal']['PCE']['Degree'] == [1,2,3])
ok('2e PCE.MetaType=PCE', m['Internal']['PCE']['MetaType'] == 'PCE')
ok('2f IgnoreDependence=False', m['Internal']['IgnoreDependence'] is False)
ok('2g Input stored', m['Internal']['Input'] is inp)
ok('2h no CombCrit (sequential)', 'CombCrit' not in m['Internal'])
ok('2i Kriging is empty dict', m['Internal']['Kriging'] == {})


# ===========================================================================
# SECTION 3 — uq_PCK_initialize : Mode='optimal'
# ===========================================================================
print('\n=== Section 3 : mode=optimal ===')

inp = make_input()
m   = make_model({'Mode': 'optimal'}, M=2, global_input=inp)
uq_PCK_initialize(m, global_input=inp)

ok('3a Mode=optimal', m['Internal']['Mode'].lower() == 'optimal')
ok('3b CombCrit set', 'CombCrit' in m['Internal'])
ok('3c CombCrit=rel_loo', m['Internal']['CombCrit'] == 'rel_loo')


# ===========================================================================
# SECTION 4 — uq_PCK_initialize : custom CombCrit
# ===========================================================================
print('\n=== Section 4 : mode=optimal + custom CombCrit ===')

inp = make_input()
m   = make_model({'Mode': 'optimal', 'CombCrit': 'fh'}, M=2, global_input=inp)
uq_PCK_initialize(m, global_input=inp)

ok('4a CombCrit=fh', m['Internal']['CombCrit'] == 'fh')


# ===========================================================================
# SECTION 5 — uq_PCK_initialize : PCE options override
# ===========================================================================
print('\n=== Section 5 : custom PCE options ===')

inp = make_input()
m   = make_model({'PCE': {'Method': 'OMP', 'Degree': [1,2]}}, M=2, global_input=inp)
uq_PCK_initialize(m, global_input=inp)

ok('5a Method=OMP', m['Internal']['PCE']['Method'] == 'OMP')
ok('5b Degree overridden', m['Internal']['PCE']['Degree'] == [1,2])
ok('5c MetaType kept', m['Internal']['PCE']['MetaType'] == 'PCE')
ok('5d TrendMethod=pce', m['Internal']['TrendMethod'] == 'pce')


# ===========================================================================
# SECTION 6 — uq_PCK_initialize : TrendMethod='user'
# ===========================================================================
print('\n=== Section 6 : user-defined polynomial trend ===')

inp     = make_input()
indices = np.array([[0,0],[1,0],[0,1],[2,0]])
types   = ['Legendre', 'Legendre']
m = make_model(
    {'PolyIndices': indices, 'PolyTypes': types},
    M=2, global_input=inp)
uq_PCK_initialize(m, global_input=inp)

ok('6a TrendMethod=user', m['Internal']['TrendMethod'] == 'user')
ok('6b PolyIndices stored', np.array_equal(m['Internal']['PolyIndices'], indices))
ok('6c PolyTypes stored', m['Internal']['PolyTypes'] == types)


# ===========================================================================
# SECTION 7 — uq_PCK_initialize : error cases
# ===========================================================================
print('\n=== Section 7 : error cases ===')

inp = make_input()

# 7a. bad Mode
raises('7a bad Mode',
       lambda: uq_PCK_initialize(
           make_model({'Mode': 'unknown'}, M=2, global_input=inp), inp),
       ValueError, 'something went wrong')

# 7b. PCE + PolyIndices together
raises('7b PCE+PolyIndices',
       lambda: uq_PCK_initialize(
           make_model({'PCE': {'Method': 'LARS'},
                       'PolyIndices': np.zeros((3,2))}, M=2, global_input=inp), inp),
       ValueError, 'cannot be given at the same time')

# 7c. PolyIndices without PolyTypes
raises('7c no PolyTypes',
       lambda: uq_PCK_initialize(
           make_model({'PolyIndices': np.zeros((3,2))}, M=2, global_input=inp), inp),
       ValueError, 'PolyTypes are missing')

# 7d. Unsupported PCE method (OLS)
raises('7d OLS not supported',
       lambda: uq_PCK_initialize(
           make_model({'PCE': {'Method': 'OLS'}}, M=2, global_input=inp), inp),
       ValueError, 'LARS or OMP')

# 7e. IgnoreDependence wrong type
raises('7e IgnoreDependence wrong type',
       lambda: uq_PCK_initialize(
           make_model({'IgnoreDependence': 'yes'}, M=2, global_input=inp), inp),
       ValueError, 'must be a logical')


# ===========================================================================
# SECTION 8 — uq_PCK_initialize : IgnoreDependence=True
# ===========================================================================
print('\n=== Section 8 : IgnoreDependence ===')

inp_dep = make_input(copula_type='gaussian')
m = make_model({'IgnoreDependence': True}, M=2, global_input=inp_dep)
uq_PCK_initialize(m, global_input=inp_dep)

ok('8a IgnoreDependence=True stored', m['Internal']['IgnoreDependence'] is True)
ok('8b copula replaced', m['Internal']['Input']['Copula']['Type'].lower() == 'independent')
ok('8c original untouched', inp_dep['Copula']['Type'].lower() == 'gaussian')

# already independent — should NOT replace
inp_ind = make_input(copula_type='independent')
m2 = make_model({'IgnoreDependence': True}, M=2, global_input=inp_ind)
uq_PCK_initialize(m2, global_input=inp_ind)
ok('8d already independent — no copy', m2['Internal']['Input'] is inp_ind)


# ===========================================================================
# SECTION 9 — Kriging.Optim.Bounds adjustment for constant inputs
# ===========================================================================
print('\n=== Section 9 : Kriging.Optim.Bounds with constants ===')

# 2D input, only dim 0 is non-constant  (nonConst = [0])
inp_c = {
    'Marginals': [{'Type': 'Uniform'}, {'Type': 'Constant'}],
    'Copula':    {'Type': 'independent', 'Parameters': np.eye(2)},
    'nonConst':  [0],       # only first dimension varies
}
bounds_full = np.array([[0.01, 0.01], [100.0, 100.0]])  # shape (2,2)
m = make_model(
    {'Kriging': {'Optim': {'Bounds': bounds_full.copy()}}},
    M=2, global_input=inp_c)
uq_PCK_initialize(m, global_input=inp_c)

bounds_adj = m['Internal']['Kriging']['Optim']['Bounds']
ok('9a bounds shape reduced', bounds_adj.shape == (2, 1))
ok('9b correct column kept', np.allclose(bounds_adj[:, 0], [0.01, 100.0]))


# ===========================================================================
# SUMMARY
# ===========================================================================
print(f'\n{"="*50}')
print(f'Results : {PASS} PASS  |  {FAIL} FAIL')
print('="*50')
if FAIL == 0:
    print('All tests PASSED.')
else:
    print('Some tests FAILED — see above.')
