"""
Non-regression figee du metamodele (PCK / GEPCK).

Contrat : a DOE identique et options identiques, l'ajustement doit rendre
exactement les memes coefficients qu'au moment ou le golden a ete produit.
C'est le filet qui rend un refactoring sur : tout deplacement de code qui
change un chiffre est vu, meme s'il ne change pas la qualite du modele.

Les tolerances distinguent deux natures de grandeur :
  - structure du modele (nombre de polynomes, indices retenus par LARS) :
    egalite STRICTE, c'est un choix discret ;
  - grandeurs continues (theta, beta_pce, LOO, predictions) : rtol 1e-8,
    ce qui absorbe une difference de BLAS ou de version de scipy mais rien
    de plus.

Si un golden doit changer : voir la procedure dans tests/make_golden.py.
"""

import functools
import json
import os

import numpy as np
import pytest

import harness

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')
CASES = ['flexion', 'linear']
KINDS = ['PCK', 'GEPCK']

pytestmark = pytest.mark.golden


def _load(case):
    with open(os.path.join(GOLDEN_DIR, f'{case}.json'), encoding='utf-8') as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def _refit(case, kind):
    from reference.limit_states import CASES as LS_CASES
    import warnings
    ref = _load(case)
    ls = LS_CASES[case]()
    X = np.asarray(ref['doe'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fm = harness.fit(kind, X, ls, max_degree=ref['max_degree'])
    return ref, harness.signature(fm, np.asarray(ref['probe']), kind)


def test_doe_du_golden_est_toujours_celui_du_harness():
    """Garde-fou : si make_doe change, tous les goldens deviennent caducs."""
    for case in CASES:
        ref = _load(case)
        assert np.allclose(harness.make_doe(ref['n_doe'], 2), np.asarray(ref['doe']),
                           atol=0, rtol=0), \
            f'{case} : make_doe a change, les goldens ne sont plus comparables'


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('kind', KINDS)
def test_structure_du_modele(case, kind):
    ref, got = _refit(case, kind)
    exp = ref['models'][kind]
    assert got['NumberOfPoly'] == exp['NumberOfPoly'], \
        'LARS ne retient plus le meme nombre de polynomes'
    assert got['poly_indices'] == exp['poly_indices'], \
        'LARS ne retient plus les memes multi-indices'


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('kind', KINDS)
def test_coefficients_du_modele(case, kind):
    ref, got = _refit(case, kind)
    exp = ref['models'][kind]
    assert np.allclose(got['theta'], exp['theta'], rtol=1e-8, atol=0), \
        f"theta {got['theta']} != golden {exp['theta']}"
    assert np.allclose(got['beta_pce'], exp['beta_pce'], rtol=1e-8, atol=1e-14)
    assert got['sigmaSQ'] == pytest.approx(exp['sigmaSQ'], rel=1e-8)


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('kind', KINDS)
def test_erreur_loo(case, kind):
    ref, got = _refit(case, kind)
    assert got['LOO'] == pytest.approx(ref['models'][kind]['LOO'], rel=1e-8)


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('kind', KINDS)
def test_predictions_aux_points_sonde(case, kind):
    ref, got = _refit(case, kind)
    exp = ref['models'][kind]
    assert np.allclose(got['mu_probe'], exp['mu_probe'], rtol=1e-8, atol=1e-14)
    assert np.allclose(got['var_probe'], exp['var_probe'], rtol=1e-6, atol=1e-16)


@pytest.mark.parametrize('kind', KINDS)
def test_variance_predictive_positive(kind, fitted):
    _, _, sigma = harness.predictors(kind, fitted[kind])
    U = np.array([[0., 0.], [2., -3.], [-5., 5.], [0.1, 0.1]])
    s = sigma(U)
    assert np.all(np.isfinite(s))
    assert np.all(s >= 0.0)
