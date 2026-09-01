"""Configuration pytest du harness de non-regression fiabilite."""

import os

# ------------------------------------------------------------------------- #
# UN SEUL THREAD BLAS, ET CES LIGNES DOIVENT RESTER LES PREMIERES            #
# ------------------------------------------------------------------------- #
# Additionner les memes nombres dans un ordre different -- ce que font deux
# threads au lieu d'un -- change le dernier bit. La vraisemblance du krigeage
# etant plate pres de son optimum, L-BFGS-B part de ce dernier bit et
# s'arrete AILLEURS sur la crete. Mesure sur ce poste, meme numpy, meme DOE,
# `flexion/PCK` :
#
#     7 threads (defaut du poste)   theta = [0.010000, 6.548512]
#     1 thread                      theta = [0.341149, 6.244649]
#
# Un facteur 34 sur la premiere composante, pour un modele de qualite
# EQUIVALENTE (LOO du meme ordre). Sans bridage, les goldens ne figent pas le
# code : ils figent le nombre de coeurs de la machine qui les a produits. Les
# cinq jobs d'integration continue du 31/08/2026 le montraient -- cinq
# comptes de coeurs, cinq theta, y compris sur Windows/Python 3.10.
#
# POURQUOI ICI, AVANT `import numpy`, ET PAS PAR `threadpoolctl`.
# `threadpool_limits(1)` rend bien le meme theta que la variable
# d'environnement -- verifie le 01/09/2026 -- mais il ne bride que les
# bibliotheques DEJA CHARGEES. Pose dans `pytest_configure`, il manquait
# l'OpenBLAS que scipy charge ensuite : la suite restait verte et le bridage
# ne servait a rien. Un echec SILENCIEUX, le pire genre.
#
# OpenBLAS lit ces variables au CHARGEMENT. Elles doivent donc etre posees
# avant le premier import qui le charge -- ici, avant `import numpy`.
#
# CE QUE CELA NE COUTE PAS : les matrices de ces tests sont minuscules (DOE
# de 24 points). La PRODUCTION n'est pas concernee : ce bridage vit dans
# `tests/` et ne franchit pas la frontiere du processus pytest.
#
# ET IL EST VERIFIE, parce qu'un bridage qu'on croit actif est pire que pas
# de bridage : voir `test_05_hygiene_depot.test_le_BLAS_tourne_sur_un_thread`.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys                                                       # noqa: E402

import numpy as np                                               # noqa: E402
import pytest                                                    # noqa: E402

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)

for p in (TESTS_DIR, os.path.join(ROOT, '_lib'), os.path.join(ROOT, '_model'), os.path.join(ROOT, '_cache'), os.path.join(ROOT, '_reliability'),
          os.path.join(ROOT, 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: test de plus de ~5 s')
    config.addinivalue_line('markers', 'golden: comparaison a une valeur figee')
    config.addinivalue_line('markers', 'oracle: comparaison a une verite analytique')
    config.addinivalue_line('markers', 'defect: defaut connu, non corrige')


@pytest.fixture(scope='session')
def flexion_ls():
    from reference.limit_states import FlexionLS
    return FlexionLS()


@pytest.fixture(scope='session')
def linear_ls():
    from reference.limit_states import LinearLS
    return LinearLS()


@pytest.fixture(scope='session')
def doe24():
    import harness
    return harness.make_doe(24, 2)


@pytest.fixture(scope='session')
def fitted(doe24, flexion_ls):
    """Metamodeles ajustes une seule fois pour toute la session (fit ~1 s)."""
    import warnings
    import harness
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return {k: harness.fit(k, doe24, flexion_ls) for k in ('PCK', 'GEPCK')}
