"""Configuration pytest du harness de non-regression fiabilite."""

import os
import sys

import numpy as np
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)

for p in (TESTS_DIR, os.path.join(ROOT, '_lib'), os.path.join(ROOT, '_model'), os.path.join(ROOT, '_cache'),
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
