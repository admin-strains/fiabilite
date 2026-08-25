"""
Coquille de compatibilite : `branche3` a ete scinde en trois modules.

    kriging.py     krigeage universel -- matrices, beta, sigma^2, K-fold, theta
    pce_basis.py   base polynomiale du chaos -- multi-indices, matrice Psi
    fit.py         orchestration PCK / GEPCK (tendance PCE + krigeage du residu)

PHASE 3 du plan de nettoyage. Le graphe d'appels montrait une
stratification nette : `fit` appelle `kriging` et `pce_basis`, qui ne
s'appellent pas entre eux.

Deleguation TARDIVE, comme les autres coquilles, pour que l'instrumentation
de `tools/telemetry.py` reste visible a travers l'ancien nom.

Conserve pour que les dix suites de `tests/unit/` restent INTOUCHEES.

RETRAIT PREVU : fin de la phase 3, avec les autres coquilles.
"""

import warnings as _warnings

import fit as _fit
import kriging as _kriging
import pce_basis as _pce_basis

_CIBLES = (_kriging, _pce_basis, _fit)

_warnings.warn(
    "Le module 'branche3' est scinde en 'kriging', 'pce_basis' et 'fit'. "
    "L'ancien nom sera retire a la fin de la phase 3 du plan de nettoyage.",
    DeprecationWarning, stacklevel=2)

__all__ = sorted({n for c in _CIBLES for n in dir(c) if not n.startswith("__")})


def __getattr__(nom):
    """Deleguation tardive : cherche dans kriging, puis pce_basis, puis fit."""
    for cible in _CIBLES:
        try:
            return getattr(cible, nom)
        except AttributeError:
            continue
    raise AttributeError(
        "module 'branche3' (scinde en kriging/pce_basis/fit) n'a pas "
        "d'attribut %r" % nom)


def __dir__():
    return list(__all__)
