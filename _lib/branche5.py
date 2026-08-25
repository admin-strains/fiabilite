"""
Coquille de compatibilite : `branche5` a ete scinde en trois modules.

    polynomials.py   polynomes orthogonaux, base PCE
    transform.py     transformations isoprobabilistes (Nataf, Rosenblatt)
    kernels.py       noyaux de correlation et leurs derivees

PHASE 3 du plan de nettoyage. `branche5.py` portait ces trois sujets sans
rapport entre eux ; l'analyse du graphe d'appels a montre **zero arete** les
reliant. Ce n'etait pas un module, c'etait trois fichiers concatenes.

Deleguation TARDIVE, comme les coquilles de la phase 2 : les noms sont
resolus au moment de l'acces, pour que l'instrumentation de
`tools/telemetry.py` reste visible a travers l'ancien nom.

Conserve pour que les dix suites de `tests/unit/` restent INTOUCHEES : elles
importent `branche5` directement, et leur valeur de temoin tient a ce qu'on
n'y ait jamais touche.

RETRAIT PREVU : fin de la phase 3, avec les autres coquilles.
"""

import warnings as _warnings

import kernels as _kernels
import polynomials as _polynomials
import transform as _transform

_CIBLES = (_polynomials, _transform, _kernels)

_warnings.warn(
    "Le module 'branche5' est scinde en 'polynomials', 'transform' et "
    "'kernels'. L'ancien nom sera retire a la fin de la phase 3 du plan de "
    "nettoyage.",
    DeprecationWarning, stacklevel=2)

__all__ = sorted({n for c in _CIBLES for n in dir(c) if not n.startswith("__")})


def __getattr__(nom):
    """Deleguation tardive : cherche dans polynomials, puis transform, puis kernels."""
    for cible in _CIBLES:
        try:
            return getattr(cible, nom)
        except AttributeError:
            continue
    raise AttributeError(
        "module 'branche5' (scinde en polynomials/transform/kernels) n'a pas "
        "d'attribut %r" % nom)


def __dir__():
    return list(__all__)
