"""
Coquille de compatibilite : `branche2` a ete renomme en `options.py`.

PHASE 2 du plan de nettoyage (docs/plan-nettoyage.md). Aucun code ici : le
module delegue a `options`.

Pourquoi une coquille plutot qu'une edition des appelants : les dix suites
unitaires de `tests/unit/` ont ete restaurees **sans modifier une ligne**
depuis la branche `fiabilite`, ce qui en fait un temoin fidele du contrat
d'origine -- ce sont elles qui ont revele les deux regressions de la branche
`flexion`. Les editer pour suivre un renommage leur oterait cette qualite.

Pourquoi `__getattr__` plutot que `from options import *` : la deleguation est
alors **tardive**. Avec un `import *`, les noms seraient lies une fois pour
toutes au chargement, et `tools/telemetry.py` -- qui instrumente en
remplacant les fonctions dans le module -- ne serait plus vu a travers
l'ancien nom. Les deux chemins d'import doivent donner le meme objet a tout
instant.

RETRAIT PREVU : fin de la phase 3, quand les scripts AC seront passes a la
nouvelle API. Une coquille sans date de retrait devient permanente.
"""

import warnings as _warnings

import options as _cible

_warnings.warn(
    "Le module 'branche2' est renomme 'options'. L'ancien nom sera retire a la "
    "fin de la phase 3 du plan de nettoyage.",
    DeprecationWarning, stacklevel=2)

__all__ = [_n for _n in dir(_cible) if not _n.startswith("_")]


def __getattr__(nom):
    """Deleguation tardive : resout dans `options` au moment de l'acces."""
    try:
        return getattr(_cible, nom)
    except AttributeError:
        raise AttributeError(
            "module 'branche2' (renomme 'options') n'a pas d'attribut %r" % nom) from None


def __dir__():
    return sorted(__all__)
