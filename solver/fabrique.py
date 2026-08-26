r"""
Choisit l'implementation du solveur, et ne charge QUE celle-la.

C'est ce module que les scripts d'etude importent -- jamais
`digital_structure` directement. La raison tient en une ligne : importer
`digital_structure`, c'est importer Digital Structure, donc exiger une
licence, un GPU et un interpreteur 3.10. L'import est donc fait a l'interieur
de la fabrique, une fois le choix connu.

    from fabrique import solveur
    s = solveur("digital_structure", chemin_ds=..., params_names=..., ...)
    s = solveur("analytique", params_names=("fc", "fy"), section=...)

CE QUE CELA REND POSSIBLE
--------------------------
Mesure du 26/08/2026 sur `AC3_pure_flexion` : une fois les appels au solveur
delegues, les seuls noms que le script empruntait encore a Digital Structure
etaient `INITCATALOG` -- passe dans l'implementation -- et `sys`, qui n'etait
jamais importe et ne fonctionnait que parce que le `import *` de Digital
Structure le laissait fuiter dans les globales (defaut 7 du plan de
nettoyage).

Autrement dit : les 2 700 lignes de plan d'experiences, metamodele,
enrichissement, FORM et tirage d'importance ne dependent plus du solveur. La
meme chaine tourne sur l'etat limite analytique, en secondes, sans licence.
"""

#: nom -> (module, classe). Le module n'est importe qu'au moment du choix.
IMPLEMENTATIONS = {
    "digital_structure": ("digital_structure", "SolveurDS"),
    "analytique": ("analytique", "SolveurAnalytique"),
}


def solveur(nom, **kwargs):
    """Construit le solveur demande. Une seule implementation est chargee."""
    if nom not in IMPLEMENTATIONS:
        raise ValueError(
            "solveur=%r inconnu (attendu : %s)"
            % (nom, ", ".join(sorted(IMPLEMENTATIONS))))
    module, classe = IMPLEMENTATIONS[nom]
    import importlib
    return getattr(importlib.import_module(module), classe)(**kwargs)


def disponible(nom):
    """L'implementation peut-elle etre chargee ici ? Sans lever.

    `digital_structure` demande Digital Structure : sur un poste sans
    licence, la reponse est non, et c'est une information, pas une erreur.
    """
    if nom not in IMPLEMENTATIONS:
        return False
    import importlib
    try:
        importlib.import_module(IMPLEMENTATIONS[nom][0])
        return True
    except Exception:
        return False
