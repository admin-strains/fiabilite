r"""Le contrat entre les scripts d'etude et les implementations de solveur.

POURQUOI CE FICHIER EXISTE -- defaut du 26/08/2026
---------------------------------------------------
Le commit `1e7b0ac` a ajoute un parametre `max_size` : les deux scripts AC se
sont mis a le passer a la fabrique, `options_maillage` a su le lire, et
`SolveurDS.evaluer` a commence a faire `self.max_size`. Mais
`SolveurDS.__init__` ne l'a jamais ni accepte ni affecte.

Resultat : TOUTE evaluation Digital Structure levait

    AttributeError: 'SolveurDS' object has no attribute 'max_size'

et la construction depuis un AC aurait leve un `TypeError` avant meme cela.
Les 376 tests etaient verts. Deux raisons, et les deux comptent :

1. `SolveurAnalytique.__init__` se termine par `**ignores` -- c'est
   deliberement permissif, pour que l'appelant n'ait pas a savoir a qui il
   parle. Le kwarg inconnu y est donc absorbe en silence. Or c'est le
   solveur analytique que la chaine de test utilise.
2. `digital_structure` ne s'importe pas sans licence, sans GPU et hors 3.10.
   Aucun test ne pouvait le construire.

Ces tests ne construisent rien et n'importent rien : ils LISENT le source avec
`ast`. Ils tournent donc partout, y compris avec `core.txt` seul, et ils
auraient attrape le defaut a l'ecriture.
"""

import ast
import os

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)

#: (etiquette, chemin) des scripts d'etude qui construisent un solveur
SCRIPTS_AC = [
    ("pure_flexion", os.path.join(_REPO, "pure_flexion", "AC3_pure_flexion.py")),
    ("moulin_blanc", os.path.join(_REPO, "Moulinblanc", "AC3_moulinblanc.py")),
]

#: (etiquette, chemin, classe) des implementations du contrat
IMPLEMENTATIONS = [
    ("digital_structure", os.path.join(_REPO, "solver", "digital_structure.py"), "SolveurDS"),
    ("analytique", os.path.join(_REPO, "solver", "analytique.py"), "SolveurAnalytique"),
]

#: la fabrique est appelee sous ce nom dans les AC
NOMS_FABRIQUE = {"_fabriquer_solveur", "solveur"}


def _arbre(chemin):
    with open(chemin, "r", encoding="utf-8", errors="replace") as fh:
        return ast.parse(fh.read(), filename=chemin)


def _classe(arbre, nom):
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.ClassDef) and noeud.name == nom:
            return noeud
    raise AssertionError("classe %r introuvable" % nom)


def _init(classe):
    for noeud in classe.body:
        if isinstance(noeud, ast.FunctionDef) and noeud.name == "__init__":
            return noeud
    raise AssertionError("__init__ introuvable dans %r" % classe.name)


def _parametres_acceptes(fonction):
    """Noms acceptes en mot-clef, et si un `**kwargs` absorbe le reste."""
    a = fonction.args
    noms = {p.arg for p in a.args} | {p.arg for p in a.kwonlyargs}
    noms.discard("self")
    return noms, a.kwarg is not None


def _kwargs_passes_a_la_fabrique(arbre):
    """Les mots-clefs que le script passe a la fabrique de solveur."""
    trouves = set()
    for noeud in ast.walk(arbre):
        if not isinstance(noeud, ast.Call):
            continue
        f = noeud.func
        nom = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
        if nom not in NOMS_FABRIQUE:
            continue
        for mc in noeud.keywords:
            if mc.arg is not None:            # ignore les **dict
                trouves.add(mc.arg)
    return trouves


# --------------------------------------------------------------------- #
@pytest.mark.parametrize("etiquette,chemin", SCRIPTS_AC)
def test_le_script_passe_bien_des_mots_clefs(etiquette, chemin):
    """Garde-fou du garde-fou : si l'extraction ne trouve rien, les deux
    tests suivants passeraient a vide et ne prouveraient plus rien."""
    passes = _kwargs_passes_a_la_fabrique(_arbre(chemin))
    assert passes, (
        "aucun appel a la fabrique detecte dans %s : soit le script a change "
        "de forme, soit NOMS_FABRIQUE est perime -- dans les deux cas ce "
        "fichier de test ne protege plus rien." % etiquette)
    assert "chemin_ds" in passes, (
        "%s : 'chemin_ds' attendu parmi %s" % (etiquette, sorted(passes)))


@pytest.mark.parametrize("impl,chemin_impl,classe", IMPLEMENTATIONS)
@pytest.mark.parametrize("etiquette,chemin_ac", SCRIPTS_AC)
def test_chaque_implementation_accepte_ce_que_le_script_passe(
        etiquette, chemin_ac, impl, chemin_impl, classe):
    """Le defaut du 26/08/2026, pris a la source.

    Une implementation qui refuse un mot-clef passe par un AC leve un
    `TypeError` a la construction -- apres, sur le Moulin Blanc, plusieurs
    minutes de construction CAD.
    """
    passes = _kwargs_passes_a_la_fabrique(_arbre(chemin_ac))
    acceptes, absorbe_tout = _parametres_acceptes(
        _init(_classe(_arbre(chemin_impl), classe)))
    if absorbe_tout:
        pytest.skip("%s.__init__ a un **kwargs : il accepte tout, et c'est "
                    "justement ce qui a masque le defaut du 26/08" % classe)
    manquants = sorted(passes - acceptes)
    assert not manquants, (
        "%s.__init__ n'accepte pas %s, que %s lui passe.\n"
        "Construction impossible : TypeError.\n"
        "  passes par l'AC : %s\n"
        "  acceptes        : %s"
        % (classe, manquants, etiquette, sorted(passes), sorted(acceptes)))


@pytest.mark.parametrize("impl,chemin_impl,classe", IMPLEMENTATIONS)
def test_tout_attribut_lu_est_affecte_quelque_part(impl, chemin_impl, classe):
    """Aucune methode ne lit un `self.x` que la classe n'affecte jamais.

    C'est la forme exacte sous laquelle le defaut s'est manifeste :
    `evaluer` lisait `self.max_size`, que rien n'affectait. Le test est
    volontairement permissif -- une affectation N'IMPORTE OU dans la classe
    suffit -- pour ne signaler que le cas indefendable.
    """
    cls = _classe(_arbre(chemin_impl), classe)

    connus = set()
    for noeud in ast.walk(cls):
        # self.x = ...  /  self.x += ...  /  for self.x in ...
        if isinstance(noeud, ast.Attribute) and isinstance(noeud.ctx, (ast.Store, ast.Del)):
            if isinstance(noeud.value, ast.Name) and noeud.value.id == "self":
                connus.add(noeud.attr)
    # methodes, proprietes et attributs de classe
    for noeud in cls.body:
        if isinstance(noeud, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            connus.add(noeud.name)
        elif isinstance(noeud, ast.AnnAssign) and isinstance(noeud.target, ast.Name):
            connus.add(noeud.target.id)
        elif isinstance(noeud, ast.Assign):
            for cible in noeud.targets:
                if isinstance(cible, ast.Name):
                    connus.add(cible.id)

    lus = {}
    for noeud in ast.walk(cls):
        if isinstance(noeud, ast.Attribute) and isinstance(noeud.ctx, ast.Load):
            if isinstance(noeud.value, ast.Name) and noeud.value.id == "self":
                lus.setdefault(noeud.attr, noeud.lineno)

    orphelins = sorted((nom, lus[nom]) for nom in lus if nom not in connus)
    assert not orphelins, (
        "%s lit des attributs que la classe n'affecte jamais -- "
        "AttributeError garantie a l'execution :\n%s"
        % (classe, "\n".join("  self.%s  (ligne %d)" % (n, l) for n, l in orphelins)))
