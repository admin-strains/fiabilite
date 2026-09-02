r"""Le `pyproject.toml` et les `requirements/*.txt` disent-ils la meme chose ?

POURQUOI CE FICHIER
--------------------
Les dependances sont declarees DEUX FOIS : dans `pyproject.toml` -- pour
`pip install -e ".[etudes]"` -- et dans `requirements/*.txt`, que le message
d'erreur du lanceur designe encore :

    OpenTURNS est introuvable : No module named 'openturns'
      python -m pip install -r requirements/studies.txt

Les deux listes coincident aujourd'hui, verifie le 02/09/2026 : sept paquets,
memes noms, memes planchers. Rien ne le garantit. Une duplication sans temoin
est une divergence en attente -- et celle-ci enverrait un nouvel arrivant
installer une chaine incomplete, ce qui echoue loin de la cause.

Les deux existent pour de bonnes raisons : le `.toml` est ce que `pip`
comprend, les `.txt` portent la separation NOYAU / ETUDES qui structure tout
le depot et que `requirements/constraints-reference.txt` epingle au paquet
pres. Ce fichier ne choisit pas entre eux : il les oblige a s'accorder.
"""

import io
import os
import re
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)

if sys.version_info >= (3, 11):
    import tomllib
else:
    tomllib = pytest.importorskip("tomli", reason="lecture TOML")


def _pyproject():
    with open(os.path.join(_REPO, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)


def _requirements(nom):
    """Les exigences d'un fichier `requirements/`, sans les `-r` ni les notes.

    Rendu comme un dict `paquet -> contrainte`, pour que le message d'echec
    dise QUEL paquet et QUELLE contrainte different, pas seulement que les
    listes ne sont pas egales.
    """
    chemin = os.path.join(_REPO, "requirements", nom)
    out = {}
    for ligne in io.open(chemin, encoding="utf-8", errors="replace"):
        ligne = ligne.split("#")[0].strip()
        if not ligne or ligne.startswith("-r"):
            continue
        out[_nom_du_paquet(ligne)] = _normaliser(ligne)
    return out


def _nom_du_paquet(exigence):
    return re.split(r"[<>=!;\s]", exigence.strip(), 1)[0].lower()


def _normaliser(exigence):
    """Meme ecriture des deux cotes : espaces retires, guillemets uniformes.

    `pyproject.toml` ecrit `python_version < '3.11'`, `requirements` ecrit
    `python_version < "3.11"`. C'est la meme contrainte.
    """
    return (exigence.strip().replace(" ", "").replace('"', "'").lower())


# --------------------------------------------------------------------------- #
# 1. LES DEUX DECLARATIONS S'ACCORDENT                                         #
# --------------------------------------------------------------------------- #
def test_les_dependances_de_base_sont_celles_de_core_txt():
    """`[project].dependencies` contre `requirements/core.txt`.

    `pytest` est la seule difference assumee : il est dans `core.txt` parce
    que le harness en a besoin, et dans l'extra `dev` du `.toml` parce qu'un
    utilisateur qui INSTALLE la chaine n'a pas a l'avoir.
    """
    projet = {_nom_du_paquet(d): _normaliser(d)
              for d in _pyproject()["project"]["dependencies"]}
    fichier = _requirements("core.txt")
    fichier.pop("pytest", None)
    assert projet == fichier, _ecart(projet, fichier, "core.txt")


def test_l_extra_etudes_est_celui_de_studies_txt():
    """Les sept paquets de la couche des etudes, avec leurs planchers."""
    extras = _pyproject()["project"]["optional-dependencies"]
    projet = {_nom_du_paquet(d): _normaliser(d) for d in extras["etudes"]}
    fichier = _requirements("studies.txt")
    assert projet == fichier, _ecart(projet, fichier, "studies.txt")


def _ecart(projet, fichier, nom):
    lignes = ["pyproject.toml et requirements/%s ont diverge :" % nom]
    for paquet in sorted(set(projet) | set(fichier)):
        a, b = projet.get(paquet), fichier.get(paquet)
        if a != b:
            lignes.append("  %-16s pyproject=%-18s %s=%s"
                          % (paquet, a or "ABSENT", nom, b or "ABSENT"))
    lignes.append("Les deux sont lues par des gens differents : le `.toml` "
                  "par `pip install -e .`, le `.txt` par le message d'erreur "
                  "du lanceur. Les faire diverger envoie quelqu'un installer "
                  "une chaine incomplete.")
    return "\n".join(lignes)


def test_le_message_du_lanceur_designe_un_fichier_QUI_EXISTE():
    """Il dit `pip install -r requirements/studies.txt`. Un message qui
    nomme un fichier absent est pire que pas de message."""
    src = io.open(os.path.join(_REPO, "launcher.py"), encoding="utf-8",
                  errors="replace").read()
    cites = set(re.findall(r"requirements/([a-z-]+\.txt)", src))
    assert cites, "le lanceur ne renvoie plus vers `requirements/`"
    for nom in cites:
        assert os.path.isfile(os.path.join(_REPO, "requirements", nom)), (
            "le lanceur renvoie vers requirements/%s, qui n'existe pas" % nom)


def test_les_versions_epinglees_couvrent_les_deux_couches():
    """`constraints-reference.txt` sert a REPRODUIRE l'environnement ou les
    etudes ont tourne. Un paquet qui y manque n'est pas epingle, et deux
    postes peuvent alors ne pas calculer la meme chose."""
    epingles = _requirements("constraints-reference.txt")
    manquants = sorted(set(_requirements("core.txt"))
                       | set(_requirements("studies.txt")) - set(epingles))
    manquants = [p for p in manquants if p not in epingles and p != "tomli"]
    assert not manquants, (
        "non epingle(s) dans constraints-reference.txt : %s. Deux postes "
        "peuvent alors installer des versions differentes." % manquants)


# --------------------------------------------------------------------------- #
# 2. LE POINT D'ENTREE EXISTE VRAIMENT                                         #
# --------------------------------------------------------------------------- #
def test_la_commande_fiabilite_pointe_sur_une_fonction_QUI_EXISTE():
    """`fiabilite = "launcher:_console"`. Une entree qui designe une fonction
    absente s'installe SANS ERREUR et echoue a la premiere utilisation."""
    scripts = _pyproject()["project"].get("scripts", {})
    assert "fiabilite" in scripts, "la commande `fiabilite` n'est plus declaree"
    module, _, fonction = scripts["fiabilite"].partition(":")
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    importe = __import__(module)
    assert callable(getattr(importe, fonction, None)), (
        "%s:%s n'est pas appelable" % (module, fonction))


def test_le_module_du_point_d_entree_est_bien_LIVRE():
    """`launcher.py` est a la racine, pas dans un paquet : sans
    `py-modules`, `pip install` ne l'emporterait pas et la commande
    `fiabilite` echouerait a l'import."""
    livres = _pyproject().get("tool", {}).get("setuptools", {}) \
                         .get("py-modules", [])
    assert "launcher" in livres, (
        "`launcher` n'est pas dans `[tool.setuptools] py-modules` : la "
        "commande `fiabilite` s'installerait et ne trouverait pas son module.")


def test_le_message_d_usage_nomme_la_commande_REELLEMENT_tapee():
    """Verifie le 02/09/2026 dans un venv neuf : `fiabilite` sans argument
    repond `usage : fiabilite ...` et sort en code 1. Il repondait
    `usage : python launcher.py ...` avant le 01/09 -- un message qui envoie
    essayer une commande que le lecteur n'a pas."""
    src = io.open(os.path.join(_REPO, "launcher.py"), encoding="utf-8",
                  errors="replace").read()
    assert 'moi = "fiabilite" if moi.startswith("fiabilite")' in src, (
        "le message d'usage ne s'adapte plus au nom reellement tape")
