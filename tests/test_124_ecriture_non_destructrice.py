r"""Une ecriture qui echoue ne doit pas emporter le fichier precedent.

LE DEFAUT -- MESURE LE 29/08/2026
----------------------------------
Neuf ecritures de ce depot suivaient le motif :

    json.dump(objet, open(fichier, "w"), indent=1)

`open(fichier, "w")` TRONQUE avant que `json.dump` ne serialise. Une seule
valeur refusee, et le fichier precedent devient un fragment :

    avant  : {"xt": [[1.0, 2.0]], "n_total": 1}
    apres  : {"xt": [[1.0, 2.0]], "coupe":

C'est pire qu'une suppression : le fichier existe, avec une date recente.

CE QUE CE MOTIF PORTAIT

    restart_state.json      jusqu'a 90 heures d'enrichissement
    hf_grid_cache*.json     225 points, soit 29 heures sur le Moulin Blanc
    doe_cache.json          le plan initial, et son filet de reprise
    hf_custom_cache.json    la grille de points libres

Le commentaire de `reprise.enregistrer` disait deja l'intention : « une
erreur de serialisation ne doit pas emporter le calcul qui vient de se
terminer ». Elle emportait le PRECEDENT.

ET IL Y AVAIT DE QUOI DECLENCHER

`coupe_la_plus_parlante` rendait les indices d'axes en `np.int64`, que
`json.dumps` refuse. Cette coupe part dans les caches de grille ET dans le
dump de reprise. Le Moulin Blanc passait par la ; la flexion pure non, parce
qu'elle codait sa coupe finale en dur, avec des `int`. Un defaut qui
n'existait donc que d'un seul cote -- le quatrieme de la semaine.
"""

import io
import json
import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_cache"), os.path.join(_REPO, "_reliability")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

np = pytest.importorskip("numpy")

import ecriture as _ecriture   # noqa: E402


class _Refuse:
    """Un objet que `json` ne sait pas serialiser."""


# --------------------------------------------------------------------------- #
# 1. L'ANCIEN SURVIT                                                           #
# --------------------------------------------------------------------------- #
def test_une_serialisation_impossible_laisse_l_ancien_intact(tmp_path):
    f = str(tmp_path / "cache.json")
    _ecriture.ecrire_json({"xt": [[1.0, 2.0]], "n_total": 1}, f)
    avant = io.open(f, encoding="utf-8").read()

    with pytest.raises(TypeError):
        _ecriture.ecrire_json({"xt": [[1.0, 2.0]], "coupe": _Refuse()}, f)

    assert io.open(f, encoding="utf-8").read() == avant, (
        "le fichier precedent a ete emporte par une ecriture qui a echoue")
    assert json.load(open(f))["n_total"] == 1


def test_un_cache_de_grille_survit_a_une_ecriture_refusee(tmp_path):
    """La forme reelle d'un cache de grille, avec une valeur indigeste.

    Le declencheur d'origine etait `np.int64` -- il ne leve PLUS, puisque la
    traduction exacte l'accepte (voir plus bas). La propriete verifiee ici est
    l'autre : quoi qu'il arrive a l'ecriture, l'ancien contenu reste.
    """
    f = str(tmp_path / "hf.json")
    _ecriture.ecrire_json({"Z": [[0.0]], "n_grid_hf": 1}, f)
    with pytest.raises(TypeError):
        _ecriture.ecrire_json({"Z": [[0.0]], "slice_def": [0, 1, _Refuse()]}, f)
    assert json.load(open(f))["n_grid_hf"] == 1


def test_aucun_temporaire_ne_reste_apres_un_echec(tmp_path):
    """Un `.tmp` oublie ferait croire a une ecriture en cours."""
    f = str(tmp_path / "cache.json")
    with pytest.raises(TypeError):
        _ecriture.ecrire_json({"x": _Refuse()}, f)
    assert not os.path.exists(f + ".tmp")
    assert not os.path.exists(f), "un fichier a ete cree alors que rien n'a pu etre ecrit"


def test_l_ecriture_qui_reussit_remplace_bien(tmp_path):
    f = str(tmp_path / "cache.json")
    _ecriture.ecrire_json({"n": 1}, f)
    _ecriture.ecrire_json({"n": 2}, f)
    assert json.load(open(f)) == {"n": 2}
    assert not os.path.exists(f + ".tmp")


def test_le_contenu_est_du_JSON_indente_comme_avant(tmp_path):
    """Les fichiers sont relus par des outils et par des humains : le format
    ne doit pas bouger avec le mecanisme d'ecriture."""
    f = str(tmp_path / "cache.json")
    _ecriture.ecrire_json({"a": 1, "b": [1, 2]}, f, indent=1)
    assert io.open(f, encoding="utf-8").read() == json.dumps(
        {"a": 1, "b": [1, 2]}, indent=1)


# --------------------------------------------------------------------------- #
# 2. LA COUPE FINALE EST SERIALISABLE                                          #
# --------------------------------------------------------------------------- #
class _Resultat:
    def __init__(self, imp, u):
        self._i, self._u = imp, u

    def getImportanceFactors(self):
        return self._i

    def getStandardSpaceDesignPoint(self):
        return self._u


def test_la_coupe_la_plus_parlante_rend_des_int():
    import form as _form
    sd = _form.coupe_la_plus_parlante(
        _Resultat([0.3, 0.6, 0.1], [-1.0, -2.0, -3.0]), 3, (0, 1, {}))
    assert isinstance(sd[0], int) and not isinstance(sd[0], np.integer)
    assert isinstance(sd[1], int) and not isinstance(sd[1], np.integer)


def test_la_coupe_la_plus_parlante_passe_dans_un_cache(tmp_path):
    """Le vrai test : elle traverse la meme mise en forme que les caches."""
    import form as _form
    sd = _form.coupe_la_plus_parlante(
        _Resultat([0.3, 0.6, 0.1], [-1.0, -2.0, -3.0]), 3, (0, 1, {}))
    f = str(tmp_path / "hf.json")
    _ecriture.ecrire_json(
        {"slice_def": [sd[0], sd[1], {str(k): v for k, v in sd[2].items()}]}, f)
    assert json.load(open(f))["slice_def"] == [0, 1, {"2": -3.0}]


def test_a_deux_variables_elle_donne_la_coupe_codee_en_dur():
    """La flexion pure ecrivait `(0, 1, {})` a la main. C'est ce que le
    mecanisme general donne -- quel que soit l'ordre d'importance."""
    import form as _form
    for imp in ([0.7, 0.3], [0.3, 0.7]):
        sd = _form.coupe_la_plus_parlante(_Resultat(imp, [-1.0, -2.0]), 2,
                                          (0, 1, {}))
        assert sd == (0, 1, {}), sd


# --------------------------------------------------------------------------- #
# 3. LA REGLE : plus une seule ecriture destructrice                           #
# --------------------------------------------------------------------------- #
DOSSIERS = ("_cache", "_doe", "_etapes", "_reliability", "_surrogate",
            "_model", "solver")


def test_aucune_ecriture_json_ne_tronque_avant_de_serialiser():
    """La regle, pas le cas.

    `json.dump(obj, open(f, "w"))` ouvre avant de serialiser. Le motif ne doit
    plus exister dans le code qui ecrit des fichiers de calcul.
    """
    import ast
    # SUR L'ARBRE, PAS SUR LE TEXTE. La premiere version cherchait le motif
    # par expression reguliere, et condamnait le docstring de `ecriture.py`
    # lui-meme -- qui le CITE pour l'expliquer. Une sonde qui condamne la
    # documentation de ce qu'elle garde n'est pas une sonde.
    coupables = []
    for dossier in DOSSIERS:
        racine = os.path.join(_REPO, dossier)
        if not os.path.isdir(racine):
            continue
        for r, _, fichiers in os.walk(racine):
            if "__pycache__" in r:
                continue
            for nom in fichiers:
                if not nom.endswith(".py"):
                    continue
                chemin = os.path.join(r, nom)
                src = io.open(chemin, encoding="utf-8", errors="replace").read()
                for n in ast.walk(ast.parse(src, chemin)):
                    if not (isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Attribute)
                            and n.func.attr == "dump"):
                        continue
                    if any(isinstance(a, ast.Call)
                           and isinstance(a.func, ast.Name)
                           and a.func.id == "open" for a in n.args):
                        coupables.append("%s:%d" % (
                            os.path.relpath(chemin, _REPO), n.lineno))
    assert not coupables, (
        "ecriture(s) qui tronquent avant de serialiser : %s.\n"
        "Passer par `_cache.ecriture.ecrire_json` : une valeur refusee leve "
        "AVANT que le fichier ne soit touche, et un processus tue pendant "
        "l'ecriture laisse l'ancien intact." % ", ".join(coupables))


def test_les_caches_passent_tous_par_le_helper():
    """Le pendant positif : la regle ne doit pas etre satisfaite en cessant
    d'ecrire."""
    n = 0
    for nom in ("doe.py", "hf.py", "reprise.py"):
        src = io.open(os.path.join(_REPO, "_cache", nom), encoding="utf-8").read()
        n += src.count("_ecriture.ecrire_json(")
    assert n >= 7, "seulement %d ecriture(s) protegee(s) dans _cache" % n


# --------------------------------------------------------------------------- #
# 4. LES TYPES NUMPY : traduits exactement, jamais devines                     #
# --------------------------------------------------------------------------- #
def test_un_entier_numpy_n_empeche_plus_d_ecrire(tmp_path):
    """`np.float64` est une sous-classe de `float` et JSON l'acceptait deja ;
    `np.int64` n'est PAS une sous-classe de `int`, et il etait refuse.

    Mesure du 29/08/2026 : sur huit ecritures de cache nourries de types
    numpy, QUATRE n'ecrivaient aucun fichier -- toutes celles qui portaient un
    entier (`n0`, `n_grid`, un champ de signature).
    """
    f = str(tmp_path / "cache.json")
    _ecriture.ecrire_json({"n0": np.int64(5), "cote": np.int32(7)}, f)
    assert json.load(open(f)) == {"n0": 5, "cote": 7}


@pytest.mark.parametrize("valeur,attendu", [
    (np.int64(-3), -3),
    (np.int32(2 ** 30), 2 ** 30),
    (np.float64(0.1), 0.1),
    (np.float32(0.5), 0.5),
    (np.bool_(True), True),
    (np.array([[1.5, 2.5]]), [[1.5, 2.5]]),
])
def test_la_traduction_est_EXACTE(tmp_path, valeur, attendu):
    """Une traduction, pas une tolerance qui devine : ces caches portent des
    heures de solveur, la valeur relue doit etre la valeur ecrite."""
    f = str(tmp_path / "cache.json")
    _ecriture.ecrire_json({"v": valeur}, f)
    assert json.load(open(f))["v"] == attendu


def test_un_objet_inconnu_LEVE_toujours(tmp_path):
    """La traduction ne doit pas devenir une tolerance generale : ce qu'on ne
    sait pas ecrire ne doit pas etre ecrit a peu pres."""
    f = str(tmp_path / "cache.json")
    with pytest.raises(TypeError):
        _ecriture.ecrire_json({"v": _Refuse()}, f)
    assert not os.path.exists(f)


def test_un_grand_entier_numpy_garde_sa_valeur(tmp_path):
    """`int(np.int64)` ne tronque pas -- contrairement a un passage par
    `float`, qui perdrait les entiers au-dela de 2^53."""
    f = str(tmp_path / "cache.json")
    grand = np.int64(2 ** 62 + 1)
    _ecriture.ecrire_json({"v": grand}, f)
    assert json.load(open(f))["v"] == 2 ** 62 + 1
