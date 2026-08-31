r"""Le comparateur de baselines : la chaine de verification, verifiee.

POURQUOI CE FICHIER
--------------------
`tools/baseline_compare.py` est ce qui atteste qu'un changement n'a rien
deplace. Un trou dedans ne casse aucun calcul -- il rend seulement les
attestations sans valeur, y compris toutes celles produites cette semaine.
Il merite donc le meme traitement que le code qu'il surveille.

CE QUI A ETE CONSTATE -- 29/08/2026
------------------------------------
`telemetry.fingerprint` compte les valeurs NON FINIES d'un tableau
(`non_finis`). Ce compte n'etait jamais relu.

Au-dela de 256 elements, les valeurs ne sont pas stockees : la comparaison
porte sur `min`, `max`, `mean` et `l2`, calcules sur les seules valeurs
FINIES. Un NaN nouveau n'y apparait donc que par le decalage qu'il induit.

    n=300      ecart 1,6e-3     vu
    n=5 000    ecart 9,6e-5     vu
    n=50 000   ecart 9,6e-6     vu
    n=200 000  ecart 2,4e-6     vu

Il etait donc toujours vu -- mais PAR ACCIDENT, et l'ecart decroit avec la
taille tandis que certaines tolerances valent 1e-6. Le compte, lui, ne depend
ni de la taille ni de la tolerance.

CE QUI A ETE VERIFIE ET TROUVE SAIN
------------------------------------
* `fingerprint(None)` rend `{"kind": "none"}` : un scalaire absent ne devient
  jamais un `scalar` a valeur nulle, et deux `kind` differents sont declares
  INCOMPARABLES plutot que compares ;
* une grandeur presente d'un seul cote, une forme differente, des statistiques
  absentes : toutes menent au code de sortie 2, jamais a un silence ;
* les valeurs non finies stockees deviennent `None`, et la boucle de
  comparaison rend l'infini des qu'un `None` fait face a un nombre.
"""

import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "tools"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

np = pytest.importorskip("numpy")

import baseline_compare as bc   # noqa: E402
import telemetry                # noqa: E402


def _grand(n, nan_a=None):
    """Un tableau plus grand que `MAX_VALEURS_STOCKEES` : seules les
    statistiques seront comparees."""
    a = np.linspace(1.0, 2.0, n)
    if nan_a is not None:
        a = a.copy()
        a[nan_a] = float("nan")
    return telemetry.fingerprint(a)


# --------------------------------------------------------------------------- #
# 1. UN NaN QUI APPARAIT EST VU, QUELLE QUE SOIT LA TAILLE                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [300, 5000, 200000])
def test_un_NaN_nouveau_est_une_divergence(n):
    ref = _grand(n)
    neuf = _grand(n, nan_a=int(n // 2))
    assert ref.get("non_finis") is None and neuf["non_finis"] == 1
    assert bc.ecart(ref, neuf) == float("inf")


def test_un_NaN_qui_disparait_est_aussi_une_divergence():
    """Le sens compte peu : ce qui compte est que le nombre ait change."""
    assert bc.ecart(_grand(1000, nan_a=10), _grand(1000)) == float("inf")


def test_deux_runs_avec_LE_MEME_NaN_ne_divergent_pas():
    """Sinon toute grandeur portant un trou legitime serait signalee a
    chaque comparaison."""
    a = _grand(1000, nan_a=10)
    assert bc.ecart(a, _grand(1000, nan_a=10)) == 0.0


def test_le_compte_est_bien_celui_des_valeurs_non_finies():
    a = np.array([1.0, float("nan"), float("inf"), 4.0])
    assert telemetry.fingerprint(a)["non_finis"] == 2


# --------------------------------------------------------------------------- #
# 2. CE QUI ETAIT DEJA SAIN -- et qui doit le rester                           #
# --------------------------------------------------------------------------- #
def test_un_petit_tableau_voit_le_NaN_par_ses_valeurs():
    """En dessous de 256 elements les valeurs sont stockees, non finies
    remplacees par `None` : la boucle rend l'infini des qu'un `None` fait
    face a un nombre."""
    ref = telemetry.fingerprint(np.array([1.0, 2.0, 3.0, 4.0]))
    neuf = telemetry.fingerprint(np.array([1.0, 2.0, float("nan"), 4.0]))
    assert "values" in ref
    assert bc.ecart(ref, neuf) == float("inf")


def test_une_valeur_absente_n_est_pas_un_zero():
    """`fingerprint(None)` ne doit jamais devenir un scalaire nul : un beta
    que FORM n'a pas trouve n'est pas un beta de zero."""
    assert telemetry.fingerprint(None) == {"kind": "none"}


def test_deux_natures_differentes_sont_INCOMPARABLES():
    """Et non « egales » ou « differentes » : le comparateur sort en 2."""
    assert bc.ecart(telemetry.fingerprint(None),
                    telemetry.fingerprint(3.0)) is None


def test_deux_absences_sont_egales():
    assert bc.ecart(telemetry.fingerprint(None),
                    telemetry.fingerprint(None)) == 0.0


def test_deux_formes_differentes_sont_INCOMPARABLES():
    a = telemetry.fingerprint(np.zeros((2, 3)))
    b = telemetry.fingerprint(np.zeros((3, 2)))
    assert bc.ecart(a, b) is None


def test_un_scalaire_nul_ne_divise_pas_par_zero():
    a = telemetry.fingerprint(0.0)
    assert bc.ecart(a, telemetry.fingerprint(0.0)) == 0.0
    assert bc.ecart(a, telemetry.fingerprint(5.0)) == 5.0


def test_un_condensat_de_suite_d_appels_n_a_pas_de_demi_mesure():
    a = telemetry.fingerprint("doe;fit;eff")
    assert bc.ecart(a, telemetry.fingerprint("doe;fit;eff")) == 0.0
    assert bc.ecart(a, telemetry.fingerprint("doe;eff")) == float("inf")


# --------------------------------------------------------------------------- #
# 3. LA REGLE : ce que l'empreinte enregistre, le comparateur le regarde      #
# --------------------------------------------------------------------------- #
#: Champs enregistres SANS etre compares, et pourquoi c'est admis.
NON_COMPARES = {
    # `shape` est compare, et `n` en decoule exactement.
    "n": "redondant avec shape",
    # Un changement de dtype a valeurs egales ne deplace aucun resultat ; il
    # viendrait d'un changement de code, que le journal montre par ailleurs.
    "dtype": "n'affecte pas les valeurs comparees",
    # Servent a comparer, pas a etre compares.
    "kind": "sert a decider de la comparaison",
    "hash": "sert de raccourci d'egalite",
    "stats": "compare champ par champ",
    "values": "compare element par element",
    "value": "compare directement",
    "keys": "compare par egalite de dictionnaire",
    "repr": "compare par egalite de dictionnaire",
}


def test_aucun_champ_d_empreinte_n_est_enregistre_pour_rien():
    """La regle qui a trouve `non_finis`.

    Un champ ecrit dans l'empreinte et jamais relu par le comparateur est une
    mesure payee et jetee -- et, ici, une divergence qui peut passer.
    """
    import io
    champs = set()
    for valeur in (3.0, "x", {"a": 1}, np.zeros(4),
                   np.array([1.0, float("nan")]), np.zeros(400), None):
        champs |= set(telemetry.fingerprint(valeur))
    src = io.open(os.path.join(_REPO, "tools", "baseline_compare.py"),
                  encoding="utf-8").read()
    oublies = sorted(c for c in champs
                     if c not in NON_COMPARES and '"%s"' % c not in src)
    assert not oublies, (
        "champ(s) d'empreinte enregistre(s) et jamais relu(s) : %s.\n"
        "Soit le comparateur doit les regarder, soit ils doivent rejoindre "
        "NON_COMPARES avec la raison." % ", ".join(oublies))
