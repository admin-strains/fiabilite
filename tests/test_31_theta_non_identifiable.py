r"""`theta` n'est pas determine par les donnees -- la mesure, et ses limites.

POURQUOI CE FICHIER EXISTE
---------------------------
Le 31/08/2026, les cinq jobs d'integration continue rendaient cinq `theta`
differents sur les memes goldens. La question « comment rendre cela
reproductible ? » a recu trois reponses successives, et les deux premieres
etaient fausses. Ce fichier fige la troisieme, pour que personne ne refasse
le chemin.

**Ce n'est pas Linux.** `noyau windows-latest py3.10` -- la configuration la
plus proche du poste de reference -- reproduit les goldens EXACTEMENT depuis
le bridage a un thread. Ce sont les autres qui divergent.

**Ce n'est pas la version des bibliotheques.** Les goldens ont ete produits
sous numpy 2.1.1 / scipy 1.15.2 et passent sous 2.2.6 / 1.15.3 sur la meme
machine.

**Ce n'est pas seulement le nombre de threads.** Le bridage a un thread a
rendu VERT le job Windows/3.10 -- premier job vert de ce depot -- mais les
quatre autres divergent toujours.

**Et ce n'est pas l'optimiseur.** C'est la mesure qui a coute le plus a
obtenir, et c'est elle qui ferme le sujet : voir ci-dessous.

CE QUE CE FICHIER NE FAIT PAS
------------------------------
Il ne corrige rien et ne verifie aucune valeur de `theta`. Il enregistre une
PROPRIETE du probleme -- l'optimum est un plateau -- et deux consequences
verifiables : le mode qui n'optimise pas est stable, et le LOO ne bouge pas
quand `theta` bouge de 97 %.
"""

import os
import sys
import warnings

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in ("_lib", "_model"):
    _c = os.path.join(_REPO, _p)
    if _c not in sys.path:
        sys.path.insert(0, _c)
if _ICI not in sys.path:
    sys.path.insert(0, _ICI)

np = pytest.importorskip("numpy")

import harness                                              # noqa: E402


#: Releve du 01/09/2026, meme poste, meme numpy, meme DOE, `linear/PCK`.
#: Seul le nombre de threads BLAS change entre les deux colonnes.
_MESURE_DES_THREADS = """
    mode            7 threads              1 thread
    sequential      [0.999989, 0.999989]   [1.000000, 1.000000]
    optimal (DE)    [54.5629, 37.7713]     [31.5094, 100.0000]
"""

#: Ce qui a ete essaye pour fermer le sujet, et mesure insuffisant : un
#: multi-depart DETERMINISTE a la place de `differential_evolution`.
#: Ecart relatif de theta entre 7 et 1 thread -- plus petit est meilleur.
_MESURE_DU_MULTIDEPART = """
    cas              DE         multi-depart
    flexion/PCK      9.71e-01   9.70e-01
    flexion/GEPCK    3.54e-02   8.16e-02
    linear/PCK       7.32e-01   2.78e-01
    linear/GEPCK     0.00e+00   6.72e-02      <- DE etait EXACT ici
"""


def _ajuster(mode, case="linear", kind="PCK"):
    import json
    from reference.limit_states import CASES
    with open(os.path.join(_REPO, "tests", "golden", case + ".json"),
              encoding="utf-8") as fh:
        ref = json.load(fh)
    opts = {"Mode": mode, "PCE": {"Degree": list(range(1, ref["max_degree"] + 1)),
                                  "Method": "LARS"}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm = harness.fit(kind, np.asarray(ref["doe"]), CASES[case](), opts=opts)
    return fm


# --------------------------------------------------------------------- #
# 1. le plateau existe : sans optimisation, theta ne bouge pas de theta0
# --------------------------------------------------------------------- #
def test_sur_le_cas_lineaire_l_optimiseur_n_a_rien_a_optimiser():
    """`sequential` n'appelle pas `differential_evolution`, et rend theta0.

    theta0 vaut la moyenne geometrique des bornes, sqrt(0.01 * 100) = 1. Le
    mode `sequential` rend [1, 1] a la cinquieme decimale : l'optimiseur ne
    trouve NULLE PART ou aller. C'est la preuve directe que la vraisemblance
    est plate sur ce cas -- l'etat limite lineaire est deja represente
    EXACTEMENT par la PCE (LOO ~ 1e-25), il ne reste aucun residu que le
    krigeage puisse expliquer.

    C'est aussi ce qui rend `theta` non identifiable : une portee de
    correlation ajustee sur un residu nul n'a pas de valeur vraie.
    """
    fm = _ajuster("sequential")
    theta = np.asarray(fm["Kriging"][0]["theta"], float).ravel()
    assert np.allclose(theta, 1.0, atol=1e-4), (
        "theta = %s, attendu ~[1, 1] (= theta0). Si l'optimiseur bouge "
        "maintenant, la vraisemblance n'est plus plate sur ce cas et TOUT ce "
        "fichier est a relire." % theta)


# --------------------------------------------------------------------- #
# 2. la consequence : le LOO ne suit pas theta
# --------------------------------------------------------------------- #
#: Un LOO sous ce seuil designe un metamodele qui interpole son plan a la
#: precision machine ou presque. Sur ces deux cas de reference, les quatre
#: ajustements y sont largement -- de 1e-10 a 1e-29.
LOO_EXCELLENT = 1e-8


def test_un_theta_tres_different_donne_un_modele_aussi_bon():
    """LA raison de ne pas figer `theta` dans un golden.

    Mesure sur la flexion pure -- le cas NON degenere, choisi expres : sur le
    cas lineaire les deux LOO valent 1e-25 et 1e-29, deux planchers
    numeriques dont la comparaison ne veut rien dire (premiere version de ce
    temoin, corrigee).

        mode         theta              LOO
        sequential   [0.685, 2.370]     5.10e-10
        optimal      [0.010, 6.549]     3.17e-09

    theta bouge de 98 %, et les DEUX modeles interpolent excellemment. Un
    golden qui fige theta a `rtol=1e-8` fige donc une grandeur que le
    probleme ne determine pas, pendant que celle qui compte est stable.

    Le seuil est ABSOLU et non relatif : comparer deux LOO entre eux, quand
    tous deux sont sous le plancher, revient a comparer du bruit.
    """
    seq = _ajuster("sequential", case="flexion")
    opt = _ajuster("optimal", case="flexion")
    t_seq = np.asarray(seq["Kriging"][0]["theta"], float).ravel()
    t_opt = np.asarray(opt["Kriging"][0]["theta"], float).ravel()
    loo_seq = float(seq["Error"][0]["LOO"])
    loo_opt = float(opt["Error"][0]["LOO"])

    ecart_theta = float(np.max(np.abs(t_opt - t_seq)
                               / np.maximum(np.abs(t_seq), 1e-30)))
    assert ecart_theta > 0.5, (
        "theta ne differe plus que de %.0f %% entre les deux modes ; ce "
        "temoin suppose un ecart d'au moins 50 %%." % (100 * ecart_theta))

    assert loo_seq < LOO_EXCELLENT and loo_opt < LOO_EXCELLENT, (
        "LOO %.3e (sequential) et %.3e (optimal) : l'un des deux depasse le "
        "seuil de %.0e. Si un theta different degrade maintenant le modele, "
        "alors theta PORTE de l'information et l'argument de ce fichier "
        "tombe -- a relire entierement." % (loo_seq, loo_opt, LOO_EXCELLENT))


# --------------------------------------------------------------------- #
# 3. les mesures, consignees pour qu'on ne refasse pas le chemin
# --------------------------------------------------------------------- #
def test_les_mesures_qui_ont_ferme_le_sujet_sont_ecrites():
    """Un garde de DOCUMENTATION, et il est assume comme tel.

    Trois pistes ont ete suivies puis abandonnees sur mesure : le systeme
    d'exploitation, la version des bibliotheques, et le remplacement de
    l'optimiseur. La derniere a coute le plus cher a ecarter -- il a fallu
    ecrire le multi-depart, le mesurer, puis balayer sa tolerance d'ex aequo
    de 1e-12 a 1e-2 pour etablir que le probleme n'etait pas la.

    Sans ces chiffres, la prochaine personne qui verra cinq theta dans cinq
    journaux proposera exactement la meme chose.
    """
    for texte, attendu in ((_MESURE_DES_THREADS, "sequential"),
                           (_MESURE_DU_MULTIDEPART, "multi-depart")):
        assert attendu in texte and "e-0" in texte or "[" in texte, texte
    assert "0.00e+00" in _MESURE_DU_MULTIDEPART, (
        "la ligne qui compte le plus a disparu : sur `linear/GEPCK`, DE etait "
        "EXACTEMENT stable et le multi-depart ne l'etait pas. C'est elle qui "
        "interdit de presenter le remplacement comme une amelioration.")
