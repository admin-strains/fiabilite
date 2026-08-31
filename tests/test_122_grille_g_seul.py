r"""Les deux voies de la grille HF jugent un point de la MEME facon.

LE DEFAUT, MESURE -- 29/08/2026
--------------------------------
La grille haute fidelite DESSINE une surface : ses quatre sites d'appel
s'ecrivent tous `self.evaluer(pt)[0]`, elle ne lit jamais de gradient et ne
nourrit aucun metamodele.

Elle etait pourtant cablee sur `run_HF` -- donc sur `evaluer_en_U`, qui LEVE
quand le solveur n'a rendu aucun gradient. Ce refus est VOULU pour un point
d'enrichissement : ce point-la a ete demande par l'algorithme, et un gradient
fabrique a zero contaminerait le metamodele.

Mais la grille a DEUX voies, et une seule passait par la. Mesure sur un
solveur qui converge sans rendre de sensibilites, le meme point des deux
cotes :

    n_workers <= 1   voie sequentielle (`evaluer_en_U`)  -> ValueError
    n_workers >  1   voie parallele (`evaluer_plan`)     -> g = 0.42, accepte

Une grille pouvait donc mourir ou aboutir selon le seul nombre de workers.
Et `tools/run_comparatif.py` impose `n_workers_DOE = 1` : toute comparaison
A/B prenait la voie stricte pendant que la production prenait l'autre.

LE CORRECTIF
-------------
`Evaluateur.evaluer_g_en_U` : meme maillage, meme appel solveur, meme forme
de retour, sans l'exigence de gradient. La grille est cablee dessus ;
l'enrichissement garde `evaluer_en_U` et son refus.

CE QUE CE FICHIER VERIFIE
--------------------------
Que les deux voies s'accordent MAINTENANT, et -- tout aussi important -- que
la garde de l'enrichissement n'a pas ete affaiblie au passage.
"""

import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_doe"), os.path.join(_REPO, "_etapes"),
           os.path.join(_REPO, "solver")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ot = pytest.importorskip("openturns", reason="la couche etudes n'est pas installee")

import evaluation as _ev            # noqa: E402
from interface import Evaluation    # noqa: E402

PARAMS = ["fc", "fy"]
U = [0.5, -1.5]


class _SolveurSansGradient:
    """Converge, mais ne rend AUCUNE sensibilite.

    Ce n'est pas un cas d'ecole : `exclure_points_sans_gradient` existe
    precisement parce que Digital Structure le fait.
    """

    def evaluer(self, params, sensibilite=False, etiquette=None):
        return Evaluation(g=0.42, alpha=1.42, grad_x=(None, None), sain=True)


class _SolveurComplet:
    def evaluer(self, params, sensibilite=False, etiquette=None):
        return Evaluation(g=0.42, alpha=1.42, grad_x=(1.0, 2.0), sain=True)


def _evaluateur(solveur):
    return _ev.Evaluateur(
        solveur_pour=lambda nom=None: solveur,
        dist=ot.JointDistribution([ot.Normal(48.0, 5.0), ot.Normal(550.0, 30.0)]),
        params_names=PARAMS, exclure_non_converges=False, archiver=False,
        journaliser=lambda *a, **k: None, sauver_partiel=lambda *a, **k: None)


# --------------------------------------------------------------------------- #
# 1. LES DEUX VOIES S'ACCORDENT                                                #
# --------------------------------------------------------------------------- #
def test_la_voie_de_la_grille_accepte_un_point_sans_gradient():
    """C'est ce que la voie parallele faisait deja ; la sequentielle levait."""
    g, grad_U, grad_X = _evaluateur(_SolveurSansGradient()).evaluer_g_en_U(U)
    assert g == 0.42
    assert list(grad_U) == [None, None]
    assert list(grad_X) == [None, None]


def test_les_deux_voies_rendent_le_meme_g():
    """Le point est le meme, le solveur est le meme : le `g` doit l'etre.

    Voie parallele = `evaluer_plan`, ce que le worker appelle. Voie de la
    grille = `evaluer_g_en_U`. Elles partaient d'un desaccord total (l'une
    rendait une valeur, l'autre levait).
    """
    ev = _evaluateur(_SolveurSansGradient())
    g_grille, _, _ = ev.evaluer_g_en_U(U)

    T_inv = ev.dist.getInverseIsoProbabilisticTransformation()
    x = T_inv(ot.Point(U))
    SOL = ev.evaluer_plan([{p: float(x[j]) for j, p in enumerate(PARAMS)}],
                          sensibilite=True)
    assert SOL[0]["g"] == g_grille


def test_avec_gradient_les_deux_evaluateurs_rendent_la_meme_chose():
    """Quand le solveur repond completement, rien ne doit differer.

    C'est le cas courant -- celui de toutes les chaines de reference. Il
    garantit que la scission n'a pas deplace le cas nominal.
    """
    ev = _evaluateur(_SolveurComplet())
    assert ev.evaluer_g_en_U(U) == ev.evaluer_en_U(U)


# --------------------------------------------------------------------------- #
# 2. LA GARDE DE L'ENRICHISSEMENT N'EST PAS AFFAIBLIE                          #
# --------------------------------------------------------------------------- #
def test_l_enrichissement_refuse_toujours_un_point_sans_gradient():
    """Le refus est VOULU la : le point a ete DEMANDE par l'algorithme, et
    l'ecarter en silence le lui ferait reproposer indefiniment."""
    with pytest.raises(ValueError, match="aucun gradient"):
        _evaluateur(_SolveurSansGradient()).evaluer_en_U(U)


def test_le_message_du_refus_renvoie_au_parametre_qui_gouverne_l_autre_voie():
    ev = _evaluateur(_SolveurSansGradient())
    with pytest.raises(ValueError) as exc:
        ev.evaluer_en_U(U)
    assert "exclure_points_sans_gradient" in str(exc.value)


# --------------------------------------------------------------------------- #
# 3. LA REGLE : la grille ne lit QUE `g`, et n'exige donc rien de plus        #
# --------------------------------------------------------------------------- #
def test_la_grille_ne_lit_que_la_premiere_valeur_de_son_evaluateur():
    """L'argument du correctif, verifie sur le texte plutot qu'affirme.

    Si un site d'appel se mettait a lire le gradient, le cablage sur
    `evaluer_g_en_U` deviendrait faux -- il rend des `None`.
    """
    import io
    import re
    src = io.open(os.path.join(_REPO, "_etapes", "grille.py"),
                  encoding="utf-8", errors="replace").read()
    appels = re.findall(r"self\.evaluer\(([^)]*)\)(\[\d\])?", src)
    assert appels, "aucun appel trouve : l'analyse a rate sa cible"
    indices = {suffixe for _, suffixe in appels}
    assert indices == {"[0]"}, (
        "la grille lit autre chose que `g` (%s). Son evaluateur est "
        "`evaluer_g_en_U`, dont le gradient peut valoir None." % sorted(indices))


@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_l_etude_cable_la_grille_sur_la_voie_sans_gradient(script):
    import io
    src = io.open(os.path.join(_REPO, script), encoding="utf-8",
                  errors="replace").read()
    assert "evaluer_g_en_U" in src, (
        "%s : la grille n'est pas cablee sur l'evaluateur sans gradient ; "
        "elle mourra a un worker la ou elle aboutit a six." % script)
    assert "evaluer=run_HF," not in src, (
        "%s : la grille est revenue sur l'evaluateur d'enrichissement."
        % script)
