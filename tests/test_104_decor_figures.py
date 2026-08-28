r"""Le decor des figures : sept valeurs qui etaient vingt variables libres.

CE QUE C'ETAIT
---------------
Chaque fonction de trace capturait par fermeture le cadre, la resolution, les
noms de variables, le nom du modele, le dossier de sortie et l'horodatage --
vingt a vingt-cinq variables libres par fonction, dans chacune des deux
etudes. C'est ce couplage, et non la longueur, qui rendait ces fonctions
inextricables du script.

CE QUE CES TESTS PROTEGENT
---------------------------
1. **La grille de coupe rend des points COMPLETS.** Une coupe fixe toutes les
   variables sauf deux ; les points passes au metamodele doivent tout de meme
   porter leurs `n_var` composantes, dont `n_var - 2` constantes. Se tromper
   la donne une figure d'une surface qui n'existe pas.
2. **Le cadre est le meme partout.** Deux planches cadrees differemment ne se
   comparent pas -- et c'etait precisement la cinquieme divergence entre les
   deux etudes.
3. **Le module ne touche jamais le solveur.** `planche_EFF` obtenait
   autrefois son fond haute fidelite elle-meme : jusqu'a 225 appels, 29
   heures sur le Moulin Blanc, sous un nom qui dit « imprime ».

VERIFIE AUTREMENT, AUSSI
-------------------------
Les tests ci-dessous verifient la mecanique. Que le DESSIN soit inchange a
ete verifie a part, en rejouant l'etude analytique avant et apres
l'extraction et en comparant les images : les douze figures sont identiques
bit a bit.
"""

import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
if os.path.join(_REPO, "_etapes") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "_etapes"))

np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

import figurer                                        # noqa: E402

SCRIPTS = ["pure_flexion/AC3_pure_flexion.py", "Moulinblanc/AC3_moulinblanc.py"]


def _decor(tmp_path, n_grid=5, params=("fc", "fy"), cadre=(-3.0, 3.0, -2.0, 2.0)):
    return figurer.Decor(cadre, n_grid, list(params), "PCK", str(tmp_path),
                         "0101_1200", tracer=lambda _m: None)


# --------------------------------------------------------------------- #
# la grille de coupe
# --------------------------------------------------------------------- #
def test_le_maillage_couvre_exactement_le_cadre(tmp_path):
    UX, UY = _decor(tmp_path).maillage()
    assert UX.shape == (5, 5)
    assert UX.min() == pytest.approx(-3.0) and UX.max() == pytest.approx(3.0)
    assert UY.min() == pytest.approx(-2.0) and UY.max() == pytest.approx(2.0)


def test_une_coupe_a_deux_variables_donne_des_points_a_deux_composantes(tmp_path):
    points = _decor(tmp_path).grille_de_coupe((0, 1, {}))
    assert points.shape == (25, 2)


def test_une_coupe_dans_un_espace_plus_grand_FIGE_les_autres_variables(tmp_path):
    """Le metamodele ne sait pas qu'on regarde un plan : il attend un point
    complet. Oublier les variables figees donnerait la figure d'une surface
    qui n'existe pas."""
    d = _decor(tmp_path, params=("fc", "fy", "s", "w"))
    points = d.grille_de_coupe((0, 2, {1: 1.5, 3: -0.5}))
    assert points.shape == (25, 4)
    assert np.all(points[:, 1] == 1.5)
    assert np.all(points[:, 3] == -0.5)
    # les deux axes de la coupe balaient bien le cadre
    assert points[:, 0].min() == pytest.approx(-3.0)
    assert points[:, 2].min() == pytest.approx(-2.0)


def test_la_coupe_respecte_l_ordre_des_axes(tmp_path):
    """`(idx_x, idx_y)` designe QUELLE variable va en abscisse. Les inverser
    transpose la figure sans que rien ne le dise."""
    d = _decor(tmp_path, params=("a", "b", "c"))
    points = d.grille_de_coupe((2, 0, {1: 0.0}))
    assert points[:, 2].min() == pytest.approx(-3.0)   # x -> variable 2
    assert points[:, 0].min() == pytest.approx(-2.0)   # y -> variable 0


# --------------------------------------------------------------------- #
# les etiquettes
# --------------------------------------------------------------------- #
def test_les_etiquettes_nomment_les_variables_de_la_coupe(tmp_path):
    x, y, figees = _decor(tmp_path).etiquettes((0, 1, {}))
    assert (x, y) == ("u_fc", "u_fy")
    assert figees == ""


def test_les_etiquettes_disent_ce_qui_est_FIGE(tmp_path):
    """Une figure qui ne dit pas a quelle valeur les autres variables sont
    figees n'est pas interpretable."""
    d = _decor(tmp_path, params=("fc", "fy", "s"))
    _x, _y, figees = d.etiquettes((0, 1, {2: 1.25}))
    assert "s=1.2" in figees or "s=1.3" in figees


# --------------------------------------------------------------------- #
# le cadre, identique partout
# --------------------------------------------------------------------- #
def test_toutes_les_figures_recoivent_le_MEME_cadre(tmp_path):
    """C'etait la cinquieme divergence entre les deux etudes : deux planches
    cadrees differemment ne se comparent pas."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = _decor(tmp_path)
    fig, (a1, a2) = plt.subplots(1, 2)
    d.cadrer(a1)
    d.cadrer(a2)
    assert a1.get_xlim() == a2.get_xlim() == (-3.0, 3.0)
    assert a1.get_ylim() == a2.get_ylim() == (-2.0, 2.0)
    plt.close(fig)


# --------------------------------------------------------------------- #
# la planche d'enrichissement
# --------------------------------------------------------------------- #
def test_la_planche_dessine_sans_toucher_au_solveur(tmp_path):
    """Elle recoit TOUT deja calcule -- y compris le fond haute fidelite,
    qu'elle allait chercher elle-meme jusqu'au 26/08/2026."""
    d = _decor(tmp_path, n_grid=6)
    Z = np.zeros((6, 6))
    xt = np.array([[0.0, 0.0], [1.0, -1.0]])
    nom = figurer.planche_EFF(d, (0, 1, {}), xt, [np.array([0.5, 0.5])],
                              Z, Z + 1.0, Z - 0.5)
    assert nom.startswith("EFF_1points_") and nom.endswith(".png")
    assert os.path.isfile(os.path.join(str(tmp_path), nom))


def test_la_planche_sait_dessiner_un_surrogate_absent(tmp_path):
    """En HF pur il n'y a pas de metamodele : la planche doit sortir quand
    meme, sans troisieme vue."""
    d = _decor(tmp_path, n_grid=4)
    Z = np.zeros((4, 4))
    nom = figurer.planche_EFF(d, (0, 1, {}), None, [], Z, Z, None)
    assert os.path.isfile(os.path.join(str(tmp_path), nom))


def test_le_nom_de_la_planche_porte_le_nombre_de_points_ajoutes(tmp_path):
    """C'est ce qui rend une serie de planches lisible : on suit
    l'enrichissement point par point."""
    d = _decor(tmp_path, n_grid=4)
    Z = np.zeros((4, 4))
    ajoutes = [np.array([0.1, 0.1]), np.array([0.2, 0.2]), np.array([0.3, 0.3])]
    nom = figurer.planche_EFF(d, (0, 1, {}), np.zeros((2, 2)), ajoutes, Z, Z, Z)
    assert "_3points_" in nom


# --------------------------------------------------------------------- #
# le module reste sans solveur, et les scripts delegue
# --------------------------------------------------------------------- #
# Que `figurer.py` ne puisse pas atteindre le solveur est verifie par
# `test_93_figures_sans_solveur.py`, qui construit le graphe d'appel et en
# calcule la fermeture transitive. Un controle par recherche de texte serait
# FAUX ici : l'en-tete du module CITE le defaut historique
# (`print_3D_HF -> run_HF`) pour expliquer pourquoi ce module existe.


def test_la_planche_globale_dessine_une_ligne_par_etape(tmp_path):
    """Elle montre le RAISONNEMENT de l'algorithme, pas son resultat : le
    critere designe un point, l'ecart-type s'effondre autour a l'etape
    suivante, la frontiere se rapproche. Un enrichissement qui tourne en rond
    s'y lit d'un coup d'oeil."""
    d = _decor(tmp_path, n_grid=4)
    Z = np.zeros((4, 4))
    etapes = [dict(n_pts=5 + k, xt=np.zeros((5 + k, 2)),
                   xt_eff=[np.array([0.1 * i, 0.1 * i]) for i in range(k)],
                   Z_eff=Z, Z_sigma=Z, Z_g=Z)
              for k in range(3)]
    nom = figurer.planche_globale(d, (0, 1, {}), etapes)
    assert os.path.isfile(os.path.join(str(tmp_path), nom))


def test_la_planche_globale_supporte_une_seule_etape(tmp_path):
    """Sans point d'enrichissement, `plt.subplots` rend un tableau a une
    dimension : le reshape qui suit n'est pas cosmetique."""
    d = _decor(tmp_path, n_grid=4)
    Z = np.zeros((4, 4))
    etapes = [dict(n_pts=5, xt=np.zeros((5, 2)), xt_eff=[],
                   Z_eff=Z, Z_sigma=Z, Z_g=Z)]
    nom = figurer.planche_globale(d, (0, 1, {}), etapes)
    assert os.path.isfile(os.path.join(str(tmp_path), nom))


@pytest.mark.parametrize("script", SCRIPTS)
def test_le_script_delegue_le_dessin(script):
    src = open(os.path.join(_REPO, script), encoding="utf-8",
               errors="replace").read()
    assert "_figurer.Decor(" in src, "%s ne construit pas de decor" % script
    assert "_figurer.planche_EFF(" in src
    # ce qui reste dans l'etude, c'est le CALCUL sur la coupe, pas le trace.
    # Les trois champs eux-memes sont dans `_reliability/eff_ot` depuis le
    # 27/08 -- ils etaient ecrits deux fois par etude, a l'identique.
    assert "_eff_ot.champs_sur_coupe(" in src
    assert "_figurer.planche_globale(" in src
    for reste in ("plt.subplots(1, 3", "plt.subplots(n_steps"):
        assert reste not in src, (
            "%s dessine encore une planche lui-meme (%s)" % (script, reste))
