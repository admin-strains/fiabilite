r"""Un fond de figure doit etre trace la ou ses valeurs ont ete calculees.

LE DEFAUT -- D'ORIGINE, CORRIGE LE 27/08/2026
-----------------------------------------------
`fond_hf_pour_figures` rendait `(Z, UX, UY)` ou `Z` venait de la grille haute
fidelite -- calculee sur `u1_min..u2_max` -- tandis que `UX, UY` etaient
construits sur le CADRE DES FIGURES. Les deux coincident en flexion pure
(`cadre_figures = "grille"`), pas sur le Moulin Blanc, ou le cadre vaut
`eff_bounds +/- 1` :

    grille HF Moulin Blanc : -6 .. +6
    cadre des figures      : -7 .. +7

Les trois figures qui affichent ce fond -- `planche_EFF`, `planche_globale`,
`visu_FORM` -- appellent toutes `ax.contour(UX_hf, UY_hf, Z_vrai, levels=[0])`.
La courbe de reference « g = 0 HF » etait donc DILATEE de 7/6, soit 17 %,
sur chaque figure du Moulin Blanc. C'est la courbe contre laquelle on juge
le metamodele.

CE QUI PROUVE QUE C'ETAIT UN DEFAUT ET NON UNE INTENTION
---------------------------------------------------------
Le fichier d'origine se contredisait : `print_3D_HF` tracait ce MEME `Z` sur
`u1_min..u2_max`, donc au bon endroit. Trois sites sur quatre etaient faux.

CE QUE LE CADRE RESTE
----------------------
Il fixe les LIMITES des axes -- voir a `Decor`. Voir la grille dans une vue
un peu plus large est legitime ; y etirer ses valeurs ne l'est pas.

Aucun nombre du calcul ne bouge : ni beta, ni Pf, ni le plan, ni le
metamodele. Seule la courbe rejoint ses propres coordonnees.
"""

import io
import os
import sys

import numpy as np
import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_etapes"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import grille as _grille                               # noqa: E402
import figurer as _figurer                             # noqa: E402


def _grille_de(bornes, cote=5, tmp=None):
    """Une grille qui n'appellera jamais rien : ces tests ne portent que sur
    ses COORDONNEES."""
    return _grille.Grille(evaluer=lambda u: (0.0, [0.0, 0.0], [0.0, 0.0]),
                          n_var=2, cote=cote, bornes=bornes,
                          fichier_cache=None, fichier_cache_complet=None,
                          tracer=lambda m: None)


# --------------------------------------------------------------------------- #
# LE MAILLAGE ET LES VALEURS VIENNENT DU MEME OBJET                            #
# --------------------------------------------------------------------------- #
def test_le_maillage_de_la_grille_couvre_exactement_ses_bornes():
    g = _grille_de((-6.0, 6.0, -6.0, 6.0), cote=5)
    UX, UY = g.maillage_2d()
    assert UX.shape == (5, 5) and UY.shape == (5, 5)
    assert UX.min() == -6.0 and UX.max() == 6.0
    assert UY.min() == -6.0 and UY.max() == 6.0


def test_des_bornes_asymetriques_sont_respectees_axe_par_axe():
    g = _grille_de((-6.0, 2.0, -1.0, 7.0), cote=4)
    UX, UY = g.maillage_2d()
    assert (UX.min(), UX.max()) == (-6.0, 2.0)
    assert (UY.min(), UY.max()) == (-1.0, 7.0)


def test_le_maillage_a_AUTANT_de_points_que_Z():
    """`ax.contour(UX, UY, Z)` exige des formes compatibles. Un maillage a
    `n_grid` points pour un `Z` a `n_grid_hf` leverait -- ou pire, passerait
    silencieusement si les deux coincident."""
    for cote in (3, 7, 15):
        UX, UY = _grille_de((-6.0, 6.0, -6.0, 6.0), cote=cote).maillage_2d()
        assert UX.shape == UY.shape == (cote, cote)


# --------------------------------------------------------------------------- #
# CE QUE LE DEFAUT VALAIT, EN CHIFFRES                                         #
# --------------------------------------------------------------------------- #
def test_en_flexion_pure_le_cadre_et_la_grille_coincidaient():
    """C'est pourquoi le defaut ne se voyait pas : l'etude de reference le
    masquait. `cadre_figures = "grille"` rend exactement les bornes."""
    bornes = (-7.5, 7.5, -7.5, 7.5)
    cadre = _figurer.cadre_des_figures("grille", bornes, [-7.5, -7.5],
                                       [7.5, 7.5], 1.0)
    assert cadre == bornes


def test_sur_le_moulin_blanc_ils_ne_coincidaient_pas():
    """Le reglage reel : `eff_bounds = +/-6`, `cadre_marge = 1`."""
    bornes = (-6.0, 6.0, -6.0, 6.0)
    cadre = _figurer.cadre_des_figures("bornes_elargies", bornes,
                                       [-6.0, -6.0], [6.0, 6.0], 1.0)
    assert cadre == (-7.0, 7.0, -7.0, 7.0)
    # la dilatation subie par la courbe de reference
    assert (cadre[1] - cadre[0]) / (bornes[1] - bornes[0]) == pytest.approx(
        7.0 / 6.0, rel=1e-12)


def test_un_contour_trace_sur_le_cadre_deplace_ses_points():
    """La consequence, sur des nombres. Un point de l'etat limite calcule en
    u = -6 se retrouvait dessine en u = -7 ; un point calcule en u = 0 ne
    bougeait pas. La courbe n'etait donc pas translatee mais ETIREE -- ce qui
    la rend difficile a reconnaitre comme fausse a l'oeil."""
    cote = 7
    vrai = np.linspace(-6.0, 6.0, cote)
    faux = np.linspace(-7.0, 7.0, cote)
    assert vrai[0] == -6.0 and faux[0] == -7.0
    assert vrai[cote // 2] == faux[cote // 2] == 0.0
    ecart = np.abs(faux - vrai)
    assert ecart.max() == pytest.approx(1.0)
    assert ecart[cote // 2] == 0.0


# --------------------------------------------------------------------------- #
# LE DEFAUT NE PEUT PLUS REVENIR                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_aucun_fond_de_figure_ne_prend_son_maillage_dans_le_cadre(script):
    """Le cadre fixe les limites des axes -- il ne definit les coordonnees
    d'aucune valeur calculee."""
    s = io.open(os.path.join(_REPO, script), encoding="utf-8").read()
    for interdit in ("np.linspace(_CX0", "np.linspace(_CY0",
                     "eff_bounds_min[0] - 1", "eff_bounds_max[0] + 1"):
        assert interdit not in s, (
            "%s : un maillage de figure est de nouveau construit sur le "
            "cadre (%r), alors que les valeurs viennent de la grille."
            % (script, interdit))
    # Le fond vient desormais de la grille elle-meme : c'est LA qu'il faut
    # verifier que le maillage et les valeurs sortent du meme objet.
    assert "_GRILLE.fond_de_figure(" in s
    src_mod = io.open(os.path.join(_REPO, "_etapes", "grille.py"),
                      encoding="utf-8").read()
    i = src_mod.index("def fond_de_figure")
    corps = src_mod[i:i + 2500]
    assert "UX, UY = self.maillage_2d()" in corps, (
        "`fond_de_figure` ne prend plus son maillage dans la grille qui "
        "calcule Z.")


@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_le_cadre_ne_sert_plus_qu_au_decor(script):
    """`_CX0.._CY1` ne doivent plus apparaitre que la ou ils ont un sens :
    leur calcul, et la construction du decor qui borne les axes."""
    s = io.open(os.path.join(_REPO, script), encoding="utf-8").read()
    lignes = [l.strip() for l in s.splitlines()
              if "_CX0" in l or "_CX1" in l or "_CY0" in l or "_CY1" in l]
    assert len(lignes) == 2, lignes
    assert lignes[0].startswith("_CX0, _CX1, _CY0, _CY1 = _figurer.cadre_des_figures(")
    assert "_figurer.Decor(" in lignes[1]
