r"""Rejouer l'enrichissement, et ce que le metamodele dit de lui-meme.

DEUX EXTRACTIONS DU 28/08/2026
-------------------------------
`rejouer_l_enrichissement` -- le metamodele tel qu'il etait a chaque etape,
reajuste sur le plan tel qu'il etait alors. C'est ce qui permet de voir OU
l'enrichissement a place ses points et ce que chacun a change. Zero appel
solveur : les points sont deja calcules.

`Diagnostics` -- trois variables GLOBALES de chaque etude, ecrites par
l'ajustement et relues par les figures, le dump de reprise et la courbe de
convergence : les termes du chaos polynomial, l'erreur de validation
croisee, et l'historique des longueurs de correlation.
"""

import os
import sys

import numpy as np
import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
if os.path.join(_REPO, "_surrogate") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "_surrogate"))

_fit = pytest.importorskip("ajuster", reason="ajustement du metamodele")


# --------------------------------------------------------------------------- #
# REJOUER L'ENRICHISSEMENT                                                     #
# --------------------------------------------------------------------------- #
def _plan(n):
    xt = np.arange(2 * n, dtype=float).reshape(n, 2)
    return xt, np.arange(n, dtype=float).reshape(n, 1), np.ones((n, 2))


def _rejouer(n_total, n0, n_eff, tracer=None):
    xt, yt, ag = _plan(n_total)
    vus = []

    def reajuster(x, y, g):
        vus.append(len(x))
        return ("modele-%d" % len(x), "sigma-%d" % len(x))

    champs = lambda g_ot, s: (np.zeros((2, 2)), np.ones((2, 2)),
                              np.full((2, 2), 3.0))
    etapes = _fit.rejouer_l_enrichissement(
        xt, yt, ag, [np.zeros(2)] * n_eff, n0, reajuster=reajuster,
        champs=champs, tracer=tracer or (lambda _m: None))
    return etapes, vus


def test_une_etape_par_point_ajoute_PLUS_l_etat_initial():
    """L'etat initial compte : c'est la reference contre laquelle on lit ce
    que l'enrichissement a apporte."""
    etapes, _ = _rejouer(n_total=13, n0=5, n_eff=8)
    assert len(etapes) == 9
    assert [e["n_pts"] for e in etapes] == [5, 6, 7, 8, 9, 10, 11, 12, 13]


def test_chaque_etape_reajuste_sur_le_plan_TEL_QU_IL_ETAIT():
    """Pas sur le plan final tronque a l'affichage : le metamodele de
    l'etape 3 doit etre celui qu'on avait a l'etape 3."""
    _, vus = _rejouer(n_total=13, n0=5, n_eff=8)
    assert vus == [5, 6, 7, 8, 9, 10, 11, 12, 13]


def test_un_enrichissement_vide_donne_le_seul_etat_initial():
    etapes, vus = _rejouer(n_total=5, n0=5, n_eff=0)
    assert len(etapes) == 1 and etapes[0]["n_pts"] == 5
    assert vus == [5]


def test_chaque_etape_porte_le_plan_et_les_points_ajoutes_a_ce_moment():
    etapes, _ = _rejouer(n_total=8, n0=5, n_eff=3)
    assert len(etapes[0]["xt"]) == 5 and len(etapes[0]["xt_eff"]) == 0
    assert len(etapes[2]["xt"]) == 7 and len(etapes[2]["xt_eff"]) == 2
    assert len(etapes[-1]["xt"]) == 8 and len(etapes[-1]["xt_eff"]) == 3


def test_le_journal_annonce_chaque_etape():
    j = []
    _rejouer(n_total=7, n0=5, n_eff=2, tracer=j.append)
    assert j == ["  [GLOBAL PLANCHE] step 0/2 (N=5) OK",
                 "  [GLOBAL PLANCHE] step 1/2 (N=6) OK",
                 "  [GLOBAL PLANCHE] step 2/2 (N=7) OK"]


def test_les_trois_champs_de_chaque_etape_viennent_de_SON_metamodele():
    """Un champ calcule sur le metamodele final, affiche a l'etape 2,
    montrerait une convergence qui n'a pas eu lieu."""
    xt, yt, ag = _plan(7)
    modeles = []

    def champs(g_ot, sigma_func):
        modeles.append(g_ot)
        return np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))

    _fit.rejouer_l_enrichissement(
        xt, yt, ag, [np.zeros(2)] * 2, 5,
        reajuster=lambda x, y, g: ("modele-%d" % len(x), None),
        champs=champs, tracer=lambda _m: None)
    assert modeles == ["modele-5", "modele-6", "modele-7"]


# --------------------------------------------------------------------------- #
# LES DIAGNOSTICS DU METAMODELE                                                #
# --------------------------------------------------------------------------- #
def test_un_diagnostic_neuf_ne_pretend_rien():
    d = _fit.Diagnostics()
    assert d.pce_label == "" and d.loo is None and d.theta == []
    assert d.sous_titre() == ""


def test_un_ajustement_avec_chaos_range_ses_trois_mesures():
    d = _fit.Diagnostics()
    d.enregistrer({"pce_label": "-0.21*1", "loo": 1.9e-2,
                   "theta": [17.7, 41.9]})
    assert d.pce_label == "-0.21*1"
    assert d.loo == pytest.approx(1.9e-2)
    assert d.theta == [[17.7, 41.9]]


def test_un_modele_SANS_chaos_ne_detruit_pas_le_diagnostic_precedent():
    """`pce_label is None` designe un krigeage pur. Ecraser `loo` et empiler
    un `theta` absent effacerait ce que le dernier ajustement informatif
    avait rendu."""
    d = _fit.Diagnostics()
    d.enregistrer({"pce_label": "-0.21*1", "loo": 1.9e-2, "theta": [1.0, 2.0]})
    d.enregistrer({"pce_label": None, "loo": None, "theta": None})
    assert d.pce_label == "-0.21*1"
    assert d.loo == pytest.approx(1.9e-2)
    assert d.theta == [[1.0, 2.0]]


def test_theta_est_CUMULATIF_les_deux_autres_sont_des_instantanes():
    """L'asymetrie est celle d'origine : on ne trace que le dernier chaos,
    mais toute la trajectoire des longueurs de correlation -- c'est elle qui
    dit si le krigeage se stabilise ou court apres le dernier point."""
    d = _fit.Diagnostics()
    for k in range(3):
        d.enregistrer({"pce_label": "terme-%d" % k, "loo": float(k),
                       "theta": [float(k), float(k)]})
    assert d.pce_label == "terme-2" and d.loo == 2.0
    assert d.theta == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]


# --------------------------------------------------------------------------- #
# LE SOUS-TITRE DES PLANCHES                                                   #
# --------------------------------------------------------------------------- #
def test_le_sous_titre_montre_le_DERNIER_theta():
    d = _fit.Diagnostics()
    d.enregistrer({"pce_label": None, "loo": None, "theta": None})
    d.theta = [[1.0, 2.0], [17.716, 41.891]]
    assert d.sous_titre() == "  theta=[17.716, 41.891]"


def test_l_ordre_du_sous_titre_est_celui_d_origine():
    """Longueurs, erreur, coupe, puis le chaos sur sa propre ligne."""
    d = _fit.Diagnostics()
    d.enregistrer({"pce_label": "-0.21*1", "loo": 1.912e-2,
                   "theta": [17.716, 41.891]})
    assert d.sous_titre("  fc=0.0") == (
        "  theta=[17.716, 41.891]  LOO=1.912e-02  fc=0.0\n-0.21*1")


def test_une_erreur_absente_ne_s_affiche_pas():
    """Un LOO manquant vaut `None`, pas zero : afficher `LOO=0.000e+00`
    annoncerait un metamodele parfait."""
    d = _fit.Diagnostics()
    d.theta = [[1.0, 2.0]]
    assert "LOO" not in d.sous_titre()


def test_un_LOO_nul_s_affiche_lui():
    """Zero est une valeur, pas une absence."""
    d = _fit.Diagnostics()
    d.enregistrer({"pce_label": "x", "loo": 0.0, "theta": [1.0]})
    assert "LOO=0.000e+00" in d.sous_titre()


# --------------------------------------------------------------------------- #
# CE QUI NE DOIT PLUS ETRE DANS LES ETUDES                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_les_etudes_n_ont_plus_de_globales_de_diagnostic(script):
    import io
    s = io.open(os.path.join(_REPO, script), encoding="utf-8",
                errors="replace").read()
    for interdit in ("_gepck_pce_label", "_gepck_loo", "_eff_history_theta",
                     "[GLOBAL PLANCHE] step"):
        assert interdit not in s, (
            "%s : un diagnostic du metamodele est revenu dans l'etude (%r)."
            % (script, interdit))
    assert "_fit.Diagnostics()" in s
    assert "_fit.rejouer_l_enrichissement(" in s
