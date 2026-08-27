r"""Le plan d'experiences : ou depenser les premiers appels solveur.

C'est le seul moment ou l'on choisit des points SANS rien savoir de l'etat
limite. Tout le reste part de la, et un plan mal reparti se paie ensuite en
points d'enrichissement -- c'est-a-dire en heures.

CE QUE CES TESTS PROTEGENT
---------------------------
1. **Le domaine de tirage suit les bornes de l'etude.** Il etait CODE EN DUR
   a +/- 7,5 dans la flexion pure et lu dans `eff_bounds` sur le Moulin Blanc,
   dans deux `build_DOE` par ailleurs identiques au caractere pres. Borner le
   domaine -- ce qu'on fait quand le solveur meurt sur des points extremes --
   n'avait donc d'effet que d'un cote.
2. **Un point sans gradient ne casse plus le plan.** Digital Structure rend
   parfois `Sensitivity = {fy1: None, fy2: None}`. Le 26/08/2026, cela a fait
   partir un plan en `TypeError` APRES cinq appels au solveur.
3. **Aucun gradient fabrique en silence.** Conserver un tel point est
   possible, mais le journal doit dire que son gradient est INVENTE : un zero
   affirme que l'etat limite est plat la ou l'on ne sait rien.
"""

import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _d in (os.path.join(_REPO, "_doe"),):
    if _d not in sys.path:
        sys.path.insert(0, _d)

ot = pytest.importorskip("openturns")
np = pytest.importorskip("numpy")

import plan as _plan                                  # noqa: E402

PARAMS = ["fc", "fy"]


def _dist():
    return ot.JointDistribution([ot.Normal(40.0, 4.0), ot.Normal(500.0, 30.0)])


# --------------------------------------------------------------------- #
# le domaine de tirage
# --------------------------------------------------------------------- #
def test_le_plan_reste_dans_les_bornes_demandees():
    """LE test de la cinquieme divergence : borner le domaine doit border le
    tirage. Ici les bornes sont volontairement asymetriques et differentes
    d'une variable a l'autre -- un tirage code en dur ne pourrait pas passer."""
    U, _X, xt = _plan.tirer_plan_lhs(_dist(), 12, [-2.0, -5.0], [1.0, 6.0])
    assert xt.shape == (12, 2)
    assert xt[:, 0].min() >= -2.0 and xt[:, 0].max() <= 1.0
    assert xt[:, 1].min() >= -5.0 and xt[:, 1].max() <= 6.0
    assert U.getSize() == 12


def test_le_plan_rend_les_memes_points_en_physique_et_en_norme():
    """`X_doe` sert au solveur, `xt` au metamodele : ils doivent designer les
    MEMES points, sinon le metamodele ajuste une surface decalee."""
    dist = _dist()
    U, X, xt = _plan.tirer_plan_lhs(dist, 8, [-2.0, -2.0], [2.0, 2.0])
    T_inv = dist.getInverseIsoProbabilisticTransformation()
    for i in range(8):
        attendu = T_inv(ot.Point(list(xt[i])))
        assert list(X[i]) == pytest.approx(list(attendu), abs=1e-12)


def test_les_points_de_depart_suivent_les_memes_bornes():
    """Un point de depart hors du domaine ou l'etat limite a ete evalue
    partirait explorer une extrapolation du metamodele."""
    sp = _plan.tirer_points_de_depart(20, [-1.5, -3.0], [1.5, 3.0])
    assert sp.shape == (20, 2)
    assert sp[:, 0].min() >= -1.5 and sp[:, 0].max() <= 1.5
    assert sp[:, 1].min() >= -3.0 and sp[:, 1].max() <= 3.0


def test_le_plan_recopiable_garde_seize_decimales():
    """Un plan retranscrit a moins de decimales n'est PAS le meme plan, et la
    comparaison de deux runs perd son sens."""
    lignes = []
    U, _X, _xt = _plan.tirer_plan_lhs(_dist(), 3, [-1.0, -1.0], [1.0, 1.0])
    _plan.tracer_plan(U, tracer=lignes.append)
    assert lignes[0].startswith("U_doe_fixed = ot.Sample([")
    corps = [l for l in lignes if l.strip().startswith("[")]
    assert len(corps) == 3
    for l in corps:
        for morceau in l.strip().strip("[],").split(","):
            assert len(morceau.strip().split(".")[1].rstrip(",]")) == 16, (
                "decimales perdues : %r" % l)


# --------------------------------------------------------------------- #
# les points sans gradient
# --------------------------------------------------------------------- #
def _SOL(gradients):
    """Un plan factice : `gradients` dit, point par point, s'il en a un."""
    return [{"g": -0.1 * i,
             "dg_fc": (0.5 if ok else None),
             "dg_fy": (0.25 if ok else None)}
            for i, ok in enumerate(gradients)]


def test_un_plan_complet_passe_entier():
    U = np.zeros((3, 2))
    assert _plan.points_avec_gradient(_SOL([True] * 3), PARAMS, U, True,
                                      tracer=lambda _m: None) == [0, 1, 2]


def test_un_point_sans_gradient_est_ECARTE_si_on_le_demande():
    messages = []
    U = np.zeros((4, 2))
    gardes = _plan.points_avec_gradient(_SOL([True, False, True, True]),
                                        PARAMS, U, True, tracer=messages.append)
    assert gardes == [0, 2, 3]
    texte = "\n".join(messages)
    assert "SANS GRADIENT" in texte and "ECARTE" in texte
    assert "le plan passe de 4 a 3" in texte, (
        "le journal doit dire de combien le plan retrecit")


def test_un_point_sans_gradient_CONSERVE_annonce_que_son_gradient_est_invente():
    """C'est la phrase qui compte : sans elle, un zero fabrique passe pour une
    mesure, et le metamodele ajuste une surface plate la ou l'on ne sait
    rien."""
    messages = []
    U = np.zeros((3, 2))
    gardes = _plan.points_avec_gradient(_SOL([True, False, True]),
                                        PARAMS, U, False, tracer=messages.append)
    assert gardes == [0, 1, 2], "sans exclusion, aucun point n'est retire"
    assert "FABRIQUE a 0" in "\n".join(messages)


def test_un_plan_sans_aucun_gradient_LEVE_au_lieu_de_rendre_du_vide():
    """Un plan vide ne se voit que trois fonctions plus loin, sous forme
    d'erreur de forme. Le dire tot nomme la cause probable."""
    U = np.zeros((2, 2))
    with pytest.raises(RuntimeError) as err:
        _plan.points_avec_gradient(_SOL([False, False]), PARAMS, U, True,
                                   tracer=lambda _m: None)
    assert "regions de sensibilite" in str(err.value), (
        "le message doit orienter vers la cause reelle : les regions du "
        "modele ne correspondent pas aux variables")


def test_un_plan_muet_reste_muet_quand_tout_va_bien():
    """Pas de bruit dans le journal quand il n'y a rien a signaler."""
    messages = []
    U = np.zeros((3, 2))
    _plan.points_avec_gradient(_SOL([True] * 3), PARAMS, U, True,
                               tracer=messages.append)
    assert messages == []


# --------------------------------------------------------------------- #
# l'augmentation de Taylor
# --------------------------------------------------------------------- #
def test_sans_eps_le_plan_est_rendu_intact():
    xt = np.array([[0.0, 0.0], [1.0, 1.0]])
    yt = np.array([[1.0], [2.0]])
    ag = np.array([[0.5, 0.25], [0.1, 0.2]])
    x2, y2, a2 = _plan.augmenter_par_taylor(xt, yt, ag, 0.0, 2,
                                            tracer=lambda _m: None)
    assert x2 is xt and y2 is yt and a2 is ag


def test_chaque_point_reel_donne_un_virtuel_par_direction():
    xt = np.array([[0.0, 0.0], [1.0, 1.0]])
    yt = np.array([[1.0], [2.0]])
    ag = np.array([[0.5, 0.25], [0.1, 0.2]])
    x2, y2, a2 = _plan.augmenter_par_taylor(xt, yt, ag, 0.01, 2,
                                            tracer=lambda _m: None)
    assert len(x2) == 2 * (1 + 2) == 6
    # premier virtuel : point 0 decale de eps selon u1
    assert list(x2[2]) == pytest.approx([0.01, 0.0])
    assert y2[2, 0] == pytest.approx(1.0 + 0.01 * 0.5), (
        "la valeur virtuelle est un developpement au PREMIER ORDRE")
    assert list(a2[2]) == pytest.approx([0.5, 0.25]), (
        "le point virtuel herite du gradient de son point reel")


def test_l_augmentation_dit_combien_de_points_sont_inventes():
    messages = []
    xt = np.array([[0.0, 0.0]])
    _plan.augmenter_par_taylor(xt, np.array([[1.0]]), np.array([[0.5, 0.25]]),
                               0.01, 2, tracer=messages.append)
    assert "1 HF + 2 virtuels = 3 pts" in "\n".join(messages), (
        "un plan de 3 points dont 2 inventes ne doit pas se lire comme un "
        "plan de 3 points mesures")


# --------------------------------------------------------------------- #
# le module reste sans solveur
# --------------------------------------------------------------------- #
def test_le_plan_ne_touche_jamais_le_solveur():
    """Il TIRE, il TRIE, il AUGMENTE. L'evaluation est ailleurs, en un seul
    exemplaire."""
    src = open(os.path.join(_REPO, "_doe", "plan.py"), encoding="utf-8").read()
    for interdit in ("solveur.evaluer", "run_HF", "Evaluateur",
                     "digital_structure"):
        assert interdit not in src, "plan.py mentionne %r" % interdit
