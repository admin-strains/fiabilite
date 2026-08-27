r"""Choisir plusieurs points d'enrichissement avant d'en avoir calcule un.

LE PROBLEME QUE KRIGING BELIEVER RESOUT
-----------------------------------------
Un appel solveur coute 466 s sur le Moulin Blanc, et six workers attendent.
Mais le critere EFF ne designe qu'UN point : le re-maximiser sans rien
changer redonnerait exactement le meme. Kriging Believer verse le point
retenu au plan avec, pour valeur, la PREDICTION du metamodele -- il le
croit. La variance s'effondre autour de ce point, et le critere designe
alors ailleurs.

CE QUE CES TESTS METTENT PAR ECRIT
-----------------------------------
1. A `n_batch = 1`, aucun reajustement n'a lieu -- pas un seul.
2. Le reajustement se fait a `theta` FIXE (`fixed_fm`), repris du metamodele
   COURANT. Une observation fictive n'est pas une mesure : elle n'a pas a
   redefinir la portee de correlation, et reoptimiser `theta` a chaque point
   du batch couterait plus que les appels solveur qu'on parallelise.
3. La valeur rendue est celle du VRAI metamodele, au premier point. Les
   suivantes viennent de metamodeles fictifs et ne peuvent pas juger
   l'arret.
4. Le gradient fictif est nul quand le metamodele n'exploite pas les
   gradients. Ce zero AFFIRME que l'etat limite est plat -- il n'est pas
   neutre. Il est inoffensif seulement parce que PCK et KRG ne le lisent
   pas ; GEPCK, lui, recoit le gradient du metamodele.
5. La valeur imputee est bien `g_ot(u)`, pas zero ni la moyenne.
"""

import os
import sys

import numpy as np
import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_reliability"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# OpenTURNS conditionne tout ce fichier : sans lui, il n'y a ni critere ni
# optimiseur a exercer. Le module entier se saute, il n'echoue pas.
ot = pytest.importorskip("openturns")
_eff_ot = pytest.importorskip("eff_ot", reason="emballage OpenTURNS du critere")


BORNES_MIN, BORNES_MAX = [-5.0, -5.0], [5.0, 5.0]


class _Surrogate:
    """Un metamodele jouet : `g` affine, `sigma` en cloche autour d'un
    centre qui se deplace a chaque reajustement."""

    def __init__(self, centre=(2.0, 2.0), fm="fm-initial"):
        self.centre = np.array(centre, float)
        self.fm = fm
        self.appels_gradient = 0

    # --- la face `g_ot` ---
    def __call__(self, u):
        u = np.array(u, float)
        return [float(1.0 + u[0] + u[1])]

    def gradient(self, u):
        self.appels_gradient += 1
        return np.array([[1.0], [1.0]])

    # --- la face `sigma_func` (methode liee, donc `__self__.fm` lisible) ---
    def sigma(self, u):
        d = np.array(u, float) - self.centre
        return float(np.exp(-0.5 * float(d @ d)))


def _reajusteur(centres):
    """Rend un `reajuster` qui deplace le centre a chaque appel, et retient
    ce qu'on lui a demande."""
    trace = {"fm": [], "n": 0, "tailles": []}

    def reajuster(g, s, xt, yt, ag, fixed_fm=None):
        trace["fm"].append(fixed_fm)
        trace["tailles"].append((len(xt), len(yt), len(ag)))
        neuf = _Surrogate(centre=centres[trace["n"] % len(centres)],
                          fm="fm-refit-%d" % trace["n"])
        trace["n"] += 1
        return neuf, neuf.sigma, xt, yt, ag

    return reajuster, trace


def _plan(n=4):
    rng = np.random.default_rng(3)
    xt = rng.normal(0, 1, (n, 2))
    yt = np.array([[1.0 + p[0] + p[1]] for p in xt])
    return xt, yt, np.ones((n, 2))


class _Journal(list):
    def __call__(self, message):
        self.append(message)


# --------------------------------------------------------------------------- #
# LA MAXIMISATION SEULE                                                        #
# --------------------------------------------------------------------------- #
def test_le_critere_est_maximise_globalement_pas_localement():
    """Le critere EFF est multimodal : il vaut zero loin de l'etat limite
    comme dans les zones deja bien connues. Une descente locale depuis
    l'origine s'arreterait sur le premier plateau venu. `GN_DIRECT` est
    global et sans derivees.

    Ici le maximum est place LOIN de l'origine ; un optimiseur local le
    manquerait.
    """
    cible = np.array([3.5, -4.0])

    class _Pic(ot.OpenTURNSPythonFunction):
        def __init__(self):
            super().__init__(2, 1)

        def _exec(self, u):
            d = np.array(u) - cible
            return [float(np.exp(-0.5 * float(d @ d)))]

    u, valeur = _eff_ot.maximiser_EFF(ot.Function(_Pic()), BORNES_MIN,
                                      BORNES_MAX, 2, 3000)
    assert np.allclose(u, cible, atol=0.3), u
    assert valeur > 0.9


def test_la_valeur_rendue_est_celle_du_critere_au_point_rendu():
    class _Plan(ot.OpenTURNSPythonFunction):
        def __init__(self):
            super().__init__(2, 1)

        def _exec(self, u):
            return [float(u[0] + u[1])]

    f = ot.Function(_Plan())
    u, valeur = _eff_ot.maximiser_EFF(f, BORNES_MIN, BORNES_MAX, 2, 2000)
    assert valeur == pytest.approx(f(ot.Point(u.tolist()))[0])


# --------------------------------------------------------------------------- #
# LE BATCH                                                                     #
# --------------------------------------------------------------------------- #
def test_un_batch_de_un_ne_reajuste_rien():
    """A `n_batch = 1` il n'y a rien a croire : le point est calcule pour de
    vrai juste apres. Un seul reajustement de trop couterait une
    optimisation de metamodele par tour d'enrichissement."""
    s = _Surrogate()
    reajuster, trace = _reajusteur([(0.0, 0.0)])
    xt, yt, ag = _plan()
    batch, valeur = _eff_ot.batch_kriging_believer(
        s, s.sigma, xt, yt, ag, n_batch=1, bornes_min=BORNES_MIN,
        bornes_max=BORNES_MAX, n_var=2, n_appels=800, epsilon_factor=2.0,
        reajuster=reajuster, tracer=_Journal())
    assert len(batch) == 1
    assert trace["n"] == 0, "un batch de un a reajuste le metamodele"


def test_un_batch_de_trois_donne_trois_points_et_deux_reajustements():
    s = _Surrogate()
    reajuster, trace = _reajusteur([(-3.0, 1.0), (1.0, -3.0)])
    xt, yt, ag = _plan()
    j = _Journal()
    batch, _ = _eff_ot.batch_kriging_believer(
        s, s.sigma, xt, yt, ag, n_batch=3, bornes_min=BORNES_MIN,
        bornes_max=BORNES_MAX, n_var=2, n_appels=800, epsilon_factor=2.0,
        reajuster=reajuster, tracer=j)
    assert len(batch) == 3
    assert trace["n"] == 2, "un point impute, un reajustement"
    assert [m.split("]")[0] + "]" for m in j] == ["  [KB 2/3]", "  [KB 3/3]"]


def test_chaque_point_impute_agrandit_le_plan_d_une_ligne():
    s = _Surrogate()
    reajuster, trace = _reajusteur([(-3.0, 1.0), (1.0, -3.0)])
    xt, yt, ag = _plan(n=4)
    _eff_ot.batch_kriging_believer(
        s, s.sigma, xt, yt, ag, n_batch=3, bornes_min=BORNES_MIN,
        bornes_max=BORNES_MAX, n_var=2, n_appels=800, epsilon_factor=2.0,
        reajuster=reajuster, tracer=_Journal())
    assert trace["tailles"] == [(5, 5, 5), (6, 6, 6)]


def test_le_plan_REEL_de_l_appelant_n_est_jamais_touche():
    """Les observations fictives ne doivent pas survivre au batch : elles
    seraient prises pour des mesures au tour suivant."""
    s = _Surrogate()
    reajuster, _ = _reajusteur([(-3.0, 1.0), (1.0, -3.0)])
    xt, yt, ag = _plan(n=4)
    copie = (xt.copy(), yt.copy(), ag.copy())
    _eff_ot.batch_kriging_believer(
        s, s.sigma, xt, yt, ag, n_batch=3, bornes_min=BORNES_MIN,
        bornes_max=BORNES_MAX, n_var=2, n_appels=800, epsilon_factor=2.0,
        reajuster=reajuster, tracer=_Journal())
    assert np.array_equal(xt, copie[0])
    assert np.array_equal(yt, copie[1])
    assert np.array_equal(ag, copie[2])


def test_le_reajustement_impose_le_theta_du_metamodele_COURANT():
    """`fixed_fm` vient du metamodele reel, et reste le meme a tous les
    tours : une observation fictive n'a pas a redefinir la portee de
    correlation, et reoptimiser `theta` couterait plus que les appels
    solveur qu'on cherche a paralleliser."""
    s = _Surrogate(fm="fm-du-vrai-metamodele")
    reajuster, trace = _reajusteur([(-3.0, 1.0), (1.0, -3.0)])
    xt, yt, ag = _plan()
    _eff_ot.batch_kriging_believer(
        s, s.sigma, xt, yt, ag, n_batch=3, bornes_min=BORNES_MIN,
        bornes_max=BORNES_MAX, n_var=2, n_appels=800, epsilon_factor=2.0,
        reajuster=reajuster, tracer=_Journal())
    assert trace["fm"] == ["fm-du-vrai-metamodele"] * 2


def test_la_valeur_rendue_vient_du_VRAI_metamodele():
    """C'est elle qui juge l'arret. Les valeurs des points suivants viennent
    de metamodeles fictifs -- les rendre reviendrait a laisser une croyance
    decider de la convergence."""
    s = _Surrogate()
    reajuster, _ = _reajusteur([(-3.0, 1.0), (1.0, -3.0)])
    xt, yt, ag = _plan()

    seule, valeur_seule = _eff_ot.batch_kriging_believer(
        s, s.sigma, xt, yt, ag, n_batch=1, bornes_min=BORNES_MIN,
        bornes_max=BORNES_MAX, n_var=2, n_appels=800, epsilon_factor=2.0,
        reajuster=reajuster, tracer=_Journal())
    batch, valeur_batch = _eff_ot.batch_kriging_believer(
        s, s.sigma, xt, yt, ag, n_batch=3, bornes_min=BORNES_MIN,
        bornes_max=BORNES_MAX, n_var=2, n_appels=800, epsilon_factor=2.0,
        reajuster=reajuster, tracer=_Journal())
    assert np.allclose(batch[0], seule[0])
    assert valeur_batch == pytest.approx(valeur_seule)


# --------------------------------------------------------------------------- #
# LE GRADIENT FICTIF                                                           #
# --------------------------------------------------------------------------- #
def test_sans_gradients_le_gradient_fictif_est_nul():
    """Constate, pas approuve : un zero AFFIRME que l'etat limite est plat.
    Il n'est inoffensif que parce que PCK et KRG ne lisent pas les
    gradients."""
    s = _Surrogate()
    vus = []

    def reajuster(g, s_, xt, yt, ag, fixed_fm=None):
        vus.append(np.array(ag[-1]))
        neuf = _Surrogate(centre=(-3.0, 1.0))
        return neuf, neuf.sigma, xt, yt, ag

    xt, yt, ag = _plan()
    _eff_ot.batch_kriging_believer(
        s, s.sigma, xt, yt, ag, n_batch=2, bornes_min=BORNES_MIN,
        bornes_max=BORNES_MAX, n_var=2, n_appels=800, epsilon_factor=2.0,
        reajuster=reajuster, gradient_du_surrogate=False, tracer=_Journal())
    assert vus[0].tolist() == [0.0, 0.0]
    assert s.appels_gradient == 0


def test_avec_gradients_c_est_celui_du_metamodele_qui_est_impute():
    s = _Surrogate()
    vus = []

    def reajuster(g, s_, xt, yt, ag, fixed_fm=None):
        vus.append(np.array(ag[-1]))
        neuf = _Surrogate(centre=(-3.0, 1.0))
        return neuf, neuf.sigma, xt, yt, ag

    xt, yt, ag = _plan()
    _eff_ot.batch_kriging_believer(
        s, s.sigma, xt, yt, ag, n_batch=2, bornes_min=BORNES_MIN,
        bornes_max=BORNES_MAX, n_var=2, n_appels=800, epsilon_factor=2.0,
        reajuster=reajuster, gradient_du_surrogate=True, tracer=_Journal())
    assert vus[0].tolist() == [1.0, 1.0]
    # DEFAUT CONSTATE, NON CORRIGE : le gradient complet est recalcule UNE
    # FOIS PAR DIMENSION pour n'en lire qu'une composante a chaque fois. Le
    # cout croit donc en n_var^2. Repris verbatim de l'original -- corriger
    # changerait le nombre d'appels au metamodele, ce qui n'a pas sa place
    # dans une extraction. Le chiffre est fige ici pour que la correction,
    # quand elle viendra, se voie.
    assert s.appels_gradient == 2, "n_var appels pour n_var composantes"


def test_la_valeur_imputee_est_la_prediction_du_metamodele():
    """« Believer » : le point est verse au plan avec ce que le metamodele
    predit, pas avec zero ni la moyenne des observations."""
    s = _Surrogate()
    vus = []

    def reajuster(g, s_, xt, yt, ag, fixed_fm=None):
        vus.append((np.array(xt[-1]), float(yt[-1][0])))
        neuf = _Surrogate(centre=(-3.0, 1.0))
        return neuf, neuf.sigma, xt, yt, ag

    xt, yt, ag = _plan()
    batch, _ = _eff_ot.batch_kriging_believer(
        s, s.sigma, xt, yt, ag, n_batch=2, bornes_min=BORNES_MIN,
        bornes_max=BORNES_MAX, n_var=2, n_appels=800, epsilon_factor=2.0,
        reajuster=reajuster, tracer=_Journal())
    u_impute, y_impute = vus[0]
    assert np.allclose(u_impute, batch[0])
    assert y_impute == pytest.approx(s(batch[0])[0])


# --------------------------------------------------------------------------- #
# CE QUI NE DOIT PLUS ETRE DANS LES ETUDES                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_les_etudes_ne_pilotent_plus_l_optimiseur(script):
    import io
    s = io.open(os.path.join(_REPO, script), encoding="utf-8").read()
    for interdit in ("GN_DIRECT", "xt_kb", "setMaximumCallsNumber",
                     "[KB "):
        assert interdit not in s, (
            "%s : le choix des points d'enrichissement est revenu dans "
            "l'etude (%r). Il appartient a `_reliability/eff_ot.py`."
            % (script, interdit))
    assert "_eff_ot.batch_kriging_believer(" in s
