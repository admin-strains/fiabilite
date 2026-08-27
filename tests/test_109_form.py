r"""L'evenement de defaillance, les points de depart, le warm start.

CE QUE C'ETAIT
---------------
Trois fonctions recopiees dans les deux etudes, dont une -- `FORM_warm_start`
-- qui peut relancer une recherche FORM multistart complete SANS ECRIRE UNE
LIGNE DE JOURNAL. Et l'evenement `g < 0`, construit a deux endroits
differents du meme fichier : une fois dans `init_FORM`, une fois a la main
pour le metamodele projete.

CE QUE CES TESTS METTENT PAR ECRIT
-----------------------------------
1. Le warm start verse au plan la PREDICTION DU METAMODELE, pas un appel
   solveur. Zero appel haute fidelite -- et un test mesure ce que le
   reajustement change reellement, au lieu de l'affirmer.
2. Il ne rend PAS le plan modifie. C'est le choix d'origine, conserve : les
   points fictifs ne survivent pas a l'appel.
3. La tolerance est une borne STRICTE : a `|g(u*)| == tol`, on accepte le
   resultat sans relancer.
4. L'origine appartient a toute recherche, multistart ou non : c'est le
   point le plus probable de l'espace standard.
"""

import io
import os
import sys

import numpy as np
import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_reliability"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ot = pytest.importorskip("openturns")
_form = pytest.importorskip("form", reason="FORM multimodal, OpenTURNS + sklearn")


# --------------------------------------------------------------------------- #
# L'EVENEMENT DE DEFAILLANCE                                                   #
# --------------------------------------------------------------------------- #
class _Affine(ot.OpenTURNSPythonFunction):
    """g = 2 - u0 - u1 : l'etat limite est la droite u0 + u1 = 2."""

    def __init__(self, n=2):
        super().__init__(n, 1)
        self.n = n

    def _exec(self, u):
        return [2.0 - float(sum(u))]


def test_l_evenement_est_g_strictement_negatif():
    g = ot.Function(_Affine())
    e = _form.evenement_de_defaillance(g, 2)
    assert e.getOperator().getImplementation().getClassName() == "Less"
    assert float(e.getThreshold()) == 0.0


def test_sans_metamodele_il_n_y_a_pas_d_evenement():
    """En HF pur il n'y a pas de metamodele, et l'etude continue sans
    evenement. Lever ici arreterait un run parfaitement valide."""
    assert _form.evenement_de_defaillance(None, 3) is None


def test_l_espace_standard_impose_sa_loi_quelle_que_soit_la_dimension():
    """La normale centree reduite en toutes dimensions est la DEFINITION de
    l'espace standard, pas un reglage : les lois physiques sont deja passees
    dans la transformation isoprobabiliste."""
    for n in (1, 2, 5):
        e = _form.evenement_de_defaillance(ot.Function(_Affine(n)), n)
        loi = e.getAntecedent().getDistribution()
        assert loi.getDimension() == n
        assert loi.getMean() == ot.Point([0.0] * n)
        assert all(abs(float(s) - 1.0) < 1e-12
                   for s in loi.getStandardDeviation())


def test_le_meme_constructeur_sert_au_metamodele_PROJETE():
    """L'evenement projete etait rebati a la main, en dimension reduite,
    alors que la construction etait deja ecrite quatre lignes plus haut."""
    e = _form.evenement_de_defaillance(ot.Function(_Affine(2)), 2)
    assert e.getAntecedent().getDistribution().getDimension() == 2


# --------------------------------------------------------------------------- #
# LES POINTS DE DEPART                                                         #
# --------------------------------------------------------------------------- #
def test_l_origine_appartient_a_toute_recherche():
    """C'est le point le plus probable de l'espace standard : partir de la
    est ce que fait un FORM sans multistart, et le multistart ne doit pas
    l'ecarter."""
    xt = np.array([[1.0, 2.0], [3.0, 4.0]])
    seul = _form.points_de_depart(xt, 2, multistart=False)
    tous = _form.points_de_depart(xt, 2, multistart=True)
    assert seul.tolist() == [[0.0, 0.0]]
    assert tous.tolist() == [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]


def test_le_multistart_part_de_TOUT_le_plan():
    xt = np.arange(12, dtype=float).reshape(4, 3)
    tous = _form.points_de_depart(xt, 3, multistart=True)
    assert len(tous) == 5
    assert tous[-1].tolist() == [0.0, 0.0, 0.0]


def test_un_plan_vide_laisse_l_origine():
    tous = _form.points_de_depart(np.zeros((0, 2)), 2, multistart=True)
    assert tous.tolist() == [[0.0, 0.0]]


# --------------------------------------------------------------------------- #
# LE WARM START                                                                #
# --------------------------------------------------------------------------- #
class _Mode:
    def __init__(self, u_star):
        self._u = list(u_star)

    def getStandardSpaceDesignPoint(self):
        return ot.Point(self._u)


class _Metamodele:
    """Rend `valeur` partout, gradient constant. Compte ses appels."""

    def __init__(self, valeur):
        self.valeur = valeur
        self.appels = 0

    def __call__(self, u):
        self.appels += 1
        return [self.valeur]

    def gradient(self, u):
        return ot.Matrix([[3.0], [4.0]])


def _rappels():
    """Un `reajuster_et_evenement` et un `rechercher_modes` qui retiennent
    ce qu'on leur a passe."""
    vu = {"plan": None, "depart": None, "n": 0}

    def reajuster_et_evenement(xt, yt, ag):
        vu["plan"] = (xt.copy(), yt.copy(), ag.copy())
        vu["n"] += 1
        return "evenement-reajuste", xt

    def rechercher_modes(depart, tol, evenement):
        vu["depart"] = np.array(depart)
        vu["evenement"] = evenement
        return ["mode-neuf"], ["sp-neuf"]

    return reajuster_et_evenement, rechercher_modes, vu


def _appeler(g_val, tolerance=0.2, modes=None, xt=None):
    g = _Metamodele(g_val)
    reajuster, chercher, vu = _rappels()
    xt = np.array([[1.0, 1.0], [2.0, 2.0]]) if xt is None else xt
    yt = np.array([[0.5], [0.6]])
    ag = np.array([[1.0, 1.0], [1.0, 1.0]])
    modes = [_Mode([1.5, -0.5])] if modes is None else modes
    res = _form.warm_start(modes, ["sp-initial"], g, xt, yt, ag, n_var=2,
                           tolerance=tolerance, multistart=True,
                           tol_all_modes=0.1,
                           reajuster_et_evenement=reajuster,
                           rechercher_modes=chercher, tracer=lambda m: None)
    return res, vu, g, (xt, yt, ag)


def test_sans_mode_il_n_y_a_rien_a_relancer():
    (modes, sps), vu, g, _ = _appeler(5.0, modes=[])
    assert modes == [] and sps == ["sp-initial"]
    assert vu["n"] == 0 and g.appels == 0


def test_un_mode_pose_sur_l_etat_limite_est_accepte_tel_quel():
    (modes, sps), vu, _, _ = _appeler(0.01)
    assert modes[0].__class__ is _Mode and sps == ["sp-initial"]
    assert vu["n"] == 0, "un mode deja convergent a paye un reajustement"


def test_la_tolerance_est_une_borne_stricte():
    """A `|g(u*)| == tol` exactement, on accepte. Le code d'origine teste
    `abs(g) > tol`, et une egalite qui declencherait relancerait FORM pour
    rien."""
    (modes, _), vu, _, _ = _appeler(0.2, tolerance=0.2)
    assert vu["n"] == 0
    (modes, _), vu, _, _ = _appeler(0.2000001, tolerance=0.2)
    assert vu["n"] == 1


def test_un_mode_LOIN_de_l_etat_limite_fait_tout_recommencer():
    (modes, sps), vu, _, _ = _appeler(5.0)
    assert modes == ["mode-neuf"] and sps == ["sp-neuf"]
    assert vu["n"] == 1
    assert vu["evenement"] == "evenement-reajuste"


def test_un_g_negatif_compte_par_sa_valeur_absolue():
    (modes, _), vu, _, _ = _appeler(-5.0)
    assert vu["n"] == 1 and modes == ["mode-neuf"]


def test_ce_qui_est_verse_au_plan_est_la_PREDICTION_du_metamodele():
    """Pas un appel solveur : le couple ajoute est
    `(u*, ce que le modele predit deja en u*)`."""
    (_, _), vu, g, _ = _appeler(5.0)
    xt, yt, ag = vu["plan"]
    assert xt[-1].tolist() == [1.5, -0.5]
    assert yt[-1].tolist() == [5.0], "la valeur versee est celle du metamodele"
    assert ag[-1].tolist() == [3.0, 4.0], "le gradient aussi"


def test_le_warm_start_ne_coute_AUCUN_appel_solveur():
    """Un seul appel, au metamodele, pour lire `g(u*)`. Le reste est du
    reajustement -- cher en secondes, jamais en heures."""
    (_, _), _, g, _ = _appeler(5.0)
    assert g.appels == 1


def test_le_plan_REEL_de_l_appelant_ressort_intact():
    """Le warm start ne rend que `(modes, best_sps)` : le point fictif ne
    survit pas a l'appel, et le plan reel n'est pas contamine. C'est le
    choix d'origine, conserve."""
    xt = np.array([[1.0, 1.0], [2.0, 2.0]])
    (_, _), vu, _, (xt_apres, yt, ag) = _appeler(5.0, xt=xt)
    assert xt_apres.tolist() == [[1.0, 1.0], [2.0, 2.0]]
    assert len(yt) == 2 and len(ag) == 2
    # le plan AUGMENTE n'existe que le temps du reajustement
    assert len(vu["plan"][0]) == 3


def test_les_points_de_depart_de_la_relance_incluent_le_point_fictif():
    """La relance repart du plan augmente : c'est ce que fait le code
    d'origine, et c'est coherent -- le point fictif est la ou FORM s'est
    arrete, donc un depart plausible."""
    (_, _), vu, _, _ = _appeler(5.0)
    assert vu["depart"].tolist() == [[1.0, 1.0], [2.0, 2.0], [1.5, -0.5],
                                     [0.0, 0.0]]


def test_le_warm_start_le_DIT_desormais():
    """Une operation qui relance une recherche FORM multistart complete ne
    doit pas etre silencieuse. Elle l'etait."""
    g = _Metamodele(5.0)
    reajuster, chercher, _ = _rappels()
    j = []
    _form.warm_start([_Mode([1.5, -0.5])], ["sp"], g,
                     np.array([[1.0, 1.0]]), np.array([[0.5]]),
                     np.array([[1.0, 1.0]]), n_var=2, tolerance=0.2,
                     multistart=True, tol_all_modes=0.1,
                     reajuster_et_evenement=reajuster,
                     rechercher_modes=chercher, tracer=j.append)
    assert len(j) == 1
    assert "[FORM WARM START]" in j[0] and "5.000000" in j[0]


def test_un_mode_convergent_reste_silencieux():
    g = _Metamodele(0.01)
    reajuster, chercher, _ = _rappels()
    j = []
    _form.warm_start([_Mode([1.5, -0.5])], ["sp"], g,
                     np.array([[1.0, 1.0]]), np.array([[0.5]]),
                     np.array([[1.0, 1.0]]), n_var=2, tolerance=0.2,
                     multistart=True, tol_all_modes=0.1,
                     reajuster_et_evenement=reajuster,
                     rechercher_modes=chercher, tracer=j.append)
    assert j == []


# --------------------------------------------------------------------------- #
# LA COUPE LA PLUS PARLANTE                                                    #
# --------------------------------------------------------------------------- #
class _Resultat:
    def __init__(self, importance, u_star):
        self._i, self._u = importance, u_star

    def getImportanceFactors(self):
        return ot.Point(self._i)

    def getStandardSpaceDesignPoint(self):
        return ot.Point(self._u)


def test_sans_resultat_FORM_la_coupe_par_defaut_est_gardee():
    assert _form.coupe_la_plus_parlante(None, 3, (0, 1, {})) == (0, 1, {})


def test_les_deux_variables_les_plus_lourdes_font_le_plan():
    r = _Resultat([0.1, 0.7, 0.05, 0.15], [1.0, 2.0, 3.0, 4.0])
    ix, iy, figees = _form.coupe_la_plus_parlante(r, 4, (0, 1, {}))
    assert (ix, iy) == (1, 3), "les deux plus grands facteurs"
    assert figees == {0: 1.0, 2: 3.0}, "les autres, figees a u*"


def test_les_axes_sortent_toujours_ordonnes():
    """Peu importe lequel pese le plus : une coupe (3, 1) et une coupe
    (1, 3) designent le meme plan, mais les figures liraient les axes a
    l'envers."""
    r = _Resultat([0.05, 0.3, 0.05, 0.6], [1.0, 2.0, 3.0, 4.0])
    ix, iy, _ = _form.coupe_la_plus_parlante(r, 4, (0, 1, {}))
    assert ix < iy


def test_a_deux_variables_rien_n_est_fige():
    r = _Resultat([0.4, 0.6], [1.0, 2.0])
    assert _form.coupe_la_plus_parlante(r, 2, (0, 1, {})) == (0, 1, {})


# --------------------------------------------------------------------------- #
# CE QUI NE DOIT PLUS ETRE DANS LES ETUDES                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_les_etudes_ne_construisent_plus_d_evenement(script):
    s = io.open(os.path.join(_REPO, script), encoding="utf-8").read()
    for interdit in ("ThresholdEvent", "JointDistribution([ot.Normal(0, 1)]",
                     "if do_multistart else"):
        assert interdit not in s, (
            "%s : la construction de l'evenement ou des points de depart est "
            "revenue dans l'etude (%r). Elle appartient a "
            "`_reliability/form.py`." % (script, interdit))
    assert "_form.evenement_de_defaillance(" in s
    assert "_form.warm_start(" in s
