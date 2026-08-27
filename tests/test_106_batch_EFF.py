r"""Le batch d'enrichissement : des points choisis, evalues, verses au plan.

POURQUOI CES TESTS COMPTENT PLUS QUE LES AUTRES
------------------------------------------------
`evaluer_batch_EFF` a deux chemins, et LA CHAINE DE VERIFICATION N'EN
TRAVERSE QU'UN. Les etudes analytiques tournent a `n_workers_DOE = 1` --
donc en sequentiel ; le Moulin Blanc, lui, tourne a 6. La branche qui
s'execute EN PRODUCTION est precisement celle qu'aucun run de controle ne
visite, et qu'aucune comparaison de journal ne peut attester.

Ces tests unitaires sont son seul filet. Ils l'exercent donc entierement :
repartition, conversion U -> X, ordre des resultats, gradients, journal.

CE QU'ILS METTENT AUSSI PAR ECRIT
----------------------------------
1. Le plafond `n_max_EFF_points` tronque le batch -- un batch de 3 quand il
   ne reste qu'une place n'ajoute qu'un point.
2. Les points virtuels de Taylor donnent EXACTEMENT les memes nombres que
   `_doe.plan.augmenter_par_taylor`, qui fait la meme chose sur tout le plan.
   Seuls les journaux different. La duplication est donc reelle, et pinnee
   ici pour qu'une unification future soit sure.
3. `dist_jointe` est un APPEL, pas une distribution : le chemin sequentiel ne
   doit jamais la construire.
4. Les deux formats de journal different (precision 4 avec compteur d'un
   cote, precision 10 de l'autre). C'est ainsi depuis l'origine ; ils sont
   figes ici parce que la comparaison des journaux atteste des extractions.
"""

import os
import sys

import numpy as np
import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_doe"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluation as _evaluation                       # noqa: E402
import plan as _plan                                   # noqa: E402


def _plan_vide(n_var=2):
    return (np.zeros((0, n_var)), np.zeros((0, 1)), np.zeros((0, n_var)), [])


def _solveur_lineaire(u):
    """g = 1 + u0 + 2*u1, gradient constant. De quoi verifier les nombres
    a la main."""
    u = np.asarray(u, float)
    return 1.0 + u[0] + 2.0 * u[1], [1.0, 2.0], [1.0, 2.0]


class _Journal(list):
    def __call__(self, message):
        self.append(message)


# --------------------------------------------------------------------------- #
# LE CHEMIN SEQUENTIEL                                                         #
# --------------------------------------------------------------------------- #
def test_un_point_rejoint_le_plan_avec_son_gradient():
    xt, yt, ag, eff = _plan_vide()
    j = _Journal()
    xt, yt, ag, eff = _evaluation.evaluer_batch_EFF(
        [np.array([1.0, 2.0])], xt, yt, ag, eff,
        n_max_points=10, n_batch=1, n_workers=1, n_var=2,
        evaluer_un_point=_solveur_lineaire, tracer=j)
    assert xt.tolist() == [[1.0, 2.0]]
    assert yt.tolist() == [[6.0]]
    assert ag.tolist() == [[1.0, 2.0]]
    assert [p.tolist() for p in eff] == [[1.0, 2.0]]
    assert len(j) == 1 and j[0].startswith("[EFF HF] u=")


def test_un_seul_worker_reste_sequentiel_meme_a_plusieurs_points():
    xt, yt, ag, eff = _plan_vide()
    j = _Journal()
    xt, yt, ag, eff = _evaluation.evaluer_batch_EFF(
        [np.array([1.0, 0.0]), np.array([0.0, 1.0])], xt, yt, ag, eff,
        n_max_points=10, n_batch=2, n_workers=1, n_var=2,
        evaluer_un_point=_solveur_lineaire, tracer=j)
    assert len(xt) == 2 and len(eff) == 2
    assert all(m.startswith("[EFF HF] ") for m in j)


def test_le_chemin_sequentiel_ne_construit_jamais_la_distribution():
    """`dist_jointe` est un APPEL. Le sequentiel n'en a aucun besoin, et la
    construire couterait pour rien."""
    xt, yt, ag, eff = _plan_vide()

    def _interdit():
        raise AssertionError("le chemin sequentiel a construit la loi jointe")

    _evaluation.evaluer_batch_EFF(
        [np.array([1.0, 2.0])], xt, yt, ag, eff,
        n_max_points=10, n_batch=1, n_workers=1, n_var=2,
        evaluer_un_point=_solveur_lineaire, dist_jointe=_interdit,
        tracer=_Journal())


def test_le_plafond_de_points_tronque_le_batch():
    """Un batch de 3 quand il ne reste qu'une place n'ajoute qu'un point.
    Sans cette troncature, `n_max_EFF_points` serait depasse d'un batch."""
    xt, yt, ag, eff = _plan_vide()
    eff.append(np.array([9.0, 9.0]))          # un point deja enrichi
    batch = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
    xt, yt, ag, eff = _evaluation.evaluer_batch_EFF(
        batch, xt, yt, ag, eff,
        n_max_points=2, n_batch=3, n_workers=1, n_var=2,
        evaluer_un_point=_solveur_lineaire, tracer=_Journal())
    assert len(xt) == 1, "le plafond n'a pas tronque le batch"
    assert len(eff) == 2


def test_un_plafond_atteint_n_ajoute_rien():
    xt, yt, ag, eff = _plan_vide()
    eff.extend([np.array([9.0, 9.0])] * 3)
    xt, yt, ag, eff = _evaluation.evaluer_batch_EFF(
        [np.array([1.0, 0.0])], xt, yt, ag, eff,
        n_max_points=3, n_batch=1, n_workers=1, n_var=2,
        evaluer_un_point=_solveur_lineaire, tracer=_Journal())
    assert len(xt) == 0


# --------------------------------------------------------------------------- #
# LES POINTS VIRTUELS DE TAYLOR                                                #
# --------------------------------------------------------------------------- #
def test_taylor_ajoute_un_voisin_par_direction():
    xt, yt, ag, eff = _plan_vide()
    j = _Journal()
    xt, yt, ag, eff = _evaluation.evaluer_batch_EFF(
        [np.array([1.0, 2.0])], xt, yt, ag, eff,
        n_max_points=10, n_batch=1, n_workers=1, n_var=2,
        evaluer_un_point=_solveur_lineaire, taylor=True, eps_taylor=0.01,
        tracer=j)
    # 1 point reel + 2 virtuels ; mais UN SEUL point d'enrichissement compte
    assert len(xt) == 3
    assert len(eff) == 1, "un point virtuel ne consomme pas le budget EFF"
    assert np.allclose(xt[1], [1.01, 2.0]) and np.allclose(yt[1], [6.01])
    assert np.allclose(xt[2], [1.0, 2.01]) and np.allclose(yt[2], [6.02])
    # le gradient du point REEL est recopie sur ses voisins
    assert ag.tolist() == [[1.0, 2.0]] * 3
    assert sum(m.startswith("[EFF Taylor]") for m in j) == 2


def test_taylor_desactive_par_un_epsilon_nul():
    xt, yt, ag, eff = _plan_vide()
    xt, yt, ag, eff = _evaluation.evaluer_batch_EFF(
        [np.array([1.0, 2.0])], xt, yt, ag, eff,
        n_max_points=10, n_batch=1, n_workers=1, n_var=2,
        evaluer_un_point=_solveur_lineaire, taylor=True, eps_taylor=0.0,
        tracer=_Journal())
    assert len(xt) == 1


def test_taylor_donne_les_MEMES_nombres_que_celui_du_plan():
    """La duplication, pinnee.

    `_doe.plan.augmenter_par_taylor` fait la meme chose sur tout le plan a la
    fois. Les deux implementations coexistent parce que leurs journaux
    different -- et le journal est ce qui atteste des extractions. Ce test
    verifie que seuls les journaux different : le jour ou on les unifiera,
    il dira si les nombres ont bouge.
    """
    u = np.array([1.0, 2.0])
    eps = 0.01

    xt, yt, ag, eff = _plan_vide()
    xt_a, yt_a, ag_a, _ = _evaluation.evaluer_batch_EFF(
        [u], xt, yt, ag, eff, n_max_points=10, n_batch=1, n_workers=1,
        n_var=2, evaluer_un_point=_solveur_lineaire, taylor=True,
        eps_taylor=eps, tracer=_Journal())

    # le meme point reel, augmente par la voie du plan d'experiences
    g, grad, _ = _solveur_lineaire(u)
    xt_b, yt_b, ag_b = _plan.augmenter_par_taylor(
        np.array([u]), np.array([[g]]), np.array([grad], float),
        eps, 2, tracer=lambda m: None)

    assert np.allclose(xt_a, xt_b)
    assert np.allclose(yt_a, yt_b)
    assert np.allclose(ag_a, ag_b)


# --------------------------------------------------------------------------- #
# LE CHEMIN PARALLELE -- SANS AUTRE FILET QUE CELUI-CI                         #
# --------------------------------------------------------------------------- #
class _Pool:
    """Un pool de solveurs, en memoire. Retient ce qu'on lui a demande."""

    def __init__(self):
        self.recu = None
        self.n_workers = None

    def __call__(self, SOL, n_workers):
        self.recu = [dict(d) for d in SOL]
        self.n_workers = n_workers
        for i, d in enumerate(SOL):
            d["g"] = float(i) + 0.5
            d["dg_fc"] = 1.0 + i
            d["dg_fy"] = 2.0 + i
        return SOL


class _LoiJointe:
    """Une loi dont la transformation inverse est l'identite : ce qui entre
    en U ressort tel quel en X, donc les nombres se lisent."""

    def __init__(self):
        self.appels = 0

    def getInverseIsoProbabilisticTransformation(self):
        self.appels += 1
        return lambda p: p


def test_le_batch_part_au_pool_des_qu_il_y_a_plusieurs_points_ET_workers():
    pool, loi = _Pool(), _LoiJointe()
    xt, yt, ag, eff = _plan_vide()
    j = _Journal()
    xt, yt, ag, eff = _evaluation.evaluer_batch_EFF(
        [np.array([1.0, 2.0]), np.array([3.0, 4.0])], xt, yt, ag, eff,
        n_max_points=10, n_batch=2, n_workers=6, n_var=2,
        evaluer_un_point=None, executer_en_parallele=pool,
        dist_jointe=lambda: loi, params_names=["fc", "fy"], tracer=j)

    # les points sont partis en variables PHYSIQUES, dans l'ordre
    assert pool.recu == [{"fc": 1.0, "fy": 2.0}, {"fc": 3.0, "fy": 4.0}]
    # ...et le plan a recu ce que le pool a rendu, dans le meme ordre
    assert xt.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert yt.tolist() == [[0.5], [1.5]]
    assert ag.tolist() == [[1.0, 2.0], [2.0, 3.0]]
    assert [p.tolist() for p in eff] == [[1.0, 2.0], [3.0, 4.0]]
    assert [m.split("]")[0] + "]" for m in j] == ["[EFF HF 1/2]", "[EFF HF 2/2]"]


def test_on_ne_demande_pas_plus_de_workers_que_de_points():
    """Six workers pour deux points, ce sont quatre copies du modele faites
    pour rien -- et sur le Moulin Blanc une copie de `.ds` n'est pas
    gratuite."""
    pool = _Pool()
    xt, yt, ag, eff = _plan_vide()
    _evaluation.evaluer_batch_EFF(
        [np.array([1.0, 2.0]), np.array([3.0, 4.0])], xt, yt, ag, eff,
        n_max_points=10, n_batch=2, n_workers=6, n_var=2,
        evaluer_un_point=None, executer_en_parallele=pool,
        dist_jointe=lambda: _LoiJointe(), params_names=["fc", "fy"],
        tracer=_Journal())
    assert pool.n_workers == 2


def test_le_parallele_n_ajoute_jamais_de_points_de_taylor():
    """Constate, pas decide : la condition d'origine porte `n_batch_EFF <= 1`.
    Un batch parallele ne passe donc jamais par Taylor."""
    pool = _Pool()
    xt, yt, ag, eff = _plan_vide()
    xt, yt, ag, eff = _evaluation.evaluer_batch_EFF(
        [np.array([1.0, 2.0]), np.array([3.0, 4.0])], xt, yt, ag, eff,
        n_max_points=10, n_batch=2, n_workers=6, n_var=2,
        evaluer_un_point=None, executer_en_parallele=pool,
        dist_jointe=lambda: _LoiJointe(), params_names=["fc", "fy"],
        taylor=True, eps_taylor=0.01, tracer=_Journal())
    assert len(xt) == 2


def test_un_point_seul_ne_part_pas_au_pool_meme_a_six_workers():
    """La condition est `len(batch) > 1 AND n_workers > 1`. Un point seul
    reste sequentiel : monter un pool pour un point coute plus qu'il ne
    rend."""
    pool = _Pool()
    xt, yt, ag, eff = _plan_vide()
    _evaluation.evaluer_batch_EFF(
        [np.array([1.0, 2.0])], xt, yt, ag, eff,
        n_max_points=10, n_batch=1, n_workers=6, n_var=2,
        evaluer_un_point=_solveur_lineaire, executer_en_parallele=pool,
        dist_jointe=lambda: _LoiJointe(), params_names=["fc", "fy"],
        tracer=_Journal())
    assert pool.recu is None


def test_un_gradient_absent_du_pool_vaut_zero():
    """Constate, pas approuve : `.get('dg_x', 0.0)` fabrique un gradient nul
    quand le pool n'en rend pas -- alors que `evaluer_en_U`, elle, LEVE dans
    ce cas. Les deux portes d'enrichissement ne traitent donc pas le meme
    defaut de la meme facon.
    """
    def _pool_muet(SOL, n_workers):
        for d in SOL:
            d["g"] = 1.0
        return SOL

    xt, yt, ag, eff = _plan_vide()
    xt, yt, ag, eff = _evaluation.evaluer_batch_EFF(
        [np.array([1.0, 2.0]), np.array([3.0, 4.0])], xt, yt, ag, eff,
        n_max_points=10, n_batch=2, n_workers=6, n_var=2,
        evaluer_un_point=None, executer_en_parallele=_pool_muet,
        dist_jointe=lambda: _LoiJointe(), params_names=["fc", "fy"],
        tracer=_Journal())
    assert ag.tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_un_gradient_None_du_pool_fait_lever():
    """L'autre face du meme defaut. `run_DOE_parallel` ecrit
    `SOL[i]['dg_q'] = d.get('dg_q')` : la CLEF existe, avec `None` dedans.
    `.get(..., 0.0)` ne la remplace donc pas, et `round(None, 6)` leve.

    Ce n'est pas un choix, c'est un angle mort : selon que le pool omette la
    clef ou y mette `None`, le meme defaut de solveur donne un gradient nul
    silencieux ou une exception. Fige ici en attendant l'arbitrage.
    """
    def _pool_None(SOL, n_workers):
        for d in SOL:
            d["g"] = 1.0
            d["dg_fc"] = None
            d["dg_fy"] = None
        return SOL

    xt, yt, ag, eff = _plan_vide()
    with pytest.raises(TypeError):
        _evaluation.evaluer_batch_EFF(
            [np.array([1.0, 2.0]), np.array([3.0, 4.0])], xt, yt, ag, eff,
            n_max_points=10, n_batch=2, n_workers=6, n_var=2,
            evaluer_un_point=None, executer_en_parallele=_pool_None,
            dist_jointe=lambda: _LoiJointe(), params_names=["fc", "fy"],
            tracer=_Journal())


# --------------------------------------------------------------------------- #
# LES JOURNAUX, FIGES                                                          #
# --------------------------------------------------------------------------- #
def test_le_journal_sequentiel_garde_son_format():
    j = _Journal()
    xt, yt, ag, eff = _plan_vide()
    _evaluation.evaluer_batch_EFF(
        [np.array([1.0, 2.0])], xt, yt, ag, eff,
        n_max_points=10, n_batch=1, n_workers=1, n_var=2,
        evaluer_un_point=_solveur_lineaire, tracer=j)
    assert j[0] == ("[EFF HF] u=%s  g=6.0000000000  grad_U=[1.0, 2.0]"
                    % (list(np.round(np.array([1.0, 2.0]), 10)),))


def test_le_journal_parallele_garde_son_format():
    j = _Journal()
    xt, yt, ag, eff = _plan_vide()
    _evaluation.evaluer_batch_EFF(
        [np.array([1.0, 2.0]), np.array([3.0, 4.0])], xt, yt, ag, eff,
        n_max_points=10, n_batch=2, n_workers=6, n_var=2,
        evaluer_un_point=None, executer_en_parallele=_Pool(),
        dist_jointe=lambda: _LoiJointe(), params_names=["fc", "fy"], tracer=j)
    assert j[0] == ("[EFF HF 1/2] u=%s  g=0.500000  grad_U=[1.0, 2.0]"
                    % (list(np.round(np.array([1.0, 2.0]), 4)),))


# --------------------------------------------------------------------------- #
# CE QUI NE DOIT PLUS ETRE DANS LES ETUDES                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_les_etudes_ne_versent_plus_les_points_elles_memes(script):
    import io
    s = io.open(os.path.join(_REPO, script), encoding="utf-8").read()
    for interdit in ("_batch_to_eval", "_n_batch_actual", "[EFF Taylor]",
                     "T_inv_eff"):
        assert interdit not in s, (
            "%s : l'evaluation du batch EFF est revenue dans l'etude (%r). "
            "Elle appartient a `_doe/evaluation.py`." % (script, interdit))
    assert "_evaluation.evaluer_batch_EFF(" in s
