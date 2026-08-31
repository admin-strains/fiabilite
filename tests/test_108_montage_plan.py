r"""Monter le plan d'experiences : greffer, assembler, journaliser.

CE QUE C'ETAIT
---------------
Trente lignes au milieu de `build_DOE`, recopiees a l'identique dans les
deux etudes, dans `if __name__ == '__main__':`. Aucun test ne pouvait les
atteindre alors qu'elles decident de ce qui entre dans le plan -- et donc
de ce sur quoi le metamodele est ajuste.

CE QUE CES TESTS METTENT PAR ECRIT
-----------------------------------
1. La greffe d'un plan interrompu recopie ce qui est deja calcule, sans rien
   supposer : `charger_doe_partiel` a deja verifie que le tirage redonne les
   memes points. `None` -- rien a reprendre -- est le cas courant, pas une
   erreur.
2. Un point ecarte l'est PARTOUT a la fois. Un point present dans `xt` mais
   absent de `all_grad` donnerait deux tableaux de longueurs differentes, et
   le metamodele apprendrait des couples decales.
3. Le `or 0.0` fabrique un gradient nul, qui AFFIRME que l'etat limite est
   plat. Ce n'est acceptable que parce que c'est un reglage explicite
   (`exclure_points_sans_gradient = false`) annonce dans le journal.
   Consequence a connaitre : un gradient reellement nul est indistinguable
   d'un gradient absent.
4. Le plan est imprime RECOPIABLE, a 16 decimales. Ce n'est pas de la
   decoration : ces tableaux rejouent une etude sans rappeler le solveur --
   trois heures de Moulin Blanc pour un plan de 24 points.
"""

import io
import os
import sys

import numpy as np
import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_cache"), os.path.join(_REPO, "_doe")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# AVANT les imports du depot : `_doe/plan.py` importe OpenTURNS au premier
# niveau. Sans cette ligne, son absence casse la COLLECTE, pas ce seul
# fichier -- et pytest interrompt la suite ENTIERE. Voir `test_05`.
pytest.importorskip("openturns")

import doe as _cache_doe                               # noqa: E402
import plan as _plan                                   # noqa: E402


PARAMS = ["fc", "fy"]


def _SOL(n):
    return [{"fc": float(i), "fy": float(i) + 0.5} for i in range(n)]


class _Journal(list):
    def __call__(self, message):
        self.append(message)


# --------------------------------------------------------------------------- #
# LA GREFFE D'UN PLAN INTERROMPU                                               #
# --------------------------------------------------------------------------- #
def test_rien_a_reprendre_n_est_pas_une_erreur():
    """C'est le cas courant : un plan qui demarre."""
    SOL = _SOL(3)
    assert _cache_doe.greffer_reprise(SOL, None, PARAMS) == 0
    assert all("g" not in d for d in SOL)


def test_les_points_deja_calcules_sont_recopies_dans_SOL():
    SOL = _SOL(4)
    repris = (np.array([[0.1, 0.2], [0.3, 0.4]]),      # xt
              np.array([[5.0], [6.0]]),                 # yt
              np.array([[1.0, 2.0], [3.0, 4.0]]),       # all_grad
              2)                                        # n_faits
    assert _cache_doe.greffer_reprise(SOL, repris, PARAMS) == 2
    assert SOL[0]["g"] == 5.0 and SOL[1]["g"] == 6.0
    assert SOL[0]["dg_fc"] == 1.0 and SOL[0]["dg_fy"] == 2.0
    assert SOL[1]["dg_fc"] == 3.0 and SOL[1]["dg_fy"] == 4.0
    assert SOL[0]["_u"] == [0.1, 0.2]


def test_les_points_NON_calcules_restent_intacts():
    """Ce sont eux que le solveur doit encore traiter. Leur poser un `g`
    par avance les ferait passer pour faits."""
    SOL = _SOL(4)
    repris = (np.array([[0.1, 0.2]]), np.array([[5.0]]),
              np.array([[1.0, 2.0]]), 1)
    _cache_doe.greffer_reprise(SOL, repris, PARAMS)
    assert "g" not in SOL[1] and "g" not in SOL[2] and "g" not in SOL[3]


def test_la_greffe_s_arrete_a_n_faits_meme_si_le_cache_en_porte_plus():
    """`n_faits` fait foi, pas la longueur des tableaux : le cache
    incremental est ecrit apres chaque point, la derniere ligne peut etre
    partielle."""
    SOL = _SOL(4)
    repris = (np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
              np.array([[5.0], [6.0], [7.0]]),
              np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), 2)
    assert _cache_doe.greffer_reprise(SOL, repris, PARAMS) == 2
    assert "g" not in SOL[2]


# --------------------------------------------------------------------------- #
# L'ASSEMBLAGE                                                                 #
# --------------------------------------------------------------------------- #
def _SOL_complet(n, avec_gradient=True):
    SOL = []
    for i in range(n):
        d = {"g": float(i)}
        if avec_gradient:
            d["dg_fc"] = float(i) + 0.1
            d["dg_fy"] = float(i) + 0.2
        SOL.append(d)
    return SOL


def test_le_plan_assemble_a_les_memes_trois_longueurs():
    xt = np.arange(8, dtype=float).reshape(4, 2)
    xt_r, yt, ag = _plan.assembler_plan(_SOL_complet(4), [0, 1, 2, 3], xt, PARAMS)
    assert len(xt_r) == len(yt) == len(ag) == 4
    assert yt.shape == (4, 1)
    assert ag.tolist() == [[0.1, 0.2], [1.1, 1.2], [2.1, 2.2], [3.1, 3.2]]


def test_un_point_ecarte_l_est_dans_LES_TROIS_tableaux():
    """C'est l'invariant qui compte. Un point present dans `xt` mais absent
    de `all_grad` decalerait les couples, et le metamodele apprendrait un
    gradient au mauvais point."""
    xt = np.arange(8, dtype=float).reshape(4, 2)
    xt_r, yt, ag = _plan.assembler_plan(_SOL_complet(4), [0, 3], xt, PARAMS)
    assert len(xt_r) == len(yt) == len(ag) == 2
    assert xt_r.tolist() == [[0.0, 1.0], [6.0, 7.0]]
    assert yt.tolist() == [[0.0], [3.0]]
    assert ag.tolist() == [[0.1, 0.2], [3.1, 3.2]]


def test_aucun_point_retenu_donne_trois_tableaux_vides_et_coherents():
    xt = np.arange(8, dtype=float).reshape(4, 2)
    xt_r, yt, ag = _plan.assembler_plan(_SOL_complet(4), [], xt, PARAMS)
    assert len(xt_r) == 0 and yt.shape == (0, 1) and len(ag) == 0


def test_un_gradient_absent_devient_zero():
    """Reglage `exclure_points_sans_gradient = false`. Le zero AFFIRME que
    l'etat limite est plat -- il n'est pas neutre. Il n'est acceptable que
    parce qu'il est demande explicitement et annonce dans le journal par
    `points_avec_gradient`."""
    SOL = _SOL_complet(2, avec_gradient=False)
    xt = np.arange(4, dtype=float).reshape(2, 2)
    _, _, ag = _plan.assembler_plan(SOL, [0, 1], xt, PARAMS)
    assert ag.tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_un_gradient_None_devient_zero_lui_aussi():
    """La clef peut EXISTER avec `None` dedans -- `run_DOE_parallel` ecrit
    `SOL[i]['dg_q'] = d.get('dg_q')`. Un `.get(clef, 0.0)` ne la remplacerait
    pas ; c'est le `or` qui traite les deux cas."""
    SOL = [{"g": 1.0, "dg_fc": None, "dg_fy": None}]
    _, _, ag = _plan.assembler_plan(SOL, [0], np.zeros((1, 2)), PARAMS)
    assert ag.tolist() == [[0.0, 0.0]]


def test_un_gradient_VRAIMENT_nul_est_indistinguable_d_un_gradient_absent():
    """La consequence du `or`, ecrite pour qu'on ne la decouvre pas en
    lisant un resultat. Un etat limite reellement plat dans une direction
    donne le meme tableau qu'un solveur muet."""
    plat = [{"g": 1.0, "dg_fc": 0.0, "dg_fy": 0.0}]
    muet = [{"g": 1.0}]
    _, _, ag_plat = _plan.assembler_plan(plat, [0], np.zeros((1, 2)), PARAMS)
    _, _, ag_muet = _plan.assembler_plan(muet, [0], np.zeros((1, 2)), PARAMS)
    assert ag_plat.tolist() == ag_muet.tolist()


def test_l_ordre_des_parametres_fixe_l_ordre_des_colonnes():
    SOL = [{"g": 1.0, "dg_fc": 7.0, "dg_fy": 9.0}]
    _, _, a = _plan.assembler_plan(SOL, [0], np.zeros((1, 2)), ["fc", "fy"])
    _, _, b = _plan.assembler_plan(SOL, [0], np.zeros((1, 2)), ["fy", "fc"])
    assert a.tolist() == [[7.0, 9.0]] and b.tolist() == [[9.0, 7.0]]


# --------------------------------------------------------------------------- #
# LE PLAN RECOPIABLE                                                           #
# --------------------------------------------------------------------------- #
def test_le_plan_s_imprime_en_python_recopiable():
    j = _Journal()
    _plan.journaliser_plan(np.array([[1.5], [-2.25]]),
                           np.array([[0.5, -0.25], [1.0, 2.0]]), tracer=j)
    assert j[0] == "yt_doe = ["
    assert j[3] == "]"
    assert j[4] == "all_grad_doe = ["
    assert j[-1] == "]"
    # ce qui est imprime doit se relire tel quel
    code = "\n".join(j)
    espace = {}
    exec(code, espace)
    assert espace["yt_doe"] == [1.5, -2.25]
    assert espace["all_grad_doe"] == [[0.5, -0.25], [1.0, 2.0]]


def test_seize_decimales_sur_yt_pour_que_la_copie_ne_soit_pas_une_approximation():
    """L'interet du plan recopiable est de RETROUVER les memes nombres. A
    six decimales, rejouer l'etude ne donnerait pas le meme metamodele."""
    j = _Journal()
    valeur = 0.1234567890123456
    _plan.journaliser_plan(np.array([[valeur]]), np.zeros((1, 2)), tracer=j)
    assert j[1].strip().rstrip(",") == "%.16f" % valeur
    espace = {}
    exec("\n".join(j), espace)
    assert espace["yt_doe"][0] == pytest.approx(valeur, rel=1e-15)


def test_un_plan_vide_s_imprime_quand_meme_valide():
    j = _Journal()
    _plan.journaliser_plan(np.zeros((0, 1)), np.zeros((0, 2)), tracer=j)
    espace = {}
    exec("\n".join(j), espace)
    assert espace["yt_doe"] == [] and espace["all_grad_doe"] == []


# --------------------------------------------------------------------------- #
# CE QUI NE DOIT PLUS ETRE DANS LES ETUDES                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_les_etudes_ne_montent_plus_le_plan_elles_memes(script):
    s = io.open(os.path.join(_REPO, script), encoding="utf-8").read()
    for interdit in ("yt_doe = [", "all_grad_doe = [", "_ag_r[i][j]",
                     "or 0.0 for p in params_names"):
        assert interdit not in s, (
            "%s : le montage du plan est revenu dans l'etude (%r)."
            % (script, interdit))
    for parti in ("_plan.assembler_plan(", "_cache_doe.greffer_reprise(",
                  "_plan.tirer_plan_lhs("):
        assert parti not in s, (
            "%s : %s est revenu dans l'etude ; l'enchainement du plan initial "
            "appartient a `_doe/plan.py` depuis le 29/08/2026." % (script, parti))


def test_le_plan_initial_est_monte_par_le_module():
    """Le pendant positif, sur le seul assembleur qui reste."""
    s = io.open(os.path.join(_REPO, "_doe", "plan.py"), encoding="utf-8").read()
    for atteste in ("def construire_plan_initial(", "assembler_plan(",
                    "_cache_doe.greffer_reprise("):
        assert atteste in s, atteste
