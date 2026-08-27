r"""Quand arreter d'enrichir : les quatre criteres, enfin testables.

CE QU'ILS ETAIENT
------------------
68 lignes de quatre branches quasi identiques, au milieu de `run_EFF` -- la
plus grosse fonction du depot (321 lignes). Chaque branche recopiait la meme
idee : compter les iterations consecutives ou beta est stable.

Agnes, 27/08/2026 : « on n'est pas au clair sur nos criteres de
convergence ». Ces tests ne les changent donc pas -- ils ECRIVENT ce qu'ils
font aujourd'hui, y compris leurs asymetries, pour qu'on puisse en discuter
sur pieces plutot que de relire quatre branches.

CE QUE LES TESTS ONT MIS PAR ECRIT
-----------------------------------
1. En mode `BB`, `hist_BS` ne recoit rien -- et reciproquement. La courbe de
   convergence de fin de run montre donc des choses differentes selon le
   critere choisi.
2. En mode `both`, les compteurs BB et BS ne bougent JAMAIS ; seul `both` est
   tenu. Le bilan affiche pourtant les trois.
3. La reprise des compteurs apres interruption n'existait que d'un cote, et
   sa boucle levait `TypeError` sur un historique contenant `None`.
"""

import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
if os.path.join(_REPO, "_reliability") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "_reliability"))

import arret as _arret                                # noqa: E402

TOL_BB = 0.05
TOL_BS = 0.01


def _a(critere, **kw):
    defauts = dict(tol_BB=TOL_BB, tol_BS=TOL_BS, n_max_points=15,
                   tol_EFF=0.001, tracer=lambda _m: None)
    defauts.update(kw)
    return _arret.ArretEFF(critere, **defauts)


# --------------------------------------------------------------------- #
# la condition de boucle : trois raisons d'arreter
# --------------------------------------------------------------------- #
def test_le_budget_de_points_arrete_la_boucle():
    a = _a("at_least_one", n_max_points=5)
    assert a.continuer(4, 1.0)
    assert not a.continuer(5, 1.0), "le budget doit primer"


def test_le_critere_EFF_lui_meme_arrete_la_boucle():
    a = _a("at_least_one", tol_EFF=0.001)
    assert a.continuer(0, 0.01)
    assert not a.continuer(0, 0.0005)
    assert not a.continuer(0, -0.0005), "c'est la valeur ABSOLUE qui compte"


def test_trois_iterations_valides_de_suite_arretent_le_mode_BB():
    a = _a("BB")
    for _ in range(2):
        a.enregistrer(0.01, None, None)
        assert a.continuer(0, 1.0), "deux iterations ne suffisent pas"
    a.enregistrer(0.01, None, None)
    assert not a.continuer(0, 1.0)


def test_une_iteration_invalide_remet_le_compteur_a_zero():
    """Un palier traverse ne doit pas passer pour une convergence."""
    a = _a("BB")
    a.enregistrer(0.01, None, None)
    a.enregistrer(0.01, None, None)
    a.enregistrer(0.9, None, None)          # le ratio remonte
    assert a.n_BB == 0
    assert a.continuer(0, 1.0)


def test_le_mode_both_demande_DEUX_iterations():
    a = _a("both")
    a.enregistrer(0.01, 4.0, 4.0)
    assert a.continuer(0, 1.0)
    a.enregistrer(0.01, 4.0, 4.0)
    assert not a.continuer(0, 1.0)


def test_at_least_one_s_arrete_des_qu_un_critere_est_atteint():
    """BS seul suffit : trois iterations ou beta ne bouge plus."""
    a = _a("at_least_one")
    for i in range(3):
        # ratio BB volontairement mauvais, ratio BS excellent
        a.enregistrer(0.9, 4.0, 4.0)
    assert a.n_BS == 3 and a.n_BB == 0
    assert not a.continuer(0, 1.0)


def test_un_critere_inconnu_ne_bloque_pas_sur_les_compteurs():
    """Il reste alors le budget de points et le critere EFF."""
    a = _a("autre_chose", n_max_points=3)
    assert a.continuer(0, 1.0)
    assert not a.continuer(3, 1.0)


# --------------------------------------------------------------------- #
# le ratio BS
# --------------------------------------------------------------------- #
def test_le_ratio_BS_est_l_ecart_relatif_entre_deux_beta():
    messages = []
    a = _a("BS", tracer=messages.append)
    _bb, bs = a.enregistrer(None, 4.0, 4.2, prefixe="N=12")
    assert bs == pytest.approx(abs(4.0 - 4.2) / 4.0)
    assert "N=12" in messages[0] and "beta_IS_prec" in messages[0]


def test_sans_iteration_precedente_le_ratio_BS_est_indefini():
    """Et le compteur retombe a zero : on ne sait pas, donc on ne compte pas."""
    a = _a("BS")
    a.enregistrer(0.01, 4.0, 4.0)     # une premiere valide
    assert a.n_BS == 1
    _bb, bs = a.enregistrer(None, 4.0, None)
    assert bs is None and a.n_BS == 0


def test_un_FORM_echoue_ne_compte_pas_comme_une_iteration_valide():
    a = _a("BS")
    a.enregistrer(None, 4.0, 4.0)
    _bb, bs = a.enregistrer(None, None, 4.0)   # beta indisponible
    assert bs is None and a.n_BS == 0


def test_un_beta_nul_ne_fait_pas_diviser_par_zero():
    a = _a("BS")
    _bb, bs = a.enregistrer(None, 0.0, 1.0)
    assert bs is None


# --------------------------------------------------------------------- #
# les asymetries, mises par ecrit
# --------------------------------------------------------------------- #
def test_en_mode_BB_l_historique_BS_reste_vide():
    """ASYMETRIE TRANSCRITE, pas corrigee : la courbe de convergence de fin de
    run ne montre pas la meme chose selon le critere choisi."""
    a = _a("BB")
    a.enregistrer(0.02, 4.0, 4.1)
    assert a.hist_BB == [0.02]
    assert a.hist_BS == []


def test_en_mode_BS_l_historique_BB_reste_vide():
    a = _a("BS")
    a.enregistrer(0.02, 4.0, 4.1)
    assert a.hist_BB == []
    assert len(a.hist_BS) == 1


def test_en_mode_both_les_compteurs_BB_et_BS_ne_bougent_jamais():
    """ASYMETRIE TRANSCRITE : le bilan de fin de run affiche pourtant les
    trois compteurs."""
    a = _a("both")
    for _ in range(3):
        a.enregistrer(0.001, 4.0, 4.0)
    assert a.n_both == 3
    assert a.n_BB == 0 and a.n_BS == 0


def test_en_mode_at_least_one_les_trois_compteurs_sont_tenus():
    a = _a("at_least_one")
    a.enregistrer(0.001, 4.0, 4.0)
    assert a.n_BB == 1 and a.n_BS == 1 and a.n_both == 1


# --------------------------------------------------------------------- #
# l'amorce sur le plan initial
# --------------------------------------------------------------------- #
def test_un_plan_qui_part_convergent_compte_deja_une_iteration():
    a = _a("BB")
    a.amorcer(0.01)
    assert a.n_BB == 1 and a.hist_BB == [0.01]


def test_un_plan_qui_part_loin_ne_compte_pas():
    a = _a("BB")
    a.amorcer(0.9)
    assert a.n_BB == 0 and a.hist_BB == [0.9]


def test_l_amorce_ne_fait_rien_en_mode_BS():
    a = _a("BS")
    a.amorcer(0.01)
    assert a.n_BB == 0 and a.hist_BB == []


def test_en_mode_both_l_amorce_remplit_l_historique_sans_compter():
    a = _a("both")
    a.amorcer(0.01)
    assert a.hist_BB == [0.01] and a.n_BB == 0


# --------------------------------------------------------------------- #
# LA REPRISE -- elle n'existait que d'un cote
# --------------------------------------------------------------------- #
def test_la_reprise_recompte_les_iterations_valides_de_suite():
    """Sans elle, un run interrompu apres deux iterations valides sur trois en
    redemande trois -- deux appels solveur pour rien, une demi-heure sur le
    Moulin Blanc."""
    a = _a("at_least_one", hist_BB=[0.9, 0.02, 0.01], hist_BS=[0.5, 0.002])
    n_bb, n_bs = a.reprendre_depuis_historique()
    assert (n_bb, n_bs) == (2, 1)


def test_la_reprise_s_arrete_a_la_premiere_iteration_invalide():
    a = _a("BB", hist_BB=[0.01, 0.01, 0.9, 0.01])
    assert a.reprendre_depuis_historique()[0] == 1


def test_un_None_dans_l_historique_ARRETE_le_compte_au_lieu_de_LEVER():
    """La version d'origine comparait `_v < tol` sans ecarter les `None` : une
    reprise apres une iteration ou le FORM avait echoue levait `TypeError`.

    Et un `None` ne peut pas compter comme valide : on ne sait pas si elle
    l'etait, et la supposer valide raccourcirait l'enrichissement sur une
    ignorance.
    """
    a = _a("BB", hist_BB=[0.01, None, 0.01])
    assert a.reprendre_depuis_historique()[0] == 1


def test_la_reprise_sur_un_historique_vide_ne_compte_rien():
    a = _a("BB")
    assert a.reprendre_depuis_historique() == (0, 0)


def test_la_reprise_annonce_ce_qu_elle_a_repris():
    messages = []
    a = _a("BB", hist_BB=[0.01, 0.01], tracer=messages.append)
    a.reprendre_depuis_historique()
    assert "RESTART" in "\n".join(messages)
    assert "count_valid_BB=2" in "\n".join(messages)


def test_une_reprise_sans_rien_a_reprendre_reste_muette():
    messages = []
    _a("BB", hist_BB=[0.9], tracer=messages.append).reprendre_depuis_historique()
    assert messages == []


# --------------------------------------------------------------------- #
# le bilan
# --------------------------------------------------------------------- #
def test_le_bilan_designe_le_critere_qui_a_arrete():
    a = _a("at_least_one")
    for _ in range(3):
        a.enregistrer(0.001, 4.0, 4.0)
    bb, bs, both = a.raisons()
    assert bb and bs and both


def test_un_arret_sur_le_budget_ne_revendique_aucun_critere():
    """Trois `False` : ce n'est pas la convergence qui a decide, et le bilan
    doit le dire plutot que d'annoncer une convergence qui n'a pas eu lieu."""
    a = _a("at_least_one", n_max_points=2)
    a.enregistrer(0.9, 4.0, 2.0)
    assert a.raisons() == (False, False, False)


def test_un_compteur_atteint_dans_un_autre_mode_ne_compte_pas():
    """En mode BS, un compteur BB a 3 ne doit pas revendiquer l'arret."""
    a = _a("BS")
    a.n_BB = 5
    assert a.raisons() == (False, False, False)


# --------------------------------------------------------------------- #
# le cout annonce
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("critere,attendu", [
    ("BB", True), ("both", True), ("at_least_one", True),
    ("BS", False), ("autre", False),
])
def test_seuls_les_criteres_qui_l_utilisent_paient_l_encadrement(critere, attendu):
    """L'encadrement `g +/- 2 sigma` coute DEUX FORM+IS de plus par
    iteration. Le mode BS n'en a pas besoin et ne doit pas le payer."""
    assert _a(critere).a_besoin_de_l_encadrement is attendu


@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_les_scripts_ne_tiennent_plus_les_compteurs(script):
    src = open(os.path.join(_REPO, script), encoding="utf-8",
               errors="replace").read()
    assert "_arret_eff.ArretEFF(" in src
    assert "count_valid_BB +=" not in src, (
        "%s tient encore ses propres compteurs" % script)
    assert "_arret.reprendre_depuis_historique()" in src, (
        "%s : la reprise des compteurs n'existait que d'un cote ; elle doit "
        "etre des deux." % script)
