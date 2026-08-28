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
# --------------------------------------------------------------------- #
# MESURER N'EST PAS DECIDER -- decision d'Agnes, 28/08/2026
#
# Les deux asymetries que ces tests figeaient ont ete tranchees :
#   1. tout ce qui est PAYE est ENREGISTRE ;
#   2. les trois compteurs sont tenus dans TOUS les modes.
# Ce qui reste commande par le critere, et lui seul : l'ARRET.
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("critere", ["BB", "BS", "both", "at_least_one"])
def test_les_deux_historiques_sont_remplis_quel_que_soit_le_critere(critere):
    """Avant, chaque mode ne remplissait que le sien : la courbe de
    convergence de fin de run montrait des choses differentes selon le
    critere choisi."""
    a = _a(critere)
    a.enregistrer(0.02, 4.0, 4.1)
    assert a.hist_BB == [0.02]
    assert len(a.hist_BS) == 1 and a.hist_BS[0] is not None


@pytest.mark.parametrize("critere", ["BB", "BS", "both", "at_least_one"])
def test_les_trois_compteurs_sont_tenus_quel_que_soit_le_critere(critere):
    """Un run en mode `both` affichait `count_valid_BB=0 count_valid_BS=0`
    alors que les deux criteres etaient satisfaits a chaque iteration."""
    a = _a(critere)
    for _ in range(3):
        a.enregistrer(0.001, 4.0, 4.0)
    assert a.n_BB == 3 and a.n_BS == 3 and a.n_both == 3


def test_un_ratio_BB_non_paye_reste_None_et_ne_compte_pas():
    """Le ratio BB demande un encadrement `g +/- 2 sigma` -- trois FORM+IS
    de plus par iteration. Quand il n'est pas paye il vaut None : on
    l'enregistre tel quel, on ne le fabrique pas."""
    a = _a("BS")
    a.enregistrer(None, 4.0, 4.02)      # ratio BS = 0.005, sous TOL_BS
    assert a.hist_BB == [None]
    assert a.n_BB == 0
    assert a.n_both == 0, "un ratio absent ne peut pas satisfaire `both`"
    assert a.n_BS == 1, "le ratio BS, lui, ne coute rien de plus que beta"


def test_le_ratio_BS_est_mesure_meme_en_mode_BB():
    """Il ne coute rien de plus que `beta`, calcule a chaque iteration de
    toute facon. Ne pas l'enregistrer, c'etait jeter une mesure gratuite."""
    a = _a("BB")
    a.enregistrer(0.5, 4.0, 4.02)
    assert a.hist_BS[0] == pytest.approx(abs(4.0 - 4.02) / 4.0)


def test_le_journal_dit_quel_critere_est_en_vigueur():
    """Les libelles sont ceux d'origine, pour que deux journaux restent
    comparables."""
    for critere, attendu in (("BS", "[N=10]"), ("both", "[N=10 both]"),
                             ("at_least_one", "[N=10 alo]"),
                             ("BB", "[N=10 bb]")):
        j = _Journal()
        a = _arret.ArretEFF(critere, 0.05, 0.05, 10, 0.001, tracer=j)
        a.enregistrer(0.02, 4.0, 4.1, prefixe="N=10")
        assert any(attendu in m for m in j), (critere, list(j))


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


@pytest.mark.parametrize("critere", ["BB", "BS", "both", "at_least_one"])
def test_l_amorce_est_rangee_quel_que_soit_le_critere(critere):
    """Elle n'est appelee que si l'encadrement initial a ete calcule -- donc
    si ce ratio a ete PAYE. En mode `BS` il etait jete."""
    a = _a(critere)
    a.amorcer(0.01)
    assert a.hist_BB == [0.01] and a.n_BB == 1


def test_l_amorce_n_a_pas_de_contrepartie_BS():
    """A la premiere mesure il n'existe pas d'iteration precedente, donc pas
    de ratio BS. `hist_BB` porte alors UNE entree de plus -- c'est pourquoi
    les deux courbes de convergence sont alignees a DROITE."""
    a = _a("at_least_one")
    a.amorcer(0.01)
    a.enregistrer(0.02, 4.0, 4.1)
    assert len(a.hist_BB) == len(a.hist_BS) + 1


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
    assert "_arret_eff.ArretEFF.pour_un_run(" in src
    assert "count_valid_BB +=" not in src, (
        "%s tient encore ses propres compteurs" % script)
    # La reprise des compteurs n'existait que d'un cote ; elle est desormais
    # DANS la fabrique, donc des deux par construction -- une etude ne peut
    # plus l'oublier. Le controle porte sur la fabrique elle-meme.
    assert "reprendre_depuis_historique" not in src, (
        "%s : la reprise des compteurs est ressortie dans l'etude ; elle "
        "appartient a `ArretEFF.pour_un_run`." % script)


# --------------------------------------------------------------------------- #
# LE BILAN DE FIN D'ENRICHISSEMENT (extrait de `run_EFF` le 27/08/2026)        #
# --------------------------------------------------------------------------- #
class _Journal(list):
    def __call__(self, message):
        self.append(message)


def _arret_avec(critere, **kw):
    j = _Journal()
    a = _arret.ArretEFF(critere, kw.pop("tol_BB", 0.05), kw.pop("tol_BS", 0.05),
                        kw.pop("n_max_points", 10), kw.pop("tol_EFF", 0.001),
                        hist_BB=kw.pop("hist_BB", None),
                        hist_BS=kw.pop("hist_BS", None), tracer=j)
    for nom, val in kw.items():
        setattr(a, nom, val)
    return a, j


def test_un_EFF_sous_la_tolerance_est_la_raison_qui_prime():
    a, j = _arret_avec("BB", n_BB=9)
    assert a.bilan(0.0005, 4) == "EFF"
    assert "EFF converge [EFF]" in j[0]
    assert "(4 point(s) ajoutes)" in j[0]


def test_un_EFF_negatif_compte_par_sa_valeur_absolue():
    a, _ = _arret_avec("BB")
    assert a.bilan(-0.0005, 1) == "EFF"


def test_le_critere_BB_est_nomme_quand_c_est_lui_qui_a_arrete():
    a, j = _arret_avec("BB", n_BB=3)
    assert a.bilan(0.5, 7) == "BB (3 iter valides)"
    assert "count_valid_BB=3" in j[0]


def test_un_budget_epuise_ne_se_fait_pas_passer_pour_une_convergence():
    """`?` n'est pas un defaut d'affichage : c'est un run arrete par le
    plafond de points. Un plafond atteint ne prouve rien, et le bilan doit
    le distinguer d'une convergence."""
    a, j = _arret_avec("BB", n_BB=1)
    assert a.bilan(0.5, 10) == "?"
    assert "EFF converge [?]" in j[0]


def test_le_bilan_dit_ce_qui_s_est_passe_pas_seulement_ce_qui_a_decide():
    """En mode `both`, le bilan affichait `count_valid_BB=0
    count_valid_BS=0` alors que les deux criteres pouvaient etre satisfaits
    a chaque iteration -- un lecteur presse y lisait le contraire de la
    verite. Les trois compteurs sont desormais tenus."""
    a = _a("both")
    for _ in range(2):
        a.enregistrer(0.001, 4.0, 4.0)
    j = _Journal()
    a.tracer = j
    assert a.bilan(0.5, 5) == "both (2 iter valides)"
    assert "count_valid_BB=2" in j[0] and "count_valid_BS=2" in j[0]
    assert "count_valid_both=2" in j[0]


def test_un_historique_vide_ne_s_imprime_pas():
    """Le cas ne se produit plus par le jeu des criteres -- les deux
    historiques sont remplis ensemble -- mais un run arrete avant la
    premiere mesure en laisse un vide. La ligne ne doit pas s'imprimer."""
    a, j = _arret_avec("BB", hist_BB=[0.1, 0.02])
    a.bilan(0.5, 3)
    assert any("[historique ratio BB]" in m for m in j)
    assert not any("[historique ratio BS]" in m for m in j)


def test_les_trous_de_l_historique_s_impriment_comme_trous():
    """Un round sans ratio vaut `None`. L'arrondir a 0.0 dirait « critere
    satisfait » sur la courbe de convergence."""
    a, j = _arret_avec("at_least_one", hist_BB=[None, 0.123456], hist_BS=[0.9])
    a.bilan(0.5, 2)
    ligne = [m for m in j if "[historique ratio BB]" in m][0]
    assert "[None, 0.1235]" in ligne
    assert "tol=0.05" in ligne
    assert any("[historique ratio BS]" in m for m in j)


# --------------------------------------------------------------------------- #
# LE CONTRAT DE PARTAGE DES HISTORIQUES (27/08/2026)                           #
# --------------------------------------------------------------------------- #
def test_les_ratios_atterrissent_dans_la_liste_de_l_appelant():
    """`ArretEFF` n'a pas d'historique a lui : il ECRIT dans celui qu'on lui
    passe. C'est ce qui permet au dump de reprise et au bilan de fin de run
    de lire les memes ratios."""
    hist = []
    a = _arret.ArretEFF("BB", 0.05, 0.05, 10, 0.001, hist_BB=hist,
                        tracer=lambda m: None)
    a.enregistrer(0.02, 4.7, 4.6, prefixe="N=10")
    assert hist == [0.02], "les ratios ne sont pas arrives chez l'appelant"


def test_vider_en_place_preserve_le_partage():
    """La bonne facon de repartir de zero."""
    hist = [0.9, 0.8]
    a = _arret.ArretEFF("BB", 0.05, 0.05, 10, 0.001, hist_BB=hist,
                        tracer=lambda m: None)
    del hist[:]
    a.enregistrer(0.02, 4.7, 4.6, prefixe="N=10")
    assert hist == [0.02]
    assert a.hist_BB is hist


def test_rebinder_casse_le_partage_EN_SILENCE():
    """LE DEFAUT DU 27/08/2026, reproduit.

    `hist = []` cree une NOUVELLE liste. L'objet garde l'ancienne, y ecrit
    fidelement, et personne ne la relit : les ratios disparaissent du bilan
    et du dump sans la moindre erreur. 623 tests verts ne l'ont pas vu --
    seule la comparaison ligne a ligne du journal analytique l'a attrape.

    Ce test n'exige rien du code de production : il FIGE le mecanisme, pour
    que la raison du `del l[:]` reste lisible.
    """
    hist = [0.9]
    a = _arret.ArretEFF("BB", 0.05, 0.05, 10, 0.001, hist_BB=hist,
                        tracer=lambda m: None)
    hist = []                       # <- le rebinding fautif
    a.enregistrer(0.02, 4.7, 4.6, prefixe="N=10")
    assert hist == [], "la liste que l'appelant relit est restee vide"
    assert a.hist_BB == [0.9, 0.02], "l'objet a ecrit dans la liste abandonnee"


@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_aucune_etude_ne_rebinde_ses_historiques(script):
    """Le garde-fou. Un `_eff_history_XX = []` reintroduirait exactement le
    defaut ci-dessus -- sans erreur, sans test rouge."""
    import os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo, script), encoding="utf-8") as fh:
        lignes = [l.strip() for l in fh]
    fautifs = [l for l in lignes
               if l.startswith("_eff_history_") and l.endswith("= []")]
    assert not fautifs, (
        "%s : historique remis a zero par rebinding (%s). Vider en place "
        "avec `del l[:]` -- voir le contrat dans `ArretEFF`."
        % (script, fautifs))


# --------------------------------------------------------------------- #
# LA REPRISE DU TROISIEME COMPTEUR (28/08/2026)
# --------------------------------------------------------------------- #
def test_une_reprise_en_mode_both_ne_repart_pas_de_zero():
    """Sans cela, une reprise en mode `both` redemandait ses deux iterations
    valides -- et sur le Moulin Blanc, une iteration EFF vaut un appel
    solveur de 466 s. C'est le defaut que `reprendre_depuis_historique`
    ferme pour BB et BS depuis le 26/08 ; le troisieme compteur y echappait.
    """
    a = _a("both", hist_BB=[0.5, 0.001, 0.002], hist_BS=[0.5, 0.001, 0.002])
    a.reprendre_depuis_historique()
    assert a.n_both == 2, "les deux dernieres iterations valaient pour both"
    assert a.n_BB == 2 and a.n_BS == 2


def test_une_iteration_ou_UN_SEUL_ratio_tient_ne_compte_pas_pour_both():
    a = _a("both", hist_BB=[0.001, 0.001], hist_BS=[0.001, 0.5])
    a.reprendre_depuis_historique()
    assert a.n_both == 0
    assert a.n_BB == 2 and a.n_BS == 0


def test_un_trou_arrete_le_compte_conjoint():
    """Un `None` -- FORM en echec -- ne se suppose pas valide : le supposer
    raccourcirait l'enrichissement sur une ignorance."""
    a = _a("both", hist_BB=[0.001, None, 0.001], hist_BS=[0.001, 0.001, 0.001])
    a.reprendre_depuis_historique()
    assert a.n_both == 1


def test_l_amorce_ne_decale_pas_le_compte_conjoint():
    """`hist_BB` porte une entree de plus -- celle du plan initial. Les deux
    historiques sont donc apparies A DROITE, sinon le compte conjoint
    comparerait l'iteration k de l'un a l'iteration k+1 de l'autre."""
    a = _a("both", hist_BB=[0.9, 0.001, 0.001], hist_BS=[0.001, 0.001])
    a.reprendre_depuis_historique()
    assert a.n_both == 2


def test_la_reprise_annonce_les_trois_compteurs():
    j = _Journal()
    a = _arret.ArretEFF("both", TOL_BB, TOL_BS, 15, 0.001,
                        hist_BB=[0.001, 0.001], hist_BS=[0.001, 0.001],
                        tracer=j)
    a.reprendre_depuis_historique()
    assert any("count_valid_both=2" in m for m in j), list(j)


# --------------------------------------------------------------------- #
# LA FABRIQUE : les trois gestes de preparation, dans le bon ordre
# --------------------------------------------------------------------- #
def test_un_run_neuf_repart_d_historiques_vides():
    bb, bs, pf = [0.9], [0.8], [{"mid": 1.0}]
    a = _arret.ArretEFF.pour_un_run("BB", TOL_BB, TOL_BS, 15, 0.001,
                                    hist_BB=bb, hist_BS=bs, hist_Pf=pf,
                                    reprise=False, tracer=lambda _m: None)
    assert bb == [] and bs == [] and pf == []
    assert a.n_BB == 0 and a.n_BS == 0 and a.n_both == 0


def test_le_vidage_se_fait_EN_PLACE():
    """Les listes de l'appelant sont celles que l'objet remplira et que le
    dump de reprise relira. Les remplacer par des neuves romprait le
    partage -- le defaut du 27/08."""
    bb, bs, pf = [0.9], [0.8], [{"mid": 1.0}]
    a = _arret.ArretEFF.pour_un_run("BB", TOL_BB, TOL_BS, 15, 0.001,
                                    hist_BB=bb, hist_BS=bs, hist_Pf=pf,
                                    reprise=False, tracer=lambda _m: None)
    assert a.hist_BB is bb and a.hist_BS is bs
    a.enregistrer(0.001, 4.0, 4.0)
    assert bb == [0.001], "les ratios n'arrivent pas chez l'appelant"


def test_une_reprise_garde_ses_historiques_et_recompte():
    """Elle ne vide rien -- ce serait jeter ce que le dump a rendu -- et
    elle recompte ses iterations valides consecutives."""
    bb, bs, pf = [0.9, 0.001, 0.002], [0.9, 0.001, 0.002], [{"mid": 1.0}]
    a = _arret.ArretEFF.pour_un_run("at_least_one", TOL_BB, TOL_BS, 15, 0.001,
                                    hist_BB=bb, hist_BS=bs, hist_Pf=pf,
                                    reprise=True, tracer=lambda _m: None)
    assert len(bb) == 3 and len(pf) == 1, "une reprise ne vide pas"
    assert a.n_BB == 2 and a.n_BS == 2 and a.n_both == 2


def test_la_reprise_est_dans_la_fabrique_donc_jamais_oubliee():
    """Elle n'existait que dans une des deux etudes. La mettre ici, c'est la
    rendre impossible a oublier."""
    bb = [0.001, 0.001, 0.001]
    a = _arret.ArretEFF.pour_un_run("BB", TOL_BB, TOL_BS, 15, 0.001,
                                    hist_BB=bb, hist_BS=[], hist_Pf=[],
                                    reprise=True, tracer=lambda _m: None)
    assert a.n_BB == 3, "les compteurs n'ont pas ete repris"


def test_un_run_neuf_ne_recompte_rien():
    """Il n'y a rien a reprendre, et les historiques viennent d'etre vides."""
    j = _Journal()
    bb = [0.001, 0.001, 0.001]
    a = _arret.ArretEFF.pour_un_run("BB", TOL_BB, TOL_BS, 15, 0.001,
                                    hist_BB=bb, hist_BS=[], hist_Pf=[],
                                    reprise=False, tracer=j)
    assert a.n_BB == 0
    assert not any("RESTART" in m for m in j)
