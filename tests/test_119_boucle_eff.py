r"""La boucle d'enrichissement, exercee sur des collaborateurs de papier.

POURQUOI CE FICHIER
--------------------
`run_EFF` etait recopie dans les deux etudes -- 136 lignes de chaque cote, a
l'identique au caractere pres. Aucun test unitaire ne le touchait : il fallait
lancer une etude entiere pour l'exercer, ce que seule la chaine analytique
fait. Trois correctifs de la semaine n'avaient ete portes que d'un cote.

La boucle est desormais dans `_reliability/enrichissement.py`, en un seul
exemplaire. Ce fichier l'exerce sans solveur, sans metamodele et sans figure :
tout ce que l'etude lui fournit est remplace par un objet de papier qui
COMPTE ce qu'on lui demande.

CE QU'IL VERIFIE, ET POURQUOI CHAQUE POINT
-------------------------------------------
1. `max_degree` vient de L'APPEL, pas du fichier d'etude. Une reprise le relit
   dans le dump et ECRASE celui de la configuration ; le figer dans l'objet
   rendrait la boucle sourde a la reprise. C'est le piege que cette extraction
   devait eviter, et le seul qui ne se voie pas sur un run de reference.
2. Les historiques sont remplis EN PLACE. `ArretEFF` recoit ces memes listes ;
   une liste rebindee le laisserait ecrire dans un objet que plus personne ne
   lit -- c'est arrive le 27/08/2026, six cent vingt-trois tests verts ne
   l'ont pas vu.
3. Une boucle qui ne tourne pas rend quand meme un bilan. Deux situations
   parfaitement legitimes y menent : budget deja epuise a la reprise, critere
   deja satisfait.
4. Les deux branches sans figure et sans dump ne levent pas.
"""

import os
import sys

import numpy as np
import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_reliability"), os.path.join(_REPO, "_doe"),
           os.path.join(_REPO, "_lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ot = pytest.importorskip("openturns", reason="la couche etudes n'est pas installee")

import enrichissement as _enr   # noqa: E402


# --------------------------------------------------------------------------- #
# LES COLLABORATEURS DE PAPIER                                                #
# --------------------------------------------------------------------------- #
class _Cfg:
    """Les seuls reglages que la boucle lit."""

    def __init__(self, **kw):
        self.do_HF = False
        self.do_PCK = True          # coupe le tirage adaptatif
        self.restart_enrich_only = False
        self.print_Pf = False
        self.print_EFF_progres = False
        self.epsilon_factor = 2.0
        self.tol_EFF = 0.001
        self.tol_BB = 0.05
        self.tol_BS = 0.05
        self.EFF_criteria = "BS"
        self.n_max_EFF_points = 3
        self.n_batch_EFF = 1
        self.n_workers_DOE = 1
        self.eps_taylor = 0.0
        self.n_max_FORM = 100
        self.tol_FORM = 0.001
        self.n_IS = 1000
        self.cov_IS = 0.05
        self.max_degree = 999       # LE PIEGE : ne doit JAMAIS etre lu
        self.__dict__.update(kw)


class _Journal(list):
    def marquer(self, phase):
        self.append(phase)


class _Critere(ot.OpenTURNSPythonFunction):
    """Le critere EFF : sa valeur est celle que le scenario a posee."""

    def __init__(self, boite):
        super().__init__(2, 1)
        self.boite = boite

    def _exec(self, u):
        return [self.boite["eff"]]


class _Controleur:
    """Le juge, remplace par un compteur.

    Il tient la meme comptabilite que le vrai : `arret.enregistrer` decide
    des compteurs, et `historique_beta` recoit le beta de l'iteration.
    """

    def __init__(self, boite):
        self.boite = boite
        self.etats = []          # tous les `etat` recus, pour les inspecter
        self.amorces = 0

    def beta_et_pf(self, g_ot, label, sign=0, fm=None, etat=None):
        self.etats.append(etat)
        return 4.7, 1e-6

    def encadrement(self, g_ot, sigma_func, label, borner, etat=None,
                    beta_central=None):
        self.etats.append(etat)
        return 0.01, 1e-6, 2e-7, 4e-6

    def amorcer_iteration(self, g_ot, sigma_func, borner, arret, *,
                          n_points, historique_Pf=None, etat=None):
        self.etats.append(etat)
        self.amorces += 1
        if historique_Pf is not None:
            historique_Pf.append({'mid': 1e-6, 'sup': 2e-7, 'inf': 4e-6})
        arret.amorcer(0.01)
        return 0.01

    def mesurer_iteration(self, g_ot, sigma_func, borner, arret, *,
                          n_points, iteration, avec_Pf,
                          historique_Pf=None, historique_beta=None, etat=None):
        self.etats.append(etat)
        beta = 4.7
        if avec_Pf and historique_Pf is not None:
            historique_Pf.append({'mid': 1e-6, 'sup': 2e-7, 'inf': 4e-6})
        arret.enregistrer(0.01, beta,
                          historique_beta[-1] if historique_beta else None)
        if historique_beta is not None:
            historique_beta.append(beta)
        return beta, 1e-6, 0.01


class _Boucle(_enr.BoucleEFF):
    """La vraie boucle, avec un juge de papier."""

    def __init__(self, *a, **kw):
        self.faux_controleur = kw.pop("controleur")
        super().__init__(*a, **kw)

    def _controleur(self):
        return self.faux_controleur


def _montage(cfg=None, boite=None, **surcharges):
    """La boucle, ses historiques et ses compteurs, prets a l'emploi."""
    cfg = cfg or _Cfg()
    boite = boite if boite is not None else {"eff": 1.0}
    hist = {"EFF": [], "BB": [], "BS": [], "Pf": [], "beta_IS": []}
    journal = _Journal()
    controleur = _Controleur(boite)
    trace = []

    payes = []

    def evaluer_un_point(u):
        payes.append(np.asarray(u, float))
        # une fois le plan enrichi, le critere retombe sous la tolerance
        boite["eff"] = boite.get("eff_apres", 1.0)
        return 0.5, np.zeros(2), None

    options = dict(
        journal=journal, historiques=hist,
        points_EFF=lambda g, s, x, y, a: ([np.array([0.5, -1.5])], 1.0),
        fonction_EFF=lambda g, s: _Critere(boite),
        ajuster=lambda g, s, x, y, a: (g, s, x, y, a),
        bornes_surrogate=lambda g, s, signe: g,
        executer_is=lambda modes, ev: None,
        evaluer_un_point=evaluer_un_point,
        tracer=trace.append,
    )
    options.update(surcharges)
    boucle = _Boucle(cfg, 2, controleur=controleur, **options)
    return boucle, hist, journal, controleur, trace, payes


def _plan():
    """Un plan minimal : deux points, deux gradients.

    `yt` est une COLONNE : `evaluer_batch_EFF` empile `[[g]]` dessus. Un
    vecteur plat passe le premier appel et casse au second.
    """
    return (np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[1.0], [0.5]]),
            np.array([[0.1, 0.1], [0.2, 0.2]]))


def _g_et_sigma():
    return (lambda u: [0.05]), (lambda u: 0.02)


# --------------------------------------------------------------------------- #
# 1. LE PIEGE : `max_degree` vient de l'appel                                 #
# --------------------------------------------------------------------------- #
def test_max_degree_vient_de_l_appel_pas_de_la_configuration():
    """Une reprise ECRASE `max_degree` : le figer serait une regression.

    Le fichier d'etude en porte un ; le dump en porte un autre. C'est celui
    du dump qui doit voyager dans l'etat passe au controleur -- sinon le
    reajustement du tirage adaptatif travaille avec le mauvais degre apres
    chaque reprise, sans qu'une ligne ne le dise.
    """
    cfg = _Cfg(max_degree=999)
    boucle, _, _, controleur, _, _ = _montage(cfg=cfg)
    xt, yt, ag = _plan()
    g_ot, sigma = _g_et_sigma()
    boucle.enrichir(g_ot, sigma, xt, yt, ag, max_degree=3)

    degres = {e["max_degree"] for e in controleur.etats if e is not None}
    assert degres == {3}, (
        "le controleur a recu %s : la boucle lit `max_degree` ailleurs que "
        "dans son argument d'appel (999 = celui du fichier d'etude)." % degres)


def test_l_etat_transmis_suit_le_plan_qui_grossit():
    """L'etat n'est pas fige a l'entree : il est relu a chaque mesure."""
    boucle, _, _, controleur, _, _ = _montage(boite={"eff": 1.0, "eff_apres": 0.0})
    xt, yt, ag = _plan()
    g_ot, sigma = _g_et_sigma()
    boucle.enrichir(g_ot, sigma, xt, yt, ag, max_degree=1)
    tailles = [len(e["xt"]) for e in controleur.etats if e is not None]
    assert tailles[0] == 2 and max(tailles) == 3, tailles


# --------------------------------------------------------------------------- #
# 2. LES HISTORIQUES SONT REMPLIS EN PLACE                                     #
# --------------------------------------------------------------------------- #
def test_les_historiques_sont_les_memes_objets_a_la_sortie():
    """Rebinder une liste, c'est ecrire dans un objet que plus personne ne lit.

    L'etude passe ses cinq listes ; le dump de reprise lit CES listes-la. Si
    la boucle en substituait de nouvelles, le dump ecrirait l'etat d'avant le
    run -- le defaut du 27/08/2026, transpose.
    """
    boucle, hist, _, _, _, _ = _montage(boite={"eff": 1.0, "eff_apres": 0.0})
    identites = {k: id(v) for k, v in hist.items()}
    xt, yt, ag = _plan()
    g_ot, sigma = _g_et_sigma()
    boucle.enrichir(g_ot, sigma, xt, yt, ag, max_degree=1)
    assert {k: id(v) for k, v in hist.items()} == identites
    assert hist["EFF"], "l'historique EFF est reste vide : rien n'a ete mesure"
    assert hist["beta_IS"], "le beta du tirage d'importance n'a pas ete range"


def test_le_beta_du_tirage_est_publie_a_chaque_tour():
    """Le dump de reprise est ecrit DANS la boucle : le beta doit y etre a
    jour AVANT chaque sauvegarde, pas seulement a la fin."""
    vus = []
    boucle, hist, _, _, _, _ = _montage(
        boite={"eff": 1.0, "eff_apres": 0.0},
        sauver=lambda x, y, a, xe: vus.append(list(hist["beta_IS"])))
    xt, yt, ag = _plan()
    g_ot, sigma = _g_et_sigma()
    boucle.enrichir(g_ot, sigma, xt, yt, ag, max_degree=1)
    assert vus and vus[0], (
        "le dump du premier tour a vu un historique de beta vide : il aurait "
        "enregistre l'etat d'avant l'iteration.")


# --------------------------------------------------------------------------- #
# 3. UNE BOUCLE QUI NE TOURNE PAS                                              #
# --------------------------------------------------------------------------- #
def test_le_critere_deja_satisfait_ne_fait_aucun_tour():
    """Le metamodele est deja assez bon : zero point paye, un bilan quand meme."""
    boucle, hist, _, _, trace, payes = _montage(boite={"eff": 0.0})
    xt, yt, ag = _plan()
    g_ot, sigma = _g_et_sigma()
    g2, s2, xt2, yt2, ag2, xt_eff = boucle.enrichir(
        g_ot, sigma, xt, yt, ag, max_degree=1)
    assert payes == [], "un point a ete paye alors que le critere etait atteint"
    assert xt_eff == []
    assert len(xt2) == 2
    assert any("[BB informatif final] ratio =" in t for t in trace)


def test_le_budget_epuise_a_la_reprise_ne_fait_aucun_tour():
    """La situation exacte du defaut `UnboundLocalError` : on reprend une
    etude qui a deja consomme ses points."""
    cfg = _Cfg(restart_enrich_only=True, n_max_EFF_points=2)
    deja = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
    boucle, _, _, _, trace, payes = _montage(cfg=cfg, boite={"eff": 1.0})
    xt, yt, ag = _plan()
    g_ot, sigma = _g_et_sigma()
    *_, xt_eff = boucle.enrichir(g_ot, sigma, xt, yt, ag, max_degree=1,
                                 xt_eff_initial=deja)
    assert payes == [], "le budget etait epuise et un point a ete paye"
    assert len(xt_eff) == 2, "les points deja payes ont ete perdus"
    bilan = [t for t in trace if "[BB informatif final]" in t]
    assert len(bilan) == 1, trace


def test_sans_reprise_les_points_deja_payes_ne_sont_pas_repris():
    """`xt_eff_initial` n'a de sens qu'en reprise : un run neuf repart de zero."""
    boucle, _, _, _, _, _ = _montage(boite={"eff": 0.0})
    xt, yt, ag = _plan()
    g_ot, sigma = _g_et_sigma()
    *_, xt_eff = boucle.enrichir(g_ot, sigma, xt, yt, ag, max_degree=1,
                                 xt_eff_initial=[np.array([9.0, 9.0])])
    assert xt_eff == []


# --------------------------------------------------------------------------- #
# 4. LES BRANCHES QUI NE FONT RIEN                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", [_Cfg(do_HF=True), _Cfg()])
def test_aucune_branche_active_rend_le_plan_intact(cfg):
    """`do_HF` : pas de metamodele a enrichir. `g_ot is None` : pas de
    metamodele du tout. Dans les deux cas le plan ressort tel quel."""
    boucle, hist, journal, _, _, payes = _montage(cfg=cfg)
    xt, yt, ag = _plan()
    g_ot = None if not cfg.do_HF else (lambda u: [0.05])
    sortie = boucle.enrichir(g_ot, (lambda u: 0.02), xt, yt, ag, max_degree=1)
    assert sortie[0] is g_ot and sortie[5] == []
    assert sortie[2] is xt
    assert payes == []
    assert journal == [], "le journal a ete estampille pour un run sans EFF"
    assert hist["EFF"] == []


def test_sans_figure_ni_dump_la_boucle_tourne_quand_meme():
    """Les deux collaborateurs facultatifs sont eprouves par `is not None`,
    jamais par leur valeur de verite."""
    cfg = _Cfg(print_EFF_progres=True)
    boucle, _, _, _, _, payes = _montage(cfg=cfg,
                                         boite={"eff": 1.0, "eff_apres": 0.0})
    xt, yt, ag = _plan()
    g_ot, sigma = _g_et_sigma()
    boucle.enrichir(g_ot, sigma, xt, yt, ag, max_degree=1)
    assert len(payes) == 1


def test_la_figure_est_appelee_quand_l_etude_la_demande():
    appels = []
    cfg = _Cfg(print_EFF_progres=True)
    boucle, _, _, _, _, _ = _montage(
        cfg=cfg, boite={"eff": 1.0, "eff_apres": 0.0},
        figure=lambda g, s, x, xe: appels.append(len(x)))
    xt, yt, ag = _plan()
    g_ot, sigma = _g_et_sigma()
    boucle.enrichir(g_ot, sigma, xt, yt, ag, max_degree=1)
    assert appels == [3], appels


def test_la_figure_n_est_pas_appelee_quand_l_etude_n_en_veut_pas():
    appels = []
    boucle, _, _, _, _, _ = _montage(
        boite={"eff": 1.0, "eff_apres": 0.0},
        figure=lambda g, s, x, xe: appels.append(1))
    xt, yt, ag = _plan()
    g_ot, sigma = _g_et_sigma()
    boucle.enrichir(g_ot, sigma, xt, yt, ag, max_degree=1)
    assert appels == []


# --------------------------------------------------------------------------- #
# 5. LE COUT : QUI PAIE L'ENCADREMENT                                          #
# --------------------------------------------------------------------------- #
def test_sans_print_Pf_le_ratio_final_est_paye_une_fois_de_plus():
    """Le bilan a besoin d'un ratio BB. Avec les courbes de Pf, il vient de la
    derniere mesure de la boucle ; sans elles, il faut le payer."""
    boucle, _, _, controleur, _, _ = _montage(boite={"eff": 0.0})
    xt, yt, ag = _plan()
    g_ot, sigma = _g_et_sigma()
    boucle.enrichir(g_ot, sigma, xt, yt, ag, max_degree=1)
    assert controleur.amorces == 0


def test_avec_print_Pf_l_amorce_est_payee_et_rangee():
    cfg = _Cfg(print_Pf=True)
    boucle, hist, _, controleur, _, _ = _montage(cfg=cfg, boite={"eff": 0.0})
    xt, yt, ag = _plan()
    g_ot, sigma = _g_et_sigma()
    boucle.enrichir(g_ot, sigma, xt, yt, ag, max_degree=1)
    assert controleur.amorces == 1
    assert len(hist["Pf"]) == 1, hist["Pf"]
