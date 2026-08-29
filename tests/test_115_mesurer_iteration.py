r"""Juger une iteration d'enrichissement, et ce qu'on accepte de payer.

CE QUE C'ETAIT
---------------
Dix-sept lignes au milieu de la boucle de `run_EFF`, recopiees dans les deux
etudes. Elles portaient une DECISION DE COUT ecrite en deux conditions qui
se recouvrent :

    if print_Pf:
        ratio, pf_mid, pf_sup, pf_inf = encadrement(...)
        list_Pf.append(...)
    if arret.a_besoin_de_l_encadrement and not print_Pf:
        ratio, _, _, _ = encadrement(...)

Il fallait deduire du `and not print_Pf` que l'encadrement n'etait paye
qu'une fois. C'est desormais un seul `or`.

CE QUE COUTE L'ENCADREMENT
---------------------------
`g +/- 2 sigma` demande DEUX FORM+IS de plus par iteration -- le central est
reutilise via `beta_central`, ce qui economise le troisieme. Deux raisons de
le payer : le critere d'arret en a besoin (`BB`, `both`, `at_least_one`), ou
`print_Pf` reclame les courbes de Pf. Une seule facture quand les deux
tiennent.

Ces tests exercent les QUATRE combinaisons, sans solveur ni OpenTURNS.
"""

import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_reliability"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_controle = pytest.importorskip("controle", reason="FORM + tirage, OpenTURNS")
import arret as _arret                                  # noqa: E402


class _Journal(list):
    def __call__(self, message):
        self.append(message)


class _Controleur(_controle.ControleurFORM):
    """Le vrai controleur, dont on remplace les deux mesures couteuses.

    `mesurer_iteration` est ce qu'on teste ; `beta_et_pf` et `encadrement`
    sont ce qu'elle orchestre, et chacun vaut un a trois FORM+IS.
    """

    def __init__(self, beta=4.7, pf=1e-6, ratio=0.02, tracer=None):
        self.tracer = tracer or (lambda _m: None)
        self._beta, self._pf, self._ratio = beta, pf, ratio
        self.appels_beta = 0
        self.appels_encadrement = 0
        self.etiquettes = []

    def beta_et_pf(self, g_ot, label, sign=0, fm=None, etat=None):
        self.appels_beta += 1
        self.etiquettes.append(label)
        return self._beta, self._pf

    def encadrement(self, g_ot, sigma_func, label, borner, etat=None,
                    beta_central=None):
        self.appels_encadrement += 1
        self.etiquettes.append(label)
        assert beta_central is not None, (
            "le FORM+IS central doit etre REUTILISE, pas recalcule -- c'est "
            "un tiers du cout de l'encadrement")
        return self._ratio, 1e-6, 2e-6, 5e-7


def _mesurer(critere, avec_Pf, **kw):
    c = _Controleur(**kw)
    a = _arret.ArretEFF(critere, 0.05, 0.05, 15, 0.001, tracer=lambda _m: None)
    pf_hist, beta_hist = [], []
    beta, pf, ratio = c.mesurer_iteration(
        object(), object(), object(), a, n_points=13, iteration=3,
        avec_Pf=avec_Pf, historique_Pf=pf_hist, historique_beta=beta_hist)
    return c, a, (beta, pf, ratio), pf_hist, beta_hist


# --------------------------------------------------------------------------- #
# LA DECISION DE COUT : LES QUATRE COMBINAISONS                                #
# --------------------------------------------------------------------------- #
def test_ni_le_critere_ni_les_courbes_ne_le_demandent_on_ne_paie_pas():
    """`BS` n'a pas besoin de l'encadrement, et sans `print_Pf` personne ne
    le reclame : deux FORM+IS economises a chaque iteration."""
    c, _, (_, _, ratio), pf_hist, _ = _mesurer("BS", avec_Pf=False)
    assert c.appels_encadrement == 0
    assert ratio is None
    assert pf_hist == []


def test_le_critere_le_demande_on_paie_une_fois():
    c, _, (_, _, ratio), pf_hist, _ = _mesurer("at_least_one", avec_Pf=False)
    assert c.appels_encadrement == 1
    assert ratio == 0.02
    assert pf_hist == [], "sans `print_Pf`, le triplet de Pf n'est pas range"


def test_les_courbes_le_demandent_on_paie_une_fois():
    """`BS` avec `print_Pf` : le critere n'en a pas besoin, les courbes si.
    Et le ratio obtenu est enregistre -- tout ce qui est paye est
    enregistre."""
    c, a, (_, _, ratio), pf_hist, _ = _mesurer("BS", avec_Pf=True)
    assert c.appels_encadrement == 1
    assert ratio == 0.02
    assert pf_hist == [{"mid": 1e-6, "sup": 2e-6, "inf": 5e-7}]
    assert a.hist_BB == [0.02]


def test_les_deux_le_demandent_on_ne_paie_QU_UNE_FOIS():
    """Le point du changement : l'ecriture d'origine, en deux `if` qui se
    recouvrent, obligeait a deduire qu'elle ne payait pas deux fois."""
    c, _, _, pf_hist, _ = _mesurer("at_least_one", avec_Pf=True)
    assert c.appels_encadrement == 1
    assert len(pf_hist) == 1


# --------------------------------------------------------------------------- #
# CE QUI EST MESURE, ET CE QUI EST RANGE                                       #
# --------------------------------------------------------------------------- #
def test_beta_est_mesure_une_seule_fois_par_iteration():
    """Le FORM+IS central. L'encadrement le REUTILISE au lieu de le
    refaire -- verifie par l'assertion sur `beta_central`."""
    c, _, (beta, pf, _), _, _ = _mesurer("at_least_one", avec_Pf=True)
    assert c.appels_beta == 1
    assert (beta, pf) == (4.7, 1e-6)


def test_les_etiquettes_du_journal_sont_celles_d_origine():
    """Deux journaux de runs differents doivent rester comparables."""
    c, _, _, _, _ = _mesurer("at_least_one", avec_Pf=True)
    assert c.etiquettes == ["N=13 mu conv", "N=13 iter 3"]


def test_beta_rejoint_son_historique():
    _, _, _, _, beta_hist = _mesurer("BS", avec_Pf=False)
    assert beta_hist == [4.7]


def test_un_FORM_en_echec_ne_range_pas_de_beta():
    """`beta = None` veut dire que le FORM n'a pas converge. Ranger un
    `None` dans l'historique de beta ferait mentir la courbe."""
    _, _, (beta, _, _), _, beta_hist = _mesurer("BS", avec_Pf=False, beta=None)
    assert beta is None
    assert beta_hist == []


def test_l_iteration_precedente_sert_de_reference_au_critere_BS():
    """`enregistrer` a besoin du beta d'AVANT pour mesurer la stabilite. Il
    vient de l'historique, pas d'une variable capturee."""
    c = _Controleur(beta=4.70)
    a = _arret.ArretEFF("BS", 0.05, 0.05, 15, 0.001, tracer=lambda _m: None)
    beta_hist = [4.68]
    c.mesurer_iteration(object(), object(), object(), a, n_points=13,
                        iteration=3, avec_Pf=False, historique_beta=beta_hist)
    assert a.hist_BS[0] == pytest.approx(abs(4.70 - 4.68) / 4.70)
    assert beta_hist == [4.68, 4.70]


def test_la_premiere_iteration_n_a_pas_de_reference():
    c = _Controleur()
    a = _arret.ArretEFF("BS", 0.05, 0.05, 15, 0.001, tracer=lambda _m: None)
    c.mesurer_iteration(object(), object(), object(), a, n_points=13,
                        iteration=1, avec_Pf=False, historique_beta=[])
    assert a.hist_BS == [None], "aucun beta precedent : le ratio n'existe pas"


# --------------------------------------------------------------------------- #
# CE QUI NE DOIT PLUS ETRE DANS LES ETUDES                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_les_etudes_ne_decident_plus_du_cout_de_l_encadrement(script):
    import io
    s = io.open(os.path.join(_REPO, script), encoding="utf-8",
                errors="replace").read()
    for interdit in ("a_besoin_de_l_encadrement and not print_Pf",
                     "import time as _t_mod", "[TIMING _form_is_iter]"):
        assert interdit not in s, (
            "%s : la conduite d'une iteration est revenue dans l'etude (%r)."
            % (script, interdit))
    assert "mesurer_iteration" not in s, (
        "%s : la conduite d'une iteration est revenue dans l'etude ; elle "
        "appartient a `_reliability/enrichissement.py`." % script)


def test_la_boucle_conduit_l_iteration():
    """Le pendant positif, sur le seul appelant qui reste."""
    import io
    s = io.open(os.path.join(_REPO, "_reliability", "enrichissement.py"),
                encoding="utf-8", errors="replace").read()
    assert "controleur.mesurer_iteration(" in s
