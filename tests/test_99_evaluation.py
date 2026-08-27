r"""L'unique passage vers le solveur : ce qu'il coute, et ce qu'il ne recalcule pas.

CE QUE CE MODULE CONCENTRE
---------------------------
Jusqu'au 27/08/2026, l'appel au solveur existait en QUATRE exemplaires :
`run_one_SOL` et `run_HF`, dans chacun des deux scripts d'etude. Quatre
copies d'un code qui decide, a chaque point, s'il faut payer 466 s.

Elles avaient deja diverge une fois : jusqu'a la phase 5, `run_HF` et
`run_one_SOL` ne maillaient PAS pareil, et les points qu'elles produisaient
nourrissaient le MEME metamodele -- deux surfaces differentes ajustees comme
s'il n'y en avait qu'une.

Aucune de ces quatre copies n'etait couverte par un test : elles vivaient
dans un bloc `__main__`, donc ni importables, ni isolables. Il n'y a plus
qu'une implementation, et voici ses tests.

CE QUI EST VERIFIE ICI
-----------------------
* la REPRISE : un point deja calcule n'est jamais re-evalue ;
* la sauvegarde APRES CHAQUE POINT, qui rend l'interruption supportable ;
* l'absence de gradient fabrique : `None`, jamais `0.0` ;
* l'ASYMETRIE VOULUE entre le plan (qui ecarte) et l'enrichissement (qui
  leve) ;
* le traitement d'un point non convergent : signale, conserve tant que les
  criteres de Digital Structure ne sont pas fiables.

Tout se teste sans licence : le solveur est remplace par un objet qui compte.
"""

import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _d in (os.path.join(_REPO, "_doe"), os.path.join(_REPO, "solver")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

ot = pytest.importorskip("openturns")

import evaluation as _evaluation                      # noqa: E402
from interface import Evaluation, SolveurNonConverge  # noqa: E402

PARAMS = ["fc", "fy"]


class _SolveurFactice:
    """Un solveur qui compte ce qu'on lui demande, et sait mal se porter."""

    def __init__(self, sain=True, gradient=True):
        self.appels = []
        self.etiquettes = []
        self.sain = sain
        self.gradient = gradient

    def evaluer(self, valeurs, sensibilite=False, etiquette=None):
        self.appels.append(dict(valeurs))
        self.etiquettes.append(etiquette)
        grad = (0.5, -0.25) if self.gradient else (None, None)
        return Evaluation(g=0.125, alpha=1.125, grad_x=grad, sain=self.sain,
                          diagnostic={"solver_status": "HORS_DOMAINE"})


def _evaluateur(solveur, **kw):
    dist = ot.JointDistribution([ot.Normal(40.0, 4.0), ot.Normal(500.0, 30.0)])
    defauts = dict(solveur_pour=lambda _nom: solveur, dist=dist,
                   params_names=PARAMS, tracer=lambda _m: None)
    defauts.update(kw)
    return _evaluation.Evaluateur(**defauts)


# --------------------------------------------------------------------- #
# LA REPRISE -- ce qui fait gagner des heures
# --------------------------------------------------------------------- #
def test_un_point_deja_calcule_n_est_jamais_reevalue():
    """C'est TOUT l'interet de la reprise. Sans ce saut, relancer une etude
    interrompue coute autant que la lancer."""
    s = _SolveurFactice()
    ev = _evaluateur(s)
    SOL = [{"fc": 40.0, "fy": 500.0, "g": -0.3},        # deja connu
           {"fc": 42.0, "fy": 510.0}]
    ev.evaluer_plan(SOL, "modele")
    assert len(s.appels) == 1, "le point deja connu a ete recalcule"
    assert s.appels[0]["fc"] == 42.0
    assert SOL[0]["g"] == -0.3, "la valeur connue a ete ecrasee"


def test_la_reprise_est_annoncee():
    messages = []
    s = _SolveurFactice()
    ev = _evaluateur(s, tracer=messages.append)
    ev.evaluer_plan([{"fc": 40.0, "fy": 500.0, "g": 1.0},
                     {"fc": 41.0, "fy": 501.0}], "modele")
    assert any("deja connus" in m and "SOCP evites" in m for m in messages), (
        "un run qui saute des points doit le dire : c'est la seule facon de "
        "savoir qu'une reprise a bien repris")


def test_l_etat_est_sauve_apres_CHAQUE_point():
    """Sauver a la fin ne servirait a rien : c'est pendant que ca s'interrompt."""
    sauvegardes = []
    s = _SolveurFactice()
    ev = _evaluateur(s, sauver_partiel=lambda SOL, n: sauvegardes.append(n))
    ev.evaluer_plan([{"fc": 40.0, "fy": 500.0},
                     {"fc": 41.0, "fy": 501.0},
                     {"fc": 42.0, "fy": 502.0}], "modele")
    assert sauvegardes == [1, 2, 3], (
        "une sauvegarde apres chaque point, avec le compte a jour : %r"
        % (sauvegardes,))


# --------------------------------------------------------------------- #
# aucun gradient fabrique
# --------------------------------------------------------------------- #
def test_sans_sensibilite_les_gradients_sont_None_et_non_zero():
    """Un gradient a 0,0 affirmerait que l'etat limite est PLAT en ce point,
    et le metamodele l'ajusterait."""
    ev = _evaluateur(_SolveurFactice())
    SOL = [{"fc": 40.0, "fy": 500.0}]
    ev.evaluer_plan(SOL, "modele", sensibilite=False)
    assert SOL[0]["dg_fc"] is None and SOL[0]["dg_fy"] is None


def test_un_solveur_muet_sur_le_gradient_ne_produit_pas_de_zeros():
    ev = _evaluateur(_SolveurFactice(gradient=False))
    SOL = [{"fc": 40.0, "fy": 500.0}]
    ev.evaluer_plan(SOL, "modele", sensibilite=True)
    assert SOL[0]["dg_fc"] is None and SOL[0]["dg_fy"] is None


def test_le_plan_range_les_coordonnees_normees_du_point():
    """`_u` sert au metamodele ; sans lui le point serait inutilisable."""
    ev = _evaluateur(_SolveurFactice())
    SOL = [{"fc": 40.0, "fy": 500.0}]
    ev.evaluer_plan(SOL, "modele", sensibilite=True)
    assert SOL[0]["_u"] == pytest.approx([0.0, 0.0], abs=1e-9), (
        "la moyenne de chaque loi doit tomber en u = 0")


# --------------------------------------------------------------------- #
# L'ASYMETRIE VOULUE
# --------------------------------------------------------------------- #
def test_l_enrichissement_LEVE_la_ou_le_plan_ECARTE():
    """Un point d'enrichissement est demande PARCE QUE l'algorithme le veut
    la. L'ecarter en silence lui ferait reproposer le meme point,
    indefiniment."""
    muet = _SolveurFactice(gradient=False)
    ev = _evaluateur(muet)

    SOL = [{"fc": 40.0, "fy": 500.0}]
    ev.evaluer_plan(SOL, "modele", sensibilite=True)      # ecarte, sans lever
    assert SOL[0]["dg_fc"] is None

    with pytest.raises(ValueError) as err:
        ev.evaluer_en_U([0.5, -0.5])
    assert "exclure_points_sans_gradient" in str(err.value), (
        "le message doit renvoyer au parametre qui gouverne l'AUTRE voie, "
        "sinon l'asymetrie passe pour une incoherence")


def test_l_enrichissement_rend_g_et_les_deux_gradients():
    s = _SolveurFactice()
    ev = _evaluateur(s)
    g, grad_U, grad_X = ev.evaluer_en_U([0.0, 0.0])
    assert g == pytest.approx(0.125)
    assert list(grad_X) == [0.5, -0.25]
    # dg/du = dg/dx * sigma pour une marginale normale
    assert list(grad_U) == pytest.approx([0.5 * 4.0, -0.25 * 30.0])


def test_l_enrichissement_journalise_le_point():
    """Le journal incremental est ce qui reste quand un run est tue."""
    journal = []
    ev = _evaluateur(_SolveurFactice(),
                     journaliser=lambda u, x, g: journal.append((list(u), g)))
    ev.evaluer_en_U([0.0, 0.0])
    assert journal == [([0.0, 0.0], pytest.approx(0.125))]


# --------------------------------------------------------------------- #
# un point non convergent : signale, pas jete
# --------------------------------------------------------------------- #
def test_un_point_non_convergent_est_conserve_et_signale():
    """Les criteres de Digital Structure ne sont pas encore fiables : un point
    ecarte sur cette base serait un appel solveur paye pour rien."""
    messages = []
    ev = _evaluateur(_SolveurFactice(sain=False), tracer=messages.append)
    SOL = [{"fc": 40.0, "fy": 500.0}]
    ev.evaluer_plan(SOL, "modele", sensibilite=True)
    assert SOL[0]["g"] == pytest.approx(0.125), "le point a ete jete"
    texte = "\n".join(messages)
    assert "NON CONVERGE" in texte
    assert "HORS_DOMAINE" in texte, "le statut du solveur doit etre dit"
    assert "alpha=1.125000" in texte, "le multiplicateur doit etre dit"
    assert "conserve" in texte


def test_la_bascule_d_exclusion_fait_lever():
    """Le jour ou les criteres seront fiables, un seul reglage suffit."""
    ev = _evaluateur(_SolveurFactice(sain=False), exclure_non_converges=True,
                     tracer=lambda _m: None)
    with pytest.raises(SolveurNonConverge):
        ev.evaluer_plan([{"fc": 40.0, "fy": 500.0}], "modele")
    with pytest.raises(SolveurNonConverge):
        ev.evaluer_en_U([0.0, 0.0])


# --------------------------------------------------------------------- #
# l'etiquette d'archive
# --------------------------------------------------------------------- #
def test_l_etiquette_numerote_les_appels():
    """Deux points ne doivent pas ecrire dans le meme sous-dossier de
    `SOCP_history` : le second effacerait le premier."""
    ev = _evaluateur(_SolveurFactice())
    a = ev.etiquette("SOL", [40.0, 500.0])
    b = ev.etiquette("SOL", [40.0, 500.0])
    assert a != b and a.startswith("SOL_001") and b.startswith("SOL_002")


def test_l_etiquette_porte_toutes_les_coordonnees():
    """L'original n'ecrivait que `u[0]` et `u[1]` : au-dela de deux variables,
    deux points distincts recevaient le MEME nom d'archive."""
    ev = _evaluation.Evaluateur(solveur_pour=lambda _n: None, dist=None,
                                params_names=["a", "b", "c"],
                                tracer=lambda _m: None)
    e1 = ev.etiquette("HF", [1.0, 2.0, 3.0], u=[0.1, 0.2, 0.3])
    e2 = ev.etiquette("HF", [1.0, 2.0, 3.0], u=[0.1, 0.2, 0.9])
    assert "_u3+0.300" in e1 and "_u3+0.900" in e2
    assert e1.replace("_001", "") != e2.replace("_002", ""), (
        "deux points distincts doivent donner deux archives distinctes")


def test_sans_archivage_aucune_etiquette_n_est_construite():
    """Le compteur ne doit pas avancer pour rien -- et surtout, on ne cree pas
    des dossiers d'archive quand personne n'en a demande."""
    s = _SolveurFactice()
    ev = _evaluateur(s, archiver=False)
    ev.evaluer_plan([{"fc": 40.0, "fy": 500.0}], "modele")
    assert s.etiquettes == [None]
    assert ev.n_appels == 0


# --------------------------------------------------------------------- #
# une seule implementation, et elle reste sans licence
# --------------------------------------------------------------------- #
def test_le_module_ne_connait_pas_Digital_Structure():
    src = open(os.path.join(_REPO, "_doe", "evaluation.py"),
               encoding="utf-8").read()
    for interdit in ("digital_structure", "STRAINS", "CetSOLV", "import schema"):
        assert interdit not in src, (
            "evaluation.py mentionne %r : le solveur lui est PASSE, il ne va "
            "pas le chercher." % interdit)


@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_le_script_ne_construit_plus_l_appel_solveur(script):
    src = open(os.path.join(_REPO, script), encoding="utf-8",
               errors="replace").read()
    assert "_evaluation.Evaluateur(" in src, script
    for parti in ("_etiquette_socp", "_grad_vers_U", "_socp_call_counter"):
        assert parti not in src, (
            "%s porte encore `%s` : ces trois-la appartiennent a "
            "`_doe/evaluation.py`." % (script, parti))
