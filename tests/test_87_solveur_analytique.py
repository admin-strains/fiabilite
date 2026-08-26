"""
Le solveur analytique dit-il la meme chose que l'oracle independant ?

PHASE 5. `solver/analytique.py` implemente le meme contrat que
`solver/digital_structure.py` sur un etat limite ferme. C'est ce qui permet
d'exercer la chaine entiere -- plan d'experiences, metamodele, enrichissement
EFF, FORM multimodal, tirage d'importance -- sans licence, sans GPU, et de
maniere REPRODUCTIBLE : la mesure du 25/08/2026 a montre que la meme chaine
sur Digital Structure s'ecarte de 12,3 % d'un run a l'autre.

POURQUOI DEUX IMPLEMENTATIONS DE LA MEME FORMULE
------------------------------------------------
`tests/reference/limit_states.py:FlexionLS` existait avant, comme oracle du
harness : il travaille en espace standard U et calcule `beta` par une
minimisation scalaire a 1e-12. `solver/analytique.py` travaille en espace
physique X, lit sa geometrie dans le `.ds`, et respecte le contrat du
solveur. Les deux sont ecrits separement, a partir de sources differentes --
leur accord est donc une verification, pas une tautologie.

Ces tests ne demandent ni Digital Structure ni OpenTURNS.
"""

import os
import sys

import numpy as np
import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
for p in (os.path.join(REPO, "solver"), os.path.join(REPO, "_model"), TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from reference.limit_states import FlexionLS                  # noqa: E402

MODELE = os.path.join(r"C:\workspace\storage\admin\SF", "test_pure_flexion.ds")
#: beta exact de l'etat limite analytique sur CETTE section, obtenu par
#: minimisation scalaire a 1e-12 -- ni metamodele, ni FORM.
BETA_EXACT = 4.772568944992476


def _solveur():
    import analytique                                         # noqa: PLC0415
    if not os.path.isdir(MODELE):
        pytest.skip("modele %s absent de ce poste" % MODELE)
    return analytique.SolveurAnalytique(MODELE, ("fc", "fy"))


def _oracle(section):
    """`FlexionLS` cale sur la meme section que le solveur."""
    from lois import SIGMA_ACIER_JCSS                         # noqa: PLC0415
    ls = FlexionLS(b=section["b"], d=section["d"], L=section["L"], F=section["F"],
                   gamma_c=section["gamma_c"], gamma_s=section["gamma_s"],
                   fcm=48.0, cov_fc=0.12, fym=550.0, sig_fy=SIGMA_ACIER_JCSS,
                   phi_mm=1.0, n_bars=1)
    # la section reelle porte 24 barres HA32 : on impose As et Med plutot que
    # de les faire redecouvrir a l'oracle
    ls.As, ls.Med = section["As"], section["Med"]
    ls.A = section["As"] * section["d"] / section["gamma_s"]
    ls.B = -section["As"] ** 2 * section["gamma_c"] / (2 * section["b"] * section["gamma_s"] ** 2)
    ls.C = -section["Med"]
    return ls


# --------------------------------------------------------------------------- #
# La geometrie vient bien du modele                                           #
# --------------------------------------------------------------------------- #
def test_la_section_est_lue_dans_le_modele_sans_digital_structure():
    """`dsCad.txt` et `dsLoad.txt` sont du texte : les deux solveurs peuvent
    donc partager la meme section, ce qui rend la comparaison legitime."""
    s = _solveur().section
    assert s["b"] == pytest.approx(0.8) and s["h"] == pytest.approx(0.8)
    assert s["L"] == pytest.approx(5.0)
    assert s["n_bars"] == 24                     # 3 lits de 8 HA32
    assert s["d"] == pytest.approx(0.691, abs=1e-3)
    assert s["Med"] == pytest.approx(s["F"] * s["L"])


def test_les_constantes_sont_celles_de_l_oracle():
    s = _solveur()
    ls = _oracle(s.section)
    assert s.A == pytest.approx(ls.A, rel=1e-15)
    assert s.B == pytest.approx(ls.B, rel=1e-15)
    assert s.Med == pytest.approx(ls.Med, rel=1e-15)


# --------------------------------------------------------------------------- #
# Le test qui compte : deux implementations independantes                     #
# --------------------------------------------------------------------------- #
def test_g_coincide_avec_l_oracle_sur_tout_le_domaine():
    """81 points couvrant [-4, 4]^2 en espace standard. Deux implementations
    ecrites separement doivent rendre le meme nombre."""
    s = _solveur()
    ls = _oracle(s.section)
    pire = 0.0
    for u1 in np.linspace(-4, 4, 9):
        for u2 in np.linspace(-4, 4, 9):
            U = np.array([[u1, u2]])
            x = ls.u_to_x(U)[0]
            obtenu = s.evaluer({"fc": float(x[0]), "fy": float(x[1])}).g
            pire = max(pire, abs(obtenu - float(ls.g(U)[0])))
    assert pire < 1e-14, "ecart max sur g = %.3e" % pire


def test_le_gradient_coincide_apres_passage_en_espace_standard():
    """Le solveur rend dg/dx, l'oracle dg/du : c'est la frontiere choisie --
    la transformation isoprobabiliste appartient a la loi jointe, pas au
    maillage. Les deux doivent coincider une fois la chaine appliquee."""
    s = _solveur()
    ls = _oracle(s.section)
    pire = 0.0
    for u1 in np.linspace(-4, 4, 9):
        for u2 in np.linspace(-4, 4, 9):
            U = np.array([[u1, u2]])
            x = ls.u_to_x(U)[0]
            gx = np.array(s.evaluer({"fc": float(x[0]), "fy": float(x[1])}).grad_x, float)
            en_u = np.array([gx[0] * ls.sig_ln * x[0], gx[1] * ls.sig_fy])
            pire = max(pire, float(np.max(np.abs(ls.grad(U)[0] - en_u))))
    assert pire < 1e-14, "ecart max sur le gradient = %.3e" % pire


def test_beta_exact_de_reference():
    """La cible que la chaine complete doit retrouver. Obtenue sans
    metamodele et sans FORM : la courbe g=0 est parametree puis ||u|| minimise
    par Brent a 1e-12."""
    assert _oracle(_solveur().section).beta_exact() == pytest.approx(BETA_EXACT, rel=1e-12)


# --------------------------------------------------------------------------- #
# Le contrat, cote analytique                                                 #
# --------------------------------------------------------------------------- #
def test_le_solveur_analytique_accepte_la_signature_du_solveur_reel():
    """Un contrat n'a d'interet que si les deux implementations sont
    interchangeables SANS que l'appelant sache a qui il parle. Les arguments
    de maillage sont acceptes et ignores, pas refuses."""
    import analytique                                         # noqa: PLC0415
    if not os.path.isdir(MODELE):
        pytest.skip("modele absent")
    s = analytique.SolveurAnalytique(
        chemin_ds=MODELE, params_names=("fc", "fy"), dossier_etude="/nulle/part",
        regions=[{"param": "X", "region_key": "fc"}], global_size=0.007,
        geo_min_approx=35, archiver=True)
    assert s.evaluer({"fc": 48.0, "fy": 550.0}).sain


def test_une_variable_hors_forme_fermee_est_refusee_a_la_construction():
    """Mieux vaut refuser tot que rendre un chiffre faux : la forme fermee ne
    connait que fc et fy."""
    import analytique                                         # noqa: PLC0415
    if not os.path.isdir(MODELE):
        pytest.skip("modele absent")
    with pytest.raises(ValueError, match="F"):
        analytique.SolveurAnalytique(MODELE, ("fc", "fy", "F"))


def test_le_solveur_compte_ses_appels():
    s = _solveur()
    assert s.nb_appels == 0
    for _ in range(3):
        s.evaluer({"fc": 48.0, "fy": 550.0})
    assert s.nb_appels == 3


def test_hors_du_domaine_plastifie_le_point_est_declare_non_sain():
    """La branche « beton ecrase » n'est pas implementee. Le solveur le DIT au
    lieu de rendre un moment resistant surestime -- c'est exactement le defaut
    que la phase 5 corrige cote Digital Structure, il n'y a pas de raison de
    l'introduire ici."""
    from interface import SolveurNonConverge                  # noqa: PLC0415
    s = _solveur()
    # beton tres faible et acier tres resistant : l'axe neutre descend sous le
    # pivot, la formule des aciers plastifies ne gouverne plus
    ev = s.evaluer({"fc": 8.0, "fy": 900.0})
    assert not ev.sain
    assert ev.diagnostic["solver_status"] == "HORS_DOMAINE"
    assert ev.diagnostic["x_comprime"] > ev.diagnostic["x_limite"]
    with pytest.raises(SolveurNonConverge):
        ev.exige_sain("point du DOE")


def test_un_appel_est_bien_moins_cher_qu_un_socp():
    """La raison d'etre de cette implementation : mille appels doivent couter
    moins qu'un seul appel a Digital Structure (~10 s sur ce modele)."""
    import time                                               # noqa: PLC0415
    s = _solveur()
    t0 = time.perf_counter()
    for i in range(1000):
        s.evaluer({"fc": 40.0 + 0.01 * i, "fy": 550.0})
    duree = time.perf_counter() - t0
    assert duree < 1.0, "1 000 appels en %.2f s" % duree
