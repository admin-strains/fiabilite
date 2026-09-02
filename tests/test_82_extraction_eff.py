"""
Le critere EFF extrait reproduit-il celui des scripts AC ?

La formule etait ecrite DEUX FOIS dans chaque script -- vectorisee dans
`_eff_vectorized`, scalaire dans `EFFFunction._exec` -- soit quatre copies
dans le depot. `_reliability/eff.py` n'en garde qu'une.

L'oracle est `tests/golden/eff_original.json`, qui contient les valeurs des
DEUX implementations d'origine, lues a une revision git. Le golden conserve
donc aussi la preuve qu'elles coincidaient : c'est ce qui justifiait
l'unification.

`eff` seul ne demande que numpy et scipy. `eff_function` et `batch_mu_sigma`
demandent OpenTURNS et sautent la ou il est absent.
"""

import json
import os
import sys

import numpy as np
import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
for p in (os.path.join(REPO, "_reliability"), os.path.join(REPO, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

GOLDEN = os.path.join(TESTS, "golden", "eff_original.json")

pytestmark = pytest.mark.skipif(not os.path.isfile(GOLDEN),
                                reason="oracle du critere EFF absent")


@pytest.fixture(scope="module")
def attendu():
    with open(GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)


def _eff():
    import eff
    return eff


def _eff_ot():
    return pytest.importorskip("eff_ot", reason="emballage OpenTURNS")


#: TOLERANCE DU GOLDEN DE LA FORMULE EFF, ET POURQUOI PAS 1e-15
#:
#: Elle valait `rel=1e-15`. L'epsilon de la double precision vaut 2.22e-16 :
#: 1e-15 fait donc QUATRE ULP, c'est-a-dire le bruit lui-meme. Mesure du
#: 31/08/2026 sur un runner d'integration continue -- 1 element sur 11 :
#:
#:     ecart absolu maximal    1.1102230246251565e-16    (un dernier bit)
#:     ecart relatif maximal   1.1362598044671046e-15
#:
#: Le golden echouait donc pour un ULP, sur une formule inchangee. 1e-13
#: laisse deux ordres de marge au-dessus de ce bruit et reste treize ordres
#: au-dessous de ce qu'une modification de la formule produirait.
TOL_EFF_GOLDEN = 1e-13


# --------------------------------------------------------------------------- #
# La formule                                                                   #
# --------------------------------------------------------------------------- #
def test_eff_reproduit_la_version_vectorisee(attendu):
    mu = np.array(attendu["mu"])
    sg = np.array(attendu["sigma"])
    for f in attendu["facteurs"]:
        obtenu = _eff().eff(mu, sg, f)
        assert list(obtenu) == pytest.approx(attendu["vectorise"][str(f)],
                                             rel=TOL_EFF_GOLDEN, abs=1e-300)


def test_eff_reproduit_aussi_la_version_scalaire(attendu):
    """Les deux implementations d'origine coincidaient : c'est ce qui a permis
    de n'en garder qu'une. Le golden en garde la trace, ce test la verifie."""
    mu = np.array(attendu["mu"])
    sg = np.array(attendu["sigma"])
    for f in attendu["facteurs"]:
        obtenu = _eff().eff(mu, sg, f)
        assert list(obtenu) == pytest.approx(attendu["scalaire"][str(f)],
                                             rel=TOL_EFF_GOLDEN, abs=1e-300)


@pytest.mark.parametrize("sigma", [0.0, -1.0, -1e-9])
def test_eff_est_nul_ou_le_metamodele_est_certain(sigma):
    """sigma <= 0 : rien a gagner a enrichir la. Comportement d'origine."""
    assert _eff().eff(np.array([1.0, 0.0, -1.0]),
                      np.full(3, sigma), 2.0).tolist() == [0.0, 0.0, 0.0]


def test_eff_est_positif_ailleurs():
    """Une esperance de faisabilite est positive ; le critere etant ensuite
    maximise, un signe faux orienterait l'enrichissement a l'envers.

    Tolerance a l'epsilon machine : la sortie descend a -1,8e-15 la ou le
    critere sous-depasse (|mu/sigma| > 9,6), pour un maximum de 7,77 -- soit
    2e-16 en relatif. C'est l'arrondi, pas la formule. Mesure sur
    200 000 tirages : 0,31 % des points, tous a cette amplitude.
    """
    rng = np.random.default_rng(3)
    mu = rng.normal(0, 3, 200000)
    sg = np.abs(rng.normal(0, 1.5, 200000)) + 1e-9
    v = _eff().eff(mu, sg, 2.0)
    assert (v >= -1e-13).all(), "minimum %.3e" % v.min()
    assert (v[np.abs(mu / sg) < 5] > 0).all()


def test_eff_decroit_quand_on_s_eloigne_de_l_etat_limite():
    """Propriete attendue du critere : a sigma fixe, il est maximal en mu = 0
    (sur l'etat limite estime) et decroit de part et d'autre."""
    e = _eff()
    mu = np.linspace(-4.0, 4.0, 201)
    v = e.eff(mu, np.full_like(mu, 1.0), 2.0)
    assert np.argmax(v) == 100                      # mu = 0
    assert (np.diff(v[:101]) > 0).all()
    assert (np.diff(v[100:]) < 0).all()


# --------------------------------------------------------------------------- #
# L'emballage OpenTURNS, qui ne doit plus reecrire la formule                  #
# --------------------------------------------------------------------------- #
def test_eff_function_donne_les_memes_valeurs_que_eff():
    """`EFFFunction._exec` recalculait la formule ; l'emballage ne fait plus
    qu'appeler `eff`. Ce test verifie qu'aucune divergence ne se reintroduit."""
    ot = pytest.importorskip("openturns")
    e = _eff()

    class _G(ot.OpenTURNSPythonFunction):
        def __init__(self):
            super().__init__(2, 1)

        def _exec(self, u):
            return [float(u[0]) * 0.7 - float(u[1]) * 0.3 + 0.2]

    g = ot.Function(_G())
    sigma = lambda u: 0.1 + 0.05 * abs(float(u[0]))          # noqa: E731

    f = ot.Function(_eff_ot().eff_function(g, sigma, 2, 2.0))
    for u in ([0.0, 0.0], [1.0, -1.0], [-2.0, 0.5], [3.0, 3.0], [-0.3, 0.9]):
        mu = g(ot.Point(u))[0]
        sg = sigma(ot.Point(u))
        assert f(u)[0] == pytest.approx(float(e.eff(np.array([mu]), np.array([sg]), 2.0)[0]),
                                        rel=1e-15)


def test_eff_function_rend_zero_si_sigma_nul():
    ot = pytest.importorskip("openturns")
    e = _eff()

    class _G(ot.OpenTURNSPythonFunction):
        def __init__(self):
            super().__init__(2, 1)

        def _exec(self, u):
            return [1.0]

    f = ot.Function(_eff_ot().eff_function(ot.Function(_G()), lambda u: 0.0, 2, 2.0))
    assert f([0.0, 0.0])[0] == 0.0


# --------------------------------------------------------------------------- #
# Ce que l'extraction devait accomplir                                        #
# --------------------------------------------------------------------------- #
def test_plus_aucune_variable_libre():
    from extraction_temoin import variables_libres
    autorises = {"np", "ot", "norm", "eff"}
    chemin = os.path.join(REPO, "_reliability", "eff.py")
    for nom in ("eff", "eff_termes"):
        restantes = set(variables_libres(chemin, nom)) - autorises
        assert not restantes, f"{nom} depend encore de {sorted(restantes)}"


@pytest.mark.parametrize("rel", ["pure_flexion/AC3_pure_flexion.py",
                                 "Moulinblanc/AC3_moulinblanc.py"])
def test_les_scripts_ac_ne_reecrivent_plus_la_formule(rel):
    """La signature de la formule : `norm.pdf(t2)`. Si elle reapparait dans un
    script AC, c'est qu'une copie est revenue."""
    with open(os.path.join(REPO, rel), encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    assert "norm.pdf(t2)" not in src, f"{rel} reecrit la formule EFF"
    assert src.count("norm.cdf(t1)") == 0, f"{rel} reecrit la formule EFF"


def test_la_somme_des_termes_vaut_le_critere():
    """La propriete qui rend impossible le retour du defaut : le journal
    d'enrichissement decompose desormais `eff` au lieu de la recopier."""
    e = _eff()
    rng = np.random.default_rng(11)
    mu = np.concatenate([rng.normal(0, 3, 3000), [0.0, 5.0, -5.0]])
    sg = np.concatenate([np.abs(rng.normal(0, 1.5, 3000)) + 1e-9, [1.0, 0.0, -1.0]])
    for f in (2.0, 1.0, 0.5):
        termes = e.eff_termes(mu, sg, f)
        somme = termes[3] + termes[4] + termes[5] + termes[6]
        assert somme == pytest.approx(e.eff(mu, sg, f), rel=1e-13, abs=1e-300)


def test_la_decomposition_est_neutre_quand_sigma_est_nul():
    e = _eff()
    termes = e.eff_termes(np.array([1.0, 2.0]), np.array([0.0, -1.0]), 2.0)
    for t in termes[3:]:
        assert t.tolist() == [0.0, 0.0]


# --------------------------------------------------------------------------- #
# LE JOURNAL DE LA DECOMPOSITION (extrait de `run_EFF` le 27/08/2026)          #
# --------------------------------------------------------------------------- #
class _Journal(list):
    def __call__(self, message):
        self.append(message)


def test_le_journal_decompose_le_critere_en_six_lignes():
    e = _eff()
    j = _Journal()
    e.journaliser_decomposition(0.3, 0.5, 2.0, [1.0, -2.0], tracer=j)
    assert len(j) == 6
    assert j[0].startswith("  EFF converge debug : u_opt=")
    assert "sigmaG=0.50000000" in j[0] and "muG=0.30000000" in j[0]
    assert "epsilon=1.00000000" in j[0], "epsilon = eps_factor * sigma"
    assert j[1].startswith("    t1=") and j[4].startswith("    term1=")
    assert j[5].startswith("    EFF = ")


def test_le_journal_dit_la_somme_du_critere_lui_meme():
    """La ligne "EFF = ..." doit valoir `eff`, pas une recopie de la formule.
    C'est exactement le defaut du 26/08 : la copie manuelle affichait de
    39 % a 1271 % d'ecart, parfois de signe oppose."""
    e = _eff()
    for mu, sg, f in ((0.3, 0.5, 2.0), (-1.7, 0.9, 1.0), (0.0, 2.0, 0.5)):
        j = _Journal()
        e.journaliser_decomposition(mu, sg, f, [0.0, 0.0], tracer=j)
        affiche = float(j[5].split("=")[1])
        assert affiche == pytest.approx(float(e.eff(mu, sg, f)), rel=1e-8)


def test_un_metamodele_interpolant_le_dit_en_une_ligne():
    """`sigma = 0` n'est pas une anomalie : le krigeage interpole ses points
    d'apprentissage, et le point d'arret peut tomber sur l'un d'eux. Une
    decomposition n'aurait alors aucun sens -- tous les termes sont nuls."""
    e = _eff()
    j = _Journal()
    e.journaliser_decomposition(1.0, 0.0, 2.0, [0.0, 0.0], tracer=j)
    assert j == ["  EFF converge debug : sigmaG=0 (modele interpolant exact "
                 "au point u_opt)"]


# --------------------------------------------------------------------------- #
# LES TROIS CHAMPS D'UNE COUPE (extraits des deux etudes le 27/08/2026)        #
# --------------------------------------------------------------------------- #
# `batch_mu_sigma` a DEUX voies, et les deux sont exercees ici :
#   - la voie `fm` (PCK, GEPCK) : un seul appel `predict(..., return_var=True)`
#     rend mu ET la variance ; `g_ot` n'est jamais appele ;
#   - la voie generique (krigeage) : `g_ot` en lot, puis `sigma_func` point
#     par point.


class _AvecFM:
    """Un `sigma_func` methode liee dont le porteur a un `fm` : c'est ce que
    `batch_mu_sigma` cherche pour choisir la voie rapide."""

    def __init__(self):
        self.fm = "modele-ajuste"

    def sigma(self, u):
        raise AssertionError("la voie `fm` ne doit pas appeler sigma_func")


def _predict_lot(sigma):
    """Le predicteur de la voie `fm` : mu = u0 + u1, variance constante."""
    def predict(fm, grid, return_var=False):
        assert fm == "modele-ajuste" and return_var
        mu = (grid[:, 0] + grid[:, 1]).reshape(-1, 1)
        return mu, np.full((len(grid), 1), sigma ** 2)
    return predict


class _EnLot:
    """Un `g_ot` de la voie generique : appele UNE fois sur tout le lot."""

    def __init__(self):
        self.appels = 0

    def __call__(self, echantillon):
        self.appels += 1
        pts = np.array(echantillon)
        return (pts[:, 0] + pts[:, 1]).reshape(-1, 1)


def _quadrillage(cote):
    g = np.zeros((cote * cote, 2))
    g[:, 0] = np.tile(np.linspace(-1.0, 1.0, cote), cote)
    g[:, 1] = np.repeat(np.linspace(-1.0, 1.0, cote), cote)
    return g


def _par_fm(cote=3, sigma=0.5):
    e = _eff_ot()
    porteur = _AvecFM()
    return e.champs_sur_coupe(None, porteur.sigma, _quadrillage(cote), cote,
                              2.0, _predict_lot(sigma))


def test_les_trois_champs_ont_la_forme_de_la_coupe():
    Z_eff, Z_sigma, Z_g = _par_fm(cote=4)
    assert Z_eff.shape == Z_sigma.shape == (4, 4)


def test_le_champ_d_ecart_type_est_la_racine_de_la_variance():
    """`batch_mu_sigma` rend `sigma`, pas `sigma^2` -- le critere EFF attend
    un ecart-type."""
    _, Z_sigma, _ = _par_fm(cote=3, sigma=0.25)
    assert np.allclose(Z_sigma, 0.25)


def test_le_champ_du_critere_vaut_eff_point_par_point():
    """Le critere TRACE doit etre celui qui est MAXIMISE. C'est le defaut du
    26/08 sous une autre forme : une copie de la formule qui derive."""
    e = _eff()
    Z_eff, Z_sigma, _ = _par_fm(cote=3, sigma=0.5)
    mu = (_quadrillage(3)[:, 0] + _quadrillage(3)[:, 1]).reshape(3, 3)
    attendu = e.eff(mu.ravel(), Z_sigma.ravel(), 2.0).reshape(3, 3)
    assert np.allclose(Z_eff, attendu)


def test_la_voie_fm_n_appelle_jamais_le_metamodele_point_par_point():
    """Un seul appel BLAS pour toute la coupe. Point par point, une planche
    de 100x100 ferait 10 000 allers-retours Python/OpenTURNS -- c'est
    pourquoi `_AvecFM.sigma` leve si on l'appelle."""
    Z_eff, _, Z_g = _par_fm(cote=5)
    assert Z_eff.shape == (5, 5)
    assert Z_g is None, "g_ot etait None : rien a tracer comme etat limite"


def test_la_voie_generique_evalue_le_metamodele_en_UN_lot():
    e = _eff_ot()
    g = _EnLot()
    Z_eff, Z_sigma, Z_g = e.champs_sur_coupe(
        g, lambda u: 0.5, _quadrillage(3), 3, 2.0, None)
    assert g.appels == 1, "le metamodele a ete appele point par point"
    assert Z_g[1, 1] == pytest.approx(0.0)
    assert Z_g[0, 0] == pytest.approx(-2.0)
    assert np.allclose(Z_sigma, 0.5)


def test_les_deux_voies_donnent_les_MEMES_champs():
    """Elles servent la meme planche : un ecart entre elles se lirait comme
    une difference de modele."""
    e = _eff_ot()
    a = _par_fm(cote=4, sigma=0.5)
    b = e.champs_sur_coupe(_EnLot(), lambda u: 0.5, _quadrillage(4), 4, 2.0,
                           None)
    for Za, Zb in zip(a[:2], b[:2]):
        assert np.allclose(Za, Zb)
