"""
FORM multimodal et tirage d'importance : l'extrait fait-il la meme chose ?

Ces fonctions manipulent des objets OpenTURNS -- `FORMResult`,
`ProbabilitySimulationResult` -- qui ne se serialisent pas en JSON. Un golden
de valeurs n'aurait donc pas de sens ici. La comparaison se fait autrement :
`tools/extraction_temoin.py` recupere les fonctions ORIGINALES a une revision
git figee (celle d'avant le retrait), et on fait tourner les deux versions
sur le meme etat limite analytique.

C'est la forme la plus directe de verification -- l'original et l'extrait,
cote a cote, sur les memes entrees.

Determinisme : OpenTURNS part d'une graine par defaut fixe, mais les tirages
avancent l'etat du generateur. `SetSeed(0)` est donc repose avant chaque
appel, sans quoi la comparaison porterait sur deux echantillons differents.

Ces tests exigent OpenTURNS et scikit-learn, pas Digital Structure.
"""

import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
for p in (os.path.join(REPO, "_reliability"), os.path.join(REPO, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

ot = pytest.importorskip("openturns", reason="couche fiabilite : OpenTURNS requis")
pytest.importorskip("sklearn", reason="DBSCAN requis pour la separation des modes")

import form as F                                    # noqa: E402
from extraction_temoin import AC_FLEXION, fonction_originale  # noqa: E402

#: revision d'AVANT le retrait des definitions des scripts AC.
#: Figee volontairement : `HEAD` designerait un etat qui ne les contient plus.
REVISION = "5da97e7"

N_VAR = 2
N_MAX_FORM = 50
TOL_FORM = 0.05
N_IS = 4000
COV_IS = 0.05
DEPARTS = np.array([[0.0, 0.0], [2.0, -2.0], [-2.0, 2.0], [1.0, 1.0]])

pytestmark = pytest.mark.slow


def _revision_disponible():
    try:
        fonction_originale(AC_FLEXION, "run_IS", {"n_var": 2, "n_IS": 10, "cov_IS": 0.1},
                           revision=REVISION)
        return True
    except Exception:
        return False


besoin_revision = pytest.mark.skipif(
    not _revision_disponible(),
    reason="revision %s introuvable (historique reecrit ?)" % REVISION)


# --------------------------------------------------------------------------- #
# Un etat limite analytique, sans metamodele ni solveur                       #
# --------------------------------------------------------------------------- #
class _G(ot.OpenTURNSPythonFunction):
    """g(u) = 3,5 - (u1 + 2 u2) / sqrt(5) : hyperplan a distance 3,5."""

    def __init__(self):
        super().__init__(N_VAR, 1)

    def _exec(self, u):
        return [3.5 - (float(u[0]) + 2.0 * float(u[1])) / np.sqrt(5.0)]


@pytest.fixture(scope="module")
def evenement():
    g = ot.Function(_G())
    X = ot.RandomVector(ot.JointDistribution([ot.Normal(0.0, 1.0)] * N_VAR))
    return ot.ThresholdEvent(ot.CompositeRandomVector(g, X), ot.Less(), 0.0)


def _muet(fn, *a, **kw):
    """Ces fonctions impriment beaucoup ; on ne compare que ce qu'elles rendent."""
    with redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# --------------------------------------------------------------------------- #
# FORM multimodal                                                              #
# --------------------------------------------------------------------------- #
@besoin_revision
def test_form_all_modes_identique_a_l_original(evenement):
    orig = fonction_originale(
        AC_FLEXION, "FORM_all_modes",
        {"n_var": N_VAR, "n_max_FORM": N_MAX_FORM, "tol_FORM": TOL_FORM,
         "do_FORM_filter": False, "eff_bounds_min": None, "eff_bounds_max": None,
         "np": np, "DBSCAN": __import__("sklearn.cluster", fromlist=["DBSCAN"]).DBSCAN},
        revision=REVISION)

    ot.RandomGenerator.SetSeed(0)
    a = _muet(orig, DEPARTS, 0.9, evenement)
    ot.RandomGenerator.SetSeed(0)
    b = _muet(F.form_all_modes, DEPARTS, 0.9, evenement, N_VAR, N_MAX_FORM, TOL_FORM)

    assert type(a) is type(b)
    modes_a, modes_b = a[0], b[0]
    assert len(modes_a) == len(modes_b) >= 1
    for ma, mb in zip(modes_a, modes_b):
        assert ma.getHasoferReliabilityIndex() == pytest.approx(
            mb.getHasoferReliabilityIndex(), rel=1e-15)
        assert list(ma.getStandardSpaceDesignPoint()) == pytest.approx(
            list(mb.getStandardSpaceDesignPoint()), rel=1e-15)


def test_form_all_modes_retrouve_le_beta_analytique(evenement):
    """L'hyperplan est a distance 3,5 de l'origine : c'est beta, par
    construction. Verification independante de l'original."""
    modes = _muet(F.form_all_modes, DEPARTS, 0.9, evenement, N_VAR,
                  N_MAX_FORM, TOL_FORM)[0]
    assert len(modes) == 1, "un hyperplan n'a qu'un mode de defaillance"
    assert modes[0].getHasoferReliabilityIndex() == pytest.approx(3.5, abs=1e-6)


# --------------------------------------------------------------------------- #
# Tirage d'importance                                                          #
# --------------------------------------------------------------------------- #
@besoin_revision
def test_run_IS_identique_a_l_original(evenement):
    modes = _muet(F.form_all_modes, DEPARTS, 0.9, evenement, N_VAR,
                  N_MAX_FORM, TOL_FORM)[0]
    orig = fonction_originale(AC_FLEXION, "run_IS",
                              {"n_var": N_VAR, "n_IS": N_IS, "cov_IS": COV_IS},
                              revision=REVISION)
    ot.RandomGenerator.SetSeed(0)
    a = orig(modes, evenement)
    ot.RandomGenerator.SetSeed(0)
    b = F.run_IS(modes, evenement, N_VAR, N_IS, COV_IS)

    assert a.getProbabilityEstimate() == pytest.approx(b.getProbabilityEstimate(), rel=1e-15)
    assert a.getCoefficientOfVariation() == pytest.approx(b.getCoefficientOfVariation(), rel=1e-15)
    assert a.getOuterSampling() == b.getOuterSampling()


def test_run_IS_encadre_la_probabilite_analytique(evenement):
    """Pf exacte de l'hyperplan : Phi(-3,5) = 2,326e-04.

    La tolerance n'est pas choisie : elle vient du coefficient de variation
    que l'estimateur rapporte lui-meme. Fixer un seuil arbitraire aurait rendu
    ce test soit complaisant, soit instable -- le premier jet, a 5 %, tombait
    sur un ecart de 7,5 %, c'est-a-dire 1,5 ecart-type. Trois ecarts-types est
    le bon seuil pour un estimateur sans biais.
    """
    from scipy.stats import norm
    modes = _muet(F.form_all_modes, DEPARTS, 0.9, evenement, N_VAR,
                  N_MAX_FORM, TOL_FORM)[0]
    ot.RandomGenerator.SetSeed(0)
    r = F.run_IS(modes, evenement, N_VAR, N_IS, COV_IS)
    pf = r.getProbabilityEstimate()
    cov = r.getCoefficientOfVariation()
    exacte = float(norm.cdf(-3.5))
    ecart = abs(pf - exacte) / exacte
    assert cov <= COV_IS * 1.05, f"le tirage n'a pas atteint sa precision : COV={cov:.4f}"
    assert ecart < 3.0 * cov, (
        f"Pf_IS={pf:.4e} contre {exacte:.4e} : {ecart / cov:.1f} ecarts-types")

    # l'intervalle de confiance a 95 % doit contenir la valeur exacte
    demi = r.getConfidenceLength(0.95) / 2.0
    assert pf - demi <= exacte <= pf + demi


@besoin_revision
def test_print_results_IS_ecrit_la_meme_chose(evenement):
    """Cette fonction n'a aucune variable libre : elle doit etre reproduite
    au caractere pres, y compris son formatage."""
    modes = _muet(F.form_all_modes, DEPARTS, 0.9, evenement, N_VAR,
                  N_MAX_FORM, TOL_FORM)[0]
    ot.RandomGenerator.SetSeed(0)
    r = F.run_IS(modes, evenement, N_VAR, N_IS, COV_IS)
    orig = fonction_originale(AC_FLEXION, "print_results_IS", revision=REVISION)

    sa, sb = io.StringIO(), io.StringIO()
    with redirect_stdout(sa):
        orig(r)
    with redirect_stdout(sb):
        F.print_results_IS(r)
    assert sa.getvalue() == sb.getvalue()


# --------------------------------------------------------------------------- #
# Enveloppe g +/- 2 sigma                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sign", [+1, -1])
def test_bound_surrogate_encadre_bien(sign):
    g = ot.Function(_G())
    sigma = lambda u: 0.1 + 0.05 * abs(float(u[0]))          # noqa: E731
    f = ot.Function(F.bound_surrogate_function(g, sigma, sign, N_VAR, predict=None))
    for u in ([0.0, 0.0], [1.0, -1.0], [-2.0, 0.5], [3.0, 3.0]):
        attendu = g(ot.Point(u))[0] + sign * 2.0 * sigma(ot.Point(u))
        assert f(u)[0] == pytest.approx(attendu, rel=1e-15)


def test_bound_surrogate_le_haut_est_au_dessus_du_bas():
    g = ot.Function(_G())
    sigma = lambda u: 0.2                                     # noqa: E731
    haut = ot.Function(F.bound_surrogate_function(g, sigma, +1, N_VAR, predict=None))
    bas = ot.Function(F.bound_surrogate_function(g, sigma, -1, N_VAR, predict=None))
    for u in ([0.0, 0.0], [2.0, -1.0], [-1.5, 2.0]):
        assert haut(u)[0] - bas(u)[0] == pytest.approx(4.0 * 0.2, rel=1e-15)


# --------------------------------------------------------------------------- #
# Ce que l'extraction devait accomplir                                        #
# --------------------------------------------------------------------------- #
def test_plus_aucune_variable_libre():
    from extraction_temoin import variables_libres
    autorises = {"np", "ot", "DBSCAN"}
    chemin = os.path.join(REPO, "_reliability", "form.py")
    for nom in ("form_all_modes", "run_IS", "run_IS_proj",
                "print_results_IS", "bound_surrogate_function"):
        restantes = set(variables_libres(chemin, nom)) - autorises
        assert not restantes, f"{nom} depend encore de {sorted(restantes)}"


# --------------------------------------------------------------------------- #
# Le garde-fou qui manquait                                                   #
# --------------------------------------------------------------------------- #
#: Les delegues vers `_reliability/form.py` que les etudes portent ENCORE.
#:
#: `BoundSurrogateFunction` en est sorti le 02/09/2026 : il liait `n_var` et
#: le predicteur a `form.bound_surrogate_function`, et il n'etait passe qu'a
#: `BoucleEFF` -- qui a les deux. La boucle le fabrique desormais, et l'etude
#: ne le nomme plus. C'est un pas de l'option A : les delegues qui ne font que
#: lier `n_var` et `cfg.*` a un appel de module disparaissent.
#:
#: Cette liste doit RETRECIR. Un nom qui la quitte est un delegue de moins
#: dans les deux etudes ; un nom qui s'y ajoute est une extraction a finir.
EXTRAITES = ["FORM_all_modes", "run_IS", "run_IS_proj", "print_results_IS"]


@pytest.mark.parametrize("rel", ["pure_flexion/AC3_pure_flexion.py",
                                 "Moulinblanc/AC3_moulinblanc.py"])
def test_les_scripts_ac_ne_portent_plus_que_des_delegues(rel):
    """Ecrit APRES avoir constate son absence.

    `run_IS_proj` avait ete extrait dans `form.py` sans etre retire des deux
    scripts AC : il existait en TROIS exemplaires, et le message de commit
    affirmait le contraire. Le test equivalent existait pour les caches ; il
    manquait ici, et rien d'autre ne pouvait le voir -- aucun script AC n'est
    execute par la suite de tests.

    On verifie la STRUCTURE, pas l'absence du nom : les delegues portent les
    memes noms que les originaux, par conception.
    """
    import ast
    chemin = os.path.join(REPO, rel)
    with open(chemin, encoding="utf-8", errors="replace") as fh:
        source = fh.read()
    arbre = ast.parse(source, chemin)
    vus = {}
    for n in ast.walk(arbre):
        if not isinstance(n, (ast.FunctionDef, ast.ClassDef)) or n.name not in EXTRAITES:
            continue
        assert isinstance(n, ast.FunctionDef), \
            f"{rel} : {n.name} est reste une classe, il devait devenir un delegue"
        corps = [x for x in n.body
                 if not (isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant))]
        assert len(corps) == 1 and isinstance(corps[0], ast.Return), \
            f"{rel} : {n.name} n est pas un delegue ({len(corps)} instructions)"
        appelle = [d.value.id for d in ast.walk(corps[0])
                   if isinstance(d, ast.Attribute) and isinstance(d.value, ast.Name)]
        assert "_form" in appelle, f"{rel} : {n.name} ne transmet pas a _form"
        vus[n.name] = True
    assert set(vus) == set(EXTRAITES), \
        f"{rel} : manquants {sorted(set(EXTRAITES) - set(vus))}"
