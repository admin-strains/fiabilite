r"""Les ajustements de metamodeles, sortis des scripts d'etude.

LA DIVERGENCE QUE CETTE EXTRACTION A TROUVEE -- 27/08/2026
-----------------------------------------------------------
`build_metamodel_KRG` bornait l'optimisation des longueurs de correlation a
`[1, 100]` dans la flexion pure et a `[0, 100]` dans le Moulin Blanc. Les deux
fonctions etaient par ailleurs identiques au caractere pres, et rien -- ni
commentaire, ni trace, ni test -- ne signalait l'ecart.

Une borne inferieure nulle laisse l'optimiseur degenerer vers une longueur de
correlation quasi nulle : le krigeage cesse alors de lisser et interpole le
bruit. Ce n'est pas anodin sur un metamodele qui sert a decider ou depenser le
prochain appel solveur.

C'est la QUATRIEME divergence trouvee entre deux copies censees etre la meme
chose -- apres la taille de maille entre `run_HF` et `run_one_SOL`, le solveur
lineaire entre les deux `InitSolver.py`, et deux fonctions de trace presentes
d'un seul cote. La valeur de chaque etude est conservee ; elle est simplement
devenue visible.

DEUX AUTRES DEFAUTS, PLUS DISCRETS
-----------------------------------
* `build_metamodel_GEK` portait `if do_GEK: sm = GEKPLS(...) else: sm =
  GEKPLS(...)` -- les deux branches construisaient le meme objet.
* `calculate_PCE(xt, y_hf, all_grad_hf, metamodel_PCE)` ne lisait ni `y_hf`
  ni `all_grad_hf`.
"""

import ast
import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _d in (os.path.join(_REPO, "_lib"), os.path.join(_REPO, "_surrogate")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

ot = pytest.importorskip("openturns")
np = pytest.importorskip("numpy")
pytest.importorskip("smt")

import ajuster                                       # noqa: E402

SCRIPTS = ["pure_flexion/AC3_pure_flexion.py", "Moulinblanc/AC3_moulinblanc.py"]
FONCTIONS = ("build_Y_aug", "build_metamodel_PCE", "calculate_PCE",
             "build_metamodel_KRG", "build_metamodel_GEK")


# --------------------------------------------------------------------- #
# la divergence est desormais explicite
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("script,attendu", [
    ("pure_flexion/AC3_pure_flexion.py", "theta_min=1.0"),
    ("Moulinblanc/AC3_moulinblanc.py", "theta_min=0.0"),
])
def test_la_borne_de_theta_est_ecrite_a_l_appel(script, attendu):
    """Chaque etude garde SA valeur historique, mais elle est lisible. Un
    ecart entre les deux copies ne doit plus pouvoir se cacher dans un corps
    de fonction duplique."""
    with open(os.path.join(_REPO, script), encoding="utf-8",
              errors="replace") as fh:
        src = fh.read()
    assert attendu in src, (
        "%s : la borne inferieure des longueurs de correlation doit rester "
        "visible a l'appel (%s)." % (script, attendu))


def test_la_borne_n_est_appliquee_que_si_on_le_demande():
    """`borner_theta=False` doit laisser l'optimiseur libre : c'est ce que
    faisaient les branches non-KRG."""
    src = ast.dump(ast.parse(
        open(os.path.join(_REPO, "_surrogate", "ajuster.py"),
             encoding="utf-8").read()))
    assert "setOptimizationBounds" in src
    xt = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, 0.7], [0.4, -1.2],
                   [-0.8, -0.3], [1.4, 1.1]])
    yt = (xt[:, 0:1] ** 2 + xt[:, 1:2])
    _, res_libre = ajuster.ajuster_KRG(xt, yt, borner_theta=False,
                                       tracer=lambda _m: None)
    _, res_borne = ajuster.ajuster_KRG(xt, yt, borner_theta=True,
                                       theta_min=3.0, theta_max=100.0,
                                       tracer=lambda _m: None)
    theta = list(res_borne.getCovarianceModel().getScale())
    assert min(theta) >= 3.0 - 1e-9, (
        "la borne inferieure n'a pas ete appliquee : theta=%s" % theta)
    assert res_libre is not res_borne


# --------------------------------------------------------------------- #
# elles ont quitte les scripts
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("script", SCRIPTS)
def test_les_ajustements_ne_sont_plus_dans_l_etude(script):
    chemin = os.path.join(_REPO, script)
    with open(chemin, encoding="utf-8", errors="replace") as fh:
        arbre = ast.parse(fh.read(), filename=chemin)
    main = [n for n in arbre.body if isinstance(n, ast.If)][0]
    definies = {n.name for n in main.body if isinstance(n, ast.FunctionDef)}
    restantes = sorted(definies & set(FONCTIONS))
    assert not restantes, "%s redefinit %s" % (script, restantes)


def test_le_module_ajuster_ne_masque_pas_le_clone_UQLab():
    """`_lib/fit.py` est le clone UQLab et se trouve sur le meme chemin
    d'import. Un module `_surrogate/fit.py` l'aurait eclipse -- ce qui s'est
    produit, et n'a ete rattrape que par l'echec de collecte de la suite."""
    import fit as clone_uqlab
    assert hasattr(clone_uqlab, "uq_PCK_calculate_coefficients")
    assert os.path.basename(os.path.dirname(clone_uqlab.__file__)) == "_lib"


# --------------------------------------------------------------------- #
# y_augmente : l'ordre des blocs n'est pas negociable
# --------------------------------------------------------------------- #
def test_y_augmente_range_les_valeurs_puis_chaque_derivee():
    """y_dot = [y^1..y^n, dg/du1^1..dg/du1^n, ..., dg/dum^1..dg/dum^n].

    Un ordre different -- point par point plutot que composante par
    composante -- donnerait un vecteur de meme longueur et un GEPCK faux
    sans aucune erreur."""
    yt = np.array([[1.0], [2.0], [3.0]])
    grad = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    attendu = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0]
    assert list(ajuster.y_augmente(yt, grad)) == attendu


def test_y_augmente_a_la_longueur_annoncee():
    n, m = 7, 3
    yt = np.zeros((n, 1))
    grad = np.zeros((n, m))
    assert ajuster.y_augmente(yt, grad).shape == (n * (1 + m),)


# --------------------------------------------------------------------- #
# etiquetage des termes du chaos : plus d'hypothese sur n_var
# --------------------------------------------------------------------- #
def test_l_etiquette_des_termes_est_inchangee_a_deux_variables():
    """Le journal d'un run doit rester comparable a ceux d'avant."""
    assert ajuster._etiquette_terme([0, 0]) == "1"
    assert ajuster._etiquette_terme([3, 0]) == "H3(u1)"
    assert ajuster._etiquette_terme([0, 2]) == "H2(u2)"
    assert ajuster._etiquette_terme([1, 2]) == "H1(u1)*H2(u2)"


def test_l_etiquette_des_termes_dit_la_verite_au_dela_de_deux_variables():
    """L'original ne lisait que `mi[0]` et `mi[1]` : avec trois variables il
    affichait un terme QUI N'ETAIT PAS celui selectionne."""
    assert ajuster._etiquette_terme([0, 0, 2]) == "H2(u3)"
    assert ajuster._etiquette_terme([1, 0, 3]) == "H1(u1)*H3(u3)"
    assert ajuster._etiquette_terme([0, 0, 0, 0]) == "1"


# --------------------------------------------------------------------- #
# PCE : la composante rendue est bien celle du plan
# --------------------------------------------------------------------- #
def test_la_composante_PCE_rend_valeurs_et_gradients_aux_points_du_plan():
    """`yr = y_hf - y_PCE` n'a de sens que si `y_PCE` est evaluee AUX MEMES
    points, dans le MEME ordre."""
    modele = ot.SymbolicFunction(["u1", "u2"], ["2*u1 + 3*u2"])
    xt = np.array([[1.0, 0.0], [0.0, 1.0], [-2.0, 4.0]])
    y_pce, grad_pce = ajuster.composante_PCE(xt, modele)
    assert y_pce[:, 0] == pytest.approx([2.0, 3.0, 8.0])
    assert grad_pce == pytest.approx(np.array([[2.0, 3.0]] * 3))


def test_la_composante_PCE_ne_reclame_plus_les_valeurs_haute_fidelite():
    """Deux parametres n'etaient jamais lus. Les garder faisait croire que la
    composante PCE etait comparee aux gradients exacts : elle ne l'est pas."""
    import inspect
    assert list(inspect.signature(ajuster.composante_PCE).parameters) == \
        ["xt", "metamodele_PCE"]


# --------------------------------------------------------------------- #
# GEK : plus de branche morte, et tous les gradients sont fournis
# --------------------------------------------------------------------- #
def test_le_GEK_recoit_une_derivee_par_variable():
    src = open(os.path.join(_REPO, "_surrogate", "ajuster.py"),
               encoding="utf-8").read()
    # compte les CONSTRUCTIONS, pas les mentions : la docstring cite le defaut.
    constructions = [n for n in ast.walk(ast.parse(src))
                     if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "GEKPLS"]
    assert len(constructions) == 1, (
        "l'original construisait le MEME objet dans les deux branches d'un "
        "`if` ; une seule construction doit subsister (%d trouvees)."
        % len(constructions))
    assert "all_grad.shape[1]" in src, (
        "le nombre de derivees fournies doit venir du tableau lui-meme, pas "
        "d'un `n_var` capture ailleurs.")
