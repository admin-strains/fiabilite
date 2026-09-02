r"""Le gradient analytique de la vraisemblance, verifie.

POURQUOI CE FICHIER EST LE PLUS IMPORTANT DU CORRECTIF
--------------------------------------------------------
Un gradient analytique FAUX serait pire que le bruit qu'il remplace : il
serait faux de facon confiante et reproductible, et l'optimiseur le suivrait
sans broncher jusqu'a un mauvais optimum, sur toutes les machines a la fois.
Le defaut d'origine, lui, se voyait -- `ABNORMAL_TERMINATION_IN_LNSRCH`, des
appels a `nit = 0`, cinq theta pour cinq jobs.

La verification est donc la condition d'existence du correctif, pas sa
formalite.

LA METHODE, ET SON PIEGE
-------------------------
On compare aux differences finies -- mais PAS au pas de scipy. C'est
justement celui qui ne marche pas : mesure du 02/09/2026, il rend un
gradient 3 372 fois trop grand et de signe oppose. On compare AU MINIMUM DE
LA COURBE EN U, vers 1e-5 a 1e-3 en relatif.

Et quand l'ecart reste grand a ce pas-la, ce n'est pas une conclusion : sur
le premier ajustement de `flexion/PCK`, le gradient vaut 6e-04 et le bruit de
J vaut 3.05e-08, si bien qu'a h = 4.8e-04 le bruit contribue encore 10 %. Le
balayage tranche :

    pas relatif   FD composante 1      ecart a l'analytique
    1e-06         +1.23200364e-03      1.01e+00
    1e-05         +6.50329404e-04      6.05e-02
    1e-04         +6.11288960e-04      3.21e-03
    1e-03         +6.12765876e-04      8.00e-04     <-- minimum
    1e-02         +5.59851778e-04      8.71e-02

Le FD CONVERGE vers l'analytique. C'etait donc le FD qui avait tort.

CE QUI EST COUVERT, ET CE QUI NE L'EST PAS
--------------------------------------------
PCK, noyau separable anisotrope, familles `gaussian` et `matern-5_2` -- les
seules que `kernel_deriv_factory` couvre deja, et les etudes n'emploient que
`matern-5_2`.

GEPCK n'est PAS couvert : sa matrice de Gram est augmentee (N*(M+1) lignes,
72 pour 24 points a deux variables) et porte les blocs de derivees du noyau.
Il faudrait deriver CEUX-LA en theta. `grad_J_of_theta_ML` le detecte et rend
None ; l'optimiseur retombe alors sur des differences finies de pas RELATIF,
degrade mais jamais faux -- et deja trois ordres meilleur que le defaut.
"""

import json
import os
import sys
import warnings

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in ("_lib", "_model"):
    _c = os.path.join(_REPO, _p)
    if _c not in sys.path:
        sys.path.insert(0, _c)
if _ICI not in sys.path:
    sys.path.insert(0, _ICI)

np = pytest.importorskip("numpy")

import harness                                              # noqa: E402


#: Les pas RELATIFS ou la comparaison a un sens. Le pas de scipy (1.49e-08
#: ABSOLU) en est volontairement absent : c'est celui que ce correctif
#: remplace.
PAS_UTILES = (1e-5, 1e-4, 1e-3)

#: EN DECA DE CE SIGMA^2, `J` N'EST PLUS CALCULABLE, ET AUCUN GRADIENT N'Y EST
#: VERIFIABLE -- ni l'analytique, ni les differences finies.
#:
#: `sigma^2 = (1/N) z^T z` avec `z = Ytilde - Q1 (Q1^T Ytilde)`. Quand la PCE
#: represente l'etat limite EXACTEMENT -- c'est le cas `linear`, LOO ~ 1e-25 --
#: `z` est une difference de quantites qui s'annulent, et son bruit RELATIF
#: explose. Mesure du 02/09/2026, `linear/PCK` a theta = [17.245, 100.000],
#: sigma^2 = 8.97e-24 :
#:
#:     pas relatif   FD composante 0    FD composante 1
#:     1e-06         -1.48073805e+05    -9.14798381e+03
#:     1e-05         -1.37831838e+04    -2.38381940e+03
#:     1e-04         +2.27522070e+03    +2.92813334e+02
#:     1e-03         +1.07362184e+02    +9.30104371e+00
#:     1e-02         +1.31026319e+01    +7.03144962e+00
#:
#: Cinq ordres de grandeur d'ecart entre deux pas voisins, aucun plateau,
#: changement de signe : ce ne sont pas des derivees. Comparer l'analytique a
#: cela ne prouverait rien, dans un sens comme dans l'autre.
#:
#: Ce n'est PAS un defaut du correctif : c'est une limite du probleme, la meme
#: qui rend theta non identifiable sur ce cas (`test_31`).
SIGMA2_PLANCHER = 1e-20


def _sigma_carre(params, theta):
    """`sigma^2` a ce theta, par le meme chemin que `J`."""
    import kriging as _kr
    CorrOptions = dict(params["CorrOptions"], IsGram=True)
    R = CorrOptions["Handle"](params["X"], params["X"],
                              np.asarray(theta, float), CorrOptions)
    am = _kr.uq_Kriging_calc_auxMatrices(R, params["F"], params["Y"],
                                         "ml_optimization")
    if am.get("Q1") is None:
        return 0.0
    Yt = np.asarray(am["Ytilde"]).reshape(-1, 1)
    z = Yt - am["Q1"] @ (am["Q1"].T @ Yt)
    return float((z.T @ z).item()) / params["N"]


def _fd(J, theta, rel):
    """Differences finies CENTREES de pas relatif, une par composante."""
    out = np.empty(np.asarray(theta).size, dtype=float)
    for i in range(out.size):
        h = rel * max(abs(theta[i]), 1e-3)
        tp, tm = np.array(theta, float), np.array(theta, float)
        tp[i] += h
        tm[i] -= h
        out[i] = (J(tp) - J(tm)) / (2 * h)
    return out


# --------------------------------------------------------------------- #
# 1. la derivee du NOYAU par rapport a theta
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("famille", ["matern-5_2", "gaussian"])
@pytest.mark.parametrize("n_var", [2, 3])
def test_dR_dtheta_coincide_avec_les_differences_finies(famille, n_var):
    """La brique de base. Trois echelles de theta, car l'erreur du pas
    ABSOLU de scipy venait precisement de ce que theta couvre quatre
    decades : un correctif qui ne marcherait qu'a theta ~ 1 ne servirait a
    rien."""
    from kernels import uq_eval_Kernel, dR_dtheta, PEPITE_PAR_DEFAUT

    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, (14, n_var))
    opts = {"Handle": uq_eval_Kernel, "Family": famille, "Type": "separable",
            "Isotropic": False, "Nugget": PEPITE_PAR_DEFAUT, "IsGram": True}

    for theta in (np.full(n_var, 1.0),
                  np.linspace(0.3, 7.0, n_var),
                  np.full(n_var, 0.05)):
        analytique = dR_dtheta(X, theta, opts)
        for i in range(n_var):
            h = 1e-5 * theta[i]
            tp, tm = theta.copy(), theta.copy()
            tp[i] += h
            tm[i] -= h
            fd = (uq_eval_Kernel(X, X, tp, opts)
                  - uq_eval_Kernel(X, X, tm, opts)) / (2 * h)
            echelle = max(float(np.max(np.abs(analytique[i]))), 1e-30)
            ecart = float(np.max(np.abs(analytique[i] - fd))) / echelle
            assert ecart < 1e-6, (
                "%s, n_var=%d, theta=%s, composante %d : ecart %.2e entre "
                "dR/dtheta analytique et differences finies. Un gradient FAUX "
                "serait pire que pas de gradient."
                % (famille, n_var, np.array2string(theta, precision=3), i, ecart))


def test_la_diagonale_de_dR_dtheta_est_nulle():
    """`R` porte `1 + pepite` sur sa diagonale, et ni l'un ni l'autre ne
    depend de theta. Si la diagonale bougeait, le gradient inclurait une
    contribution qui n'existe pas."""
    from kernels import uq_eval_Kernel, dR_dtheta, PEPITE_PAR_DEFAUT
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, (9, 2))
    opts = {"Handle": uq_eval_Kernel, "Family": "matern-5_2",
            "Type": "separable", "Isotropic": False,
            "Nugget": PEPITE_PAR_DEFAUT, "IsGram": True}
    for D in dR_dtheta(X, np.array([1.3, 0.7]), opts):
        assert np.allclose(np.diag(D), 0.0, atol=0.0), np.diag(D)


def test_une_famille_sans_derivee_ecrite_LEVE(monkeypatch):
    """Elle ne doit pas rendre un resultat approximatif en silence.

    `exponential`, `linear` et `matern-3_2` existent dans l'evaluation du
    noyau mais n'ont pas de derivee ecrite -- pas plus en x qu'en theta.
    """
    from kernels import uq_eval_Kernel, dR_dtheta
    X = np.zeros((4, 2))
    opts = {"Handle": uq_eval_Kernel, "Family": "exponential",
            "Type": "separable", "Isotropic": False, "Nugget": 0.0}
    with pytest.raises(ValueError, match="famille"):
        dR_dtheta(X, np.array([1.0, 1.0]), opts)


# --------------------------------------------------------------------- #
# 2. le gradient de la VRAISEMBLANCE, sur le vrai probleme
# --------------------------------------------------------------------- #
def _captures(case="flexion", kind="PCK", combien=6):
    """Les entrees des vrais appels a l'optimiseur, pas un cas fabrique."""
    import kriging as _kr
    import fit as _fit
    from reference.limit_states import CASES

    capture = []
    vrai = _kr.kriging_optimize_theta

    def espion(params, theta0, bornes, method="gradbased"):
        if method.lower() == "gradbased" and len(capture) < combien:
            capture.append((params, np.asarray(theta0, float).copy()))
        return vrai(params, theta0, bornes, method)

    _kr.kriging_optimize_theta = espion
    _fit.kriging_optimize_theta = espion
    try:
        with open(os.path.join(_REPO, "tests", "golden", case + ".json"),
                  encoding="utf-8") as fh:
            ref = json.load(fh)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            harness.fit(kind, np.asarray(ref["doe"]), CASES[case](),
                        max_degree=ref["max_degree"])
    finally:
        _kr.kriging_optimize_theta = vrai
        _fit.kriging_optimize_theta = vrai
    return capture


@pytest.mark.parametrize("case", ["flexion", "linear"])
def test_grad_J_coincide_avec_les_differences_finies(case):
    """LE temoin du correctif.

    On prend le MEILLEUR des pas utiles, et non un pas fixe : quand le
    gradient est petit devant le bruit de J, un pas donne peut etre domine
    par ce bruit. C'est ce qui s'est produit le 02/09 sur le premier
    ajustement -- 6.05e-02 a 1e-5, 8.00e-04 a 1e-3. Prendre le meilleur
    revient a se placer au minimum de la courbe en U, ce qui est la seule
    comparaison qui ait un sens.
    """
    import kriging as _kr

    captures = _captures(case=case)
    assert captures, "aucun appel gradbased capture"

    testes, ecartes = 0, 0
    for params, theta0 in captures:
        g = _kr.grad_J_of_theta_ML(theta0, params)
        if g is None:
            continue                      # cas non couvert : le repli sert

        def J(t):
            return _kr.uq_Kriging_eval_J_of_theta_ML(np.asarray(t, float),
                                                     params)

        estimations = [_fd(J, theta0, rel) for rel in PAS_UTILES]

        # LA REFERENCE EST-ELLE UTILISABLE ? Une derivee honnete a un
        # PLATEAU : trois pas espaces d'une decade doivent donner a peu pres
        # la meme valeur. Quand ils n'en donnent pas -- a sigma^2 sous le
        # plancher de cancellation, les valeurs sautaient de 1e+05 a 1e+01 --
        # ce n'est pas le gradient analytique qu'on mesure, c'est le bruit de
        # J. On l'ecarte, et on le COMPTE.
        echelle_fd = max(float(np.max(np.abs(np.array(estimations)))), 1e-30)
        dispersion = float(np.max(np.abs(np.array(estimations)
                                         - np.array(estimations[-1])))) / echelle_fd
        if dispersion > 0.1:
            ecartes += 1
            continue

        testes += 1
        meilleur = min(
            float(np.max(np.abs(g - fd)))
            / max(float(np.max(np.abs(g))), float(np.max(np.abs(fd))), 1e-30)
            for fd in estimations)
        assert meilleur < 1e-2, (
            "%s : ecart relatif %.2e entre le gradient analytique et les "
            "differences finies, au meilleur des pas %s -- et la reference "
            "FD, elle, est stable (dispersion %.1e). Le correctif repose "
            "ENTIEREMENT sur la justesse de ce gradient."
            % (case, meilleur, PAS_UTILES, dispersion))

    assert testes, (
        "aucun cas verifiable sur %r : soit le gradient analytique ne "
        "s'applique nulle part, soit tous les points ont un sigma^2 sous "
        "%.0e -- auquel cas c'est `J` qui n'est pas calculable, pas le "
        "gradient qui est faux." % (case, SIGMA2_PLANCHER))


def test_GEPCK_rend_None_plutot_qu_un_gradient_faux():
    """La garde qui evite le pire.

    La matrice de Gram de GEPCK est AUGMENTEE -- N*(M+1) lignes -- et porte
    les blocs de derivees du noyau. `dR_dtheta` construit la Gram simple : la
    formule ne s'y applique pas. Sans cette garde, le calcul levait sur une
    incompatibilite de dimensions (72x72 contre 24x24) ; avec une matrice qui
    aurait par hasard la bonne taille, il aurait rendu un gradient FAUX.
    """
    import kriging as _kr
    captures = _captures(case="flexion", kind="GEPCK", combien=3)
    assert captures, "aucun appel gradbased capture en GEPCK"
    for params, theta0 in captures:
        assert _kr.grad_J_of_theta_ML(theta0, params) is None, (
            "grad_J_of_theta_ML rend un gradient pour GEPCK. Soit la derivee "
            "du noyau augmente a ete ecrite -- et ce temoin est a mettre a "
            "jour -- soit la garde a saute et le gradient est FAUX.")


def test_le_repli_ne_rend_jamais_un_gradient_nul():
    """Un gradient nul ferait croire a L-BFGS-B qu'il a converge.

    C'est exactement le defaut qu'on ferme : l'optimiseur s'arretait sur
    place. Le repli en differences finies de pas relatif doit donc CALCULER,
    jamais abdiquer.
    """
    import kriging as _kr
    captures = _captures(case="flexion", kind="GEPCK", combien=2)
    assert captures
    params, theta0 = captures[0]
    lb = np.full(theta0.size, 0.01)

    def J(t):
        return _kr.uq_Kriging_eval_J_of_theta_ML(np.asarray(t, float), params)

    jac = _kr._jacobien(J, params, lb)
    g = np.asarray(jac(theta0), dtype=float)
    assert np.all(np.isfinite(g)), g
    assert np.any(np.abs(g) > 0.0), (
        "le repli rend un gradient identiquement nul : L-BFGS-B conclurait "
        "a la convergence sur place.")
