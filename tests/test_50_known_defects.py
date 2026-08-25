"""
Defauts connus, ecrits comme des specifications executables.

Chaque test decrit le comportement ATTENDU (pas le comportement actuel) et
porte xfail(strict=True). Consequence : la suite est verte aujourd'hui, et le
jour ou quelqu'un corrige le defaut, le test passe -> xpass -> ECHEC, ce qui
oblige a retirer le marqueur et a acter la correction. Aucun defaut ne peut
etre corrige en silence, ni reapparaitre sans bruit.

Ne rien mettre ici sans : symptome reproductible, localisation fichier:ligne,
et effet concret sur la production.
"""

import numpy as np
import pytest

import harness


@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "branche5.uq_eval_global_Kernel l.1360 : isGram = (n1 == n2) and "
    "np.array_equal(X1, X2). Evaluer un GEPCK exactement sur son propre DOE "
    "prend la branche Gram et renvoie R_tilde (N*(M+1), N*(M+1)) au lieu de "
    "r0_tilde (N, N*(M+1)) ; branche4.py l.206 leve alors "
    "'operands could not be broadcast together with shapes (N,) (N*(M+1),)'. "
    "Impact production : tout appel du metamodele sur un point deja au DOE "
    "plante -- grille EFF passant par un point du DOE, relecture de cache, "
    "verification d'interpolation. Le contournement actuel est involontaire "
    "(les points tombent rarement pile sur le DOE)."))
def test_gepck_sevalue_sur_son_propre_doe(fitted, doe24, flexion_ls):
    g_hat, _, _ = harness.predictors('GEPCK', fitted['GEPCK'])
    mu = g_hat(doe24)
    assert np.allclose(mu, flexion_ls.g(doe24), atol=1e-8)


@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "Le krigeage sans pepite doit interpoler ses points d'apprentissage. "
    "Mesure sur la branche cleaning (DOE 24 pts, flexion, Matern-5/2, "
    "Mode=optimal) : ecart max 2.96e-03 pour GEPCK contre 5.31e-07 pour PCK, "
    "soit 4 ordres de grandeur. Les gradients, eux, interpolent a 6.9e-06. "
    "Piste : conditionnement de R_tilde (theta ~ 60-75 sur 24 points, blocs "
    "de derivees) dans branche3.uq_GEPCK_calculate_coefficients. "
    "Impact production : c'est la meme cause probable que l'ecart de 1.3 % "
    "sur beta (GEPCK) contre 0.011 % (PCK) mesure par test_40, alors que "
    "GEPCK dispose de PLUS d'information que PCK."))
def test_gepck_interpole_son_doe(fitted, doe24, flexion_ls):
    g_hat, _, _ = harness.predictors('GEPCK', fitted['GEPCK'])
    err = np.abs(g_hat(doe24 + 1e-9) - flexion_ls.g(doe24)).max()
    assert err < 1e-6, f'erreur d interpolation GEPCK = {err:.3e}'


@pytest.mark.defect
def test_pck_interpole_son_doe(fitted, doe24, flexion_ls):
    """Temoin : le meme controle passe pour PCK. Sert a prouver que le test
    ci-dessus mesure un defaut de GEPCK et non un artefact du harness."""
    g_hat, _, _ = harness.predictors('PCK', fitted['PCK'])
    err = np.abs(g_hat(doe24 + 1e-9) - flexion_ls.g(doe24)).max()
    assert err < 1e-5, f'erreur d interpolation PCK = {err:.3e}'


@pytest.mark.defect
def test_gradient_gepck_est_juste(fitted, flexion_ls):
    """
    Contre-expertise du gradient analytique GEPCK.

    A ne pas confondre avec un test naif : l'ecart au FD AUGMENTE quand le pas
    diminue (9.7e-07 a h=1e-3, 4.1e-03 a h=1e-7), signature d'un bruit
    d'arrondi de la prediction et non d'un gradient faux. On compare donc au
    pas ou le FD est le plus fiable, et on verifie la decroissance.
    """
    g_hat, grad_ana, _ = harness.predictors('GEPCK', fitted['GEPCK'])
    U = np.array([[1.0, -1.0], [-2.0, 0.5], [0.25, 2.75]])
    G = grad_ana(U)

    def fd(h):
        F = np.zeros_like(G)
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            F[:, j] = (g_hat(U + e) - g_hat(U - e)) / (2 * h)
        return np.abs(G - F).max()

    assert fd(1e-3) < 1e-5, 'gradient analytique GEPCK faux au pas le plus fiable'
    assert fd(1e-3) < fd(1e-6), \
        'le FD ne se degrade plus quand h diminue : reexaminer le diagnostic'
