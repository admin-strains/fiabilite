"""
La convention `der` / `der_prime` des noyaux -- tranchee, et ancree ici.

DEFAUT 4 du plan de nettoyage. `kernels.uq_eval_global_Kernel` contredisait sa
propre docstring : celle-ci annoncait, pour le bloc `cb=l` de `r0_tilde`,
`dk(X_test, X_train)/dX_test_{l-1}` obtenu par `(der=l-1, dp=None)`, tandis que
le code appelait `(der=None, dp=l-1)`. Meme inversion dans la docstring de
`uq_assemble_global_Kernel`.

Tant que la convention restait ambigue, les defauts 2 et 3 -- interpolation
GEPCK a 3,0e-03 et erreur de 1,30 % sur beta -- n'etaient pas diagnosticables :
impossible de dire si le krigeage etait mal conditionne ou s'il resolvait un
systeme faux.

VERDICT, mesure le 26/08/2026
------------------------------
Le CODE avait raison, les DEUX docstrings etaient fausses. La raison est
mathematique : la l-ieme observation augmentee est la derivee AU POINT
D'APPRENTISSAGE, donc

    Cov( y(x*), dy/dx_l (x^j) )  =  d k(x*, x^j) / d x^j_l

soit une derivee par rapport au SECOND argument -- `der=None, dp=l`.

Ces tests l'etablissent sans rien supposer du code : par differences finies,
puis par la propriete structurelle qui en decoule.

Le test hérité `tests/unit/test_eval_global_kernel.py` encode la convention
inverse. Il reste en `xfail(strict)` et n'est PAS corrige : les suites
d'origine servent de temoin, on ne rewrite pas un temoin. Ce fichier-ci porte
la verite.

Ces tests ne demandent que numpy.
"""

import os
import sys

import numpy as np
import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
for p in (os.path.join(REPO, "_lib"),):
    if p not in sys.path:
        sys.path.insert(0, p)

from kernels import (kernel_deriv_factory, uq_assemble_global_Kernel,  # noqa: E402
                     uq_eval_global_Kernel)

FAMILLES = ("gaussian", "matern-5_2")
#: pas de derivation. 1e-6 place l'erreur de troncature et l'erreur d'arrondi
#: au meme ordre pour une difference centree en double precision.
EPS = 1e-6


def _jeu(graine=0, n1=4, n2=5, m=2):
    rng = np.random.default_rng(graine)
    return (rng.normal(size=(n1, m)), rng.normal(size=(n2, m)),
            np.array([1.3, 0.7])[:m])


# --------------------------------------------------------------------------- #
# 1. Ce que calcule reellement kernel_deriv_factory                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("famille", FAMILLES)
@pytest.mark.parametrize("l", [0, 1])
def test_der_prime_derive_le_second_argument(famille, l):
    """`(der=None, dp=l)` doit valoir dk/dX2_l, verifie par difference finie."""
    X1, X2, theta = _jeu()
    k = kernel_deriv_factory(famille, None, None)
    plus, moins = X2.copy(), X2.copy()
    plus[:, l] += EPS
    moins[:, l] -= EPS
    fd = (k(X1, plus, theta) - k(X1, moins, theta)) / (2 * EPS)

    obtenu = kernel_deriv_factory(famille, None, l)(X1, X2, theta)
    assert np.max(np.abs(fd - obtenu)) < 1e-8, \
        "(der=None, dp=%d) ne derive pas le second argument" % l


@pytest.mark.parametrize("famille", FAMILLES)
@pytest.mark.parametrize("l", [0, 1])
def test_der_derive_le_premier_argument(famille, l):
    """Le pendant : `(der=l, dp=None)` doit valoir dk/dX1_l."""
    X1, X2, theta = _jeu()
    k = kernel_deriv_factory(famille, None, None)
    plus, moins = X1.copy(), X1.copy()
    plus[:, l] += EPS
    moins[:, l] -= EPS
    fd = (k(plus, X2, theta) - k(moins, X2, theta)) / (2 * EPS)

    obtenu = kernel_deriv_factory(famille, l, None)(X1, X2, theta)
    assert np.max(np.abs(fd - obtenu)) < 1e-8


@pytest.mark.parametrize("famille", FAMILLES)
@pytest.mark.parametrize("l", [0, 1])
def test_les_deux_conventions_ne_sont_pas_interchangeables(famille, l):
    """Sans quoi le defaut 4 n'aurait ete qu'une question de style. Les deux
    derivees sont opposees : confondre l'une pour l'autre change le signe."""
    X1, X2, theta = _jeu()
    a = kernel_deriv_factory(famille, None, l)(X1, X2, theta)
    b = kernel_deriv_factory(famille, l, None)(X1, X2, theta)
    assert np.max(np.abs(a - b)) > 0.1, "les deux conventions coincident ?"
    assert np.allclose(a, -b, atol=1e-12), \
        "attendu a = -b : dk/dx' = -dk/dx pour un noyau stationnaire"


# --------------------------------------------------------------------------- #
# 2. Ce que le bloc-ligne rb=0 doit porter                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("famille", FAMILLES)
def test_r0_tilde_derive_le_point_d_apprentissage(famille):
    """La l-ieme observation augmentee est dy/dx_l AU POINT D'APPRENTISSAGE.
    `r0_tilde[:, l*n2:(l+1)*n2]` doit donc valoir dk(X_test, X_train)/dX_train_l
    -- ce que la docstring d'origine niait."""
    X1, X2, theta = _jeu()
    n2, m = X2.shape[0], X2.shape[1]
    r0 = uq_eval_global_Kernel(X1, X2, theta, {"Family": famille, "Nugget": 0.0})
    assert r0.shape == (X1.shape[0], n2 * (m + 1))

    k = kernel_deriv_factory(famille, None, None)
    assert np.allclose(r0[:, :n2], k(X1, X2, theta), atol=1e-14)
    for l in range(m):
        plus, moins = X2.copy(), X2.copy()
        plus[:, l] += EPS
        moins[:, l] -= EPS
        fd = (k(X1, plus, theta) - k(X1, moins, theta)) / (2 * EPS)
        bloc = r0[:, (l + 1) * n2:(l + 2) * n2]
        assert np.max(np.abs(fd - bloc)) < 1e-8, \
            "bloc cb=%d : ce n'est pas la derivee du point d'apprentissage" % (l + 1)


@pytest.mark.parametrize("famille", FAMILLES)
def test_le_bloc_ligne_zero_de_R_tilde_dit_la_meme_chose(famille):
    """`uq_assemble_global_Kernel` et `uq_eval_global_Kernel` doivent employer
    la MEME convention : c'est ce qui rend l'interpolation possible."""
    X1, X2, theta = _jeu()
    n1 = X1.shape[0]
    R = uq_assemble_global_Kernel(X1, X2, theta, famille)
    r0 = uq_eval_global_Kernel(X1, X2, theta, {"Family": famille, "Nugget": 0.0})
    assert np.max(np.abs(R[:n1, :] - r0)) < 1e-13


# --------------------------------------------------------------------------- #
# 3. La consequence : la condition NECESSAIRE d'interpolation exacte          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("famille", FAMILLES)
def test_r0_tilde_evalue_sur_le_doe_est_le_haut_de_R_tilde(famille):
    """Si le metamodele est evalue EXACTEMENT sur ses points d'apprentissage,
    `r0_tilde` doit reproduire les n premieres lignes de `R_tilde` -- alors
    `r0 @ R^-1 @ y_aug` rend `y` exactement.

    Cette propriete tient a 1e-15. L'interpolation GEPCK, elle, ne tient qu'a
    3,0e-03 (defaut 2) : la cause n'est donc PAS la convention des derivees,
    mais la resolution du systeme. C'est ce que ce test etablit, et c'est ce
    qui rend le defaut 2 diagnosticable.
    """
    X, _, theta = _jeu(graine=1, n1=6, n2=6)
    n, m = X.shape
    opts = {"Family": famille, "Nugget": 0.0}
    R = uq_eval_global_Kernel(X, X, theta, opts)
    assert R.shape == (n * (m + 1), n * (m + 1))

    #: forcer la branche non-Gram sans changer les valeurs de facon sensible
    Xbis = X + 0.0
    Xbis[0, 0] += 1e-15
    r0 = uq_eval_global_Kernel(Xbis, X, theta, opts)
    assert np.max(np.abs(r0 - R[:n, :])) < 1e-13


@pytest.mark.parametrize("famille", FAMILLES)
def test_R_tilde_est_symetrique(famille):
    """Une matrice de Gram qui ne l'est pas signale une convention incoherente
    entre le bloc (rb, cb) et le bloc (cb, rb)."""
    X, _, theta = _jeu(graine=2, n1=5, n2=5)
    R = uq_eval_global_Kernel(X, X, theta, {"Family": famille, "Nugget": 0.0})
    assert np.max(np.abs(R - R.T)) < 1e-14
