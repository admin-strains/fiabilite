"""
Conversion GEPCK complet -> patches de Bernstein (sans grille tensorielle).

Remplace le pipeline :
  grille N_GRID^n -> B-spline -> ndspline_to_bernstein_patches

par une evaluation directe du GEPCK aux points de Bernstein de chaque
sous-boite, suivie d'une inversion Vandermonde. Le modele complet
(PCE + Kriging) est inclus dans les patches d'exclusion PP.

Cout : k_breaks^n * (degree+1)^n evaluations GEPCK au lieu de N_GRID^n.
Pour n=2 : k_breaks=20, degree=3 -> 6400 evals (equivalent grille 80x80).
Pour n=4 : k_breaks=5, degree=3 -> 160 000 evals (vs grille 80^4 = 40M).
"""

import numpy as np
from projected_polyhedron import BernsteinPatch


def _bernstein_vandermonde_1d(d):
    """
    Matrice de Vandermonde-Bernstein (d+1)x(d+1).
    M[i, j] = B_{j,d}(i/d) = C(d,j) * (i/d)^j * (1-i/d)^{d-j}
    Utilisee pour inverser la relation vals = M @ bern @ M.T.
    """
    from math import comb
    M = np.zeros((d + 1, d + 1))
    for i in range(d + 1):
        t = i / d if d > 0 else 0.5
        for j in range(d + 1):
            M[i, j] = comb(d, j) * (t ** j) * ((1.0 - t) ** (d - j))
    return M


def gepck_to_bernstein_patches(predict_fn, domain, k_breaks, degree=3):
    """
    Convertit un modele callable (ex : GEPCK complet) en patches de Bernstein 2D.

    Evalue predict_fn directement aux points de Bernstein de chaque sous-boite,
    sans grille tensorielle N_GRID^2. Le modele complet (PCE + Kriging) est
    inclus dans les patches d'exclusion PP.

    Parametres
    ----------
    predict_fn : callable (N, 2) -> (N,)
        Evaluateur du modele. Ex: lambda pts: predict_gepck(fm, pts).flatten()
    domain : ((a1, b1), (a2, b2))
        Domaine physique a couvrir.
    k_breaks : int
        Nombre de sous-boites par dimension (k_breaks^2 patches au total).
    degree : int (defaut 3)
        Degre Bernstein par dimension.

    Cout : k_breaks^2 * (degree+1)^2 evaluations de predict_fn.
    Ex  : k_breaks=20, degree=3 -> 20^2 * 4^2 = 6400 evals (equivalent a grille 80x80)

    Retourne
    --------
    patches : list of BernsteinPatch
    """
    (a1, b1), (a2, b2) = domain
    u1_breaks = np.linspace(a1, b1, k_breaks + 1)
    u2_breaks = np.linspace(a2, b2, k_breaks + 1)

    d = degree
    M = _bernstein_vandermonde_1d(d)
    M_inv = np.linalg.inv(M)

    patches = []
    for i in range(k_breaks):
        for j in range(k_breaks):
            ba1, bb1 = u1_breaks[i], u1_breaks[i + 1]
            ba2, bb2 = u2_breaks[j], u2_breaks[j + 1]
            if bb1 - ba1 < 1e-15 or bb2 - ba2 < 1e-15:
                continue

            # Points de Bernstein dans cette boite
            pts = np.array([
                [ba1 + ki / d * (bb1 - ba1),
                 ba2 + kj / d * (bb2 - ba2)]
                for ki in range(d + 1)
                for kj in range(d + 1)
            ], dtype=float)   # shape ((d+1)^2, 2)

            vals = predict_fn(pts).reshape(d + 1, d + 1)

            # Inversion Vandermonde : vals = M @ true_bern @ M.T
            true_bern = M_inv @ vals @ M_inv.T

            patches.append(BernsteinPatch(true_bern, ((ba1, bb1), (ba2, bb2))))

    return patches


def f2_gepck_to_bernstein_patches(grad_fn, domain, k_breaks, degree=3):
    """
    Patches de Bernstein pour f2(u) = u1*dg/du2 - u2*dg/du1  (cas n=2).

    f2 est de degre degree+1 (produit lineaire x gradient).
    On utilise (degree+2)^2 points d'evaluation par boite.

    Parametres
    ----------
    grad_fn : callable (N, 2) -> (N, 2)
        Gradient complet : G[:, 0] = dg/du1, G[:, 1] = dg/du2.
        Ex: lambda pts: predict_gradient_gepck(fm, pts)
    domain : ((a1, b1), (a2, b2))
    k_breaks : int
    degree : int (degre de g ; f2 utilisera degree+1)

    Cout : k_breaks^2 * (degree+2)^2 evaluations de grad_fn.

    Retourne
    --------
    patches_f2 : list of BernsteinPatch (de degre degree+1 par dimension)
    """
    (a1, b1), (a2, b2) = domain
    u1_breaks = np.linspace(a1, b1, k_breaks + 1)
    u2_breaks = np.linspace(a2, b2, k_breaks + 1)

    d = degree + 1   # degre de f2
    M = _bernstein_vandermonde_1d(d)
    M_inv = np.linalg.inv(M)

    patches = []
    for i in range(k_breaks):
        for j in range(k_breaks):
            ba1, bb1 = u1_breaks[i], u1_breaks[i + 1]
            ba2, bb2 = u2_breaks[j], u2_breaks[j + 1]
            if bb1 - ba1 < 1e-15 or bb2 - ba2 < 1e-15:
                continue

            pts = np.array([
                [ba1 + ki / d * (bb1 - ba1),
                 ba2 + kj / d * (bb2 - ba2)]
                for ki in range(d + 1)
                for kj in range(d + 1)
            ], dtype=float)   # shape ((d+1)^2, 2)

            G = grad_fn(pts)   # shape ((d+1)^2, 2)
            f2_vals = (pts[:, 0] * G[:, 1] - pts[:, 1] * G[:, 0]).reshape(d + 1, d + 1)

            true_bern = M_inv @ f2_vals @ M_inv.T

            patches.append(BernsteinPatch(true_bern, ((ba1, bb1), (ba2, bb2))))

    return patches