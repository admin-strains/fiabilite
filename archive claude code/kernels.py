"""
kernels.py — Fonctions de corrélation pour le Kriging
Fidèle à uq_eval_Kernel.m et uq_assemble_Kernel.m de UQLab

Type ellipsoïdal uniquement (défaut UQLab pour PC-Kriging) :
    h = seuclidean(x1, x2, theta)  distance euclidienne standardisée
    K(h) selon la famille choisie
"""

import numpy as np
from scipy.spatial.distance import cdist


def eval_kernel(X1: np.ndarray, X2: np.ndarray, theta: np.ndarray,
                family: str = 'matern-5_2', nugget: float = 1e-4) -> np.ndarray:
    """
    Calcule la matrice de corrélation K(X1, X2, theta).

    Parameters
    ----------
    X1 : (N1, M)
    X2 : (N2, M)
    theta : (M,) longueurs de corrélation (anisotrope) ou scalaire (isotrope)
    family : 'matern-5_2' | 'matern-3_2' | 'gaussian' | 'exponential'
    nugget : ajouté sur la diagonale uniquement si X1 is X2 (Gram matrix)

    Returns
    -------
    K : (N1, N2)
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    theta = np.atleast_1d(theta).astype(float)

    N1, M = X1.shape
    N2 = X2.shape[0]

    # Distance euclidienne standardisée : h_ij = ||x1_i - x2_j||_{theta}
    # scipy cdist avec 'seuclidean' utilise 1/V comme poids, on lui passe theta² comme variances
    # Mais UQLab calcule h = pdist2(X1,X2,'seuclidean',theta')
    # ce qui donne h_ij = sqrt(sum((x1_i - x2_j)^2 / theta_k^2))
    # → on passe V = theta**2
    if theta.size == 1:
        theta = np.repeat(theta, M)

    h = cdist(X1, X2, metric='seuclidean', V=theta**2)

    K = _apply_kernel(h, family)

    # Nugget sur la diagonale (Gram matrix uniquement)
    is_gram = (N1 == N2) and np.array_equal(X1, X2)
    if is_gram and nugget > 0:
        K += nugget * np.eye(N1)

    return K


def _apply_kernel(h: np.ndarray, family: str) -> np.ndarray:
    """Applique le kernel scalaire sur la matrice de distances h (ellipsoïdal)."""
    family = family.lower()
    if family == 'matern-5_2':
        sqrt5 = np.sqrt(5.0)
        return (1.0 + sqrt5 * h + (5.0 / 3.0) * h**2) * np.exp(-sqrt5 * h)
    elif family == 'matern-3_2':
        sqrt3 = np.sqrt(3.0)
        return (1.0 + sqrt3 * h) * np.exp(-sqrt3 * h)
    elif family == 'gaussian':
        return np.exp(-0.5 * h**2)
    elif family == 'exponential':
        return np.exp(-h)
    else:
        raise ValueError(f"Famille de kernel inconnue : '{family}'. "
                         "Choisir parmi : 'matern-5_2', 'matern-3_2', 'gaussian', 'exponential'")
