"""
Base polynomiale du chaos -- extrait de branche3.py.

Correspondance loi marginale <-> famille de polynomes, ensemble des
multi-indices jusqu'a un degre, et matrice de design Psi.

PHASE 3 du plan de nettoyage : `branche3.py` melait le krigeage, la base
polynomiale et l'orchestration de l'ajustement. Le graphe d'appels
(`tools/analyse_dependances.py`) montre une stratification nette --
`fit` appelle `kriging` et `pce_basis`, qui ne s'appellent pas entre eux.

Le corps des fonctions est repris VERBATIM : la scission ne doit changer
aucun resultat, et la baseline le verifie.
"""

import numpy as np

from polynomials import uq_PCE_create_Psi, uq_PCK_eval_unipoly
from transform import uq_GeneralIsopTransform


# ===========================================================================
# 9.  PCE basis utilities (support for uq_PCK_calculate_coefficients)
# ===========================================================================

def poly_type_from_marginal(mtype):
    """
    Map UQLab marginal type string → polynomial family.
    Mirrors uq_poly_marginals logic.
    """
    m = mtype.lower()
    if m in ('uniform',):
        return 'Legendre'
    if m in ('gaussian', 'normal'):
        return 'Hermite'
    if m in ('exponential', 'gamma'):
        return 'Laguerre'
    # default: Legendre
    return 'Legendre'


def aux_marginal_from_poly_type(poly_type):
    """
    Canonical marginal for the auxiliary (polynomial) space.
    Legendre  → Uniform(-1, 1)
    Hermite   → Gaussian(0, 1)
    Laguerre  → Gamma / Exponential (approx Uniform for simplicity)
    """
    pt = poly_type.lower()
    if pt == 'legendre':
        return {'Type': 'Uniform',   'Parameters': [-1.0, 1.0]}
    if pt == 'hermite':
        return {'Type': 'Gaussian',  'Parameters': [0.0,  1.0]}
    return {'Type': 'Uniform', 'Parameters': [-1.0, 1.0]}


def pce_multi_indices(M, max_degree):
    """
    Generate all M-dimensional multi-indices alpha with |alpha|_1 <= max_degree.
    Returns ndarray of shape (P, M).
    """
    if M == 1:
        return np.arange(max_degree + 1).reshape(-1, 1)

    result = []
    # Recursive enumeration via a stack
    stack  = [([], max_degree)]  # (current_alpha, remaining_degree)
    while stack:
        prefix, remaining = stack.pop()
        if len(prefix) == M:
            result.append(prefix)
            continue
        dims_left = M - len(prefix)
        for v in range(remaining + 1):
            if dims_left == 1:
                result.append(prefix + [v])
            else:
                stack.append((prefix + [v], remaining - v))

    # Sort by total degree then lexicographically
    arr = np.array(result)
    key = arr.sum(axis=1) * (max_degree + 1)**M + np.dot(
        arr, (max_degree + 1)**np.arange(M - 1, -1, -1))
    return arr[np.argsort(key)]


def pce_eval_design_matrix(X, Indices, PolyTypes, orig_marginals, orig_copula,
                            aux_marginals):
    """
    Evaluate the PCE design matrix Psi (N × P).

    Parameters
    ----------
    X             : (N, M) ndarray in original input space
    Indices       : (P, M) ndarray of multi-indices
    PolyTypes     : list of M strings ('Legendre', 'Hermite', …)
    orig_marginals: list of M dicts from Input object
    orig_copula   : copula dict
    aux_marginals : list of M canonical marginal dicts (for polynomial space)
    """
    aux_copula = {'Type': 'Independent',
                  'Parameters': np.eye(len(aux_marginals))}
    U   = uq_GeneralIsopTransform(
        X, orig_marginals, orig_copula, aux_marginals, aux_copula)
    uv  = uq_PCK_eval_unipoly(U, Indices, PolyTypes)
    Psi = uq_PCE_create_Psi(Indices, uv)
    return Psi, U
