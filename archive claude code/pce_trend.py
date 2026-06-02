"""
pce_trend.py — Trend PCE pour PC-Kriging
Traduction directe de UQLab MATLAB (Sudret, ETH Zurich) en Python pur.
Aucune dépendance externe hormis numpy et scikit-learn.

Fichiers MATLAB sources traduits :
  - uq_poly_rec_coeffs.m    → _rec_coeffs_hermite / _rec_coeffs_legendre
  - uq_eval_rec_rule.m      → _eval_rec_rule
  - uq_generate_basis_Apmj.m→ _generate_multi_indices
  - uq_PCE_lars.m           → _lars_fit (via sklearn + LOO analytique)

Distributions supportées :
  - 'uniform'  [a, b]        → Legendre orthonormaux sur [-1, 1]
  - 'gaussian' [mu, sigma]   → Hermite orthonormaux sur (-∞, +∞)
  - 'lognormal'[mu, sigma]   → transform log → Hermite

Usage :
  pce = PCETrend(distributions, degree=range(1, 4))
  pce.fit(X, Y)
  F = pce.eval_active(X)   # matrice design (N, n_poly_actifs)
"""

import numpy as np
from itertools import product as iproduct
from sklearn.linear_model import Lars


# =============================================================================
# 1. COEFFICIENTS DE RÉCURRENCE (uq_poly_rec_coeffs.m)
# =============================================================================

def _rec_coeffs_hermite(max_degree: int) -> np.ndarray:
    """
    Coefficients de récurrence pour les polynômes de Hermite orthonormaux.
    Distribution associée : Gaussian N(0,1).

    Traduit depuis uq_poly_rec_coeffs.m :
        a_n     = 0           pour tout n
        sqrt_b0 = 1
        sqrt_bn = sqrt(n)     pour n >= 1

    Retourne AB : (max_degree+1, 2)
        AB[:, 0] = a_n
        AB[:, 1] = sqrt_b_n
    """
    n = max_degree + 1
    AB = np.zeros((n, 2))
    AB[:, 0] = 0.0                             # a_n = 0
    AB[0, 1] = 1.0                             # sqrt_b0 = 1
    AB[1:, 1] = np.sqrt(np.arange(1, n))       # sqrt_bn = sqrt(n)
    return AB


def _rec_coeffs_legendre(max_degree: int) -> np.ndarray:
    """
    Coefficients de récurrence pour les polynômes de Legendre orthonormaux.
    Distribution associée : Uniform[-1, 1].

    Traduit depuis uq_poly_rec_coeffs.m :
        a_n     = 0
        sqrt_b0 = 1
        sqrt_bn = sqrt(1 / (4 - n^{-2}))   pour n >= 1

    Retourne AB : (max_degree+1, 2)
    """
    n = max_degree + 1
    AB = np.zeros((n, 2))
    AB[:, 0] = 0.0                                                # a_n = 0
    AB[0, 1] = 1.0                                                # sqrt_b0 = 1
    ns = np.arange(1, n, dtype=float)
    AB[1:, 1] = np.sqrt(1.0 / (4.0 - ns**(-2)))                  # sqrt_bn
    return AB


# =============================================================================
# 2. ÉVALUATION PAR RÉCURRENCE (uq_eval_rec_rule.m)
# =============================================================================

def _eval_rec_rule(u: np.ndarray, AB: np.ndarray) -> np.ndarray:
    """
    Évalue les polynômes orthonormaux par récurrence à 3 termes.
    Traduction exacte de uq_eval_rec_rule.m (Sudret).

    Récurrence :
        P_{-1}  = 0
        P_0     = 1 / AB[0, 1]
        P_{k+1} = ((u - AB[k,0]) * P_k - P_{k-1} * AB[k,1]) / AB[k+1, 1]

    Parameters
    ----------
    u  : (N,)  points dans l'espace standard
    AB : (P+1, 2)  coefficients de récurrence

    Returns
    -------
    P : (N, P+1)   P[:, k] = k-ème polynôme évalué en u
    """
    u = np.asarray(u).ravel()
    N = len(u)
    max_degree = AB.shape[0] - 1
    P = np.zeros((N, max_degree + 1))

    # P_0 = 1 / sqrt_b0
    P[:, 0] = 1.0 / AB[0, 1]

    if max_degree == 0:
        return P

    # P_1 : k=0 → P_{-1} = 0
    P[:, 1] = ((u - AB[0, 0]) * P[:, 0]) / AB[1, 1]

    # P_{k+1} pour k >= 1
    for k in range(1, max_degree):
        P[:, k + 1] = (
            (u - AB[k, 0]) * P[:, k] - P[:, k - 1] * AB[k, 1]
        ) / AB[k + 1, 1]

    return P


# =============================================================================
# 3. TRANSFORMATION VERS L'ESPACE STANDARD
# =============================================================================

def _to_standard(x: np.ndarray, dist: dict) -> np.ndarray:
    """
    Transforme une variable physique x vers l'espace standard
    de son polynôme orthogonal.

    Uniform[a, b]      → u ∈ [-1, 1]   : u = 2*(x-a)/(b-a) - 1
    Gaussian(mu, sig)  → u ∈ (-∞,+∞)   : u = (x - mu) / sig
    LogNormal(mu, sig) → u ∈ (-∞,+∞)   : u = (log(x) - mu) / sig
    """
    dtype = dist['type'].lower()
    p = dist['parameters']

    if dtype == 'uniform':
        a, b = p[0], p[1]
        return 2.0 * (x - a) / (b - a) - 1.0

    elif dtype in ('gaussian', 'normal'):
        mu, sigma = p[0], p[1]
        return (x - mu) / sigma

    elif dtype == 'lognormal':
        mu_log, sigma_log = p[0], p[1]
        return (np.log(x) - mu_log) / sigma_log

    else:
        raise ValueError(
            f"Distribution '{dtype}' non supportée. "
            "Choisir parmi : 'uniform', 'gaussian', 'lognormal'."
        )


def _poly_type(dist: dict) -> str:
    """Retourne le type de polynôme associé à une distribution."""
    dtype = dist['type'].lower()
    if dtype == 'uniform':
        return 'legendre'
    elif dtype in ('gaussian', 'normal', 'lognormal'):
        return 'hermite'
    else:
        raise ValueError(f"Distribution '{dtype}' non supportée.")


# =============================================================================
# 4. INDICES MULTI-DIMENSIONNELS (uq_generate_basis_Apmj.m)
# =============================================================================

def _generate_multi_indices(M: int, max_degree: int) -> np.ndarray:
    """
    Génère tous les multi-indices α ∈ N^M tels que |α| = Σ α_j ≤ max_degree.
    Inclut le terme constant α = (0, ..., 0).

    Fidèle à uq_generate_basis_Apmj.m (troncature totale |α| ≤ P).

    Returns
    -------
    indices : (n_poly, M)  chaque ligne = un multi-indice
    """
    indices = []
    for alpha in iproduct(range(max_degree + 1), repeat=M):
        if sum(alpha) <= max_degree:
            indices.append(alpha)
    # Tri : d'abord par degré total |α|, puis lexicographique
    indices.sort(key=lambda a: (sum(a), a))
    return np.array(indices, dtype=int)


# =============================================================================
# 5. MATRICE DE DESIGN Ψ
# =============================================================================

def _build_design_matrix(X: np.ndarray, distributions: list,
                          indices: np.ndarray, max_degree: int) -> np.ndarray:
    """
    Construit la matrice de design Ψ (N, n_poly).

    Ψ[n, α] = Π_{j=1}^M  P_{α_j}^{(j)}(u_{n,j})

    Étapes :
      1. Transformer chaque colonne de X dans l'espace standard → U (N, M)
      2. Évaluer les polynômes univariés pour chaque variable → val[j][N, max_degree+1]
      3. Produit tensoriel selon les multi-indices

    Parameters
    ----------
    X            : (N, M)
    distributions: liste de M dicts
    indices      : (n_poly, M)  multi-indices
    max_degree   : degré maximum

    Returns
    -------
    Psi : (N, n_poly)
    """
    N, M = X.shape
    n_poly = indices.shape[0]

    # Étape 1 + 2 : polynômes univariés par variable
    univ_vals = []  # univ_vals[j] : (N, max_degree+1)
    for j in range(M):
        u_j = _to_standard(X[:, j], distributions[j])
        ptype = _poly_type(distributions[j])
        if ptype == 'hermite':
            AB = _rec_coeffs_hermite(max_degree)
        else:
            AB = _rec_coeffs_legendre(max_degree)
        univ_vals.append(_eval_rec_rule(u_j, AB))  # (N, max_degree+1)

    # Étape 3 : produit tensoriel
    Psi = np.ones((N, n_poly))
    for p_idx in range(n_poly):
        for j in range(M):
            deg_j = indices[p_idx, j]
            Psi[:, p_idx] *= univ_vals[j][:, deg_j]

    return Psi


# =============================================================================
# 6. LARS AVEC LOO ANALYTIQUE (uq_PCE_lars.m)
# =============================================================================

def _lars_fit(Psi: np.ndarray, Y: np.ndarray) -> tuple:
    """
    Régression LARS avec LOO early stopping.
    Traduction de uq_PCE_lars.m + uq_lar.m (UQLab).

    Algorithme :
      1. LARS → chemin de coefficients (path)
      2. À chaque étape : refit OLS sur l'ensemble actif (hybrid LARS)
      3. LOO analytique via hat matrix
      4. Early stopping : arrêt si LOO ne s'améliore pas sur 2 étapes

    Parameters
    ----------
    Psi : (N, P)  matrice de design
    Y   : (N,)   réponses

    Returns
    -------
    ranked_indices : list[int]   indices actifs classés par |coeff| décroissant
    coefficients   : (P,)        coefficients OLS (zéro pour les inactifs)
    loo_error      : float        LOO normalisée par Var(Y)
    """
    N, P = Psi.shape
    Y_var = np.var(Y)
    if Y_var < 1e-20:
        Y_var = 1.0

    # LARS
    max_iter = min(N - 1, P)
    lars = Lars(fit_intercept=False,
                fit_path=True, n_nonzero_coefs=max_iter)
    lars.fit(Psi, Y)

    coef_path = lars.coef_path_   # (P, n_steps)
    n_steps = coef_path.shape[1]

    best_loo = np.inf
    best_active = []
    best_coeffs_a = np.array([])
    no_improve = 0

    for step in range(1, n_steps):
        c = coef_path[:, step]
        active_idx = np.where(np.abs(c) > 1e-12)[0].tolist()

        if len(active_idx) == 0:
            continue
        if len(active_idx) >= N:
            # Système sur-déterminé → LOO instable, on arrête
            break

        Psi_a = Psi[:, active_idx]   # (N, k)

        # Refit OLS sur l'ensemble actif (hybrid LARS = plus précis)
        coeffs_a, _, _, _ = np.linalg.lstsq(Psi_a, Y, rcond=None)

        residuals = Y - Psi_a @ coeffs_a

        # LOO analytique : e_loo[i] = residuals[i] / (1 - H[i,i])
        # H = Psi_a (Psi_aᵀ Psi_a)⁻¹ Psi_aᵀ
        # H[i,i] = ||Q[i,:]||²  avec [Q, R] = qr(Psi_a)
        try:
            Q, _ = np.linalg.qr(Psi_a, mode='reduced')   # (N, k)
            h_diag = np.sum(Q**2, axis=1)                 # (N,)
            h_diag = np.clip(h_diag, -0.9999, 0.9999)
            e_loo = residuals / (1.0 - h_diag)
            loo = float(np.mean(e_loo**2) / Y_var)
        except np.linalg.LinAlgError:
            loo = np.inf

        if loo < best_loo:
            best_loo = loo
            best_active = active_idx.copy()
            best_coeffs_a = coeffs_a.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 2:   # early stopping (UQLab : LarsEarlyStop)
                break

    # Vecteur de coefficients complet
    coefficients = np.zeros(P)
    if len(best_active) > 0:
        coefficients[best_active] = best_coeffs_a

    # Ranking par |coefficient| décroissant
    ranked = sorted(best_active, key=lambda i: abs(coefficients[i]), reverse=True)

    return ranked, coefficients, best_loo


# =============================================================================
# 7. CLASSE PCETrend
# =============================================================================

class PCETrend:
    """
    Trend PCE calibré par LARS avec LOO early stopping.
    Traduction fidèle de UQLab, sans aucune dépendance externe (numpy seul).

    Attributes
    ----------
    distributions : list[dict]
    degree_range  : list[int]
    ranked_indices: list[int]   indices des polynômes actifs (classés LARS)
    coefficients  : np.ndarray  coefficients OLS (P,)
    loo_error     : float       LOO normalisée du meilleur modèle
    _indices      : np.ndarray  multi-indices (n_poly, M)
    _max_degree   : int
    """

    def __init__(self, distributions: list, degree=range(1, 4)):
        self.distributions = distributions
        self.degree_range = list(degree)

        # Résultats du fit
        self.ranked_indices = None
        self.coefficients = None
        self.loo_error = None
        self._indices = None
        self._max_degree = None
        self._X_train = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Calibre le trend PCE sur (X, Y).

        Pour chaque degré dans degree_range :
          - génère les multi-indices
          - construit Ψ
          - calibre LARS + LOO
        Retient le degré avec le meilleur LOO.
        """
        X = np.atleast_2d(X)
        Y = np.asarray(Y).ravel()
        self._X_train = X.copy()
        M = X.shape[1]

        best_loo = np.inf
        best_result = None

        for deg in self.degree_range:
            indices = _generate_multi_indices(M, deg)
            Psi = _build_design_matrix(X, self.distributions, indices, deg)
            ranked, coeffs, loo = _lars_fit(Psi, Y)

            if loo < best_loo:
                best_loo = loo
                best_result = {
                    'indices': indices,
                    'coefficients': coeffs,
                    'ranked_indices': ranked,
                    'max_degree': deg,
                }

        self._indices = best_result['indices']
        self.coefficients = best_result['coefficients']
        self.ranked_indices = best_result['ranked_indices']
        self._max_degree = best_result['max_degree']
        self.loo_error = best_loo

    # ------------------------------------------------------------------
    def _eval(self, X: np.ndarray) -> np.ndarray:
        """Retourne Ψ(X) complet — (N, n_poly_total)."""
        if self._indices is None:
            raise RuntimeError("PCETrend non calibré. Appeler fit() d'abord.")
        X = np.atleast_2d(X)
        return _build_design_matrix(X, self.distributions,
                                    self._indices, self._max_degree)

    def eval_active(self, X: np.ndarray) -> np.ndarray:
        """Ψ réduit aux polynômes actifs sélectionnés — (N, n_active)."""
        Psi = self._eval(X)
        return Psi[:, self.ranked_indices]

    def eval_subset(self, X: np.ndarray, n_poly: int) -> np.ndarray:
        """
        Ψ réduit aux n_poly premiers polynômes du classement LARS — (N, n_poly).
        Utilisé par le mode 'optimal' de PCKriging.
        """
        Psi = self._eval(X)
        return Psi[:, self.ranked_indices[:n_poly]]

    @property
    def n_polynomials(self) -> int:
        """Nombre de polynômes actifs sélectionnés."""
        return len(self.ranked_indices) if self.ranked_indices is not None else 0
