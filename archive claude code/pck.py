"""
pck.py — PC-Kriging (Polynomial Chaos Kriging)
Fidèle à UQLab : uq_PCK_calculate_coefficients.m + uq_Kriging_calculate.m
               + uq_Kriging_eval.m + uq_Kriging_eval_J_of_theta_ML.m

Modèle : Y(x) = f_PCE(x) [trend] + Z(x) [Gaussien de moyenne nulle]

Prédiction BLUP :
  Ŷ(x0) = f0·β + r0 · R⁻¹ · (Y - F·β)
  Var(x0) = σ² · (1 - D1 + D2)

Mode 'sequential' : tous les polynômes LARS → 1 calibration Kriging
Mode 'optimal'    : sous-ensembles croissants → on retient celui qui minimise LOO
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, solve_triangular
import warnings
from typing import Optional, Sequence

from kernels import eval_kernel
from pce_trend import PCETrend


# ---------------------------------------------------------------------------
# Helpers numériques
# ---------------------------------------------------------------------------

def _cholesky_safe(R: np.ndarray):
    """Tente la décomposition de Cholesky. Retourne (L, ok) avec L upper-triangular."""
    try:
        L = np.linalg.cholesky(R).T  # L upper, R = Lᵀ L
        return L, True
    except np.linalg.LinAlgError:
        return None, False


def _compute_aux_matrices(R: np.ndarray, F: np.ndarray, Y: np.ndarray) -> dict:
    """
    Calcule les matrices auxiliaires nécessaires à la calibration et prédiction.
    Fidèle à uq_Kriging_calc_auxMatrices.m (mode 'ml_estimation').

    Retourne un dict avec :
        cholR, Rinv, FTRinv, FTRinvF, Ytilde, Ftilde, Q1, G, beta, sigmaSQ
    """
    N = R.shape[0]
    L, chol_ok = _cholesky_safe(R)

    aux = {}
    aux['chol_ok'] = chol_ok

    if chol_ok:
        aux['cholR'] = L  # upper triangular, R = Lᵀ L
        aux['Rinv'] = None

        # Ytilde = L⁻ᵀ Y  (forward substitution avec Lᵀ = L.T lower)
        # L est upper → Lᵀ est lower → solve_triangular(L.T, Y, lower=True)
        Ytilde = solve_triangular(L.T, Y, lower=True)
        Ftilde = solve_triangular(L.T, F, lower=True)

        # QR de Ftilde : [Q1, G] = qr(Ftilde, 0)
        Q1, G = np.linalg.qr(Ftilde, mode='reduced')

        # FTRinv = Fᵀ R⁻¹, shape (P, N)
        # R = Lᵀ L (L upper)  →  R⁻¹ F = L⁻¹ L⁻ᵀ F
        # Étape 1 : résoudre Lᵀ v = F  (Lᵀ lower)
        # Étape 2 : résoudre L x = v   (L upper)  → x = R⁻¹ F
        v_rinvF = solve_triangular(L.T, F, lower=True)       # L⁻ᵀ F, (N, P)
        Rinv_F = solve_triangular(L, v_rinvF, lower=False)   # R⁻¹ F, (N, P)
        FTRinv = Rinv_F.T                                     # Fᵀ R⁻¹, (P, N)
        FTRinvF = F.T @ Rinv_F                                # Fᵀ R⁻¹ F, (P, P)

        aux['Ytilde'] = Ytilde
        aux['Ftilde'] = Ftilde
        aux['Q1'] = Q1
        aux['G'] = G
        aux['FTRinv'] = FTRinv
        aux['FTRinvF'] = FTRinvF
    else:
        # Cholesky échoué → pseudo-inverse
        aux['cholR'] = None
        Rinv = np.linalg.pinv(R)
        aux['Rinv'] = Rinv
        FTRinv = F.T @ Rinv
        FTRinvF = FTRinv @ F
        aux['FTRinv'] = FTRinv
        aux['FTRinvF'] = FTRinvF
        aux['Ytilde'] = None
        aux['Ftilde'] = None
        aux['Q1'] = None
        aux['G'] = None

    return aux


def _compute_beta(aux: dict, Y: np.ndarray) -> np.ndarray:
    """
    Calcule β = (Fᵀ R⁻¹ F)⁻¹ Fᵀ R⁻¹ Y via QR si disponible.
    Fidèle à uq_Kriging_calc_beta.m.
    """
    if aux['chol_ok'] and aux['Q1'] is not None:
        # β = G⁻¹ Q1ᵀ Ỹ
        Q1, G, Ytilde = aux['Q1'], aux['G'], aux['Ytilde']
        rhs = Q1.T @ Ytilde
        beta, _, _, _ = np.linalg.lstsq(G, rhs, rcond=None)
    else:
        # β = (Fᵀ R⁻¹ F)⁻¹ Fᵀ R⁻¹ Y
        FTRinv = aux['FTRinv']
        FTRinvF = aux['FTRinvF']
        beta, _, _, _ = np.linalg.lstsq(FTRinvF, FTRinv @ Y, rcond=None)
    return beta


def _compute_sigma_sq(aux: dict, N: int, Y: np.ndarray, beta: np.ndarray) -> float:
    """
    σ² = (1/N) ||z||²  fidèle à calc_sigmaSqMLBypass (QR) ou calc_sigmaSqMLWithoutCholesky.
    """
    if aux['chol_ok'] and aux['Q1'] is not None:
        # bypass : z = Ỹ - Q1 Q1ᵀ Ỹ
        Ytilde = aux['Ytilde']
        Q1 = aux['Q1']
        z = Ytilde - Q1 @ (Q1.T @ Ytilde)
        sigma_sq = float((z @ z) / N)
    else:
        # z = Y - F β, σ² = z Rᵀ z / N (impossible sans R ici)
        F = aux.get('F', None)
        Rinv = aux['Rinv']
        if F is not None:
            z = Y - F @ beta
            sigma_sq = float((z @ Rinv @ z) / N)
        else:
            sigma_sq = float(np.var(Y))
    return max(sigma_sq, 1e-20)


def _mle_objective(log_theta: np.ndarray, X: np.ndarray, Y: np.ndarray,
                   F: np.ndarray, kernel_family: str, nugget: float) -> float:
    """
    J(θ) = 0.5 * (N * log(2π σ²) + log|R| + N)
    Fidèle à uq_Kriging_eval_J_of_theta_ML.m
    """
    theta = np.exp(log_theta)
    N = X.shape[0]

    try:
        R = eval_kernel(X, X, theta, family=kernel_family, nugget=nugget)
        aux = _compute_aux_matrices(R, F, Y)
        aux['F'] = F

        # log det R
        if aux['chol_ok']:
            logdetR = 2.0 * np.sum(np.log(np.diag(aux['cholR'])))
        else:
            eps = 1e-320
            logdetR = np.log(max(np.linalg.det(R), eps))

        beta = _compute_beta(aux, Y)
        sigma_sq = _compute_sigma_sq(aux, N, Y, beta)

        J = 0.5 * (N * np.log(2 * np.pi * sigma_sq) + logdetR + N)
        return float(J)
    except Exception:
        return 1e30


def _compute_loo(R: np.ndarray, F: np.ndarray, Y: np.ndarray) -> float:
    """
    Calcul analytique de l'erreur LOO normalisée.
    LOO = mean((e_i / (1 - H_ii))²) / Var(Y)
    """
    N = Y.shape[0]
    Y_var = np.var(Y)
    if Y_var < 1e-20:
        return 0.0

    aux = _compute_aux_matrices(R, F, Y)
    aux['F'] = F
    beta = _compute_beta(aux, Y)

    # Résidus = Y - F β - r_cross R⁻¹ (Y - F β)
    # Pour LOO analytique Kriging, on peut utiliser la hat matrix du trend
    # UQLab utilise Error.LOO via une formule analytique sur le Kriging
    # Approximation : on utilise la hat matrix du modèle de régression (F, R)
    # H = F (Fᵀ R⁻¹ F)⁻¹ Fᵀ R⁻¹  → hat matrix du trend seulement
    # Plus rigoureux : h_diag du BLUP complet
    # Ici on utilise l'approximation de la variance du BLUP via Cholesky
    try:
        if aux['chol_ok']:
            L = aux['cholR']
            # Hat matrix du trend : H = Ftilde (Ftilde^T Ftilde)^{-1} Ftilde^T ... mais
            # pour le krigeage complet, la hat matrix est R^{-1/2} F (F^T R^{-1} F)^{-1} F^T R^{-1/2}
            # Diag de cette hat matrix = ||F_tilde(i,:)||² / ||Ftilde||²  approx.
            Ftilde = aux['Ftilde']
            Q1 = aux['Q1']
            # h_ii = ||Q1[i,:]||²
            h_diag = np.sum(Q1**2, axis=1)
        else:
            Rinv = aux['Rinv']
            FTRinv = aux['FTRinv']
            FTRinvF = aux['FTRinvF']
            H = F @ np.linalg.solve(FTRinvF, FTRinv)
            h_diag = np.diag(H)

        # Résidus de régression
        residuals = Y - F @ beta
        h_diag = np.clip(h_diag, -0.9999, 0.9999)
        e_loo = residuals / (1.0 - h_diag)
        loo = np.mean(e_loo**2) / Y_var
        return float(loo)
    except Exception:
        return np.inf


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class PCKriging:
    """
    PC-Kriging : trend PCE (LARS) + Kriging sur les résidus (MLE).

    Fidèle à UQLab, mode 'sequential' et 'optimal'.

    Parameters
    ----------
    mode : 'sequential' | 'optimal'
    pce_degree : degrés polynomiaux à tester (défaut 1:3)
    corr_family : famille du kernel ('matern-5_2' par défaut)
    nugget : régularisation diagonale de R (défaut 1e-4)
    n_optim_starts : nombre de démarrages aléatoires pour l'optimisation MLE
    """

    def __init__(self,
                 mode: str = 'sequential',
                 pce_degree: Sequence[int] = range(1, 4),
                 corr_family: str = 'matern-5_2',
                 nugget: float = 1e-4,
                 n_optim_starts: int = 5):
        self.mode = mode.lower()
        assert self.mode in ('sequential', 'optimal'), "mode doit être 'sequential' ou 'optimal'"
        self.pce_degree = list(pce_degree)
        self.corr_family = corr_family
        self.nugget = nugget
        self.n_optim_starts = n_optim_starts

        # Résultats du fit
        self.pce_trend: Optional[PCETrend] = None
        self.theta: Optional[np.ndarray] = None     # longueurs de corrélation optimales
        self.sigma_sq: Optional[float] = None        # variance GP
        self.beta: Optional[np.ndarray] = None       # coefficients trend
        self.F_train: Optional[np.ndarray] = None    # matrice de design (N, n_poly)
        self.R_train: Optional[np.ndarray] = None    # matrice de corrélation (N, N)
        self._aux: Optional[dict] = None
        self._X_train: Optional[np.ndarray] = None
        self._Y_train: Optional[np.ndarray] = None
        self._n_poly_used: int = 0
        self.loo_error: Optional[float] = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray, distributions: list) -> 'PCKriging':
        """
        Calibre le modèle PC-Kriging.

        Parameters
        ----------
        X : (N, M) points d'entraînement
        Y : (N,) réponses
        distributions : liste de dicts {'type': str, 'parameters': list}

        Returns
        -------
        self
        """
        X = np.atleast_2d(X)
        Y = np.asarray(Y).ravel()
        N, M = X.shape

        self._X_train = X.copy()
        self._Y_train = Y.copy()

        # ── Étape 1 : calibrer le PCE trend ──────────────────────────────
        self.pce_trend = PCETrend(distributions, degree=self.pce_degree)
        self.pce_trend.fit(X, Y)
        n_total_poly = self.pce_trend.n_polynomials

        if n_total_poly == 0:
            # Fallback : constante seule si LARS ne sélectionne rien
            print("  [PCK] Avertissement : LARS n'a sélectionné aucun polynôme → trend constant")
            n_total_poly = 1
            self.pce_trend.ranked_indices = [0]

        # ── Étape 2 : Kriging avec trend PCE ─────────────────────────────
        if self.mode == 'sequential':
            self._fit_sequential(X, Y, M, n_total_poly)
        else:
            self._fit_optimal(X, Y, M, n_total_poly)

        return self

    # ------------------------------------------------------------------
    def _fit_sequential(self, X, Y, M, n_total_poly):
        """
        Mode sequential : tous les polynômes LARS → 1 calibration Kriging.
        Fidèle à UQLab sequential mode.
        """
        # Matrice de design = tous les polynômes actifs
        F = self.pce_trend.eval_active(X)  # (N, n_total_poly)
        self._calibrate_kriging(X, Y, F, M)
        self._n_poly_used = n_total_poly
        self.loo_error = _compute_loo(self.R_train, F, Y)

    # ------------------------------------------------------------------
    def _fit_optimal(self, X, Y, M, n_total_poly):
        """
        Mode optimal : sous-ensembles croissants, minimisation LOO.
        Fidèle à UQLab optimal mode.
        """
        best_loo = np.inf
        best_state = None

        for k in range(1, n_total_poly + 1):
            F_k = self.pce_trend.eval_subset(X, k)  # (N, k)

            # Calibrer Kriging pour ce sous-ensemble
            self._calibrate_kriging(X, Y, F_k, M)
            loo = _compute_loo(self.R_train, F_k, Y)

            if loo < best_loo:
                best_loo = loo
                best_state = {
                    'theta': self.theta.copy(),
                    'sigma_sq': self.sigma_sq,
                    'beta': self.beta.copy(),
                    'F_train': F_k.copy(),
                    'R_train': self.R_train.copy(),
                    'aux': self._aux.copy(),
                    'n_poly': k,
                }

        # Restaurer le meilleur état
        self.theta = best_state['theta']
        self.sigma_sq = best_state['sigma_sq']
        self.beta = best_state['beta']
        self.F_train = best_state['F_train']
        self.R_train = best_state['R_train']
        self._aux = best_state['aux']
        self._n_poly_used = best_state['n_poly']
        self.loo_error = best_loo

    # ------------------------------------------------------------------
    def _calibrate_kriging(self, X: np.ndarray, Y: np.ndarray,
                            F: np.ndarray, M: int) -> None:
        """
        Optimise θ via MLE multi-départ, puis calcule β et σ².
        Fidèle à uq_Kriging_optimizer.m + uq_Kriging_calculate.m.
        """
        N = X.shape[0]

        # Bornes log(θ) ∈ [log(1e-2), log(1e2)] (anisotrope, M params)
        bounds = [(-4.6, 4.6)] * M  # log(0.01) ≈ -4.6, log(100) ≈ 4.6

        best_J = np.inf
        best_log_theta = np.zeros(M)

        for _ in range(self.n_optim_starts):
            # Départ aléatoire
            log_theta0 = np.random.uniform(-2, 2, size=M)

            try:
                res = minimize(
                    _mle_objective,
                    log_theta0,
                    args=(X, Y, F, self.corr_family, self.nugget),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-9}
                )
                if res.fun < best_J:
                    best_J = res.fun
                    best_log_theta = res.x
            except Exception:
                pass

        self.theta = np.exp(best_log_theta)

        # Calcul final des matrices et coefficients
        self.R_train = eval_kernel(X, X, self.theta,
                                   family=self.corr_family, nugget=self.nugget)
        self._aux = _compute_aux_matrices(self.R_train, F, Y)
        self._aux['F'] = F
        self.beta = _compute_beta(self._aux, Y)
        self.sigma_sq = _compute_sigma_sq(self._aux, N, Y, self.beta)
        self.F_train = F

    # ------------------------------------------------------------------
    def predict(self, X_test: np.ndarray, return_std: bool = False):
        """
        Prédiction Kriging BLUP.
        Fidèle à uq_Kriging_eval.m.

        Parameters
        ----------
        X_test : (N_test, M)
        return_std : si True, retourne aussi la déviation standard

        Returns
        -------
        Y_mean : (N_test,)
        Y_std  : (N_test,) — uniquement si return_std=True
        """
        if self.theta is None:
            raise RuntimeError("Modèle non calibré. Appeler fit() d'abord.")

        X_test = np.atleast_2d(X_test)
        N_test = X_test.shape[0]

        X_train = self._X_train
        Y_train = self._Y_train
        F_train = self.F_train
        beta = self.beta
        sigma_sq = self.sigma_sq
        aux = self._aux

        # Trend au point test
        # f0 = évaluation des n_poly_used premiers polynômes (dans le classement LARS)
        f0 = self.pce_trend.eval_subset(X_test, self._n_poly_used)  # (N_test, n_poly)

        # Cross-corrélation : nugget = 0 (fidèle au code MATLAB)
        r0 = eval_kernel(X_test, X_train, self.theta,
                         family=self.corr_family, nugget=0.0)  # (N_test, N)

        # R⁻¹ (Y - F β)
        Y_Fbeta = Y_train - F_train @ beta

        if aux['chol_ok']:
            L = aux['cholR']
            # R = Lᵀ L (L upper)  →  R⁻¹ v = L⁻¹ L⁻ᵀ v
            # Étape 1 : Lᵀ w = v  (Lᵀ lower)  →  w = L⁻ᵀ v
            # Étape 2 : L x = w  (L upper)   →  x = L⁻¹ w = R⁻¹ v
            w_rinv = solve_triangular(L.T, Y_Fbeta, lower=True)
            Rinv_rhs = solve_triangular(L, w_rinv, lower=False)
        else:
            Rinv_rhs = aux['Rinv'] @ Y_Fbeta

        # Ŷ = f0 β + r0 R⁻¹ (Y - F β)
        Y_mean = f0 @ beta + r0 @ Rinv_rhs

        if not return_std:
            return Y_mean

        # ── Variance : σ² (1 - D1 + D2) ──────────────────────────────────
        # D1 = diag(r0 R⁻¹ r0ᵀ)
        if aux['chol_ok']:
            L = aux['cholR']
            # r0 R⁻¹ r0ᵀ : pour chaque ligne i de r0,  D1[i] = ||L⁻¹ r0[i,:]||²
            LT_inv_r0 = solve_triangular(L, r0.T, lower=False).T  # (N_test, N)
            D1 = np.sum(LT_inv_r0**2, axis=1)  # (N_test,)
        else:
            Rinv = aux['Rinv']
            D1 = np.einsum('ij,jk,ik->i', r0, Rinv, r0)  # diag de r0 Rinv r0ᵀ

        # u0 = Fᵀ R⁻¹ r0ᵀ - f0ᵀ  → (n_poly, N_test)
        FTRinv = aux['FTRinv']  # (n_poly, N)
        u0 = FTRinv @ r0.T - f0.T  # (n_poly, N_test)

        # D2 = diag(u0ᵀ (Fᵀ R⁻¹ F)⁻¹ u0)
        FTRinvF = aux['FTRinvF']  # (n_poly, n_poly)
        FTRinvF_inv_u0, _, _, _ = np.linalg.lstsq(FTRinvF, u0, rcond=None)  # (n_poly, N_test)
        D2 = np.einsum('ij,ij->j', u0, FTRinvF_inv_u0)  # (N_test,)

        Y_var = sigma_sq * (1.0 - D1 + D2)
        Y_var = np.maximum(Y_var, 0.0)  # cap négatif à 0 (dispersion numérique)
        Y_std = np.sqrt(Y_var)

        return Y_mean, Y_std

    # ------------------------------------------------------------------
    def __repr__(self):
        if self.theta is None:
            return "PCKriging(non calibré)"
        return (f"PCKriging(mode='{self.mode}', "
                f"n_poly={self._n_poly_used}, "
                f"theta={np.round(self.theta, 4)}, "
                f"sigma_sq={self.sigma_sq:.4e}, "
                f"LOO={self.loo_error:.4e})")
