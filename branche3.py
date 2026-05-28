"""
branche3.py — Python word-for-word translation of B3 (PCK calculate)

Sources (bottom → top dependency order):
  1. Kriging/calc/uq_Kriging_calc_DiagOfCongruent.m
  2. Kriging/calc/uq_Kriging_calc_auxMatrices.m
  3. Kriging/calc/uq_Kriging_calc_beta.m
  4. Kriging/calc/uq_Kriging_calc_sigmaSq.m
  5. Kriging/calc/uq_Kriging_calc_KFold.m
  6. Kriging/optimizer/uq_Kriging_eval_J_of_theta_ML.m
  7. Kriging/optimizer/uq_Kriging_optimizer.m  (→ scipy)
  8. Kriging/uq_Kriging_calculate.m  (embedded as fit_kriging_pck)
  9. PCK/uq_PCK_calculate_coefficients.m

MATLAB → Python conventions:
  - cholR : upper-triangular factor, i.e. np.linalg.cholesky(R).T
             (= MATLAB's chol(R)).  None when Cholesky fails.
  - Rinv  : None when cholR available, otherwise pinv(R).
  - All array indices 0-based.
  - cell arrays of scalars → Python lists.
"""

import copy
import warnings
import numpy as np
from scipy import linalg as sla
from scipy.optimize import minimize, differential_evolution

# --- imports from companion modules ---
import sys, os
_dir = os.path.dirname(__file__)
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from branche5 import (
    uq_PCE_create_Psi, uq_PCK_eval_unipoly,
    uq_GeneralIsopTransform, uq_eval_Kernel,
    uq_poly_rec_coeffs,           # used to identify poly type from marginal
    uq_eval_hermite_deriv, uq_eval_legendre_deriv,
)
from branche_lars import uq_lar


# ===========================================================================
# 1.  uq_Kriging_calc_DiagOfCongruent
#     Source: Kriging/calc/uq_Kriging_calc_DiagOfCongruent.m
# ===========================================================================
def uq_Kriging_calc_DiagOfCongruent(A, B):
    """
    DiagC = uq_Kriging_calc_DiagOfCongruent(A, B)
    Returns diag(A * B^{-1} * A^T) as a 1-D array.
    A : (N2, N1),  B : (N1, N1).
    """
    AT = A.T                                       # (N1, N2)
    # B well-conditioned → backslash; else pseudo-inverse
    if 1.0 / np.linalg.cond(B) > np.finfo(float).eps:   # rcond(B) > eps
        C = np.linalg.solve(B, AT)               # (N1, N2)  B \ AT
    else:
        C = np.linalg.pinv(B) @ AT               # (N1, N2)

    return np.sum(AT * C, axis=0)                  # (N2,)  row sum of AT.*C


# ===========================================================================
# 2.  uq_Kriging_calc_auxMatrices (and helpers)
#     Source: Kriging/calc/uq_Kriging_calc_auxMatrices.m
# ===========================================================================

def _calc_CholR(R):
    """
    [cholR, Rinv] = calc_CholR(R)
    cholR : upper-triangular Cholesky factor (= MATLAB chol(R)) or None.
    Rinv  : pinv(R) only when cholR is None.
    """
    try:
        L      = np.linalg.cholesky(R)        # lower-triangular: L @ L.T = R
        cholR  = L.T                          # upper-triangular  (MATLAB convention)
        Rinv   = None
    except np.linalg.LinAlgError:
        cholR  = None
        Rinv   = np.linalg.pinv(R)
    return cholR, Rinv


def _calc_FTRinv(F, cholR, Rinv):
    """
    FTRinv = F^T R^{-1}   (P × N)
    MATLAB: (F' / cholR) / cholR'  =  F^T L^{-1} L^{-T}  =  F^T R^{-1}
    """
    if cholR is not None:
        # (F' / cholR) / cholR'
        # Step 1: F.T @ cholR^{-1} = solve(cholR.T, F).T
        tmp    = np.linalg.solve(cholR.T, F)       # cholR.T \ F  →  (N, P)
        FTRinv = np.linalg.solve(cholR, tmp).T     # cholR  \ tmp → (N, P), then .T → (P, N)
    else:
        FTRinv = F.T @ Rinv
    return FTRinv


def _calc_AuxMatricesQR(cholR, Y, F):
    """
    [Ytilde, Ftilde, Q1, G] = calc_AuxMatricesQR(cholR, Y, F)
    MATLAB: Ytilde = cholR' \ Y;  Ftilde = cholR' \ F;  [Q1,G] = qr(Ftilde,0)
    """
    if cholR is not None:
        L_lower = cholR.T                          # lower-triangular
        Ytilde  = sla.solve_triangular(L_lower, Y, lower=True)
        Ftilde  = sla.solve_triangular(L_lower, F, lower=True)
        Q1, G   = np.linalg.qr(Ftilde, mode='reduced')   # economy QR
    else:
        Ytilde = Ftilde = Q1 = G = None
    return Ytilde, Ftilde, Q1, G


def _calc_FTRinvF_inv(FTRinvF):
    """
    None if FTRinvF is well-conditioned, else pinv(FTRinvF).
    MATLAB: if rcond(FTRinvF) > 1e-10 → []; else pinv(FTRinvF)
    Note: rcond > 1e-10 means well-conditioned → return None (no pinv needed).
    """
    try:
        rc = 1.0 / np.linalg.cond(FTRinvF)
    except Exception:
        rc = 0.0
    if rc > 1e-10:           # well-conditioned: no need to store inverse
        return None
    return np.linalg.pinv(FTRinvF)


def uq_Kriging_calc_auxMatrices(R, F, Y, runCase):
    """
    auxMatrices = uq_Kriging_calc_auxMatrices(R, F, Y, runCase)

    runCase in {'default', 'ml_optimization', 'ml_estimation'}

    Returns a dict with keys depending on runCase.
    """
    runCase = runCase.lower()
    am      = {}

    cholR, Rinv = _calc_CholR(R)
    am['cholR'] = cholR
    am['Rinv']  = Rinv

    if runCase in ('default', 'ml_estimation'):
        FTRinv         = _calc_FTRinv(F, cholR, Rinv)
        FTRinvF        = FTRinv @ F
        FTRinvF_inv    = _calc_FTRinvF_inv(FTRinvF)
        am['FTRinv']       = FTRinv
        am['FTRinvF']      = FTRinvF
        am['FTRinvF_inv']  = FTRinvF_inv

    if runCase in ('ml_optimization', 'ml_estimation'):
        Ytilde, Ftilde, Q1, G = _calc_AuxMatricesQR(cholR, Y, F)
        am['Ytilde'] = Ytilde
        am['Ftilde'] = Ftilde
        am['Q1']     = Q1
        am['G']      = G

    return am


# ===========================================================================
# 3.  uq_Kriging_calc_beta
#     Source: Kriging/calc/uq_Kriging_calc_beta.m
# ===========================================================================
def uq_Kriging_calc_beta(F, trendType, Y, betaEstimMethod, auxMatrices):
    """
    beta = uq_Kriging_calc_beta(F, trendType, Y, betaEstimMethod, auxMatrices)
    """
    if isinstance(Y, np.ndarray) and Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    isQR   = betaEstimMethod.lower() == 'qr'
    isNoQ1 = auxMatrices.get('Q1') is None

    if trendType.lower() == 'simple':
        betaEstimMethod = 'no_estimation'
    elif isQR and isNoQ1:
        betaEstimMethod = 'standard'

    if betaEstimMethod.lower() == 'qr':
        # beta = G^{-1} Q1^T Ytilde
        Q1     = auxMatrices['Q1']
        G_mat  = auxMatrices['G']
        Ytilde = auxMatrices['Ytilde']
        if G_mat.shape[0] != G_mat.shape[1] or np.linalg.cond(G_mat) > 1e10:
            beta = np.linalg.pinv(G_mat) @ (Q1.T @ Ytilde)
        else:
            beta = np.linalg.solve(G_mat, Q1.T @ Ytilde)

    elif betaEstimMethod.lower() == 'standard':
        # beta = (F^T R^{-1} F)^{-1} F^T R^{-1} Y
        FTRinv      = auxMatrices['FTRinv']
        FTRinvF     = auxMatrices['FTRinvF']
        FTRinvF_inv = auxMatrices['FTRinvF_inv']
        if FTRinvF_inv is None:
            beta = np.linalg.solve(FTRinvF, FTRinv @ Y)
        else:
            beta = FTRinvF_inv @ FTRinv @ Y

    else:  # 'no_estimation' — simple Kriging
        beta = np.ones((F.shape[1], 1))

    return beta.ravel()  # return as 1-D array


# ===========================================================================
# 4.  uq_Kriging_calc_sigmaSq
#     Source: Kriging/calc/uq_Kriging_calc_sigmaSq.m
# ===========================================================================
def uq_Kriging_calc_sigmaSq(KrgParameters, estimMethod):
    """
    sigmaSq = uq_Kriging_calc_sigmaSq(KrgParameters, estimMethod)

    estimMethod : 'cv' | 'ml_chol' | 'ml_nochol' | 'ml_bypass_chol' | 'ml_bypass_nochol'
    """
    em = estimMethod.lower()
    if em == 'cv':
        cvErrors = np.concatenate([np.atleast_1d(e) for e in KrgParameters['CVErrors']])
        cvSigma2 = np.concatenate([np.atleast_1d(s) for s in KrgParameters['CVSigma2']])
        return float(np.mean(cvErrors / cvSigma2))

    elif em == 'ml_chol':
        N      = KrgParameters['N']
        beta   = KrgParameters['beta'].reshape(-1, 1)
        Ytilde = KrgParameters['Ytilde'].reshape(-1, 1)
        Ftilde = KrgParameters['Ftilde']
        z      = Ytilde - Ftilde @ beta
        return float((z.T @ z) / N)

    elif em == 'ml_nochol':
        N    = KrgParameters['N']
        Y    = KrgParameters['Y'].reshape(-1, 1)
        F    = KrgParameters['F']
        Rinv = KrgParameters['Rinv']
        beta = KrgParameters['beta'].reshape(-1, 1)
        z    = Y - F @ beta
        return float((z.T @ Rinv @ z) / N)

    elif em == 'ml_bypass_chol':
        N      = KrgParameters['N']
        Q1     = KrgParameters['Q1']
        Ytilde = KrgParameters['Ytilde'].reshape(-1, 1)
        z      = Ytilde - Q1 @ (Q1.T @ Ytilde)
        return float((z.T @ z) / N)

    elif em == 'ml_bypass_nochol':
        # same as ml_nochol (can't bypass without chol)
        return uq_Kriging_calc_sigmaSq(KrgParameters, 'ml_nochol')

    else:
        raise ValueError(f'Unknown estimMethod: {estimMethod}')


# ===========================================================================
# 5.  uq_Kriging_calc_KFold
#     Source: Kriging/calc/uq_Kriging_calc_KFold.m
# ===========================================================================
def uq_Kriging_helper_create_randIdx(K, N):
    """
    randIdx = uq_Kriging_helper_create_randIdx(K, N)
    K=1 → N-fold (LOO): each fold contains one index.
    """
    if K == 1:
        return [[i] for i in range(N)]
    folds  = []
    idx    = np.random.permutation(N)
    size   = N // K
    for k in range(K):
        start = k * size
        end   = start + size if k < K - 1 else N
        folds.append(idx[start:end].tolist())
    return folds


def _calc_B1(auxMatrices, F, N):
    """
    B1 matrix for K-fold CV (Dubrule formula).
    B1 = R^{-1} * [I - F (F^T R^{-1} F)^{-1} F^T R^{-1}]
    """
    FTRinv      = auxMatrices['FTRinv']          # P × N
    FTRinvF     = auxMatrices['FTRinvF']         # P × P
    FTRinvF_inv = auxMatrices['FTRinvF_inv']     # None or P × P
    L           = auxMatrices['cholR']           # upper-tri or None

    if FTRinvF_inv is None:
        MM = np.eye(N) - F @ np.linalg.solve(FTRinvF, FTRinv)
    else:
        MM = np.eye(N) - F @ FTRinvF_inv @ FTRinv

    if L is not None:
        # B1 = L \ (L' \ MM) = R^{-1} @ MM  (L is upper-tri cholR)
        B1 = np.linalg.solve(L, np.linalg.solve(L.T, MM))
    else:
        B1 = auxMatrices['Rinv'] @ MM

    return B1


def uq_Kriging_calc_KFold(randIdx, Y, F, auxMatrices):
    """
    [cvErrors, cvSigma2] = uq_Kriging_calc_KFold(randIdx, Y, F, auxMatrices)
    Returns lists of per-fold squared errors and variance estimates.
    """
    Y       = np.asarray(Y).ravel()
    N       = len(Y)
    B1      = _calc_B1(auxMatrices, F, N)
    nClasses = len(randIdx)

    if nClasses == N:
        # LOO: Dubrule analytic formula
        yPredVar = 1.0 / np.diag(B1)          # (N,)
        yPredMu  = Y - yPredVar * (B1 @ Y)    # (N,)
        cvErrors = [(yPredMu[i] - Y[i])**2 for i in range(N)]
        cvSigma2 = [float(yPredVar[i])        for i in range(N)]
    else:
        # K-fold
        cvErrors = [None] * nClasses
        cvSigma2 = [None] * nClasses
        for nc, idxV in enumerate(randIdx):
            idxV       = list(idxV)
            idxT_mask  = np.ones(N, dtype=bool)
            idxT_mask[idxV] = False
            idxT       = np.where(idxT_mask)[0]
            yV         = Y[idxV]
            yT         = Y[idxT]
            B11        = B1[np.ix_(idxV, idxV)]
            B12        = B1[np.ix_(idxV, idxT)]
            yPredMu    = np.linalg.solve(B11, -B12 @ yT)
            yPredVar_k = np.diag(np.linalg.solve(B11, np.eye(len(idxV))))
            cvErrors[nc] = (yV - yPredMu)**2
            cvSigma2[nc] = yPredVar_k

    return cvErrors, cvSigma2


# ===========================================================================
# 6.  uq_Kriging_eval_J_of_theta_ML
#     Source: Kriging/optimizer/uq_Kriging_eval_J_of_theta_ML.m
# ===========================================================================
def uq_Kriging_eval_J_of_theta_ML(theta, KrgModelParameters):
    """
    J = uq_Kriging_eval_J_of_theta_ML(theta, KrgModelParameters)
    ML objective function for Kriging hyper-parameter optimization.
    """
    X           = KrgModelParameters['X']
    Y           = KrgModelParameters['Y']
    N           = KrgModelParameters['N']
    F           = KrgModelParameters['F']
    CorrOptions = KrgModelParameters['CorrOptions']
    evalR_handle = CorrOptions['Handle']
    trendType   = KrgModelParameters['trend_type']
    isRegression = KrgModelParameters.get('IsRegression', False)
    estimNoise  = KrgModelParameters.get('EstimNoise', False)

    try:
        if not isRegression:
            # Interpolation
            R = evalR_handle(X, X, theta, CorrOptions)
        elif estimNoise:
            # Regression with Tau parameter
            tau = theta[-1]
            R   = ((1 - tau) * evalR_handle(X, X, theta[:-1], CorrOptions)
                   + tau * np.eye(N))
        else:
            # Regression with known sigmaNSQ
            sigmaNSQ = KrgModelParameters.get('sigmaNSQ', np.zeros((N, N)))
            R = theta[-1] * evalR_handle(X, X, theta[:-1], CorrOptions) + sigmaNSQ

        am       = uq_Kriging_calc_auxMatrices(R, F, Y, 'ml_optimization')
        cholR    = am['cholR']

        # log det(R)
        if cholR is not None:
            logDetR = 2.0 * float(np.sum(np.log(np.diag(cholR))))
        else:
            eps_val = 1e-320
            logDetR = float(np.log(max(np.linalg.det(R), eps_val)))

        # sigma²
        kp = {'N': N}
        if cholR is not None:
            kp['Ytilde'] = am['Ytilde']
            kp['Q1']     = am['Q1']
            estimSigma   = 'ml_bypass_chol'
        else:
            # need beta → standard formula
            tmp_am = dict(am)
            tmp_am['FTRinv']    = _calc_FTRinv(F, None, am['Rinv'])
            tmp_am['FTRinvF']   = tmp_am['FTRinv'] @ F
            tmp_am['FTRinvF_inv'] = _calc_FTRinvF_inv(tmp_am['FTRinvF'])
            beta   = uq_Kriging_calc_beta(F, trendType, Y, 'standard', tmp_am)
            kp['Y']    = Y
            kp['F']    = F
            kp['beta'] = beta
            kp['Rinv'] = am['Rinv']
            estimSigma = 'ml_bypass_nochol'

        sigmaSq = uq_Kriging_calc_sigmaSq(kp, estimSigma)

        if not isRegression or estimNoise:
            J = 0.5 * (N * np.log(2 * np.pi * sigmaSq) + logDetR + N)
        else:
            J = 0.5 * (N * np.log(2 * np.pi) + logDetR + N * sigmaSq)

    except Exception:
        J = np.finfo(float).max

    return float(J)


# ===========================================================================
# 7.  Optimizer  (replaces uq_Kriging_optimizer for PCK usage)
#     Uses scipy — matches 'GradBased' case of uq_Kriging_optimizer.m
# ===========================================================================
def kriging_optimize_theta(KrgModelParameters, theta0, bounds_2xM,
                           method='gradbased'):
    """
    Optimize Kriging hyper-parameters theta.
    Returns (theta_opt, J_opt, exitflag).

    method : 'gradbased' → L-BFGS-B (scipy)
             'de'        → differential_evolution (scipy)
             'none'      → evaluate at theta0, no optimization
    """
    lb = bounds_2xM[0, :]        # lower bounds  (1D array)
    ub = bounds_2xM[1, :]        # upper bounds
    bounds_list = list(zip(lb, ub))

    def J(t):
        return uq_Kriging_eval_J_of_theta_ML(t, KrgModelParameters)

    if method.lower() == 'none':
        theta_opt = np.asarray(theta0)
        J_opt     = J(theta_opt)
        return theta_opt, J_opt, 1

    elif method.lower() == 'de':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = differential_evolution(J, bounds_list, seed=42,
                                          maxiter=200, tol=1e-6,
                                          mutation=(0.5, 1.0), recombination=0.9)
        return res.x, res.fun, int(res.success)

    else:   # 'gradbased' (default)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = minimize(J, theta0, method='L-BFGS-B',
                           bounds=bounds_list,
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8})
        return res.x, res.fun, int(res.success)


# ===========================================================================
# 8.  fit_kriging_pck
#     Functional equivalent of uq_Kriging_calculate.m for the PCK context.
#     Fits one Kriging model given a fixed trend handle F_handle.
# ===========================================================================
def fit_kriging_pck(U, Y, F_handle, CorrOptions,
                    theta_bounds, theta0,
                    estim_method='ml',
                    optim_method='gradbased'):
    """
    Fit a Kriging model with a fixed polynomial trend.

    Parameters
    ----------
    U            : (N, M) ndarray — training inputs in auxiliary space
    Y            : (N,)   ndarray — training outputs (one output dimension)
    F_handle     : callable U → (N, P)  (the polynomial trend matrix)
    CorrOptions  : dict with 'Handle', 'Family', 'Type' etc.
    theta_bounds : (2, M) ndarray — [lb; ub] for each dimension
    theta0       : (M,)   ndarray — initial theta
    estim_method : 'ml' or 'cv'
    optim_method : 'gradbased' | 'de' | 'none'

    Returns
    -------
    fitted : dict with keys theta, beta, sigmaSQ, LOO, F, R, auxMatrices
    """
    Y   = np.asarray(Y).ravel()
    N   = U.shape[0]
    F   = F_handle(U)                          # N × P  (trend matrix)

    # Build KrgModelParameters (matches uq_Kriging_optimizer input)
    KrgModelParameters = {
        'X'           : U,
        'Y'           : Y,
        'N'           : N,
        'F'           : F,
        'CorrOptions' : CorrOptions,
        'trend_type'  : 'ordinary',            # 'ordinary' = non-simple Kriging
        'IsRegression': False,
        'EstimNoise'  : False,
    }

    # Find optimal theta
    theta_opt, J_opt, exitflag = kriging_optimize_theta(
        KrgModelParameters, theta0, theta_bounds, method=optim_method)

    # Compute R at optimal theta
    evalR = CorrOptions['Handle']
    R     = evalR(U, U, theta_opt, CorrOptions)

    # Choose run case for aux matrices
    if estim_method.lower() == 'ml':
        runCase = 'ml_estimation'
    else:
        runCase = 'default'

    am   = uq_Kriging_calc_auxMatrices(R, F, Y, runCase)

    # Compute beta (GLS trend coefficients)
    betaMethod = 'qr' if estim_method.lower() == 'ml' else 'standard'
    beta       = uq_Kriging_calc_beta(F, 'ordinary', Y, betaMethod, am)

    # Compute sigma² (GP variance)
    if estim_method.lower() == 'ml':
        kp = {'N': N, 'beta': beta, 'F': F, 'Y': Y}
        if am['cholR'] is not None:
            kp['Ytilde'] = am['Ytilde']
            kp['Ftilde'] = am['Ftilde']
            sigmaMethod  = 'ml_chol'
        else:
            kp['Rinv']   = am['Rinv']
            sigmaMethod  = 'ml_nochol'
        sigmaSQ = uq_Kriging_calc_sigmaSq(kp, sigmaMethod)
    else:
        randIdx_cv = uq_Kriging_helper_create_randIdx(1, N)
        CVE, CVS   = uq_Kriging_calc_KFold(randIdx_cv, Y, F, am)
        kp         = {'CVErrors': CVE, 'CVSigma2': CVS}
        sigmaSQ    = uq_Kriging_calc_sigmaSq(kp, 'cv')

    # LOO error (N-fold CV, Dubrule)
    if am.get('FTRinv') is None:
        # make sure default aux matrices are available
        am_def = uq_Kriging_calc_auxMatrices(R, F, Y, 'default')
    else:
        am_def = am
    randIdx_loo         = uq_Kriging_helper_create_randIdx(1, N)
    CVErrors, _         = uq_Kriging_calc_KFold(randIdx_loo, Y, F, am_def)
    varY                = float(np.var(Y, ddof=0))
    LOO                 = float(np.mean(np.array(CVErrors))) / varY if varY > 0 else 0.0

    return {
        'theta'       : theta_opt,
        'beta'        : beta,
        'sigmaSQ'     : sigmaSQ,
        'LOO'         : LOO,
        'J_opt'       : J_opt,
        'exitflag'    : exitflag,
        'F'           : F,
        'R'           : R,
        'auxMatrices' : am_def,
        'F_handle'    : F_handle,   # stored for B4 prediction
    }


def fit_kriging_gepck(U, Y_aug, F_global_handle, CorrOptions,
                      theta_bounds, theta0,
                      estim_method='ml',
                      optim_method='gradbased'):
    """
    Fit a GEPCK model (gradient-enhanced Kriging with PCE trend).

    Clone de fit_kriging_pck avec R̃, F̃, ẏ augmentés (Zuhal 2021).

    Parameters
    ----------
    U                : (N, M) ndarray    — points d'entraînement (espace auxiliaire)
    Y_aug            : (N*(M+1),) ndarray — réponse augmentée [y ; dy/du_0 ; ... ; dy/du_{M-1}]
    F_global_handle  : callable U → (N*(M+1), P_sel)   — trend augmentée F̃
    CorrOptions      : dict avec 'Handle'=uq_eval_global_Kernel, 'Family', etc.
    theta_bounds     : (2, M) ndarray    — bornes [lb ; ub] pour chaque dimension
    theta0           : (M,)  ndarray     — theta initial
    estim_method     : 'ml' ou 'cv'
    optim_method     : 'gradbased' | 'de' | 'none'

    Returns
    -------
    fitted : dict avec theta, beta, sigmaSQ, LOO, F_tilde, R_tilde, auxMatrices,
             F_global_handle, F_handle_standard
    """
    Y_aug = np.asarray(Y_aug).ravel()
    N     = U.shape[0]             # nombre de points (pas d'observations)
    N_aug = len(Y_aug)             # N*(M+1) — nombre total d'observations
    F     = F_global_handle(U)     # (N*(M+1), P_sel) — trend augmentée F̃

    # KrgModelParameters — N_aug remplace N partout où il entre dans la vraisemblance
    KrgModelParameters = {
        'X'           : U,
        'Y'           : Y_aug,
        'N'           : N_aug,
        'F'           : F,
        'CorrOptions' : CorrOptions,
        'trend_type'  : 'ordinary',
        'IsRegression': False,
        'EstimNoise'  : False,
    }

    # Optimisation de theta
    theta_opt, J_opt, exitflag = kriging_optimize_theta(
        KrgModelParameters, theta0, theta_bounds, method=optim_method)

    # R̃ au theta optimal
    evalR   = CorrOptions['Handle']
    R_tilde = evalR(U, U, theta_opt, CorrOptions)

    # Matrices auxiliaires
    if estim_method.lower() == 'ml':
        runCase = 'ml_estimation'
    else:
        runCase = 'default'

    am = uq_Kriging_calc_auxMatrices(R_tilde, F, Y_aug, runCase)

    # Beta (GLS)
    betaMethod = 'qr' if estim_method.lower() == 'ml' else 'standard'
    beta       = uq_Kriging_calc_beta(F, 'ordinary', Y_aug, betaMethod, am)

    # Sigma²
    if estim_method.lower() == 'ml':
        kp = {'N': N_aug, 'beta': beta, 'F': F, 'Y': Y_aug}
        if am['cholR'] is not None:
            kp['Ytilde'] = am['Ytilde']
            kp['Ftilde'] = am['Ftilde']
            sigmaMethod  = 'ml_chol'
        else:
            kp['Rinv']  = am['Rinv']
            sigmaMethod = 'ml_nochol'
        sigmaSQ = uq_Kriging_calc_sigmaSq(kp, sigmaMethod)
    else:
        randIdx_cv = uq_Kriging_helper_create_randIdx(1, N_aug)
        CVE, CVS   = uq_Kriging_calc_KFold(randIdx_cv, Y_aug, F, am)
        kp         = {'CVErrors': CVE, 'CVSigma2': CVS}
        sigmaSQ    = uq_Kriging_calc_sigmaSq(kp, 'cv')

    # LOO (Dubrule) — dénominateur = N_aug (N*(M+1) observations)
    if am.get('FTRinv') is None:
        am_def = uq_Kriging_calc_auxMatrices(R_tilde, F, Y_aug, 'default')
    else:
        am_def = am
    randIdx_loo     = uq_Kriging_helper_create_randIdx(1, N_aug)
    CVErrors, _     = uq_Kriging_calc_KFold(randIdx_loo, Y_aug, F, am_def)
    varY            = float(np.var(Y_aug[:N], ddof=0))   # variance sur les N valeurs seules
    LOO             = float(np.mean(np.array(CVErrors))) / varY if varY > 0 else 0.0

    return {
        'theta'              : theta_opt,
        'beta'               : beta,
        'sigmaSQ'            : sigmaSQ,
        'LOO'                : LOO,
        'J_opt'              : J_opt,
        'exitflag'           : exitflag,
        'F_tilde'            : F,
        'R_tilde'            : R_tilde,
        'auxMatrices'        : am_def,
        'F_global_handle'    : F_global_handle,   # F̃ augmentée — pour B4 prédiction
    }


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


# ===========================================================================
# 10.  uq_PCK_calculate_coefficients (main B3 function)
#      Source: PCK/uq_PCK_calculate_coefficients.m
# ===========================================================================

def uq_PCK_calculate_coefficients(X, Y, pck_config,
                                   input_marginals, input_copula,
                                   CorrOptions=None,
                                   theta_bounds=None,
                                   theta0=None,
                                   optim_method='gradbased',
                                   estim_method='ml'):
    """
    Fit a PC-Kriging (PCK) metamodel.

    Word-for-word translation of uq_PCK_calculate_coefficients.m.

    Parameters
    ----------
    X               : (N, M) ndarray — experimental design inputs
    Y               : (N,) or (N, Nout) ndarray — model outputs
    pck_config      : dict from uq_PCK_initialize (B2), keys:
                        Mode, TrendMethod, PCE, Kriging, CombCrit,
                        [PolyIndices, PolyTypes for 'user' mode]
    input_marginals : list of M dicts, each with 'Type', 'Parameters'
    input_copula    : dict with 'Type', 'Parameters'
    CorrOptions     : dict for correlation function; if None uses Matern52
    theta_bounds    : (2, M) ndarray; if None uses [[0.01]*M, [100]*M]
    theta0          : (M,) ndarray; if None uses geometric mean of bounds
    optim_method    : 'gradbased' | 'de' | 'none'
    estim_method    : 'ml' (default) | 'cv'

    Returns
    -------
    fitted_model : dict — everything needed by B4 (PCK_eval) for prediction
    """
    X = np.atleast_2d(X).astype(float)
    Y = np.atleast_1d(Y).astype(float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    N, M    = X.shape
    Nout    = Y.shape[1]

    # -----------------------------------------------------------------------
    # Identify non-constant dimensions  (lines 44-47 in uq_PCK_calculate)
    # -----------------------------------------------------------------------
    diff_X  = np.diff(X, axis=0)
    nonConst = np.where(np.any(diff_X, axis=0))[0]   # 0-indexed
    if len(nonConst) == 0:
        raise ValueError('Only constants in the input model.')
    Xred = X[:, nonConst]                             # (N, Mred)
    Mred = len(nonConst)

    # Reduced marginals
    red_marginals = [input_marginals[i] for i in nonConst]

    # -----------------------------------------------------------------------
    # Determine PolyTypes and Auxiliary space  (mirrors uq_PCE_initialize)
    # -----------------------------------------------------------------------
    PolyTypes_all = [poly_type_from_marginal(m['Type']) for m in red_marginals]
    aux_marginals = [aux_marginal_from_poly_type(pt) for pt in PolyTypes_all]
    aux_copula    = {'Type': 'Independent',
                     'Parameters': np.eye(Mred)}

    # -----------------------------------------------------------------------
    # Default CorrOptions: Matern-5/2 anisotropic
    # -----------------------------------------------------------------------
    if CorrOptions is None:
        CorrOptions = {
            'Handle'    : uq_eval_Kernel,
            'Family'    : 'matern-5_2',
            'Type'      : 'separable',
            'Isotropic' : False,
            'Nugget'    : 0.0,
        }

    # -----------------------------------------------------------------------
    # Default theta_bounds and theta0
    # -----------------------------------------------------------------------
    if theta_bounds is None:
        theta_bounds = np.array([[0.01] * Mred, [100.0] * Mred])
    if theta0 is None:
        theta0 = np.sqrt(theta_bounds[0] * theta_bounds[1])   # geometric mean

    # -----------------------------------------------------------------------
    # Generate PCE trend (TrendMethod = 'pce' or 'user')
    # (lines 55-97 in uq_PCK_calculate_coefficients.m)
    # -----------------------------------------------------------------------
    mode         = pck_config.get('Mode', 'sequential').lower()
    trend_method = pck_config.get('TrendMethod', 'pce').lower()

    # Container for results per output
    idxranking   = [None] * Nout      # ordered index lists for each output
    AllIndices   = [None] * Nout      # the full polynomial basis (P, Mred)

    if trend_method == 'pce':
        pce_opts   = pck_config.get('PCE', {})
        max_degree = max(pce_opts.get('Degree', [1, 2, 3]))

        # Build full multi-index set up to max_degree
        FullIndices = pce_multi_indices(Mred, max_degree)   # (P_full, Mred)
        PolyTypes   = PolyTypes_all                          # list of Mred strings

        # Evaluate design matrix Psi (N × P_full) in auxiliary space
        Psi_full, U_train = pce_eval_design_matrix(
            Xred, FullIndices, PolyTypes,
            red_marginals, input_copula, aux_marginals)

        # Run LARS for each output
        for oo in range(Nout):
            lar_opts = {
                'normalize'   : True,
                'hybrid_lars' : True,
                'loo_modified': True,
                'loo_hybrid'  : True,
                'early_stop'  : True,
            }
            lar_res = uq_lar(Psi_full, Y[:, oo], lar_opts)
            idxranking[oo] = lar_res['lars_idx']    # 0-indexed column positions
            AllIndices[oo] = FullIndices             # same basis for all outputs

    else:  # 'user' trend
        # user provided PolyIndices + PolyTypes
        PolyIndices = pck_config['PolyIndices']    # (P, Mred)
        PolyTypes   = pck_config['PolyTypes']
        for oo in range(Nout):
            idxranking[oo] = list(range(PolyIndices.shape[0]))
            AllIndices[oo] = PolyIndices
        # Evaluate U for training
        _, U_train = pce_eval_design_matrix(
            Xred, AllIndices[0], PolyTypes,
            red_marginals, input_copula, aux_marginals)

    # -----------------------------------------------------------------------
    # Build trend handle factory
    # -----------------------------------------------------------------------
    def make_trend_handle(selected_idx, Indices, poly_types):
        """
        Returns F_handle(U) → (N, len(selected_idx)) trend matrix.
        Replaces uq_evalModel(myPIP, X) for Custom PCE.
        Each column corresponds to one basis polynomial.
        (Lines 130-141 in uq_PCK_calculate_coefficients.m)
        """
        Idx_sel = Indices[np.array(selected_idx), :]   # (P_sel, Mred)
        p_types = poly_types

        def F_handle(U):
            uv  = uq_PCK_eval_unipoly(U, Idx_sel, p_types)
            F   = uq_PCE_create_Psi(Idx_sel, uv)
            return F

        return F_handle

    def make_trend_handle_deriv(selected_idx, Indices, poly_types, der):
        """
        Returns F_der_handle(U) → (N, len(selected_idx)) : la dérivée partielle
        de la matrice de base PCE par rapport à la der-ième dimension de U.

        Implémente ∂Ψ_k/∂U_der = φ'_{α_k,der}(U_der) · ∏_{i≠der} φ_{α_ki}(U_i)
        en remplaçant uv[:, der, :] par les valeurs dérivées puis en annulant
        les colonnes où Idx_sel[:, der]==0 (dérivée d'une constante = 0).
        """
        Idx_sel = Indices[np.array(selected_idx), :]
        p_types = poly_types

        def F_der_handle(U):
            uv  = uq_PCK_eval_unipoly(U, Idx_sel, p_types)   # (N, M, P+1)
            P   = uv.shape[2] - 1
            pt  = p_types[der].lower()
            if pt == 'hermite':
                uv[:, der, :] = uq_eval_hermite_deriv(P, U[:, der])
            elif pt == 'legendre':
                uv[:, der, :] = uq_eval_legendre_deriv(P, U[:, der])
            Psi = uq_PCE_create_Psi(Idx_sel, uv)
            # uq_PCE_create_Psi saute les degrés 0 (aa = Indices[:,mm] > 0).
            # φ'_0 = 0 (dérivée d'une constante) : annuler explicitement.
            Psi[:, Idx_sel[:, der] == 0] = 0.0
            return Psi

        return F_der_handle

    def make_trend_global_handle(selected_idx, Indices, poly_types):
        """
        Returns F_global(U) → (N*(M+1), P_sel) : matrice F̃ augmentée GEPCK.
        Empile F(U) puis ∂F/∂U_k(U) pour k=0..M-1 (Zuhal 2021, Eq. 9).
        """
        Idx_sel = Indices[np.array(selected_idx), :]
        M_loc   = Idx_sel.shape[1]
        F0 = make_trend_handle(selected_idx, Indices, poly_types)
        Fk = [make_trend_handle_deriv(selected_idx, Indices, poly_types, k)
              for k in range(M_loc)]
        def F_global(U):
            blocks = [F0(U)] + [fk(U) for fk in Fk]
            return np.vstack(blocks)
        return F_global

    # -----------------------------------------------------------------------
    # Compose Kriging + trend  (mode = 'sequential' or 'optimal')
    # (lines 126-222 in uq_PCK_calculate_coefficients.m)
    # -----------------------------------------------------------------------
    comb_crit = pck_config.get('CombCrit', 'rel_loo').lower()

    fitted_kriging   = [None] * Nout   # one per output
    NumberOfPoly     = np.zeros(Nout, dtype=int)

    for oo in range(Nout):
        idx_ranked = idxranking[oo]     # ordered 0-indexed column positions
        Indices_oo = AllIndices[oo]     # full basis indices (P, Mred)

        if mode == 'sequential':
            # Take ALL polynomials at once as the trend
            F_handle = make_trend_handle(
                idx_ranked, Indices_oo, PolyTypes_all[:Mred])

            fitted = fit_kriging_pck(
                U_train, Y[:, oo], F_handle,
                CorrOptions, theta_bounds, theta0.copy(),
                estim_method=estim_method,
                optim_method=optim_method)

            fitted_kriging[oo]  = fitted
            NumberOfPoly[oo]    = len(idx_ranked)

        else:  # 'optimal'
            best_LOO      = np.inf
            best_fitted   = None
            best_ii       = 0
            # Initialisation GA+BFGS sur trend constant (Zuhal 2021 Section III.B.1)
            # DE sur Ψ₀=1 (kriging ordinaire) pour theta0 global avant la boucle LARS
            F_constant = lambda U: np.ones((U.shape[0], 1))
            fitted_constant = fit_kriging_pck(
                U_train, Y[:, oo], F_constant,
                CorrOptions, theta_bounds, theta0.copy(),
                estim_method=estim_method,
                optim_method='de')
            theta_current = fitted_constant['theta']

            for ii in range(1, len(idx_ranked) + 1):
                F_handle = make_trend_handle(
                    idx_ranked[:ii], Indices_oo, PolyTypes_all[:Mred])

                fitted = fit_kriging_pck(
                    U_train, Y[:, oo], F_handle,
                    CorrOptions, theta_bounds, theta_current,
                    estim_method=estim_method,
                    optim_method=optim_method)

                theta_current = fitted['theta']   # warm start pour l'itération suivante

                crit = fitted['LOO'] if comb_crit == 'rel_loo' else fitted['LOO']

                if crit < best_LOO:
                    best_LOO    = crit
                    best_fitted = fitted
                    best_ii     = ii

            fitted_kriging[oo] = best_fitted
            NumberOfPoly[oo]   = best_ii

    # -----------------------------------------------------------------------
    # Assemble the result dict  (lines 227-237 in uq_PCK_calculate)
    # -----------------------------------------------------------------------
    fitted_model = {
        # Per-output Kriging models
        'Kriging'        : fitted_kriging,
        # Error per output
        'Error'          : [{'LOO': fitted_kriging[oo]['LOO']} for oo in range(Nout)],
        # Auxiliary space (polynomial canonical space)
        'AuxSpace'       : {
            'Marginals'  : aux_marginals,
            'Copula'     : aux_copula,
        },
        # Training data (in auxiliary space)
        'ExpDesign'      : {
            'X'  : X,
            'Y'  : Y,
            'U'  : U_train,
            'Xred': Xred,
        },
        # Config used
        'pck_config'     : pck_config,
        'PolyTypes'      : PolyTypes_all[:Mred],
        'AllIndices'     : AllIndices,
        'idxranking'     : idxranking,
        'NumberOfPoly'   : NumberOfPoly,
        'nonConst'       : nonConst,
        # Marginals needed for prediction
        'OrigMarginals'  : input_marginals,
        'OrigCopula'     : input_copula,
        'RedMarginals'   : red_marginals,
        'CorrOptions'    : CorrOptions,
        'M'              : M,
        'Mred'           : Mred,
        'Nout'           : Nout,
    }

    return fitted_model
