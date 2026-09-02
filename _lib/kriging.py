"""
Krigeage universel -- estimation et validation -- extrait de branche3.py.

Matrices auxiliaires (Cholesky, QR), beta et sigma^2 generalises,
validation croisee K-fold et LOO, vraisemblance en theta et son
optimisation. Coeur numerique du metamodele, y compris la variante
gradient-enhanced.

PHASE 3 du plan de nettoyage : `branche3.py` melait le krigeage, la base
polynomiale et l'orchestration de l'ajustement. Le graphe d'appels
(`tools/analyse_dependances.py`) montre une stratification nette --
`fit` appelle `kriging` et `pce_basis`, qui ne s'appellent pas entre eux.

Le corps des fonctions est repris VERBATIM : la scission ne doit changer
aucun resultat, et la baseline le verifie.
"""

import numpy as np
from scipy import linalg as sla
from scipy.optimize import minimize, differential_evolution
import warnings



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
        return float(((z.T @ z) / N).item())

    elif em == 'ml_nochol':
        N    = KrgParameters['N']
        Y    = KrgParameters['Y'].reshape(-1, 1)
        F    = KrgParameters['F']
        Rinv = KrgParameters['Rinv']
        beta = KrgParameters['beta'].reshape(-1, 1)
        z    = Y - F @ beta
        return float(((z.T @ Rinv @ z) / N).item())

    elif em == 'ml_bypass_chol':
        N      = KrgParameters['N']
        Q1     = KrgParameters['Q1']
        Ytilde = KrgParameters['Ytilde'].reshape(-1, 1)
        z      = Ytilde - Q1 @ (Q1.T @ Ytilde)
        return float(((z.T @ z) / N).item())

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
    # DEFAUT 1 : au fit, c'est toujours la matrice de Gram du plan
    # d'experiences qui est voulue. On le dit, plutot que de laisser le
    # noyau le deviner en inspectant le contenu des tableaux.
    CorrOptions = dict(CorrOptions, IsGram=True)
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
                           jac=_jacobien(J, KrgModelParameters, lb),
                           bounds=bounds_list,
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8})
        return res.x, res.fun, int(res.success)


#: Passe a False pour retrouver le comportement d'avant le 02/09/2026 --
#: aucun `jac`, donc la differenciation par defaut de scipy. Sert aux mesures
#: comparatives ; la production n'a aucune raison de l'employer.
GRADIENT_ANALYTIQUE = True


def _jacobien(J, KrgModelParameters, lb):
    """Le gradient a donner a L-BFGS-B. JAMAIS celui de scipy par defaut.

    Trois niveaux, du meilleur au moins bon, et aucun n'est le pas absolu de
    1.49e-08 qui rendait du bruit :

      1. `grad_J_of_theta_ML` -- analytique, verifie contre des differences
         finies de pas adapte (`tests/test_32_gradient_vraisemblance.py`) ;
      2. si elle ne s'applique pas -- regression, Cholesky en echec, famille
         sans derivee ecrite -- des differences finies de pas RELATIF, qui
         restent trois ordres meilleures que le defaut ;
      3. `GRADIENT_ANALYTIQUE = False` rend le comportement d'origine, pour
         les mesures comparatives seulement.

    Le repli est CALCULE, jamais devine : rendre un gradient nul ferait
    croire a L-BFGS-B qu'il a converge, ce qui est precisement le defaut
    qu'on ferme.
    """
    if not GRADIENT_ANALYTIQUE:
        return None

    def jac(theta):
        theta = np.asarray(theta, dtype=float)
        g = grad_J_of_theta_ML(theta, KrgModelParameters)
        if g is not None:
            return g
        # Repli : pas RELATIF a theta, plancher a la borne basse pour qu'une
        # composante nulle ne donne pas un pas nul.
        pas = 1e-5 * np.maximum(np.abs(theta), lb)
        J0 = J(theta)
        approx = np.empty(theta.size, dtype=float)
        for i in range(theta.size):
            tp = theta.copy()
            tp[i] += pas[i]
            approx[i] = (J(tp) - J0) / pas[i]
        return approx

    return jac


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
    R     = evalR(U, U, theta_opt, dict(CorrOptions, IsGram=True))

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
    R_tilde = evalR(U, U, theta_opt, dict(CorrOptions, IsGram=True))

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
# 6bis.  Gradient ANALYTIQUE de la vraisemblance par rapport a theta
#
# POURQUOI CE BLOC EXISTE
# ------------------------
# `kriging_optimize_theta` appelait `minimize(...)` SANS `jac`. Scipy
# differenciait donc J lui-meme, avec son pas par defaut de 1.49e-08 ABSOLU,
# pour un theta vivant sur [0.01, 100] et une J dont le bruit d'evaluation
# vaut 3.05e-08. Mesure du 02/09/2026, a theta = [47.6737, 21.9981] :
#
#     pas h        dJ/dtheta_0       dJ/dtheta_1
#     1.49e-08     1.0283e+00        -1.6137e+00     <-- le pas de scipy
#     1.00e-04     3.0482e-04         4.8278e-04     <-- la vraie pente
#
# Un facteur 3 372, et un changement de signe : le gradient recu etait du
# BRUIT. Consequence directe, `ABNORMAL_TERMINATION_IN_LNSRCH` et des appels
# a `nit = 0`. Diagnostic complet et hypotheses ecartees :
# `docs/diagnostic-optimisation-theta.md`.
# ===========================================================================
def grad_J_of_theta_ML(theta, KrgModelParameters):
    """`dJ/dtheta`, analytique. Meme convention que `uq_Kriging_eval_J_of_theta_ML`.

    LA FORMULE, et d'ou elle vient
    -------------------------------
    `J = 0.5 * (N log(2 pi sigma^2) + logdet R + N)` avec
    `sigma^2 = (1/N) r^T R^-1 r` et `r = Y - F beta`, beta etant l'estimateur
    des moindres carres generalises. beta etant OPTIMAL, sa contribution a la
    derivee totale est nulle -- c'est ce qui rend la formule courte :

        dJ/dtheta_i = 0.5 * ( tr(R^-1 dR_i) - (1/sigma^2) a^T dR_i a )

    avec `a = R^-1 r`. Rien n'est recalcule qui existe deja : la Cholesky de
    R est celle que J vient de faire, et `a = U^-1 z` ou `z` est le residu
    projete que `sigma^2` utilise (`z = L^-1 r`, `L = U^T`).

    QUAND ELLE NE S'APPLIQUE PAS -- et alors on le DIT
    ---------------------------------------------------
    Rend None, sans lever, dans les trois cas ou la formule ci-dessus n'est
    pas la bonne : regression (theta porte alors un parametre de bruit en
    plus), Cholesky en echec (J bascule sur une autre formule via la
    pseudo-inverse), ou famille de noyau sans derivee ecrite. L'appelant
    retombe alors sur les differences finies -- degrade, mais jamais FAUX.
    Un gradient faux serait pire que pas de gradient : il serait faux de
    facon confiante et reproductible.
    """
    from kernels import dR_dtheta, dRtilde_dtheta

    X = KrgModelParameters['X']
    Y = KrgModelParameters['Y']
    N = KrgModelParameters['N']
    F = KrgModelParameters['F']
    CorrOptions = dict(KrgModelParameters['CorrOptions'], IsGram=True)
    if KrgModelParameters.get('IsRegression', False):
        return None

    # PCK ou GEPCK ? La Gram de GEPCK est AUGMENTEE -- n*(M+1) lignes, car
    # elle porte les blocs de derivees du noyau : 72x72 pour 24 points a deux
    # variables. La FORMULE de dJ/dtheta est la meme dans les deux cas -- J a
    # la meme forme, R est simplement plus grande -- seule la derivee de R
    # change.
    #
    # On les distingue par la seule chose qui les separe sans ambiguite : en
    # PCK, `N` est le nombre de points du plan ; en GEPCK il vaut n*(M+1)
    # alors que `X` n'a toujours que n lignes.
    _n, _M = np.atleast_2d(np.asarray(X)).shape
    if _n == int(N):
        _derivee_de_R = dR_dtheta                 # Gram simple
    elif int(N) == _n * (_M + 1):
        _derivee_de_R = dRtilde_dtheta            # Gram augmentee
    else:
        return None                               # forme non reconnue

    theta = np.asarray(theta, dtype=float).ravel()
    try:
        R = CorrOptions['Handle'](X, X, theta, CorrOptions)
        am = uq_Kriging_calc_auxMatrices(R, F, Y, 'ml_optimization')
        cholR = am['cholR']
        if cholR is None:
            return None                      # J passe par la pseudo-inverse
        derivees = _derivee_de_R(X, theta, CorrOptions)
    except Exception:
        return None

    Ytilde = np.asarray(am['Ytilde']).reshape(-1, 1)
    Q1 = am['Q1']
    z = Ytilde - Q1 @ (Q1.T @ Ytilde)
    sigmaSq = float((z.T @ z).item()) / N
    if not np.isfinite(sigmaSq) or sigmaSq <= 0.0:
        return None

    # a = R^-1 r. Avec z = L^-1 r et cholR = U = L^T : a = U^-1 z.
    a = sla.solve_triangular(cholR, z, lower=False)

    grad = np.zeros(theta.size, dtype=float)
    for i, dRi in enumerate(derivees):
        # tr(R^-1 dR_i) par la Cholesky deja calculee, sans former R^-1.
        W = sla.cho_solve((cholR, False), dRi)
        forme = float((a.T @ (dRi @ a)).item())
        grad[i] = 0.5 * (float(np.trace(W)) - forme / sigmaSq)
    if not np.all(np.isfinite(grad)):
        return None
    return grad
