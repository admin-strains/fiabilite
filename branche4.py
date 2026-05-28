"""
branche4.py — Python word-for-word translation of B4 (PCK_eval)

Sources (UQLab 2.2.0):
  1. PCK/uq_PCK_eval.m
  2. Kriging/eval/uq_Kriging_eval.m

MATLAB -> Python conventions:
  - nargout==1  <-> return_var=False, return_cov=False  (mean only)
  - nargout==2  <-> return_var=True,  return_cov=False  (mean + variance)
  - nargout==3  <-> return_var=True,  return_cov=True   (mean + variance + covariance)
  - Trajectory / replication path of uq_Kriging_eval is NOT translated
    (not needed for PCK).
  - Regression / heteroscedastic path is NOT translated
    (PCK always uses interpolation Kriging).
"""

import numpy as np

import sys, os
_dir = os.path.dirname(__file__)
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from branche3 import uq_Kriging_calc_DiagOfCongruent, uq_Kriging_calc_auxMatrices
from branche5 import uq_GeneralIsopTransform


# ===========================================================================
# Helper
# ===========================================================================
def _verify_YSigma2(YSigma2):
    """
    verify_YSigma2 (local subfunction of uq_Kriging_eval.m)
    Cap negative variance to 0, preserve NaN.
    """
    nan_mask = np.isnan(YSigma2)
    YSigma2  = np.maximum(0.0, YSigma2)
    YSigma2[nan_mask] = np.nan
    return YSigma2


# ===========================================================================
# 1.  uq_Kriging_eval (PCK interpolation path)
#     Source: Kriging/eval/uq_Kriging_eval.m
#     Only the main branch (no varargin / trajectory).
#     Regression extensions stripped (PCK is always interpolation).
# ===========================================================================
def uq_Kriging_eval_one_output(kriging_oo, U_test, U_train, Y_train, F_train,
                                 CorrOptions,
                                 return_var=False, return_cov=False):
    """
    Kriging predictor for one output (PCK interpolation).

    Implements the for-loop body of uq_Kriging_eval.m for one output oo,
    under the assumption:
      - no regression (isRegression = False)
      - no trajectory resampling
      - cache not used (auxMatrices already available)

    Parameters
    ----------
    kriging_oo  : dict from fit_kriging_pck — keys: theta, beta, sigmaSQ,
                  F, R, auxMatrices, F_handle
    U_test      : (N_test, Mred)  test inputs in auxiliary space
    U_train     : (N, Mred)       training inputs in auxiliary space
    Y_train     : (N,)            training outputs
    F_train     : (N, P)          trend matrix at training points
    CorrOptions : dict with 'Handle', 'Family', 'Type', etc.
    return_var  : bool — compute YSigma2
    return_cov  : bool — compute full YCov (implies return_var)

    Returns
    -------
    YMu           : (N_test,)
    YSigma2       : (N_test,)          [if return_var or return_cov]
    YCov          : (N_test, N_test)   [if return_cov]
    """
    theta    = kriging_oo['theta']
    beta     = kriging_oo['beta']
    sigmaSQ  = kriging_oo['sigmaSQ']
    am       = kriging_oo['auxMatrices']
    R        = kriging_oo['R']          # (N, N) training correlation matrix
    F_handle = kriging_oo['F_handle']   # callable U -> (N, P) trend matrix

    N_test = U_test.shape[0]
    N      = U_train.shape[0]

    # --- f0: trend basis at test points  (uq_Kriging_eval.m line ~106)
    # evalF_handle = current_model.Internal.Kriging(1).Trend.Handle
    # f0 = evalF_handle(U0, current_model)
    f0 = F_handle(U_test)              # (N_test, P)

    # --- r0: cross-correlation, nugget forced to 0  (line ~140)
    # CrossCorOpts = GPCorrOptions; CrossCorOpts.Nugget = 0
    CrossCorOpts         = dict(CorrOptions)
    CrossCorOpts['Nugget'] = 0.0
    evalR = CorrOptions['Handle']
    r0    = evalR(U_test, U_train, theta, CrossCorOpts)   # (N_test, N)

    # --- Rinv: R^{-1}  (line ~150)
    # if any(isnan(cholR)): Rinv = auxMatrices.Rinv
    # else: L = cholR; Rinv = L \ (L' \ eye(N))
    cholR = am['cholR']
    if cholR is not None:
        Rinv = np.linalg.solve(cholR,
               np.linalg.solve(cholR.T, np.eye(N)))
    else:
        Rinv = am['Rinv']

    # --- YMu = f0*beta + r0 * Rinv * (Y - F*beta)  (line ~154)
    residual = Y_train - F_train @ beta
    YMu      = f0 @ beta + r0 @ (Rinv @ residual)

    if not return_var and not return_cov:
        return YMu

    # --- u0 = FTRinv * r0' - f0'  (line ~159)
    # FTRinv : (P, N),  r0' : (N, N_test),  f0' : (P, N_test)
    FTRinv  = am['FTRinv']    # (P, N)
    FTRinvF = am['FTRinvF']   # (P, P)
    u0      = FTRinv @ r0.T - f0.T   # (P, N_test)

    if return_cov:
        # Full covariance  (nargout == 3 branch, lines ~194-212)
        # D1 = r0 * Rinv * r0'                      (N_test, N_test)
        # D2 = u0' * (FTRinvF \ u0)                 (N_test, N_test)
        # CorrU0 = evalR_handle(U0, U0, theta, ...)  (N_test, N_test)
        # YCov = sigmaSQ * (CorrU0 - D1 + D2)
        D1_mat     = r0 @ Rinv @ r0.T                              # (N_test, N_test)
        FTRinvF_inv = am.get('FTRinvF_inv')
        if FTRinvF_inv is None:
            D2_mat = u0.T @ np.linalg.solve(FTRinvF, u0)          # (N_test, N_test)
        else:
            D2_mat = u0.T @ FTRinvF_inv @ u0
        CorrU0  = evalR(U_test, U_test, theta, CorrOptions)        # (N_test, N_test)
        YCov    = sigmaSQ * (CorrU0 - D1_mat + D2_mat)
        # Variance from diagonal  (line ~213)
        YSigma2 = _verify_YSigma2(np.diag(YCov))
        return YMu, YSigma2, YCov

    # Variance only  (nargout == 2 branch, lines ~163-185)
    # D1 = diag(r0 R^{-1} r0')                via DiagOfCongruent
    # D2 = diag(u0' (FTRinvF)^{-1} u0)        via DiagOfCongruent
    # YSigma2 = sigmaSQ * (1 - D1 + D2)
    D1      = uq_Kriging_calc_DiagOfCongruent(r0, R)       # (N_test,)
    D2      = uq_Kriging_calc_DiagOfCongruent(u0.T, FTRinvF)  # (N_test,)
    YSigma2 = sigmaSQ * (np.ones(N_test) - D1 + D2)
    YSigma2 = _verify_YSigma2(YSigma2)
    return YMu, YSigma2


# ===========================================================================
# 2.  uq_GEPCK_eval_one_output
# ===========================================================================
def uq_GEPCK_eval_one_output(gepck_oo, U_test, U_train, Y_aug, F_tilde_train,
                               CorrOptions,
                               return_var=False, return_cov=False):
    """
    Prédiction GEPCK pour une sortie.

    Clone de uq_Kriging_eval_one_output avec R_tilde, r̃₀, Y_aug, F_tilde_train.

    Parameters
    ----------
    gepck_oo      : dict de fit_kriging_gepck — clés : theta, beta, sigmaSQ,
                    F_tilde, R_tilde, auxMatrices, F_global_handle
    U_test        : (N_test, Mred)
    U_train       : (N, Mred)
    Y_aug         : (N*(M+1),)
    F_tilde_train : (N*(M+1), P)
    CorrOptions   : dict avec 'Handle'=uq_eval_global_Kernel
    return_var    : bool
    return_cov    : bool
    """
    theta           = gepck_oo['theta']
    beta            = gepck_oo['beta']
    sigmaSQ         = gepck_oo['sigmaSQ']
    am              = gepck_oo['auxMatrices']
    R_tilde         = gepck_oo['R_tilde']
    F_global_handle = gepck_oo['F_global_handle']

    N_test = U_test.shape[0]
    N      = U_train.shape[0]
    N_aug  = R_tilde.shape[0]          # N*(M+1)

    # --- f0 : trend standard au point test — NON augmenté (Zuhal : f(x*) non augmenté)
    f0 = F_global_handle(U_test)[:N_test, :]   # (N_test, P)

    # --- r̃₀ : cross-corrélation augmentée, nugget forcé à 0
    CrossCorOpts           = dict(CorrOptions)
    CrossCorOpts['Nugget'] = 0.0
    evalR = CorrOptions['Handle']
    r0    = evalR(U_test, U_train, theta, CrossCorOpts)   # (N_test, N*(M+1))

    # --- Rinv
    cholR = am['cholR']
    if cholR is not None:
        Rinv = np.linalg.solve(cholR,
               np.linalg.solve(cholR.T, np.eye(N_aug)))
    else:
        Rinv = am['Rinv']

    # --- YMu = f0 @ beta + r̃₀ @ Rinv @ (Y_aug - F̃ @ beta)
    residual = Y_aug - F_tilde_train @ beta
    YMu      = f0 @ beta + r0 @ (Rinv @ residual)

    if not return_var and not return_cov:
        return YMu

    # --- u0 = FTRinv @ r̃₀ᵀ − f0ᵀ
    FTRinv  = am['FTRinv']     # (P, N*(M+1))
    FTRinvF = am['FTRinvF']    # (P, P)
    u0      = FTRinv @ r0.T - f0.T   # (P, N_test)

    if return_cov:
        D1_mat      = r0 @ Rinv @ r0.T
        FTRinvF_inv = am.get('FTRinvF_inv')
        if FTRinvF_inv is None:
            D2_mat = u0.T @ np.linalg.solve(FTRinvF, u0)
        else:
            D2_mat = u0.T @ FTRinvF_inv @ u0
        CorrU0  = evalR(U_test, U_test, theta, CorrOptions)
        YCov    = sigmaSQ * (CorrU0 - D1_mat + D2_mat)
        YSigma2 = _verify_YSigma2(np.diag(YCov))
        return YMu, YSigma2, YCov

    D1      = uq_Kriging_calc_DiagOfCongruent(r0, R_tilde)
    D2      = uq_Kriging_calc_DiagOfCongruent(u0.T, FTRinvF)
    YSigma2 = sigmaSQ * (np.ones(N_test) - D1 + D2)
    YSigma2 = _verify_YSigma2(YSigma2)
    return YMu, YSigma2


# ===========================================================================
# 3.  uq_GEPCK_eval
#     Clone de uq_PCK_eval pour GEPCK (Nout=1 fixé).
# ===========================================================================
def uq_GEPCK_eval(fitted_model, X_test, return_var=False, return_cov=False):
    """
    Évalue un métamodèle GEPCK entraîné sur X_test.

    Clone de uq_PCK_eval. Différences :
      - fitted_model['ExpDesign']['Y_aug'] au lieu de ['Y']
      - kriging_oo['F_tilde'] au lieu de kriging_oo['F']
      - appelle uq_GEPCK_eval_one_output au lieu de uq_Kriging_eval_one_output

    Parameters
    ----------
    fitted_model : dict de uq_GEPCK_calculate_coefficients (B3)
    X_test       : (N_test, M) ndarray
    return_var   : bool
    return_cov   : bool

    Returns
    -------
    YMu     : (N_test, 1)
    YSigma2 : (N_test, 1)          [si return_var ou return_cov]
    YCov    : (N_test, N_test, 1)  [si return_cov]
    """
    X_test  = np.atleast_2d(X_test).astype(float)
    N_test  = X_test.shape[0]

    Nout     = fitted_model['Nout']
    Mred     = fitted_model['Mred']
    nonConst = fitted_model['nonConst']

    Xred_test = X_test[:, nonConst]

    red_marg = fitted_model['RedMarginals']
    aux_marg = fitted_model['AuxSpace']['Marginals']
    aux_cop  = fitted_model['AuxSpace']['Copula']
    red_cop  = {'Type': 'Independent', 'Parameters': np.eye(Mred)}

    U_test = uq_GeneralIsopTransform(
        Xred_test, red_marg, red_cop, aux_marg, aux_cop)

    U_train     = fitted_model['ExpDesign']['U']
    Y_aug       = fitted_model['ExpDesign']['Y_aug']
    CorrOptions = fitted_model['CorrOptions']

    YMu = np.zeros((N_test, Nout))
    if return_var or return_cov:
        YSigma2 = np.zeros((N_test, Nout))
    if return_cov:
        YCov = np.zeros((N_test, N_test, Nout))

    for oo in range(Nout):
        gepck_oo      = fitted_model['Kriging'][oo]
        F_tilde_train = gepck_oo['F_tilde']

        if return_cov:
            YMu_oo, YSig_oo, YCov_oo = uq_GEPCK_eval_one_output(
                gepck_oo, U_test, U_train, Y_aug, F_tilde_train,
                CorrOptions, return_var=True, return_cov=True)
            YMu[:, oo]     = YMu_oo
            YSigma2[:, oo] = YSig_oo
            YCov[:, :, oo] = YCov_oo
        elif return_var:
            YMu_oo, YSig_oo = uq_GEPCK_eval_one_output(
                gepck_oo, U_test, U_train, Y_aug, F_tilde_train,
                CorrOptions, return_var=True, return_cov=False)
            YMu[:, oo]     = YMu_oo
            YSigma2[:, oo] = YSig_oo
        else:
            YMu[:, oo] = uq_GEPCK_eval_one_output(
                gepck_oo, U_test, U_train, Y_aug, F_tilde_train,
                CorrOptions, return_var=False, return_cov=False)

    if return_cov:
        return YMu, YSigma2, YCov
    elif return_var:
        return YMu, YSigma2
    else:
        return YMu


# ===========================================================================
# 2.  uq_PCK_eval
#     Source: PCK/uq_PCK_eval.m  (main entry point)
# ===========================================================================
def uq_PCK_eval(fitted_model, X_test, return_var=False, return_cov=False):
    """
    Evaluate a fitted PCK metamodel at X_test.

    Word-for-word translation of:
      uq_PCK_eval.m  +  uq_Kriging_eval.m  (PCK / interpolation path)

    Parameters
    ----------
    fitted_model : dict from uq_PCK_calculate_coefficients (B3)
    X_test       : (N_test, M) ndarray of test inputs
    return_var   : bool — also return YSigma2 (corresponds to nargout>=2)
    return_cov   : bool — also return full YCov (corresponds to nargout>=3)

    Returns
    -------
    YMu           : (N_test, Nout)
    YSigma2       : (N_test, Nout)             [if return_var or return_cov]
    YCov          : (N_test, N_test, Nout)     [if return_cov]
    """
    X_test  = np.atleast_2d(X_test).astype(float)
    N_test  = X_test.shape[0]

    Nout     = fitted_model['Nout']
    Mred     = fitted_model['Mred']
    nonConst = fitted_model['nonConst']   # 0-indexed

    # --- Remove constant dimensions  (uq_PCK_eval.m line 14)
    # X = X(:, current_model.Internal.Input.nonConst)
    Xred_test = X_test[:, nonConst]       # (N_test, Mred)

    # --- Map Xred_test -> U_test  (uq_Kriging_eval.m lines ~65-80)
    # SCALING = current_model.Internal.Scaling  (= AuxSpace)
    # U0 = uq_GeneralIsopTransform(X0, Input.Marginals, Input.Copula,
    #                               SCALING.Marginals, SCALING.Copula)
    red_marg = fitted_model['RedMarginals']
    aux_marg = fitted_model['AuxSpace']['Marginals']
    aux_cop  = fitted_model['AuxSpace']['Copula']
    # Reduced input copula = independent (constants already removed)
    red_cop  = {'Type': 'Independent', 'Parameters': np.eye(Mred)}

    U_test = uq_GeneralIsopTransform(
        Xred_test, red_marg, red_cop, aux_marg, aux_cop)

    # --- Training data
    # U = current_model.ExpDesign.U  (scaled training inputs)
    # Y = current_model.ExpDesign.Y
    U_train = fitted_model['ExpDesign']['U']   # (N, Mred)
    Y_full  = fitted_model['ExpDesign']['Y']   # (N, Nout)
    CorrOptions = fitted_model['CorrOptions']

    # --- Allocate output arrays  (uq_PCK_eval.m lines 9-11)
    YMu     = np.zeros((N_test, Nout))
    if return_var or return_cov:
        YSigma2 = np.zeros((N_test, Nout))
    if return_cov:
        YCov = np.zeros((N_test, N_test, Nout))

    # --- Loop over outputs  (uq_PCK_eval.m lines 18-41)
    for oo in range(Nout):
        kriging_oo = fitted_model['Kriging'][oo]
        Y_train    = Y_full[:, oo].ravel()
        F_train    = kriging_oo['F']    # (N, P) — training trend matrix

        if return_cov:
            YMu_oo, YSig_oo, YCov_oo = uq_Kriging_eval_one_output(
                kriging_oo, U_test, U_train, Y_train, F_train,
                CorrOptions, return_var=True, return_cov=True)
            YMu[:, oo]     = YMu_oo
            YSigma2[:, oo] = YSig_oo
            YCov[:, :, oo] = YCov_oo
        elif return_var:
            YMu_oo, YSig_oo = uq_Kriging_eval_one_output(
                kriging_oo, U_test, U_train, Y_train, F_train,
                CorrOptions, return_var=True, return_cov=False)
            YMu[:, oo]     = YMu_oo
            YSigma2[:, oo] = YSig_oo
        else:
            YMu[:, oo] = uq_Kriging_eval_one_output(
                kriging_oo, U_test, U_train, Y_train, F_train,
                CorrOptions, return_var=False, return_cov=False)

    # --- Return
    if return_cov:
        return YMu, YSigma2, YCov
    elif return_var:
        return YMu, YSigma2
    else:
        return YMu
