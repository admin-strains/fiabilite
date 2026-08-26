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

from kriging import uq_Kriging_calc_DiagOfCongruent, uq_Kriging_calc_auxMatrices
from kernels import uq_eval_deriv_global_Kernel
from transform import uq_GeneralIsopTransform


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
# 0.  Poids duaux -- PHASE 7 du plan de nettoyage
# ===========================================================================
def poids_duaux(am, residual, cle='_poids_duaux'):
    r"""Renvoie `R^-1 @ residu`, calcule UNE FOIS puis mis en cache.

    Le code reconstruisait l'inverse COMPLETE, par
    `solve(cholR, solve(cholR.T, I))` -- soit N_aug descentes-remontees -- A
    CHAQUE APPEL DE PREDICTION, pour n'en tirer qu'un produit
    matrice-vecteur.

    Or `residu = Y_aug - F_tilde @ beta` ne depend QUE de l'ajustement : le
    vecteur `R^-1 @ residu` est une constante du metamodele. Une seule
    descente-remontee suffit, et une seule fois.

    Mesure sur le plan de 24 points (systeme 72x72, GEPCK) :
        reconstruction de l'inverse   0,228 ms   a chaque appel
        une descente-remontee         0,098 ms   une seule fois

    Le cache vit dans `auxMatrices`, ou le modele range deja ses facteurs et
    qui ne bouge plus une fois l'ajustement fait. Le residu est conserve avec
    le resultat : si un appelant en passe un autre, le cache se recalcule au
    lieu de rendre une valeur qui ne lui correspond pas.
    """
    memo = am.get(cle)
    if memo is not None and memo[0].shape == residual.shape \
            and np.array_equal(memo[0], residual):
        return memo[1]

    cholR = am.get('cholR')
    if cholR is not None:
        poids = np.linalg.solve(cholR, np.linalg.solve(cholR.T, residual))
    else:
        poids = am['Rinv'] @ residual
    am[cle] = (np.array(residual, copy=True), poids)
    return poids


def _inverse_complete(am, taille):
    """`R^-1` en entier, mise en cache.

    N'est necessaire QUE pour la covariance croisee (`return_cov`), qui
    demande `r0 @ R^-1 @ r0.T` : une forme quadratique entre points de test,
    irreductible a un produit avec un vecteur fixe. Ce chemin n'est pas
    emprunte par la chaine de fiabilite, qui ne demande que la variance
    ponctuelle.
    """
    cholR = am.get('cholR')
    if cholR is None:
        return am['Rinv']
    memo = am.get('_Rinv_complet')
    if memo is None or memo.shape[0] != taille:
        memo = np.linalg.solve(cholR, np.linalg.solve(cholR.T, np.eye(taille)))
        am['_Rinv_complet'] = memo
    return memo


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
    CrossCorOpts           = dict(CorrOptions)
    CrossCorOpts['Nugget'] = 0.0
    CrossCorOpts['IsGram'] = False     # cross-correlation, jamais une Gram
    evalR = CorrOptions['Handle']
    r0    = evalR(U_test, U_train, theta, CrossCorOpts)   # (N_test, N)

    # --- YMu = f0*beta + r0 * Rinv * (Y - F*beta)  (line ~154)
    #
    # `Rinv @ residual` ne depend que de l'ajustement : il est calcule une
    # fois et garde, au lieu de reconstruire l'inverse complete a chaque
    # appel (phase 7 du plan de nettoyage).
    residual = Y_train - F_train @ beta
    YMu      = f0 @ beta + r0 @ poids_duaux(am, residual)

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
        D1_mat     = r0 @ _inverse_complete(am, N) @ r0.T                              # (N_test, N_test)
        FTRinvF_inv = am.get('FTRinvF_inv')
        if FTRinvF_inv is None:
            D2_mat = u0.T @ np.linalg.solve(FTRinvF, u0)          # (N_test, N_test)
        else:
            D2_mat = u0.T @ FTRinvF_inv @ u0
        # ici, au contraire, la matrice de Gram des points de test est bien ce
        # qu'on veut : on le DIT, au lieu de le laisser deviner sur le contenu.
        GramTestOpts = dict(CorrOptions)
        GramTestOpts['IsGram'] = True
        CorrU0  = evalR(U_test, U_test, theta, GramTestOpts)        # (N_test, N_test)
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
    #
    # `IsGram=False` est EXIGE ici : on veut r0_tilde (N_test, N*(M+1)), quelle
    # que soit la position des points de test. Sans ce drapeau, evaluer le
    # metamodele exactement sur son propre plan d'experiences faisait basculer
    # la fonction sur la branche Gram et rendait une matrice carree
    # (defaut 1 du plan de nettoyage).
    CrossCorOpts            = dict(CorrOptions)
    CrossCorOpts['Nugget']  = 0.0
    CrossCorOpts['IsGram']  = False
    evalR = CorrOptions['Handle']
    r0    = evalR(U_test, U_train, theta, CrossCorOpts)   # (N_test, N*(M+1))

    # --- YMu = f0 @ beta + r̃₀ @ Rinv @ (Y_aug - F̃ @ beta)
    #
    # `Rinv @ residual` est une constante du metamodele : une seule
    # descente-remontee, mise en cache (phase 7 du plan de nettoyage).
    residual = Y_aug - F_tilde_train @ beta
    YMu      = f0 @ beta + r0 @ poids_duaux(am, residual)

    if not return_var and not return_cov:
        return YMu

    # --- u0 = FTRinv @ r̃₀ᵀ − f0ᵀ
    FTRinv  = am['FTRinv']     # (P, N*(M+1))
    FTRinvF = am['FTRinvF']    # (P, P)
    u0      = FTRinv @ r0.T - f0.T   # (P, N_test)

    if return_cov:
        D1_mat      = r0 @ _inverse_complete(am, N_aug) @ r0.T
        FTRinvF_inv = am.get('FTRinvF_inv')
        if FTRinvF_inv is None:
            D2_mat = u0.T @ np.linalg.solve(FTRinvF, u0)
        else:
            D2_mat = u0.T @ FTRinvF_inv @ u0
        # ici, au contraire, la matrice de Gram des points de test est bien ce
        # qu'on veut : on le DIT, au lieu de le laisser deviner sur le contenu.
        GramTestOpts = dict(CorrOptions)
        GramTestOpts['IsGram'] = True
        CorrU0  = evalR(U_test, U_test, theta, GramTestOpts)
        YCov    = sigmaSQ * (CorrU0 - D1_mat + D2_mat)
        YSigma2 = _verify_YSigma2(np.diag(YCov))
        return YMu, YSigma2, YCov

    D1      = uq_Kriging_calc_DiagOfCongruent(r0, R_tilde)
    D2      = uq_Kriging_calc_DiagOfCongruent(u0.T, FTRinvF)
    YSigma2 = sigmaSQ * (np.ones(N_test) - D1 + D2)
    YSigma2 = _verify_YSigma2(YSigma2)
    return YMu, YSigma2


# ===========================================================================
# 2b. uq_GEPCK_eval_one_output_deriv
#     ∂ŷ/∂u_{der_var} analytique — gradient du GEPCK dans l'espace auxiliaire.
# ===========================================================================
def uq_GEPCK_eval_one_output_deriv(gepck_oo, U_test, U_train, Y_aug,
                                    F_tilde_train, CorrOptions, der_var):
    """
    ∂ŷ/∂u_{der_var} analytique pour une sortie GEPCK.

    Formule :
        ∂ŷ/∂u_i = [∂Ψ/∂u_i(u*)]ᵀ β  +  [∂r̃₀/∂u_i(u*)]ᵀ α_pred
    avec α_pred = R̃⁻¹(ẏ - F̃β).

    Parameters
    ----------
    gepck_oo      : dict de fit_kriging_gepck — doit contenir 'F_deriv_handles'
    U_test        : (N_test, Mred)
    U_train       : (N, Mred)
    Y_aug         : (N*(M+1),)
    F_tilde_train : (N*(M+1), P)
    CorrOptions   : dict avec 'Handle'=uq_eval_global_Kernel
    der_var       : int — indice de la variable (0..Mred-1)

    Returns
    -------
    dYMu : (N_test,)
    """
    theta = gepck_oo['theta']
    beta  = gepck_oo['beta']
    am    = gepck_oo['auxMatrices']
    N_aug = gepck_oo['R_tilde'].shape[0]

    # --- alpha_pred = R̃⁻¹(ẏ - F̃β)
    #
    # C'est EXACTEMENT le vecteur de la moyenne : meme cache, donc calcule
    # une fois pour les deux (phase 7 du plan de nettoyage).
    alpha_pred = poids_duaux(am, Y_aug - F_tilde_train @ beta)   # (N_aug,)

    # --- Terme 1 : [∂Ψ/∂u_i]ᵀ β
    dPsi  = gepck_oo['F_deriv_handles'][der_var](U_test)  # (N_test, P)
    term1 = dPsi @ beta                                   # (N_test,)

    # --- Terme 2 : [∂r̃₀/∂u_i]ᵀ α_pred
    CrossCorOpts = {**CorrOptions, 'Nugget': 0.0}
    dr0   = uq_eval_deriv_global_Kernel(
        U_test, U_train, theta, CrossCorOpts, der_var)    # (N_test, N_aug)
    term2 = dr0 @ alpha_pred                              # (N_test,)

    return term1 + term2


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
# 3b. uq_GEPCK_eval_deriv
#     ∂ŷ/∂u_{der_var} en chaque point de X_test (espace auxiliaire U).
# ===========================================================================
def uq_GEPCK_eval_deriv(fitted_model, X_test, der_var):
    """
    ∂ŷ/∂u_{der_var} analytique en chaque point de X_test.

    Parameters
    ----------
    fitted_model : dict de fit_gepck / uq_GEPCK_calculate_coefficients
    X_test       : (N_test, M)
    der_var      : int — indice de la variable dans l'espace auxiliaire (0..Mred-1)

    Returns
    -------
    dYMu : (N_test, Nout)
    """
    X_test   = np.atleast_2d(X_test).astype(float)
    Mred     = fitted_model['Mred']
    nonConst = fitted_model['nonConst']
    Xred_test = X_test[:, nonConst]

    red_marg = fitted_model['RedMarginals']
    aux_marg = fitted_model['AuxSpace']['Marginals']
    aux_cop  = fitted_model['AuxSpace']['Copula']
    red_cop  = {'Type': 'Independent', 'Parameters': np.eye(Mred)}
    U_test   = uq_GeneralIsopTransform(Xred_test, red_marg, red_cop, aux_marg, aux_cop)

    U_train     = fitted_model['ExpDesign']['U']
    Y_aug       = fitted_model['ExpDesign']['Y_aug']
    CorrOptions = fitted_model['CorrOptions']

    Nout = fitted_model['Nout']
    dYMu = np.zeros((U_test.shape[0], Nout))
    for oo in range(Nout):
        gepck_oo      = fitted_model['Kriging'][oo]
        F_tilde_train = gepck_oo['F_tilde']
        dYMu[:, oo] = uq_GEPCK_eval_one_output_deriv(
            gepck_oo, U_test, U_train, Y_aug, F_tilde_train, CorrOptions, der_var)
    return dYMu


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
