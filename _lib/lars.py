"""
branche_lars.py — Python word-for-word translation of UQLab LARS functions

Sources (in order of dependency, bottom → top):
  1. modules/PCE/uq_PCE_loo_error.m
  2. lib/uq_matrix_utils/uq_blockwise_inverse.m
  3. modules/PCE/PolyCoeff/Regression/uq_PCE_OLS_regression.m
  4. lib/uq_regression/LAR/uq_lar.m
  5. modules/PCE/PolyCoeff/Regression/uq_PCE_lars.m  (thin wrapper)

MATLAB → Python index convention:
  - All column indices in a_coeff, i_coeff, constindices, lars_idx are 0-based.
  - coeff_array is (nvars+1, P) — row k holds the coefficient vector after
    the k-th LARS step (row 0 = zeros = initial state).
    MATLAB coeff_array(k+1, ...) = Python coeff_array[k, ...]
  - a_scores / loo_scores : same shift — index k corresponds to k LARS
    iterations completed (k=0 : constant-only score).
"""

import warnings
import numpy as np


# ---------------------------------------------------------------------------
# 1. uq_PCE_loo_error
#    Source: modules/uq_model/builtin/uq_metamodel/PCE/uq_PCE_loo_error.m
# ---------------------------------------------------------------------------
def uq_PCE_loo_error(Psi, M, Y, coefficients=None,
                     modified_flag=True, modi_diag=None):
    """
    [loo, normEmpErr, opt_results] = uq_PCE_loo_error(Psi, M, Y,
                                         coefficients, modified_flag, modi_diag)

    LOO error for OLS regression.  Always follows the nargout==3 path of the
    MATLAB function (always computes T and the full opt_results struct).

    Parameters
    ----------
    Psi         : (N, NCoeff) ndarray – design matrix
    M           : (NCoeff, NCoeff) ndarray – (PsiT Psi)^{-1}
    Y           : (N,) ndarray – model evaluations
    coefficients: (NCoeff,) ndarray or None
                  If None, recomputed by OLS: M @ Psi.T @ Y
    modified_flag : bool – apply T correction (default True)
    modi_diag   : (N,) ndarray or None – centering correction to h

    Returns
    -------
    loo          : float
    normEmpErr   : float
    opt_results  : dict
    """
    # if no coefficients are provided, calculate them by OLS  (line 27-29)
    if coefficients is None:
        coefficients = M @ (Psi.T @ Y)

    # initialize the optional results to the empty matrix  (line 32)
    opt_results = {}

    # size of the currently accepted basis  (lines 34-35)
    N      = Psi.shape[0]
    NCoeff = coefficients.size

    # h factor — diagonal of hat matrix, computed without the full N×N mat
    # (lines 38-40) h_i = (Psi M Psi^T)_ii = rowsum(Psi*M .* Psi)
    PsiM = Psi @ M                          # N × NCoeff
    h    = np.sum(PsiM * Psi, axis=1)       # N,

    # adjust h for centered covariates  (lines 43-45)
    if modi_diag is not None:
        h = h + modi_diag

    # residuals and variance  (lines 47-49)
    res  = Psi @ coefficients - Y
    varY = np.var(Y, ddof=0)                # MATLAB: var(Y, 1) = pop. variance

    # (lines 52-61)
    if varY == 0:                           # if the data has 0 variance
        normEmpErr = 0.0
        loo        = 0.0
    else:
        normEmpErr = np.sum(res**2) / len(res) / varY
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loo_arr = (res / (1.0 - h))**2
        loo = float(np.mean(loo_arr)) / varY
        if np.isnan(loo):                   # NaN → Inf  (line 60)
            loo = np.inf

    # -----------------------------------------------------------------------
    # nargout == 3 branch  (lines 80-100) — always computed in Python
    # -----------------------------------------------------------------------
    trM = float(np.trace(M))               # (line 81)
    if trM < 0 or abs(trM) > 1e6:         # (lines 82-84) stability fallback
        trM = float(np.trace(np.linalg.pinv(Psi.T @ Psi)))
    if N > NCoeff:                         # (lines 85-88)
        T = N / (N - NCoeff) * (1.0 + trM)
    else:
        T = np.inf

    opt_results['T']                    = T
    opt_results['loo']                  = loo
    opt_results['ModifiedLoo']          = loo * T
    opt_results['normEmpErr']           = normEmpErr
    opt_results['ModifiednormEmpErr']   = normEmpErr * T

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        opt_results['LooPred'] = Y + np.sqrt(T) * res / (1.0 - h)  # (line 95)

    if modified_flag:                      # (lines 96-98)
        loo        = T * loo
        normEmpErr = T * normEmpErr

    return loo, normEmpErr, opt_results


# ---------------------------------------------------------------------------
# 2. uq_blockwise_inverse
#    Source: lib/uq_matrix_utils/uq_blockwise_inverse.m
# ---------------------------------------------------------------------------
def uq_blockwise_inverse(Ainv, B, C, D):
    """
    M = uq_blockwise_inverse(Ainv, B, C, D)

    Blockwise inversion of [[A B]; [C D]] given Ainv = A^{-1}.
    Used in uq_lar to perform rank-1 updates of the Gram matrix inverse
    as new regressors are added one at a time.

    In the LAR context:
      Ainv : (k, k)  — current inverse
      B    : (k, 1)  — new cross-correlation column
      C    : (1, k)  — B^T
      D    : (1, 1)  — self-correlation of new regressor  (scalar)
    """
    # ensure at least 2-D  (for uniform matrix algebra below)
    Ainv = np.atleast_2d(np.asarray(Ainv, dtype=float))
    B    = np.atleast_2d(np.asarray(B,    dtype=float))
    C    = np.atleast_2d(np.asarray(C,    dtype=float))
    D    = np.atleast_2d(np.asarray(D,    dtype=float))

    # (lines 22-31)  Schur complement SC = D - C Ainv B
    if D.shape == (1, 1):                  # isscalar(D) in MATLAB
        sc_val  = float(D[0, 0]) - float(np.asarray(C @ Ainv @ B).item())
        if sc_val == 0.0:
            sc_val = np.finfo(float).tiny
        SCinv   = float(1.0 / sc_val)     # scalar
        # (lines 34-36)  cached products
        T1      = Ainv @ B * SCinv        # (k, 1)  — Ainv*B scaled by scalar
        T2      = C @ Ainv                # (1, k)
        # (lines 38-39)  assembly
        return np.block([
            [Ainv + T1 @ T2,            -T1             ],
            [(-SCinv) * T2,             np.array([[SCinv]])],
        ])
    else:
        SC    = D - C @ Ainv @ B
        SCinv = np.linalg.solve(SC, np.eye(SC.shape[0]))
        T1    = Ainv @ B @ SCinv
        T2    = C @ Ainv
        return np.block([
            [Ainv + T1 @ T2,   -T1  ],
            [(-SCinv) @ T2,     SCinv],
        ])


# ---------------------------------------------------------------------------
# 3. uq_PCE_OLS_regression
#    Source: modules/.../PCE/PolyCoeff/Regression/uq_PCE_OLS_regression.m
# ---------------------------------------------------------------------------
def uq_PCE_OLS_regression(Psi, Y, options=None):
    """
    results = uq_PCE_OLS_regression(Psi, Y, options)

    Ordinary Least Squares regression.  Returns a dict with fields
    'coefficients', 'LOO', 'normEmpErr', 'optErrorParams'.
    """
    results    = {}
    COVFLAG    = False
    modified_loo = False

    if options is not None:
        if 'CY' in options:
            COVFLAG = True
            CY      = options['CY']
        if 'modified_loo' in options:
            modified_loo = bool(options['modified_loo'])

    # (lines 27-33)  apply covariance weight matrix if provided
    if COVFLAG:
        CYinv = np.linalg.solve(CY, np.eye(CY.shape[0]))
        L     = np.linalg.cholesky(CYinv)      # MATLAB: chol returns upper tri
        # MATLAB chol(A) = R s.t. R'*R = A (upper triangular)
        # numpy cholesky returns lower triangular L s.t. L*L' = A
        # We need L s.t. L*L' = CYinv, then apply L' as in MATLAB L*Psi
        # MATLAB: L = chol(CYinv), so L is upper tri and L'*L = CYinv
        # → Psi_new = L*Psi uses the upper-tri factor
        # Python numpy cholesky gives lower L; we want upper = L.T
        L    = L.T          # upper triangular (same as MATLAB's chol)
        Psi  = L @ Psi
        Y    = L @ Y

    # (lines 35-45)  invert the linear system PsiTPsi a = PsiT Y
    PsiTPsi = Psi.T @ Psi
    try:
        rcond_val = 1.0 / np.linalg.cond(PsiTPsi)
    except np.linalg.LinAlgError:
        rcond_val = 0.0

    if rcond_val > 1e-12:                  # (line 37) faster path
        results['coefficients'] = np.linalg.solve(PsiTPsi, Psi.T @ Y)
        M = np.linalg.solve(PsiTPsi, np.eye(PsiTPsi.shape[0]))
    else:                                  # (lines 43-44) stable path
        M = np.linalg.pinv(PsiTPsi)
        results['coefficients'] = M @ Psi.T @ Y

    # (lines 48-53)  compute LOO
    LOO, normEmpErr, optErrorParams = uq_PCE_loo_error(
        Psi, M, Y, results['coefficients'], modified_loo)

    results['LOO']             = LOO
    results['normEmpErr']      = normEmpErr
    results['optErrorParams']  = optErrorParams

    return results


# ---------------------------------------------------------------------------
# 4. uq_lar
#    Source: lib/uq_regression/LAR/uq_lar.m
# ---------------------------------------------------------------------------
def uq_lar(Psi, Y, options=None):
    """
    results = uq_lar(Psi, Y, options)

    Hybrid Least Angle Regression (LARS) for sparse PCE.
    Word-for-word translation of uq_lar.m.

    All column indices in the returned results are 0-based (Python convention).

    Parameters
    ----------
    Psi     : (N, P) ndarray — design matrix (polynomial evaluations)
    Y       : (N,)  ndarray — model responses
    options : dict or None.  Keys: 'normalize', 'early_stop', 'hybrid_lars',
              'no_selection', 'loo_modified', 'loo_hybrid', 'display', 'CY'

    Returns
    -------
    results : dict with keys listed in the module docstring.
    """
    results = {}

    # -----------------------------------------------------------------------
    # Default option values  (lines 48-55 in uq_lar.m)
    # -----------------------------------------------------------------------
    normalize_columns = True    # normalize = 1
    early_stop        = True    # early_stop = 1
    hybrid_lars       = True    # hybrid_lars = 1
    modified_loo      = True    # modified_loo = 1
    hybrid_loo        = True    # hybrid_loo = 1
    no_selection      = False   # no_selection = 0
    generalized_ls    = False   # generalized_ls = 0
    DisplayLevel      = 0
    CY                = None

    # -----------------------------------------------------------------------
    # Parse options  (lines 57-101)
    # -----------------------------------------------------------------------
    if options is not None:
        if 'normalize'    in options: normalize_columns = bool(options['normalize'])
        if 'early_stop'   in options: early_stop        = bool(options['early_stop'])
        if 'hybrid_lars'  in options: hybrid_lars       = bool(options['hybrid_lars'])
        if 'no_selection' in options: no_selection      = bool(options['no_selection'])
        if 'loo_modified' in options: modified_loo      = bool(options['loo_modified'])
        if 'loo_hybrid'   in options: hybrid_loo        = bool(options['loo_hybrid'])
        if 'display'      in options: DisplayLevel      = options['display']
        if 'CY'           in options:
            CY            = options['CY']
            generalized_ls = True

    # OLS options  (line 103-104)
    olsoptions = {'modified_loo': modified_loo}

    # (line 107)
    N, P = Psi.shape
    Psi  = Psi.astype(float).copy()    # work on a copy (MATLAB modifies Psi)
    Y    = Y.astype(float).copy()

    # -----------------------------------------------------------------------
    # Trivial case: P == 1  (lines 109-138)
    # -----------------------------------------------------------------------
    if P == 1:
        if generalized_ls:
            olsoptions['CY'] = CY
        ols_results = uq_PCE_OLS_regression(Psi, Y, olsoptions)

        results['coefficients']    = ols_results['coefficients']
        results['LOO']             = ols_results['LOO']
        results['normEmpErr']      = ols_results['normEmpErr']
        results['optErrorParams']  = ols_results['optErrorParams']
        results['coeff_array']     = results['coefficients'].copy()
        results['max_score']       = 1.0 - results['LOO']
        results['a_scores']        = np.array([results['max_score']])
        results['loo_scores']      = np.array([results['LOO']])
        results['best_basis_index'] = 0                 # MATLAB: 1 → Python: 0
        results['nz_idx']          = np.array([True])
        results['LOO_lars']        = results['LOO']
        results['lars_idx']        = np.array([0])      # MATLAB: 1 → Python: 0
        return results

    # -----------------------------------------------------------------------
    # Generalized LS: decorrelate  (lines 141-148)
    # -----------------------------------------------------------------------
    if generalized_ls:
        CYinv = np.linalg.solve(CY, np.eye(CY.shape[0]))
        L     = np.linalg.cholesky(CYinv).T    # upper triangular (MATLAB chol)
        Psi   = L @ Psi
        Y     = L @ Y

    # -----------------------------------------------------------------------
    # Centering & normalization  (lines 150-183)
    # -----------------------------------------------------------------------
    # constant regressors: columns with zero diff
    # MATLAB: constidx = ~any(diff(Psi, 1))  — diff along rows (axis=0)
    diffs       = np.diff(Psi, axis=0)               # (N-1, P)
    constidx    = ~np.any(diffs, axis=0)              # (P,) bool
    constindices = np.where(constidx)[0]              # 0-indexed
    constval     = Psi[0, constidx].copy()            # values of const cols
    bool_const_and_center = bool(np.any(constidx))

    mu_Psi = None
    mu_Y   = None
    if bool_const_and_center:
        # (lines 163-170)
        mu_Psi = np.mean(Psi, axis=0)
        Psi    = Psi - mu_Psi                         # broadcasts over N rows
        mu_Y   = float(np.mean(Y))
        Y      = Y - mu_Y
        modi_diag         = np.full(N, 1.0 / N)       # 1/N * ones(N,1)
        run_lars_iterations = float(np.var(Y, ddof=0)) # 0 iff const Y
    else:
        modi_diag         = np.zeros(N)
        run_lars_iterations = 1.0

    # (lines 178-183)  normalize columns to unit variance
    nznorm_indices = None
    normPsi        = None
    if normalize_columns:
        # stddev (with N-1 denominator, works on centered data)
        normPsi        = np.sqrt(np.sum(Psi**2, axis=0) / (N - 1))
        nznorm_indices = normPsi != 0.0
        Psi[:, nznorm_indices] /= normPsi[nznorm_indices]

    # -----------------------------------------------------------------------
    # Initialise LAR iterations  (lines 189-205)
    # -----------------------------------------------------------------------
    nvars      = min(N - 2, P)          # max active predictors
    mu         = np.zeros_like(Y)       # current LAR direction
    a_coeff    = []                     # active column indices (0-based)
    i_coeff    = list(range(P))         # candidate indices (0-based)
    M_gram     = None                   # inverse Gram matrix (grows k×k)
    maxk       = 8 * nvars
    coeff_array = np.zeros((nvars + 1, P))

    a_scores   = np.full(nvars + 1, -np.inf)
    loo_scores = np.full(nvars + 1, np.inf)

    refscore   = np.inf

    if DisplayLevel > 1:
        print(f'Maximum LARS candidate basis size: {P}')

    # initial score: OLS with constant term  (lines 213-220)
    if bool_const_and_center:
        ols_init  = uq_PCE_OLS_regression(
            np.ones((N, 1)), Y, olsoptions)
        loo_scores[0] = ols_init['LOO']          # MATLAB: loo_scores(1)
        a_scores[0]   = 1.0 - loo_scores[0]     # MATLAB: a_scores(1)

    maxiter = min(maxk, nvars)

    # -----------------------------------------------------------------------
    # Main LARS loop  (lines 224-366)
    # -----------------------------------------------------------------------
    k = 0
    while k < maxiter and run_lars_iterations:
        k += 1              # MATLAB: k = k + 1  at the TOP of the loop

        if DisplayLevel > 3:
            print(f'Computing LAR iteration {k}')

        # correlation with the residual  (lines 231-234)
        cj      = Psi.T @ (Y - mu)              # (P,)
        i_arr   = np.array(i_coeff)             # 0-indexed candidate indices
        rel_corr = np.abs(cj[i_arr])
        local_idx = int(np.argmax(rel_corr))
        C       = float(rel_corr[local_idx])
        idx     = int(i_arr[local_idx])         # absolute 0-indexed column

        # Information matrix update  (lines 239-265)
        if k == 1:
            # (line 240)  M = pinv(Psi(:,idx)'*Psi(:,idx))
            val    = float(Psi[:, idx] @ Psi[:, idx])
            M_gram = np.linalg.pinv(np.array([[val]]))  # (1,1)
        else:
            # (lines 242-246)  rank-1 blockwise update
            a_arr_old = np.array(a_coeff)       # OLD active set (before add)
            x = (Psi[:, a_arr_old].T @ Psi[:, idx]).reshape(-1, 1)  # (k-1,1)
            r = float(Psi[:, idx] @ Psi[:, idx])
            M_gram = uq_blockwise_inverse(M_gram, x, x.T, np.array([[r]]))

            # (lines 249-265)  singularity recovery
            if not np.all(np.isfinite(M_gram)):
                combined = a_coeff + [idx]
                try:
                    Pc = Psi[:, combined]
                    M_gram = np.linalg.pinv(Pc.T @ Pc)
                except Exception:
                    if DisplayLevel > 1:
                        print('singular design matrix. Skipping basis element.')
                    i_coeff.remove(idx)
                    if a_coeff:
                        Pa = Psi[:, np.array(a_coeff)]
                        M_gram = np.linalg.pinv(Pa.T @ Pa)
                    continue     # skip rest of this iteration (MATLAB continue)

        # update active set  (line 268)
        a_coeff.append(idx)
        a_arr = np.array(a_coeff)               # updated active set

        # correlation signs  (lines 272-274)
        s = np.sign(cj[a_arr])
        s[s == 0.0] = 1.0                       # MATLAB: s(~s) = 1

        # c, w, u, aj  (lines 286-293)
        c_lars = float(1.0 / np.sqrt(s @ M_gram @ s))   # scalar, called 'c' in MATLAB
        w      = c_lars * M_gram @ s             # (k,)
        u      = Psi[:, a_arr] @ w               # (N,)
        aj     = Psi.T @ u                       # (P,)

        # gamma  (lines 296-310)
        if k < nvars:
            # (line 299)  remove idx from i_coeff temporarily for the calc
            # NOTE: in MATLAB i_coeff is updated AFTER gamma, so here too
            # we use the *current* i_coeff (still contains idx at this point)
            i_arr_gamma = np.array(i_coeff)      # includes idx (not yet removed)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                tmp1  = (C - cj[i_arr_gamma]) / (c_lars - aj[i_arr_gamma])
                tmp2  = (C + cj[i_arr_gamma]) / (c_lars + aj[i_arr_gamma])
            tmp   = np.concatenate([tmp1, tmp2])
            pos   = tmp > 0.0
            if np.any(pos):
                gamma = float(np.min(tmp[pos]))
            else:
                gamma = 0.0
                if DisplayLevel >= 3:
                    print(f'Warning: numerical instability!! Gamma set to 0 '
                          f'at LAR iteration {k}.')
        else:
            gamma = C / c_lars              # (line 309) OLS solution

        # remove idx from candidate set  (line 313)
        i_coeff.remove(idx)

        # update residual  (line 316)
        mu = mu + gamma * u

        # update coefficient array  (line 320)
        # MATLAB: coeff_array(k+1, a_coeff) = coeff_array(k, a_coeff) + gamma*w'
        # Python: coeff_array[k, a_arr]     = coeff_array[k-1, a_arr] + gamma*w
        coeff_array[k, a_arr] = coeff_array[k - 1, a_arr] + gamma * w

        # LOO estimate  (lines 333-342)
        if not hybrid_loo:
            # pass current LARS coefficients for the active set
            coeff_k = coeff_array[k, a_arr]     # MATLAB: coeff_array(k+1,a_coeff)'
            loo, _, _ = uq_PCE_loo_error(
                Psi[:, a_arr], M_gram, Y, coeff_k,
                modified_loo, modi_diag)
        else:
            loo, _, _ = uq_PCE_loo_error(
                Psi[:, a_arr], M_gram, Y, None,
                modified_loo, modi_diag)

        if loo < 0:
            warnings.warn('leave one out error negative!!')

        # MATLAB: loo_scores(k+1) = loo; a_scores(k+1) = 1-loo
        loo_scores[k] = loo
        a_scores[k]   = 1.0 - loo

        # early stop  (lines 346-365)
        mm = round(nvars * 0.1)
        mm = max(mm, 100)
        mm = min(mm, nvars)

        if loo < refscore:
            refscore = loo

        if k > mm:
            # MATLAB: if (loo_scores(k-mm) <= refscore) && early_stop
            # loo_scores(k-mm) in MATLAB (1-indexed) → loo_scores[k-mm-1] Python
            if loo_scores[k - mm - 1] <= refscore and early_stop:
                if DisplayLevel > 1:
                    print(f'Early stop at coefficient {k - mm}/{P}')
                break

    # -----------------------------------------------------------------------
    # Select the best iteration  (lines 369-375)
    # -----------------------------------------------------------------------
    if not no_selection:
        k_best   = int(np.argmax(a_scores))     # 0-indexed best iteration
        maxScore = float(a_scores[k_best])
    else:
        maxScore = 1.0 - loo
        k_best   = len(a_scores) - 1

    # -----------------------------------------------------------------------
    # Hybrid LARS: assign final coefficients via OLS  (lines 378-423)
    # -----------------------------------------------------------------------
    # (line 382)  nz_idx from best coeff_array row
    nz_idx = np.abs(coeff_array[k_best, :]) > 0.0   # (P,) bool

    # scale back Psi  (lines 384-393)
    if normalize_columns and nznorm_indices is not None:
        Psi[:, nznorm_indices] *= normPsi[nznorm_indices]

    if bool_const_and_center:
        Psi = Psi + mu_Psi                          # bsxfun(@plus, Psi, mu_Psi)
        Y   = Y + mu_Y
        nz_idx[constindices[0]] = True              # always include constant

    # (lines 400-423)  final coefficient assignment
    coefficients = np.zeros(P)
    if hybrid_lars:
        # recompute via OLS on the selected basis
        ols_final = uq_PCE_OLS_regression(Psi[:, nz_idx], Y, olsoptions)
        coefficients[nz_idx]      = ols_final['coefficients']
        results['LOO']            = ols_final['LOO']
        results['normEmpErr']     = ols_final['normEmpErr']
        results['optErrorParams'] = ols_final['optErrorParams']
    else:
        # rescale the LARS solution
        coefficients[nz_idx] = coeff_array[k_best, nz_idx]
        if normalize_columns and nznorm_indices is not None:
            coefficients[nznorm_indices] /= normPsi[nznorm_indices]
        if bool_const_and_center:
            residual = Y - Psi @ coefficients
            coefficients[constindices[0]] = (
                float(np.mean(residual)) / float(constval[0]))
        # LOO for the non-hybrid case
        Pnz         = Psi[:, nz_idx]
        M_final     = np.linalg.pinv(Pnz.T @ Pnz)
        results['LOO'], results['normEmpErr'], results['optErrorParams'] = \
            uq_PCE_loo_error(Pnz, M_final, Y, coefficients[nz_idx], True)

    if DisplayLevel > 1:
        print(f'LAR basis size: {int(np.sum(nz_idx))}/{P}')

    # -----------------------------------------------------------------------
    # Assign remaining outputs  (lines 431-455)
    # -----------------------------------------------------------------------
    results['coeff_array']      = coeff_array
    results['max_score']        = maxScore
    results['coefficients']     = coefficients
    results['a_scores']         = a_scores
    results['loo_scores']       = loo_scores
    results['best_basis_index'] = k_best            # 0-indexed (MATLAB: 1-indexed)
    results['nz_idx']           = nz_idx

    # lars_idx: ordered list of active column indices up to best iteration
    # MATLAB: a_coeff(1:(k-1)) where k is MATLAB's 1-indexed k_best
    #         → Python: a_coeff[:k_best]   (k_best is 0-indexed)
    if bool_const_and_center:
        results['lars_idx'] = (
            [int(constindices[0])] + [int(v) for v in a_coeff[:k_best]])
    else:
        results['lars_idx'] = [int(v) for v in a_coeff[:k_best]]

    results['LOO_lars'] = 1.0 - maxScore

    return results


# ---------------------------------------------------------------------------
# 5. uq_PCE_lars  (thin wrapper — mirrors uq_PCE_lars.m interface)
#    In the Python PCK clone, B3 (PCK_calculate) calls uq_lar directly.
#    This wrapper is provided for completeness / unit-testing.
# ---------------------------------------------------------------------------
def uq_PCE_lars(Psi, Y,
                LarsEarlyStop=True,
                ModifiedLoo=True,
                HybridLoo=True,
                Normalize=True,
                DisplayLevel=0,
                CY=None):
    """
    Thin wrapper around uq_lar that mirrors the interface of uq_PCE_lars.m.

    Parameters are the scalar flags stored in
    current_model.Internal.PCE(oo).LARS.* in UQLab.

    Returns the raw uq_lar result dict, augmented with:
      'LOO_lars'  = 1 - max_score
      'lars_idx'  = ordered 0-indexed column selection
    """
    lar_options = {
        'early_stop':   LarsEarlyStop,
        'normalize':    Normalize,
        'hybrid_lars':  True,            # always True in uq_PCE_lars.m (line 39)
        'loo_modified': ModifiedLoo,
        'loo_hybrid':   HybridLoo,
        'display':      DisplayLevel,
    }
    if CY is not None:
        lar_options['CY'] = CY

    return uq_lar(Psi, Y, lar_options)
