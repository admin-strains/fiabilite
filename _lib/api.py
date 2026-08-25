"""
branche1.py -- Python entry point for PCK (PC-Kriging)

Sources (UQLab 2.2.0):
  uq_initialize_uq_metamodel.m  [dispatch / init]
  uq_eval_uq_metamodel.m        [evaluation dispatch]

In MATLAB, uq_initialize_uq_metamodel handles:
  - Session management (uq_getModel, uq_addprop, uq_calculateMetamodel)
  - ExpDesign generation (LHS, MC, user-supplied, data file, ...)
  - Display / verbosity
  - Multiple metamodel types (PCE, Kriging, LRA, PCK, SVR, SSE, ...)

In Python, B1 covers only the PCK path with user-supplied (X, Y):
  1. fit_pck   -- calls B2 (uq_PCK_initialize) then B3 (uq_PCK_calculate_coefficients)
  2. predict_pck -- thin wrapper around B4 (uq_PCK_eval)

MATLAB -> Python conventions:
  - 'user' ExpDesign path only (lines ~175-210 of uq_initialize_uq_metamodel.m)
  - uq_getModel / uq_addprop / session tracking: not needed in Python
  - uq_calculateMetamodel: replaced by direct call to uq_PCK_calculate_coefficients
"""

import numpy as np
import sys, os
_dir = os.path.dirname(__file__)
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from options import uq_PCK_initialize
from fit import uq_PCK_calculate_coefficients, uq_GEPCK_calculate_coefficients
from predict import uq_PCK_eval, uq_GEPCK_eval, uq_GEPCK_eval_deriv
from kernels import uq_eval_Kernel, uq_eval_global_Kernel


# ===========================================================================
# 1.  fit_pck
#     Equivalent of uq_createModel(OPTIONS, MetaType='PCK')
#     restricted to the user-supplied ExpDesign path.
#
#     MATLAB chain:
#       uq_createModel
#       -> uq_initialize_uq_metamodel  (lines ~285-295: case 'pck' -> B2)
#       -> uq_calculateMetamodel       (case 'pck' -> B3)
# ===========================================================================
# =============================================================================
# TEMPORAIRE -- Génération de DOE (hors périmètre B1 MATLAB)
# Dans UQLab, la génération est faite par uq_getExpDesignSample +
# uq_eval_ExpDesign (lib/ExpDesign/) -- non traduit en Python.
# Cette fonction est fournie uniquement pour les tests locaux.
# À supprimer une fois les tests terminés.
# =============================================================================
def generate_doe(N, marginals, method='lhs', seed=None):
    """
    TEMPORAIRE -- Génère un plan d'expériences.

    Parameters
    ----------
    N         : int   nombre de points
    marginals : list of M dicts {'Type': str, 'Parameters': [a, b]}
    method    : 'lhs' (Latin Hypercube, défaut) | 'mc' (Monte Carlo)
    seed      : int ou None

    Returns
    -------
    X : (N, M) ndarray dans l'espace d'entrée original
    """
    from scipy.stats import qmc
    from scipy.special import ndtri   # quantile normale standard

    M = len(marginals)

    if method == 'lhs':
        sampler = qmc.LatinHypercube(d=M, seed=seed)
        U = sampler.random(N)          # (N, M) dans [0, 1]^M
    else:                              # 'mc'
        rng_loc = np.random.default_rng(seed)
        U = rng_loc.uniform(0, 1, (N, M))

    X = np.zeros((N, M))
    for j, marg in enumerate(marginals):
        mtype  = marg['Type'].lower()
        params = marg['Parameters']
        if mtype == 'uniform':
            a, b = params[0], params[1]
            X[:, j] = a + (b - a) * U[:, j]
        elif mtype in ('gaussian', 'normal'):
            mu, sigma = params[0], params[1]
            X[:, j] = mu + sigma * ndtri(np.clip(U[:, j], 1e-10, 1 - 1e-10))
        elif mtype == 'lognormal':
            mu, sigma = params[0], params[1]
            X[:, j] = np.exp(mu + sigma * ndtri(np.clip(U[:, j], 1e-10, 1 - 1e-10)))
        else:
            raise ValueError(f'generate_doe: type marginal non supporté "{mtype}"')

    return X


def fit_pck(X, Y, options, marginals, copula):
    """
    Main PCK entry point.

    Word-for-word translation of the PCK path in uq_initialize_uq_metamodel.m,
    restricted to user-supplied experimental design (X, Y given directly).

    Parameters
    ----------
    X         : (N, M) ndarray   training inputs
    Y         : (N,) or (N, Nout) ndarray   training outputs
    options   : dict   PCK configuration. Supported keys:
                  Mode          : 'sequential' (default) | 'optimal'
                  PCE           : dict  --  Degree, Method ('LARS' or 'OMP')
                  Kriging       : dict  --  Optim.Bounds, Optim.Method,
                                            EstimMethod, Corr.Family, ...
                  CombCrit      : str   ('rel_loo', for Mode='optimal')
                  PolyIndices   : (P, M) ndarray  (TrendMethod='user')
                  PolyTypes     : list of M strings (TrendMethod='user')
                  IgnoreDependence : bool (default False)
    marginals : list of M dicts, each {'Type': str, 'Parameters': [...]}
    copula    : dict {'Type': str, 'Parameters': ndarray}

    Returns
    -------
    fitted_model : dict   output of uq_PCK_calculate_coefficients (B3),
                          ready for predict_pck() / uq_PCK_eval() (B4).
    """
    X = np.atleast_2d(X).astype(float)
    Y = np.atleast_1d(Y).astype(float)
    M = X.shape[1]

    global_input = {'Marginals': marginals, 'Copula': copula}

    # -----------------------------------------------------------------------
    # Build current_model skeleton
    # uq_initialize_uq_metamodel.m: object created by uq_getModel,
    # here we build the equivalent Python dict.
    # -----------------------------------------------------------------------
    current_model = {
        'Options':  options,
        'Internal': {
            'Runtime': {'M': M},
            'Input':   global_input,
        },
    }

    # -----------------------------------------------------------------------
    # B2 -- uq_PCK_initialize
    # uq_initialize_uq_metamodel.m line ~290:
    #   case 'pck': uq_PCK_initialize(current_model)
    # Fills current_model['Internal'] with validated config.
    # -----------------------------------------------------------------------
    uq_PCK_initialize(current_model, global_input=global_input)

    # -----------------------------------------------------------------------
    # Extract pck_config from Internal
    # uq_calculateMetamodel -> case 'pck' -> uq_PCK_calculate_coefficients
    # The MATLAB function reads everything from current_model.Internal.
    # We mirror that by extracting the same fields.
    # -----------------------------------------------------------------------
    internal   = current_model['Internal']
    pck_config = {
        'Mode':        internal['Mode'],
        'TrendMethod': internal['TrendMethod'],
        'PCE':         internal.get('PCE', {}),
        'Kriging':     internal.get('Kriging', {}),
        'CombCrit':    internal.get('CombCrit', 'rel_loo'),
    }
    if 'PolyIndices' in internal:
        pck_config['PolyIndices'] = internal['PolyIndices']
    if 'PolyTypes' in internal:
        pck_config['PolyTypes']   = internal['PolyTypes']

    # -----------------------------------------------------------------------
    # Extract Kriging tuning options from internal config
    # Mirrors uq_Kriging_initialize + uq_Kriging_initialize_optimizer
    # -----------------------------------------------------------------------
    krig_opts    = internal.get('Kriging', {})
    theta_bounds = None
    theta0       = None
    optim_method = 'gradbased'   # default: L-BFGS-B  (MATLAB: fmincon)
    estim_method = 'ml'          # default: maximum likelihood
    CorrOptions  = None          # None -> B3 uses Matern-5/2 anisotropic

    if 'Optim' in krig_opts:
        ob = krig_opts['Optim']
        if 'Bounds' in ob:
            theta_bounds = ob['Bounds']
        if 'Method' in ob:
            optim_method = ob['Method'].lower()
        if 'InitialValue' in ob:
            theta0 = np.asarray(ob['InitialValue'], dtype=float)

    if 'EstimMethod' in krig_opts:
        estim_method = krig_opts['EstimMethod'].lower()

    if 'Corr' in krig_opts:
        # Build CorrOptions dict (mirrors uq_Kriging_init_Corr.m)
        corr = krig_opts['Corr']
        CorrOptions = {
            'Handle'    : uq_eval_Kernel,
            'Family'    : corr.get('Family',    'matern-5_2'),
            'Type'      : corr.get('Type',      'separable'),
            'Isotropic' : corr.get('Isotropic', False),
            'Nugget'    : corr.get('Nugget',    0.0),
        }

    # -----------------------------------------------------------------------
    # B3 -- uq_PCK_calculate_coefficients
    # uq_calculateMetamodel.m: case 'pck' -> uq_PCK_calculate_coefficients
    # -----------------------------------------------------------------------
    return uq_PCK_calculate_coefficients(
        X, Y, pck_config, marginals, copula,
        CorrOptions=CorrOptions,
        theta_bounds=theta_bounds,
        theta0=theta0,
        optim_method=optim_method,
        estim_method=estim_method,
    )


# ===========================================================================
# 3.  fit_gepck
#     Clone de fit_pck pour GEPCK.
# ===========================================================================
def fit_gepck(X, Y_aug, options, marginals, copula):
    """
    Main GEPCK entry point.

    Clone de fit_pck. Différences :
      - Y remplacé par Y_aug (N*(M+1),)
      - CorrOptions['Handle'] = uq_eval_global_Kernel
      - appelle uq_GEPCK_calculate_coefficients

    Parameters
    ----------
    X         : (N, M) ndarray   training inputs
    Y_aug     : (N*(M+1),) ndarray   [y ; dy/du_0 ; ... ; dy/du_{M-1}]
    options   : dict   PCK/GEPCK configuration (mêmes clés que fit_pck)
    marginals : list of M dicts
    copula    : dict
    """
    X     = np.atleast_2d(X).astype(float)
    Y_aug = np.asarray(Y_aug).ravel().astype(float)
    M     = X.shape[1]

    global_input = {'Marginals': marginals, 'Copula': copula}

    current_model = {
        'Options':  options,
        'Internal': {
            'Runtime': {'M': M},
            'Input':   global_input,
        },
    }

    uq_PCK_initialize(current_model, global_input=global_input)

    internal   = current_model['Internal']
    pck_config = {
        'Mode':        internal['Mode'],
        'TrendMethod': internal['TrendMethod'],
        'PCE':         internal.get('PCE', {}),
        'Kriging':     internal.get('Kriging', {}),
        'CombCrit':    internal.get('CombCrit', 'rel_loo'),
    }
    if 'PolyIndices' in internal:
        pck_config['PolyIndices'] = internal['PolyIndices']
    if 'PolyTypes' in internal:
        pck_config['PolyTypes']   = internal['PolyTypes']

    krig_opts    = internal.get('Kriging', {})
    theta_bounds = None
    theta0       = None
    optim_method = 'gradbased'
    estim_method = 'ml'
    CorrOptions  = None

    if 'Optim' in krig_opts:
        ob = krig_opts['Optim']
        if 'Bounds' in ob:
            theta_bounds = ob['Bounds']
        if 'Method' in ob:
            optim_method = ob['Method'].lower()
        if 'InitialValue' in ob:
            theta0 = np.asarray(ob['InitialValue'], dtype=float)

    if 'EstimMethod' in krig_opts:
        estim_method = krig_opts['EstimMethod'].lower()

    if 'Corr' in krig_opts:
        corr = krig_opts['Corr']
        CorrOptions = {
            'Handle'    : uq_eval_global_Kernel,   # ← GEPCK
            'Family'    : corr.get('Family',    'matern-5_2'),
            'Type'      : corr.get('Type',      'separable'),
            'Isotropic' : corr.get('Isotropic', False),
            'Nugget'    : corr.get('Nugget',    0.0),
        }

    return uq_GEPCK_calculate_coefficients(
        X, Y_aug, pck_config, marginals, copula,
        CorrOptions=CorrOptions,
        theta_bounds=theta_bounds,
        theta0=theta0,
        optim_method=optim_method,
        estim_method=estim_method,
    )


# ===========================================================================
# 4.  predict_gepck
#     Clone trivial de predict_pck.
# ===========================================================================
def predict_gepck(fitted_model, X_test,
                  return_var=False, return_cov=False):
    """
    Évalue un modèle GEPCK entraîné sur X_test.

    Clone de predict_pck — seul changement : uq_GEPCK_eval.
    """
    return uq_GEPCK_eval(fitted_model, X_test,
                         return_var=return_var, return_cov=return_cov)


# ===========================================================================
# 5.  predict_deriv_gepck  /  predict_gradient_gepck
#     Gradient analytique ∂ŷ/∂u dans l'espace auxiliaire U.
# ===========================================================================
def predict_deriv_gepck(fitted_model, X_test, der_var):
    """
    ∂ŷ/∂u_{der_var} analytique du GEPCK.

    Parameters
    ----------
    fitted_model : dict retourné par fit_gepck
    X_test       : (N_test, M)
    der_var      : int — indice de la variable dans l'espace auxiliaire (0-indexed)

    Returns
    -------
    dYMu : (N_test, Nout)
    """
    return uq_GEPCK_eval_deriv(fitted_model, X_test, der_var)


def predict_gradient_gepck(fitted_model, X_test):
    """
    Gradient complet ∂ŷ/∂u dans l'espace auxiliaire U.

    Returns
    -------
    G : (N_test, Mred)  — G[:, i] = ∂ŷ/∂u_i
    """
    Mred   = fitted_model['Mred']
    N_test = np.atleast_2d(X_test).shape[0]
    G = np.zeros((N_test, Mred))
    for i in range(Mred):
        G[:, i] = predict_deriv_gepck(fitted_model, X_test, i)[:, 0]
    return G


# ===========================================================================
# 2.  predict_pck
#     Equivalent of uq_evalModel(myPCK, X_test)
#
#     MATLAB chain:
#       uq_evalModel
#       -> uq_eval_uq_metamodel  (case 'pck' -> uq_PCK_eval)
# ===========================================================================
def predict_pck(fitted_model, X_test,
                return_var=False, return_cov=False):
    """
    Evaluate a fitted PCK model at X_test.

    Word-for-word translation of the PCK path in uq_eval_uq_metamodel.m:
        case 'pck': uq_PCK_eval(current_model, X)

    Parameters
    ----------
    fitted_model : dict returned by fit_pck()
    X_test       : (N_test, M) ndarray of test inputs
    return_var   : bool -- also return predictive variance
    return_cov   : bool -- also return full covariance matrix

    Returns
    -------
    YMu      : (N_test, Nout)
    YSigma2  : (N_test, Nout)             [if return_var or return_cov]
    YCov     : (N_test, N_test, Nout)     [if return_cov]
    """
    # uq_eval_uq_metamodel.m: case 'pck' -> uq_PCK_eval(current_model, X)
    return uq_PCK_eval(fitted_model, X_test,
                       return_var=return_var, return_cov=return_cov)
