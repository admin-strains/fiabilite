"""
Ajustement PCK et GEPCK -- orchestration -- extrait de branche3.py.

Assemble la tendance PCE et le krigeage du residu : selection de la base
par LARS, puis ajustement du krigeage. C'est le point d'entree appele par
`api.fit_pck` et `api.fit_gepck`.

PHASE 3 du plan de nettoyage : `branche3.py` melait le krigeage, la base
polynomiale et l'orchestration de l'ajustement. Le graphe d'appels
(`tools/analyse_dependances.py`) montre une stratification nette --
`fit` appelle `kriging` et `pce_basis`, qui ne s'appellent pas entre eux.

Le corps des fonctions est repris VERBATIM : la scission ne doit changer
aucun resultat, et la baseline le verifie.
"""

import copy
import numpy as np

from kernels import uq_eval_Kernel
from kriging import fit_kriging_pck, fit_kriging_gepck
from pce_basis import (
    poly_type_from_marginal, aux_marginal_from_poly_type,
    pce_multi_indices, pce_eval_design_matrix,
)
from lars import uq_lar
from polynomials import (
    uq_PCE_create_Psi, uq_PCK_eval_unipoly,
    uq_eval_hermite_deriv, uq_eval_legendre_deriv,
)


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
        _deg_list  = pce_opts.get('Degree', [1, 2, 3])
        max_degree = max(_deg_list) if _deg_list else 0

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


def uq_GEPCK_calculate_coefficients(X, Y_aug, pck_config,
                                     input_marginals, input_copula,
                                     CorrOptions=None,
                                     theta_bounds=None,
                                     theta0=None,
                                     optim_method='gradbased',
                                     estim_method='ml'):
    """
    Fit a GEPCK metamodel (sortie unique).

    Clone de uq_PCK_calculate_coefficients avec Y_aug, make_trend_global_handle,
    fit_kriging_gepck et uq_eval_global_Kernel.

    Parameters
    ----------
    X               : (N, M) ndarray
    Y_aug           : (N*(M+1),) ndarray — [y ; dy/du_0 ; ... ; dy/du_{M-1}]
                      fourni par l'utilisateur (pas assemblé ici)
    pck_config      : dict de uq_PCK_initialize (B2)
    input_marginals : list de M dicts
    input_copula    : dict copule
    CorrOptions     : dict; si None utilise Matern52 + uq_eval_global_Kernel
    theta_bounds    : (2, M) ndarray; si None [[0.01]*M, [100]*M]
    theta0          : (M,) ndarray; si None moyenne géométrique des bornes
    optim_method    : 'gradbased' | 'de' | 'none'
    estim_method    : 'ml' | 'cv'
    """
    from kernels import uq_eval_global_Kernel

    X     = np.atleast_2d(X).astype(float)
    Y_aug = np.asarray(Y_aug).ravel().astype(float)
    N, M  = X.shape
    Nout  = 1   # GEPCK : sortie unique

    # -----------------------------------------------------------------------
    # Dimensions non-constantes
    # -----------------------------------------------------------------------
    diff_X   = np.diff(X, axis=0)
    nonConst = np.where(np.any(diff_X, axis=0))[0]
    if len(nonConst) == 0:
        raise ValueError('Only constants in the input model.')
    Xred = X[:, nonConst]
    Mred = len(nonConst)

    red_marginals = [input_marginals[i] for i in nonConst]

    # -----------------------------------------------------------------------
    # PolyTypes et espace auxiliaire
    # -----------------------------------------------------------------------
    PolyTypes_all = [poly_type_from_marginal(m['Type']) for m in red_marginals]
    aux_marginals = [aux_marginal_from_poly_type(pt) for pt in PolyTypes_all]
    aux_copula    = {'Type': 'Independent', 'Parameters': np.eye(Mred)}

    # -----------------------------------------------------------------------
    # CorrOptions défaut : Matern-5/2 + uq_eval_global_Kernel
    # -----------------------------------------------------------------------
    if CorrOptions is None:
        CorrOptions = {
            'Handle'    : uq_eval_global_Kernel,
            'Family'    : 'matern-5_2',
            'Type'      : 'separable',
            'Isotropic' : False,
            'Nugget'    : 0.0,
        }

    # -----------------------------------------------------------------------
    # theta_bounds et theta0
    # -----------------------------------------------------------------------
    if theta_bounds is None:
        theta_bounds = np.array([[0.01] * Mred, [100.0] * Mred])
    if theta0 is None:
        theta0 = np.sqrt(theta_bounds[0] * theta_bounds[1])

    # -----------------------------------------------------------------------
    # LARS sur Y_aug[:N] (valeurs seules — les gradients n'entrent pas dans LARS)
    # -----------------------------------------------------------------------
    mode         = pck_config.get('Mode', 'sequential').lower()
    trend_method = pck_config.get('TrendMethod', 'pce').lower()
    Y_vals       = Y_aug[:N]

    idxranking = [None]
    AllIndices = [None]

    if trend_method == 'pce':
        pce_opts   = pck_config.get('PCE', {})
        _deg_list  = pce_opts.get('Degree', [1, 2, 3])
        max_degree = max(_deg_list) if _deg_list else 0

        FullIndices = pce_multi_indices(Mred, max_degree)
        PolyTypes   = PolyTypes_all

        Psi_full, U_train = pce_eval_design_matrix(
            Xred, FullIndices, PolyTypes,
            red_marginals, input_copula, aux_marginals)

        lar_opts = {
            'normalize'   : True,
            'hybrid_lars' : True,
            'loo_modified': True,
            'loo_hybrid'  : True,
            'early_stop'  : True,
        }
        lar_res       = uq_lar(Psi_full, Y_vals, lar_opts)
        idxranking[0] = lar_res['lars_idx']
        AllIndices[0] = FullIndices

    else:  # 'user'
        PolyIndices   = pck_config['PolyIndices']
        PolyTypes     = pck_config['PolyTypes']
        idxranking[0] = list(range(PolyIndices.shape[0]))
        AllIndices[0] = PolyIndices
        _, U_train = pce_eval_design_matrix(
            Xred, AllIndices[0], PolyTypes,
            red_marginals, input_copula, aux_marginals)

    # -----------------------------------------------------------------------
    # Trend handle factories (GEPCK — not shared with PCK)
    # -----------------------------------------------------------------------
    def make_trend_handle(selected_idx, Indices, poly_types):
        Idx_sel = Indices[np.array(selected_idx), :]
        p_types = poly_types
        def F_handle(U):
            uv = uq_PCK_eval_unipoly(U, Idx_sel, p_types)
            return uq_PCE_create_Psi(Idx_sel, uv)
        return F_handle

    def make_trend_handle_deriv(selected_idx, Indices, poly_types, der):
        Idx_sel = Indices[np.array(selected_idx), :]
        p_types = poly_types
        def F_der_handle(U):
            uv  = uq_PCK_eval_unipoly(U, Idx_sel, p_types)
            P   = uv.shape[2] - 1
            pt  = p_types[der].lower()
            if pt == 'hermite':
                uv[:, der, :] = uq_eval_hermite_deriv(P, U[:, der])
            elif pt == 'legendre':
                uv[:, der, :] = uq_eval_legendre_deriv(P, U[:, der])
            Psi = uq_PCE_create_Psi(Idx_sel, uv)
            Psi[:, Idx_sel[:, der] == 0] = 0.0
            return Psi
        return F_der_handle

    def make_trend_global_handle(selected_idx, Indices, poly_types):
        Idx_sel = Indices[np.array(selected_idx), :]
        M_loc   = Idx_sel.shape[1]
        F0 = make_trend_handle(selected_idx, Indices, poly_types)
        Fk = [make_trend_handle_deriv(selected_idx, Indices, poly_types, k)
              for k in range(M_loc)]
        def F_global(U):
            return np.vstack([F0(U)] + [fk(U) for fk in Fk])
        return F_global

    # -----------------------------------------------------------------------
    # Fit GEPCK (mode sequential ou optimal)
    # -----------------------------------------------------------------------
    comb_crit  = pck_config.get('CombCrit', 'rel_loo').lower()
    idx_ranked = idxranking[0]
    Indices_oo = AllIndices[0]

    if mode == 'sequential':
        F_global_handle = make_trend_global_handle(
            idx_ranked, Indices_oo, PolyTypes_all[:Mred])

        fitted_kriging = fit_kriging_gepck(
            U_train, Y_aug, F_global_handle,
            CorrOptions, theta_bounds, theta0.copy(),
            estim_method=estim_method,
            optim_method=optim_method)

        NumberOfPoly = len(idx_ranked)

    else:  # 'optimal'
        best_LOO    = np.inf
        best_fitted = None
        best_ii     = 0

        # Init GA sur trend constant augmenté (Zuhal 2021 Section III.B.1)
        F_global_constant = make_trend_global_handle(
            [0], Indices_oo, PolyTypes_all[:Mred])
        fitted_constant = fit_kriging_gepck(
            U_train, Y_aug, F_global_constant,
            CorrOptions, theta_bounds, theta0.copy(),
            estim_method=estim_method,
            optim_method='de')
        theta_current = fitted_constant['theta']

        for ii in range(1, len(idx_ranked) + 1):
            F_global_handle = make_trend_global_handle(
                idx_ranked[:ii], Indices_oo, PolyTypes_all[:Mred])

            fitted = fit_kriging_gepck(
                U_train, Y_aug, F_global_handle,
                CorrOptions, theta_bounds, theta_current,
                estim_method=estim_method,
                optim_method=optim_method)

            theta_current = fitted['theta']

            if fitted['LOO'] < best_LOO:
                best_LOO    = fitted['LOO']
                best_fitted = fitted
                best_ii     = ii

        fitted_kriging = best_fitted
        NumberOfPoly   = best_ii

    # Indices finaux utilisés dans le modèle sélectionné
    final_idx = idx_ranked if mode == 'sequential' else idx_ranked[:best_ii]

    # Closures ∂Ψ/∂u_k pour k=0..Mred-1 — utilisées par predict_deriv_gepck
    fitted_kriging['F_deriv_handles'] = [
        make_trend_handle_deriv(final_idx, Indices_oo, PolyTypes_all[:Mred], k)
        for k in range(Mred)
    ]

    # -----------------------------------------------------------------------
    # Résultat
    # -----------------------------------------------------------------------
    return {
        'Kriging'       : [fitted_kriging],
        'Error'         : [{'LOO': fitted_kriging['LOO']}],
        'AuxSpace'      : {'Marginals': aux_marginals, 'Copula': aux_copula},
        'ExpDesign'     : {
            'X'    : X,
            'Y_aug': Y_aug,
            'U'    : U_train,
            'Xred' : Xred,
        },
        'pck_config'    : pck_config,
        'PolyTypes'     : PolyTypes_all[:Mred],
        'AllIndices'    : AllIndices,
        'idxranking'    : idxranking,
        'NumberOfPoly'  : NumberOfPoly,
        'nonConst'      : nonConst,
        'OrigMarginals' : input_marginals,
        'OrigCopula'    : input_copula,
        'RedMarginals'  : red_marginals,
        'CorrOptions'   : CorrOptions,
        'M'             : M,
        'Mred'          : Mred,
        'Nout'          : Nout,
    }
