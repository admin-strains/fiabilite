"""
branche5.py
Traduction mot-à-mot de la BRANCHE 5 d'UQLab 2.2.0 en Python.

Fonctions traduites (dans l'ordre de dépendance) :
  1. uq_poly_rec_coeffs      — coefficients de récurrence analytiques
  2. uq_eval_rec_rule         — évaluation par récurrence à 3 termes
  3. uq_eval_legendre         — polynômes de Legendre orthonormaux
  4. uq_eval_hermite          — polynômes de Hermite orthonormaux
  5. uq_PCK_eval_unipoly      — évaluation univariée par dimension
  6. uq_PCE_create_Psi        — assemblage de la matrice de design Ψ
  7. uq_GeneralIsopTransform  — transformation isoprobabiliste générale
  8. uq_assemble_Kernel       — formules des noyaux (Matérn, Gauss, …)
  9. uq_eval_Kernel           — calcul de la matrice de corrélation K

Note d'indexing : MATLAB est 1-indexé, Python est 0-indexé.
  MATLAB AB(k, col)  →  Python AB[k-1, col-1]
  MATLAB P(:, k)     →  Python P[:, k-1]
Tous les décalages sont indiqués en commentaire là où ils apparaissent.
"""

import numpy as np
from scipy.spatial.distance import cdist
import scipy.stats as stats


# =============================================================================
# 1. uq_poly_rec_coeffs
# =============================================================================

def uq_poly_rec_coeffs(n_max, polytype, params_or_Wx=None):
    """
    AB = uq_poly_rec_coeffs(n_max, polytype, params_or_Wx)

    Retourne les coefficients de récurrence pour les polynômes classiques
    de Wiener-Askey orthogonaux par rapport à une PDF.

    Référence : Gautschi, W. (2004). Orthogonal polynomials: computation and
    approximation.

    Paramètres
    ----------
    n_max : int
        Index maximum pour les coefficients de récurrence a_n et b_n.
    polytype : str
        'legendre', 'hermite', 'laguerre', 'jacobi', 'fourier', 'zero'.
    params_or_Wx : array-like ou dict, optionnel
        Paramètres pour Laguerre/Jacobi, ou dict {'pdf', 'invcdf', 'bounds'}
        pour polynômes arbitraires.

    Retourne
    --------
    AB : list [matrice (n_max+1)×2, [borne_inf, borne_sup]]
        AB[0] : colonnes = [a_n, sqrt_b_n] pour n = 0..n_max
        AB[1] : bornes du domaine
    """
    if polytype.lower() in ('jacobi', 'laguerre'):
        if params_or_Wx is None:
            raise ValueError(
                f'{polytype} polynomials are parametrically defined! '
                'Please provide them as an input argument.'
            )
        parms = params_or_Wx

    # vecteur n = 0, 1, ..., n_max
    n = np.arange(0, n_max + 1, dtype=float)

    pt = polytype.lower()

    if pt == 'hermite':
        an = np.zeros(n_max + 1)
        sqrt_b0 = 1.0
        sqrt_bn = np.sqrt(n[1:])          # sqrt_bn(k) = sqrt(k) pour k=1..n_max
        bounds = [-np.inf, np.inf]

    elif pt == 'legendre':
        an = np.zeros(n_max + 1)
        sqrt_b0 = 1.0
        # MATLAB : sqrt_bn = @(n) sqrt(1./(4-n.^-2))
        sqrt_bn = np.sqrt(1.0 / (4.0 - n[1:] ** (-2)))
        bounds = [-1.0, 1.0]

    elif pt == 'laguerre':
        an = 2.0 * n + parms[1]
        sqrt_b0 = 1.0
        sqrt_bn = -np.sqrt(n[1:] * (n[1:] + parms[1] - 1.0))
        bounds = [0.0, np.inf]

    elif pt == 'jacobi':
        a = parms[1] - 1.0
        b = parms[0] - 1.0
        bpa = a + b
        bma = b - a
        bpa_bma = bpa * bma
        if (a + b) == 0:
            an = (
                (n == 0).astype(float) * bma / (bpa + 2.0)
                + (n != 0).astype(float) * bpa_bma / ((2.0*n + bpa) * (2.0*n + bpa + 2.0))
                + 1.0
            ) * 0.5
        else:
            an = (bpa_bma / ((2.0*n + bpa) * (2.0*n + bpa + 2.0)) + 1.0) * 0.5
        sqrt_b0 = 1.0
        nn = n[1:]
        if (a + b + 1.0) == 0:
            sqrt_bn = np.sqrt(
                4.0 * nn * (nn+a) * (nn+b) * (
                    (nn == 1).astype(float) / ((2.0*nn+bpa)**2 * (2.0*nn+bpa+1.0))
                    + (nn != 1).astype(float) * (nn+a+b)
                      / ((2.0*nn+bpa)**2 * (2.0*nn+bpa+1.0) * (2.0*nn+bpa-1.0))
                )
            ) * 0.5
        else:
            sqrt_bn = np.sqrt(
                4.0 * nn * (nn+a) * (nn+b) * (nn+bpa)
                / ((2.0*nn+bpa)**2 * (2.0*nn+bpa+1.0) * (2.0*nn+bpa-1.0))
            ) * 0.5
        bounds = [0.0, 1.0]

    elif pt == 'fourier':
        sqrt_b0 = 1.0
        an = n.copy()
        sqrt_bn = n[1:].copy()
        bounds = [0.0, 1.0]

    elif pt == 'zero':
        sqrt_b0 = 0.0
        an = np.zeros(n_max + 1)
        sqrt_bn = np.zeros(n_max)
        bounds = [-np.inf, np.inf]

    else:
        raise ValueError('Unknown polynomial type!')

    # Assemblage : AB[0] = matrice (n_max+1)×2
    # colonne 0 : a_0, a_1, ..., a_{n_max}
    # colonne 1 : sqrt_b0, sqrt_b1, ..., sqrt_b_{n_max}
    b_col = np.concatenate([[sqrt_b0], sqrt_bn])
    coeffs = np.column_stack([an, b_col])   # shape (n_max+1, 2)

    return [coeffs, bounds]


# =============================================================================
# 2. uq_eval_rec_rule
# =============================================================================

def uq_eval_rec_rule(X, AB, nonrecursive=False):
    """
    Y = uq_eval_rec_rule(X, AB, nonrecursive)

    Évalue les polynômes correspondant à la matrice de Jacobi définie par AB.
    Si nonrecursive=True, retourne seulement le polynôme de plus haut degré.

    Paramètres
    ----------
    X : np.ndarray, shape (N,) ou (N,1)
        Points d'évaluation (vecteur colonne).
    AB : np.ndarray, shape (n_max+1, 2)
        Coefficients de récurrence. Colonne 0 = a_n, colonne 1 = b_n.
    nonrecursive : bool, optionnel
        Si True, retourne seulement la dernière colonne (degré le plus élevé).

    Retourne
    --------
    Y : np.ndarray, shape (N, n_max+1)  ou  (N, 1) si nonrecursive=True
    """
    X = np.asarray(X, dtype=float).reshape(-1, 1)   # force colonne (N,1)

    N = X.shape[0]
    # size(AB,1) en MATLAB = AB.shape[0] en Python
    P = np.zeros((N, AB.shape[0]))
    Pminus1 = np.zeros((N, 1))

    # MATLAB : P(:,1) = 1/AB(1,2)
    # Python  : P[:,0] = 1/AB[0,1]   (décalage +1 sur les deux indices)
    P[:, 0] = 1.0 / AB[0, 1]

    # MATLAB : for k=1:(size(AB,1)-1)
    # Python  : for k in range(1, AB.shape[0])
    for k in range(1, AB.shape[0]):
        # MATLAB AB(k,1) → Python AB[k-1,0]
        # MATLAB AB(k,2) → Python AB[k-1,1]
        # MATLAB AB(k+1,2) → Python AB[k,1]
        # MATLAB P(:,k)   → Python P[:,k-1]
        # MATLAB P(:,k+1) → Python P[:,k]
        # MATLAB P(:,k-1) → Python P[:,k-2]
        if k > 1:
            P[:, k] = (
                (X[:, 0] - AB[k-1, 0]) * P[:, k-1]
                - P[:, k-2] * AB[k-1, 1]
            ) / AB[k, 1]
        else:
            # k=1 : P_{-1} = Pminus1 = 0
            P[:, k] = (
                (X[:, 0] - AB[k-1, 0]) * P[:, k-1]
                - Pminus1[:, 0] * AB[k-1, 1]
            ) / AB[k, 1]

    if nonrecursive:
        return P[:, -1:]   # seulement le dernier degré

    return P   # shape (N, n_max+1)


# =============================================================================
# 3. uq_eval_legendre
# =============================================================================

def uq_eval_legendre(ORDER, X, nonrecursive=False):
    """
    VALUE = uq_eval_legendre(ORDER, X, nonrecursive)

    Évalue les polynômes de Legendre orthonormaux univariés jusqu'à l'ordre
    ORDER aux points X. Retourne un tableau N×(ORDER+1) : la colonne j
    contient le polynôme de degré j-1 évalué en X.

    X doit être dans l'intervalle (-1, 1).
    """
    if not np.isscalar(ORDER) or ORDER < 0:
        raise ValueError('uq_eval_legendre only operates on positive order polynomials')

    X = np.asarray(X, dtype=float)
    if X.ndim != 1 and not (X.ndim == 2 and X.shape[1] == 1):
        raise ValueError('uq_eval_legendre is designed to work with X in column vector format')
    X = X.reshape(-1)

    if X.max() > 1 or X.min() < -1:
        raise ValueError('uq_eval_legendre only operates in the (-1, 1) interval')

    AB = uq_poly_rec_coeffs(ORDER, 'legendre')
    VALUE = uq_eval_rec_rule(X, AB[0], nonrecursive)
    return VALUE


# =============================================================================
# 4. uq_eval_hermite
# =============================================================================

def uq_eval_hermite(ORDER, X, nonrecursive=False):
    """
    VALUE = uq_eval_hermite(ORDER, X, nonrecursive)

    Évalue les polynômes de Hermite orthonormaux univariés jusqu'à l'ordre
    ORDER aux points X. Retourne un tableau N×(ORDER+1) : la colonne j
    contient le polynôme de degré j-1 évalué en X.
    """
    if not np.isscalar(ORDER) or ORDER < 0:
        raise ValueError('uq_eval_hermite only operates on positive order polynomials')

    X = np.asarray(X, dtype=float)
    if X.ndim != 1 and not (X.ndim == 2 and X.shape[1] == 1):
        raise ValueError('uq_eval_hermite is designed to work with X in column vector format')
    X = X.reshape(-1)

    AB = uq_poly_rec_coeffs(ORDER, 'hermite')
    VALUE = uq_eval_rec_rule(X, AB[0], nonrecursive)
    return VALUE


# =============================================================================
# 5. uq_PCK_eval_unipoly
# =============================================================================

def uq_PCK_eval_unipoly(U, polyindices, PolyTypes):
    """
    univ_p_val = uq_PCK_eval_unipoly(U, polyindices, PolyTypes)

    Évalue les polynômes univariés pour chaque dimension.
    Permet d'avoir des types de polynômes différents par dimension.

    Paramètres
    ----------
    U : np.ndarray, shape (N, M)
        Points d'évaluation dans l'espace canonique.
    polyindices : np.ndarray, shape (P, M)
        Multi-indices de la base polynomiale.
    PolyTypes : list of str, longueur M
        Type de polynôme par dimension : 'legendre' ou 'hermite'.

    Retourne
    --------
    univ_p_val : np.ndarray, shape (N, M, P+1)
        univ_p_val[:, i, k] = polynôme de degré k pour la dimension i.
    """
    N, M = U.shape

    # MATLAB : P = full(max(sum(polyindices, 2)))
    # full() densifie si polyindices est sparse
    polyindices = np.asarray(polyindices)
    # degré maximal = max de la somme des degrés sur tous les multi-indices
    P = int(np.max(polyindices.sum(axis=1)))

    univ_p_val = np.zeros((N, M, P + 1))

    for i in range(M):
        pt = PolyTypes[i].lower()
        if pt == 'legendre':
            # uq_eval_legendre(P, U(:,i)) → shape (N, P+1)
            univ_p_val[:, i, :] = uq_eval_legendre(P, U[:, i])
        elif pt == 'hermite':
            # uq_eval_hermite(P, U(:,i)) → shape (N, P+1)
            univ_p_val[:, i, :] = uq_eval_hermite(P, U[:, i])
        # Laguerre, Jacobi, Fourier : non implémentés (commentés dans UQLab)

    return univ_p_val


# =============================================================================
# 6. uq_PCE_create_Psi
# =============================================================================

def uq_PCE_create_Psi(Indices, univ_p_val):
    """
    Psi = uq_PCE_create_Psi(Indices, univ_p_val)

    Assemble la matrice de design Ψ à partir du jeu d'indices de la base
    et des évaluations des polynômes univariés.

    Ψ_j(x) = ∏ᵢ φ_{αⱼᵢ}(xᵢ)   avec αⱼᵢ = Indices[j, i]

    Paramètres
    ----------
    Indices : np.ndarray, shape (P, M)
        Multi-indices de la base polynomiale.
    univ_p_val : np.ndarray, shape (N, M, deg_max+1)
        Évaluations univariées — sortie de uq_PCK_eval_unipoly.

    Retourne
    --------
    Psi : np.ndarray, shape (N, P)
    """
    # MATLAB : Indices peut être sparse → densifier
    Indices = np.asarray(Indices)

    M = univ_p_val.shape[1]   # nombre de variables d'entrée
    N = univ_p_val.shape[0]   # taille du design expérimental
    P = Indices.shape[0]       # nombre d'éléments de la base

    if M != Indices.shape[1]:
        raise ValueError("Error: Index and univ_p_val don't seem to have consistent sizes!!")

    # Initialisation à 1 : élément neutre du produit tensoriel
    Psi = np.ones((N, P))

    # MATLAB :
    # for mm = 1:size(Indices,2)
    #     aa = Indices(:,mm) > 0
    #     try
    #       Psi(:,aa) = Psi(:,aa) .* reshape(univ_p_val(:,mm, Indices(aa,mm)+1), size(Psi(:,aa)));
    #     catch me
    #       warning(me.message);
    #     end
    # end
    for mm in range(M):
        aa = Indices[:, mm] > 0          # booléen sur les P colonnes de Psi

        # MATLAB Indices(aa,mm)+1 → Python Indices[aa,mm] (MATLAB 1-indexé sur la 3e dim)
        # univ_p_val(:, mm, Indices(aa,mm)+1) en MATLAB = degré Indices[aa,mm] en Python
        # car Python stocke le degré k à l'index k (0-indexé), MATLAB à l'index k+1
        deg_aa = Indices[aa, mm].astype(int)   # degrés pour les termes actifs

        # reshape pour broadcaster correctement : (N, nb_actifs)
        poly_vals = univ_p_val[:, mm, deg_aa]   # shape (N, nb_actifs)

        try:
            Psi[:, aa] = Psi[:, aa] * poly_vals
        except Exception as me:
            import warnings
            warnings.warn(str(me))

    return Psi


# =============================================================================
# 7. uq_GeneralIsopTransform  (+ helpers)
# =============================================================================

# --- Helpers -----------------------------------------------------------------

def _uq_find_nonconstant_marginals(Marginals):
    """Retourne les indices (0-based) des marginales non constantes."""
    nonconstant = []
    for i, m in enumerate(Marginals):
        if m.get('Type', '').lower() != 'constant':
            nonconstant.append(i)
    return np.array(nonconstant, dtype=int)


def _uq_isIndependenceCopula(Copula):
    """Vérifie si la copule est une copule d'indépendance."""
    return Copula.get('Type', '').lower() == 'independent'


def _uq_IndepCopula(M):
    """Crée une copule d'indépendance de dimension M."""
    return {
        'Type': 'Independent',
        'Parameters': np.eye(M),
        'Variables': list(range(M)),
    }


def _uq_IsopTransform(X, X_Marginals, Y_Marginals):
    """
    Transforme X marginale par marginale (cas copule indépendante) :
    X → Uniform[0,1] via CDF source → Y via CDF inverse cible.
    Supporte : Uniform, Gaussian/Normal, Lognormal, Gumbel, Exponential.
    """
    N, M = X.shape
    Y = np.empty_like(X)

    for i in range(M):
        mx = X_Marginals[i]
        my = Y_Marginals[i]

        # CDF de la source → valeurs uniformes [0,1]
        u = _marginal_cdf(X[:, i], mx)

        # CDF inverse de la cible → valeurs dans l'espace cible
        Y[:, i] = _marginal_icdf(u, my)

    return Y


def _marginal_cdf(x, marginal):
    """CDF d'une marginale UQLab."""
    t = marginal.get('Type', '').lower()
    p = marginal.get('Parameters', [])

    if t == 'uniform':
        a, b = p[0], p[1]
        return np.clip((x - a) / (b - a), 0.0, 1.0)
    elif t in ('gaussian', 'normal'):
        mu, sigma = p[0], p[1]
        return stats.norm.cdf(x, loc=mu, scale=sigma)
    elif t == 'lognormal':
        mu, sigma = p[0], p[1]
        return stats.lognorm.cdf(x, s=sigma, scale=np.exp(mu))
    elif t == 'gumbel':
        mu, beta = p[0], p[1]
        return stats.gumbel_r.cdf(x, loc=mu, scale=beta)
    elif t == 'exponential':
        lam = p[0]
        return stats.expon.cdf(x, scale=1.0/lam)
    else:
        raise NotImplementedError(f'CDF not implemented for marginal type: {t}')


def _marginal_icdf(u, marginal):
    """CDF inverse d'une marginale UQLab."""
    t = marginal.get('Type', '').lower()
    p = marginal.get('Parameters', [])

    if t == 'uniform':
        a, b = p[0], p[1]
        return a + (b - a) * u
    elif t in ('gaussian', 'normal'):
        mu, sigma = p[0], p[1]
        return stats.norm.ppf(u, loc=mu, scale=sigma)
    elif t == 'lognormal':
        mu, sigma = p[0], p[1]
        return stats.lognorm.ppf(u, s=sigma, scale=np.exp(mu))
    elif t == 'gumbel':
        mu, beta = p[0], p[1]
        return stats.gumbel_r.ppf(u, loc=mu, scale=beta)
    elif t == 'exponential':
        lam = p[0]
        return stats.expon.ppf(u, scale=1.0/lam)
    else:
        raise NotImplementedError(f'ICDF not implemented for marginal type: {t}')


def _uq_NatafTransform(X, Marginals, Copula):
    """Transforme X vers l'espace Gaussien standard via Nataf."""
    raise NotImplementedError('uq_NatafTransform: non implémenté (copule Gaussienne non-indépendante)')


def _uq_invNatafTransform(Xstd, Marginals, Copula):
    """Transforme depuis l'espace Gaussien standard vers l'espace cible via Nataf inverse."""
    raise NotImplementedError('uq_invNatafTransform: non implémenté (copule Gaussienne non-indépendante)')


def _uq_RosenblattTransform(X, Marginals, Copula):
    """Transforme X vers Uniform[0,1]^M via la transformation de Rosenblatt."""
    raise NotImplementedError('uq_RosenblattTransform: non implémenté (cas copule non-indépendante)')


def _uq_invRosenblattTransform(U, Marginals, Copula):
    """Transforme depuis Uniform[0,1]^M vers l'espace cible via Rosenblatt inverse."""
    raise NotImplementedError('uq_invRosenblattTransform: non implémenté (cas copule non-indépendante)')


def _uq_BlockGeneralIsopTransform(X, X_Marginals, X_Copula, Y_Marginals, Y_Copula):
    """
    Transforme un bloc de variables de façon efficace.
    Retourne (Y, success).
    """
    # si copule indépendante des deux côtés : transform standard marginale par marginale
    if _uq_isIndependenceCopula(X_Copula) and _uq_isIndependenceCopula(Y_Copula):
        Y = _uq_IsopTransform(X, X_Marginals, Y_Marginals)
        success = True
        return Y, success

    # fonction locale pour vérifier le type de copule
    def checkCop(cop, typ):
        return (cop.get('Type', '').lower() == typ.lower()) or \
               (cop.get('Type', '').lower() == 'pair' and
                cop.get('Family', '').lower() == typ.lower())

    # fonction locale pour vérifier les marginales
    def checkMarg(marg, typ):
        types = [m.get('Type', '').lower() for m in marg]
        return all(t == typ.lower() for t in types) and \
               not any('Bounds' in m for m in marg)

    # cas Gaussien + marginales Gaussiennes : Nataf
    if (checkCop(X_Copula, 'gaussian') and checkMarg(X_Marginals, 'gaussian') and
            checkCop(Y_Copula, 'gaussian') and checkMarg(Y_Marginals, 'gaussian')):
        Xstd = _uq_NatafTransform(X, X_Marginals, X_Copula)
        Y = _uq_invNatafTransform(Xstd, Y_Marginals, Y_Copula)
        success = True
        return Y, success

    # cas Student + marginales Student : Nataf (succès=False dans UQLab)
    elif (checkCop(X_Copula, 'student') and checkMarg(X_Marginals, 'student') and
          checkCop(Y_Copula, 'student') and checkMarg(Y_Marginals, 'student')):
        Xstd = _uq_NatafTransform(X, X_Marginals, X_Copula)
        Y = _uq_invNatafTransform(Xstd, Y_Marginals, Y_Copula)
        success = False
        return Y, success

    else:
        Y = np.full(X.shape, np.nan)
        success = False
        return Y, success


# --- Fonction principale -----------------------------------------------------

def uq_GeneralIsopTransform(X, X_Marginals, X_Copula, Y_Marginals, Y_Copula):
    """
    Y = uq_GeneralIsopTransform(X, X_Marginals, X_Copula, Y_Marginals, Y_Copula)

    Transforme un ensemble X d'échantillons d'un vecteur aléatoire de distribution
    arbitraire vers un échantillon Y dans un autre espace probabiliste.
    Les deux espaces sont définis via le formalisme Marginals / Copula d'UQLab.

    Paramètres
    ----------
    X : np.ndarray, shape (N, M)
    X_Marginals : list of dict, longueur M
        Chaque dict : {'Type': str, 'Parameters': list, ...}
    X_Copula : dict ou list of dict
        {'Type': str, 'Variables': list, ...}
    Y_Marginals : list of dict, longueur M
    Y_Copula : dict ou list of dict
    """
    if np.any(np.isnan(X)):
        raise ValueError('Requested Generalized probabilistic transformation for array X containing nans.')

    n, M = X.shape
    if M != len(X_Marginals):
        raise ValueError('The input dimension is inconsistent with the data.')
    if M != len(Y_Marginals):
        raise ValueError('The input dimension do not match the output dimension.')

    # Indices des variables non-constantes
    idNonConst_X = _uq_find_nonconstant_marginals(X_Marginals)
    idNonConst_Y = _uq_find_nonconstant_marginals(Y_Marginals)
    idConst_X = np.setdiff1d(np.arange(M), idNonConst_X)
    idConst_Y = np.setdiff1d(np.arange(M), idNonConst_Y)

    # Erreur si une constante X doit être transformée en non-constante Y
    idConst_XnotY = np.setdiff1d(idConst_X, idConst_Y)
    if len(idConst_XnotY) > 0:
        raise ValueError(
            f'Constant marginal X_i cannot be mapped to non-constant marginal Y_i, i={idConst_XnotY}'
        )

    # Initialise Y avec NaN
    Y = np.full((n, M), np.nan)

    # Assigne les constantes directement
    for ii in idConst_Y:
        Y[:, ii] = Y_Marginals[ii]['Parameters'][0]

    # Normalise les copules en liste si nécessaire
    if isinstance(X_Copula, dict):
        X_Copula_list = [X_Copula]
    else:
        X_Copula_list = X_Copula

    if isinstance(Y_Copula, dict):
        Y_Copula_list = [Y_Copula]
    else:
        Y_Copula_list = Y_Copula

    # Assigne le champ Variables si absent
    if len(X_Copula_list) == 1 and 'Variables' not in X_Copula_list[0]:
        X_Copula_list[0] = dict(X_Copula_list[0])
        X_Copula_list[0]['Variables'] = list(range(M))
    if len(Y_Copula_list) == 1 and 'Variables' not in Y_Copula_list[0]:
        Y_Copula_list[0] = dict(Y_Copula_list[0])
        Y_Copula_list[0]['Variables'] = list(range(M))

    success = True
    icop = len(Y_Copula_list)   # index à partir duquel lancer Rosenblatt

    for cc, CopX in enumerate(X_Copula_list):
        CopX = dict(CopX)
        VarsX = list(CopX.get('Variables', range(M)))

        # Retire les constantes de VarsX
        const_in_VarsX = [v for v in VarsX if v in idConst_X]
        if const_in_VarsX:
            if _uq_isIndependenceCopula(CopX):
                VarsX = [v for v in VarsX if v not in idConst_X]
                CopX = _uq_IndepCopula(len(VarsX))
            else:
                raise ValueError(
                    f'X variables {const_in_VarsX} are constant but coupled by {CopX["Type"]} copula'
                )

        # Essai de la transformation bloc à bloc
        if success and cc < len(Y_Copula_list):
            CopY = dict(Y_Copula_list[cc])
            VarsY = list(CopY.get('Variables', range(M)))

            const_in_VarsY = [v for v in VarsY if v in idConst_Y]
            if const_in_VarsY:
                if _uq_isIndependenceCopula(CopY):
                    VarsY = [v for v in VarsY if v not in idConst_Y]
                    CopY = _uq_IndepCopula(len(VarsY))
                else:
                    raise ValueError(
                        f'Constant variables Y_{VarsY} cannot be coupled by {CopY["Type"]} copula'
                    )

            if len(VarsX) == len(VarsY):
                same_order = (VarsX == VarsY)
                both_indep = (
                    _uq_isIndependenceCopula(CopX)
                    and _uq_isIndependenceCopula(CopY)
                    and set(VarsX) == set(VarsY)
                )
                if same_order or both_indep:
                    if VarsX:
                        CopX_local = _uq_IndepCopula(len(VarsX))
                        CopY_local = _uq_IndepCopula(len(VarsY))
                        X_block = X[:, VarsX]
                        Y_block, success = _uq_BlockGeneralIsopTransform(
                            X_block,
                            [X_Marginals[v] for v in VarsX], CopX_local,
                            [Y_Marginals[v] for v in VarsX], CopY_local,
                        )
                        Y[:, VarsX] = Y_block
                else:
                    success = False
            else:
                success = False

            if not success:
                icop = cc

        # Si la transformation bloc échoue : Rosenblatt
        if not success and VarsX:
            CopX_local = _uq_IndepCopula(len(VarsX))
            Y[:, VarsX] = _uq_RosenblattTransform(
                X[:, VarsX],
                [X_Marginals[v] for v in VarsX],
                CopX_local,
            )

    # Rosenblatt inverse pour les variables Y non traitées
    for cc in range(icop, len(Y_Copula_list)):
        CopY = dict(Y_Copula_list[cc])
        VarsY = list(CopY.get('Variables', range(M)))

        const_in_VarsY = [v for v in VarsY if v in idConst_Y]
        if const_in_VarsY:
            if _uq_isIndependenceCopula(CopY):
                VarsY = [v for v in VarsY if v not in idConst_Y]
                CopY = _uq_IndepCopula(len(VarsY))
            else:
                raise ValueError(
                    f'Constant variables Y_{VarsY} cannot be coupled by {CopY["Type"]} copula'
                )
        if VarsY:
            CopY_local = _uq_IndepCopula(len(VarsY))
            Y[:, VarsY] = _uq_invRosenblattTransform(
                Y[:, VarsY],
                [Y_Marginals[v] for v in VarsY],
                CopY_local,
            )

    if np.any(np.isnan(Y)):
        raise ValueError('Generalized Isoprobabilistic Transform Y of X contains nans')

    return Y


# =============================================================================
# 8. uq_assemble_Kernel
# =============================================================================

def uq_assemble_Kernel(h, K_family, K_type):
    """
    K = uq_assemble_Kernel(h, K_family, K_type)

    Calcule les valeurs du noyau à partir des distances normalisées h.

    Paramètres
    ----------
    h : np.ndarray
        Distances normalisées.
        - Séparable  : shape (n_pairs, M) — une colonne par dimension
        - Ellipsoïdal : shape (n_pairs,)  — distance scalaire
    K_family : str
        'nugget', 'linear', 'exponential', 'gaussian', 'matern-5_2', 'matern-3_2'
    K_type : str
        'separable' ou 'ellipsoidal'

    Retourne
    --------
    K : np.ndarray, shape (n_pairs,)
    """
    kt = K_type.lower()
    kf = K_family.lower()

    if kt == 'separable':
        if kf == 'nugget':
            K = np.prod(h, axis=1)
        elif kf == 'linear':
            K = np.prod(np.maximum(0.0, 1.0 - h), axis=1)
        elif kf == 'exponential':
            K = np.prod(np.exp(-h), axis=1)
        elif kf == 'gaussian':
            K = np.prod(np.exp(-0.5 * h**2), axis=1)
        elif kf == 'matern-5_2':
            K = np.prod(
                (1.0 + np.sqrt(5.0)*h + 5.0/3.0 * h**2) * np.exp(-np.sqrt(5.0)*h),
                axis=1
            )
        elif kf == 'matern-3_2':
            K = np.prod(
                (1.0 + np.sqrt(3.0)*h) * np.exp(-np.sqrt(3.0)*h),
                axis=1
            )
        else:
            raise ValueError('Error: Unknown kernel/correlation function family!')

    elif kt == 'ellipsoidal':
        h = np.abs(h)
        if kf == 'linear':
            K = np.maximum(0.0, 1.0 - h)
        elif kf == 'exponential':
            K = np.exp(-h)
        elif kf == 'gaussian':
            K = np.exp(-0.5 * h**2)
        elif kf == 'matern-5_2':
            K = (1.0 + np.sqrt(5.0)*h + 5.0/3.0 * h**2) * np.exp(-np.sqrt(5.0)*h)
        elif kf == 'matern-3_2':
            K = (1.0 + np.sqrt(3.0)*h) * np.exp(-np.sqrt(3.0)*h)
        else:
            raise ValueError('Error: Unknown correlation function family!')
    else:
        raise ValueError(f'Unknown kernel type: {K_type}')

    return K


# =============================================================================
# 9. uq_eval_Kernel
# =============================================================================

def uq_eval_Kernel(X1, X2, theta, options):
    """
    K = uq_eval_Kernel(X1, X2, theta, options)

    Calcule la matrice de noyau/corrélation N1×N2 entre X1 (N1×M) et X2 (N2×M).

    Paramètres
    ----------
    X1 : np.ndarray, shape (N1, M)
    X2 : np.ndarray, shape (N2, M)
    theta : np.ndarray, shape (M,) ou scalaire
        Paramètres du noyau (longueurs de corrélation).
    options : dict avec les champs :
        'Family'    : str ou callable — famille du noyau
        'Type'      : 'separable' ou 'ellipsoidal'
        'Isotropic' : bool
        'Nugget'    : float ou np.ndarray

    Retourne
    --------
    K : np.ndarray, shape (N1, N2)
    """
    K_type      = options['Type']
    K_family    = options['Family']
    K_isIsotropic = options['Isotropic']
    nugget      = options.get('Nugget', 0)

    N1 = X1.shape[0]
    N2 = X2.shape[0]
    M  = X1.shape[1]

    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 0:
        theta = theta.reshape(1)

    # Vérifications de cohérence
    if K_isIsotropic and theta.size > 1 \
            and not callable(K_family) \
            and K_family.lower() not in ('polynomial', 'sigmoid'):
        raise ValueError(
            'Error: For isotropic kernel/correlation function, '
            'the length scale parameter (theta) is expected to be scalar!'
        )
    if isinstance(K_family, str) and K_family.lower() == 'polynomial' and theta.size != 2:
        raise ValueError('Error: The polynomial kernel must have two parameters')
    if isinstance(K_family, str) and K_family.lower() == 'sigmoid' and theta.size != 2:
        raise ValueError('Error: The sigmoid kernel must have two parameters')
    if X1.shape[1] != X2.shape[1]:
        raise ValueError('Error: Xi s in K(X1,X2) must have the same number of dimensions!')
    if not K_isIsotropic and M != theta.size \
            and not (callable(K_family) or
                     (isinstance(K_family, str) and K_family.lower() in ('polynomial', 'sigmoid'))):
        raise ValueError(
            'Error: For anisotropic kernel/correlation function '
            'theta vector must have length equal to the number of marginals!'
        )

    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)   # colonne pour broadcast

    # Déterminer si c'est une matrice de Gram (X1 == X2)
    isGram = (N1 == N2) and np.array_equal(X1, X2)

    # Familles stationnaires
    stationary_families = {'nugget', 'linear', 'exponential', 'gaussian', 'matern-5_2', 'matern-3_2'}
    isstationary = (callable(K_family) or
                    (isinstance(K_family, str) and K_family.lower() in stationary_families))

    # MATLAB : [idx2,idx1] = meshgrid(uint32(1:N2),uint32(1:N1))
    # idx1(i,j) = i  (varie par lignes, 0..N1-1)
    # idx2(i,j) = j  (varie par colonnes, 0..N2-1)
    idx1, idx2 = np.meshgrid(np.arange(N1, dtype=np.int32),
                              np.arange(N2, dtype=np.int32),
                              indexing='ij')   # shape (N1, N2)

    if isstationary:
        if isGram:
            # Triangle strictement inférieur uniquement (idx1 > idx2)
            zidx = idx1 > idx2
        else:
            zidx = idx1 > -1   # tous les éléments (True partout)

        kt = K_type.lower()

        if kt == 'separable':
            if callable(K_family):
                K_flat = np.zeros(N1 * N2)
                K_flat[zidx.ravel()] = 1.0
                if K_isIsotropic:
                    for jj in range(M):
                        K_flat[zidx.ravel()] *= K_family(
                            X1[idx1[zidx], jj], X2[idx2[zidx], jj], float(theta[0])
                        )
                else:
                    for jj in range(M):
                        K_flat[zidx.ravel()] *= K_family(
                            X1[idx1[zidx], jj], X2[idx2[zidx], jj], float(theta[jj])
                        )
            else:
                kf = K_family.lower()
                if kf == 'nugget':
                    h = (X1[idx1[zidx], :] == X2[idx2[zidx], :]).astype(float)
                else:
                    # MATLAB : abs(bsxfun(@rdivide, X1(idx1,...) - X2(idx2,...), theta'))
                    h = np.abs(
                        (X1[idx1[zidx], :] - X2[idx2[zidx], :]) / theta.ravel()
                    )

                K_flat = np.zeros(N1 * N2)
                if isGram:
                    K_flat[zidx.ravel()] = uq_assemble_Kernel(h, K_family, K_type)
                else:
                    K_flat = uq_assemble_Kernel(h, K_family, K_type)

        elif kt == 'ellipsoidal':
            theta_row = theta.ravel()
            if K_isIsotropic:
                theta_row = np.full(M, float(theta_row[0]))

            # pdist2 avec distance euclidienne standardisée
            h = cdist(X1, X2, metric='seuclidean', V=theta_row**2)

            if callable(K_family):
                K_flat = np.zeros(N1 * N2)
                K_flat[zidx.ravel()] = K_family(h[zidx])
            else:
                if isGram:
                    K_flat = np.zeros(N1 * N2)
                    K_flat[zidx.ravel()] = uq_assemble_Kernel(h[zidx], K_family, K_type)
                else:
                    K_flat = uq_assemble_Kernel(h, K_family, K_type)
        else:
            raise ValueError(f'Unknown type of correlation function: "{K_type}"')

    else:
        # Noyaux non-stationnaires
        if isGram:
            zidx = idx1 >= idx2   # diagonale incluse
        else:
            zidx = idx1 > -1

        K_flat = np.zeros(N1 * N2)
        kf = K_family.lower() if isinstance(K_family, str) else ''

        if kf == 'linear_ns':
            K_flat[zidx.ravel()] = np.sum(
                X1[idx1[zidx], :] * X2[idx2[zidx], :], axis=1
            )
        elif kf == 'polynomial':
            K_flat[zidx.ravel()] = (
                np.sum(X1[idx1[zidx], :] * X2[idx2[zidx], :], axis=1) + theta.ravel()[0]
            ) ** theta.ravel()[1]
        elif kf == 'sigmoid':
            K_flat[zidx.ravel()] = np.tanh(
                np.sum(X1[idx1[zidx], :] * X2[idx2[zidx], :], axis=1) / theta.ravel()[0]
                + theta.ravel()[1]
            )
        else:
            raise ValueError('Error: Unknown correlation function family!')

    # Remet K en forme N1×N2
    K = K_flat.reshape(N1, N2)

    # Si matrice de Gram : compléter par symétrie + diagonale + nugget
    if isGram:
        if isstationary:
            K = K + K.T + np.eye(N1)
        else:
            # Diagonale déjà calculée → évite le double comptage
            K = K + K.T - np.diag(np.diag(K))

        # Nugget
        nugget = np.asarray(nugget)
        if nugget.ndim == 0 and float(nugget) != 0:
            K = K + np.eye(N1) * float(nugget)
        elif nugget.ndim > 0 and nugget.size > 1:
            K = K + np.diag(nugget.ravel())

    return K
