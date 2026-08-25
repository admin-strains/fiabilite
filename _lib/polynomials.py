"""
Polynomes orthogonaux et base PCE -- extrait de branche5.py.

Recurrences a trois termes, Hermite et Legendre, evaluation d'un
polynome unidimensionnel et construction de la matrice de design Psi.

PHASE 3 du plan de nettoyage : `branche5.py` portait TROIS sujets sans
rapport entre eux. L'analyse du graphe d'appels
(`tools/analyse_dependances.py`) a montre **zero arete** entre les trois --
c'etaient trois modules colles dans un fichier.

Le corps des fonctions est repris VERBATIM : la scission ne doit changer
aucun resultat, et la baseline le verifie.
"""

import numpy as np


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
# 3b. uq_eval_legendre_deriv
# =============================================================================

def uq_eval_legendre_deriv(ORDER, X):
    """
    VALUE = uq_eval_legendre_deriv(ORDER, X)

    Dérivée des polynômes de Legendre orthonormaux L_k.
    Retourne un tableau N×(ORDER+1) : colonne k = L'_k(x).

    Formule :
        L_n = c_n * P_n   avec c_n = sqrt((2n+1)/2)  (P_n = Legendre standard)
        P'_0 = 0,  P'_1 = 1,  P'_n = (2n-1)*P_{n-1} + P'_{n-2}  pour n >= 2
        L'_n = c_n * P'_n
    """
    if not np.isscalar(ORDER) or ORDER < 0:
        raise ValueError('uq_eval_legendre_deriv only operates on positive order polynomials')

    X = np.asarray(X, dtype=float)
    if X.ndim != 1 and not (X.ndim == 2 and X.shape[1] == 1):
        raise ValueError('uq_eval_legendre_deriv is designed to work with X in column vector format')
    X = X.reshape(-1)
    N = X.shape[0]

    out = np.zeros((N, ORDER + 1))
    if ORDER == 0:
        return out   # L'_0 = 0

    # Legendre orthonormaux L_n, puis standard P_n = L_n / c_n
    L  = uq_eval_legendre(ORDER, X)                       # (N, ORDER+1)
    ns = np.arange(ORDER + 1, dtype=float)
    c  = np.sqrt(2 * ns + 1)                              # c_n : L_n = c_n * P_n
    P  = L / c[np.newaxis, :]                             # (N, ORDER+1)

    # Récurrence sur les dérivées standard
    dP = np.zeros((N, ORDER + 1))
    dP[:, 1] = 1.0                                        # P'_1 = 1
    for n in range(2, ORDER + 1):
        dP[:, n] = (2 * n - 1) * P[:, n - 1] + dP[:, n - 2]

    # Retour vers orthonormaux
    out = c[np.newaxis, :] * dP                           # L'_n = c_n * P'_n
    return out


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
# 4b. uq_eval_hermite_deriv
# =============================================================================

def uq_eval_hermite_deriv(ORDER, X):
    """
    VALUE = uq_eval_hermite_deriv(ORDER, X)

    Dérivée des polynômes de Hermite orthonormaux H_k.
    Retourne un tableau N×(ORDER+1) : colonne k = H'_k(x).

    Formule : H'_0 = 0,  H'_k = sqrt(k) * H_{k-1}  pour k >= 1
    """
    if not np.isscalar(ORDER) or ORDER < 0:
        raise ValueError('uq_eval_hermite_deriv only operates on positive order polynomials')

    X = np.asarray(X, dtype=float)
    if X.ndim != 1 and not (X.ndim == 2 and X.shape[1] == 1):
        raise ValueError('uq_eval_hermite_deriv is designed to work with X in column vector format')
    X = X.reshape(-1)

    out = np.zeros((X.shape[0], ORDER + 1))
    if ORDER >= 1:
        H = uq_eval_hermite(ORDER, X)                     # (N, ORDER+1)
        for k in range(1, ORDER + 1):
            out[:, k] = np.sqrt(k) * H[:, k - 1]
    return out


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
