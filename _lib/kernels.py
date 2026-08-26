"""
Noyaux de correlation et leurs derivees -- extrait de branche5.py.

Noyaux separables/ellipsoidaux, derivees premieres et secondes, et les
noyaux augmentes du krigeage gradient-enhanced (GEPCK).

PHASE 3 du plan de nettoyage : `branche5.py` portait TROIS sujets sans
rapport entre eux. L'analyse du graphe d'appels
(`tools/analyse_dependances.py`) a montre **zero arete** entre les trois --
c'etaient trois modules colles dans un fichier.

Le corps des fonctions est repris VERBATIM : la scission ne doit changer
aucun resultat, et la baseline le verifie.
"""

import numpy as np
from scipy.spatial.distance import cdist


# =============================================================================
# Pepite par defaut -- DEFAUTS 2 et 3 du plan de nettoyage
# =============================================================================
#: Valeur ajoutee a la diagonale de la matrice de correlation avant de la
#: factoriser. Elle valait ZERO, au nom de l'interpolation exacte. La mesure
#: du 26/08/2026 montre que c'est l'inverse qui se produit.
#:
#: POURQUOI UNE PEPITE EST NECESSAIRE ICI
#: Les etats limites de fiabilite sont TRES lisses. Le maximum de
#: vraisemblance pousse donc les longueurs de correlation vers le haut --
#: theta = [59,9 ; 73,1] mesures sur un plan de 24 points dans [-4, 4]^2,
#: bien loin sous le plafond de 100. La correlation vaut alors ~1 entre tous
#: les points : la matrice devient numeriquement de rang 1.
#:
#: Conditionnement mesure sur ce plan : cond(R) = 7,0e13 pour PCK,
#: cond(R_tilde) = 1,7e15 pour GEPCK -- au bord des 4,5e15 que la double
#: precision autorise. Le residu de resolution vaut alors 2,8e-02, et c'est
#: LUI l'erreur d'interpolation. Ni l'equilibrage de Jacobi (aucun gain
#: mesure : 1,64e15 -> 1,63e15), ni le raffinement iteratif, ni `lstsq` n'y
#: changent quoi que ce soit : la matrice est simplement hors de portee.
#:
#: CE QUE CELA COUTAIT EN PRODUCTION
#: L'erreur EMPIRE quand le plan d'experiences grandit -- exactement a
#: l'envers de ce qu'on attend, et la boucle d'enrichissement EFF, elle,
#: ajoute des points :
#:
#:     etat limite   N    pepite 0      pepite 1e-8
#:     flexion      24    1,30 %        0,0072 %
#:     flexion      40    56,4 %        0,0015 %
#:     lineaire     40    466 %         exact
#:
#: Le cas lineaire est le plus parlant : l'etat limite est un hyperplan que
#: le metamodele contient exactement, et sans pepite il rendait beta = 19,8
#: au lieu de 3,5.
#:
#: CHOIX DE LA VALEUR
#: Balayage sur 2 etats limites x 4 tailles de plan x 2 metamodeles. A 1e-8,
#: l'interpolation reste sous 4,7e-08 et l'erreur sur beta sous 0,030 %
#: partout -- soit MIEUX que PCK, ce que GEPCK devrait toujours etre puisqu'il
#: dispose des gradients en plus. 1e-10 ne suffit pas sur les grands plans ;
#: 1e-6 marche aussi mais introduit un biais d'interpolation inutile.
#:
#: La pepite n'est ajoutee qu'a la matrice de Gram, jamais a la
#: cross-correlation `r0` -- `predict.py` force `Nugget = 0.0` de ce cote,
#: comme le veut la formulation.
PEPITE_PAR_DEFAUT = 1e-8


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

    # Déterminer si c'est une matrice de Gram (X1 == X2).
    #
    # Ici l'enjeu n'est pas la FORME -- elle vaut (N1, N2) dans les deux cas --
    # mais le contenu : la branche Gram ne calcule que le triangle inferieur,
    # symetrise, ajoute l'identite et le nugget. Se tromper change donc la
    # matrice, pas sa taille, ce qui est plus discret. Comme pour
    # `uq_eval_global_Kernel`, l'appelant peut le DIRE (defaut 1 du plan).
    isGram = options.get('IsGram')
    if isGram is None:
        isGram = (N1 == N2) and np.array_equal(X1, X2)
    else:
        isGram = bool(isGram)
        if isGram and (N1 != N2 or not np.array_equal(X1, X2)):
            raise ValueError(
                "IsGram=True mais X1 et X2 different : une matrice de Gram "
                "suppose le meme jeu de points des deux cotes (%s contre %s)."
                % (X1.shape, X2.shape))

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


# ===========================================================================
# GEK — fonctions de dérivées du noyau (kernel_deriv_factory + helpers)
# Utilisées pour assembler la matrice de corrélation augmentée R̃ de GEK/GEPCK
# (Zuhal 2021, Eq. 7).  Noyaux supportés : 'gaussian' et 'matern-5_2'.
# ===========================================================================

def _prod_excl(K_uni):
    """
    K_uni : (n1, n2, M)
    Retourne K_excl : (n1, n2, M) avec K_excl[:,:,l] = prod_{m!=l} K_uni[:,:,m].
    Algorithme prefix/suffix O(M) sans division.
    """
    n1, n2, M = K_uni.shape
    pre = np.ones((n1, n2, M + 1))
    suf = np.ones((n1, n2, M + 1))
    for l in range(M):
        pre[:, :, l+1] = pre[:, :, l] * K_uni[:, :, l]
        suf[:, :, M-1-l] = suf[:, :, M-l] * K_uni[:, :, M-1-l]
    return pre[:, :, :M] * suf[:, :, 1:]


def kernel_deriv_factory(family, der, der_prime):
    """
    Retourne f(X1, X2, theta) -> (n1, n2) pour un bloc de R_tilde.

    Parameters
    ----------
    family     : 'gaussian' ou 'matern-5_2'
    der        : int ou None  -- composante du premier argument x (0-indexe)
    der_prime  : int ou None  -- composante du second argument x' (0-indexe)
    """
    if family.lower() == 'gaussian':
        def f(X1, X2, theta):
            X1    = np.asarray(X1, dtype=float)
            X2    = np.asarray(X2, dtype=float)
            theta = np.asarray(theta, dtype=float).ravel()

            delta = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]       # (n1, n2, M)

            opts = {'Handle': uq_eval_Kernel, 'Family': 'gaussian',
                    'Type': 'separable', 'Isotropic': False, 'Nugget': 0.0}
            k = uq_eval_Kernel(X1, X2, theta, opts)                    # (n1, n2)

            if der is None and der_prime is None:
                return k
            elif der is not None and der_prime is None:
                return -delta[:, :, der] / theta[der]**2 * k
            elif der is None and der_prime is not None:
                return +delta[:, :, der_prime] / theta[der_prime]**2 * k
            elif der == der_prime:
                i = der
                return (1 - delta[:, :, i]**2 / theta[i]**2) / theta[i]**2 * k
            else:
                return (-delta[:, :, der] * delta[:, :, der_prime]
                        / (theta[der]**2 * theta[der_prime]**2) * k)

        return f

    elif family.lower() == 'matern-5_2':
        def f(X1, X2, theta):
            X1    = np.asarray(X1, dtype=float)
            X2    = np.asarray(X2, dtype=float)
            theta = np.asarray(theta, dtype=float).ravel()

            delta  = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]      # (n1, n2, M)
            a      = np.sqrt(5) * np.abs(delta) / theta                # (n1, n2, M)

            K_uni  = (1 + a + a**2 / 3) * np.exp(-a)                  # (n1, n2, M)
            K_excl = _prod_excl(K_uni)                                 # (n1, n2, M)

            if der is None and der_prime is None:
                return K_uni.prod(axis=2)
            elif der is not None and der_prime is None:
                i = der
                return (-(5 / (3 * theta[i]**2)) * delta[:, :, i]
                        * (1 + a[:, :, i]) * np.exp(-a[:, :, i]) * K_excl[:, :, i])
            elif der is None and der_prime is not None:
                j = der_prime
                return (+(5 / (3 * theta[j]**2)) * delta[:, :, j]
                        * (1 + a[:, :, j]) * np.exp(-a[:, :, j]) * K_excl[:, :, j])
            elif der == der_prime:
                i = der
                return ((5 / (3 * theta[i]**2))
                        * (1 + a[:, :, i] - a[:, :, i]**2)
                        * np.exp(-a[:, :, i]) * K_excl[:, :, i])
            else:
                i, j = der, der_prime
                K_excl_ij = K_excl[:, :, i] / K_uni[:, :, j]
                return (-(25 / (9 * theta[i]**2 * theta[j]**2))
                        * delta[:, :, i] * delta[:, :, j]
                        * (1 + a[:, :, i]) * (1 + a[:, :, j])
                        * np.exp(-(a[:, :, i] + a[:, :, j])) * K_excl_ij)

        return f

    else:
        raise ValueError(
            f'GEK : noyau "{family}" non supporte. Utiliser "gaussian" ou "matern-5_2".')


def kernel_second_deriv_factory(family, der1, der2):
    """
    Retourne f(X1, X2, theta) -> (n1, n2) calculant
    d²k(X1, X2) / (dX1_{der1} dX1_{der2})
    (derivee seconde par rapport au PREMIER argument uniquement).

    Utilisee dans uq_assemble_deriv_global_Kernel pour les blocs cb >= 1 :
        d/dX1_{der_var} [ dk/dX1_{l-1} ] = d²k/(dX1_{der_var} dX1_{l-1})

    Parameters
    ----------
    family : 'gaussian' ou 'matern-5_2'
    der1   : int  -- 1re composante de derivation (0-indexe)
    der2   : int  -- 2e  composante de derivation (0-indexe)
    """
    if family.lower() == 'gaussian':
        def f(X1, X2, theta):
            X1    = np.asarray(X1, dtype=float)
            X2    = np.asarray(X2, dtype=float)
            theta = np.asarray(theta, dtype=float).ravel()

            delta = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]   # (n1, n2, M)
            opts  = {'Handle': uq_eval_Kernel, 'Family': 'gaussian',
                     'Type': 'separable', 'Isotropic': False, 'Nugget': 0.0}
            k = uq_eval_Kernel(X1, X2, theta, opts)               # (n1, n2)

            if der1 == der2:
                i = der1
                # d²k/dx*_i² = (delta_i²/theta_i² - 1) / theta_i² * k
                return (delta[:, :, i]**2 / theta[i]**2 - 1) / theta[i]**2 * k
            else:
                i, j = der1, der2
                # d²k/(dx*_i dx*_j) = delta_i * delta_j / (theta_i² * theta_j²) * k
                return (delta[:, :, i] * delta[:, :, j]
                        / (theta[i]**2 * theta[j]**2) * k)
        return f

    elif family.lower() == 'matern-5_2':
        def f(X1, X2, theta):
            X1    = np.asarray(X1, dtype=float)
            X2    = np.asarray(X2, dtype=float)
            theta = np.asarray(theta, dtype=float).ravel()

            delta  = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]  # (n1, n2, M)
            a      = np.sqrt(5) * np.abs(delta) / theta            # (n1, n2, M)
            K_uni  = (1 + a + a**2 / 3) * np.exp(-a)              # (n1, n2, M)
            K_excl = _prod_excl(K_uni)                             # (n1, n2, M)

            if der1 == der2:
                i = der1
                # d²k/dx*_i² = -(5/(3*theta_i²)) * (1 + a_i - a_i²) * exp(-a_i) * K_excl_i
                return (-(5 / (3 * theta[i]**2))
                        * (1 + a[:, :, i] - a[:, :, i]**2)
                        * np.exp(-a[:, :, i]) * K_excl[:, :, i])
            else:
                i, j = der1, der2
                # d²k/(dx*_i dx*_j) = (25/(9*ti²*tj²)) * di*dj*(1+ai)*(1+aj)*exp(-(ai+aj))*K_excl_ij
                K_excl_ij = K_excl[:, :, i] / K_uni[:, :, j]
                return ((25 / (9 * theta[i]**2 * theta[j]**2))
                        * delta[:, :, i] * delta[:, :, j]
                        * (1 + a[:, :, i]) * (1 + a[:, :, j])
                        * np.exp(-(a[:, :, i] + a[:, :, j])) * K_excl_ij)
        return f

    else:
        raise ValueError(
            f'kernel_second_deriv_factory : noyau "{family}" non supporte.')


def uq_assemble_global_Kernel(X1, X2, theta, family):
    """
    Assemble la matrice de correlation augmentee GEK de taille n1(m+1) x n2(m+1).

    Ordre dimension-major (Zuhal 2021 Eq. 6) :
        lignes  0 .. n1-1       : bloc valeurs (X1)
        lignes  k*n1 .. (k+1)*n1 : bloc derivee dy/dx_{k-1}  pour k=1..m
        colonnes : meme ordre sur X2

    Le bloc (rb, cb) de taille n1 x n2 est calcule par
        kernel_deriv_factory(family, der, dp)(X1, X2, theta)
    avec
        der = rb - 1  si rb > 0  sinon None   (derivee 1er arg = X1, point-LIGNE)
        dp  = cb - 1  si cb > 0  sinon None   (derivee 2e  arg = X2, point-COLONNE)

    Convention (Zuhal 2021, Eq. 7) :
        bloc(k, 0) = dR/dx^i_k  (1er arg, point-ligne x^i)  -> der=k-1, dp=None
        bloc(0, k) = dR/dx^j_k  (2e  arg, point-col   x^j)  -> der=None, dp=k-1

    CES DEUX LIGNES ETAIENT INVERSEES dans la docstring d'origine, ainsi que
    l'attribution de `der`/`dp` a `rb`/`cb` ci-dessus. Le code, lui, etait
    juste : verifie par differences finies le 26/08/2026 (defaut 4 du plan de
    nettoyage), cf. `tests/test_51_convention_derivees.py`. Le bloc-ligne
    rb=0 porte bien les derivees par rapport au SECOND argument, ce qui est
    ce que reclame Cov(y(x), dy/dx'_k) = dk(x, x')/dx'_k.

    Parameters
    ----------
    X1     : (n1, m) ndarray
    X2     : (n2, m) ndarray
    theta  : (m,)  ndarray    longueurs de correlation
    family : str              'gaussian' ou 'matern-5_2'

    Returns
    -------
    R_tilde : (n1*(m+1), n2*(m+1)) ndarray
    """
    X1    = np.asarray(X1, dtype=float)
    X2    = np.asarray(X2, dtype=float)
    n1, m = X1.shape
    n2    = X2.shape[0]
    theta = np.asarray(theta, dtype=float).ravel()

    N_rows  = n1 * (m + 1)
    N_cols  = n2 * (m + 1)
    R_tilde = np.empty((N_rows, N_cols))

    for rb in range(m + 1):
        der = rb - 1 if rb > 0 else None
        for cb in range(m + 1):
            dp  = cb - 1 if cb > 0 else None
            f   = kernel_deriv_factory(family, der, dp)
            R_tilde[rb*n1 : (rb+1)*n1,
                    cb*n2 : (cb+1)*n2] = f(X1, X2, theta)

    return R_tilde


def uq_assemble_deriv_global_Kernel(X1, X2, theta, family, der_var):
    """
    Assemble dr0_tilde/dX1_{der_var} de taille (n1, n2*(m+1)).

    Derivee du vecteur de cross-correlation r0_tilde (bloc rb=0) par rapport
    a la composante der_var du premier argument X1 (point test).

    Bloc cb=0 : d[k(X1,X2)]/dX1_{der_var}
              = kernel_deriv_factory(family, der_var, None)

    Bloc cb=l (l=1..m) : d[dk(X1,X2)/dX1_{l-1}]/dX1_{der_var}
                        = d²k/(dX1_{der_var} dX1_{l-1})
                        = kernel_second_deriv_factory(family, der_var, l-1)

    Parameters
    ----------
    X1      : (n1, m) ndarray  -- points test
    X2      : (n2, m) ndarray  -- points train
    theta   : (m,)  ndarray
    family  : str              'gaussian' ou 'matern-5_2'
    der_var : int              composante de derivation (0-indexe)

    Returns
    -------
    dr0 : (n1, n2*(m+1)) ndarray
    """
    X1    = np.asarray(X1, dtype=float)
    X2    = np.asarray(X2, dtype=float)
    n1, m = X1.shape
    n2    = X2.shape[0]
    theta = np.asarray(theta, dtype=float).ravel()

    dr0 = np.empty((n1, n2 * (m + 1)))

    for cb in range(m + 1):
        dp = cb - 1 if cb > 0 else None
        f  = kernel_deriv_factory(family, der_var, dp)
        dr0[:, cb*n2 : (cb+1)*n2] = f(X1, X2, theta)

    return dr0


def uq_eval_deriv_global_Kernel(X1, X2, theta, options, der_var):
    """
    Calcule dr0_tilde/dX1_{der_var} pour GEPCK (cas non-Gram uniquement).

    Derivee du vecteur de cross-correlation augmente r0_tilde par rapport a
    la composante der_var du point test X1.
    Utilisee dans predict_deriv pour le terme correlation de dY_hat/dx_{der_var}.

    Parameters
    ----------
    X1      : (n1, m) ndarray  -- points test
    X2      : (n2, m) ndarray  -- points train
    theta   : (m,)  ndarray
    options : dict  ('Family', eventuellement 'Nugget' ignore ici)
    der_var : int   composante de derivation (0-indexe)

    Returns
    -------
    dr0 : (n1, n2*(m+1)) ndarray
    """
    family = options['Family']
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    theta = np.asarray(theta, dtype=float).ravel()
    return uq_assemble_deriv_global_Kernel(X1, X2, theta, family, der_var)


def uq_eval_global_Kernel(X1, X2, theta, options):
    """
    Calcule R_tilde (Gram) ou r0_tilde (non-Gram) pour GEPCK.

    Equivalent de uq_eval_Kernel pour GEPCK : meme signature (X1, X2, theta,
    options), gere nugget et detection Gram.
    Utilise comme CorrOptions['Handle'] dans fit_gepck_kriging et la prediction.

    Cas Gram   (X1 == X2) : retourne R_tilde  (n1*(m+1), n1*(m+1))
        Tous les (m+1)^2 blocs, nugget ajoute sur la diagonale si non nul.

    Cas non-Gram (X_test, X_train) : retourne r0_tilde  (n1, n2*(m+1))
        Seulement le bloc-ligne rb=0 de la matrice complete :
            r0_tilde[i, cb*n2:(cb+1)*n2] = Cov(y(x_i), y_aug_bloc_cb(X_train))
        avec  cb=0 -> k(X_test, X_train)                  (der=None, dp=None)
              cb=l -> dk(X_test, X_train)/dX_train_{l-1}  (der=None, dp=l-1)
        Forme compatible avec la prediction : r0_tilde @ (R_tilde_inv @ y_aug)

        La derivee porte sur le SECOND argument, X_train, et non sur X_test :
        la l-ieme observation augmentee est dy/dx_l AU POINT D'APPRENTISSAGE,
        donc Cov(y(x*), dy/dx_l(x^j)) = dk(x*, x^j) / dx^j_l.
        La docstring d'origine annoncait `dk/dX_test` et `(der=l-1, dp=None)`,
        ce qui contredisait le code -- et le code avait raison. Tranche le
        26/08/2026 par differences finies (defaut 4 du plan de nettoyage) :
        voir `tests/test_51_convention_derivees.py`.

    Parameters
    ----------
    X1      : (n1, m) ndarray
    X2      : (n2, m) ndarray
    theta   : (m,)  ndarray
    options : dict avec les champs :
        'Family' : 'gaussian' ou 'matern-5_2'
        'Nugget' : float (defaut 0.0)

    Returns
    -------
    Gram     : R_tilde  de forme (n1*(m+1), n1*(m+1))
    non-Gram : r0_tilde de forme (n1, n2*(m+1))
    """
    family = options['Family']
    nugget = float(options.get('Nugget', 0.0))

    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    n1, m = X1.shape
    n2    = X2.shape[0]

    # DEFAUT 1 du plan de nettoyage, corrige le 26/08/2026.
    #
    # Les deux branches ne rendent pas la meme FORME : (n(m+1), n(m+1)) contre
    # (n1, n2(m+1)). Les choisir en inspectant le CONTENU des tableaux est donc
    # une devinette sur ce que l'appelant voulait -- et elle se trompait des
    # que le metamodele etait evalue sur un point deja present dans son plan
    # d'experiences : `predict_gepck` reclamait r0_tilde et recevait R_tilde,
    # d'ou un « operands could not be broadcast together ». Cela arrive pour de
    # bon : une grille EFF qui passe par un point du DOE, une relecture de
    # cache, une verification d'interpolation.
    #
    # L'appelant DIT desormais ce qu'il veut, via `options['IsGram']`. Le repli
    # sur l'ancienne heuristique est conserve pour les appels qui ne le
    # precisent pas encore.
    isGram = options.get('IsGram')
    if isGram is None:
        isGram = (n1 == n2) and np.array_equal(X1, X2)
    else:
        isGram = bool(isGram)
        if isGram and (n1 != n2 or not np.array_equal(X1, X2)):
            raise ValueError(
                "IsGram=True mais X1 et X2 different : une matrice de Gram "
                "suppose le meme jeu de points des deux cotes (%s contre %s)."
                % (X1.shape, X2.shape))

    if isGram:
        # R_tilde complet : (m+1)^2 blocs, forme (n1*(m+1), n1*(m+1))
        R_tilde = uq_assemble_global_Kernel(X1, X2, theta, family)
        if nugget != 0.0:
            R_tilde += nugget * np.eye(n1 * (m + 1))
        return R_tilde
    else:
        # r0_tilde : m+1 blocs (rb=0) — forme (n1, n2*(m+1))
        # Convention article (Zuhal 2021, p.4) :
        #   cb=0 : k(X_test, X_train)                    — der=None, dp=None
        #   cb=l : dk(X_test, X_train)/dX_test_{l-1}     — der=l-1,  dp=None
        # = premier bloc-ligne de uq_assemble_global_Kernel(X1, X2, ...)
        # Propriete : r0_tilde(X_train) = R_tilde[:N_train, :] -> interpolation exacte.
        r0_tilde = np.empty((n1, n2 * (m + 1)))
        for cb in range(m + 1):
            dp  = cb - 1 if cb > 0 else None
            f   = kernel_deriv_factory(family, None, dp)
            r0_tilde[:, cb*n2 : (cb+1)*n2] = f(X1, X2, theta)
        return r0_tilde
