"""
Transformations isoprobabilistes -- extrait de branche5.py.

Nataf, Rosenblatt, et le cas independant. Passage de l'espace physique X
a l'espace auxiliaire U et retour.

PHASE 3 du plan de nettoyage : `branche5.py` portait TROIS sujets sans
rapport entre eux. L'analyse du graphe d'appels
(`tools/analyse_dependances.py`) a montre **zero arete** entre les trois --
c'etaient trois modules colles dans un fichier.

Le corps des fonctions est repris VERBATIM : la scission ne doit changer
aucun resultat, et la baseline le verifie.
"""

import numpy as np
import scipy.stats as stats


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


def _memes_marginales(a, b):
    """Deux marginales decrivent-elles la MEME loi ?

    Comparaison stricte du type et des parametres. Une egalite approchee
    n'aurait pas de sens ici : on cherche le cas ou la transformation est
    l'identite exacte, pas le cas ou elle en est proche.
    """
    if a.get('Type', '').lower() != b.get('Type', '').lower():
        return False
    if ('Bounds' in a) or ('Bounds' in b):
        return False          # bornes tronquees : la loi n'est plus la meme
    pa = np.atleast_1d(np.asarray(a.get('Parameters', []), dtype=float))
    pb = np.atleast_1d(np.asarray(b.get('Parameters', []), dtype=float))
    return pa.shape == pb.shape and bool(np.array_equal(pa, pb))


def _uq_IsopTransform(X, X_Marginals, Y_Marginals):
    """
    Transforme X marginale par marginale (cas copule indépendante) :
    X → Uniform[0,1] via CDF source → Y via CDF inverse cible.
    Supporte : Uniform, Gaussian/Normal, Lognormal, Gumbel, Exponential.

    RACCOURCI SUR LES MARGINALES IDENTIQUES -- phase 7, 26/08/2026.
    Quand la loi source et la loi cible sont la meme, le passage par la CDF
    puis la CDF inverse est l'identite en theorie, et couteux et FAUX en
    pratique. C'est le cas dominant de cette chaine : tout s'y passe en espace
    standard, marginales Gaussiennes (0, 1) des deux cotes.

    Ce n'est pas qu'une affaire de vitesse. `norm.cdf(u)` sature a 1.0 en
    double precision des que u grandit, et `norm.ppf` ne peut plus revenir :

        u = 5   erreur 3,0e-11
        u = 7   erreur 5,8e-06
        u = 8   erreur 8,4e-03
        u > 8,3 +inf

    Or les bornes de recherche EFF de ces etudes valent exactement
    [-7,5 ; +7,5] : le metamodele perdait donc des chiffres significatifs au
    bord meme du domaine explore, et la grille de trace frolait la falaise.
    Cote negatif il n'y a rien de tel -- `cdf` y reste representable -- d'ou
    une erreur ASYMETRIQUE, la pire sorte a diagnostiquer.
    """
    N, M = X.shape
    Y = np.empty_like(X)

    for i in range(M):
        mx = X_Marginals[i]
        my = Y_Marginals[i]

        if _memes_marginales(mx, my):
            Y[:, i] = X[:, i]
            continue

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
