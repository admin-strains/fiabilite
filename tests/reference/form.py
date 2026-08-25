"""
FORM de reference -- HL-RF avec recherche lineaire sur fonction merite.

Volontairement independant d'OpenTURNS : le harness doit pouvoir tourner sur
un poste sans STRAINS ni environnement conda. L'algorithme est le HL-RF
classique (Hasofer-Lind / Rackwitz-Fiessler) avec la fonction merite de
Zhang & Der Kiureghian, soit exactement la famille d'algorithmes utilisee
par ot.AbdoRackwitz dans les scripts AC.

Il ne s'agit PAS d'un remplacement du FORM de production : c'est un oracle
lent et simple, dont on sait qu'il converge sur les cas tests du harness.
"""

import numpy as np


def hlrf(g_func, grad_func, u0, max_iter=200, tol_g=1e-8, tol_u=1e-8,
         c_merit=None, ls_max=20, ls_rho=0.5):
    """
    Cherche u* = argmin ||u|| sous g(u) = 0.

    Parameters
    ----------
    g_func    : callable (N,M) -> (N,)   fonction de performance
    grad_func : callable (N,M) -> (N,M)  gradient de g
    u0        : (M,) point de depart
    tol_g     : |g(u)| accepte a convergence. 1e-8 et non 1e-10 : quand le
                gradient vient de differences finies sur un metamodele, le
                bruit d arrondi de la prediction (~1e-9) empeche de descendre
                plus bas. Reste 6 ordres de grandeur sous le tol_FORM=0.05
                utilise en production sur Moulin Blanc.
    tol_u     : deplacement accepte a convergence.

    Returns
    -------
    dict : {'u_star', 'beta', 'n_iter', 'converged', 'g_star'}
    """
    u = np.asarray(u0, dtype=float).ravel().copy()

    def g1(v):
        return float(np.atleast_1d(g_func(np.atleast_2d(v)))[0])

    def dg1(v):
        return np.asarray(grad_func(np.atleast_2d(v)), dtype=float).ravel()

    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        gu = g1(u)
        du = dg1(u)
        ndu2 = float(du @ du)
        if ndu2 <= 0.0 or not np.isfinite(ndu2):
            break

        # direction HL-RF
        u_new_full = ((du @ u - gu) / ndu2) * du
        d = u_new_full - u

        # fonction merite : m(u) = 0.5||u||^2 + c |g(u)|
        c = c_merit if c_merit is not None else \
            (np.linalg.norm(u) / np.linalg.norm(du) + 10.0)
        m0 = 0.5 * u @ u + c * abs(gu)
        step = 1.0
        for _ in range(ls_max):
            ut = u + step * d
            mt = 0.5 * ut @ ut + c * abs(g1(ut))
            if mt <= m0:
                break
            step *= ls_rho
        else:
            ut = u + d  # pas de descente trouvee : pas plein

        shift = float(np.linalg.norm(ut - u))
        u = ut
        if abs(g1(u)) < tol_g and shift < tol_u:
            converged = True
            break

    return {'u_star': u,
            'beta': float(np.linalg.norm(u)),
            'n_iter': n_iter,
            'converged': bool(converged),
            'g_star': g1(u)}


def multistart_hlrf(g_func, grad_func, starts, **kw):
    """HL-RF depuis plusieurs points de depart ; renvoie le beta minimal."""
    best = None
    for u0 in np.atleast_2d(starts):
        r = hlrf(g_func, grad_func, u0, **kw)
        if not r['converged']:
            continue
        if best is None or r['beta'] < best['beta']:
            best = r
    return best
