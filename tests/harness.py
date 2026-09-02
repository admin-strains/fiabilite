"""
Colle entre les oracles (tests/reference/) et le code teste (_lib/).

Ce module reproduit EXACTEMENT le mode d'appel du metamodele utilise en
production dans les scripts AC (init_g_ot, branches do_PCK / do_GEPCK) :
memes marginales, meme copule, memes options, meme mise en forme de Y_aug.
Si le harness passe mais que la production casse, c'est que ce fichier a
divergé des scripts AC -- le maintenir en meme temps qu'eux.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_DIR = os.path.join(ROOT, '_lib')
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)


# --------------------------------------------------------------------------- #
# DOE fige -- volontairement independant de scipy.qmc et de api.generate_doe
# (dont l'implementation LHS peut changer d'une version de scipy a l'autre).
# Latin Hypercube centre + permutation PCG64 : stable entre versions numpy.
# --------------------------------------------------------------------------- #
def make_doe(n, n_var, seed=20260825):
    rng = np.random.default_rng(seed)
    U01 = np.empty((n, n_var))
    centers = (np.arange(n) + 0.5) / n
    for j in range(n_var):
        U01[:, j] = centers[rng.permutation(n)]
    from scipy.special import ndtri
    return ndtri(U01)          # espace standard N(0,1)


# --------------------------------------------------------------------------- #
# Construction du metamodele -- copie conforme des options de production       #
# --------------------------------------------------------------------------- #
def _inputs(n_var):
    marginals = [{'Type': 'Gaussian', 'Parameters': [0.0, 1.0]}] * n_var
    copula = {'Type': 'Independent', 'Parameters': np.eye(n_var)}
    return marginals, copula


def default_opts(max_degree=3):
    """Options 'Mode=optimal' des scripts AC (init_g_ot, fixed_fm is None)."""
    return {'Mode': 'optimal',
            'PCE': {'Degree': list(range(1, max_degree + 1)), 'Method': 'LARS'}}


def build_Y_aug(yt, all_grad):
    """Identique a AC3_pure_flexion.build_Y_aug (eq. 6 Zuhal et al.)."""
    y_flat = np.asarray(yt).ravel()
    grad_blocks = [np.asarray(all_grad)[:, j] for j in range(np.asarray(all_grad).shape[1])]
    return np.concatenate([y_flat] + grad_blocks)


def fit(kind, X, ls, max_degree=3, opts=None):
    """
    kind : 'PCK' ou 'GEPCK'. Renvoie le fitted_model brut de _lib.
    Les warnings d'optimisation sont laisses visibles (contrairement a la
    production qui les masque) : un nouveau warning est une information.
    """
    from api import fit_pck, fit_gepck

    X = np.atleast_2d(np.asarray(X, dtype=float))
    n_var = X.shape[1]
    marginals, copula = _inputs(n_var)
    opts = opts if opts is not None else default_opts(max_degree)

    y = np.asarray(ls.g(X), dtype=float).ravel()
    if kind == 'PCK':
        return fit_pck(X, y, opts, marginals, copula)
    if kind == 'GEPCK':
        grad = np.asarray(ls.grad(X), dtype=float)
        return fit_gepck(X, build_Y_aug(y, grad), opts, marginals, copula)
    raise ValueError('kind doit valoir PCK ou GEPCK, pas %r' % (kind,))


def predictors(kind, fm, fd_step=1e-5):
    """
    Renvoie (g_hat, grad_hat, sigma_hat) utilisables par reference.form.hlrf.

    - GEPCK : grad_hat = gradient ANALYTIQUE du metamodele (predict_gradient_gepck)
    - PCK   : grad_hat = differences finies centrees (comme OpenTURNS par defaut)
    """
    from api import predict_pck, predict_gepck, predict_gradient_gepck

    pred = predict_pck if kind == 'PCK' else predict_gepck

    def g_hat(U):
        return np.asarray(pred(fm, np.atleast_2d(U))).reshape(-1)

    def sigma_hat(U):
        _, var = pred(fm, np.atleast_2d(U), return_var=True)
        return np.sqrt(np.maximum(np.asarray(var).reshape(-1), 0.0))

    if kind == 'GEPCK':
        def grad_hat(U):
            return np.asarray(predict_gradient_gepck(fm, np.atleast_2d(U)))
    else:
        def grad_hat(U):
            U = np.atleast_2d(np.asarray(U, dtype=float))
            G = np.zeros_like(U)
            for j in range(U.shape[1]):
                e = np.zeros(U.shape[1]); e[j] = fd_step
                G[:, j] = (g_hat(U + e) - g_hat(U - e)) / (2.0 * fd_step)
            return G

    return g_hat, grad_hat, sigma_hat


# --------------------------------------------------------------------------- #
# Extraction de la signature d'un modele ajuste (ce qu'on gele en golden)      #
# --------------------------------------------------------------------------- #
def signature(fm, X_probe, kind):
    """
    Resume numerique reproductible d'un metamodele ajuste.
    C'est cet objet qui est compare au fichier golden.
    """
    from api import predict_pck, predict_gepck

    pred = predict_pck if kind == 'PCK' else predict_gepck
    mu, var = pred(fm, X_probe, return_var=True)

    npoly = fm['NumberOfPoly']
    npoly = int(np.atleast_1d(npoly)[0])
    idx = fm['idxranking'][0][:npoly]
    sel = fm['AllIndices'][0][np.array(idx), :]

    return {
        'kind': kind,
        'LOO': float(fm['Error'][0]['LOO']),
        'NumberOfPoly': npoly,
        'theta': np.asarray(fm['Kriging'][0]['theta'], dtype=float).ravel().tolist(),
        'sigmaSQ': float(np.atleast_1d(fm['Kriging'][0]['sigmaSQ']).ravel()[0]),
        'beta_pce': np.asarray(fm['Kriging'][0]['beta'], dtype=float).ravel().tolist(),
        'poly_indices': np.asarray(sel, dtype=int).tolist(),
        'mu_probe': np.asarray(mu, dtype=float).ravel().tolist(),
        'var_probe': np.asarray(var, dtype=float).ravel().tolist(),
    }


PROBE = np.array([[0.0, 0.0],
                  [1.0, -1.0],
                  [-2.0, 0.5],
                  [0.25, 2.75],
                  [-3.0, -3.0]])

#: Les memes cinq points, en trois variables. Ils gardent la meme intention :
#: le centre, un point courant, un point excentre, un point loin sur une seule
#: variable, et un coin. Le troisieme cas de reference (`console`) a trois
#: variables -- voir `tests/reference/limit_states.ConsoleLS`.
PROBE_3 = np.array([[0.0, 0.0, 0.0],
                    [1.0, -1.0, 0.5],
                    [-2.0, 0.5, 1.5],
                    [0.25, 2.75, -0.75],
                    [-3.0, -3.0, -3.0]])


def probe(n_var):
    """Les points sonde du cas, choisis par sa dimension.

    Les goldens PORTENT leur propre sonde (`ref['probe']`) : ce choix ne sert
    qu'a la GENERATION. Un golden reste donc lisible meme si cette table
    change, et `test_doe_du_golden_est_toujours_celui_du_harness` verifie que
    le plan, lui, n'a pas bouge.
    """
    if n_var == 2:
        return PROBE
    if n_var == 3:
        return PROBE_3
    raise ValueError("aucun jeu de points sonde fige pour %d variables : en "
                     "ajouter un ici plutot que d'en tirer au hasard" % n_var)
