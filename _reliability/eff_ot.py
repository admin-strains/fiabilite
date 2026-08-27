"""
Emballage OpenTURNS du critere EFF.

Separe de `eff.py` parce que la formule elle-meme ne demande que numpy et
scipy : la mettre dans le meme fichier que `import openturns` aurait rendu le
noyau indisponible partout ou OpenTURNS ne l'est pas. La docstring de `eff.py`
affirmait cette separation ; ce fichier la rend vraie.
"""

import numpy as np
import openturns as ot

from eff import eff


def eff_function(g_ot, sigma_func, n_var, epsilon_factor):
    """Emballe `eff` pour OpenTURNS, qui evalue point par point.

    Remplace la classe `EFFFunction` des scripts AC, dont `_exec` reecrivait
    la formule. `n_var` et `epsilon_factor` etaient des variables libres de
    `main` ; ce sont maintenant des parametres.

    S'utilise comme avant : `ot.Function(eff_function(...))`.
    """

    class _EFF(ot.OpenTURNSPythonFunction):
        def __init__(self):
            super().__init__(n_var, 1)
            self.g_ot = g_ot
            self.sigma_func = sigma_func

        def _exec(self, u):
            u = ot.Point(u)
            sigmaG = self.sigma_func(u)
            if sigmaG <= 0.0:
                return [0.0]
            muG = self.g_ot(u)[0]
            return [float(eff(np.array([muG]), np.array([sigmaG]), epsilon_factor)[0])]

    return _EFF()


def batch_mu_sigma(g_ot, sigma_func, grid, predict):
    """Calcule (mu, sigma) en batch sur une grille de points.
    - GEPCK : 1 appel predict_gepck(return_var=True) -> mu + var en 1 fois (BLAS multi-thread)
    - Autres modeles : batch via ot.Sample pour mu, loop fallback pour sigma
    """
    _fm = getattr(getattr(sigma_func, '__self__', None), 'fm', None)
    if _fm is not None:
        mu_arr, sig2_arr = predict(_fm, grid, return_var=True)
        mu    = mu_arr[:, 0]
        sigma = np.sqrt(np.maximum(0.0, sig2_arr[:, 0]))
        return mu, sigma
    grid_ot = ot.Sample(grid.tolist())
    mu    = np.array(g_ot(grid_ot))[:, 0]
    sigma = np.array([sigma_func(pt) for pt in grid])
    return mu, sigma


def _ecrire(message):
    print(message, flush=True)


def maximiser_EFF(f_eff, bornes_min, bornes_max, n_var, n_appels):
    """Le point du domaine ou le critere EFF est le plus grand.

    `GN_DIRECT` est un algorithme GLOBAL sans derivees : le critere EFF est
    multimodal par construction -- il vaut zero loin de l'etat limite comme
    dans les zones deja bien connues -- et une descente locale depuis
    l'origine s'arreterait sur le premier plateau venu.

    Le budget `n_appels` porte sur le METAMODELE, pas sur le solveur : ces
    evaluations sont gratuites a l'echelle d'un point haute fidelite.

    Retourne `(u, valeur du critere en u)`.
    """
    probleme = ot.OptimizationProblem(f_eff, ot.Function(), ot.Function(),
                                      ot.Interval(bornes_min, bornes_max))
    probleme.setMinimization(False)
    algo = ot.NLopt(probleme, "GN_DIRECT")
    algo.setStartingPoint([0.0] * n_var)
    algo.setMaximumCallsNumber(n_appels)
    algo.run()
    u = np.array(algo.getResult().getOptimalPoint())
    return u, f_eff(ot.Point(u.tolist()))[0]


def batch_kriging_believer(g_ot, sigma_func, xt, yt, all_grad, *,
                           n_batch, bornes_min, bornes_max, n_var, n_appels,
                           epsilon_factor, reajuster, gradient_du_surrogate=False,
                           tracer=_ecrire):
    """Plusieurs points d'enrichissement d'un coup, sans appeler le solveur.

    A `n_batch = 1`, c'est une simple maximisation du critere.

    Au-dela, la difficulte est qu'on veut plusieurs points AVANT d'en avoir
    calcule un seul -- sinon le pool de solveurs reste vide. Re-maximiser le
    critere sans rien changer redonnerait le meme point. Kriging Believer
    contourne cela en CROYANT le metamodele : le point retenu est verse au
    plan avec, pour valeur, la prediction du metamodele lui-meme. Le
    metamodele est reajuste, sa variance s'effondre autour de ce point, et le
    critere designe alors ailleurs.

    Le reajustement se fait a `theta` et polynomes FIXES (`fixed_fm`) : une
    observation fictive n'est pas une mesure, elle n'a pas a redefinir la
    portee de correlation. C'est aussi ce qui rend l'operation abordable --
    reoptimiser `theta` a chaque point du batch couterait plus que les
    appels solveur qu'on cherche a paralleliser.

    `gradient_du_surrogate` : pour un metamodele qui exploite les gradients
    (GEPCK), le gradient fictif est celui du metamodele ; sinon, des zeros.
    Un zero n'est pas neutre -- il affirme que l'etat limite est plat -- mais
    un metamodele qui n'utilise pas les gradients ne le lira pas.

    Retourne `(liste des points, valeur du critere au PREMIER point)`.
    Cette valeur est celle du vrai metamodele : c'est elle qui juge l'arret,
    et les suivantes viennent de metamodeles fictifs.
    """
    f = ot.Function(eff_function(g_ot, sigma_func, n_var, epsilon_factor))
    u1, valeur = maximiser_EFF(f, bornes_min, bornes_max, n_var, n_appels)
    batch = [u1]
    if n_batch <= 1:
        return batch, valeur

    xt_kb, yt_kb, ag_kb = np.copy(xt), np.copy(yt), np.copy(all_grad)
    g_kb, s_kb = g_ot, sigma_func
    # `theta` et les polynomes du metamodele COURANT, a imposer aux
    # reajustements fictifs.
    fm_kb = getattr(getattr(s_kb, '__self__', None), 'fm', None)

    for k in range(1, n_batch):
        u_prev = batch[-1]
        y_fictif = float(g_kb(ot.Point(u_prev.tolist()))[0])
        xt_kb = np.vstack([xt_kb, [u_prev]])
        yt_kb = np.vstack([yt_kb, [[y_fictif]]])
        if gradient_du_surrogate:
            grad_fictif = np.array([[float(g_kb.gradient(ot.Point(u_prev.tolist()))[i, 0])
                                     for i in range(n_var)]])
        else:
            grad_fictif = np.zeros((1, n_var))
        ag_kb = np.vstack([ag_kb, grad_fictif])
        g_kb, s_kb, xt_kb, yt_kb, ag_kb = reajuster(
            None, None, xt_kb, yt_kb, ag_kb, fixed_fm=fm_kb)
        f_kb = ot.Function(eff_function(g_kb, s_kb, n_var, epsilon_factor))
        u_k, valeur_kb = maximiser_EFF(f_kb, bornes_min, bornes_max, n_var,
                                       n_appels)
        batch.append(u_k)
        tracer("  [KB %d/%d] u=%s  EFF=%.6f"
               % (k + 1, n_batch, list(np.round(u_k, 3)), valeur_kb))

    return batch, valeur
