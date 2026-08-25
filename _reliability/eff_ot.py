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
