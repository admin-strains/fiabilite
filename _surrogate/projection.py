r"""Projection du metamodele sur les variables autres que la POSITION.

LE PROBLEME
------------
Certaines etudes portent une variable de position -- l'abscisse d'un convoi,
par exemple -- qui n'est pas une variable aleatoire au meme titre que les
autres : on ne cherche pas la probabilite que le convoi soit a tel endroit, on
cherche la position la PLUS DEFAVORABLE. L'etat limite pertinent est donc
l'enveloppe

    g_proj(u_autres) = min_p  g(u_autres, p)

et la fiabilite se calcule sur cette enveloppe, dans un espace a une dimension
de moins.

LA MINIMISATION
----------------
Une descente locale depuis un point quelconque se fait piéger : sur un ouvrage
a plusieurs travees, `g` en fonction de la position a plusieurs creux. D'ou la
strategie en deux temps -- une grille grossiere pour trouver le bon creux,
puis un affinage borne autour de lui.

CE QUE L'EXTRACTION A RENDU VISIBLE
------------------------------------
Le domaine balaye etait ecrit `-5.0, 5.0` en dur, sans rapport avec les bornes
de l'etude (`eff_bounds`) ni avec la loi de la variable de position. C'est
desormais un argument, avec cette valeur pour defaut : la changer deplacerait
les resultats publies, mais au moins elle se voit.
"""

import numpy as np
import openturns as ot
from scipy.optimize import minimize_scalar


def projeter_surrogate(g_ot, n_var, idx_position, borne=5.0, n_grille=30,
                       xatol=1e-4, maxiter=200):
    """L'enveloppe de `g_ot` sur la variable de position.

    Retourne `g_ot` INCHANGE si `idx_position` est None -- une etude sans
    variable de position n'a rien a projeter.

    Sinon retourne une `ot.Function` de dimension `n_var - 1`, dont chaque
    evaluation coute `n_grille` evaluations du metamodele plus l'affinage.
    C'est du metamodele, pas du solveur : cher en secondes, pas en heures.
    """
    if idx_position is None:
        return g_ot

    idx_autres = [i for i in range(n_var) if i != idx_position]
    n_proj = len(idx_autres)

    class _Projete(ot.OpenTURNSPythonFunction):
        def __init__(self):
            super().__init__(n_proj, 1)

        def _exec(self, u_reduit):
            def _objectif(u_pos):
                u_complet = [0.0] * n_var
                for k, idx in enumerate(idx_autres):
                    u_complet[idx] = float(u_reduit[k])
                u_complet[idx_position] = u_pos
                return float(g_ot(ot.Point(u_complet))[0])

            # grille grossiere d'abord : `g` peut avoir plusieurs creux le long
            # de la position (une travee par creux), et un affinage lance au
            # mauvais endroit converge vers le mauvais minimum.
            u_grille = np.linspace(-borne, borne, n_grille)
            g_grille = [_objectif(u) for u in u_grille]
            u_best = u_grille[np.argmin(g_grille)]
            res = minimize_scalar(_objectif,
                                  bounds=(max(-borne, u_best - 0.5),
                                          min(borne, u_best + 0.5)),
                                  method='bounded',
                                  options={'xatol': xatol, 'maxiter': maxiter})
            return [res.fun]

    return ot.Function(_Projete())
