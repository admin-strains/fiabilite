"""
Critere d'enrichissement EFF (Expected Feasibility Function).

Extrait de `AC3_pure_flexion.py` / `AC3_moulinblanc.py`. PHASE 3 du plan de
nettoyage.

LA FORMULE ETAIT ECRITE DEUX FOIS DANS CHAQUE SCRIPT
----------------------------------------------------
`_eff_vectorized` la calculait sur des tableaux, `EFFFunction._exec` la
recalculait point par point pour OpenTURNS -- soit quatre copies de la meme
algebre dans le depot. Elles ont ete comparees avant d'etre unifiees :
ecart **exactement nul** sur 4 005 points et trois valeurs d'`eps_factor`,
cas limites compris (sigma nul, sigma negatif, mu nul).

Il n'en reste qu'une, `eff()`, et `eff_termes()` en donne la decomposition
pour les journaux -- au lieu de la recopier, ce qui etait la source du defaut.

Ce module ne demande que **numpy et scipy** : il tourne partout. L'emballage
OpenTURNS est dans `eff_ot.py`, pour que la separation des dependances soit
reelle et pas seulement annoncee.
"""

import numpy as np
from scipy.stats import norm


def eff(mu, sigma, eps_factor):
    """Critere EFF (Expected Feasibility Function), vectorise.

    Corps repris VERBATIM de `_eff_vectorized`. Les points ou sigma <= 0
    rendent 0 : le metamodele y est certain, il n'y a rien a gagner a
    l'enrichir.
    """
    eps        = eps_factor * sigma
    safe_sigma = np.where(sigma > 0, sigma, 1.0)
    t1 = -mu / safe_sigma
    t2 = (eps + mu) / safe_sigma
    t3 = (eps - mu) / safe_sigma
    eff_vals = (2*mu*norm.cdf(t1) - (eps+mu)*norm.cdf(-t2) + (eps-mu)*norm.cdf(t3)
                + sigma*(-2*norm.pdf(t1) + norm.pdf(t2) + norm.pdf(t3)))
    return np.where(sigma > 0, eff_vals, 0.0)


def eff_termes(mu, sigma, eps_factor):
    """Decomposition de `eff` en ses quatre termes, pour les journaux.

    Existe pour une raison precise. Les scripts AC decomposaient la formule a
    la main dans un bloc de diagnostic, et cette copie etait FAUSSE : son
    quatrieme terme valait `sigma*(pdf(t2) - pdf(t3))` au lieu de
    `sigma*(-2*pdf(t1) + pdf(t2) + pdf(t3))`. La ligne "EFF = ..." du journal
    d'enrichissement affichait donc de 39 % a 1271 % d'ecart avec le critere
    reellement maximise, parfois avec le signe oppose.

    La somme des quatre termes vaut `eff` par construction, et un test le
    verifie -- ce qui rend le meme ecart impossible a reintroduire.

    Renvoie (t1, t2, t3, terme1, terme2, terme3, terme4).
    """
    sigma = np.asarray(sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    eps = eps_factor * sigma
    safe = np.where(sigma > 0, sigma, 1.0)
    t1 = -mu / safe
    t2 = (eps + mu) / safe
    t3 = (eps - mu) / safe
    z = np.where(sigma > 0, 1.0, 0.0)
    return (t1, t2, t3,
            z * 2 * mu * norm.cdf(t1),
            z * -(eps + mu) * norm.cdf(-t2),
            z * (eps - mu) * norm.cdf(t3),
            z * sigma * (-2 * norm.pdf(t1) + norm.pdf(t2) + norm.pdf(t3)))
