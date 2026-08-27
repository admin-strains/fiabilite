r"""ACTION `figurer` : tout ce qui RESTITUE, rien qui CALCULE.

POURQUOI CE MODULE EXISTE
--------------------------
Le 26/08/2026, sept des onze fonctions `print_*` des scripts d'etude pouvaient
declencher un appel au solveur -- jusqu'a 225, soit 29 heures sur le Moulin
Blanc. Le chemin etait indirect, donc invisible a la lecture :

    print_planche_EFF -> _hf_from_custom_points -> run_HF
    print_3D_HF       -> run_HF                 (n_grid_hf^2 appels)
    print_results     -> run_HF                 (2 appels par mode FORM)

C'est ce qui m'a fait annoncer, a tort, que la grille haute fidelite arrivait
en DERNIER et qu'on pouvait l'interrompre sans rien perdre. Elle arrive AVANT
l'enrichissement. Une fonction nommee « print » n'eveille aucun soupcon.

LA REGLE DE CE FICHIER
-----------------------
Aucune fonction d'ici n'evalue l'etat limite. Ce n'est pas une intention :
`tests/test_93_figures_sans_solveur.py` construit le graphe d'appel du module
et calcule sa fermeture transitive. Une seule arete vers le solveur, meme a
travers trois intermediaires, fait echouer la suite.

Corollaire pratique : ce module prend en argument ce qui a deja ete calcule.
Il ne va pas le chercher.
"""

import openturns as ot


def _ecrire(message):
    """Sortie par defaut : vidangee, pour qu'un run suivi en direct ne
    reste pas muet pendant des heures dans le tampon."""
    print(message, flush=True)


def tracer_domaine_physique(dist, params_names, borne_min, borne_max,
                            ecrire=_ecrire):
    """Ce que les bornes de recherche valent EN UNITES PHYSIQUES.

    Personne ne lit `[-7.5, +7.5]` comme « de 8,9 a 461 MPa ». C'est pourtant
    ce que cela veut dire, et c'est ce qui a fait mourir un run de deux heures
    le 26/08/2026 : le rapport entre les deux nappes d'armatures atteignait 52
    a un coin du domaine, et Digital Structure a termine le processus.

    Renvoie la liste des couples (min, max) physiques, un par variable -- de
    quoi tester la fonction sans lire une sortie texte.

    ZERO appel solveur.
    """
    n_var = len(params_names)
    # T_inv est EXACTEMENT la transformation que l'evaluation de l'etat limite
    # emploie pour passer de u a x. En utiliser une autre ici ferait afficher
    # un domaine qui n'est pas celui qu'on evalue.
    #
    # La voie « naive » -- computeQuantile(Normal().computeCDF(u)) -- est de
    # surcroit FAUSSE dans la queue haute : `1 - p` y perd toute precision.
    # Mesure du 26/08/2026 sur Normal(235 ; 30,15) :
    #
    #     u = +6,0   ecart 2,7e-07
    #     u = +7,5   ecart 5,6e-03
    #     u = +8,5   ecart 11,3 MPa      (479,99 au lieu de 491,28)
    #
    # C'est le defaut corrige en phase 7 dans `_lib/transform.py`, et il a
    # failli etre reintroduit ici. T_inv : 5,7e-14 partout.
    T_inv = dist.getInverseIsoProbabilisticTransformation()
    bas_x = T_inv(ot.Point([borne_min] * n_var))
    haut_x = T_inv(ot.Point([borne_max] * n_var))

    ecrire("  DOMAINE DE RECHERCHE -- ce que les bornes valent physiquement")
    extremes = []
    for i, nom in enumerate(params_names):
        marginale = dist.getMarginal(i)
        lo, hi = sorted((float(bas_x[i]), float(haut_x[i])))
        extremes.append((lo, hi))
        alerte = ""
        if lo <= 0:
            alerte = "  <-- NEGATIF OU NUL"
        elif lo < 0.1 * float(marginale.getMean()[0]):
            alerte = "  <-- moins d'un dixieme de la moyenne"
        ecrire("    %-8s u in [%+.2f, %+.2f]  ->  [%.2f, %.2f]%s"
               % (nom, borne_min, borne_max, lo, hi, alerte))

    # Rapport le plus defavorable atteignable a un COIN du domaine : la
    # variable la plus haute contre la plus basse. C'est lui qui gouverne le
    # conditionnement des cones SOCP, pas la valeur absolue des bornes.
    if len(extremes) >= 2:
        bas = min(l for l, _ in extremes)
        haut = max(h for _, h in extremes)
        pire = haut / bas if bas > 0 else float('inf')
        ecrire("    rapport le plus defavorable aux coins : %.1f" % pire)
        if pire >= 5:
            ecrire("    ATTENTION : au-dela de ~5 les cones SOCP sont mal")
            ecrire("    conditionnes (docs/mesh/). A 52, le 26/08/2026,")
            ecrire("    Digital Structure a termine le processus.")
    ecrire("")
    return extremes


def resume_FORM(best_result, dist, params_names, ecrire=_ecrire):
    """Les resultats FORM, en clair.

    Cette moitie-la etait soudee a l'erreur FOSM, qui, elle, evalue l'etat
    limite EXACT en deux points. Une fonction nommee « print » lancait donc
    deux SOCP -- un quart d'heure sur le Moulin Blanc. Elles sont separees :
    le resume est ici, l'erreur FOSM reste du cote des actions qui coutent.

    Renvoie u*, dont l'appelant a besoin pour la suite.

    ZERO appel solveur.
    """
    u_star = best_result.getStandardSpaceDesignPoint()
    n_iter = best_result.getOptimizationResult().getIterationNumber()
    x_star = dist.getInverseIsoProbabilisticTransformation()(u_star)

    ecrire("n_iter FORM  = %d" % n_iter)
    for i, p in enumerate(params_names):
        ecrire("%s*          = %.4f" % (p, x_star[i]))
    ecrire("u*           = %s" % [round(v, 4) for v in u_star])
    ecrire("Imp.         = %s" % [round(v, 4) for v in best_result.getImportanceFactors()])
    ecrire("beta         = %.4f" % best_result.getHasoferReliabilityIndex())
    ecrire("Pf           = %.4e" % best_result.getEventProbability())
    return u_star
