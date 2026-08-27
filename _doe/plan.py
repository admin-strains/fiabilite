r"""ACTION `plan` : ou depenser les premiers appels solveur.

Le plan d'experiences initial est le seul moment ou l'on choisit des points
SANS rien savoir de l'etat limite. Tout le reste -- enrichissement, grille,
FORM -- part de lui. Un plan mal reparti se paie ensuite en points
d'enrichissement, c'est-a-dire en heures.

D'ou le tirage par hypercube latin recuit (`SimulatedAnnealingLHS` +
`SpaceFillingMinDist`) : il maximise la distance minimale entre points, au
lieu de se contenter d'une repartition marginale correcte.

CE QUE CE MODULE NE FAIT PAS
-----------------------------
Il n'appelle pas le solveur. Il TIRE des points, il TRIE ceux qui sont
exploitables, il AUGMENTE le plan par developpement de Taylor -- l'evaluation
elle-meme est dans `_doe/evaluation.py`, en un seul exemplaire.

DEUX DEFAUTS QUE CES FONCTIONS PORTENT DANS LEUR HISTOIRE
----------------------------------------------------------
1. **Le domaine de tirage etait code en dur.** `ot.Uniform(-7.5, 7.5)` dans la
   flexion pure, `eff_bounds` sur le Moulin Blanc, dans deux `build_DOE` par
   ailleurs identiques. Borner le domaine -- ce qu'on fait quand le solveur
   meurt sur des points extremes -- n'avait donc d'effet que d'un cote.
2. **Un point sans gradient cassait le plan.** Digital Structure rend parfois
   `Sensitivity = {fy1: None, fy2: None}` ; le 26/08/2026, cela a fait partir
   un plan en `TypeError` APRES cinq appels au solveur. Le `.get(..., 0.0)`
   cense proteger ne protegeait rien : la clef EXISTE, avec la valeur None.
   Et tant mieux -- un gradient nul affirmerait que l'etat limite est plat en
   ce point, ce que le metamodele ajusterait.
"""

import numpy as np
import openturns as ot


def _ecrire(message):
    print(message, flush=True)


def tirer_plan_lhs(dist_X, n_doe, bornes_min, bornes_max):
    """Un hypercube latin recuit sur le domaine de l'etude.

    Retourne `(U_doe, X_doe, xt)` : les points en variables normees (objet
    OpenTURNS), les memes en variables physiques, et le tableau numpy que le
    metamodele consomme.

    `bornes_min` / `bornes_max` sont des listes, une valeur par variable. Les
    passer -- plutot que d'ecrire +/- 7,5 -- est ce qui rend le bornage du
    domaine effectif.
    """
    n_var = len(bornes_min)
    T_inv = dist_X.getInverseIsoProbabilisticTransformation()
    dist_U = ot.JointDistribution(
        [ot.Uniform(bornes_min[i], bornes_max[i]) for i in range(n_var)])
    lhs = ot.LHSExperiment(dist_U, n_doe)
    recuit = ot.SimulatedAnnealingLHS(lhs, ot.SpaceFillingMinDist())
    U_doe = recuit.generate()
    return U_doe, T_inv(U_doe), np.array(U_doe)


def tracer_plan(U_doe, tracer=_ecrire):
    """Le plan, sous une forme recopiable dans un fichier d'etude.

    Seize decimales : un plan retranscrit a moins n'est PAS le meme plan, et
    la comparaison de deux runs perd son sens.
    """
    tracer("U_doe_fixed = ot.Sample([")
    for i in range(U_doe.getSize()):
        vals = []
        for j in range(U_doe.getDimension()):
            v = U_doe[i][j]
            vals.append(" %.16f" % v if v >= 0 else "%.16f" % v)
        tracer("    [%s]," % ", ".join(vals))
    tracer("])")


def points_avec_gradient(SOL, params_names, U_doe, exclure, tracer=_ecrire):
    """Les indices des points du plan exploitables par le metamodele.

    Un point sans gradient n'est pas une anomalie de forme : c'est un point ou
    Digital Structure n'a rien rendu. Deux conduites possibles, et `exclure`
    tranche :

    * `True` -- on l'ECARTE : le plan retrecit, mais rien de faux n'y entre ;
    * `False` -- on le CONSERVE, et son gradient sera fabrique a zero. Le
      journal le dit en toutes lettres, parce que ce zero est un mensonge :
      il affirme que l'etat limite est plat la ou on ne sait rien.

    Leve si `exclure` est vrai et qu'il ne reste RIEN : il n'y a alors pas de
    plan a ajuster, et le dire tot vaut mieux qu'une erreur de forme trois
    fonctions plus loin.
    """
    n_doe = len(SOL)
    complets = [i for i in range(n_doe)
                if all(SOL[i].get('dg_%s' % p) is not None for p in params_names)]
    manquants = [i for i in range(n_doe) if i not in complets]
    if not manquants:
        return complets

    for i in manquants:
        tracer("  [PLAN] point %d SANS GRADIENT (le solveur n'en rend aucun) "
               "u=%s g=%+.6f -- %s"
               % (i, [round(float(v), 4) for v in U_doe[i]], SOL[i]['g'],
                  "ECARTE" if exclure
                  else "CONSERVE avec un gradient FABRIQUE a 0"))
    if not exclure:
        return list(range(n_doe))

    tracer("  [PLAN] %d point(s) ecarte(s) : le plan passe de %d a %d."
           % (len(manquants), n_doe, len(complets)))
    if not complets:
        raise RuntimeError(
            "aucun point du plan d'experiences n'a de gradient : il n'y a "
            "rien a ajuster. Verifier que les regions de sensibilite du "
            "modele correspondent aux variables.")
    return complets


def augmenter_par_taylor(xt, yt, all_grad, eps, n_var, tracer=_ecrire):
    """Ajoute au plan des points VIRTUELS, obtenus au premier ordre.

    Pour chaque point reel et chaque direction, un voisin a distance `eps`
    dont la valeur vient du developpement de Taylor. Cela donne au PCK -- qui
    n'exploite pas les gradients directement -- une information de pente sans
    payer un appel solveur de plus.

    `eps` trop grand fait mentir le developpement ; trop petit, les deux
    points sont numeriquement confondus et le krigeage se retrouve avec une
    matrice de covariance singuliere. `eps = 0` desactive.
    """
    if not eps:
        return xt, yt, all_grad
    n_reels = len(xt)
    for i_pt in range(n_reels):
        for i_dim in range(n_var):
            u_virt = xt[i_pt] + eps * np.eye(n_var)[i_dim]
            y_virt = yt[i_pt, 0] + eps * all_grad[i_pt, i_dim]
            xt = np.vstack([xt, [u_virt]])
            yt = np.vstack([yt, [[y_virt]]])
            all_grad = np.vstack([all_grad, [all_grad[i_pt]]])
    tracer("  [Taylor DOE] %d HF + %d virtuels = %d pts"
           % (n_reels, n_reels * n_var, len(xt)))
    return xt, yt, all_grad


def tirer_points_de_depart(n_sp, bornes_min, bornes_max):
    """Les points de depart du FORM multimodal.

    Meme tirage que le plan, meme domaine : un point de depart hors du domaine
    ou l'etat limite a ete evalue partirait explorer une extrapolation du
    metamodele.
    """
    n_var = len(bornes_min)
    dist_U = ot.JointDistribution(
        [ot.Uniform(bornes_min[i], bornes_max[i]) for i in range(n_var)])
    recuit = ot.SimulatedAnnealingLHS(ot.LHSExperiment(dist_U, n_sp),
                                      ot.SpaceFillingMinDist())
    return np.array(recuit.generate())
