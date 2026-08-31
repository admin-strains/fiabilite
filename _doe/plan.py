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

import os

import numpy as np
import openturns as ot

import doe as _cache_doe
import parallele as _parallele


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


def assembler_plan(SOL, complets, xt, params_names):
    """Le plan reduit aux points retenus : `(xt, yt, all_grad)`.

    `complets` vient de `points_avec_gradient` : ce sont les indices des
    points que l'etude accepte de garder. Les autres sont ecartes ICI, et
    non plus tard -- un point sans gradient qui reste dans `xt` mais pas
    dans `all_grad` donnerait deux tableaux de longueurs differentes.

    `or 0.0` porte le gradient FABRIQUE : quand `exclure_points_sans_gradient`
    est faux, un point sans gradient est conserve avec un gradient nul. Ce
    zero AFFIRME que l'etat limite est plat -- `points_avec_gradient`
    l'annonce dans le journal, et c'est un reglage d'etude, pas un defaut
    silencieux.
    """
    yt = np.array([SOL[i]["g"] for i in complets]).reshape(-1, 1)
    all_grad = np.array([[SOL[i].get("dg_%s" % p) or 0.0 for p in params_names]
                         for i in complets], dtype=float)
    return xt[complets], yt, all_grad


def journaliser_plan(yt, all_grad, tracer=_ecrire):
    """Le plan, imprime sous une forme RECOPIABLE en Python.

    Ce n'est pas de la decoration : ces deux tableaux permettent de rejouer
    une etude -- ajustement, enrichissement, FORM -- sans rappeler le
    solveur. Sur le Moulin Blanc, un plan de 24 points represente environ
    trois heures ; les relire d'un journal coute zero.

    D'ou les 16 decimales sur `yt` : tronquer ferait de la copie une
    approximation, et l'interet est justement de retrouver les memes nombres.
    """
    tracer("yt_doe = [")
    for k in range(len(yt)):
        tracer("    %.16f," % (yt[k][0],))
    tracer("]")
    tracer("all_grad_doe = [")
    for k in range(len(all_grad)):
        tracer("    [" + ", ".join("%.10f" % v for v in all_grad[k]) + "],")
    tracer("]")


def _moisson_par_defaut(fichier_cache, tracer):
    """Ou chercher ce qu'un run parallele interrompu avait deja paye.

    Les workers ecrivent dans `<modele>.ds/_doe_workers/` -- et le cache du
    plan, lui, est `<modele>.ds/doe_cache.json`. Le dossier du modele est donc
    deja connu ici, et l'etude n'a rien a recabler : elle passerait deux lignes
    identiques de chaque cote, ce qui est precisement ce qu'on retire.

    `moissonner` reste surchargeable -- les tests s'en servent.
    """
    dossier = os.path.dirname(os.path.abspath(fichier_cache))
    base = os.path.basename(dossier)
    racine = os.path.dirname(dossier)
    nom = base[:-3] if base.endswith(".ds") else base

    def moissonner(SOL, params_names):
        return _parallele.moissonner_sorties(SOL, params_names, racine, nom,
                                             tracer=tracer)
    return moissonner


# --------------------------------------------------------------------------- #
# L'ENCHAINEMENT COMPLET : du tirage au plan pret a nourrir un metamodele     #
# --------------------------------------------------------------------------- #
def construire_plan_initial(cfg, n_doe, *, dist_jointe, params_names,
                            bornes_min, bornes_max, fichier_cache, signature,
                            executer_plan, executer_en_parallele=None,
                            moissonner=None, journaliser=None, tracer=_ecrire):
    """Le plan d'experiences initial : `n_doe` appels solveur, ou zero.

    C'etait `build_DOE`, cinquante-deux lignes ecrites dans les DEUX etudes.
    Ce qui restait a l'etude n'etait pas l'enchainement -- il est identique --
    mais le nom du modele et la facon d'appeler le solveur. Ce sont donc les
    seules choses qu'on lui demande encore, sous forme d'appelables.

    Retourne TOUJOURS `(xt, yt, all_grad)`.

    UNE ARITE, PAS DEUX. `build_DOE` rendait tantot un triplet, tantot le seul
    `xt` -- et son propre commentaire gardait la trace de ce que cela avait
    coute : « l'ancienne branche faisait `xt = build_DOE()` et recevait un
    TRIPLET -- xt devenait un tuple, silencieusement ». En haute fidelite
    pure il n'y a ni valeurs ni gradients a rendre : ce sont deux `None`, pas
    une signature differente.

    `eval_hf` a disparu avec cette unification. Il valait `False` en un seul
    site d'appel, et ce site etait deja garde par `do_HF` -- les deux
    conditions ne se sont jamais separees.

    LES DEUX CACHES, QUI NE FONT PAS LA MEME CHOSE
    -----------------------------------------------
    * le cache COMPLET (`load_doe_cache`) rend un plan entier deja paye, et
      la fonction sort aussitot ;
    * le cache PARTIEL (`charger_doe_partiel`) rend les points d'un plan
      INTERROMPU, greffes dans `SOL` pour que le solveur ne les repaie pas.
      Sur la voie PARALLELE ce filet-la est ecrit par les workers, dans LEUR
      copie du `.ds` : `moissonner(SOL)` va l'y chercher.
      Il verifie les coordonnees qu'il rend (`xt_attendu`) : sans cela, un
      plan retire d'ailleurs serait apparie aux mauvais points.

    Le cache complet n'est consulte qu'en presence d'un metamodele : en HF
    pur, `xt` seul ne se relit pas d'un fichier qui porte aussi `yt`.

    Les deux caches sont clefs sur `n_doe`. `build_DOE` les clefait sur `n0`
    quel que soit le `n_doe` demande ; les deux sites d'appel prenaient le
    defaut `n_doe=n0`, donc rien ne change -- mais un plan d'une autre taille
    n'ira plus relire le cache d'un plan de taille n0.
    """
    n_var = len(params_names)
    if not cfg.do_HF:
        depuis_cache = _cache_doe.load_doe_cache(
            fichier_cache, n_doe, cfg.config_is_identical, signature=signature)
        if depuis_cache is not None:
            return depuis_cache

    dist_X = dist_jointe()
    U_doe, X_doe, xt = tirer_plan_lhs(dist_X, n_doe, bornes_min, bornes_max)
    if cfg.print_DOE:
        tracer_plan(U_doe, tracer=tracer)
    if cfg.do_HF:
        # Haute fidelite pure : le plan sert de POINTS, pas de donnees
        # d'apprentissage. Aucun appel solveur ici -- ils viendront un par un.
        return xt, None, None

    SOL = [{params_names[j]: X_doe[i][j] for j in range(n_var)}
           for i in range(n_doe)]

    # Reprise d'un plan interrompu : la greffe est dans `_cache/doe.py`, avec
    # ce qu'une interruption coutait avant qu'elle existe.
    if cfg.config_is_identical:
        _cache_doe.greffer_reprise(
            SOL,
            _cache_doe.charger_doe_partiel(fichier_cache, n_doe,
                                           signature=signature,
                                           xt_attendu=xt),
            params_names)

    if cfg.n_workers_DOE and cfg.n_workers_DOE > 1:
        # Le filet du parallele : les sorties des workers qu'un run precedent
        # avait menes au bout. Le leur est ecrit dans LEUR copie du `.ds`, que
        # le pere ne relit pas -- d'ou cette moisson, verifiee coordonnee par
        # coordonnee. Le pool saute ensuite ce qui porte deja `g`.
        if cfg.config_is_identical:
            recolte = (moissonner or _moisson_par_defaut(fichier_cache, tracer))
            for i, rendu in recolte(SOL, params_names).items():
                SOL[i].update(rendu)
        SOL = executer_en_parallele(SOL, cfg.n_workers_DOE)
    else:
        SOL = executer_plan(SOL)

    complets = points_avec_gradient(SOL, params_names, U_doe,
                                    cfg.exclure_points_sans_gradient,
                                    tracer=tracer)
    xt, yt, all_grad = assembler_plan(SOL, complets, xt, params_names)

    if journaliser is not None:
        for i in complets:
            journaliser(list(U_doe[i]), list(X_doe[i]), SOL[i]['g'], phase="DOE")
    if cfg.print_DOE:
        journaliser_plan(yt, all_grad, tracer=tracer)
    _cache_doe.save_doe_cache(fichier_cache, n_doe, xt, yt, all_grad,
                              signature=signature)
    if cfg.do_PCK:
        xt, yt, all_grad = augmenter_par_taylor(xt, yt, all_grad,
                                                cfg.eps_taylor, n_var,
                                                tracer=tracer)
    return xt, yt, all_grad
