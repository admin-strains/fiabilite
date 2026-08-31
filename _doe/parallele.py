r"""Evaluation en parallele : plusieurs solveurs, plusieurs copies du modele.

POURQUOI DES SOUS-PROCESSUS ET NON DES FILS
--------------------------------------------
Digital Structure travaille sur un dossier `.ds` qu'il REECRIT a chaque
evaluation -- `dsCad.txt` est modifie sur place pour y inscrire les valeurs
des variables. Deux evaluations concurrentes dans le meme dossier se
detruisent l'une l'autre. Chaque worker recoit donc SA copie du modele.

C'est aussi pour cela que ce ne sont pas des fils : les bibliotheques du
solveur ne sont pas reentrantes, et la charge MKL doit etre repartie
explicitement (`MKL_NUM_THREADS`), faute de quoi chaque worker croit disposer
de toute la machine et ils se disputent les coeurs.

CE CODE N'A JAMAIS TOURNE
--------------------------
Les workers passaient jusqu'a la phase 5 par `launcher3.py`, une copie du
lanceur qui portait en dur les chemins du poste de l'auteur
(`C:\_workingDir\_SF\test flexion\_lib`). Ce chemin n'existant nulle part
ailleurs, ce chemin de code ne pouvait pas s'executer hors de ce poste --
ce qui explique qu'il n'ait jamais ete couvert par un test, ni meme, selon
toute vraisemblance, execute depuis.

Il etait de surcroit RECOPIE quatre fois : `run_DOE_parallel` et
`run_HF_grid_parallel`, dans chacun des deux scripts d'etude, pour 115 lignes
qui ne differaient que par le nom des sous-dossiers et la facon de ranger le
resultat. Le tronc commun est ici, et il est testable : le lanceur de
processus est un argument.
"""

import json
import os
import shutil
import subprocess
import sys


def _ecrire(message):
    print(message, flush=True)


#: Nombre de fils MKL supposes disponibles sur la machine. Il etait ecrit
#: `32` en dur, deux fois par fichier : sur une machine a 8 coeurs, quatre
#: workers en reclamaient chacun 8, soit 32 fils pour 8 coeurs.
FILS_MKL_DISPONIBLES = 32


def repartir(n_points, n_workers):
    """Les indices des points, distribues en tourniquet.

    Le tourniquet -- et non des tranches contigues -- parce que le cout d'un
    point varie fortement avec sa position dans le domaine : des tranches
    contigues donneraient un worker charge et trois oisifs.

    Retourne une liste de listes, une par worker, les vides comprises.
    """
    n_workers = max(1, min(n_workers, n_points)) if n_points else 1
    lots = [[] for _ in range(n_workers)]
    for i in range(n_points):
        lots[i % n_workers].append(i)
    return lots


def preparer_worker(storage, base_ds, nom_worker, points, params_names):
    """La copie isolee du modele, et la tache a y executer.

    Seuls `dsCad.txt` et `dsLoad.txt` sont copies : ce sont les deux fichiers
    que le solveur reecrit. Le reste du `.ds` (STEP, resultats) est relu
    depuis le modele principal, transmis par `_DOE_MAIN_DS`.
    """
    wds = os.path.join(storage, nom_worker + ".ds")
    os.makedirs(wds, exist_ok=True)
    for fichier in ("dsCad.txt", "dsLoad.txt"):
        shutil.copy2(os.path.join(base_ds, fichier), os.path.join(wds, fichier))
    tache = {"points": [dict({"idx": i}, **{p: float(points[i][p])
                                            for p in params_names})
                        for i in sorted(points)]}
    fichier_tache = os.path.join(wds, "_doe_task.json")
    fichier_sortie = os.path.join(wds, "_doe_out.json")
    with open(fichier_tache, "w") as fh:
        json.dump(tache, fh)
    if os.path.exists(fichier_sortie):
        # une sortie d'un run precedent serait relue comme celle-ci
        os.remove(fichier_sortie)
    return wds, fichier_tache, fichier_sortie


def moissonner_sorties(points, params_names, storage, base_modelname,
                       sous_dossier="_doe_workers", tolerance=1e-9,
                       tracer=_ecrire):
    """Les points qu'un run parallele INTERROMPU avait deja payes.

    Retourne `{indice: resultat}`, a greffer dans `points` avant de relancer
    le pool -- qui saute alors ce qui porte deja `g`.

    LE DEFAUT QUE CELA FERME -- 29/08/2026
    ---------------------------------------
    Le filet de reprise du plan (`save_doe_cache_incremental`, ecrit apres
    CHAQUE point) s'execute DANS le worker, donc dans SA copie du `.ds`. Le
    pere, lui, ne relit que le sien. Mesure, interruption apres 3 points sur
    5 :

        voie sequentielle   filet chez le pere : oui   3 points repris
        voie parallele      filet chez le pere : non   0 point repris
                            (le fichier etait dans _doe_workers/doew0.ds/)

    Sur le Moulin Blanc -- `n0 = 5`, six workers -- une interruption du plan
    coutait donc jusqu'a cinq appels solveur, environ 39 minutes.

    ON VERIFIE, ON NE SUPPOSE PAS
    ------------------------------
    Une sortie de worker ne porte que `{idx: {g, dg_*}}` : la greffer sur le
    seul indice serait exactement le defaut du cache de points libres, ou des
    valeurs se retrouvaient appariees a de mauvaises coordonnees. Le fichier
    de tache, lui, est a cote et porte les PARAMETRES de chaque indice. On
    compare donc les deux, parametre par parametre, et on ne greffe que ce
    qui coincide.

    C'est aussi ce qui rend la moisson sure malgre le menage de
    `preparer_worker` : une sortie d'un autre run n'a aucune raison de porter
    les memes coordonnees, et sera ecartee ici -- avec une ligne pour le dire.
    """
    dossier = os.path.join(storage, base_modelname + ".ds", sous_dossier)
    if not os.path.isdir(dossier):
        return {}
    recoltes, ecartes = {}, 0
    for nom in sorted(os.listdir(dossier)):
        wds = os.path.join(dossier, nom)
        sortie = os.path.join(wds, "_doe_out.json")
        tache = os.path.join(wds, "_doe_task.json")
        if not (os.path.exists(sortie) and os.path.exists(tache)):
            continue
        try:
            rendus = json.load(open(sortie))
            attendus = {p["idx"]: p for p in json.load(open(tache))["points"]}
        except Exception as e:
            tracer("  [MOISSON] %s illisible (%s: %s) -> ignore"
                   % (nom, type(e).__name__, e))
            continue
        for i_str, rendu in rendus.items():
            i = int(i_str)
            attendu = attendus.get(i)
            if attendu is None or i >= len(points):
                ecartes += 1
                continue
            if any(abs(float(attendu[p]) - float(points[i][p])) > tolerance
                   for p in params_names):
                ecartes += 1
                continue
            recoltes[i] = rendu
    if recoltes or ecartes:
        tracer("  [MOISSON] %d point(s) recuperes d'un run parallele "
               "interrompu, %d ecarte(s) (coordonnees differentes)"
               % (len(recoltes), ecartes))
    return recoltes


def evaluer_en_parallele(points, params_names, storage, base_modelname,
                         n_workers, script_etude, repo,
                         sous_dossier="_doe_workers", prefixe="doew",
                         nom_journal="_doe_worker.log", etiquette="DOE PARALLELE",
                         lancer=None, tracer=_ecrire):
    """Evalue `points` (liste de dicts en variables PHYSIQUES) en parallele.

    Retourne `{indice: resultat}` ou `resultat` porte `g` et les `dg_<var>`
    rendus par le worker.

    UN POINT QUI PORTE DEJA `g` EST SAUTE -- comme dans `evaluer_plan`.
    C'est tout l'interet de la reprise, et cela a manque jusqu'au 29/08/2026 :
    l'appelant greffait les points d'un plan interrompu, le journal annoncait
    « 3/5 points repris du cache -> autant de SOCP evites », puis le worker
    RECONSTRUISAIT son dictionnaire a partir des seuls parametres physiques
    (`_wSOL = [{p: float(pt[p]) for p in params_names}]`) et les repayait
    tous. Mesure, filet de 3 points sur 5 :

        voie sequentielle   2 points payes sur 5
        voie parallele      5 points payes sur 5

    Sur le Moulin Blanc (`n0 = 5`, six workers, `config_is_identical`), cela
    fait jusqu'a cinq appels solveur -- environ 39 minutes -- repayes en
    annoncant le contraire.

    `lancer(argv, env, cwd, journal)` est le lanceur de processus ; il est un
    argument pour que ce module soit testable sans solveur -- c'est la seule
    raison pour laquelle ces 115 lignes, jamais executees depuis leur
    ecriture, peuvent enfin etre verifiees.
    """
    lancer = lancer or _lancer_sous_processus
    base_ds = os.path.join(storage, base_modelname + ".ds")
    n_points = len(points)

    # Ce qui est deja paye ne repart pas au solveur. On garde les INDICES
    # globaux : les workers rendent un dictionnaire indexe, et le controle de
    # completude en fin de fonction porte sur `range(n_points)`.
    connus = {i: {c: v for c, v in points[i].items()
                  if c == "g" or c.startswith("dg_")}
              for i in range(n_points) if "g" in points[i]}
    a_payer = [i for i in range(n_points) if i not in connus]
    if connus:
        tracer("  [%s] %d/%d point(s) deja connus (cache partiel) : autant de "
               "SOCP evites" % (etiquette, len(connus), n_points))
    if not a_payer:
        return dict(connus)

    lots = [[a_payer[k] for k in lot] for lot in repartir(len(a_payer), n_workers)]
    n_reels = sum(1 for lot in lots if lot)
    fils = max(1, FILS_MKL_DISPONIBLES // max(1, n_reels))
    tracer("  [%s] %d pts -> %d workers (MKL=%d threads/worker)"
           % (etiquette, len(a_payer), n_reels, fils))

    processus = []
    for w, idxs in enumerate(lots):
        if not idxs:
            continue
        nom_worker = os.path.join(base_modelname + ".ds", sous_dossier,
                                  "%s%d" % (prefixe, w))
        wds, _tache, sortie = preparer_worker(
            storage, base_ds, nom_worker,
            {i: points[i] for i in idxs}, params_names)
        env = dict(os.environ,
                   _DOE_WORKER=os.path.join(wds, "_doe_task.json"),
                   _DOE_OUT=sortie,
                   _DOE_WORKER_MODELNAME=nom_worker,
                   _DOE_MAIN_DS=base_ds,
                   _FIAB_LOG_REDIRECTED="1",
                   MKL_NUM_THREADS=str(fils), OMP_NUM_THREADS=str(fils))
        tracer("    -> worker %d: %d points" % (w, len(idxs)))
        proc = lancer([sys.executable, os.path.join(repo, "launcher.py"),
                       "--garder-cwd", script_etude],
                      env, wds, os.path.join(wds, nom_journal))
        processus.append((proc, sortie, w, idxs))

    for proc, _sortie, w, _idxs in processus:
        rc = proc.wait()
        tracer("    <- worker %d fini (rc=%d)" % (w, rc))

    resultats = dict(connus)
    for _proc, sortie, w, idxs in processus:
        if not os.path.exists(sortie):
            raise RuntimeError(
                "[%s] worker %d sans sortie %s (voir %s). Un worker qui meurt "
                "en silence rendrait un plan incomplet sans le dire."
                % (etiquette, w, sortie, nom_journal))
        for i_str, d in json.load(open(sortie)).items():
            resultats[int(i_str)] = d
        tracer("    collecte worker %d: %d pts" % (w, len(idxs)))

    manquants = sorted(set(range(n_points)) - set(resultats))
    if manquants:
        raise RuntimeError(
            "[%s] %d point(s) sans resultat : %s. Un plan troue passerait "
            "pour un plan complet." % (etiquette, len(manquants), manquants))
    return resultats


class _ProcessusJournalise:
    """Un sous-processus et son journal, fermes ensemble.

    L'original gardait les descripteurs dans une liste parallele et les
    fermait a la main dans une seconde boucle : un `raise` entre les deux les
    laissait ouverts, et sous Windows un fichier ouvert par un processus mort
    reste verrouille.
    """

    def __init__(self, argv, env, cwd, chemin_journal):
        self._journal = open(chemin_journal, "w")
        self._proc = subprocess.Popen(argv, env=env, stdout=self._journal,
                                      stderr=subprocess.STDOUT, cwd=cwd)

    def wait(self, *a, **kw):
        try:
            return self._proc.wait(*a, **kw)
        finally:
            if not self._journal.closed:
                self._journal.close()


def _lancer_sous_processus(argv, env, cwd, chemin_journal):
    return _ProcessusJournalise(argv, env, cwd, chemin_journal)


def evaluer_plan_en_parallele(SOL, params_names, storage, base_modelname,
                              n_workers, script_etude, repo, lancer=None):
    """Un plan d'experiences reparti, puis RECOPIE dans `SOL`.

    Les workers rendent un dictionnaire indexe ; l'appelant, lui, travaille
    sur `SOL`, la liste qu'il a construite. Cette fonction fait le pont, en
    place -- `SOL` est modifiee et rendue.

    `lancer` remonte jusqu'ici depuis le 29/08/2026 : la couture de test
    s'arretait a `evaluer_en_parallele`, alors que les deux POINTS D'ENTREE
    reels sont cette fonction et sa voisine. Un module dont le commentaire
    d'en-tete dit « CE CODE N'A JAMAIS TOURNE » ne peut pas se permettre une
    couture qui s'arrete un niveau trop haut.

    Un gradient ABSENT est recopie tel quel, a `None` : c'est ce que le
    solveur a dit. Le remplacer par zero affirmerait un etat limite plat.
    `_doe/plan.assembler_plan` decide ensuite quoi en faire, selon
    `exclure_points_sans_gradient`.
    """
    resultats = evaluer_en_parallele(SOL, params_names, storage,
                                     base_modelname, n_workers,
                                     script_etude=script_etude, repo=repo,
                                     lancer=lancer)
    for i, rendu in resultats.items():
        SOL[i]["g"] = rendu["g"]
        for nom in params_names:
            SOL[i]["dg_%s" % nom] = rendu.get("dg_%s" % nom)
    return SOL


def evaluer_points_en_parallele(u_points, dist, params_names, storage,
                                modelname, n_workers, script_etude, repo,
                                lancer=None, config_identique=False):
    """Des points de l'espace STANDARD, repartis sur plusieurs solveurs.

    Le solveur ne connait que les variables physiques : les points passent
    par la transformation isoprobabiliste avant d'etre confies au pool.

    Retourne la liste des `g`, DANS L'ORDRE de `u_points` -- les workers
    rendent un dictionnaire indexe, et l'appelant attend une grille.

    LE FILET, JUMEAU DE CELUI DU PLAN -- 29/08/2026
    ------------------------------------------------
    Les workers de grille ecrivent dans `_hf_workers/`, que personne ne
    relisait : une interruption perdait tout ce qu'ils avaient paye, comme
    ceux du plan avant `moissonner_sorties`. C'est atteignable sur la vraie
    etude -- `moulin_blanc.toml` porte `do_custom_hf = true` et six workers --
    et vaut 2 h 30 pour une grille libre de vingt points.

    La moisson se fait ICI et non dans la grille, parce que c'est ici que les
    points existent en variables PHYSIQUES : c'est sous cette forme que les
    fichiers de tache des workers les portent, donc la seule ou la
    verification des coordonnees ait un sens.

    Elle n'a lieu que si l'etude accepte de reutiliser du calcul
    (`config_identique`) : `config_is_identical = false` veut dire
    « recalcule », pas « recalcule si tu ne trouves rien ».
    """
    # `T_inv(list(u))` et non `T_inv(ot.Point(list(u)))` : OpenTURNS accepte
    # une sequence, et ce module n'a alors AUCUNE dependance a OpenTURNS --
    # il ne fait que du sous-processus et du fichier. Verifie egal.
    T_inv = dist.getInverseIsoProbabilisticTransformation()
    points = []
    for u in u_points:
        x = T_inv(list(u))
        points.append({p: float(x[j]) for j, p in enumerate(params_names)})
    if config_identique:
        # Greffe : le pool saute ensuite ce qui porte deja `g`.
        for i, rendu in moissonner_sorties(points, params_names, storage,
                                           modelname,
                                           sous_dossier="_hf_workers").items():
            points[i].update(rendu)
    resultats = evaluer_en_parallele(
        points, params_names, storage, modelname, n_workers,
        script_etude=script_etude, repo=repo,
        sous_dossier="_hf_workers", prefixe="hfw",
        nom_journal="_hf_worker.log", etiquette="HF GRID PARALLELE",
        lancer=lancer)
    return [resultats[i]["g"] for i in range(len(points))]


def fabrique_memoisee(fabriquer, modelname_defaut, chemin_pour, **reglages):
    """Un solveur PAR MODELE, construit une fois puis reutilise.

    Un worker de plan d'experiences parallele travaille sur SA copie du
    `.ds` : le nom du modele lui est impose par le processus pere, d'ou le
    parametre.

    La memoisation n'est pas une optimisation : le solveur porte un COMPTEUR
    D'APPELS, et en reconstruire un remettrait ce compteur a zero. C'est lui
    qui dit, en fin de run, combien d'appels ont reellement ete payes.
    """
    cache = {}

    def solveur(nom=None):
        nom = nom or modelname_defaut
        if nom not in cache:
            cache[nom] = fabriquer(chemin_ds=chemin_pour(nom), **reglages)
        return cache[nom]

    solveur.cache = cache
    return solveur
