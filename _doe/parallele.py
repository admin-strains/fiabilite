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


def evaluer_en_parallele(points, params_names, storage, base_modelname,
                         n_workers, script_etude, repo,
                         sous_dossier="_doe_workers", prefixe="doew",
                         nom_journal="_doe_worker.log", etiquette="DOE PARALLELE",
                         lancer=None, tracer=_ecrire):
    """Evalue `points` (liste de dicts en variables PHYSIQUES) en parallele.

    Retourne `{indice: resultat}` ou `resultat` porte `g` et les `dg_<var>`
    rendus par le worker.

    `lancer(argv, env, cwd, journal)` est le lanceur de processus ; il est un
    argument pour que ce module soit testable sans solveur -- c'est la seule
    raison pour laquelle ces 115 lignes, jamais executees depuis leur
    ecriture, peuvent enfin etre verifiees.
    """
    lancer = lancer or _lancer_sous_processus
    base_ds = os.path.join(storage, base_modelname + ".ds")
    n_points = len(points)
    lots = repartir(n_points, n_workers)
    n_reels = sum(1 for lot in lots if lot)
    fils = max(1, FILS_MKL_DISPONIBLES // max(1, n_reels))
    tracer("  [%s] %d pts -> %d workers (MKL=%d threads/worker)"
           % (etiquette, n_points, n_reels, fils))

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

    resultats = {}
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
