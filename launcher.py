r"""
Lanceur portable des etudes de fiabilite.

Remplace les dix lanceurs recopies (launcher.py, launcher2.py, launcher3.py,
launcher_moulin_blanc*.py, launcher_cantilever_s.py, ...), qui differaient
entre eux par des chemins absolus et une ligne de print.

    python launcher.py Moulinblanc/AC3_moulinblanc.py
    python launcher.py pure_flexion/AC3_pure_flexion.py

Aucun chemin en dur : la racine du depot vient de __file__, celle de Digital
Structure de la variable d'environnement DS_ROOT ou d'une detection
automatique.

    set DS_ROOT=D:\autre\emplacement\front


POURQUOI CE FICHIER EXISTE (et pourquoi l'ordre des imports compte)
-------------------------------------------------------------------
OpenTURNS doit etre importe AVANT que les repertoires de DLL de Digital
Structure ne soient ajoutes au chemin de recherche. Sinon l'import
d'OpenTURNS echoue sur :

    ImportError: DLL load failed while importing _common:
    La procedure specifiee est introuvable.

Les anciens lanceurs attribuaient cela a un "conflit MKL". Mesure le
25/08/2026, la cause est ailleurs -- trois DLL portent le meme nom des deux
cotes, avec des contenus differents :

    libblas.dll     OpenTURNS 0,75 Mo   |  DS 0,10 Mo
    liblapack.dll   OpenTURNS 14,4 Mo   |  DS 0,17 Mo
    zlib1.dll       OpenTURNS 0,10 Mo   |  DS 0,13 Mo

OpenTURNS embarque un OpenBLAS complet construit avec MinGW (d'ou ses
dependances libgcc_s_seh-1.dll et libssp-0.dll) ; DS expose des DLL MSVC de
meme nom mais bien plus petites. Des que bin\\ de DS passe devant, libot.dll
resout liblapack.dll vers la version DS, qui n'exporte pas les symboles
attendus.

Importer OpenTURNS d'abord charge ses DLL avant que la collision ne puisse
se produire. Le contournement est verifie par
tests/test_60_environnement.py, qui echoue si la contrainte disparait ou si
elle cesse d'etre necessaire -- dans ce dernier cas, supprimer ce
contournement plutot que de le trainer.
"""

import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))

#: sous-repertoires de DLL a exposer, relatifs a la racine DS
DLL_SUBDIRS = (
    r"STRAINS\rupt\core\bin",
    r"STRAINS\rupt\core",
    r"STRAINS\common\Dll",
    r"STRAINS\rupt\core\bin\meshgems",
    r"STRAINS\rupt\core\bin\mosek",
)


def find_ds_root(obligatoire=True):
    """Racine contenant le paquet STRAINS : DS_ROOT, sinon detection.

    `obligatoire=False` rend None au lieu de sortir quand Digital Structure
    est introuvable.

    POURQUOI CETTE OPTION EXISTE
    -----------------------------
    Toutes les etudes n'ont pas besoin de Digital Structure : celles qui
    tournent sur le solveur ANALYTIQUE n'en touchent pas une ligne. Or ce
    lanceur exigeait DS avant meme de savoir quelle etude on lui demandait,
    et sortait en code 1. Consequence mesuree le 31/08/2026 : le run
    bout-en-bout de `pure_flexion` -- pourtant analytique -- ne pouvait pas
    tourner en integration continue, et `test_103_grille_bout_en_bout` y
    produisait cinq erreurs.

    Le defaut n'etait pas rattrapable cote test : c'est le lanceur qui
    posait la condition.

    L'ABSENCE RESTE BRUYANTE. On ne fait pas semblant que DS est la : le
    lanceur l'annonce, et une etude qui en a reellement besoin echouera a
    l'import de `STRAINS` avec le message de Python. Ce qui disparait, c'est
    le refus PREALABLE, qui condamnait aussi ce qui n'en depend pas.
    """
    env = os.environ.get("DS_ROOT")
    if env:
        if os.path.isdir(os.path.join(env, "STRAINS")):
            return os.path.abspath(env)
        # Un DS_ROOT pose EXPRES et faux reste une erreur, meme en mode
        # tolerant : c'est une intention explicite qui n'aboutit pas.
        raise SystemExit(
            "DS_ROOT vaut %r mais ne contient pas de dossier STRAINS.\n"
            "Attendu : le dossier 'front' du workspace Digital Structure." % env)

    candidats = []
    parent = os.path.dirname(REPO)
    for _ in range(3):                      # depot, workspace, au-dessus
        candidats.append(os.path.join(parent, "front"))
        parent = os.path.dirname(parent)
    candidats.append(r"C:\workspace\front")

    for c in candidats:
        if os.path.isdir(os.path.join(c, "STRAINS")):
            return os.path.abspath(c)

    message = ("Digital Structure introuvable. Cherche dans :\n  "
               + "\n  ".join(candidats)
               + "\n\nDefinir DS_ROOT sur le dossier 'front' du workspace :\n"
                 "  set DS_ROOT=C:\\workspace\\front")
    if obligatoire:
        raise SystemExit(message)
    print("[launcher] " + message.replace("\n", "\n[launcher] "), file=sys.stderr)
    print("[launcher] --> on continue SANS : une etude sur solveur analytique "
          "n'en a pas besoin.", file=sys.stderr)
    return None


def check_python(ds_root=True):
    """Les .pyd de DS sont lies a python310.dll : 3.10 obligatoire.

    `ds_root` falsy = Digital Structure n'est pas la. La contrainte tombe
    alors, puisqu'elle porte sur des modules compiles qu'on ne chargera
    pas : une etude analytique tourne sous n'importe quel Python 3.9+.
    """
    if not ds_root:
        return
    if sys.version_info[:2] != (3, 10):
        raise SystemExit(
            "Python %d.%d detecte, or les modules compiles de Digital Structure\n"
            "sont lies a python310.dll et ne se chargeront pas.\n"
            "Utiliser un interpreteur 3.10 (voir requirements/studies.txt)."
            % sys.version_info[:2])


def setup(ds_root):
    """Prepare l'environnement d'execution. L'ordre des blocs est significatif."""
    # 1. OpenTURNS D'ABORD -- voir l'en-tete de ce fichier.
    try:
        import openturns  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "OpenTURNS est introuvable : %s\n"
            "  python -m pip install -r requirements/studies.txt" % exc)

    # 2. seulement ensuite, les DLL de Digital Structure -- s'il est la
    manquants = []
    for sub in (DLL_SUBDIRS if ds_root else ()):
        d = os.path.join(ds_root, sub)
        if os.path.isdir(d):
            os.add_dll_directory(d)
        else:
            manquants.append(d)
    if manquants:
        print("[launcher] repertoires de DLL absents (ignores) :", file=sys.stderr)
        for d in manquants:
            print("           " + d, file=sys.stderr)

    # 3. chemins d'import : DS s'il est la, puis les modules du depot.
    # `ds_root` a None quand DS est absent -- l'inserer tel quel poserait
    # None dans sys.path, ou il fait lever chaque import suivant.
    for p in ((ds_root,) if ds_root else ()) + (
              REPO, os.path.join(REPO, "_lib"), os.path.join(REPO, "_model"),
              os.path.join(REPO, "_cache"),
              os.path.join(REPO, "_reliability"),
              os.path.join(REPO, "_config"),
              os.path.join(REPO, "_etapes"),
              os.path.join(REPO, "_surrogate"),
              os.path.join(REPO, "_doe"),
              os.path.join(REPO, "solver")):
        if p not in sys.path:
            sys.path.insert(0, p)


def run(script, extra=(), garder_cwd=False):
    """Execute l'etude avec __name__ == '__main__'.

    Les scripts AC ont 98 % de leur code dans `if __name__ == '__main__':` :
    les importer ne fait rien. Cette execution par exec est donc obligatoire
    tant que la phase 3 du plan de nettoyage n'a pas sorti ce code de main.

    `extra` est transmis via sys.argv, pour les outils qui prennent des
    arguments (tools/solve_one.py).

    `garder_cwd` laisse le repertoire courant tel que l'appelant l'a pose.
    C'est ce dont ont besoin les workers de DOE parallele : chacun travaille
    dans sa copie isolee du modele, et se placer tous dans le dossier de
    l'etude les ferait ecrire leurs fichiers de debogage au meme endroit.
    """
    script = os.path.abspath(script)
    if not os.path.isfile(script):
        raise SystemExit("Etude introuvable : %s" % script)

    dossier = os.path.dirname(script)
    if dossier not in sys.path:
        sys.path.insert(0, dossier)

    sys.argv = [script] + list(extra)
    if not garder_cwd:
        os.chdir(dossier)
    with open(script, "r", encoding="utf-8", errors="replace") as fh:
        source = fh.read()
    exec(compile(source, script, "exec"), {"__name__": "__main__", "__file__": script})


def check(ds_root, etude=None):
    """--check : valide l'installation sans lancer un seul calcul.

    Avec une etude en argument, execute aussi son EN-TETE (imports +
    INITCATALOG) en posant __name__ != "__main__" : le bloc de calcul, qui
    represente 98 % du fichier, n'est donc pas atteint.
    """
    import openturns as ot
    from STRAINS.rupt.core import CetSOLV  # noqa: F401
    import api

    print("[check] openturns %s" % ot.__version__)
    print("[check] Digital Structure : modules compiles chargeables")
    print("[check] librairie du depot : %d fonctions exposees"
          % len([n for n in dir(api) if not n.startswith("_")]))
    for mod in ("smt", "autograd", "sklearn", "matplotlib", "psutil", "threadpoolctl"):
        __import__(mod)
    print("[check] pile des etudes complete")

    if etude:
        chemin = os.path.abspath(etude)
        if not os.path.isfile(chemin):
            raise SystemExit("Etude introuvable : %s" % chemin)
        dossier = os.path.dirname(chemin)
        if dossier not in sys.path:
            sys.path.insert(0, dossier)
        os.environ.setdefault("_FIAB_LOG_REDIRECTED", "1")   # matplotlib en Agg
        with open(chemin, "r", encoding="utf-8", errors="replace") as fh:
            source = fh.read()
        espace = {"__name__": "_check_pas_main", "__file__": chemin}
        exec(compile(source, chemin, "exec"), espace)
        print("[check] en-tete de %s : %d symboles definis avant __main__"
              % (os.path.basename(chemin), len(espace)))

    print("[check] OK -- installation utilisable")


def main(argv):
    argv = list(argv)
    garder_cwd = "--garder-cwd" in argv
    if garder_cwd:
        argv.remove("--garder-cwd")

    if len(argv) < 2:
        # Le nom REELLEMENT tape : `python launcher.py`, ou la commande
        # `fiabilite` installee par `pyproject.toml`. Un message qui nomme
        # l'autre envoie le lecteur essayer une commande qu'il n'a pas.
        moi = os.path.basename(argv[0])
        moi = "fiabilite" if moi.startswith("fiabilite") else "python " + moi
        raise SystemExit(
            "usage : %s [--garder-cwd] <etude.py>\n"
            "        %s --check [etude.py]" % (moi, moi))

    # `--check` repond a la question « mon installation est-elle bonne ? » :
    # il exige donc Digital Structure et refuse net s'il manque. Un RUN, lui,
    # tolere son absence -- une etude sur solveur analytique n'en a pas
    # besoin, et le refus prealable interdisait de la jouer en integration
    # continue. Voir `find_ds_root`.
    strict = argv[1] == "--check"
    ds_root = find_ds_root(obligatoire=strict)
    check_python(ds_root)
    print("[launcher] depot  : %s" % REPO, flush=True)
    print("[launcher] DS     : %s" % (ds_root or "ABSENT (etudes analytiques "
                                                 "seulement)"), flush=True)

    if argv[1] == "--check":
        setup(ds_root)
        check(ds_root, argv[2] if len(argv) > 2 else None)
        return

    print("[launcher] etude  : %s" % argv[1], flush=True)
    setup(ds_root)
    run(argv[1], argv[2:], garder_cwd=garder_cwd)


def _console():
    """Point d'entree de la commande `fiabilite` (voir `pyproject.toml`).

    Meme chemin que `python launcher.py`, au nom pres : `freeze_support`
    d'abord -- les workers DOE paralleles sont lances en `spawn` sous
    Windows, et sans lui chaque worker relancerait le programme entier.
    """
    import multiprocessing as mp
    mp.freeze_support()
    main(sys.argv)


if __name__ == "__main__":
    _console()
