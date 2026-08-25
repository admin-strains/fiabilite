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


def find_ds_root():
    """Racine contenant le paquet STRAINS : DS_ROOT, sinon detection."""
    env = os.environ.get("DS_ROOT")
    if env:
        if os.path.isdir(os.path.join(env, "STRAINS")):
            return os.path.abspath(env)
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

    raise SystemExit(
        "Digital Structure introuvable. Cherche dans :\n  "
        + "\n  ".join(candidats)
        + "\n\nDefinir DS_ROOT sur le dossier 'front' du workspace :\n"
          "  set DS_ROOT=C:\\workspace\\front")


def check_python():
    """Les .pyd de DS sont lies a python310.dll : 3.10 obligatoire."""
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

    # 2. seulement ensuite, les DLL de Digital Structure
    manquants = []
    for sub in DLL_SUBDIRS:
        d = os.path.join(ds_root, sub)
        if os.path.isdir(d):
            os.add_dll_directory(d)
        else:
            manquants.append(d)
    if manquants:
        print("[launcher] repertoires de DLL absents (ignores) :", file=sys.stderr)
        for d in manquants:
            print("           " + d, file=sys.stderr)

    # 3. chemins d'import : DS, puis la librairie du depot
    for p in (ds_root, os.path.join(REPO, "_lib")):
        if p not in sys.path:
            sys.path.insert(0, p)


def run(script):
    """Execute l'etude avec __name__ == '__main__'.

    Les scripts AC ont 98 % de leur code dans `if __name__ == '__main__':` :
    les importer ne fait rien. Cette execution par exec est donc obligatoire
    tant que la phase 3 du plan de nettoyage n'a pas sorti ce code de main.
    """
    script = os.path.abspath(script)
    if not os.path.isfile(script):
        raise SystemExit("Etude introuvable : %s" % script)

    dossier = os.path.dirname(script)
    if dossier not in sys.path:
        sys.path.insert(0, dossier)

    os.chdir(dossier)
    with open(script, "r", encoding="utf-8", errors="replace") as fh:
        source = fh.read()
    exec(compile(source, script, "exec"), {"__name__": "__main__", "__file__": script})


def check(ds_root):
    """--check : valide l'installation sans lancer un seul calcul."""
    import openturns as ot
    from STRAINS.rupt.core import CetSOLV  # noqa: F401
    import branche1

    print("[check] openturns %s" % ot.__version__)
    print("[check] Digital Structure : modules compiles chargeables")
    print("[check] librairie du depot : %d fonctions exposees"
          % len([n for n in dir(branche1) if not n.startswith("_")]))
    for mod in ("smt", "autograd", "sklearn", "matplotlib", "psutil", "threadpoolctl"):
        __import__(mod)
    print("[check] pile des etudes complete")
    print("[check] OK -- installation utilisable")


def main(argv):
    if len(argv) < 2:
        raise SystemExit(
            "usage : python launcher.py <etude.py>\n"
            "        python launcher.py --check")

    check_python()
    ds_root = find_ds_root()
    print("[launcher] depot  : %s" % REPO, flush=True)
    print("[launcher] DS     : %s" % ds_root, flush=True)

    if argv[1] == "--check":
        setup(ds_root)
        check(ds_root)
        return

    print("[launcher] etude  : %s" % argv[1], flush=True)
    setup(ds_root)
    run(argv[1])


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()     # workers DOE paralleles : spawn Windows
    main(sys.argv)
