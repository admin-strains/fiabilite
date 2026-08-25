"""
Graphe d'appels interne d'un module, et validation d'un decoupage propose.

    python tools/analyse_dependances.py _lib/branche5.py
    python tools/analyse_dependances.py _lib/branche5.py --partition partition.json

Sert la phase 3 du plan de nettoyage, ou l'on scinde des modules qui portent
plusieurs sujets. Decouper a vue est le meilleur moyen de creer une
circularite ou de casser une dependance cachee ; cet outil repond a deux
questions avant d'ecrire la moindre ligne :

  1. quelles fonctions du module s'appellent entre elles ?
  2. etant donne un decoupage propose, quelles aretes le traversent, et le
     graphe des nouveaux modules est-il acyclique ?

Une arete qui traverse le decoupage n'est pas une faute en soi -- c'est un
import a ecrire. Un CYCLE, si.

Le fichier de partition est un JSON {nom_du_module: [fonctions...]}.
"""

import argparse
import ast
import json
import os
import sys
from collections import defaultdict


def analyser(path):
    """Renvoie (definitions, appels) : nom -> ligne, et nom -> {noms appeles}."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        arbre = ast.parse(fh.read(), path)

    definitions = {}
    for noeud in arbre.body:
        if isinstance(noeud, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[noeud.name] = noeud.lineno

    appels = {nom: set() for nom in definitions}
    imports_differes = []

    for noeud in arbre.body:
        if not isinstance(noeud, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for sous in ast.walk(noeud):
            if isinstance(sous, ast.Name) and sous.id in definitions and sous.id != noeud.name:
                appels[noeud.name].add(sous.id)
            elif isinstance(sous, (ast.Import, ast.ImportFrom)):
                mod = getattr(sous, "module", None) or ",".join(a.name for a in sous.names)
                imports_differes.append((noeud.name, sous.lineno, mod))

    return definitions, appels, imports_differes


def module_de(nom, partition):
    for mod, fonctions in partition.items():
        if nom in fonctions:
            return mod
    return None


def valider(definitions, appels, partition):
    """Aretes traversantes et detection de cycle entre modules proposes."""
    oubliees = sorted(set(definitions) - {f for l in partition.values() for f in l})
    inconnues = sorted({f for l in partition.values() for f in l} - set(definitions))

    traversantes = defaultdict(list)
    for source, cibles in appels.items():
        ms = module_de(source, partition)
        for cible in sorted(cibles):
            mc = module_de(cible, partition)
            if ms and mc and ms != mc:
                traversantes[(ms, mc)].append((source, cible))

    # cycle ?
    aretes = defaultdict(set)
    for (a, b) in traversantes:
        aretes[a].add(b)
    cycles = []
    etat = {}

    def visiter(n, pile):
        etat[n] = 1
        for m in sorted(aretes.get(n, ())):
            if etat.get(m) == 1:
                cycles.append(pile[pile.index(m):] + [m] if m in pile else [n, m])
            elif etat.get(m) is None:
                visiter(m, pile + [m])
        etat[n] = 2

    for n in sorted(partition):
        if etat.get(n) is None:
            visiter(n, [n])

    return oubliees, inconnues, traversantes, cycles


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("module")
    ap.add_argument("--partition", help="JSON {module: [fonctions]}")
    args = ap.parse_args()

    definitions, appels, differes = analyser(args.module)
    print("=" * 76)
    print("%s : %d definitions au niveau module" % (os.path.basename(args.module), len(definitions)))
    print("=" * 76)

    if differes:
        print("\nIMPORTS DIFFERES (dans le corps d'une fonction) -- aretes cachees :")
        for fonction, ligne, mod in differes:
            print("  l.%-5d %-38s importe %s" % (ligne, fonction, mod))

    if not args.partition:
        print("\nAppels internes :")
        for nom in sorted(definitions, key=lambda n: definitions[n]):
            cibles = sorted(appels[nom])
            print("  l.%-5d %-38s -> %s" % (definitions[nom], nom,
                                            ", ".join(cibles) if cibles else "(aucun)"))
        entrants = defaultdict(int)
        for cibles in appels.values():
            for c in cibles:
                entrants[c] += 1
        isoles = [n for n in definitions if not appels[n] and not entrants[n]]
        if isoles:
            print("\nFonctions isolees (ni appelantes ni appelees dans ce module) :")
            for n in sorted(isoles):
                print("  %s" % n)
        return 0

    with open(args.partition, "r", encoding="utf-8") as fh:
        partition = json.load(fh)

    oubliees, inconnues, traversantes, cycles = valider(definitions, appels, partition)

    print("\nPartition proposee :")
    for mod, fonctions in sorted(partition.items()):
        print("  %-16s %2d fonctions" % (mod, len(fonctions)))

    if oubliees:
        print("\nNON AFFECTEES (%d) -- le decoupage est incomplet :" % len(oubliees))
        for n in oubliees:
            print("  l.%-5d %s" % (definitions[n], n))
    if inconnues:
        print("\nINEXISTANTES dans le module (%d) :" % len(inconnues))
        for n in inconnues:
            print("  %s" % n)

    print("\nAretes traversant le decoupage (= imports a ecrire) :")
    if not traversantes:
        print("  aucune -- les sous-ensembles sont independants")
    for (a, b), paires in sorted(traversantes.items()):
        noms = sorted({c for _, c in paires})
        print("  %-16s -> %-16s %2d appel(s) : %s"
              % (a, b, len(paires), ", ".join(noms[:6]) + (" ..." if len(noms) > 6 else "")))

    if cycles:
        print("\nCYCLE ENTRE MODULES -- decoupage a revoir :")
        for c in cycles:
            print("  " + " -> ".join(c))
        return 1

    print("\nGraphe acyclique : le decoupage tient.")
    return 0 if not (oubliees or inconnues) else 2


if __name__ == "__main__":
    sys.exit(main())
