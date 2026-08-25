"""
Temoin d'extraction : comparer une fonction extraite a son original.

La phase 3 sort du code de `if __name__ == '__main__':`. Or ce code n'est
couvert ni par les goldens ni par la baseline -- il n'est meme pas
importable. Extraire sans filet, c'est deplacer a l'aveugle.

Ce module fournit le filet : il recupere une fonction **encore imbriquee dans
un script AC**, sans executer le script, et la rend appelable. Le test
d'extraction compare alors l'original et la version extraite sur les memes
entrees. Le code de production sert d'oracle a sa propre refonte.

    from extraction_temoin import fonction_originale
    loi_fy_orig = fonction_originale(AC, "loi_fy", {"SIGMA": 30.1496})
    assert loi_fy_orig(550).getParameter() == lois.loi_fy(550).getParameter()

Comment ca marche
-----------------
Le script AC n'est jamais execute : `ast` y trouve la definition, son texte
est desindente puis execute dans un espace de noms fabrique pour l'occasion,
ou l'on injecte ses variables libres. C'est le seul moyen d'obtenir la
fonction telle qu'elle est en production, sans STRAINS et sans lancer un
calcul.

Ce filet est TEMPORAIRE par construction : quand un script AC aura ete
entierement vide de sa logique, il n'aura plus d'original a offrir. Les tests
d'extraction devront alors etre remplaces par des tests ecrits sur la version
extraite -- ce qu'ils sont deja, en pratique, puisqu'ils fixent le
comportement.
"""

import ast
import io
import os
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AC_FLEXION = os.path.join(REPO, "pure_flexion", "AC3_pure_flexion.py")
AC_MOULIN = os.path.join(REPO, "Moulinblanc", "AC3_moulinblanc.py")


def _source(chemin, revision=None):
    """Source d'un fichier, dans l'arbre de travail ou a une revision git.

    La lecture a une revision est indispensable a la phase 3 : une fois une
    fonction extraite ET retiree du script AC, l'arbre de travail n'a plus
    d'original a offrir. La revision d'avant l'extraction, elle, l'a toujours.
    """
    if revision is None:
        with io.open(chemin, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    import subprocess
    rel = os.path.relpath(os.path.abspath(chemin), REPO).replace("\\", "/")
    p = subprocess.run(["git", "show", "%s:%s" % (revision, rel)],
                       capture_output=True, cwd=REPO, timeout=60)
    if p.returncode != 0:
        raise LookupError("git show %s:%s a echoue : %s"
                          % (revision, rel, p.stderr.decode("utf-8", "replace")[:200]))
    return p.stdout.decode("utf-8", "replace")


def texte_fonction(chemin, nom, revision=None):
    """Texte source d'une fonction (ou classe), meme imbriquee, desindente."""
    src = _source(chemin, revision)
    arbre = ast.parse(src, chemin)
    lignes = src.split("\n")
    for noeud in ast.walk(arbre):
        if isinstance(noeud, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and noeud.name == nom:
            brut = "\n".join(lignes[noeud.lineno - 1:noeud.end_lineno])
            return textwrap.dedent(brut)
    raise LookupError("%s introuvable dans %s" % (nom, os.path.basename(chemin)))


def variables_libres(chemin, nom, revision=None):
    """Noms utilises par la fonction sans y etre definis ni recus en argument.

    Sert a savoir ce qu'il faut injecter -- et, plus important pour le
    chantier, ce que l'extraction devra transformer en parametre explicite.
    """
    import builtins

    texte = texte_fonction(chemin, nom, revision)
    arbre = ast.parse(texte)
    lies, utilises = set(), set()

    for n in ast.walk(arbre):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lies.add(n.name)
            args = getattr(n, "args", None)
            if args is not None:
                lies.update(a.arg for a in list(args.args) + list(args.kwonlyargs))
                if args.vararg:
                    lies.add(args.vararg.arg)
                if args.kwarg:
                    lies.add(args.kwarg.arg)
        elif isinstance(n, ast.arg):
            lies.add(n.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            lies.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            lies.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                lies.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            utilises.add(n.id)

    return sorted(utilises - lies - set(dir(builtins)))


def environnement_standard():
    """Les modules que les scripts AC ont sous la main au moment de definir
    leurs fonctions imbriquees."""
    import math
    import numpy as np

    env = {"np": np, "math": math}
    try:
        import openturns as ot
        env["ot"] = ot
    except ImportError:
        pass
    return env


def fonction_originale(chemin, nom, libres=None, env=None, revision=None):
    """
    Renvoie la fonction telle qu'elle est ecrite dans le script AC.

    `libres` fournit ses variables libres (SIGMA, PARAM_CONFIG, ...). Si l'une
    manque, l'appel echouera sur un NameError explicite -- utiliser
    `variables_libres()` pour savoir lesquelles.
    """
    espace = dict(env or environnement_standard())
    espace.update(libres or {})
    exec(compile(texte_fonction(chemin, nom, revision), chemin + ":" + nom, "exec"), espace)
    return espace[nom]


def _classe(d):
    """Nom de classe d'une loi OpenTURNS, quel que soit son emballage.

    `ot.Normal(...)` rend un `Normal`, mais une loi issue d'une composition ou
    d'une PythonDistribution arrive emballee dans `ot.Distribution` : seul le
    premier repond a `getClassName()` directement.
    """
    try:
        return d.getImplementation().getClassName()
    except AttributeError:
        return d.getClassName()


def memes_parametres(d1, d2, tol=0.0):
    """Deux lois OpenTURNS decrivent-elles la meme distribution ?

    On compare le nom de la classe et le vecteur de parametres : deux objets
    distincts peuvent representer la meme loi, l'identite ne dit rien.
    """
    if _classe(d1) != _classe(d2):
        return False
    p1, p2 = list(d1.getParameter()), list(d2.getParameter())
    if len(p1) != len(p2):
        return False
    return all(abs(a - b) <= tol * max(1.0, abs(a)) for a, b in zip(p1, p2))
