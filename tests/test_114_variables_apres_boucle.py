r"""Aucune variable lue apres un `while` ne doit n'exister que dedans.

CE QUE CE CONTROLE GENERALISE
------------------------------
`test_113` verifie cet invariant sur `run_EFF`, ou il avait ete viole :
`_ratio_bb` n'etait affecte que dans la boucle d'enrichissement, et le bilan
le lisait apres. Un run qui n'entrait jamais dans la boucle -- budget
d'enrichissement deja epuise a la reprise, ou critere EFF deja satisfait --
mourait sur un `UnboundLocalError`, apres tout le travail.

Ce fichier passe le meme controle sur TOUT le depot. Le defaut etait
isole ; ce test est ce qui le garde isole.

POURQUOI SEULEMENT LES `while`
-------------------------------
Un `while` peut trivialement ne pas tourner : sa condition est evaluee avant
le premier tour, et elle depend de l'etat. C'est le cas dangereux.

Un `for` le peut aussi -- sur un iterable vide -- mais le controle y devient
imprecis pour deux raisons, mesurees le 28/08/2026 sur ce depot :

  * la variable de boucle FUIT apres le `for` par construction du langage, et
    presque toujours sans consequence : l'iterable est non vide par
    construction ;
  * un nom peut etre reaffecte par une boucle SUIVANTE avant d'etre lu, ce
    que cette analyse ne modelise pas.

Le controle etendu aux `for` rendait 65 signalements, tous relevant de l'un
de ces deux motifs -- aucun defaut. Etendu aux seuls `while` : ZERO. On garde
donc le controle la ou il conclut, et on dit pourquoi on ne l'etend pas.

CE QUE L'ANALYSE SUR-APPROXIME
-------------------------------
« Connu avant la boucle » = tout nom affecte n'importe ou plus haut dans la
MEME portee, meme dans une branche non prise. On sur-approxime donc ce qui
est connu : le controle peut manquer un defaut, il ne doit pas en inventer.
Si un signalement apparait, il se lit -- ce n'est pas forcement un defaut,
mais c'est toujours une variable dont l'existence depend d'une boucle.
"""

import ast
import io
import os

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)

#: Dossiers hors perimetre : dependances tierces, sorties, historique.
IGNORES = ("__pycache__", ".git", "_lib", "historique", "output", ".pixi",
           "storage", ".venv")


def _cibles(n):
    if isinstance(n, ast.Assign):
        return n.targets
    if isinstance(n, (ast.AugAssign, ast.AnnAssign)):
        return [n.target]
    if isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)):
        return [n.target]
    if isinstance(n, ast.withitem) and n.optional_vars is not None:
        return [n.optional_vars]
    return []


def _affectations(porteur, ):
    """`(nom, ligne)` de toute affectation de la portee, sans descendre dans
    les fonctions imbriquees -- elles ont la leur."""
    out = []
    pile = list(getattr(porteur, "body", []))
    while pile:
        n = pile.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append((n.name, n.lineno))
            continue
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            out += [((a.asname or a.name).split(".")[0], n.lineno)
                    for a in n.names]
        if isinstance(n, ast.ExceptHandler) and n.name:
            out.append((n.name, n.lineno))
        for c in _cibles(n):
            for x in ast.walk(c):
                if isinstance(x, ast.Name):
                    out.append((x.id, n.lineno))
        for x in ast.walk(n):
            if isinstance(x, ast.comprehension):
                for y in ast.walk(x.target):
                    if isinstance(y, ast.Name):
                        out.append((y.id, n.lineno))
        for champ in ("body", "orelse", "finalbody", "handlers"):
            pile += getattr(n, champ, []) or []
    return out


def _portees(arbre):
    yield arbre
    for n in ast.walk(arbre):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield n


def fragiles(source, chemin="<source>"):
    """Les `(ligne, nom)` lus apres un `while` et affectes seulement dedans."""
    arbre = ast.parse(source, filename=chemin)
    trouves = []
    for porteur in _portees(arbre):
        args = set()
        if isinstance(porteur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = porteur.args
            args = {x.arg for x in
                    a.args + a.kwonlyargs + getattr(a, "posonlyargs", [])}
            if a.vararg:
                args.add(a.vararg.arg)
            if a.kwarg:
                args.add(a.kwarg.arg)
        toutes = _affectations(porteur)
        corps = porteur.body
        for i, n in enumerate(corps):
            if not isinstance(n, ast.While):
                continue
            apres = corps[i + 1:]
            if not apres:
                continue
            avant = args | {nom for nom, l in toutes if l < n.lineno}
            dans = {nom for nom, _ in _affectations(n)}
            lus = set()
            for suite in apres:
                for x in ast.walk(suite):
                    if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                        lus.add(x.id)
            for nom in sorted((lus & dans) - avant):
                trouves.append((n.lineno, nom))
    return trouves


def _sources():
    for dossier, sous, fichiers in os.walk(_REPO):
        sous[:] = [d for d in sous if d not in IGNORES and not d.startswith(".")]
        if any(p in dossier for p in IGNORES):
            continue
        for f in sorted(fichiers):
            if f.endswith(".py"):
                yield os.path.join(dossier, f)


# --------------------------------------------------------------------------- #
def test_le_controle_reconnait_le_defaut_qu_il_garde():
    """Un garde-fou qui ne peut pas se declencher ne vaut rien. Voici le
    defaut du 28/08/2026, reduit a sa forme minimale."""
    source = (
        "def run(n):\n"
        "    total = 0\n"
        "    while total < n:\n"
        "        ratio = total / n\n"
        "        total += 1\n"
        "    print(ratio)\n"
    )
    assert fragiles(source) == [(3, "ratio")]


def test_un_nom_defini_avant_la_boucle_ne_se_signale_pas():
    """Le correctif applique : une ligne au-dessus de la boucle."""
    source = (
        "def run(n):\n"
        "    total = 0\n"
        "    ratio = None\n"
        "    while total < n:\n"
        "        ratio = total / n\n"
        "        total += 1\n"
        "    print(ratio)\n"
    )
    assert fragiles(source) == []


def test_un_argument_compte_comme_defini():
    source = (
        "def run(n, ratio=None):\n"
        "    while n:\n"
        "        ratio = n\n"
        "        n -= 1\n"
        "    return ratio\n"
    )
    assert fragiles(source) == []


# --------------------------------------------------------------------------- #
def test_aucune_source_du_depot_ne_lit_une_variable_de_while_apres_lui():
    """Le controle, sur tout le depot. Zero cas le 28/08/2026, et c'est un
    seuil STRICT : il n'y a pas de raison legitime d'en avoir un."""
    signales = []
    for chemin in _sources():
        source = io.open(chemin, encoding="utf-8", errors="replace").read()
        try:
            t = fragiles(source, chemin)
        except SyntaxError:
            continue          # `test_05_hygiene_depot` s'occupe de la syntaxe
        rel = os.path.relpath(chemin, _REPO).replace(os.sep, "/")
        signales += ["%s:%d  %s" % (rel, ligne, nom) for ligne, nom in t]

    assert not signales, (
        "Variable(s) lue(s) apres un `while` alors qu'elles n'existent que "
        "s'il a tourne :\n  %s\n"
        "Si la boucle ne tourne pas -- et une condition de `while` est "
        "evaluee AVANT le premier tour -- c'est un UnboundLocalError. Le "
        "definir au-dessus de la boucle suffit. Voir `test_113` pour le cas "
        "qui a motive ce controle." % "\n  ".join(signales))
