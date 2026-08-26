"""
Fige les options de maillage et de solveur portees par les scripts AC.

    python tools/golden_options_ds.py             # ecrit tests/golden/options_ds.json
    python tools/golden_options_ds.py --verifier  # compare sans ecrire

PHASE 5. Les quatre copies de l'appel a Digital Structure (`run_one_SOL` et
`run_HF`, dans les deux scripts AC) portent chacune un dictionnaire d'options
de maillage et une vingtaine d'affectations `kwargs[...]`. Ce sont des
REGLAGES DE CALCUL : une valeur qui bouge change le resultat, sans que rien
ne le signale.

Avant de les rassembler dans `solver/digital_structure.py`, on les releve tels
quels, par AST, sans jamais executer le script (il exige Digital Structure).
Le golden sert ensuite a prouver deux choses :

  * la version rassemblee est identique a l'originale, valeur par valeur ;
  * les quatre copies AVAIENT diverge -- et ou.
"""

import argparse
import ast
import io
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(REPO, "tests", "golden")
CIBLE = os.path.join(GOLDEN, "options_ds.json")

SCRIPTS = {
    "pure_flexion": "pure_flexion/AC3_pure_flexion.py",
    "moulin_blanc": "Moulinblanc/AC3_moulinblanc.py",
}
FONCTIONS = ("run_one_SOL", "run_HF")


def _fonctions_imbriquees(source):
    """{nom: noeud} pour toutes les fonctions du script, imbriquees comprises."""
    out = {}
    for noeud in ast.walk(ast.parse(source)):
        if isinstance(noeud, ast.FunctionDef):
            out.setdefault(noeud.name, noeud)
    return out


def _litteral_ou_expression(noeud):
    """Valeur si c'est un litteral, sinon la source de l'expression.

    Une option peut etre `0.05` (litteral) ou `global_size` (variable). La
    difference est precisement ce que ce golden doit capturer : c'est elle qui
    revele que `run_HF` codait en dur ce que `run_one_SOL` lisait dans la
    configuration.
    """
    try:
        return ast.literal_eval(noeud)
    except Exception:
        return {"_expression": ast.unparse(noeud)}


def options_de(noeud_fonction):
    """Les options de maillage et de solveur d'une fonction d'appel a DS."""
    maillage, solveur = {}, {}
    for n in ast.walk(noeud_fonction):
        # Meshkwargs = { ... }
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == "Meshkwargs" and isinstance(n.value, ast.Dict):
            for cle, val in zip(n.value.keys, n.value.values):
                maillage[ast.literal_eval(cle)] = _litteral_ou_expression(val)
        # Meshkwargs["x"] = ... / kwargs["x"] = ...
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Subscript) \
                and isinstance(n.targets[0].value, ast.Name):
            nom = n.targets[0].value.id
            try:
                cle = ast.literal_eval(n.targets[0].slice)
            except Exception:
                continue
            cible = maillage if nom == "Meshkwargs" else (solveur if nom == "kwargs" else None)
            if cible is not None:
                cible[cle] = _litteral_ou_expression(n.value)
        # kwargs = {"scaling": 1, ...}
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == "kwargs" and isinstance(n.value, ast.Dict):
            for cle, val in zip(n.value.keys, n.value.values):
                solveur[ast.literal_eval(cle)] = _litteral_ou_expression(val)
    return {"maillage": maillage, "solveur": solveur}


def source_a_revision(chemin_relatif, revision):
    if revision is None:
        return io.open(os.path.join(REPO, chemin_relatif),
                       encoding="utf-8", errors="replace").read()
    return subprocess.check_output(
        ["git", "show", "%s:%s" % (revision, chemin_relatif)],
        cwd=REPO).decode("utf-8", "replace")


def _revision_courante():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO).decode().strip()
    except Exception:
        return "?"


def produire(revision):
    out = {"_revision": revision or _revision_courante(), "copies": {}}
    for etude, chemin in sorted(SCRIPTS.items()):
        fonctions = _fonctions_imbriquees(source_a_revision(chemin, revision))
        for nom in FONCTIONS:
            if nom not in fonctions:
                continue
            out["copies"]["%s/%s" % (etude, nom)] = options_de(fonctions[nom])
    return out


def divergences(donnees):
    """Ou les quatre copies ne disent pas la meme chose."""
    copies = donnees["copies"]
    lignes = []
    for famille in ("maillage", "solveur"):
        cles = set()
        for c in copies.values():
            cles |= set(c[famille])
        for cle in sorted(cles):
            vues = {nom: c[famille].get(cle, "<absente>") for nom, c in copies.items()}
            distinctes = {json.dumps(v, sort_keys=True) for v in vues.values()}
            if len(distinctes) > 1:
                lignes.append((famille, cle, vues))
    return lignes


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--revision", default=None)
    ap.add_argument("--verifier", action="store_true")
    args = ap.parse_args()

    os.makedirs(GOLDEN, exist_ok=True)
    neuf = produire(args.revision)

    print("  %d copies relevees (revision %s)" % (len(neuf["copies"]), neuf["_revision"]))
    for famille, cle, vues in divergences(neuf):
        print("  DIVERGENCE %-8s %-28s" % (famille, cle))
        for nom, val in sorted(vues.items()):
            print("      %-28s %r" % (nom, val))

    if args.verifier:
        if not os.path.isfile(CIBLE):
            print("  golden absent : %s" % CIBLE)
            return 1
        ancien = json.load(io.open(CIBLE, encoding="utf-8"))
        memes = ancien["copies"] == neuf["copies"]
        print("  golden : %s" % ("identique" if memes else "ECART"))
        return 0 if memes else 1

    with io.open(CIBLE, "w", encoding="utf-8") as fh:
        json.dump(neuf, fh, indent=1, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    print("  fige -> %s" % os.path.relpath(CIBLE, REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
