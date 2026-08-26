"""
Fige la configuration portee par les scripts AC, avant qu'ils ne la perdent.

    python tools/golden_config.py            # ecrit tests/golden/config_*.json
    python tools/golden_config.py --verifier  # compare sans ecrire

PHASE 4b. Tant que les scripts AC portaient leurs ~80 affectations litterales,
ils etaient l'oracle : `tests/test_85_configuration.py` comparait le TOML a
CE QUE FAISAIT LE SCRIPT, et c'est ce qui a prouve que la phase 4a ne changeait
aucune valeur.

En debranchant les scripts sur `_config/schema.py`, cet oracle disparait --
et une comparaison a un oracle vide passerait sans rien prouver. Cet outil le
recopie donc dans un golden, PRIS A LA REVISION QUI PRECEDE LE DEBRANCHEMENT.

L'extraction est la meme que celle de `test_85` : lecture par AST, sans jamais
executer le script (il exige Digital Structure et OpenTURNS). Seules les
affectations LITTERALES du premier niveau de `__main__`, avant la premiere
definition de fonction, sont retenues : au-dela commence le code, pas la
configuration.
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

ETUDES = {
    "pure_flexion": "pure_flexion/AC3_pure_flexion.py",
    "moulin_blanc": "Moulinblanc/AC3_moulinblanc.py",
}


def config_litterale(source):
    """Affectations litterales au premier niveau de `main`, avant la 1re def.

    Identique a `test_85._config_du_script`, mais prend la SOURCE et non un
    chemin : c'est ce qui permet de lire une revision git.
    """
    main = [n for n in ast.parse(source).body
            if isinstance(n, ast.If) and getattr(n.test.left, "id", None) == "__name__"]
    if not main:
        raise ValueError("pas de bloc __main__")
    out = {}
    for n in main[0].body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            break
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except Exception:
                pass                      # expression : ce n'est plus un litteral
    return out


def source_a_revision(chemin_relatif, revision):
    """Contenu d'un fichier a une revision git, sans toucher a l'arbre."""
    if revision is None:
        return io.open(os.path.join(REPO, chemin_relatif),
                       encoding="utf-8", errors="replace").read()
    return subprocess.check_output(
        ["git", "show", "%s:%s" % (revision, chemin_relatif)],
        cwd=REPO).decode("utf-8", "replace")


def chemin_golden(nom):
    return os.path.join(GOLDEN, "config_%s.json" % nom)


def _revision_courante():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO).decode().strip()
    except Exception:
        return "?"


def produire(nom, revision):
    src = source_a_revision(ETUDES[nom], revision)
    valeurs = config_litterale(src)
    return {
        "_etude": nom,
        "_script": ETUDES[nom],
        "_revision": revision or _revision_courante(),
        "_n_affectations": len(valeurs),
        "valeurs": valeurs,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--revision", default=None,
                    help="revision git a lire (defaut : l'arbre de travail)")
    ap.add_argument("--verifier", action="store_true",
                    help="compare au golden existant au lieu de le reecrire")
    args = ap.parse_args()

    os.makedirs(GOLDEN, exist_ok=True)
    ecarts = 0
    for nom in sorted(ETUDES):
        neuf = produire(nom, args.revision)
        cible = chemin_golden(nom)
        if args.verifier:
            if not os.path.isfile(cible):
                print("  %-14s ABSENT : %s" % (nom, cible))
                ecarts += 1
                continue
            ancien = json.load(io.open(cible, encoding="utf-8"))
            memes = ancien["valeurs"] == neuf["valeurs"]
            print("  %-14s %d affectations  %s"
                  % (nom, neuf["_n_affectations"],
                     "identique" if memes else "ECART"))
            if not memes:
                a, b = ancien["valeurs"], neuf["valeurs"]
                for cle in sorted(set(a) | set(b)):
                    if a.get(cle, "<absent>") != b.get(cle, "<absent>"):
                        print("      %-22s golden=%r  script=%r"
                              % (cle, a.get(cle, "<absent>"), b.get(cle, "<absent>")))
                ecarts += 1
        else:
            with io.open(cible, "w", encoding="utf-8") as fh:
                json.dump(neuf, fh, indent=1, sort_keys=True, ensure_ascii=False)
                fh.write("\n")
            print("  %-14s %d affectations figees (revision %s) -> %s"
                  % (nom, neuf["_n_affectations"], neuf["_revision"],
                     os.path.relpath(cible, REPO)))
    return 1 if ecarts else 0


if __name__ == "__main__":
    sys.exit(main())
