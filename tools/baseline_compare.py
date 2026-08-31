"""
Comparaison de deux journaux de run : qu'est-ce qui a bouge, et de combien ?

    python tools/baseline_compare.py reference.jsonl apres_modif.jsonl
    python tools/baseline_compare.py --last flexion_analytique
    python tools/baseline_compare.py ref.jsonl neuf.jsonl --rtol 1e-6 --tout

Sortie : une ligne par grandeur qui a bouge, avec l'ecart relatif et l'endroit
de la chaine ou il apparait POUR LA PREMIERE FOIS. C'est ce point d'apparition
qui compte : un beta qui bouge est une consequence, pas une cause.

Codes de sortie
---------------
0  les deux journaux sont equivalents dans la tolerance
1  au moins une grandeur a bouge au-dela de la tolerance
2  les journaux ne sont pas comparables (etapes absentes ou en trop)
"""

import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: tolerances par motif de nom, du plus specifique au plus general
TOLERANCES = [
    ("out_theta", 1e-8),        # optimisation : sensible au BLAS
    ("sigma_grille", 1e-8),
    ("mu_grille", 1e-9),
    ("eff_grille", 1e-8),
    ("cov_is", 1e-6),
    ("", 1e-10),                # tout le reste : quasi bit-a-bit
]


def tolerance(nom):
    for motif, tol in TOLERANCES:
        if motif and motif in nom:
            return tol
    return TOLERANCES[-1][1]


def charger(path):
    entetes, probes, evenements, footer = {}, {}, [], {}
    with open(path, "r", encoding="utf-8") as fh:
        for ligne in fh:
            e = json.loads(ligne)
            if e["kind"] == "header":
                entetes = e
            elif e["kind"] == "footer":
                footer = e
            elif e["kind"] == "probe":
                probes[(e["stage"], e["name"], e["occ"])] = e["fp"]
            else:
                evenements.append(e)
    return entetes, probes, evenements, footer


def ecart(a, b):
    """Ecart relatif entre deux empreintes. None si non comparable."""
    if a.get("kind") != b.get("kind"):
        return None
    k = a["kind"]
    if k == "scalar":
        d = abs(a["value"] - b["value"])
        return d / max(abs(a["value"]), 1e-300) if a["value"] else d
    if k == "str":
        # condensat de suite d'appels : identique ou pas, il n'y a pas de demi-mesure
        return 0.0 if a.get("value") == b.get("value") else float("inf")
    if k != "array":
        return 0.0 if a == b else None
    if a.get("shape") != b.get("shape"):
        return None
    # UN NaN QUI APPARAIT EST UNE DIVERGENCE, quelle que soit la taille.
    #
    # `telemetry.fingerprint` compte les valeurs non finies -- et ce compte
    # n'etait JAMAIS relu (constat du 29/08/2026). Au-dela de 256 elements,
    # les valeurs ne sont pas stockees et la comparaison se fait sur les
    # statistiques, calculees sur les seules valeurs FINIES : un NaN nouveau
    # n'y apparait que par le decalage qu'il induit sur `mean` et `l2`.
    #
    # Mesure : un NaN pose sur la valeur la plus proche de la moyenne donne
    # un ecart de 1,6e-3 a n=300 et de 2,4e-6 a n=200 000 -- toujours au-dessus
    # des tolerances, donc toujours vu. Mais il l'est PAR ACCIDENT, et l'ecart
    # decroit avec la taille. Le compte, lui, ne depend ni de la taille ni de
    # la tolerance.
    if a.get("non_finis", 0) != b.get("non_finis", 0):
        return float("inf")
    if "values" in a and "values" in b:
        num = den = 0.0
        for x, y in zip(a["values"], b["values"]):
            if x is None or y is None:
                if x != y:
                    return float("inf")
                continue
            num = max(num, abs(x - y))
            den = max(den, abs(x))
        return num / den if den else num
    sa, sb = a.get("stats"), b.get("stats")
    if not sa or not sb:
        return None
    pires = []
    for cle in ("min", "max", "mean", "l2"):
        d = abs(sa[cle] - sb[cle])
        pires.append(d / abs(sa[cle]) if sa[cle] else d)
    return max(pires)


ORDRE = ["doe", "lib:fit_pck", "lib:fit_gepck", "lib:predict_pck", "lib:predict_gepck",
         "lib:predict_gradient_gepck", "solveur", "eff", "form", "proba", "oracle", "cout"]


def rang(stage):
    return ORDRE.index(stage) if stage in ORDRE else len(ORDRE)


def comparer(ref_path, new_path, rtol_global=None, tout=False):
    h1, p1, _, f1 = charger(ref_path)
    h2, p2, _, f2 = charger(new_path)

    print("=" * 78)
    print("REFERENCE : %s" % os.path.basename(ref_path))
    print("COMPARE   : %s" % os.path.basename(new_path))
    print("=" * 78)

    # --- environnement --------------------------------------------------------
    e1 = h1.get("environnement", {})
    e2 = h2.get("environnement", {})
    diffs_env = []
    for cle in ("python", "git_commit"):
        if e1.get(cle) != e2.get(cle):
            diffs_env.append("  %-14s %s  ->  %s" % (cle, e1.get(cle), e2.get(cle)))
    for pkg in sorted(set(e1.get("paquets", {})) | set(e2.get("paquets", {}))):
        a, b = e1.get("paquets", {}).get(pkg), e2.get("paquets", {}).get(pkg)
        if a != b:
            diffs_env.append("  %-14s %s  ->  %s" % (pkg, a, b))
    if diffs_env:
        print("\nENVIRONNEMENT DIFFERENT -- a garder en tete en lisant la suite :")
        print("\n".join(diffs_env))
    if e2.get("git_modifie"):
        print("\n  (arbre de travail modifie au moment du run compare)")

    # --- alignement -----------------------------------------------------------
    cles1, cles2 = set(p1), set(p2)
    absentes = sorted(cles1 - cles2, key=lambda c: (rang(c[0]), c))
    nouvelles = sorted(cles2 - cles1, key=lambda c: (rang(c[0]), c))
    communes = sorted(cles1 & cles2, key=lambda c: (rang(c[0]), c[1], c[2]))

    if absentes:
        print("\nGRANDEURS ABSENTES du run compare (%d) :" % len(absentes))
        for stage, nom, occ in absentes[:20]:
            print("  %-28s %s[%d]" % (stage, nom, occ))
        if len(absentes) > 20:
            print("  ... et %d autres" % (len(absentes) - 20))
    if nouvelles:
        print("\nGRANDEURS NOUVELLES (%d) :" % len(nouvelles))
        for stage, nom, occ in nouvelles[:20]:
            print("  %-28s %s[%d]" % (stage, nom, occ))
        if len(nouvelles) > 20:
            print("  ... et %d autres" % (len(nouvelles) - 20))

    # --- comparaison ----------------------------------------------------------
    identiques, derives, ecarts, incomparables = 0, [], [], []
    for cle in communes:
        stage, nom, occ = cle
        a, b = p1[cle], p2[cle]
        tol = rtol_global if rtol_global is not None else tolerance(nom)
        if a.get("hash") and a.get("hash") == b.get("hash"):
            identiques += 1
            if tout:
                print("  identique   %-22s %-24s" % (stage, nom))
            continue
        d = ecart(a, b)
        if d is None:
            incomparables.append((stage, nom, occ, a, b))
        elif d == 0.0:
            identiques += 1          # scalaires : pas de hachage, mais ecart nul
            if tout:
                print("  identique   %-22s %-24s" % (stage, nom))
        elif d <= tol:
            derives.append((stage, nom, occ, d, tol))
        else:
            ecarts.append((stage, nom, occ, d, tol))

    print("\n%-28s %-26s %12s %10s" % ("ETAPE", "GRANDEUR", "ECART REL.", "TOLERANCE"))
    print("-" * 78)
    for stage, nom, occ, d, tol in ecarts:
        print("  ECART     %-24s %-24s %10.3e %10.1e" % (stage, nom, d, tol))
    for stage, nom, occ, d, tol in derives:
        print("  derive    %-24s %-24s %10.3e %10.1e" % (stage, nom, d, tol))
    for stage, nom, occ, a, b in incomparables:
        print("  INCOMPAR. %-24s %-24s  %s vs %s"
              % (stage, nom, a.get("shape", a.get("kind")), b.get("shape", b.get("kind"))))

    # --- premier point de divergence -----------------------------------------
    if ecarts:
        premier = min(ecarts, key=lambda t: (rang(t[0]), t[2]))
        print("\nPREMIERE DIVERGENCE : %s / %s (ecart %.3e)"
              % (premier[0], premier[1], premier[3]))
        print("C'est la qu'il faut chercher : ce qui suit en decoule.")

    # --- resume ---------------------------------------------------------------
    print("\n" + "-" * 78)
    print("%d identiques au bit pres | %d derives dans la tolerance | %d ECARTS | "
          "%d incomparables" % (identiques, len(derives), len(ecarts), len(incomparables)))
    r1 = {k: v for k, v in (f1.get("resume") or {}).items()}
    r2 = {k: v for k, v in (f2.get("resume") or {}).items()}
    for cle in sorted(set(r1) & set(r2)):
        if r1[cle].get("kind") == "scalar":
            va, vb = r1[cle]["value"], r2[cle]["value"]
            marque = "=" if va == vb else "->"
            print("  %-10s %.12g %s %.12g" % (cle, va, marque, vb))
    print("duree : %.1f s  ->  %.1f s" % (f1.get("duree", 0), f2.get("duree", 0)))

    if absentes or nouvelles or incomparables:
        return 2
    return 1 if ecarts else 0


def dernier_et_avant_dernier(nom):
    d = os.path.join(REPO, "baselines", nom)
    runs = sorted(glob.glob(os.path.join(d, "run_*.jsonl")))
    if len(runs) < 2:
        raise SystemExit("Il faut au moins deux journaux dans %s" % d)
    return runs[-2], runs[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("reference", nargs="?")
    ap.add_argument("compare", nargs="?")
    ap.add_argument("--last", metavar="NOM",
                    help="compare les deux derniers journaux de baselines/NOM")
    ap.add_argument("--rtol", type=float, default=None,
                    help="tolerance unique, remplace la politique par grandeur")
    ap.add_argument("--tout", action="store_true", help="lister aussi les identiques")
    args = ap.parse_args()

    if args.last:
        ref, new = dernier_et_avant_dernier(args.last)
    elif args.reference and args.compare:
        ref, new = args.reference, args.compare
    else:
        ap.error("donner deux journaux, ou --last NOM")

    sys.exit(comparer(ref, new, args.rtol, args.tout))


if __name__ == "__main__":
    main()
