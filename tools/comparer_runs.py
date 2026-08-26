"""
Compare deux journaux de run complet d'un script AC.

    python tools/comparer_runs.py origine.txt actuel.txt

Repond a la question que ni le harness ni la baseline ne posent : le script
AC, execute EN ENTIER sur le vrai solveur, rend-il les memes chiffres qu'avant
la restructuration ?

Le journal d'un AC melange les traces du solveur (des milliers de lignes de
chronometrage) et les grandeurs de la chaine de fiabilite. Cet outil ne
retient que les secondes, et les compare une a une.

Ce qui est extrait, dans l'ordre de la chaine :

    DOE          les points evalues et leur g
    metamodele   erreur LOO, nombre de polynomes, longueurs de correlation
    EFF          points d'enrichissement et valeur du critere
    FORM         beta, Pf, point de conception, par mode
    IS           Pf, beta, coefficient de variation
    cout         nombre d'appels au solveur

Un ecart sur le DOE se propage a tout le reste : l'outil signale donc la
PREMIERE grandeur qui diverge, comme le fait `baseline_compare.py`.
"""

import argparse
import io
import re
import sys

#: (etiquette, motif, groupes numeriques) -- ordre = ordre de la chaine
MOTIFS = [
    ("DOE point", re.compile(r"\[DOE\]\s*(?:pt\s*)?(\d+).*?u\s*=\s*\[([^\]]+)\].*?g\s*=\s*([-\d.eE+]+)"), None),
    ("SOL", re.compile(r"SOL\[(\d+)\]\s*:\s*g\s*=\s*([-\d.eE+]+)"), None),
    ("LOO", re.compile(r"LOO\s*=\s*([-\d.eE+]+)"), None),
    ("n_poly", re.compile(r"n_poly\s*=\s*\[?(\d+)"), None),
    ("theta", re.compile(r"theta\s*=\s*\[([^\]]+)\]"), None),
    ("EFF initial", re.compile(r"EFF initial\s*:\s*EFF\(u_opt\)\s*=\s*([-\d.eE+]+)"), None),
    ("EFF point", re.compile(r"point EFF ajoute[^\[]*\[([^\]]+)\]"), None),
    ("EFF u_opt", re.compile(r"EFF debug u_opt=\[([^\]]+)\]"), None),
    ("beta FORM", re.compile(r"beta(?:_FORM)?\s*=\s*([-\d.eE+]+)"), None),
    ("u*", re.compile(r"u\*\s*=\s*\[([^\]]+)\]"), None),
    ("Pf FORM", re.compile(r"Pf(?:_FORM)?\s*=\s*([-\d.eE+]+)"), None),
    ("Pf IS", re.compile(r"Pf_IS\s*=\s*([-\d.eE+]+)"), None),
    ("beta IS", re.compile(r"beta_IS\s*=\s*([-\d.eE+]+)"), None),
    ("COV", re.compile(r"COV\s*=\s*([-\d.eE+]+)"), None),
]

#: lignes du solveur, sans interet pour la comparaison
BRUIT = re.compile(r"CTIMER|^\s*\d+\s*\|\s*(normal|corrector)|Liberation de memoire"
                   r"|Ecriture|Lecture|^\s*$|MeshGems|meshgems")


def extraire(chemin):
    """Renvoie {etiquette: [valeurs...]} dans l'ordre d'apparition."""
    out = {}
    with io.open(chemin, encoding="utf-8", errors="replace") as fh:
        for ligne in fh:
            if BRUIT.search(ligne):
                continue
            for etiquette, motif, _ in MOTIFS:
                for m in motif.finditer(ligne):
                    out.setdefault(etiquette, []).append(m.group(m.lastindex))
    return out


def _nombres(texte):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", texte)]


def comparer(a, b, rtol):
    ea, eb = extraire(a), extraire(b)
    print("=" * 76)
    print("ORIGINE : %s" % a)
    print("ACTUEL  : %s" % b)
    print("=" * 76)

    ecarts, premiere = [], None
    for etiquette, _, _ in MOTIFS:
        va, vb = ea.get(etiquette, []), eb.get(etiquette, [])
        if not va and not vb:
            continue
        if len(va) != len(vb):
            msg = "%-14s NOMBRE D'OCCURRENCES : %d contre %d" % (etiquette, len(va), len(vb))
            print("  " + msg)
            ecarts.append(msg)
            premiere = premiere or etiquette
            continue
        pire, ou = 0.0, None
        for i, (x, y) in enumerate(zip(va, vb)):
            na, nb = _nombres(x), _nombres(y)
            if len(na) != len(nb):
                pire, ou = float("inf"), i
                break
            for p, q in zip(na, nb):
                d = abs(p - q) / max(abs(p), 1e-300)
                if d > pire:
                    pire, ou = d, i
        etat = "identique" if pire == 0 else ("dans la tolerance" if pire <= rtol else "ECART")
        print("  %-14s %3d valeur(s)  ecart relatif max %9.3e  %s"
              % (etiquette, len(va), pire, etat))
        if pire > rtol:
            ecarts.append("%s (occurrence %s)" % (etiquette, ou))
            premiere = premiere or etiquette

    print("-" * 76)
    if not ecarts:
        print("Les deux runs rendent les memes chiffres (tolerance %.0e)." % rtol)
        return 0
    print("%d grandeur(s) divergent." % len(ecarts))
    print("PREMIERE DIVERGENCE : %s -- c'est la qu'il faut chercher." % premiere)
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("origine")
    ap.add_argument("actuel")
    ap.add_argument("--rtol", type=float, default=1e-9)
    args = ap.parse_args()
    sys.exit(comparer(args.origine, args.actuel, args.rtol))


if __name__ == "__main__":
    main()
