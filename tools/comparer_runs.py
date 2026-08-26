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

#: Le RESULTAT, par opposition au flux intermediaire ci-dessus.
#:
#: La distinction n'est pas cosmetique. Les 41 valeurs de `beta FORM` d'un run
#: incluent toutes les iterations d'enrichissement, dont l'ORDRE change d'une
#: execution a l'autre : le critere EFF y presente deux maxima quasi egaux et
#: un ecart minuscule fait basculer lequel l'emporte. Comparees une a une, ces
#: 41 valeurs signalent donc un ecart alors que rien n'a bouge.
#:
#: Les modes finaux, eux, se sont reveles identiques entre la revision
#: d'origine et toutes les etapes de restructuration -- alors qu'une simple
#: repetition du meme code les fait bouger. C'est la grandeur a regarder.
MOTIFS_FINAUX = [
    ("mode final", re.compile(
        r"mode\s+(\d+)\s*:\s*beta=([-\d.eE+]+)\s+Pf=([-\d.eE+]+)\s+u\*=\[([^\]]+)\]")),
    ("Pf_IS final", re.compile(r"^\s*Pf_IS\s+=\s+([-\d.eE+]+)")),
    ("beta_IS final", re.compile(r"^\s*beta_IS\s+=\s+([-\d.eE+]+)")),
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


def extraire_finaux(chemin):
    """Le resultat de l'etude : modes FORM et tirage d'importance final.

    La ligne complete est conservee, pas seulement un groupe : c'est elle
    qu'on veut voir cote a cote quand elle diverge.
    """
    out = {}
    with io.open(chemin, encoding="utf-8", errors="replace") as fh:
        for ligne in fh:
            for etiquette, motif in MOTIFS_FINAUX:
                if motif.search(ligne):
                    out.setdefault(etiquette, []).append(ligne.strip())
    # Un run reaffiche ses modes a chaque iteration d'enrichissement : seule
    # la DERNIERE serie compte. On remonte jusqu'au « mode 1 » inclus.
    if "mode final" in out:
        garde = []
        for ligne in reversed(out["mode final"]):
            garde.insert(0, ligne)
            if re.match(r"mode\s+1\s*:", ligne):
                break
        out["mode final"] = garde
    for cle in ("Pf_IS final", "beta_IS final"):
        if cle in out:
            out[cle] = out[cle][-1:]
    return out


def comparer_finaux(a, b):
    """Le resultat de l'etude, cote a cote. Renvoie la liste des ecarts REELS.

    Seuls les modes FORM sont juges : ils se sont reveles identiques entre la
    revision d'origine et toutes les etapes de restructuration. `Pf_IS` et
    `beta_IS`, eux, portent la dispersion d'un run a l'autre -- 12,3 %
    mesures -- et sont donc affiches pour information, jamais comme verdict.
    """
    fa, fb = extraire_finaux(a), extraire_finaux(b)
    print("  RESULTAT FINAL")
    if not fa and not fb:
        print("    (aucun resultat final reconnu dans ces journaux)")
        return []

    ecarts = []
    va, vb = fa.get("mode final", []), fb.get("mode final", [])
    if va or vb:
        if va == vb:
            for ligne in va:
                print("    modes FORM  identique   %s" % ligne)
        else:
            ecarts.append("mode final")
            print("    modes FORM  DIFFERENTS")
            for ligne in va:
                print("      origine : %s" % ligne)
            for ligne in vb:
                print("      actuel  : %s" % ligne)

    for cle in ("Pf_IS final", "beta_IS final"):
        x, y = fa.get(cle, ["?"])[0], fb.get(cle, ["?"])[0]
        etat = "identique" if x == y else "-- soumis au bruit de run, pas un verdict"
        print("    %-13s %s  /  %s   %s" % (cle, x, y, etat))
    return ecarts


def _nombres(texte):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", texte)]


def _resolution_relative(texte):
    """Ecart relatif que vaut UN digit du dernier chiffre imprime.

    Une valeur journalisee `0.310336` ne dit rien en dessous de 1e-6 absolu :
    deux runs peuvent l'ecrire 0.310336 et 0.310337 sans que rien n'ait change
    ailleurs que dans l'arrondi. Un plancher de bruit ne doit donc jamais
    descendre sous cette resolution -- sinon l'outil signale un ECART la ou il
    n'y a qu'un affichage.

    C'est arrive au premier essai : `EFF initial` a ete signale pour 3,2e-06,
    soit exactement 1e-6 / 0,310336.
    """
    pire = 0.0
    for m in re.finditer(r"[-+]?(\d*)\.?(\d+)(?:[eE]([-+]?\d+))?", texte):
        entier, frac, exposant = m.group(1), m.group(2), m.group(3)
        if "." not in m.group(0):
            continue                                  # entier : resolution nulle
        valeur = abs(float(m.group(0)))
        if valeur == 0:
            continue
        absolue = 10.0 ** (-len(frac))
        if exposant:
            absolue *= 10.0 ** int(exposant)
        pire = max(pire, absolue / valeur)
    return pire


def _ecart_max(va, vb):
    """(pire ecart relatif, occurrence) entre deux listes de valeurs extraites.

    Renvoie (None, None) si les deux listes n'ont pas la meme longueur : ce
    n'est plus un ecart numerique mais un changement de deroulement.
    """
    if len(va) != len(vb):
        return None, None
    pire, ou = 0.0, None
    for i, (x, y) in enumerate(zip(va, vb)):
        na, nb = _nombres(x), _nombres(y)
        if len(na) != len(nb):
            return float("inf"), i
        for p, q in zip(na, nb):
            d = abs(p - q) / max(abs(p), 1e-300)
            if d > pire:
                pire, ou = d, i
    return pire, ou


def comparer(a, b, rtol, repetition=None):
    ea, eb = extraire(a), extraire(b)
    er = extraire(repetition) if repetition else None
    print("=" * 76)
    print("ORIGINE    : %s" % a)
    print("ACTUEL     : %s" % b)
    if repetition:
        print("REPETITION : %s   (meme code que ORIGINE)" % repetition)
    print("=" * 76)
    if not repetition:
        print("Sans repetition, le seuil est arbitraire (%.0e). Sur cette chaine, deux" % rtol)
        print("runs du MEME code ecartent de plusieurs dizaines de pour cent : lire ce")
        print("tableau comme un constat, pas comme un verdict. Fournir --repetition.")
        print("-" * 76)

    ecarts, premiere = [], None
    for etiquette, _, _ in MOTIFS:
        va, vb = ea.get(etiquette, []), eb.get(etiquette, [])
        if not va and not vb:
            continue
        pire, ou = _ecart_max(va, vb)
        if pire is None:
            msg = "%-14s NOMBRE D'OCCURRENCES : %d contre %d" % (etiquette, len(va), len(vb))
            print("  " + msg)
            ecarts.append(msg)
            premiere = premiere or etiquette
            continue

        # Le seuil vient de la MESURE quand une repetition est fournie : c'est
        # l'ecart que produit le meme code relance. Rien en dessous ne peut
        # etre impute a une modification du code.
        if er is not None:
            plancher, _ = _ecart_max(va, er.get(etiquette, []))
            if plancher is None:              # la repetition n'a pas le meme deroulement
                plancher = float("inf")
        else:
            plancher = rtol
        # ... et jamais en dessous de ce que le journal sait ecrire.
        plancher = max(plancher, max((_resolution_relative(x) for x in va), default=0.0))

        significatif = pire > plancher
        if pire == 0:
            etat = "identique"
        elif not significatif:
            etat = "dans le bruit" if er is not None else "dans la tolerance"
        else:
            etat = "ECART"
        print("  %-14s %3d valeur(s)  ecart %9.3e  plancher %9.3e  %s"
              % (etiquette, len(va), pire, plancher, etat))
        if significatif:
            ecarts.append("%s (occurrence %s)" % (etiquette, ou))
            premiere = premiere or etiquette

    print("-" * 76)
    ecarts_finaux = comparer_finaux(a, b)
    print("-" * 76)
    if ecarts and not ecarts_finaux:
        print("Le FLUX intermediaire s'ecarte (%s) mais le RESULTAT est identique."
              % ", ".join(e.split(" (")[0] for e in ecarts))
        print("C'est la signature du reordonnancement des points d'enrichissement :")
        print("deux maxima quasi egaux du critere EFF, dont l'ordre bascule d'un run")
        print("a l'autre. Le meme reordonnancement s'observe entre deux executions du")
        print("MEME code -- il ne dit donc rien du code.")
        return 0
    if not ecarts:
        if er is not None:
            print("Aucun ecart ne depasse le bruit d'une repetition du meme code.")
            print("C'est la seule conclusion que cette chaine permet -- elle n'est pas")
            print("reproductible au bit pres (docs/reproductibilite-chaine-complete.md).")
        else:
            print("Les deux runs rendent les memes chiffres (tolerance %.0e)." % rtol)
        return 0
    print("%d grandeur(s) depassent le plancher." % len(ecarts))
    print("PREMIERE DIVERGENCE : %s -- c'est la qu'il faut chercher." % premiere)
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("origine")
    ap.add_argument("actuel")
    ap.add_argument("--repetition", default=None,
                    help="journal d'un SECOND run du code d'ORIGINE. Fournit le "
                         "plancher de bruit, mesure au lieu d'etre suppose.")
    ap.add_argument("--rtol", type=float, default=1e-9,
                    help="seuil de repli quand aucune repetition n'est fournie")
    args = ap.parse_args()
    sys.exit(comparer(args.origine, args.actuel, args.rtol, args.repetition))


if __name__ == "__main__":
    main()
