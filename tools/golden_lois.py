"""
Fige le comportement des lois TELLES QU'ELLES ETAIENT dans les scripts AC.

    python tools/golden_lois.py --revision HEAD

Produit `tests/golden/lois_originales.json`, qui devient l'oracle durable de
`_model/lois.py`. Sans lui, une fois les definitions retirees des scripts AC,
plus rien ne dirait ce que le code faisait avant l'extraction.

La revision est celle d'AVANT le retrait des definitions : l'arbre de travail
ne les a plus.
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

import numpy as np                                        # noqa: E402
from extraction_temoin import AC_FLEXION, fonction_originale  # noqa: E402

SIGMA_ORIGINE = float(np.sqrt(19.0 ** 2 + 22.0 ** 2 + 8.0 ** 2))

CAS = {
    "loi_fy": ([(550.0, None), (500.0, None), (550.0, 0.05), (235.0, None), (235.0, 0.10)],
               {"SIGMA": SIGMA_ORIGINE}),
    "loi_fc": ([(48.0, 0.12), (48.0, None), (20.0, None), (36.0, None),
                (60.0, 0.07), (25.0, None)], {}),
    "loi_F_permanente": ([(1.0, None), (1.0, 0.05), (0.9, 0.10), (2.5, None)], {}),
    "loi_F_exploitation": ([(1.0, None, "office"), (1.0, None, "residence"),
                            (1.0, None, "storage"), (1.0, None, "library"),
                            (1.0, None, "industrial_heavy"), (1.0, None, "inconnu"),
                            (2.0, 0.3, "office")], {}),
    "loi_F_intermittente": ([("office",), ("crowd",), ("classroom",), ("inconnu",)], {}),
}

TUKEY = [(0.0, 1.0, 0.5), (-1.0, 1.0, 0.0), (0.2, 0.4, 1.0),
         (0.2, 0.40, 0.5), (-5.0, 5.0, 0.25)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--revision", default="HEAD")
    args = ap.parse_args()

    out = {"revision": args.revision, "source": os.path.basename(AC_FLEXION), "lois": {}}

    for nom, (appels, libres) in CAS.items():
        f = fonction_originale(AC_FLEXION, nom, libres, revision=args.revision)
        out["lois"][nom] = [{"args": list(a), "parametres": list(f(*a).getParameter())}
                            for a in appels]

    f = fonction_originale(AC_FLEXION, "loi_uni_approx", revision=args.revision)
    out["lois"]["loi_uni_approx"] = []
    for a, b, alpha in TUKEY:
        d = f(a, b, alpha)
        xs = list(np.linspace(a - 0.1 * (b - a), b + 0.1 * (b - a), 25))
        out["lois"]["loi_uni_approx"].append({
            "args": [a, b, alpha],
            "x": xs,
            "pdf": [d.computePDF([x]) for x in xs],
            "cdf": [d.computeCDF([x]) for x in xs],
            "quantiles": {str(p): d.computeQuantile(p)[0]
                          for p in (0.01, 0.25, 0.5, 0.75, 0.99)},
        })

    # dist_jointe : ce qui compte en aval est la transformation isoprobabiliste
    lfc = fonction_originale(AC_FLEXION, "loi_fc", revision=args.revision)
    lfy = fonction_originale(AC_FLEXION, "loi_fy", {"SIGMA": SIGMA_ORIGINE},
                             revision=args.revision)
    config = {"fc": {"loi": lfc, "args": (48, 0.12)},
              "fy": {"loi": lfy, "args": (550, None)}}
    dj = fonction_originale(AC_FLEXION, "dist_jointe",
                            {"PARAM_CONFIG": config, "params_names": ["fc", "fy"]},
                            revision=args.revision)
    d = dj()
    t = d.getInverseIsoProbabilisticTransformation()
    out["lois"]["dist_jointe"] = {
        "config": {"fc": [48, 0.12], "fy": [550, None]},
        "parametres": list(d.getParameter()),
        "u": [[0.0, 0.0], [-2.5, -3.5], [1.2, -0.7], [3.0, 3.0]],
        "x": [list(t(u)) for u in ([0.0, 0.0], [-2.5, -3.5], [1.2, -0.7], [3.0, 3.0])],
    }

    p = os.path.join(REPO, "tests", "golden", "lois_originales.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("ecrit %s  (%d lois, revision %s)" % (p, len(out["lois"]), args.revision))


if __name__ == "__main__":
    main()
