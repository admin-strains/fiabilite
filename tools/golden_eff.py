"""
Fige le critere EFF TEL QU'IL ETAIT dans les scripts AC.

    python tools/golden_eff.py --revision HEAD

Produit `tests/golden/eff_original.json`. Meme raison que pour les lois et les
caches : une fois la definition retiree des scripts AC, plus rien ne dirait ce
que le code faisait avant.

On enregistre les DEUX implementations d'origine -- la vectorisee
(`_eff_vectorized`) et la scalaire (`EFFFunction._exec`, transcrite ici sans
son emballage OpenTURNS). Elles doivent coincider : c'est ce qui justifie de
n'en garder qu'une, et le golden en conserve la preuve.
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

import numpy as np                                        # noqa: E402
from scipy.stats import norm                              # noqa: E402
from extraction_temoin import AC_FLEXION, fonction_originale  # noqa: E402

MU = [-5.0, -2.0, -0.5, -1e-12, 0.0, 1e-12, 0.5, 2.0, 5.0, 0.3, -0.3]
SIGMA = [1e-8, 1e-3, 0.05, 0.2, 1.0, 3.0, 0.0, -1.0, 0.5, 2.5, 0.01]
FACTEURS = [2.0, 1.0, 0.5]


def scalaire(muG, sigmaG, epsilon_factor):
    """Transcription exacte de EFFFunction._exec, sans OpenTURNS."""
    if sigmaG <= 0.0:
        return 0.0
    epsilon = epsilon_factor * sigmaG
    t1 = -muG / sigmaG
    t2 = (epsilon + muG) / sigmaG
    t3 = (epsilon - muG) / sigmaG
    return float(2 * muG * norm.cdf(t1) - (epsilon + muG) * norm.cdf(-t2)
                 + (epsilon - muG) * norm.cdf(t3)
                 + sigmaG * (-2 * norm.pdf(t1) + norm.pdf(t2) + norm.pdf(t3)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--revision", default="HEAD")
    args = ap.parse_args()

    vect = fonction_originale(AC_FLEXION, "_eff_vectorized",
                              {"norm": norm, "np": np}, revision=args.revision)
    mu, sg = np.array(MU), np.array(SIGMA)
    out = {"revision": args.revision, "mu": MU, "sigma": SIGMA, "facteurs": FACTEURS,
           "vectorise": {}, "scalaire": {}}
    for f in FACTEURS:
        out["vectorise"][str(f)] = [float(v) for v in vect(mu, sg, f)]
        out["scalaire"][str(f)] = [scalaire(m, s, f) for m, s in zip(MU, SIGMA)]
        ecart = max(abs(a - b) for a, b in zip(out["vectorise"][str(f)],
                                               out["scalaire"][str(f)]))
        print("  eps_factor=%.1f : les deux implementations d'origine different de %.3e"
              % (f, ecart))

    p = os.path.join(REPO, "tests", "golden", "eff_original.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("ecrit %s (%d points x %d facteurs)" % (p, len(MU), len(FACTEURS)))


if __name__ == "__main__":
    main()
