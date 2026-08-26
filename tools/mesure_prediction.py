"""
Cout d'un appel de prediction, decompose.

    python tools/mesure_prediction.py
    python tools/mesure_prediction.py --n-doe 40 --repets 20

PHASE 7 du plan de nettoyage. Le plan designe un coupable : `predict.py`
reconstruit `Rinv` par `solve(cholR, solve(cholR.T, I))` A CHAQUE APPEL de
prediction, alors que le facteur de Cholesky est deja calcule au fit.

Cet outil mesure avant de croire. Il separe :

    noyau      construction de r0 (cross-correlation)
    inverse    reconstruction de Rinv
    produit    r0 @ (Rinv @ residu)
    variance   le supplement quand on demande sigma

La distinction compte : la chaine appelle la prediction en deux regimes tres
differents -- un point a la fois dans FORM et dans la recherche EFF, dix
mille points d'un coup dans le tirage d'importance. Une optimisation qui aide
l'un peut ne rien changer a l'autre.

Ne demande ni Digital Structure ni OpenTURNS.
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(REPO, "tests"), os.path.join(REPO, "_lib")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _chrono(fonction, repets):
    """Meilleur temps sur `repets` essais, en millisecondes.

    Le MEILLEUR et non la moyenne : on mesure un cout de calcul, et les
    valeurs hautes ne disent que ce que l'ordonnanceur faisait a cote.
    """
    t = []
    for _ in range(repets):
        t0 = time.perf_counter()
        fonction()
        t.append(time.perf_counter() - t0)
    return min(t) * 1e3


def mesurer(modele, n_doe, tailles, repets):
    import harness                                            # noqa: PLC0415
    from api import predict_gepck, predict_pck                # noqa: PLC0415
    from reference.limit_states import FlexionLS              # noqa: PLC0415

    ls = FlexionLS()
    doe = harness.make_doe(n_doe, ls.n_var)
    fm = harness.fit(modele, doe, ls)
    predire = predict_gepck if modele == "GEPCK" else predict_pck

    K = fm["Kriging"]
    K = K[0] if isinstance(K, list) else K
    R = np.asarray(K["R_tilde" if modele == "GEPCK" else "R"])
    cholR = K["auxMatrices"]["cholR"]
    n_aug = R.shape[0]

    print("  %s : plan de %d points, systeme %dx%d"
          % (modele, n_doe, n_aug, n_aug))
    if cholR is not None:
        cout_inv = _chrono(
            lambda: np.linalg.solve(cholR, np.linalg.solve(cholR.T, np.eye(n_aug))),
            repets)
        cout_solve = _chrono(
            lambda: np.linalg.solve(cholR, np.linalg.solve(cholR.T,
                                                           np.zeros((n_aug, 1)))),
            repets)
        print("     reconstruction de Rinv      %8.3f ms" % cout_inv)
        print("     une seule descente-remontee %8.3f ms   (rapport %.0fx)"
              % (cout_solve, cout_inv / cout_solve if cout_solve else float("nan")))

    rng = np.random.default_rng(0)
    print("     %-10s %10s %10s %10s" % ("N_test", "moyenne", "+variance", "par point"))
    for n_test in tailles:
        U = rng.normal(size=(n_test, ls.n_var))
        t_mu = _chrono(lambda: predire(fm, U), repets)
        t_var = _chrono(lambda: predire(fm, U, return_var=True), repets)
        print("     %-10d %8.3f ms %8.3f ms %9.4f ms"
              % (n_test, t_mu, t_var, t_mu / n_test))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--n-doe", type=int, default=24)
    ap.add_argument("--repets", type=int, default=10)
    ap.add_argument("--tailles", type=int, nargs="+", default=[1, 10, 1000, 10000])
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    print("Cout d'un appel de prediction -- meilleur temps sur %d essais\n" % args.repets)
    for modele in ("PCK", "GEPCK"):
        mesurer(modele, args.n_doe, args.tailles, args.repets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
