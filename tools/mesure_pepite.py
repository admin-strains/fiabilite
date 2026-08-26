"""
Effet de la pepite sur le conditionnement, l'interpolation et beta.

    python tools/mesure_pepite.py            # tableau complet
    python tools/mesure_pepite.py --rapide   # un seul plan d'experiences

DEFAUTS 2 et 3 du plan de nettoyage. Cet outil produit la mesure qui a
justifie de passer la pepite par defaut de 0 a 1e-8, et permet de la refaire
si quelqu'un veut changer cette valeur -- ou verifier qu'elle tient encore sur
un autre etat limite.

CE QUE LA MESURE ETABLIT
-------------------------
Sans pepite, l'erreur sur beta EMPIRE quand le plan d'experiences grandit,
alors que la boucle d'enrichissement EFF, elle, ajoute des points. Sur un
etat limite lineaire -- un hyperplan que le metamodele contient exactement --
GEPCK rendait beta = 19,8 au lieu de 3,5 sur 40 points.

La cause n'est pas seulement le conditionnement. Sans pepite, la
vraisemblance croit indefiniment avec les longueurs de correlation (une
matrice plus singuliere la gonfle artificiellement) : l'optimiseur va au
plafond. Deux des quatre cas de reference y etaient colles, a theta = 100.
La pepite restaure un maximum INTERIEUR -- c'est l'estimation de theta
elle-meme qui etait cassee.

Ne demande ni Digital Structure ni OpenTURNS.
"""

import argparse
import os
import sys
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(REPO, "tests"), os.path.join(REPO, "_lib")):
    if p not in sys.path:
        sys.path.insert(0, p)

PEPITES = (0.0, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5)


def _mesurer(modele, ls, doe, pepite):
    import harness                                            # noqa: PLC0415
    from reference.form import hlrf                           # noqa: PLC0415

    opts = harness.default_opts(3)
    opts["Kriging"] = {"Corr": {"Family": "matern-5_2", "Nugget": pepite}}
    fm = harness.fit(modele, doe, ls, opts=opts)
    K = fm["Kriging"]
    K = K[0] if isinstance(K, list) else K
    R = np.asarray(K["R_tilde" if modele == "GEPCK" else "R"])

    g_hat, grad_hat, _ = harness.predictors(modele, fm)
    interpolation = float(np.max(np.abs(g_hat(doe) - ls.g(doe))))
    r = hlrf(g_hat, grad_hat, np.zeros(ls.n_var))
    beta_exact = ls.beta_exact()
    return {
        "theta": np.asarray(K["theta"], float),
        "cond": float(np.linalg.cond(R)),
        "interpolation": interpolation,
        "beta": r["beta"],
        "erreur": abs(r["beta"] - beta_exact) / beta_exact * 100.0,
        "converge": bool(r["converged"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--rapide", action="store_true", help="un seul plan (24 points)")
    ap.add_argument("--pepite", type=float, default=None,
                    help="ne mesurer qu'une valeur de pepite")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    import harness                                            # noqa: PLC0415
    from reference.limit_states import FlexionLS, LinearLS    # noqa: PLC0415

    cas = [("flexion", FlexionLS()), ("lineaire", LinearLS())]
    tailles = (24,) if args.rapide else (12, 24, 40)
    pepites = (args.pepite,) if args.pepite is not None else PEPITES

    print("%-9s %-4s %-6s %-8s %-11s %-14s %-11s %s"
          % ("cas", "N", "modele", "pepite", "cond(R)", "interpolation",
             "beta", "erreur %"))
    print("-" * 84)
    for nom, ls in cas:
        for N in tailles:
            doe = harness.make_doe(N, ls.n_var)
            for modele in ("PCK", "GEPCK"):
                for pepite in pepites:
                    try:
                        m = _mesurer(modele, ls, doe, pepite)
                        plafond = " theta AU PLAFOND" if np.any(m["theta"] > 99.0) else ""
                        print("%-9s %-4d %-6s %-8.0e %-11.2e %-14.3e %-11.6f %.4f%s"
                              % (nom, N, modele, pepite, m["cond"],
                                 m["interpolation"], m["beta"], m["erreur"], plafond))
                    except Exception as exc:
                        print("%-9s %-4d %-6s %-8.0e ECHEC : %s"
                              % (nom, N, modele, pepite, str(exc)[:40]))
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
