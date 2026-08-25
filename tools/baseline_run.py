"""
Production d'une baseline : la chaine de fiabilite jouee de bout en bout,
instrumentee, sur le cas de reference.

    python tools/baseline_run.py                      # baseline analytique (~30 s)
    python tools/baseline_run.py --repeat 3           # + plancher de bruit
    python launcher.py tools/baseline_run.py --solveur ds --repeat 1

Deux baselines, deux usages
---------------------------
`--solveur analytique` (defaut) -- l'etat limite de `tests/reference` remplace
    Digital Structure. Quelques dizaines de secondes, aucune dependance lourde :
    c'est la baseline qui se rejoue a **chaque commit**.

`--solveur ds` -- le vrai solveur sur `test_pure_flexion`. Quelques minutes :
    c'est la baseline des **fins de phase**, celle qui prouve que la chaine
    complete n'a pas bouge.

Les deux parcourent exactement les memes etapes et produisent le meme format
de journal, donc le meme comparateur les traite.

Le plancher de bruit
--------------------
`--repeat N` rejoue la chaine N fois et enregistre la dispersion observee.
Sans ce chiffre, un ecart constate apres une modification est ininterpretable :
on ne sait pas s'il vient de la modification ou du run. Une baseline sans son
plancher de bruit ne sert a rien.
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "_lib"), os.path.join(REPO, "tests"), os.path.join(REPO, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import telemetry  # noqa: E402


# --------------------------------------------------------------------------- #
# Les deux fonctions de performance                                           #
# --------------------------------------------------------------------------- #
def solveur_analytique(journal, cfg):
    """Etat limite de reference : meme geometrie que test_pure_flexion."""
    from reference.limit_states import FlexionLS
    ls = FlexionLS(**cfg["section"])
    n_appels = [0]

    def g(U):
        U = np.atleast_2d(U)
        n_appels[0] += U.shape[0]
        return np.asarray(ls.g(U)).ravel()

    def grad(U):
        return np.asarray(ls.grad(np.atleast_2d(U)))

    return g, grad, n_appels, ls


def solveur_ds(journal, cfg):
    """Digital Structure sur test_pure_flexion, via l'adaptateur solve_one."""
    import shutil
    import solve_one
    import launcher

    solve_one._catalogues(launcher.find_ds_root())

    src = os.path.join(cfg["ds"]["storage"], cfg["ds"]["model"] + ".ds")
    work = cfg["ds"]["workdir"]
    if os.path.isdir(work):
        shutil.rmtree(work)
    shutil.copytree(src, work)

    from reference.limit_states import FlexionLS
    ls = FlexionLS(**cfg["section"])          # sert seulement a la transformation U -> X
    n_appels = [0]

    def g(U):
        U = np.atleast_2d(U)
        out = []
        for u in U:
            fc, fy = ls.u_to_x([u])[0]
            gv, alpha, t_mesh, t_solv, sante = solve_one.evaluate(
                work, {"fc": float(fc), "fy": float(fy)},
                global_size=cfg["ds"]["global_size"],
                geo_min_approx=cfg["ds"]["geo_min"])
            n_appels[0] += 1
            journal.probe("solveur", u=u, fc=fc, fy=fy, g=gv, alpha=alpha,
                          gap=sante["gap_relatif"], tetra=sante["numTetra"],
                          iterations=sante["solverIterations"],
                          t_solveur=t_solv)
            sain = bool(sante["converged"]) and sante["solver_status"] == "OPTIMAL"
            if not sain:
                journal.event("solveur", "APPEL NON CONVERGE -- valeur retenue quand meme "
                                         "par le code de production",
                              u=list(map(float, u)), statut=sante["solver_status"],
                              converged=sante["converged"], alpha=alpha)
            out.append(gv)
        return np.array(out)

    def grad(U):
        """Gradient par differences finies (l'adaptateur ne remonte pas encore
        les sensibilites du solveur)."""
        U = np.atleast_2d(U)
        h = cfg["ds"]["fd_step"]
        G = np.zeros_like(U)
        for j in range(U.shape[1]):
            e = np.zeros(U.shape[1])
            e[j] = h
            G[:, j] = (g(U + e) - g(U - e)) / (2 * h)
        return G

    return g, grad, n_appels, ls


SOLVEURS = {"analytique": solveur_analytique, "ds": solveur_ds}


# --------------------------------------------------------------------------- #
# La chaine, instrumentee etape par etape                                     #
# --------------------------------------------------------------------------- #
def chaine(journal, cfg, solveur):
    import harness
    from reference.form import hlrf, multistart_hlrf

    g, grad, n_appels, ls = SOLVEURS[solveur](journal, cfg)

    # --- 1. plan d'experiences ------------------------------------------------
    X = harness.make_doe(cfg["n_doe"], 2, seed=cfg["doe_seed"])
    journal.probe("doe", X=X)

    y = g(X)
    dy = grad(X)
    journal.probe("doe", y=y, dy=dy)

    # --- 2. metamodele --------------------------------------------------------
    # (les appels a fit_* / predict_* sont journalises par instrument_lib)
    from api import fit_pck, fit_gepck
    marginals = [{"Type": "Gaussian", "Parameters": [0.0, 1.0]}] * 2
    copula = {"Type": "Independent", "Parameters": np.eye(2)}
    opts = harness.default_opts(cfg["max_degree"])

    if cfg["modele"] == "GEPCK":
        fm = fit_gepck(X, harness.build_Y_aug(y, dy), opts, marginals, copula)
    else:
        fm = fit_pck(X, y.ravel(), opts, marginals, copula)

    g_hat, grad_hat, sigma_hat = harness.predictors(cfg["modele"], fm)

    # --- 3. critere d'enrichissement EFF sur une grille -----------------------
    gr = np.linspace(-cfg["u_max"], cfg["u_max"], cfg["n_grid"])
    UU = np.column_stack([m.ravel() for m in np.meshgrid(gr, gr, indexing="ij")])
    mu = g_hat(UU)
    sig = sigma_hat(UU)
    eff = expected_feasibility(mu, sig, cfg["epsilon_factor"])
    k = int(np.argmax(eff))
    journal.probe("eff", mu_grille=mu, sigma_grille=sig, eff_grille=eff,
                  eff_max=eff[k], u_eff=UU[k])

    # --- 4. FORM --------------------------------------------------------------
    departs = np.array(cfg["departs"])
    resultats = [hlrf(g_hat, grad_hat, u0) for u0 in departs]
    journal.probe("form",
                  beta_par_depart=np.array([r["beta"] for r in resultats]),
                  iterations=np.array([r["n_iter"] for r in resultats]),
                  converges=np.array([float(r["converged"]) for r in resultats]))
    best = multistart_hlrf(g_hat, grad_hat, departs)
    journal.probe("form", beta=best["beta"], u_star=best["u_star"], g_star=best["g_star"])

    # --- 5. probabilite de defaillance ---------------------------------------
    from scipy.stats import norm
    pf_form = float(norm.cdf(-best["beta"]))
    pf_is, cov_is = importance_sampling(g_hat, best["u_star"], cfg["n_is"], cfg["is_seed"])
    journal.probe("proba", pf_form=pf_form, pf_is=pf_is, cov_is=cov_is,
                  beta_is=float(-norm.ppf(pf_is)) if pf_is > 0 else float("nan"))

    # --- 6. reperes independants du metamodele --------------------------------
    journal.probe("oracle", beta_exact=ls.beta_exact(), u_star_exact=ls.u_star_exact())
    journal.probe("cout", n_appels_solveur=n_appels[0])

    return {"beta": best["beta"], "pf_form": pf_form, "pf_is": pf_is,
            "u_star": best["u_star"], "n_appels": n_appels[0]}


def expected_feasibility(mu, sigma, eps_factor):
    """Critere EFF (Bichon), vectorise -- meme forme que `_eff_vectorized`."""
    from scipy.stats import norm
    eps = eps_factor * sigma
    s = np.where(sigma > 0, sigma, 1.0)
    a, b, c = -mu / s, (-eps - mu) / s, (eps - mu) / s
    eff = (mu * (2 * norm.cdf(a) - norm.cdf(b) - norm.cdf(c))
           - s * (2 * norm.pdf(a) - norm.pdf(b) - norm.pdf(c))
           + eps * (norm.cdf(c) - norm.cdf(b)))
    return np.where(sigma > 0, eff, 0.0)


def importance_sampling(g_hat, u_star, n, seed):
    """Tirage d'importance centre sur le point de conception."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, len(u_star))) + u_star
    ind = (g_hat(U) < 0).astype(float)
    w = np.exp(-U @ u_star + 0.5 * float(u_star @ u_star))
    contrib = ind * w
    pf = float(contrib.mean())
    cov = float(contrib.std(ddof=1) / np.sqrt(n) / pf) if pf > 0 else float("inf")
    return pf, cov


# --------------------------------------------------------------------------- #
CONFIG = {
    "modele": "GEPCK",
    "n_doe": 24,
    "doe_seed": 20260825,
    "max_degree": 3,
    "u_max": 7.5,
    "n_grid": 120,
    "epsilon_factor": 2.0,
    "departs": [[0.0, 0.0], [2.0, -2.0], [-2.0, 2.0], [1.0, 1.0]],
    "n_is": 20000,
    "is_seed": 7,
    # geometrie reelle de test_pure_flexion (b=h=0.8, L=5.0, 24 HA32, d=0.691333)
    "section": {"b": 0.8, "d": 0.6913333333333334, "L": 5.0, "phi_mm": 32.0,
                "n_bars": 24, "F": 0.9, "gamma_c": 1.0, "gamma_s": 1.0,
                "fcm": 48.0, "cov_fc": 0.12, "fym": 550.0,
                "sig_fy": 30.149626863362670},
    "ds": {"model": "test_pure_flexion", "storage": r"C:\workspace\storage\admin\SF",
           "workdir": r"C:\tmp\baseline_ds.ds", "global_size": "0.05",
           "geo_min": 4, "fd_step": 0.05},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--solveur", choices=sorted(SOLVEURS), default="analytique")
    ap.add_argument("--repeat", type=int, default=1,
                    help="nombre de repetitions, pour mesurer le plancher de bruit")
    ap.add_argument("--nom", default=None)
    ap.add_argument("--note", default=None)
    args = ap.parse_args()

    nom = args.nom or ("flexion_" + args.solveur)
    resultats = []
    chemins = []

    for i in range(args.repeat):
        journal = telemetry.Journal(nom, config={**CONFIG, "solveur": args.solveur,
                                                 "repetition": i},
                                    note=args.note)
        telemetry.pin_seeds(journal)
        restaurer = telemetry.instrument_lib(journal)
        try:
            r = chaine(journal, CONFIG, args.solveur)
        finally:
            restaurer()
        chemins.append(journal.close(**r))
        resultats.append(r)
        print("[baseline] run %d/%d  beta=%.10f  Pf_IS=%.6e  %d appels solveur  -> %s"
              % (i + 1, args.repeat, r["beta"], r["pf_is"], r["n_appels"],
                 os.path.basename(chemins[-1])), flush=True)

    if args.repeat > 1:
        plancher = {}
        for cle in ("beta", "pf_form", "pf_is"):
            v = np.array([r[cle] for r in resultats], dtype=float)
            plancher[cle] = {"min": float(v.min()), "max": float(v.max()),
                             "etendue_relative": float(np.ptp(v) / abs(v.mean())) if v.mean() else 0.0}
        p = os.path.join(os.path.dirname(chemins[0]), "plancher_de_bruit.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"n_repetitions": args.repeat, "grandeurs": plancher,
                       "journaux": [os.path.basename(c) for c in chemins]}, fh, indent=1)
        print("\n[baseline] plancher de bruit sur %d runs :" % args.repeat)
        for cle, d in plancher.items():
            print("   %-8s etendue relative %.3e" % (cle, d["etendue_relative"]))
        print("   ecrit : %s" % p)

    print("\n[baseline] journal de reference : %s" % chemins[-1])


if __name__ == "__main__":
    main()
