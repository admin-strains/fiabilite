"""
FORM multimodal et tirage d'importance.

Extrait de `AC3_pure_flexion.py` / `AC3_moulinblanc.py`, ou ces fonctions
etaient definies dans `if __name__ == '__main__':`. Elles y sont identiques
caractere pour caractere. PHASE 3 du plan de nettoyage.

Corps VERBATIM ; seules les variables libres de `main` deviennent des
parametres explicites :

    FORM_all_modes  n_var, n_max_FORM, tol_FORM, do_FORM_filter,
                    eff_bounds_min, eff_bounds_max
    run_IS          n_var, n_IS, cov_IS
    run_IS_proj     idem, plus `idx_position` -- l'original appelait
                    `_find_position_var_index()`, qui lisait `params_names` et
                    `PARAM_CONFIG` ; l'index est desormais fourni par
                    l'appelant, ce qui coupe le lien avec la configuration.

`BoundSurrogateFunction` conserve son contournement par introspection
(`getattr(getattr(sigma_func, '__self__', None), 'fm', None)`) et sa selection
du predicteur : le nettoyer serait une refonte, pas une extraction.

Cette couche a besoin d'OpenTURNS et de scikit-learn (DBSCAN), pas de Digital
Structure.
"""

import numpy as np
import openturns as ot
from sklearn.cluster import DBSCAN


def form_all_modes(starting_points, tol_all_modes, event, n_var,
                   n_max_FORM, tol_FORM, do_FORM_filter=False,
                   eff_bounds_min=None, eff_bounds_max=None):
    """
    Multi-start FORM + DBSCAN pour identifier les modes de défaillance.
    - Chaque cluster DBSCAN = un mode distinct.
    - u* isolés (label -1) = descentes mal convergées, ignorées.
    """
    all_u_star   = []   # u* de chaque run réussi
    all_results  = []   # FORMResult correspondant
    all_sp       = []   # point de départ correspondant
    n_total = len(starting_points)

    for k, sp in enumerate(starting_points):
        print(f"  FORM {k+1}/{n_total}...", flush=True)
        try:
            solver = ot.AbdoRackwitz()
            solver.setStartingPoint(sp.tolist())
            solver.setMaximumIterationNumber(n_max_FORM)
            solver.setCheckStatus(False)
            solver.setMaximumConstraintError(tol_FORM)
            form_i = ot.FORM(solver, event)
            form_i.run()
            r_i    = form_i.getResult()
            u_star = np.array(r_i.getStandardSpaceDesignPoint())
            all_u_star.append(u_star)
            all_results.append(r_i)
            all_sp.append(sp)
            print(f"  [sp={[round(v,3) for v in np.array(sp)]}, "
                f"u*={[round(v,3) for v in u_star]}, "
                f"beta={r_i.getHasoferReliabilityIndex():.4f}]", flush=True)
        except Exception as e:
            print(f"  [sp={[round(v,3) for v in np.array(sp)]}, "
                f"ECHEC ({type(e).__name__})]", flush=True)

    # --- Filtrer les u* hors bornes EFF ---
    if do_FORM_filter:
        _filtered = [(u, r, s) for u, r, s in zip(all_u_star, all_results, all_sp)
                     if all(eff_bounds_min[j] <= u[j] <= eff_bounds_max[j] for j in range(n_var))]
        _n_rejected = len(all_u_star) - len(_filtered)
        if _n_rejected > 0:
            print(f"  [FORM FILTER] {_n_rejected} u* hors bornes EFF rejetes", flush=True)
        all_u_star  = [x[0] for x in _filtered]
        all_results = [x[1] for x in _filtered]
        all_sp      = [x[2] for x in _filtered]

    if not all_u_star:
        return [], []

    # --- Cas 1 point : pas de DBSCAN ---
    if len(all_u_star) == 1:
        print(f"\n1 mode(s) distinct(s) (1 seul point de depart, pas de DBSCAN) :", flush=True)
        u = [round(v, 3) for v in all_results[0].getStandardSpaceDesignPoint()]
        print(f"  mode 1 : beta={all_results[0].getHasoferReliabilityIndex():.4f}  "
              f"Pf={all_results[0].getEventProbability():.3e}  u*={u}", flush=True)
        return [all_results[0]], [all_sp[0]]

    # --- DBSCAN ---
    U_all  = np.array(all_u_star)          # shape (n_runs_ok, n_var)
    db     = DBSCAN(eps=tol_all_modes, min_samples=2).fit(U_all)
    labels = db.labels_

    n_noise = np.sum(labels == -1)
    if n_noise > 0:
        print(f"  {n_noise} descente(s) mal convergée(s) ignorée(s) (bruit DBSCAN)", flush=True)

    # --- Un mode par cluster : FORMResult avec beta minimal ---
    modes     = []
    best_sps  = []
    for lbl in sorted(set(labels) - {-1}):
        idx_cluster = [i for i, l in enumerate(labels) if l == lbl]
        best_i = min(idx_cluster,
                    key=lambda i: all_results[i].getHasoferReliabilityIndex())
        modes.append(all_results[best_i])
        best_sps.append(all_sp[best_i])

    order = sorted(range(len(modes)), key=lambda i: modes[i].getHasoferReliabilityIndex())
    modes    = [modes[i]    for i in order]
    best_sps = [best_sps[i] for i in order]

    print(f"\n{len(modes)} mode(s) distinct(s) "
        f"(DBSCAN eps={tol_all_modes}, min_samples=2) :", flush=True)
    for i, m in enumerate(modes):
        u = [round(v, 3) for v in m.getStandardSpaceDesignPoint()]
        print(f"  mode {i+1} : beta={m.getHasoferReliabilityIndex():.4f}  "
            f"Pf={m.getEventProbability():.3e}  u*={u}", flush=True)

    return modes, best_sps


def bound_surrogate_function(g_ot, sigma_func, sign, n_var, predict):
    """Enveloppe g +/- 2 sigma, pour encadrer le beta du metamodele.

        sign = +1  ->  borne superieure  g_sup = mu_g + 2 sigma_g
        sign = -1  ->  borne inferieure  g_inf = mu_g - 2 sigma_g

    Remplace la classe `BoundSurrogateFunction` des scripts AC. `n_var` et le
    predicteur etaient des variables libres de `main` ; le predicteur remplace
    aussi le branchement sur `do_PCK`, desormais fait par l'appelant.

    Aucun `_gradient` n'est defini : OpenTURNS passe en differences finies si
    FORM est appele dessus. C'etait deja le cas.

    S'utilise comme avant : `ot.Function(bound_surrogate_function(...))`.
    """

    class _Bound(ot.OpenTURNSPythonFunction):
        def __init__(self):
            super().__init__(n_var, 1)
            self._g_ot = g_ot
            self._sigma_func = sigma_func
            self._sign = sign

        def _exec(self, u):
            u_pt = ot.Point(list(u))
            mu = self._g_ot(u_pt)[0]
            sigma = self._sigma_func(u_pt)
            return [mu + self._sign * 2.0 * sigma]

        def _exec_sample(self, U):
            _fm = getattr(getattr(self._sigma_func, '__self__', None), 'fm', None)
            if _fm is not None:
                U_np = np.array(U)
                mu_arr, sig2_arr = predict(_fm, U_np, return_var=True)
                mu = mu_arr[:, 0]
                sigma = np.sqrt(np.maximum(0.0, sig2_arr[:, 0]))
                result = mu + self._sign * 2.0 * sigma
                return result.reshape(-1, 1).tolist()
            return [[self._exec(u)[0]] for u in U]

    return _Bound()


def run_IS(modes, event, n_var, n_IS, cov_IS):
    """
    Importance Sampling post-FORM sur le surrogate.
    Distribution instrumentale : mixture de N(u*_i) pondérée par les Pf_FORM_i.
    Mono-modal : N simple centré sur u*.
    Retourne un ProbabilitySimulationResult.
    """
    if len(modes) == 1:
        g_imp = ot.Normal(n_var)
        g_imp.setMu(modes[0].getStandardSpaceDesignPoint())
        importance_dist = g_imp
    else:
        gaussians  = []
        pf_weights = []
        for m in modes:
            g_i = ot.Normal(n_var)
            g_i.setMu(m.getStandardSpaceDesignPoint())
            gaussians.append(g_i)
            pf_weights.append(m.getEventProbability())
        importance_dist = ot.Mixture(gaussians, pf_weights)

    experiment = ot.ImportanceSamplingExperiment(importance_dist, n_IS)
    std_event  = ot.StandardEvent(event)
    algo = ot.ProbabilitySimulationAlgorithm(std_event, experiment)
    algo.setMaximumCoefficientOfVariation(cov_IS)
    algo.setMaximumOuterSampling(n_IS)
    algo.run()
    return algo.getResult()


def run_IS_proj(modes, event_proj, n_var, n_IS, cov_IS, idx_position):
    """IS sur le surrogate projete (sans la variable de position).
    Extrait u* et Pf des modes FORM (n_var dims), enleve la composante position,
    et fait l'IS en dimension reduite sur event_proj.
    Si pas de variable de position, equivalent a run_IS."""
    idx_pos = idx_position

    idx_other = [i for i in range(n_var) if i != idx_pos] if idx_pos is not None else list(range(n_var))
    n_proj = len(idx_other)

    # extraire u* et Pf de chaque mode, projeter u*
    u_stars_proj = []
    pf_weights = []
    for m in modes:
        u_full = list(m.getStandardSpaceDesignPoint())
        u_proj = [u_full[i] for i in idx_other]
        u_stars_proj.append(u_proj)
        pf_weights.append(m.getEventProbability())

    if len(modes) == 1:
        g_imp = ot.Normal(n_proj)
        g_imp.setMu(u_stars_proj[0])
        importance_dist = g_imp
    else:
        gaussians = []
        for u_proj in u_stars_proj:
            g_i = ot.Normal(n_proj)
            g_i.setMu(u_proj)
            gaussians.append(g_i)
        importance_dist = ot.Mixture(gaussians, pf_weights)

    experiment = ot.ImportanceSamplingExperiment(importance_dist, n_IS)
    std_event  = ot.StandardEvent(event_proj)
    algo = ot.ProbabilitySimulationAlgorithm(std_event, experiment)
    algo.setMaximumCoefficientOfVariation(cov_IS)
    algo.setMaximumOuterSampling(n_IS)
    algo.run()
    return algo.getResult()


def print_results_IS(result_IS):
    pf   = result_IS.getProbabilityEstimate()
    cov  = result_IS.getCoefficientOfVariation()
    ci   = result_IS.getConfidenceLength(0.95)
    beta = float(-ot.Normal().computeQuantile(pf)[0])
    print(f"=== Importance Sampling ===", flush=True)
    print(f"  Pf_IS   = {pf:.4e}", flush=True)
    print(f"  beta_IS = {beta:.4f}", flush=True)
    print(f"  COV     = {cov:.4f}", flush=True)
    print(f"  IC 95%  = [{pf - ci/2:.4e}, {pf + ci/2:.4e}]", flush=True)
    print(f"  N_IS    = {result_IS.getOuterSampling()}", flush=True)


def _ecrire(message):
    print(message, flush=True)


def evenement_de_defaillance(g_ot, n_var):
    """`g < 0` dans l'espace standard, ou None si aucun metamodele.

    La loi est normale centree reduite en toutes dimensions : c'est la
    definition meme de l'espace standard, pas un choix d'etude. Les lois
    physiques sont deja passees dans la transformation isoprobabiliste.

    Rendre `None` plutot que lever : en HF pur il n'y a pas de metamodele, et
    l'etude continue sans evenement.
    """
    X = ot.RandomVector(ot.JointDistribution([ot.Normal(0, 1)] * n_var))
    if g_ot is None:
        return None
    return ot.ThresholdEvent(ot.CompositeRandomVector(g_ot, X), ot.Less(), 0.0)


def points_de_depart(xt, n_var, multistart):
    """Les points d'ou partent les recherches FORM.

    En multistart, tout le plan d'experiences PLUS l'origine ; sinon
    l'origine seule. L'origine est le point le plus probable de l'espace
    standard : elle appartient a toute recherche.
    """
    origine = [[0.0] * n_var]
    return np.vstack([xt, origine]) if multistart else np.array(origine)


def warm_start(modes, best_sps, g_ot, xt, yt, all_grad, *, n_var, tolerance,
               multistart, tol_all_modes, reajuster_et_evenement,
               rechercher_modes, tracer=_ecrire):
    """Relance FORM quand le mode dominant ne tombe pas sur `g = 0`.

    FORM cherche le point de l'etat limite le plus proche de l'origine. Si
    le metamodele evalue a `u*` rend une valeur loin de zero, c'est que la
    recherche s'est arretee ailleurs que sur la frontiere : le resultat ne
    veut rien dire. On verse alors `u*` au plan, on reajuste, et on relance
    la recherche complete.

    CE QUI EST VERSE AU PLAN EST LA PREDICTION DU METAMODELE, PAS UN APPEL
    SOLVEUR. Zero appel haute fidelite, donc -- mais aussi zero information
    nouvelle sur le vrai etat limite : le couple ajoute est `(u*, ce que le
    modele predit deja en u*)`. Ce que cela change reellement depend du
    metamodele (un krigeage interpolant y annule sa variance sans bouger sa
    moyenne) ; `test_109_form` le mesure au lieu de l'affirmer.

    Ne rend PAS le plan modifie -- seulement `(modes, best_sps)`. C'est le
    choix d'origine, conserve : les points fictifs ne survivent donc pas a
    l'appel, et le plan reel n'est pas contamine.
    """
    if not modes:
        return modes, best_sps
    u_star = modes[0].getStandardSpaceDesignPoint()
    g_val = g_ot(ot.Point(u_star))[0] if g_ot is not None else None
    if g_val is None or abs(g_val) <= tolerance:
        return modes, best_sps

    tracer("  [FORM WARM START] g(u*)=%.6f au-dela de %s : le mode dominant "
           "n'est pas sur l'etat limite, on reajuste et on recommence"
           % (g_val, tolerance))
    xt = np.vstack([xt, [np.array(u_star)]])
    yt = np.vstack([yt, [[g_val]]])
    grad_ot = g_ot.gradient(ot.Point(u_star))
    all_grad = np.vstack([all_grad,
                          np.array([[grad_ot[i, 0] for i in range(n_var)]])])
    evenement, xt = reajuster_et_evenement(xt, yt, all_grad)
    return rechercher_modes(points_de_depart(xt, n_var, multistart),
                            tol_all_modes, evenement)


def coupe_la_plus_parlante(best_result, n_var, coupe_par_defaut):
    """Les deux variables qui pesent le plus, les autres figees a `u*`.

    A plus de deux variables, une figure doit choisir son plan. Le prendre
    au hasard montrerait une coupe ou il ne se passe rien ; les facteurs
    d'importance de FORM designent celui ou tout se joue.
    """
    if best_result is None:
        return coupe_par_defaut
    importance = np.array(best_result.getImportanceFactors())
    # `int(...)` et non les `np.int64` que rend `argsort` : cette coupe part
    # dans du JSON -- les caches de grille et le dump de reprise la portent --
    # et `json.dumps(np.int64(0))` LEVE. Le Moulin Blanc passait par ici, la
    # flexion pure non : elle codait sa coupe finale en dur, avec des `int`.
    # Mesure du 29/08/2026 : `[HF CACHE] sauvegarde echouee (TypeError: Object
    # of type int64 is not JSON serializable)`, et le fichier vise laisse pour
    # mort.
    deux = [int(i) for i in np.argsort(importance)[::-1][:2]]
    u_star = np.array(best_result.getStandardSpaceDesignPoint())
    return (min(deux), max(deux),
            {i: float(u_star[i]) for i in range(n_var) if i not in deux})
