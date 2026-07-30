"""
Fonctions FORM, Importance Sampling, et projection surrogate.
"""
import openturns as ot
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize_scalar
from config_utilisateur import PARAM_CONFIG, params_names, n_var, tol_FORM, eff_bounds_min, eff_bounds_max
from config_pardefaut import n_max_FORM, do_FORM_filter, n_IS, cov_IS


def _is_position_var(sens):
    """Detecte si une region de sensibilite est une variable de position (axis='position')."""
    return sens.get('axis') == 'position'

def _find_position_var_index():
    """Retourne l'index de la variable de position dans params_names, ou None."""
    for i, p in enumerate(params_names):
        if _is_position_var(PARAM_CONFIG[p]['sens']):
            return i
    return None

def projection_surrogate(g_ot):
    """Si variable de position dans PARAM_CONFIG, retourne un g_ot projete
    g_proj(u_other) = min_p g_ot(u_full) sur la variable de position.
    Sinon retourne g_ot inchange."""
    idx_pos = _find_position_var_index()

    if idx_pos is None:
        return g_ot

    idx_other = [i for i in range(n_var) if i != idx_pos]
    n_proj = len(idx_other)

    class ProjectedSurrogateFunction(ot.OpenTURNSPythonFunction):
        def __init__(self):
            super().__init__(n_proj, 1)

        def _exec(self, u_reduced):
            def _obj(u_pos):
                u_full = [0.0] * n_var
                for k, idx in enumerate(idx_other):
                    u_full[idx] = float(u_reduced[k])
                u_full[idx_pos] = u_pos
                return float(g_ot(ot.Point(u_full))[0])
            # grille grossiere puis affinage (robuste pour W multi-creux)
            u_grid = np.linspace(-5.0, 5.0, 30)
            g_grid = [_obj(u) for u in u_grid]
            u_best = u_grid[np.argmin(g_grid)]
            res = minimize_scalar(_obj,
                                  bounds=(max(-5.0, u_best - 0.5),
                                          min(5.0, u_best + 0.5)),
                                  method='bounded',
                                  options={'xatol': 1e-4, 'maxiter': 200})
            return [res.fun]

    return ot.Function(ProjectedSurrogateFunction())

def FORM_all_modes(starting_points, tol_all_modes, event):
    """
    Multi-start FORM + DBSCAN pour identifier les modes de defaillance.
    - Chaque cluster DBSCAN = un mode distinct.
    - u* isoles (label -1) = descentes mal convergees, ignorees.
    """
    all_u_star   = []
    all_results  = []
    all_sp       = []
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
    U_all  = np.array(all_u_star)
    db     = DBSCAN(eps=tol_all_modes, min_samples=2).fit(U_all)
    labels = db.labels_

    n_noise = np.sum(labels == -1)
    if n_noise > 0:
        print(f"  {n_noise} descente(s) mal convergee(s) ignoree(s) (bruit DBSCAN)", flush=True)

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

def run_IS(modes, event):
    """
    Importance Sampling post-FORM sur le surrogate.
    Distribution instrumentale : mixture de N(u*_i) ponderee par les Pf_FORM_i.
    Mono-modal : N simple centre sur u*.
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

def run_IS_proj(modes, event_proj):
    """IS sur le surrogate projete (sans la variable de position).
    Extrait u* et Pf des modes FORM (n_var dims), enleve la composante position,
    et fait l'IS en dimension reduite sur event_proj.
    Si pas de variable de position, equivalent a run_IS."""
    idx_pos = _find_position_var_index()

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
