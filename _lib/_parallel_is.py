# -*- coding: utf-8 -*-
# Module REUTILISABLE : Importance Sampling ADAPTATIF + PARALLELISABLE (sonde + ramp-up).
# C'est la "demarche discutee" prete a etre branchee dans run_IS de l'AC.
#
# Principe (valide par _test_probe_rampup.py) :
#   1. SONDE sequentielle adaptative : tire des blocs un par un, COV verifie a CHAQUE bloc.
#      -> si COV<=cible PENDANT la sonde : STOP (= comportement sequentiel actuel, AUCUN pool).
#   2. sinon RAMP-UP : pool de K process (BLAS=1/worker), chaque ronde = K x CHUNK blocs
#      EN PARALLELE, on recolle (sommes partielles additives, recollage EXACT), COV verifie
#      par ronde, stop des COV<=cible ou cap atteint.
#
# Mono-modal (densite N(u*, I)) : couvre l'IS des BANDES (g, g+2s, g-2s), qui EST le goulot.
# Bande via 'sign' : 0 -> g moyen ; +1 -> g+2s ; -1 -> g-2s. (return_var de predict_gepck.)
#
# Estimateur EXACT (poids d'importance) :
#   tire U ~ N(u*, I) ; g_band = mu(U) + sign*2*sqrt(var(U)) ; w = exp(0.5|u*|^2 - U.u*)
#   c = (g_band<0) ? w : 0 ;  Pf = mean(c) ;  COV = std(c)/Pf/sqrt(N).

import os, sys, json, time, warnings, math
sys.path.insert(0, r"C:\workspace\fiabilite\_lib")
import numpy as np
from branche1 import fit_gepck, predict_gepck
from threadpoolctl import threadpool_limits
import concurrent.futures as cf

NV = 2

# --------- construction du surrogate depuis l'etat (xt,yt,all_grad,max_degree) ----------
def build_fm_from_state(xt, yt, all_grad, max_degree):
    xt = np.asarray(xt, float); yt = np.asarray(yt, float); ag = np.asarray(all_grad, float)
    Y = np.concatenate([yt.flatten()] + [ag[:, j] for j in range(NV)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_gepck(xt, Y,
                         {'Mode': 'optimal', 'PCE': {'Degree': list(range(1, max_degree + 1)), 'Method': 'LARS'}},
                         [{'Type': 'Gaussian', 'Parameters': [0.0, 1.0]}] * NV,
                         {'Type': 'Independent', 'Parameters': np.eye(NV)})

# --------- un bloc : sommes partielles (n, sum_c, sumsq_c) -- combinables exactement ---------
def block_partial(fm, u_star, half_ustar2, sign, seed, n):
    rng = np.random.default_rng(seed)
    U   = rng.standard_normal((n, NV)) + u_star
    if sign == 0:
        g = predict_gepck(fm, U)[:, 0]
    else:
        mu, var = predict_gepck(fm, U, return_var=True)
        g = mu[:, 0] + sign * 2.0 * np.sqrt(np.maximum(0.0, var[:, 0]))
    c = np.where(g < 0.0, np.exp(half_ustar2 - U @ u_star), 0.0)
    return n, float(c.sum()), float((c * c).sum())

def cov_from(N, S, SS):
    pf = S / N
    if pf <= 0:
        return pf, float('inf')
    return pf, math.sqrt(max(SS / N - pf * pf, 0.0) / N) / pf

# --------- worker multi-process (le surrogate est reconstruit 1x/worker dans _init) ---------
_W = {}
def _init(state):
    _W['fm']    = build_fm_from_state(state['xt'], state['yt'], state['all_grad'], state['max_degree'])
    _W['ustar'] = np.asarray(state['u_star'], float)
    _W['half']  = 0.5 * float(_W['ustar'] @ _W['ustar'])
    _W['sign']  = int(state['sign'])
    _W['block'] = int(state['block'])
    _W['chunk'] = int(state['chunk'])
def _worker(seed):
    with threadpool_limits(limits=1):
        N = 0; S = 0.0; SS = 0.0
        for j in range(_W['chunk']):
            n_i, s_i, ss_i = block_partial(_W['fm'], _W['ustar'], _W['half'], _W['sign'],
                                           seed * 1000 + j, _W['block'])
            N += n_i; S += s_i; SS += ss_i
        return N, S, SS

# ============================ LA DEMARCHE : sonde + ramp-up ============================
def adaptive_is(fm, state, u_star, sign=0, cov_target=0.05, cap_blocks=10000,
                block=10000, K=16, chunk=8, probe_blocks=16, verbose=False):
    """IS adaptatif mono-modal. Renvoie dict(pf, cov, n_blocks, n_evals, mode, t).
    fm : surrogate deja construit (pour la SONDE sequentielle, dans le process courant).
    state : dict pour reconstruire fm dans les workers (xt/yt/all_grad/max_degree) +
            u_star/sign/block/chunk (ajoutes ici)."""
    ustar = np.asarray(u_star, float)
    half  = 0.5 * float(ustar @ ustar)
    t0 = time.perf_counter()

    # ---- 1. SONDE sequentielle adaptative (COV verifie a chaque bloc) ----
    N = 0; S = 0.0; SS = 0.0; nb = 0
    pf, cov = 0.0, float('inf')
    for _ in range(probe_blocks):
        n_i, s_i, ss_i = block_partial(fm, ustar, half, sign, seed=nb, n=block)
        N += n_i; S += s_i; SS += ss_i; nb += 1
        pf, cov = cov_from(N, S, SS)
        if cov <= cov_target:
            return dict(pf=pf, cov=cov, n_blocks=nb, n_evals=N,
                        mode="sonde (sequentiel, pas de pool)", t=time.perf_counter() - t0)
        if nb >= cap_blocks:
            return dict(pf=pf, cov=cov, n_blocks=nb, n_evals=N,
                        mode="sonde -> CAP", t=time.perf_counter() - t0)

    # ---- 2. RAMP-UP : rondes paralleles (le cas est "dur") ----
    wstate = dict(state); wstate.update(u_star=list(ustar), sign=sign, block=block, chunk=chunk)
    nr = 0
    with cf.ProcessPoolExecutor(max_workers=K, initializer=_init, initargs=(wstate,)) as ex:
        while True:
            parts = list(ex.map(_worker, [10_000_000 + nr * K + i for i in range(K)]))
            for n_i, s_i, ss_i in parts:
                N += n_i; S += s_i; SS += ss_i; nb += n_i // block
            nr += 1
            pf, cov = cov_from(N, S, SS)
            if cov <= cov_target or nb >= cap_blocks:
                break
    return dict(pf=pf, cov=cov, n_blocks=nb, n_evals=N, n_rounds=nr,
                mode=f"sonde + ramp-up ({nr} rondes par.)", t=time.perf_counter() - t0)

# ================================ validation standalone ================================
if __name__ == "__main__":
    DS = r"C:\workspace\fiabilite\cas_test\test_pure_flexion.ds"
    d  = json.load(open(os.path.join(DS, "restart_state_2fy.json")))
    state = dict(xt=d["xt"], yt=d["yt"], all_grad=d["all_grad"], max_degree=d["max_degree"])
    u_star = d["modes"][0]["u_star"]
    fm = build_fm_from_state(state['xt'], state['yt'], state['all_grad'], state['max_degree'])
    print(f"u*={[round(v,3) for v in u_star]} | Pf_dump(g moyen)={d['IS']['Pf']:.3e}\n", flush=True)

    from scipy.stats import norm
    for sign, lbl in [(0, "g moyen (mu)"), (+1, "g+2s (borne sup)"), (-1, "g-2s (borne inf)")]:
        r = adaptive_is(fm, state, u_star, sign=sign, cov_target=0.05,
                        cap_blocks=3000, K=16, chunk=8, probe_blocks=16)
        beta = float(-norm.ppf(r['pf'])) if r['pf'] > 0 else float('nan')
        print(f"[{lbl:18s}] Pf={r['pf']:.4e}  beta={beta:.3f}  COV={r['cov']:.4f}  "
              f"blocs={r['n_blocks']}  t={r['t']:.2f}s  [{r['mode']}]", flush=True)
    print("\nFINI", flush=True)
