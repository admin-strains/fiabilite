# -*- coding: utf-8 -*-
# ==========================================================================================
# Smoke test T5-1/T5-2 (2026-07-06, MM) : la conversion gradients X->U deplacee dans
# run_one_SOL donne un all_grad IDENTIQUE a l'ancienne conversion dans build_DOE.
#
# Le changement de code est purement : calculer  grad_U = J_Tinv(u)^T . grad_X  au point
# u = T(x)  (dans run_one_SOL, x = valeur X du point)  AU LIEU de  u = U_doe[i]
# (ancienne boucle dans build_DOE, x = T_inv(U_doe[i])).
#
# Ces deux voies sont identiques SSI  T(T_inv(U)) == U  (roundtrip exact) ET
# J_Tinv.gradient est le meme aux deux points. On le prouve ici SANS solve, sur les
# lois EXACTES des deux AC :
#   - moulin_blanc : 2 x Normal (fy1, fy2)
#   - cantilever_s : Normal (fy_top) + Uniform(0,1) (s, variable de position)
# Tolerance stricte 1e-9 (roundtrip + gradient). Si ca passe, all_grad est inchange.
#
#   C:\python3\python.exe tests\test_gradient_XtoU_identity.py
# ==========================================================================================
import numpy as np
import openturns as ot

def _check_dist(label, marginals, grad_X_samples):
    dist_X = ot.JointDistribution(marginals)
    T      = dist_X.getIsoProbabilisticTransformation()
    T_inv  = dist_X.getInverseIsoProbabilisticTransformation()
    n_var  = len(marginals)

    # points DOE en U (comme le LHS de build_DOE)
    dist_U = dist_X.getStandardDistribution()
    ot.RandomGenerator.SetSeed(12345)
    U_doe = ot.LHSExperiment(dist_U, 12).generate()

    max_du = 0.0
    max_dgrad = 0.0
    for i in range(U_doe.getSize()):
        U_i = U_doe[i]
        x   = T_inv(U_i)              # X du point (ce que run_one_SOL recoit)
        u   = T(x)                    # run_one_SOL recalcule u = T(x)
        # (1) roundtrip : u == U_i ?
        du = max(abs(u[j] - U_i[j]) for j in range(n_var))
        max_du = max(max_du, du)
        # (2) gradient X->U : identique en u (nouveau) et en U_i (ancien) ?
        gX = ot.Point(list(grad_X_samples[i % len(grad_X_samples)]))
        gU_new = T_inv.gradient(u).transpose()   * gX      # run_one_SOL
        gU_old = T_inv.gradient(U_i).transpose() * gX      # ancien build_DOE
        dgrad = max(abs(gU_new[j] - gU_old[j]) for j in range(n_var))
        max_dgrad = max(max_dgrad, dgrad)

    ok = (max_du < 1e-9) and (max_dgrad < 1e-9)
    print(("OK " if ok else "!! ECHEC ") +
          f": {label:16s} roundtrip max|u-U|={max_du:.2e}  grad max|dU_new-dU_old|={max_dgrad:.2e}")
    return ok

print("\n############ SMOKE TEST T5-1/T5-2 : conversion X->U identique (run_one_SOL vs build_DOE) ############")
SIGMA = np.sqrt(19.0**2 + 22.0**2 + 8.0**2)  # ~30 MPa, comme AC_moulin_blanc_2fy
gX = [[-1.234, 0.567], [2.1, -3.4], [0.0, 5.5], [-7.7, 1.1]]

ok = True
# moulin_blanc : fy1, fy2 ~ Normal(550, ~30)
ok &= _check_dist("moulin_blanc 2fy", [ot.Normal(550.0, SIGMA), ot.Normal(550.0, SIGMA)], gX)
# cantilever_s : fy_top ~ Normal(550, ~30), s ~ Uniform(0,1)
ok &= _check_dist("cantilever fy+s", [ot.Normal(550.0, SIGMA), ot.Uniform(0.0, 1.0)], gX)
# variante : LogNormal (fc) pour couvrir une loi non-lineaire
_sig = np.sqrt(np.log(1 + 0.12**2)); _mu = np.log(48.0) - 0.5 * _sig**2
ok &= _check_dist("normal+lognormal", [ot.Normal(550.0, SIGMA), ot.LogNormal(_mu, _sig, 0.0)], gX)

assert ok, "ECHEC : la conversion X->U deplacee ne reproduit PAS l'ancienne (all_grad changerait)"
print("\n############ VALIDE : all_grad inchange par le deplacement de la conversion ############")
