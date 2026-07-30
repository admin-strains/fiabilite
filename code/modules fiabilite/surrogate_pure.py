"""
Fonctions et classes de construction des surrogates (PCE, KRG, GEK, GEPCK).
Aucune dependance sur run_HF.
"""
import numpy as np
import openturns as ot
from scipy.stats import norm
from smt.surrogate_models import GEKPLS
from branche1 import predict_gepck, predict_gradient_gepck
from config_utilisateur import n_var, eff_bounds_min, eff_bounds_max, dist_jointe
from config_pardefaut import q, max_degree, epsilon_factor, n_sp, print_gepck_calls
from config_autres import do_KRG, do_GEK


def build_starting_points():
    dist_U = ot.JointDistribution([ot.Uniform(eff_bounds_min[i], eff_bounds_max[i]) for i in range(n_var)])
    lhs = ot.LHSExperiment(dist_U, n_sp)
    sa = ot.SimulatedAnnealingLHS(lhs, ot.SpaceFillingMinDist())
    return np.array(sa.generate())  # shape (n_sp, n_var)

def build_Y_aug(yt, all_grad):
    """
    Construit le vecteur gradient-enhanced y_dot (eq. 6 Zuhal et al.).
    y_dot = [y^1,...,y^n, dg/du1^1,...,dg/du1^n, ..., dg/dum^1,...,dg/dum^n]^T
    Shape : (n0*(1+n_var),)
    """
    y_flat      = yt.flatten()                                         # (n0,)
    grad_blocks = [all_grad[:, j] for j in range(all_grad.shape[1])]  # n_var blocs de (n0,)
    return np.concatenate([y_flat] + grad_blocks)                      # (n0*(1+n_var),)

def build_metamodel_PCE(xt, y_hf):
    # 1. INITIALISATION : DOE ET DISTRIBUTION
    inputSample = ot.Sample(xt)
    outputSample = ot.Sample(y_hf)
    n0 = xt.shape[0]
    dist_X = dist_jointe()
    dist_U = dist_X.getStandardDistribution()

    # 2. BASE DE CANDIDATS : TYPE, ENUMERATION, DEGRE
    n_var = inputSample.getDimension()
    enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(n_var, q)
    basis = ot.OrthogonalProductPolynomialFactory([ot.HermiteFactory()] * n_var, enumerateFunction)
    basis_size = enumerateFunction.getBasisSizeFromTotalDegree(max_degree)
    basisStrategy = ot.FixedStrategy(basis, basis_size)

    # 3. PROPOSITION / PROJECTION / SELECTION
    selectionStrategy = ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut())
    projectionStrategy = ot.LeastSquaresStrategy(selectionStrategy)

    # 4. RESULTAT
    algo = ot.FunctionalChaosAlgorithm(inputSample, outputSample, dist_U, basisStrategy, projectionStrategy)
    algo.run()
    result = algo.getResult()
    n_active = result.getCoefficients().getSize()
    print(f"PCE construite : basis_size={basis_size}, coefficients actifs LARS={n_active}", flush=True)
    indices = result.getIndices()
    coeffs  = result.getCoefficients()
    terms = []
    for k in range(indices.getSize()):
        mi = enumerateFunction(indices[k])
        a, b = int(mi[0]), int(mi[1])
        if   a == 0 and b == 0: label = "1"
        elif a == 0:            label = f"H{b}(u2)"
        elif b == 0:            label = f"H{a}(u1)"
        else:                   label = f"H{a}(u1)*H{b}(u2)"
        terms.append(f"{coeffs[k, 0]:+.4f}*{label}")
    print(f"  PCE termes : {' '.join(terms)}", flush=True)
    metamodel = result.getMetaModel()
    return metamodel

def calculate_PCE(xt, y_hf, all_grad_hf, metamodel_PCE):
    U_doe = ot.Sample(xt)
    y_PCE = np.array(metamodel_PCE(U_doe))
    n_var = U_doe.getDimension()
    n0 = U_doe.getSize()
    dist_X = dist_jointe()
    T = dist_X.getIsoProbabilisticTransformation()
    T_inv = dist_X.getInverseIsoProbabilisticTransformation()
    all_grad_PCE = np.zeros((n0, n_var))
    for i in range(n0):
        grad_pce_u = metamodel_PCE.gradient(U_doe[i])
        for j in range(n_var):
            all_grad_PCE[i, j] = grad_pce_u[j, 0]
    return y_PCE, all_grad_PCE

class PCKRGFunction(ot.OpenTURNSPythonFunction):
    def __init__(self, g_pce, g_krg):
        super().__init__(n_var, 1)
        self.g_pce = g_pce
        self.g_krg = g_krg

    def _exec(self, u):
        return [self.g_pce(u)[0] + self.g_krg(u)[0]]

    def _exec_sample(self, U):
        U_ot = ot.Sample(U)
        Z_pce = np.array(self.g_pce(U_ot))[:, 0]
        Z_krg = np.array(self.g_krg(U_ot))[:, 0]
        return (Z_pce + Z_krg).reshape(-1, 1).tolist()

    def _gradient(self, u):
        u_ot     = ot.Point(list(u))
        grad_pce = self.g_pce.gradient(u_ot)
        grad_krg = self.g_krg.gradient(u_ot)
        return [[grad_pce[i, 0] + grad_krg[i, 0]] for i in range(n_var)]

class oldGEPCKFunction(ot.OpenTURNSPythonFunction):
    def __init__(self, g_pce, sm_gepck):
        super().__init__(n_var, 1)
        self.g_pce  = g_pce
        self.sm     = sm_gepck

    def _exec(self, u):
        y_pce = self.g_pce(ot.Point(list(u)))[0]
        y_gek = self.sm.predict_values(np.array(u).reshape(1, -1)).item()
        return [y_pce + y_gek]

    def _exec_sample(self, U):
        U_ot = ot.Sample(U)
        Z_pce = np.array(self.g_pce(U_ot))[:, 0]
        Z_gek = self.sm.predict_values(np.array(U))[:, 0]
        return (Z_pce + Z_gek).reshape(-1, 1).tolist()

    def _exec_sigma(self, u):
        sm = self.sm
        n  = sm.nt; d = sm.X_norma.shape[1]
        W  = sm.coeff_pls; th = sm.optimal_theta
        s2 = float(sm.optimal_par['sigma2'])
        th_eff = (W**2) @ th
        x_n = (np.array(u).reshape(-1) - sm.X_offset) / sm.X_scale
        Xn  = sm.X_norma
        df  = x_n[None, :] - Xn
        kf  = np.exp(-np.dot(df**2, th_eff))
        kd  = (2.0 * kf[:, None] * df * th_eff[None, :]).reshape(-1)
        dff = Xn[:, None, :] - Xn[None, :, :]
        K_ff = np.exp(-np.einsum('ijk,k->ij', dff**2, th_eff))
        K_fd = (2.0 * K_ff[:, :, None] * dff * th_eff[None, None, :]).reshape(n, n*d)
        B_mat = dff * th_eff[None, None, :]
        term1 = 2.0 * np.diag(th_eff)
        term2 = 4.0 * np.einsum('ija,ijb->ijab', B_mat, B_mat)
        K_dd  = (K_ff[:, :, None, None] * (term1 - term2)).transpose(0,2,1,3).reshape(n*d, n*d)
        K_tot = np.block([[K_ff, K_fd], [K_fd.T, K_dd]])
        K_tot += 1e-10 * np.eye(K_tot.shape[0])
        k = np.concatenate([kf, kd])
        try:
            B = max(0.0, 1.0 - k @ np.linalg.solve(K_tot, k))
            return float(np.sqrt(s2 * B))
        except np.linalg.LinAlgError:
            return float(np.sqrt(sm.predict_variances(np.array(u).reshape(1, -1)).item()))

    def _gradient(self, u):
        u_np     = np.array(u).reshape(1, -1)
        grad_pce = self.g_pce.gradient(ot.Point(list(u)))   # OT Matrix (n_var, 1)
        return [[grad_pce[i, 0] + self.sm.predict_derivatives(u_np, i).item()]
                for i in range(n_var)]

class GEPCKFunction(ot.OpenTURNSPythonFunction):
    def __init__(self, fm):
        super().__init__(n_var, 1)
        self.fm = fm
        self.n_eval_calls = 0
        self.n_grad_calls = 0

    def _exec(self, u):
        u_np  = np.array(u).reshape(1, -1)
        g_val = float(predict_gepck(self.fm, u_np)[0, 0])
        self.n_eval_calls += 1
        if print_gepck_calls:
            print(f"[GEPCK eval #{self.n_eval_calls:3d}] u=[{float(u[0]):+.4f}, {float(u[1]):+.4f}]"
                  f"  g={g_val:+.6f}", flush=True)
        return [g_val]

    def _exec_sample(self, U):
        U_np = np.array(U)
        return predict_gepck(self.fm, U_np)[:, 0:1].tolist()

    def _exec_sigma(self, u):
        u_np = np.array(u).reshape(1, -1)
        _, YSig2 = predict_gepck(self.fm, u_np, return_var=True)
        return float(np.sqrt(max(0.0, float(YSig2[0, 0]))))

    def _gradient(self, u):
        u_np  = np.array(u).reshape(1, -1)
        G     = predict_gradient_gepck(self.fm, u_np)   # (1, Mred)
        grad  = [float(G[0, i]) for i in range(self.fm['Mred'])]
        g_val = float(predict_gepck(self.fm, u_np)[0, 0])
        self.n_grad_calls += 1
        print(f"[GEPCK grad #{self.n_grad_calls:3d}] u=[{float(u[0]):+.4f}, {float(u[1]):+.4f}]"
              f"  g={g_val:+.6f}  grad=[{grad[0]:+.6f}, {grad[1]:+.6f}]", flush=True)
        return [[v] for v in grad]

class BoundSurrogateFunction(ot.OpenTURNSPythonFunction):
    """
    g_bound(u) = g_ot(u) + sign * 2 * sigma_func(u)
    Wrappeur externe pour les bornes de confiance du surrogate.
    """
    def __init__(self, g_ot, sigma_func, sign):
        super().__init__(n_var, 1)
        self._g_ot       = g_ot
        self._sigma_func = sigma_func
        self._sign       = sign   # +1 ou -1

    def _exec(self, u):
        u_pt  = ot.Point(list(u))
        mu    = self._g_ot(u_pt)[0]
        sigma = self._sigma_func(u_pt)
        return [mu + self._sign * 2.0 * sigma]

    def _exec_sample(self, U):
        _fm = getattr(getattr(self._sigma_func, '__self__', None), 'fm', None)
        if _fm is not None:
            U_np = np.array(U)
            mu_arr, sig2_arr = predict_gepck(_fm, U_np, return_var=True)
            mu    = mu_arr[:, 0]
            sigma = np.sqrt(np.maximum(0.0, sig2_arr[:, 0]))
            result = mu + self._sign * 2.0 * sigma
            return result.reshape(-1, 1).tolist()
        return [[self._exec(u)[0]] for u in U]

def build_metamodel_KRG(xt, yt):
    n_var = xt.shape[1]
    basis = ot.ConstantBasisFactory(n_var).build()
    covarianceModel = ot.SquaredExponential([1.0] * n_var)
    algo_KRG = ot.KrigingAlgorithm(xt, yt, covarianceModel, basis)
    if do_KRG:
        algo_KRG.setOptimizationBounds(ot.Interval([1.0] * n_var, [5.0] * n_var))
    algo_KRG.run()
    result = algo_KRG.getResult()
    cov_opt = result.getCovarianceModel()
    print(f"  KRG theta={list(cov_opt.getScale())}  sigma={list(cov_opt.getAmplitude())}", flush=True)
    return result.getMetaModel(), result

def build_metamodel_GEK(xt, yt, all_grad):
    xlimits = np.column_stack([xt.min(axis=0) - 1, xt.max(axis=0) + 1])
    sm = GEKPLS(
        n_comp=2,
        theta0=[1e-2, 1e-2],
        theta_bounds=[1.0, 5.0],
        corr="squar_exp",
        poly="constant",
        xlimits=xlimits,
        print_global=False,
    )
    sm.set_training_values(xt, yt)
    for j in range(n_var):
        sm.set_training_derivatives(xt, all_grad[:, j].reshape(-1, 1), j)
    sm.train()
    print(f"  GEK theta={list(sm.optimal_theta)}  sigma={float(np.sqrt(sm.optimal_par['sigma2'])):.6f}", flush=True)
    return sm

class GEKPLSFunction(ot.OpenTURNSPythonFunction):
    def __init__(self, surrogate):
        super().__init__(n_var, 1)
        self.sm = surrogate

    def _exec(self, u):
        return [self.sm.predict_values(np.array(u).reshape(1, -1)).item()]

    def _exec_sample(self, U):
        return self.sm.predict_values(np.array(U)).tolist()

    def _exec_sigma(self, u):
        sm = self.sm
        n  = sm.nt; d = sm.X_norma.shape[1]
        W  = sm.coeff_pls; th = sm.optimal_theta
        s2 = float(sm.optimal_par['sigma2'])
        th_eff = (W**2) @ th
        x_n = (np.array(u).reshape(-1) - sm.X_offset) / sm.X_scale
        Xn  = sm.X_norma
        df  = x_n[None, :] - Xn
        kf  = np.exp(-np.dot(df**2, th_eff))
        kd  = (2.0 * kf[:, None] * df * th_eff[None, :]).reshape(-1)
        dff = Xn[:, None, :] - Xn[None, :, :]
        K_ff = np.exp(-np.einsum('ijk,k->ij', dff**2, th_eff))
        K_fd = (2.0 * K_ff[:, :, None] * dff * th_eff[None, None, :]).reshape(n, n*d)
        B_mat = dff * th_eff[None, None, :]
        term1 = 2.0 * np.diag(th_eff)
        term2 = 4.0 * np.einsum('ija,ijb->ijab', B_mat, B_mat)
        K_dd  = (K_ff[:, :, None, None] * (term1 - term2)).transpose(0,2,1,3).reshape(n*d, n*d)
        K_tot = np.block([[K_ff, K_fd], [K_fd.T, K_dd]])
        K_tot += 1e-10 * np.eye(K_tot.shape[0])
        k = np.concatenate([kf, kd])
        try:
            B = max(0.0, 1.0 - k @ np.linalg.solve(K_tot, k))
            return float(np.sqrt(s2 * B))
        except np.linalg.LinAlgError:
            return float(np.sqrt(sm.predict_variances(np.array(u).reshape(1, -1)).item()))

    def _gradient(self, u):
        u_np = np.array(u).reshape(1, -1)
        return [[self.sm.predict_derivatives(u_np, kx).item()] for kx in range(n_var)]

class EFFFunction(ot.OpenTURNSPythonFunction):
    def __init__(self, g_ot, sigma_func):
        super().__init__(n_var, 1)
        self.g_ot = g_ot
        self.sigma_func = sigma_func

    def _exec(self, u):
        u = ot.Point(u)
        sigmaG  = self.sigma_func(u)
        if sigmaG <= 0.0:
            return [0.0]
        muG     = self.g_ot(u)[0]
        epsilon = epsilon_factor * sigmaG
        t1 = -muG / sigmaG
        t2 = (epsilon + muG) / sigmaG
        t3 = (epsilon - muG) / sigmaG
        return [2*muG*norm.cdf(t1) - (epsilon+muG)*norm.cdf(-t2) + (epsilon-muG)*norm.cdf(t3) + sigmaG*(-2*norm.pdf(t1) + norm.pdf(t2) + norm.pdf(t3))]
