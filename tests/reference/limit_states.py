"""
Etats limites de reference -- oracles independants.

Ces fonctions ne dependent NI de STRAINS, NI d'OpenTURNS, NI de _lib/.
Elles servent de verite terrain : le harness compare le comportement du
metamodele (_lib) a ces valeurs calculees analytiquement ou par une
minimisation scalaire haute precision.

Convention : tout est exprime dans l'espace standard U ~ N(0,1)^M, comme
dans les scripts AC (marginales 'Gaussian' [0,1] + copule independante).
g(u) > 0 = domaine sur, g(u) < 0 = defaillance, g(u) = 0 = etat limite.
"""

import numpy as np
from scipy.optimize import minimize_scalar


# =========================================================================== #
# 1. Etat limite lineaire -- beta connu EXACTEMENT                            #
# =========================================================================== #
class LinearLS:
    """
    g(u) = beta_ref - (a . u) / ||a||

    L'etat limite est un hyperplan a la distance beta_ref de l'origine :
    beta exact = beta_ref, u* = beta_ref * a / ||a||.
    Test de non-regression le plus severe possible : toute la chaine
    (DOE -> metamodele -> FORM) doit retrouver beta_ref des que le
    metamodele contient le degre 1.
    """

    name = 'linear'

    def __init__(self, beta_ref=3.5, a=(1.0, 2.0)):
        self.beta_ref = float(beta_ref)
        self.a = np.asarray(a, dtype=float)
        self.n_var = self.a.size
        self._an = self.a / np.linalg.norm(self.a)

    def g(self, U):
        U = np.atleast_2d(np.asarray(U, dtype=float))
        return self.beta_ref - U @ self._an

    def grad(self, U):
        U = np.atleast_2d(np.asarray(U, dtype=float))
        return np.tile(-self._an, (U.shape[0], 1))

    def beta_exact(self):
        return self.beta_ref

    def u_star_exact(self):
        return self.beta_ref * self._an


# =========================================================================== #
# 2. Etat limite flexion BA -- branche "aciers plastifies" de                 #
#    flexion_claude.g (pure_flexion/AC3_pure_flexion.py l.1221-1230)          #
# =========================================================================== #
class FlexionLS:
    """
    Section rectangulaire BA en flexion simple, aciers plastifies.

    Moment resistant (identique a AC3_pure_flexion.flexion_claude) :

        M_R(fc, fy) = A * fy + B * fy^2 / fc     A = As*d/gamma_s
                                                 B = -As^2*gamma_c/(2*b*gamma_s^2)
        g = (M_R - Med) / Med

    Variables aleatoires (memes lois que les scripts AC) :
        fc ~ LogNormale (moyenne fcm, CoV cov_fc)   -> u1
        fy ~ Normale    (moyenne fym, ecart sig_fy) -> u2

    Unites : m, MN, MPa (= MN/m^2), coherentes avec les scripts AC.

    beta_exact() n'utilise NI metamodele NI FORM : la courbe g=0 est
    parametree par u1 (racine physique du trinome en fy, cf.
    flexion_claude.u2p_LS) puis ||u|| est minimise par Brent a 1e-12.
    """

    name = 'flexion'
    n_var = 2

    def __init__(self,
                 b=0.30, d=0.55, L=3.00,
                 phi_mm=20.0, n_bars=4,
                 F=0.099,
                 gamma_c=1.0, gamma_s=1.0,
                 fcm=48.0, cov_fc=0.12,
                 fym=550.0, sig_fy=30.0):
        self.b, self.d, self.L = b, d, L
        self.gamma_c, self.gamma_s = gamma_c, gamma_s
        self.fcm, self.cov_fc = fcm, cov_fc
        self.fym, self.sig_fy = fym, sig_fy

        self.As = n_bars * np.pi * (phi_mm / 2e3) ** 2
        self.Med = F * L

        self.A = self.As * d / gamma_s
        self.B = -self.As ** 2 * gamma_c / (2.0 * b * gamma_s ** 2)
        self.C = -self.Med

        self.sig_ln = np.sqrt(np.log(1.0 + cov_fc ** 2))
        self.mu_ln = np.log(fcm) - 0.5 * self.sig_ln ** 2

    # ---- transformation isoprobabiliste (copule independante) --------------
    def u_to_x(self, U):
        U = np.atleast_2d(np.asarray(U, dtype=float))
        fc = np.exp(self.mu_ln + self.sig_ln * U[:, 0])
        fy = self.fym + self.sig_fy * U[:, 1]
        return np.column_stack([fc, fy])

    def fy_to_u2(self, fy):
        return (np.asarray(fy, dtype=float) - self.fym) / self.sig_fy

    # ---- fonction de performance -------------------------------------------
    def g(self, U):
        X = self.u_to_x(U)
        fc, fy = X[:, 0], X[:, 1]
        M_R = self.A * fy + self.B * fy ** 2 / fc
        return (M_R - self.Med) / self.Med

    def grad(self, U):
        """dg/du (N,2) analytique -- oracle des tests de gradient GEPCK."""
        U = np.atleast_2d(np.asarray(U, dtype=float))
        X = self.u_to_x(U)
        fc, fy = X[:, 0], X[:, 1]
        dg_dfc = -self.B * fy ** 2 / fc ** 2 / self.Med
        dg_dfy = (self.A + 2.0 * self.B * fy / fc) / self.Med
        return np.column_stack([dg_dfc * self.sig_ln * fc,
                                dg_dfy * self.sig_fy])

    # ---- etat limite parametre : u2 tel que g(u1, u2) = 0 ------------------
    def u2_on_LS(self, u1):
        fc = float(np.exp(self.mu_ln + self.sig_ln * float(u1)))
        a = self.B / fc
        b = self.A
        c = self.C
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return np.nan
        fy = (-b + np.sqrt(disc)) / (2.0 * a)
        return float(self.fy_to_u2(fy))

    def _argmin_u1(self, u1_range, n_scan):
        u1 = np.linspace(u1_range[0], u1_range[1], n_scan)
        u2 = np.array([self.u2_on_LS(v) for v in u1])
        r = np.hypot(u1, u2)
        r[~np.isfinite(r)] = np.inf
        k = int(np.argmin(r))

        def norm(v):
            w = self.u2_on_LS(v)
            return np.inf if not np.isfinite(w) else float(np.hypot(v, w))

        res = minimize_scalar(norm,
                              bounds=(u1[max(k - 1, 0)], u1[min(k + 1, n_scan - 1)]),
                              method='bounded', options={'xatol': 1e-12})
        return float(res.x), float(res.fun)

    def beta_exact(self, u1_range=(-6.0, 6.0), n_scan=20001):
        return self._argmin_u1(u1_range, n_scan)[1]

    def u_star_exact(self, u1_range=(-6.0, 6.0), n_scan=20001):
        u1s, _ = self._argmin_u1(u1_range, n_scan)
        return np.array([u1s, self.u2_on_LS(u1s)])


CASES = {'linear': LinearLS, 'flexion': FlexionLS}
