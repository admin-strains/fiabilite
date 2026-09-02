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
from scipy.optimize import minimize, minimize_scalar


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



# =========================================================================== #
# 3. Poutre console -- TROIS variables, non lineaire, beta par minimisation   #
# =========================================================================== #
class ConsoleLS:
    r"""Fleche en bout d'une poutre console, limitee a L/250.

        delta(F, E, h) = 4 F L^3 / (E b h^3)
        g = 1 - delta / delta_lim

    POURQUOI CE TROISIEME CAS EXISTE
    ---------------------------------
    Les deux premiers ont DEUX variables, et ils se comportent de facon
    OPPOSEE sur tout ce qui touche au krigeage : sur `linear` la PCE
    represente l'etat limite exactement, `sigma^2` tombe au plancher
    d'annulation (1e-24) et `theta` n'a pas de valeur vraie ; sur `flexion`
    l'ajustement a deux bassins d'attraction, et l'un des cinq runners
    d'integration continue tombe dans l'autre.

    Un cas peut donc cacher un defaut que l'autre expose, et rien ne
    tranchait entre les deux. Il manquait surtout un cas a TROIS variables :
    la matrice de Gram augmentee de GEPCK est de taille n(M+1), et avec M=2
    elle ne porte que deux blocs de derivees. `theta` y a trois longueurs de
    correlation a ajuster, ce qui exerce l'anisotropie pour de bon.

    La non-linearite est franche et n'est pas polynomiale : `g` varie en
    `h^-3` et en `1/E`. La tendance PCE ne peut donc pas absorber tout le
    signal, et le residu que le krigeage explique est reel -- c'est le
    regime ou `theta` COMPTE, celui que ni `linear` ni `flexion` ne montrent.

    Variables (memes conventions que les autres cas : U ~ N(0,1)^3) :
        F ~ LogNormale (moyenne Fm, CoV cov_F)      -> u1   charge, MN
        E ~ LogNormale (moyenne Em, CoV cov_E)      -> u2   module, MPa
        h ~ Normale    (moyenne hm, ecart sig_h)    -> u3   hauteur, m

    Unites : m, MN, MPa (= MN/m^2) -- coherentes avec `FlexionLS`.

    beta_exact() n'utilise NI metamodele NI FORM. L'etat limite est un
    GRAPHE : `g = 0` s'inverse exactement en `h = (K F / E)^(1/3)`, donc
    `u3 = phi(u1, u2)`. On minimise `||u||^2` sur ce graphe par L-BFGS-B avec
    son GRADIENT ANALYTIQUE, apres un balayage de 121x121 pour l'amorce.
    Confier la pente a des differences finies serait ici une faute de gout
    apres la journee du 02/09/2026.
    """

    name = 'console'
    n_var = 3

    def __init__(self, L=3.0, b=0.30, delta_lim=None,
                 Fm=0.05, cov_F=0.20,
                 Em=30000.0, cov_E=0.10,
                 hm=0.50, sig_h=0.02):
        self.L, self.b = float(L), float(b)
        self.delta_lim = float(delta_lim if delta_lim is not None else L / 250.0)
        #: 4 L^3 / (b delta_lim) : tout ce qui ne depend pas des variables
        self.K = 4.0 * self.L ** 3 / (self.b * self.delta_lim)
        self.hm, self.sig_h = float(hm), float(sig_h)
        self.sig_lnF = np.sqrt(np.log(1.0 + cov_F ** 2))
        self.mu_lnF = np.log(Fm) - 0.5 * self.sig_lnF ** 2
        self.sig_lnE = np.sqrt(np.log(1.0 + cov_E ** 2))
        self.mu_lnE = np.log(Em) - 0.5 * self.sig_lnE ** 2

    # ---- transformation isoprobabiliste (copule independante) --------------
    def u_to_x(self, U):
        U = np.atleast_2d(np.asarray(U, dtype=float))
        F = np.exp(self.mu_lnF + self.sig_lnF * U[:, 0])
        E = np.exp(self.mu_lnE + self.sig_lnE * U[:, 1])
        h = self.hm + self.sig_h * U[:, 2]
        return np.column_stack([F, E, h])

    # ---- fonction de performance -------------------------------------------
    def g(self, U):
        X = self.u_to_x(U)
        F, E, h = X[:, 0], X[:, 1], X[:, 2]
        return 1.0 - self.K * F / (E * h ** 3)

    def grad(self, U):
        """dg/du (N,3) analytique -- oracle des tests de gradient GEPCK.

        Avec `r = K F / (E h^3)` (la fleche relative), `g = 1 - r` et

            dg/dF = -r/F     dg/dE = +r/E     dg/dh = +3 r/h

        puis la chaine vers `u` : `dF/du1 = sig_lnF F`, `dE/du2 = sig_lnE E`,
        `dh/du3 = sig_h`.
        """
        X = self.u_to_x(U)
        F, E, h = X[:, 0], X[:, 1], X[:, 2]
        r = self.K * F / (E * h ** 3)
        return np.column_stack([-r * self.sig_lnF,
                                +r * self.sig_lnE,
                                +3.0 * r / h * self.sig_h])

    # ---- etat limite parametre : u3 tel que g(u1, u2, u3) = 0 --------------
    def u3_on_LS(self, u1, u2):
        ln_h = (np.log(self.K) + self.mu_lnF + self.sig_lnF * u1
                - self.mu_lnE - self.sig_lnE * u2) / 3.0
        return (np.exp(ln_h) - self.hm) / self.sig_h

    def _r2_et_grad(self, v):
        """`||u||^2` sur le graphe, et sa pente exacte."""
        u1, u2 = float(v[0]), float(v[1])
        ln_h = (np.log(self.K) + self.mu_lnF + self.sig_lnF * u1
                - self.mu_lnE - self.sig_lnE * u2) / 3.0
        h = np.exp(ln_h)
        phi = (h - self.hm) / self.sig_h
        dphi1 = h * self.sig_lnF / 3.0 / self.sig_h
        dphi2 = -h * self.sig_lnE / 3.0 / self.sig_h
        return (u1 * u1 + u2 * u2 + phi * phi,
                np.array([2.0 * u1 + 2.0 * phi * dphi1,
                          2.0 * u2 + 2.0 * phi * dphi2]))

    def _argmin(self, cote=121, borne=6.0):
        grille = np.linspace(-borne, borne, cote)
        meilleur, v0 = np.inf, (0.0, 0.0)
        for a in grille:
            for c in grille:
                r2, _ = self._r2_et_grad((a, c))
                if r2 < meilleur:
                    meilleur, v0 = r2, (a, c)
        res = minimize(lambda v: self._r2_et_grad(v)[0], np.array(v0),
                       jac=lambda v: self._r2_et_grad(v)[1],
                       method='L-BFGS-B',
                       options={'ftol': 1e-16, 'gtol': 1e-14})
        return res.x, float(np.sqrt(res.fun))

    def beta_exact(self):
        return self._argmin()[1]

    def u_star_exact(self):
        v, _ = self._argmin()
        return np.array([v[0], v[1], self.u3_on_LS(v[0], v[1])])


CASES = {'linear': LinearLS, 'flexion': FlexionLS, 'console': ConsoleLS}
