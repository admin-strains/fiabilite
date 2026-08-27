r"""Les six enveloppes OpenTURNS des metamodeles.

CE QU'ELLES SONT
-----------------
OpenTURNS a besoin d'un objet `ot.Function` pour propager une incertitude.
Chaque famille de metamodele -- PCK, GEPCK, GEK, PC-Kriging, ou l'etat limite
haute fidelite lui-meme -- est donc habillee d'une classe qui expose `_exec`,
`_exec_sample`, `_gradient` et, quand la famille sait la calculer, `_exec_sigma`.

POURQUOI ELLES SONT ICI
------------------------
Elles etaient definies dans le bloc `__main__` des deux scripts d'etude, 193
lignes CHACUN, **identiques au caractere pres**. Une classe definie dans
`__main__` n'est ni importable, ni testable, ni isolable : c'est le constat
qui a fonde la phase 3 du nettoyage, et il valait aussi pour ces six-la.

Elles n'ont jamais dependu de l'etude : leurs seules variables libres etaient
`n_var`, l'evaluateur haute fidelite, et un interrupteur de trace. Ces trois-la
sont devenues des arguments de constructeur.

CE QUE L'EXTRACTION A REVELE
-----------------------------
1. `_exec_sigma` etait ecrite DEUX FOIS par fichier, a l'identique, dans
   `oldGEPCKFunction` et `GEKPLSFunction` -- 25 lignes de variance posterieure
   a noyau augmente, recopiees quatre fois au total. Elle est ici une fonction
   libre, `sigma_gek`, ecrite une fois.
2. Quatre lignes de trace supposaient `n_var == 2` en dur (`u[0]`, `u[1]`) :
   avec trois variables elles auraient tronque, avec une seule leve un
   IndexError. Elles sont generiques. Pour n_var = 2, la chaine produite est
   la meme -- le journal d'un run reste comparable a ceux d'avant.
"""

import numpy as np
import openturns as ot

from api import (predict_gepck, predict_gradient_gepck, predict_pck)


def _liste(valeurs, gabarit):
    """« [+1.2345, -0.6789] » pour un nombre quelconque de variables.

    Les traces d'origine ecrivaient `u[0]` et `u[1]` en dur.
    """
    return "[" + ", ".join(format(float(v), gabarit) for v in valeurs) + "]"


def sigma_gek(sm, u):
    """Ecart-type de prediction d'un krigeage a noyau AUGMENTE par les
    gradients (GEK).

    Le systeme de covariance porte les observations ET leurs derivees :

        K_tot = [[K_ff, K_fd],
                 [K_fd^T, K_dd]]

    d'ou sigma^2 = s2 * (1 - k^T K_tot^-1 k) avec k = [kf, kd].

    En cas de systeme singulier -- il arrive quand deux points du plan sont
    quasi confondus -- on retombe sur la variance sans gradients, qui est une
    borne superieure : mieux vaut une incertitude surestimee qu'une exception
    au milieu d'un enrichissement.
    """
    n = sm.nt
    d = sm.X_norma.shape[1]
    W = sm.coeff_pls
    th = sm.optimal_theta
    s2 = float(sm.optimal_par['sigma2'])
    th_eff = (W ** 2) @ th
    x_n = (np.array(u).reshape(-1) - sm.X_offset) / sm.X_scale
    Xn = sm.X_norma
    df = x_n[None, :] - Xn
    kf = np.exp(-np.dot(df ** 2, th_eff))
    kd = (2.0 * kf[:, None] * df * th_eff[None, :]).reshape(-1)
    dff = Xn[:, None, :] - Xn[None, :, :]
    K_ff = np.exp(-np.einsum('ijk,k->ij', dff ** 2, th_eff))
    K_fd = (2.0 * K_ff[:, :, None] * dff * th_eff[None, None, :]).reshape(n, n * d)
    B_mat = dff * th_eff[None, None, :]
    term1 = 2.0 * np.diag(th_eff)
    term2 = 4.0 * np.einsum('ija,ijb->ijab', B_mat, B_mat)
    K_dd = (K_ff[:, :, None, None] * (term1 - term2)).transpose(0, 2, 1, 3).reshape(n * d, n * d)
    K_tot = np.block([[K_ff, K_fd], [K_fd.T, K_dd]])
    K_tot += 1e-10 * np.eye(K_tot.shape[0])
    k = np.concatenate([kf, kd])
    try:
        B = max(0.0, 1.0 - k @ np.linalg.solve(K_tot, k))
        return float(np.sqrt(s2 * B))
    except np.linalg.LinAlgError:
        return float(np.sqrt(sm.predict_variances(np.array(u).reshape(1, -1)).item()))


class HFFunction(ot.OpenTURNSPythonFunction):
    """L'etat limite EXACT, vu par OpenTURNS.

    Le cache d'un point n'est pas une optimisation accessoire : OpenTURNS
    demande la valeur puis le gradient en deux appels separes, et une
    evaluation coute jusqu'a 466 s sur le Moulin Blanc. Sans lui, chaque point
    serait resolu deux fois.
    """

    def __init__(self, n_var, evaluer):
        super().__init__(n_var, 1)
        self.n_var = n_var
        self._evaluer = evaluer
        self._cache_u = None
        self._cache_g = None
        self._cache_grad = None
        self.n_hf_calls = 0

    def _run_if_needed(self, u):
        u_arr = np.array(u)
        if self._cache_u is None or not np.allclose(u_arr, self._cache_u, atol=1e-12):
            g, grad_U, _ = self._evaluer(u)
            self._cache_u = u_arr.copy()
            self._cache_g = float(g)
            self._cache_grad = [float(grad_U[i]) for i in range(self.n_var)]
            self.n_hf_calls += 1
            print("[HF #%3d] u=%s  g=%+.6f  grad=%s"
                  % (self.n_hf_calls, _liste(u_arr, "+.4f"), g,
                     _liste(self._cache_grad, "+.6f")), flush=True)

    def _exec(self, u):
        self._run_if_needed(u)
        return [self._cache_g]

    def _gradient(self, u):
        self._run_if_needed(u)
        # Format OpenTURNS : (n_var, 1)
        return [[g] for g in self._cache_grad]


class PCKRGFunction(ot.OpenTURNSPythonFunction):
    """PC-Kriging naif : la somme d'un chaos polynomial et d'un krigeage sur
    le residu, chacun garde son propre objet OpenTURNS."""

    def __init__(self, n_var, g_pce, g_krg):
        super().__init__(n_var, 1)
        self.n_var = n_var
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
        u_ot = ot.Point(list(u))
        grad_pce = self.g_pce.gradient(u_ot)
        grad_krg = self.g_krg.gradient(u_ot)
        return [[grad_pce[i, 0] + grad_krg[i, 0]] for i in range(self.n_var)]


class oldGEPCKFunction(ot.OpenTURNSPythonFunction):
    """Premiere version du GEPCK : chaos polynomial + GEK (smt) sur le residu.

    Conservee comme temoin -- `do_old_GEPCK` la selectionne -- face a
    l'implementation `GEPCKFunction`, qui passe par `api.predict_gepck`.
    """

    def __init__(self, n_var, g_pce, sm_gepck):
        super().__init__(n_var, 1)
        self.n_var = n_var
        self.g_pce = g_pce
        self.sm = sm_gepck

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
        return sigma_gek(self.sm, u)

    def _gradient(self, u):
        u_np = np.array(u).reshape(1, -1)
        grad_pce = self.g_pce.gradient(ot.Point(list(u)))   # OT Matrix (n_var, 1)
        return [[grad_pce[i, 0] + self.sm.predict_derivatives(u_np, i).item()]
                for i in range(self.n_var)]


class GEPCKFunction(ot.OpenTURNSPythonFunction):
    """GEPCK a cinq branches, evalue par `api.predict_gepck`."""

    def __init__(self, n_var, fm, tracer_appels=False):
        super().__init__(n_var, 1)
        self.n_var = n_var
        self.fm = fm
        self.tracer_appels = tracer_appels
        self.n_eval_calls = 0
        self.n_grad_calls = 0

    def _exec(self, u):
        u_np = np.array(u).reshape(1, -1)
        g_val = float(predict_gepck(self.fm, u_np)[0, 0])
        self.n_eval_calls += 1
        if self.tracer_appels:
            print("[GEPCK eval #%3d] u=%s  g=%+.6f"
                  % (self.n_eval_calls, _liste(u, "+.4f"), g_val), flush=True)
        return [g_val]

    def _exec_sample(self, U):
        U_np = np.array(U)
        return predict_gepck(self.fm, U_np)[:, 0:1].tolist()

    def _exec_sigma(self, u):
        u_np = np.array(u).reshape(1, -1)
        _, YSig2 = predict_gepck(self.fm, u_np, return_var=True)
        return float(np.sqrt(max(0.0, float(YSig2[0, 0]))))

    def _gradient(self, u):
        u_np = np.array(u).reshape(1, -1)
        G = predict_gradient_gepck(self.fm, u_np)   # (1, Mred)
        grad = [float(G[0, i]) for i in range(self.fm['Mred'])]
        g_val = float(predict_gepck(self.fm, u_np)[0, 0])
        self.n_grad_calls += 1
        print("[GEPCK grad #%3d] u=%s  g=%+.6f  grad=%s"
              % (self.n_grad_calls, _liste(u, "+.4f"), g_val,
                 _liste(grad, "+.6f")), flush=True)
        return [[v] for v in grad]


class PCKFunction(ot.OpenTURNSPythonFunction):
    """PC-Kriging sans gradient analytique : FORM y recourt aux differences
    finies d'OpenTURNS."""

    def __init__(self, n_var, fm, tracer_appels=False):
        super().__init__(n_var, 1)
        self.n_var = n_var
        self.fm = fm
        self.tracer_appels = tracer_appels
        self.n_eval_calls = 0

    def _exec(self, u):
        u_np = np.array(u).reshape(1, -1)
        g_val = float(predict_pck(self.fm, u_np)[0, 0])
        self.n_eval_calls += 1
        if self.tracer_appels:
            print("[PCK eval #%3d] u=%s  g=%+.6f"
                  % (self.n_eval_calls, _liste(u, "+.4f"), g_val), flush=True)
        return [g_val]

    def _exec_sample(self, U):
        U_np = np.array(U)
        return predict_pck(self.fm, U_np)[:, 0:1].tolist()

    def _exec_sigma(self, u):
        u_np = np.array(u).reshape(1, -1)
        _, YSig2 = predict_pck(self.fm, u_np, return_var=True)
        return float(np.sqrt(max(0.0, float(YSig2[0, 0]))))


class GEKPLSFunction(ot.OpenTURNSPythonFunction):
    """GEK avec reduction PLS (smt.GEKPLS)."""

    def __init__(self, n_var, surrogate):
        super().__init__(n_var, 1)
        self.n_var = n_var
        self.sm = surrogate

    def _exec(self, u):
        return [self.sm.predict_values(np.array(u).reshape(1, -1)).item()]

    def _exec_sample(self, U):
        return self.sm.predict_values(np.array(U)).tolist()

    def _exec_sigma(self, u):
        return sigma_gek(self.sm, u)

    def _gradient(self, u):
        u_np = np.array(u).reshape(1, -1)
        return [[self.sm.predict_derivatives(u_np, kx).item()]
                for kx in range(self.n_var)]
