r"""FORM + tirage d'importance sur le metamodele COURANT : le juge de l'enrichissement.

A QUOI CELA SERT
-----------------
L'enrichissement EFF ajoute des points la ou le metamodele est a la fois
proche de l'etat limite et incertain. Mais « a quel moment s'arreter » ne se
lit pas sur le critere EFF seul : un critere purement geometrique peut etre
satisfait alors que la probabilite de defaillance, elle, bouge encore.

D'ou ce controle : a chaque iteration, on refait un FORM et un tirage
d'importance sur le metamodele courant, et on regarde si beta se stabilise.
Deux mesures en sortent :

* **BB** -- « bande de beta » : l'ecart entre le beta obtenu sur `g + 2 sigma`
  et celui obtenu sur `g - 2 sigma`, rapporte a beta. Il mesure ce que
  l'incertitude du metamodele coute EN FIABILITE, et non en valeur de `g`.
  Tant qu'il est large, deux surfaces egalement plausibles donnent deux
  reponses differentes.
* **BS** -- la stabilite de beta d'une iteration a l'autre.

CE QUE CELA COUTE
------------------
ZERO appel solveur : tout se passe sur le metamodele. Mais un FORM+IS n'est
pas gratuit en secondes, et l'encadrement en demande TROIS. C'est pour cela
que `_three_form_is` sait reutiliser le beta central deja calcule -- une
economie d'un tiers, a chaque iteration.

POURQUOI C'EST SORTI DE `run_EFF`
----------------------------------
`run_EFF` faisait 321 lignes, la plus grosse fonction du depot, dont 80 pour
ces deux fonctions imbriquees. Elles fermaient sur `xt`, `yt` et `all_grad`
-- QUI CHANGENT a chaque iteration de la boucle. La capture etait donc juste,
mais invisible : rien ne disait que le tirage d'importance adaptatif
travaillait sur l'etat courant du plan. Ici, cet etat est un argument.
"""

import os
import time

import openturns as ot


def _ecrire(message):
    print(message, flush=True)


class ControleurFORM:
    """FORM + IS sur un metamodele, avec les reglages de l'etude.

    `executer_is(modes, evenement)` est le tirage d'importance de repli --
    celui d'OpenTURNS. `adaptatif` est la version parallelisee (sonde puis
    montee en charge) ; None la desactive, ce que fait l'etude quand le
    metamodele est un PCK, dont le format n'est pas celui qu'elle attend.
    """

    def __init__(self, n_var, n_max_FORM, tol_FORM, executer_is,
                 adaptatif=None, cov_cible=0.05, n_is=10000,
                 K=16, chunk=8, sondes=16, tracer=_ecrire):
        self.n_var = n_var
        self.n_max_FORM = n_max_FORM
        self.tol_FORM = tol_FORM
        self.executer_is = executer_is
        self.adaptatif = adaptatif
        self.cov_cible = cov_cible
        self.n_is = n_is
        self.K = K
        self.chunk = chunk
        self.sondes = sondes
        self.tracer = tracer

    # ------------------------------------------------------------------ #
    def _form(self, g_ot, label):
        """Le FORM lui-meme. Retourne le resultat, ou None s'il echoue.

        `setCheckStatus(False)` : AbdoRackwitz leve si le critere de
        convergence n'est pas atteint dans le nombre d'iterations imparti. Au
        milieu d'un enrichissement, ce n'est pas une raison d'arreter -- on
        prend le point trouve et on continue.
        """
        # L'evenement se construit HORS du `try` : un echec ici serait une
        # erreur de programmation, pas un defaut de convergence, et l'attraper
        # le ferait passer pour « FORM echoue ».
        distribution = ot.JointDistribution([ot.Normal(0, 1)] * self.n_var)
        evenement = ot.ThresholdEvent(
            ot.CompositeRandomVector(g_ot, ot.RandomVector(distribution)),
            ot.Less(), 0.0)
        try:
            solveur = ot.AbdoRackwitz()
            solveur.setStartingPoint([0.0] * self.n_var)
            solveur.setMaximumIterationNumber(self.n_max_FORM)
            solveur.setCheckStatus(False)
            solveur.setMaximumConstraintError(self.tol_FORM)
            algo = ot.FORM(solveur, evenement)
            algo.run()
            return algo.getResult(), evenement
        except Exception as e:
            self.tracer("  [%s] FORM echoue (%s)" % (label, type(e).__name__))
            return None, None

    def beta_et_pf(self, g_ot, label, sign=0, fm=None, etat=None):
        """`(beta_IS, pf_IS)` pour le metamodele donne, ou `(None, None)`.

        `etat` porte le plan COURANT (`xt`, `yt`, `all_grad`, `max_degree`) :
        le tirage adaptatif reconstruit le metamodele a partir de lui, et il
        change a chaque iteration de l'enrichissement.
        """
        resultat, evenement = self._form(g_ot, label)
        if resultat is None:
            return None, None
        beta_form = resultat.getHasoferReliabilityIndex()
        pf_form = resultat.getEventProbability()

        if self.adaptatif is not None and fm is not None and etat is not None:
            u_star = list(resultat.getStandardSpaceDesignPoint())
            cap = int(os.environ.get("_IS_CAP", str(self.n_is)))
            r = self.adaptatif(fm, etat, u_star, sign=sign,
                               cov_target=self.cov_cible, cap_blocks=cap,
                               K=self.K, chunk=self.chunk,
                               probe_blocks=self.sondes)
            pf_is = r['pf']
            beta_is = (float(-ot.Normal().computeQuantile(pf_is)[0])
                       if pf_is > 0 else float('nan'))
            self.tracer("  [%s] beta_FORM=%.4f  Pf_FORM=%.3e | Pf_IS=%.3e  "
                        "beta_IS=%.4f  COV=%.3f  [PAR:%s]"
                        % (label, beta_form, pf_form, pf_is, beta_is,
                           r['cov'], r['mode']))
            self.tracer("  [IS DETAIL PAR] %s : blocs=%d evals~%s COV=%.4f "
                        "(cible %s)"
                        % (label, r['n_blocks'], format(r['n_evals'], ","),
                           r['cov'], self.cov_cible))
            return beta_is, pf_is

        resultat_is = self.executer_is([resultat], evenement)
        pf_is = resultat_is.getProbabilityEstimate()
        beta_is = float(-ot.Normal().computeQuantile(pf_is)[0])
        self.tracer("  [%s] beta_FORM=%.4f  Pf_FORM=%.3e | Pf_IS=%.3e  "
                    "beta_IS=%.4f  COV=%.3f"
                    % (label, beta_form, pf_form, pf_is, beta_is,
                       resultat_is.getCoefficientOfVariation()))
        return beta_is, pf_is

    # ------------------------------------------------------------------ #
    def encadrement(self, g_ot, sigma_func, label, borner, etat=None,
                    beta_central=None):
        """FORM+IS sur `g`, `g + 2 sigma` et `g - 2 sigma`.

        Retourne `(ratio, pf_mid, pf_sup, pf_inf)`, ou quatre `None` si l'un
        des trois FORM a echoue.

        Le ratio `|beta_sup - beta_inf| / |beta|` est le critere BB : il dit
        ce que l'incertitude du metamodele coute en FIABILITE. Un metamodele
        peut coller a l'etat limite partout sauf la ou beta se joue.

        `beta_central` evite de recalculer le FORM+IS central quand
        l'appelant vient de le faire -- un tiers du cout de ce controle.
        """
        g_sup = ot.Function(borner(g_ot, sigma_func, +1))
        g_inf = ot.Function(borner(g_ot, sigma_func, -1))
        fm = getattr(getattr(sigma_func, '__self__', None), 'fm', None)

        if beta_central is not None:
            b_mid, pf_mid = beta_central
            self.tracer("  [%s mu] reutilise mu conv (pas de recalcul FORM/IS "
                        "redondant)" % label)
        else:
            b_mid, pf_mid = self.beta_et_pf(g_ot, "%s mu" % label, sign=0,
                                            fm=fm, etat=etat)
        b_sup, pf_sup = self.beta_et_pf(g_sup, "%s sup" % label, sign=+1,
                                        fm=fm, etat=etat)
        b_inf, pf_inf = self.beta_et_pf(g_inf, "%s inf" % label, sign=-1,
                                        fm=fm, etat=etat)
        if None in (b_mid, b_sup, b_inf) or b_mid == 0:
            return None, None, None, None
        ratio = abs(b_sup - b_inf) / abs(b_mid)
        self.tracer("  [%s] |beta_IS_sup - beta_IS_inf| / beta_IS = %.4f"
                    % (label, ratio))
        return ratio, pf_mid, pf_sup, pf_inf



    def amorcer_iteration(self, g_ot, sigma_func, borner, arret, *,
                          n_points, historique_Pf=None, etat=None):
        """L'encadrement du plan INITIAL, avant tout enrichissement.

        Son ratio compte deja pour une iteration valide : un plan qui part
        convergent ne doit pas payer trois iterations pour le prouver.

        UNE REDONDANCE CONSTATEE, NON CORRIGEE. L'appelant vient de mesurer
        le FORM+IS central (`beta_et_pf`, etiquette « initial mu conv ») et
        cette methode ne le lui passe PAS en `beta_central` -- contrairement
        a `mesurer_iteration`. Le central est donc recalcule : trois FORM+IS
        au lieu de deux. Ce n'est pas du solveur, c'est du metamodele --
        des secondes, pas des heures.

        Le corriger n'est pas gratuit pour autant : le tirage d'importance
        est stochastique, et les deux mesures du central ne rendent pas
        exactement le meme beta. Reutiliser la premiere DEPLACERAIT le ratio
        BB initial. C'est donc un arbitrage, pas un nettoyage -- et les deux
        appels n'emploient meme pas le meme `fm` : l'appelant passe None,
        `encadrement` le tire de `sigma_func`.
        """
        ratio, pf_mid, pf_sup, pf_inf = self.encadrement(
            g_ot, sigma_func, "N=%d initial BB" % n_points, borner, etat=etat)
        if historique_Pf is not None:
            historique_Pf.append({'mid': pf_mid, 'sup': pf_sup, 'inf': pf_inf})
        arret.amorcer(ratio)
        return ratio

    def mesurer_iteration(self, g_ot, sigma_func, borner, arret, *,
                          n_points, iteration, avec_Pf,
                          historique_Pf=None, historique_beta=None, etat=None):
        """Une iteration d'enrichissement, jugee.

        `beta` par FORM + tirage d'importance ; l'encadrement `g +/- 2 sigma`
        s'il est du ; les compteurs d'arret mis a jour.

        LA DECISION DE COUT, ENFIN NOMMEE
        ----------------------------------
        L'encadrement vaut DEUX FORM+IS de plus par iteration -- le central
        est reutilise. Deux raisons de le payer, et une seule facture :

          * le critere d'arret en a besoin (`BB`, `both`, `at_least_one`) ;
          * `print_Pf` reclame les courbes de Pf mid/sup/inf.

        Cela s'ecrivait `if print_Pf: ...` puis
        `if a_besoin and not print_Pf: ...` -- deux conditions qui se
        recouvrent, et dont il fallait deduire qu'elles ne paient qu'une
        fois. Ici c'est un seul `or`, et le triplet de Pf n'est range que si
        on l'a demande.

        Le ratio BB obtenu est enregistre quel que soit le critere : tout ce
        qui est paye est enregistre (voir `_reliability/arret.py`).

        Retourne `(beta, pf, ratio_bb)`. `ratio_bb` vaut None quand
        l'encadrement n'a pas ete paye -- pas quand il a echoue, que
        `encadrement` signale de la meme facon ; c'est une limite heritee.
        """
        fm = getattr(getattr(sigma_func, '__self__', None), 'fm', None)
        t0 = time.perf_counter()
        beta, pf = self.beta_et_pf(g_ot, "N=%d mu conv" % n_points, fm=fm,
                                   etat=etat)
        self.tracer("  [TIMING _form_is_iter] dt=%.2fs (fm=%s)"
                    % (time.perf_counter() - t0, "oui" if fm else "non"))

        ratio_bb = None
        if avec_Pf or arret.a_besoin_de_l_encadrement:
            ratio_bb, pf_mid, pf_sup, pf_inf = self.encadrement(
                g_ot, sigma_func, "N=%d iter %d" % (n_points, iteration),
                borner, etat=etat, beta_central=(beta, pf))
            if avec_Pf and historique_Pf is not None:
                historique_Pf.append({'mid': pf_mid, 'sup': pf_sup,
                                      'inf': pf_inf})

        arret.enregistrer(ratio_bb, beta,
                          historique_beta[-1] if historique_beta else None,
                          prefixe="N=%d" % n_points)
        if beta is not None and historique_beta is not None:
            historique_beta.append(beta)
        return beta, pf, ratio_bb


# --------------------------------------------------------------------------- #
# L'ERREUR FOSM : ce que le metamodele coute en POSITION du point de conception
# --------------------------------------------------------------------------- #
class ErreurFOSM:
    """Ecart entre le u* du metamodele et celui d'un developpement du PREMIER
    ORDRE de l'etat limite EXACT autour de l'origine.

    C'est une mesure de justesse independante de l'enrichissement : le
    metamodele peut avoir convergé sur lui-meme et rester decale par rapport a
    la surface reelle. Le developpement au premier ordre en 0 donne un point de
    conception approche, mais calcule sur l'ETAT LIMITE EXACT.

    COUT : deux appels solveur -- un en u*, un en 0. Le second est mis en cache
    et reutilise d'un mode FORM a l'autre : sur une etude a quatre modes, cela
    fait quatre appels au lieu de huit, soit une demi-heure economisee sur le
    Moulin Blanc.

    Ce n'est PAS une figure. Elle etait pourtant soudee a l'impression des
    resultats FORM, sous un nom qui dit « print » -- d'ou deux SOCP declenches
    par ce qui ressemblait a de l'affichage.
    """

    def __init__(self, evaluer_en_U, params_names, tracer=_ecrire):
        self.evaluer = evaluer_en_U
        self.params_names = params_names
        self.n_var = len(params_names)
        self.tracer = tracer
        #: (g, gradient) a l'origine -- calcule une fois, relu ensuite.
        self._origine = None

    def gradient_a_l_origine(self):
        """Le gradient en u = 0, calcule au plus une fois."""
        if self._origine is None:
            g0, grad0, _ = self.evaluer(ot.Point([0.0] * self.n_var))
            self._origine = (g0, grad0)
        else:
            self.tracer("  [FOSM] run_HF([0,0]) reutilise du cache "
                        "(pas de SOCP redondant)")
        return self._origine

    def mesurer(self, best_result):
        """L'ecart relatif, ou None si le point de conception est en 0.

        Deux appels solveur au premier mode, un seul aux suivants.
        """
        u_star = best_result.getStandardSpaceDesignPoint()
        _, grad_u_star, _ = self.evaluer(u_star)
        for i, p in enumerate(self.params_names):
            self.tracer("dg/du_%s en u* (HF@u*GEK) = %.6f" % (p, grad_u_star[i]))

        g0, grad0 = self.gradient_a_l_origine()
        u_fosm = grad0 * (-g0 / grad0.normSquare())
        self.tracer("u* FOSM (HF) = %s" % [round(v, 4) for v in u_fosm])
        if u_star.norm() == 0:
            # Un point de conception a l'origine voudrait dire que la
            # defaillance est atteinte a la moyenne de toutes les variables :
            # l'ecart relatif n'a alors aucun sens.
            self.tracer("Erreur FOSM  = indefinie (u* est a l'origine)")
            return None
        erreur = (u_fosm - u_star).norm() / u_star.norm()
        self.tracer("Erreur FOSM  = %.4f" % erreur)
        return erreur
