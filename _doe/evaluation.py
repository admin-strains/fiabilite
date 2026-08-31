r"""L'UNIQUE passage du programme vers le solveur.

POURQUOI UN SEUL ENDROIT
-------------------------
Tout ce que coute une etude de fiabilite passe par ici : un appel vaut
jusqu'a 466 s sur le Moulin Blanc, et une etude complete en demande plusieurs
centaines. Un point evalue deux fois, c'est un quart d'heure perdu ; un point
evalue avec un maillage different de son voisin, c'est un metamodele qui
ajuste deux surfaces differentes en croyant n'en voir qu'une.

Ce second cas n'est pas theorique : jusqu'a la phase 5, `run_HF` et
`run_one_SOL` ne maillaient PAS pareil, et les points qu'elles produisaient
nourrissaient le MEME metamodele. Les deux vivent maintenant dans la meme
classe, sur le meme solveur.

LES DEUX PORTES
----------------
* `evaluer_plan` -- une liste de points donnes en variables PHYSIQUES, avec
  reprise : un point deja calcule n'est jamais re-evalue.
* `evaluer_en_U` -- un point donne en variables NORMEES, pour
  l'enrichissement, la grille et le FOSM.

LA DIFFERENCE DE TRAITEMENT EST VOULUE
---------------------------------------
Un point du plan d'experiences sans gradient est ECARTE ; un point
d'enrichissement sans gradient fait LEVER. Ce n'est pas une incoherence :
l'enrichissement a demande CE point-la parce que l'algorithme le veut la.
L'ecarter en silence lui ferait reproposer le meme point, indefiniment.
"""

import numpy as np
import openturns as ot


def _ecrire(message):
    print(message, flush=True)


def gradient_vers_U(grad_X, u_point, T_inv):
    """Le gradient de l'espace physique X vers l'espace standard U.

    Ici, et pas dans le solveur : la transformation isoprobabiliste appartient
    a la loi jointe, pas au maillage.
    """
    J_Tinv_T = T_inv.gradient(u_point).transpose()
    return J_Tinv_T * ot.Point(list(grad_X))


class Evaluateur:
    """L'etat limite, vu par le reste du programme.

    `solveur_pour(nom)` rend le solveur d'un modele -- un par copie du `.ds`,
    car un worker de plan parallele travaille sur la sienne.
    """

    def __init__(self, solveur_pour, dist, params_names,
                 exclure_non_converges=False, archiver=False,
                 journaliser=None, sauver_partiel=None, tracer=_ecrire):
        self.solveur_pour = solveur_pour
        self.dist = dist
        self.params_names = params_names
        self.n_var = len(params_names)
        self.exclure_non_converges = exclure_non_converges
        self.archiver = archiver
        #: `journaliser(u, x, g)` -- le journal incremental des points evalues
        self.journaliser = journaliser or (lambda *_a: None)
        #: `sauver_partiel(SOL, n_faits)` -- la reprise du plan d'experiences
        self.sauver_partiel = sauver_partiel or (lambda *_a: None)
        self.tracer = tracer
        self.n_appels = 0

    # ------------------------------------------------------------------ #
    def etiquette(self, prefixe, p_vals, u=None):
        """Nom du sous-dossier de `SOCP_history`, dans la forme d'origine.

        L'original ecrivait `u[0]` et `u[1]` en dur : au-dela de deux
        variables, deux points distincts pouvaient recevoir le meme nom
        d'archive.
        """
        self.n_appels += 1
        coords = ""
        if u is not None:
            coords = "".join("_u%d%+.3f" % (i + 1, float(v))
                             for i, v in enumerate(u))
        coords += "_" + "_".join("%s%.1f" % (self.params_names[i], p_vals[i])
                                 for i in range(len(p_vals)))
        return "%s_%03d%s" % (prefixe, self.n_appels, coords)

    def _signaler_si_non_convergent(self, ev, description, motif,
                                    mention="EXCLU"):
        """Signaler, pas jeter.

        Les criteres rendus par Digital Structure ne sont pas encore fiables
        (Agnes, 26/08/2026) : un point ecarte sur cette base
        serait un appel solveur paye pour rien. `exclure_points_non_converges`
        rebasculera le jour ou ils le seront.
        """
        if ev.sain:
            return
        self.tracer("  [SOLVEUR] %s NON CONVERGE (%s), alpha=%.6f -- %s"
                    % (description, ev.diagnostic.get("solver_status"), ev.alpha,
                       mention if self.exclure_non_converges
                       else "conserve : critere DS juge non fiable"))
        if self.exclure_non_converges:
            ev.exige_sain(motif)

    # ------------------------------------------------------------------ #
    def evaluer_plan(self, SOL, modelname=None, sensibilite=False):
        """Un plan d'experiences en variables PHYSIQUES.

        Chaque entree de `SOL` recoit `g`, `_u` (ses coordonnees normees) et
        `dg_<var>` (le gradient en U, ou None). Une entree qui porte deja `g`
        est SAUTEE : c'est tout l'interet de la reprise, et cela vaut des
        heures apres une interruption.
        """
        solveur = self.solveur_pour(modelname)
        T = self.dist.getIsoProbabilisticTransformation()
        T_inv = self.dist.getInverseIsoProbabilisticTransformation()

        deja = sum(1 for s in SOL if 'g' in s)
        if deja:
            self.tracer("  [SOLVEUR] %d/%d point(s) deja connus (cache "
                        "partiel) : autant de SOCP evites" % (deja, len(SOL)))

        for i in range(len(SOL)):
            if 'g' in SOL[i]:
                continue
            p_vals = [float(SOL[i][p]) for p in self.params_names]
            etiquette = self.etiquette("SOL", p_vals) if self.archiver else None
            ev = solveur.evaluer({p: SOL[i][p] for p in self.params_names},
                                 sensibilite=sensibilite, etiquette=etiquette)
            self._signaler_si_non_convergent(
                ev, "point %s" % (p_vals,),
                "point %s du plan d'experiences" % (p_vals,),
                mention="EXCLU du plan d'experiences")
            SOL[i]['g'] = ev.g

            u_point = T(ot.Point(p_vals))
            SOL[i]['_u'] = [float(u_point[j]) for j in range(self.n_var)]
            if sensibilite and ev.gradient_complet:
                grad_U = gradient_vers_U(ev.grad_x, u_point, T_inv)
                for j, p in enumerate(self.params_names):
                    SOL[i]['dg_%s' % p] = float(grad_U[j])
            else:
                for p in self.params_names:
                    SOL[i]['dg_%s' % p] = None
            self.sauver_partiel(SOL, sum(1 for s in SOL if 'g' in s))
        return SOL

    # ------------------------------------------------------------------ #
    def evaluer_g_en_U(self, u):
        """L'etat limite en UN point, SANS exiger le gradient.

        Retourne la meme forme que `evaluer_en_U` -- `(g, grad_U, grad_X)` --
        mais `grad_U` peut porter des `None`.

        A QUOI ELLE SERT, ET QUEL DEFAUT ELLE FERME -- 29/08/2026
        ----------------------------------------------------------
        La grille haute fidelite ne veut qu'un `g` : elle DESSINE une surface,
        elle ne nourrit aucun metamodele. Ses quatre sites d'appel s'ecrivent
        tous `self.evaluer(pt)[0]`.

        Elle etait pourtant cablee sur `evaluer_en_U`, qui LEVE quand le
        solveur ne rend pas de gradient -- un refus voulu pour un point
        d'ENRICHISSEMENT, ou un gradient fabrique contaminerait le
        metamodele. Resultat, deux voies de la MEME grille jugeaient le meme
        point differemment. Mesure sur un solveur qui converge sans rendre de
        sensibilites :

            n_workers <= 1  voie sequentielle  -> ValueError, la grille meurt
            n_workers >  1  voie parallele     -> g = 0.42, point accepte

        Et `tools/run_comparatif.py` impose `n_workers_DOE = 1` : toute
        comparaison A/B prenait donc la voie stricte pendant que la
        production prenait l'autre.

        `sensibilite=True` est CONSERVE. Le gradient n'est pas utilise ici,
        mais le demander fait partie de ce qui a ete paye jusqu'ici : ne plus
        le demander changerait le cout et le journal de la grille, ce qui est
        une decision d'etude, pas un nettoyage.
        """
        return self._en_U(u, exiger_gradient=False)

    def evaluer_en_U(self, u):
        """L'etat limite en UN point de l'espace standard.

        Retourne `(g, gradient en U, gradient en X)`, et LEVE si le solveur
        n'a rendu aucun gradient : les points qu'elle produit rejoignent le
        plan d'experiences.

        Elle maille EXACTEMENT comme `evaluer_plan` -- ce qui n'etait pas le
        cas avant la phase 5, et alimentait le meme metamodele avec deux
        surfaces differentes.
        """
        return self._en_U(u, exiger_gradient=True)

    def _en_U(self, u, exiger_gradient):
        solveur = self.solveur_pour(None)
        n_var = len(u)
        T_inv = self.dist.getInverseIsoProbabilisticTransformation()
        u_point = ot.Point(list(u))
        x_point = T_inv(u_point)
        p_vals = [float(x_point[j]) for j in range(n_var)]

        etiquette = self.etiquette("HF", p_vals, u=u) if self.archiver else None
        ev = solveur.evaluer(
            {self.params_names[i]: x_point[i] for i in range(n_var)},
            sensibilite=True, etiquette=etiquette)
        self._signaler_si_non_convergent(
            ev, "run_HF %s" % (list(u),),
            "point d'enrichissement %s" % (list(u),))

        grad_X = list(ev.grad_x)
        grad_U = [None] * n_var
        if ev.gradient_complet:
            grad_U = gradient_vers_U(grad_X, u, T_inv)
        if exiger_gradient and any(v is None for v in grad_U):
            # On LEVE ici, la ou le plan d'experiences ECARTE. La difference
            # est voulue : ce point-la a ete DEMANDE par l'algorithme
            # d'enrichissement. L'ecarter en silence lui ferait reproposer le
            # meme point, indefiniment.
            #
            # Une SURFACE DE FOND, elle, ne veut qu'un `g` : c'est
            # `evaluer_g_en_U` qui la sert, et elle passe ici sans exiger.
            raise ValueError(
                "run_HF en u=%s : le solveur n'a rendu aucun gradient "
                "(grad_HF_X=%s). Un gradient fabrique a 0 affirmerait que "
                "l'etat limite est plat ici, et le metamodele l'ajusterait. "
                "Le plan d'experiences, lui, ecarte ces points -- voir "
                "`exclure_points_sans_gradient`." % (list(u), grad_X))
        self.journaliser(u, x_point, ev.g)
        return ev.g, grad_U, grad_X


# --------------------------------------------------------------------------- #
# LE BATCH D'ENRICHISSEMENT : des points choisis, evalues, verses au plan      #
# --------------------------------------------------------------------------- #
def evaluer_batch_EFF(batch, xt, yt, all_grad, xt_eff, *,
                      n_max_points, n_batch, n_workers, n_var,
                      evaluer_un_point, executer_en_parallele=None,
                      dist_jointe=None, params_names=None,
                      taylor=False, eps_taylor=0.0, tracer=_ecrire):
    """Les points proposes par le critere EFF, evalues et verses au plan.

    Deux chemins, et c'est le batch qui tranche :

    * a plusieurs points ET plusieurs workers, ils partent ensemble au pool
      (`executer_en_parallele`) ;
    * sinon, un par un (`evaluer_un_point`), avec la possibilite d'ajouter
      des points VIRTUELS de Taylor autour du point reel.

    ATTENTION -- LE CHEMIN PARALLELE N'EST EXERCE PAR AUCUNE CHAINE DE
    VERIFICATION. `n_workers_DOE` vaut 1 dans les etudes analytiques, 6 sur
    le Moulin Blanc : la branche qui tourne EN PRODUCTION est precisement
    celle qu'aucun run de controle ne traverse. Ses tests unitaires sont donc
    le seul filet -- voir `test_106_batch_EFF`.

    Les deux journaux different (precision 4 et compteur d'un cote, precision
    10 de l'autre). C'est ainsi depuis l'origine ; les formats sont repris
    verbatim, parce que la comparaison des journaux est ce qui atteste des
    extractions.

    Retourne `(xt, yt, all_grad, xt_eff)` -- les quatre etats du plan.
    """
    n_actuel = min(n_batch, n_max_points - len(xt_eff))
    a_evaluer = batch[:n_actuel]

    if len(a_evaluer) > 1 and n_workers > 1:
        # `dist_jointe` est un APPEL, pas une distribution : la construire
        # coute, et le chemin sequentiel n'en a aucun besoin.
        T_inv = dist_jointe().getInverseIsoProbabilisticTransformation()
        SOL = []
        for u_pt in a_evaluer:
            x_pt = T_inv(ot.Point(list(u_pt)))
            SOL.append({p: float(x_pt[j]) for j, p in enumerate(params_names)})
        SOL = executer_en_parallele(SOL, min(n_workers, len(a_evaluer)))
        for k, u_pt in enumerate(a_evaluer):
            g_k = SOL[k]['g']
            grad_k = [SOL[k].get('dg_%s' % p, 0.0) for p in params_names]
            xt_eff.append(np.array(u_pt))
            xt = np.vstack([xt, [np.array(u_pt)]])
            yt = np.vstack([yt, [[g_k]]])
            all_grad = np.vstack([all_grad, [grad_k]])
            tracer("[EFF HF %d/%d] u=%s  g=%.6f  grad_U=%s"
                   % (k + 1, len(a_evaluer), list(np.round(u_pt, 4)), g_k,
                      [round(v, 6) for v in grad_k]))
        return xt, yt, all_grad, xt_eff

    for u_pt in a_evaluer:
        g_val, grad_U, _ = evaluer_un_point(np.array(u_pt))
        xt_eff.append(np.array(u_pt))
        xt = np.vstack([xt, [np.array(u_pt)]])
        yt = np.vstack([yt, [[g_val]]])
        grad_val = np.array([[float(grad_U[i]) for i in range(n_var)]])
        all_grad = np.vstack([all_grad, grad_val])
        tracer("[EFF HF] u=%s  g=%.10f  grad_U=%s"
               % (list(np.round(u_pt, 10)), g_val,
                  [round(float(grad_U[i]), 10) for i in range(n_var)]))

        # Points virtuels au premier ordre autour du point qui vient d'etre
        # calcule. MEME formule que `_doe.plan.augmenter_par_taylor`, qui la
        # fait sur tout le plan a la fois -- seuls les journaux different.
        # L'egalite numerique des deux est verifiee par `test_106_batch_EFF`
        # ; les unifier changerait le journal, donc pas aujourd'hui.
        if taylor and eps_taylor > 0:
            for i_dim in range(n_var):
                u_virt = np.array(u_pt) + eps_taylor * np.eye(n_var)[i_dim]
                y_virt = g_val + eps_taylor * float(grad_U[i_dim])
                xt = np.vstack([xt, [u_virt]])
                yt = np.vstack([yt, [[y_virt]]])
                all_grad = np.vstack([all_grad, grad_val])
                tracer("[EFF Taylor] u=%s  y_taylor=%.10f  (eps=%s, dim=%d)"
                       % (list(np.round(u_virt, 10)), y_virt, eps_taylor, i_dim))
    return xt, yt, all_grad, xt_eff
