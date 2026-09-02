r"""La boucle d'enrichissement EFF : choisir un point, le payer, reajuster, juger.

CE QUE FAIT CE MODULE
----------------------
Le metamodele est ajuste sur un plan initial. Tant qu'il reste trop incertain
la ou il compte -- pres de l'etat limite, dans la zone qui porte la
probabilite de defaillance -- on lui ajoute des points. Le critere EFF
designe ou ; le solveur paie ; le controleur FORM+IS dit quand s'arreter.

Cette boucle etait ecrite DEUX FOIS, a l'identique au caractere pres :
136 lignes dans `AC3_pure_flexion.py`, 136 dans `AC3_moulinblanc.py`. Toute
correction devait etre portee deux fois, et trois fois cette semaine elle ne
l'a ete que d'un cote (le lanceur parallele, la reprise des compteurs EFF,
`xt_eff` a la reprise). Elle est ici en un seul exemplaire.

CE QUI RESTE A L'ETUDE, ET POURQUOI
------------------------------------
Tout ce que la boucle sait faire, elle le fait par des COLLABORATEURS que
l'etude lui donne : ou trouver le prochain point, comment l'evaluer, comment
reajuster, ou dessiner. La boucle ne connait ni le solveur, ni le
metamodele, ni le format des figures.

Ce partage n'est pas cosmetique. `evaluer_batch` porte le nom du modele, la
liste des parametres, le lanceur parallele : autant de choses qui n'ont de
sens que dans une etude. La boucle, elle, porte l'ordre des gestes -- et
c'est l'ordre qui etait duplique.

LES REGLAGES DU TIRAGE D'IMPORTANCE PARALLELE
-----------------------------------------------
`_IS_K`, `_IS_CHUNK`, `_IS_PROBE` et `_IS_PARALLEL` etaient lus dans l'en-tete
des deux etudes, et n'y servaient QU'ICI. Ils sont donc lus ici. Ce sont des
reglages d'execution -- ils viennent de l'environnement, pas du fichier
d'etude, parce qu'ils dependent de la machine et non du calcul.
"""

import os

import numpy as np
import openturns as ot

import arret as _arret_eff
import controle as _controle
import eff as _eff
import eff_ot as _eff_ot
import evaluation as _evaluation
import form as _form
from _parallel_is import adaptive_is

#: Le tirage d'importance parallele, regle par l'environnement.
_IS_PARALLEL = os.environ.get("_IS_PARALLEL", "1") != "0"
_IS_K = int(os.environ.get("_IS_K", "16"))
_IS_CHUNK = int(os.environ.get("_IS_CHUNK", "8"))
_IS_PROBE = int(os.environ.get("_IS_PROBE", "16"))


def _ecrire(message):
    print(message, flush=True)


class BoucleEFF:
    """L'enrichissement d'un metamodele par le critere EFF.

    `historiques` est le dictionnaire des cinq listes tenues par l'etude
    (`EFF`, `BB`, `BS`, `Pf`, `beta_IS`). Elles sont remplies EN PLACE, jamais
    rebindees : `ArretEFF` recoit ces memes objets, et une liste rebindee le
    laisserait ecrire dans une liste que plus personne ne lit -- c'est arrive
    le 27/08/2026, et six cent vingt-trois tests verts ne l'ont pas vu.

    CE QUE SEULE L'ETUDE SAIT -- a fournir :

    ``ajuster(g_ot, sigma_func, xt, yt, all_grad)``
        le metamodele reajuste sur le plan augmente, en cinq sorties.
    ``evaluer_un_point(u)``
        un point paye au solveur, en `(g, gradient)`. C'est LA frontiere avec
        Digital Structure : la boucle ne connait pas le solveur.
    ``executer_en_parallele(SOL, n_workers)``
        le meme travail pour plusieurs points a la fois, quand il y a de quoi
        paralleliser. Elle porte le nom du modele, que la boucle ignore.
    ``params_names`` et ``param_config``
        les noms des variables et leur catalogue, dont le chemin parallele a
        besoin pour repasser de l'espace standard aux parametres physiques.
    ``predire``
        le predicteur du modele courant, en VALEUR. Les fonctions de
        prediction vivent dans `_lib` et `_reliability` n'a pas a en
        dependre ; c'est la seule chose que la boucle ne deduit pas de `cfg`.
    ``figure(g_ot, sigma_func, xt, xt_eff)``
        la planche intermediaire, si l'etude en veut une.
    ``sauver(xt, yt, all_grad, xt_eff)``
        le dump de reprise, appele a la fin de chaque tour.

    CE QUE LA BOUCLE SE DONNE ELLE-MEME -- injectable, mais inutile de le
    faire hors des tests. Chacun etait un DELEGUE d'une ligne recopie dans
    les deux etudes, dont la seule fonction etait de lier `n_var` et `cfg.*`
    a un appel de module :

    ``fonction_EFF(g_ot, sigma_func)``
        le critere EFF sous une forme qu'`ot.Function` accepte
        (`cfg.epsilon_factor`, `n_var`).
    ``bornes_surrogate(g_ot, sigma_func, signe)``
        l'encadrement `g +/- 2 sigma`, dont le controleur tire le ratio BB
        (`n_var`, `predire`).
    ``executer_is(modes, evenement)``
        le tirage d'importance de repli (`cfg.n_IS`, `cfg.cov_IS`, `n_var`).
    ``points_EFF(g_ot, sigma_func, xt, yt, all_grad)``
        les points du prochain batch, et la valeur du critere au premier
        (les six reglages EFF, le domaine, et `ajuster`).
    ``dist_jointe()``
        la loi jointe, des que `param_config` est donne.

    Ils restent des parametres parce que `tests/test_119_boucle_eff.py` en
    stube quatre pour eprouver l'ORDRE DES GESTES sans metamodele : une
    boucle qui les fabriquerait sans recours ne serait plus testable seule.
    """

    def __init__(self, cfg, n_var, *, journal, historiques,
                 ajuster, evaluer_un_point,
                 points_EFF=None, fonction_EFF=None, bornes_surrogate=None,
                 executer_is=None, executer_en_parallele=None,
                 dist_jointe=None, params_names=None, param_config=None,
                 predire=None, figure=None, sauver=None, tracer=_ecrire):
        self.cfg = cfg
        self.n_var = n_var
        self.journal = journal
        self.hist = historiques
        self.ajuster = ajuster
        self.evaluer_un_point = evaluer_un_point
        self.executer_en_parallele = executer_en_parallele
        self.params_names = params_names
        self.param_config = param_config
        self.predire = predire
        # QUATRE COLLABORATEURS QUE LA BOUCLE SE DONNE ELLE-MEME
        #
        # Ils ne dependaient de rien que la boucle ne sache deja : le nombre
        # de variables, les reglages du fichier d'etude, et le predicteur du
        # modele courant. Les etudes les fabriquaient pourtant chacune, en
        # quatre fonctions d'une ligne -- des DELEGUES, dont la seule raison
        # d'exister etait de lier `n_var` et `cfg.*` a un appel de module.
        # Mesure du 02/09/2026 : 21 delegues de cette nature dans chaque AC,
        # portant les MEMES 21 noms des deux cotes.
        #
        # Ils restent injectables, et ce n'est pas de la politesse : les
        # temoins de `test_119_boucle_eff.py` en stubent quatre pour eprouver
        # l'ORDRE DES GESTES sans metamodele. Une boucle qui les fabriquerait
        # sans recours ne serait plus testable seule.
        self.fonction_EFF = fonction_EFF or self._critere_EFF
        self.bornes_surrogate = bornes_surrogate or self._encadrement
        self.executer_is = executer_is or self._tirage_d_importance
        self.points_EFF = points_EFF or self._prochain_batch
        self.dist_jointe = dist_jointe or (
            self._loi_jointe if param_config is not None else None)
        # Facultatifs tous les deux. Ils sont eprouves par `is not None` au
        # point d'usage, jamais par leur valeur de verite : un appelable a le
        # droit d'etre faux, et ce piege a deja remplace deux fois un journal
        # vide par le journal par defaut.
        self.figure = figure
        self.sauver = sauver
        self.tracer = tracer

    # ------------------------------------------------------------------ #
    # CE QUE LA BOUCLE DERIVE, ET DE QUOI                                  #
    # ------------------------------------------------------------------ #
    @property
    def _bornes(self):
        """Le domaine de recherche, `eff_bound_*` repete sur chaque variable.

        Les etudes l'ecrivaient en deux lignes chacune. Il ne depend que du
        fichier d'etude et du nombre de variables.
        """
        return ([self.cfg.eff_bound_min] * self.n_var,
                [self.cfg.eff_bound_max] * self.n_var)

    def _critere_EFF(self, g_ot, sigma_func):
        """`EFFFunction` des etudes : le critere sous une forme `ot.Function`."""
        return _eff_ot.eff_function(g_ot, sigma_func, self.n_var,
                                    self.cfg.epsilon_factor)

    def _encadrement(self, g_ot, sigma_func, signe):
        """`BoundSurrogateFunction` des etudes : `g +/- 2 sigma`.

        Demande le PREDICTEUR du modele courant. C'est la seule chose que la
        boucle ne peut pas deduire de `cfg` : les fonctions de prediction
        vivent dans `_lib`, et `_reliability` n'a pas a en dependre. L'etude
        la passe donc comme VALEUR (`predire=`), non comme fermeture.
        """
        if self.predire is None:
            raise ValueError(
                "`bornes_surrogate` doit etre fourni, ou bien `predire` pour "
                "que la boucle le fabrique : l'encadrement a besoin du "
                "predicteur du modele courant.")
        return _form.bound_surrogate_function(g_ot, sigma_func, signe,
                                              self.n_var, self.predire)

    def _tirage_d_importance(self, modes, evenement):
        """`run_IS` des etudes : le tirage d'importance de repli."""
        return _form.run_IS(modes, evenement, self.n_var,
                            self.cfg.n_IS, self.cfg.cov_IS)

    def _prochain_batch(self, g_ot, sigma_func, xt, yt, all_grad):
        """`_find_batch_EFF_points` des etudes : maximisation du critere puis
        Kriging Believer, avec le reajustement que la boucle a deja."""
        bmin, bmax = self._bornes
        cfg = self.cfg
        return _eff_ot.batch_kriging_believer(
            g_ot, sigma_func, xt, yt, all_grad,
            n_batch=cfg.n_batch_EFF, bornes_min=bmin, bornes_max=bmax,
            n_var=self.n_var, n_appels=cfg.n_NLopt_EFF,
            epsilon_factor=cfg.epsilon_factor, reajuster=self.ajuster,
            gradient_du_surrogate=cfg.do_GEPCK)

    def _loi_jointe(self):
        """`dist_jointe` des etudes, quand `param_config` est fourni."""
        import lois as _lois                                # noqa: PLC0415
        return _lois.dist_jointe(self.param_config, self.params_names)

    # ------------------------------------------------------------------ #
    def _controleur(self):
        """Le juge de l'arret : FORM + tirage d'importance sur le courant.

        `adaptatif` est desactive pour un PCK : le format de son metamodele
        n'est pas celui que le tirage parallele attend.
        """
        cfg = self.cfg
        return _controle.ControleurFORM(
            self.n_var, cfg.n_max_FORM, cfg.tol_FORM, executer_is=self.executer_is,
            adaptatif=(adaptive_is if (_IS_PARALLEL and not cfg.do_PCK) else None),
            cov_cible=cfg.cov_IS, n_is=cfg.n_IS,
            K=_IS_K, chunk=_IS_CHUNK, sondes=_IS_PROBE)

    # ------------------------------------------------------------------ #
    def _payer_le_batch(self, batch, xt, yt, all_grad, xt_eff):
        """Les points proposes, evalues et verses au plan.

        Le partage suit la meme regle que partout ici : les REGLAGES (combien
        de points au plus, quelle taille de batch, combien de workers, faut-il
        des points virtuels de Taylor) viennent du fichier d'etude et sont
        donc lus ici ; ce que seule l'etude sait -- comment on evalue, sous
        quel nom de modele -- lui a ete demande a la construction.

        `taylor` n'a de sens qu'a un seul point : un batch de Kriging Believer
        apporte deja plusieurs points, et les entourer de points virtuels
        melangerait deux facons d'enrichir.
        """
        cfg = self.cfg
        return _evaluation.evaluer_batch_EFF(
            batch, xt, yt, all_grad, xt_eff,
            n_max_points=cfg.n_max_EFF_points, n_batch=cfg.n_batch_EFF,
            n_workers=cfg.n_workers_DOE, n_var=self.n_var,
            evaluer_un_point=self.evaluer_un_point,
            executer_en_parallele=self.executer_en_parallele,
            dist_jointe=self.dist_jointe, params_names=self.params_names,
            taylor=(cfg.do_PCK and cfg.n_batch_EFF <= 1),
            eps_taylor=cfg.eps_taylor)

    # ------------------------------------------------------------------ #
    def enrichir(self, g_ot, sigma_func, xt, yt, all_grad, *,
                 max_degree, xt_eff_initial=()):
        """Enrichit jusqu'a l'arret, et rend le plan augmente.

        Retourne `(g_ot, sigma_func, xt, yt, all_grad, xt_eff)`.

        `max_degree` EST UN ARGUMENT D'APPEL, PAS UN REGLAGE DE L'OBJET. Une
        reprise le relit dans le dump et ECRASE celui du fichier d'etude ; le
        figer a la construction rendrait donc la boucle sourde a la reprise.
        Il n'est pas lu ici : il voyage dans l'etat passe au controleur, que
        le reajustement du tirage adaptatif consomme.

        `xt_eff_initial` amorce la liste des points d'enrichissement quand on
        reprend un run interrompu. Sans elle, une reprise repartirait de zero
        point EFF et repaierait le budget entier -- jusqu'a `n_max_EFF_points`
        appels solveur.
        """
        cfg = self.cfg
        # Aucune branche active : ni metamodele a enrichir, ni critere a
        # evaluer. Le plan ressort intact, et sans point d'enrichissement.
        if g_ot is None or cfg.do_HF:
            return g_ot, sigma_func, xt, yt, all_grad, []
        self.journal.marquer("EFF")

        xt_eff = list(xt_eff_initial) if cfg.restart_enrich_only else []

        # L'etat du plan est PASSE au controleur, alors qu'il etait capture
        # par fermeture dans les etudes -- juste, mais invisible : rien ne
        # disait que le tirage adaptatif travaillait sur l'etat courant.
        controleur = self._controleur()

        def _etat():
            return dict(xt=xt, yt=yt, all_grad=all_grad, max_degree=max_degree)

        def _form_is(g_ot_i, label, sign=0, fm=None):
            return controleur.beta_et_pf(g_ot_i, label, sign=sign, fm=fm,
                                         etat=_etat())

        def _encadrer(g_ot_i, sigma_func_i, label, b_mid_precalc=None):
            return controleur.encadrement(
                g_ot_i, sigma_func_i, label, self.bornes_surrogate,
                etat=_etat(), beta_central=b_mid_precalc)

        # --- On resout u = argmax(EFF) (batch KB si n_batch_EFF > 1) ---
        batch_pts, eff_val_init = self.points_EFF(g_ot, sigma_func, xt, yt, all_grad)
        u_opt = ot.Point(batch_pts[0].tolist())
        f = ot.Function(self.fonction_EFF(g_ot, sigma_func))
        sigG = sigma_func(u_opt)
        muG = g_ot(ot.Point(u_opt))[0]
        eps = cfg.epsilon_factor * sigG
        self.tracer(f"  EFF debug u_opt={list(np.round(np.array(u_opt),3))} : "
                    f"sigmaG={sigG:.6f}  muG={muG:.6f}  epsilon={eps:.6f}")
        self.tracer(f"  EFF initial : EFF(u_opt)={eff_val_init:.6f}, tol={cfg.tol_EFF}")

        iter_count = 0

        b_mid, pf_mid_conv = _form_is(g_ot, f"N={len(xt)} initial mu conv")
        if cfg.restart_enrich_only:
            beta_IS = list(self.hist["beta_IS"]) + ([b_mid] if b_mid is not None else [])
        else:
            beta_IS = [b_mid] if b_mid is not None else []
        # Les historiques remis dans le bon etat, l'objet d'arret construit
        # dessus, les compteurs repris s'il y a lieu : `_reliability/arret.py`.
        arret = _arret_eff.ArretEFF.pour_un_run(
            cfg.EFF_criteria, cfg.tol_BB, cfg.tol_BS, cfg.n_max_EFF_points,
            cfg.tol_EFF, hist_BB=self.hist["BB"], hist_BS=self.hist["BS"],
            hist_Pf=self.hist["Pf"], reprise=cfg.restart_enrich_only)
        hist_Pf = self.hist["Pf"]
        # Lu APRES la boucle, meme si elle ne tourne jamais : `test_113`.
        ratio_bb = None
        # EFF initial, avant ajout du premier point.
        self.hist["EFF"].append(f(u_opt)[0])

        if cfg.print_Pf:
            controleur.amorcer_iteration(
                g_ot, sigma_func, self.bornes_surrogate, arret,
                n_points=len(xt), historique_Pf=hist_Pf, etat=_etat())

        while arret.continuer(len(xt_eff), f(u_opt)[0]):
            sigG = sigma_func(u_opt)
            muG = g_ot(ot.Point(u_opt))[0]
            self.tracer(f"  EFF={f(u_opt)[0]:.6f} > {cfg.tol_EFF} -- "
                        f"u_opt={list(np.round(np.array(u_opt),3))}  "
                        f"sigmaG={sigG:.6f}  muG={muG:.6f}")
            # EFF apres rebuild a cette iteration.
            self.hist["EFF"].append(f(u_opt)[0])

            xt, yt, all_grad, xt_eff = self._payer_le_batch(
                batch_pts, xt, yt, all_grad, xt_eff)
            g_ot, sigma_func, xt, yt, all_grad = self.ajuster(
                g_ot, sigma_func, xt, yt, all_grad)

            iter_count += 1
            b_mid, pf_mid_conv, ratio_bb = controleur.mesurer_iteration(
                g_ot, sigma_func, self.bornes_surrogate, arret,
                n_points=len(xt), iteration=iter_count, avec_Pf=cfg.print_Pf,
                historique_Pf=hist_Pf, historique_beta=beta_IS, etat=_etat())

            if cfg.print_EFF_progres and self.figure is not None:
                self.figure(g_ot, sigma_func, xt, xt_eff)

            # Dump de reprise a chaque tour : un run tue entre deux points ne
            # doit pas couter le round entier.
            self.hist["beta_IS"][:] = beta_IS
            if self.sauver is not None:
                self.sauver(xt, yt, all_grad, xt_eff)

            # --- On re-resout u = argmax(EFF) ---
            batch_pts, _ = self.points_EFF(g_ot, sigma_func, xt, yt, all_grad)
            u_opt = ot.Point(batch_pts[0].tolist())
            f = ot.Function(self.fonction_EFF(g_ot, sigma_func))

        # Le bilan : la decomposition du critere au point d'arret
        # (`_reliability/eff.py`), puis ce qui a arrete et sur quel
        # historique (`_reliability/arret.py`, ou vivent les compteurs).
        # sigma AVANT mu : l'ordre des deux appels est celui d'origine, et un
        # metamodele peut journaliser.
        sigG2 = sigma_func(u_opt)
        muG2 = g_ot(ot.Point(u_opt))[0]
        _eff.journaliser_decomposition(muG2, sigG2, cfg.epsilon_factor, u_opt)
        arret.bilan(f(u_opt)[0], len(xt_eff))

        # Avec `print_Pf`, le ratio vient du dernier encadrement de la boucle ;
        # sans lui, il faut le payer une fois de plus.
        if not cfg.print_Pf:
            ratio_bb, _, _, _ = _encadrer(g_ot, sigma_func, "BB final",
                                          b_mid_precalc=(b_mid, pf_mid_conv))
        self.tracer(f"  [BB informatif final] ratio = {ratio_bb}  "
                    f"tol_BB = {cfg.tol_BB}")

        self.hist["beta_IS"][:] = beta_IS
        return g_ot, sigma_func, xt, yt, all_grad, xt_eff
