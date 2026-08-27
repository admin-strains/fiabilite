r"""ACTION `grille` : la surface exacte de l'etat limite, point par point.

C'EST LE CALCUL LE PLUS CHER DU PROGRAMME
------------------------------------------
Une grille 15x15 = 225 appels solveur. A 466 s l'appel sur le Moulin Blanc,
cela fait 29 heures -- et elle arrive AVANT l'enrichissement, pas apres. Le
26/08/2026 j'ai annonce le contraire, faute d'avoir lu le code : elle etait
declenchee depuis une fonction nommee `print_planche_EFF`, appelee une ligne
avant `run_EFF`.

`n_grid_hf` n'est donc pas une resolution d'image : c'est un budget de calcul,
et son cout croit comme le carre.

CE QUE CE MODULE GARANTIT
--------------------------
* **La reprise apres interruption.** Chaque point calcule est ecrit
  immediatement dans un fichier `.partial`. Un run tue a la 200e evaluation
  reprend a la 201e ; il ne recommence pas 26 heures de calcul.
* **Le cache ne ment pas.** Toute lecture verifie une SIGNATURE portant le
  solveur et la geometrie de la grille. Sans elle, une grille calculee sur
  d'autres bornes serait relue telle quelle -- ce qui est arrive le
  26/08/2026 : apres avoir borne le domaine a +/- 6, le cache partiel
  contenait encore des valeurs calculees a +/- 7,5, sous un autre solveur
  lineaire, et elles auraient ete servies comme etant celles du nouveau
  domaine.

UNE HYPOTHESE QUI ATTEND SON JOUR
----------------------------------
La grille COMPLETE (n_var dimensions) construit tous ses axes sur
`u1_min..u1_max` -- y compris l'axe 2, dont les bornes propres sont
`u2_min..u2_max`. Les deux coincident dans toutes les etudes actuelles, donc
rien ne casse ; le jour ou elles differeront, l'interpolation d'une coupe
sortira du domaine. `verifier_bornes()` le dit tout haut plutot que de
laisser la surprise pour plus tard.
"""

import os
import time

import numpy as np

import hf as _cache_hf


def _ecrire(message):
    print(message, flush=True)


class Grille:
    """La grille haute fidelite d'une etude : geometrie, cache, evaluateur.

    `evaluer(u) -> (g, grad, ...)` est le seul lien avec le solveur ; le
    module ne connait ni le modele, ni la licence.
    """

    def __init__(self, evaluer, n_var, cote, bornes,
                 fichier_cache, fichier_cache_complet,
                 signature=None, config_identique=None,
                 marquer_phase=None, tracer=_ecrire):
        self.evaluer = evaluer
        self.n_var = n_var
        self.cote = cote
        self.u1_min, self.u1_max, self.u2_min, self.u2_max = bornes
        self.fichier_cache = fichier_cache
        self.fichier_cache_complet = fichier_cache_complet
        self.signature = signature
        self.config_identique = config_identique
        self.marquer_phase = marquer_phase or (lambda _p: None)
        self.tracer = tracer
        #: la grille complete en memoire, et ses axes -- remplis par
        #: `calculer_complete`, lus par `coupe_depuis_complete`.
        self.complete = None
        self.axes_complets = None

    # ------------------------------------------------------------------ #
    # geometrie
    # ------------------------------------------------------------------ #
    def axes_2d(self):
        return (np.linspace(self.u1_min, self.u1_max, self.cote),
                np.linspace(self.u2_min, self.u2_max, self.cote))

    def maillage_2d(self):
        ux, uy = self.axes_2d()
        return np.meshgrid(ux, uy)

    def verifier_bornes(self):
        """La grille complete suppose que TOUS les axes ont les bornes de u1.

        Renvoie None si l'hypothese tient, un message sinon. Ce n'est pas une
        exception : la grille 2D, elle, est correcte dans tous les cas.
        """
        if self.n_var < 2:
            return None
        if (self.u1_min, self.u1_max) == (self.u2_min, self.u2_max):
            return None
        return ("grille complete : tous les axes sont construits sur "
                "[%g, %g] (les bornes de u1) alors que u2 vaut [%g, %g]. "
                "Une coupe interpolee sortira du domaine."
                % (self.u1_min, self.u1_max, self.u2_min, self.u2_max))

    # ------------------------------------------------------------------ #
    # le cache 2D, sous signature
    # ------------------------------------------------------------------ #
    def lire_cache_2d(self, coupe, fichier=None, cote=None):
        """Z si le cache est utilisable, None sinon.

        « Utilisable » veut dire : meme coupe, meme cote, ET meme signature.
        Le 26/08/2026, un cache partiel calcule a +/- 7,5 sous cuDSS a failli
        etre servi comme etant la grille du domaine +/- 6 sous MUMPS.
        """
        return _cache_hf.load_hf_cache(self.cote if cote is None else cote,
                                       self.fichier_cache if fichier is None else fichier,
                                       coupe, self.config_identique,
                                       signature=self.signature)

    def ecrire_cache_2d(self, Z, coupe, fichier=None, cote=None):
        return _cache_hf.save_hf_cache(Z, self.cote if cote is None else cote,
                                       self.fichier_cache if fichier is None else fichier,
                                       coupe, signature=self.signature)

    # ------------------------------------------------------------------ #
    # grille 2D, avec reprise
    # ------------------------------------------------------------------ #
    def calculer_2d(self, points, cote=None, contexte="", fichier=None,
                    coupe=None):
        """La surface sur une coupe 2D. Retourne `(Z, description)`.

        `description` est le dictionnaire que l'etude range dans son etat de
        reprise ; il porte la coupe et le cote, de quoi savoir plus tard de
        quelle grille il s'agit.

        COUT : `cote^2` appels solveur si le cache est vide, moins ce que le
        fichier `.partial` d'un run interrompu permet de sauter.
        """
        cote = self.cote if cote is None else cote
        fichier = self.fichier_cache if fichier is None else fichier

        cached = _cache_hf.load_hf_cache(cote, fichier, coupe,
                                         self.config_identique,
                                         signature=self.signature)
        if cached is not None:
            return cached, self._description(coupe, cote, cached)

        self.marquer_phase("HF")
        n_total = len(points)
        # reprise : ce qu'un run interrompu a deja paye
        Z_flat = _cache_hf.load_hf_cache_partial(fichier, coupe, n_total,
                                                 self.config_identique,
                                                 signature=self.signature)
        if Z_flat is None:
            Z_flat = [None] * n_total
        n_sautes = sum(1 for v in Z_flat if v is not None)
        n_a_faire = n_total - n_sautes
        t_debut = time.perf_counter()
        self.tracer("\n##### HF GRID START: %dx%d = %d points solveur (%s)"
                    " [skip %d, calcul %d] #####"
                    % (cote, cote, n_total, contexte, n_sautes, n_a_faire))

        n_faits = 0
        for i, pt in enumerate(points):
            if Z_flat[i] is not None:
                continue
            t_pt0 = time.perf_counter()
            g_val = self.evaluer(pt)[0]
            Z_flat[i] = g_val
            n_faits += 1
            # ecriture APRES chaque point : c'est ce qui rend l'interruption
            # supportable.
            _cache_hf.save_hf_cache_partial(Z_flat, n_total, fichier, coupe,
                                            signature=self.signature)
            t_pt = time.perf_counter() - t_pt0
            t_ecoule = time.perf_counter() - t_debut
            t_eta = (t_ecoule / n_faits) * (n_a_faire - n_faits)
            self.tracer("  [HF GRID %2d/%d]  u=[%s]  g=%+.4f  dt=%.0fs  "
                        "elapsed=%.1fmin  ETA=%.1fmin"
                        % (n_sautes + n_faits, n_total,
                           ", ".join("%+.3f" % v for v in pt),
                           g_val, t_pt, t_ecoule / 60, t_eta / 60))

        self.tracer("\n##### HF GRID DONE in %.1f min (%d appels solveur, "
                    "%d skip) #####\n"
                    % ((time.perf_counter() - t_debut) / 60, n_faits, n_sautes))
        Z = np.array(Z_flat, dtype=float).reshape(cote, cote)
        _cache_hf.save_hf_cache(Z, cote, fichier, coupe,
                                signature=self.signature)
        partiel = fichier + '.partial'
        if os.path.exists(partiel):
            os.remove(partiel)
        return Z, self._description(coupe, cote, Z)

    @staticmethod
    def _description(coupe, cote, Z):
        return {'params': {'slice_def': coupe, 'n_grid_hf': cote},
                'Z': Z.tolist() if hasattr(Z, "tolist") else Z}

    # ------------------------------------------------------------------ #
    # grille complete (n_var dimensions)
    # ------------------------------------------------------------------ #
    def calculer_complete(self):
        """La surface sur tout le domaine : `cote^n_var` appels solveur.

        A trois variables et un cote de 15, cela fait 3 375 appels. Le cout
        n'est plus quadratique mais exponentiel en `n_var` : c'est la seule
        action du programme dont le budget peut depasser la semaine.
        """
        cached = _cache_hf.load_hf_grid_full(
            self.fichier_cache_complet, self.n_var, self.cote,
            self.config_identique, signature=self.signature)
        axes = [np.linspace(self.u1_min, self.u1_max, self.cote)
                for _ in range(self.n_var)]
        if cached is not None:
            self.complete, self.axes_complets = cached, axes
            return cached

        avertissement = self.verifier_bornes()
        if avertissement:
            self.tracer("  [GRILLE] ATTENTION : " + avertissement)

        self.marquer_phase("HF_FULL")
        grids = np.meshgrid(*axes, indexing='ij')
        points = np.column_stack([g.ravel() for g in grids])
        n_total = len(points)
        Z_flat = []
        t_debut = time.perf_counter()
        self.tracer("\n##### HF FULL GRID START: %d^%d = %d points solveur #####"
                    % (self.cote, self.n_var, n_total))
        for i, pt in enumerate(points):
            t_pt0 = time.perf_counter()
            g_val = self.evaluer(pt)[0]
            Z_flat.append(g_val)
            t_pt = time.perf_counter() - t_pt0
            t_ecoule = time.perf_counter() - t_debut
            t_eta = (t_ecoule / (i + 1)) * (n_total - i - 1)
            self.tracer("  [HF FULL %3d/%d]  u=[%s]  g=%+.4f  dt=%.0fs  "
                        "elapsed=%.1fmin  ETA=%.1fmin"
                        % (i + 1, n_total,
                           ", ".join("%+.3f" % pt[j] for j in range(self.n_var)),
                           g_val, t_pt, t_ecoule / 60, t_eta / 60))
        self.tracer("\n##### HF FULL GRID DONE in %.1f min (%d appels solveur) #####\n"
                    % ((time.perf_counter() - t_debut) / 60, n_total))
        Z_full = np.array(Z_flat).reshape([self.cote] * self.n_var)
        self.complete, self.axes_complets = Z_full, axes
        _cache_hf.save_hf_grid_full(self.fichier_cache_complet, Z_full,
                                    self.n_var, self.cote,
                                    signature=self.signature)
        return Z_full

    def coupe_depuis_complete(self, coupe):
        """Une coupe 2D interpolee dans la grille complete. ZERO appel solveur.

        C'est tout l'interet de payer la grille complete une fois : chaque
        coupe supplementaire est ensuite gratuite.
        """
        from scipy.interpolate import RegularGridInterpolator
        if self.complete is None:
            raise ValueError(
                "aucune grille complete en memoire : appeler "
                "`calculer_complete()` d'abord. (Elle coute cote^n_var appels "
                "solveur ; ce n'est pas une chose qu'on declenche par surprise "
                "au fond d'une fonction de trace.)")
        idx_x, idx_y, fixes = coupe
        interp = RegularGridInterpolator(self.axes_complets, self.complete,
                                         method='linear')
        UX, UY = self.maillage_2d()
        pts = np.zeros((self.cote * self.cote, self.n_var))
        pts[:, idx_x] = UX.ravel()
        pts[:, idx_y] = UY.ravel()
        for idx, val in fixes.items():
            pts[:, idx] = val
        return interp(pts).reshape(self.cote, self.cote)
