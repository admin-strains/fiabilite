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

import json
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
                 fichier_cache_points=None, evaluer_lot=None,
                 coupe_initiale=None, coupe_courante=None,
                 points_libres=None, active=True,
                 signature=None, config_identique=None,
                 marquer_phase=None, tracer=_ecrire):
        self.evaluer = evaluer
        self.n_var = n_var
        self.cote = cote
        self.u1_min, self.u1_max, self.u2_min, self.u2_max = bornes
        self.fichier_cache = fichier_cache
        self.fichier_cache_complet = fichier_cache_complet
        self.fichier_cache_points = fichier_cache_points
        #: `evaluer_lot(points) -> [g]` -- l'evaluation en parallele, quand
        #: elle est disponible. None = un point apres l'autre.
        self.evaluer_lot = evaluer_lot
        #: La coupe que l'etude regarde par defaut. Elle sert de REFERENCE :
        #: une coupe qui lui est egale n'a aucune raison d'etre recalculee
        #: dans un second cache -- c'est la garde des 29 heures.
        self.coupe_courante = coupe_courante
        #: Des points CHOISIS a la main plutot qu'un quadrillage regulier.
        #: Quand ils existent, ils remplacent la grille : voir
        #: `depuis_points_libres`.
        self.points_libres = points_libres
        #: Une grille eteinte ne coute rien : les figures sortent sans fond.
        self.active = active
        self.signature = signature
        self.config_identique = config_identique
        self.marquer_phase = marquer_phase or (lambda _p: None)
        self.tracer = tracer
        #: la grille complete en memoire, et ses axes -- remplis par
        #: `calculer_complete`, lus par `coupe_depuis_complete`.
        self.complete = None
        self.axes_complets = None
        #: memo de `depuis_points_libres` : eviter de relire le JSON a chaque
        #: figure qui redemande le meme fond.
        self._resultat_points = None
        #: Les deux coupes du run : la COURANTE et la FINALE (celle choisie
        #: sur les facteurs d'importance). Elles etaient deux variables
        #: globales du script, rangees par une fonction dont le seul role
        #: etait de choisir laquelle des deux ecrire.
        self.coupes = {"courante": coupe_initiale, "finale": None}

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

    # ------------------------------------------------------------------ #
    # une grille de points CHOISIS, plutot qu'un quadrillage
    # ------------------------------------------------------------------ #
    def depuis_points_libres(self, points, marge=0.1, n_interp=50):
        """La surface a partir d'une liste de points quelconques.

        Un quadrillage regulier depense la moitie de son budget loin de l'etat
        limite. Quand on sait deja ou il passe -- apres un premier run, ou
        d'apres un modele analytique -- on peut placer les points a la main et
        interpoler entre eux. C'est ce que fait `hf_custom_points`.

        Retourne `(Z, UX, UY)` sur une grille reguliere de `n_interp` cotes,
        cadree sur l'enveloppe des points elargie de `marge`.

        COUT : un appel solveur par point non deja calcule. Le cache partiel
        est ecrit APRES CHAQUE POINT.
        """
        if points is None:
            return None, None, None
        if self._resultat_points is not None:
            return self._resultat_points

        pts = np.array(points)
        n_total = len(pts)

        g_vals = self._lire_cache_points_complet(n_total)
        if g_vals is None:
            g_vals = self._calculer_points_manquants(
                pts, self._lire_cache_points_partiel(n_total))
            self._ecrire_cache_points(pts, g_vals, n_total)

        self._resultat_points = self._interpoler(pts, np.array(g_vals, float),
                                                 marge, n_interp)
        return self._resultat_points

    # -- les quatre morceaux, separes : la version d'origine les melait, et
    # -- construisait DEUX FOIS la meme grille d'interpolation.
    def _lire_cache_points_complet(self, n_total):
        if not self.config_identique or not self.fichier_cache_points:
            return None
        if not os.path.exists(self.fichier_cache_points):
            return None
        try:
            d = json.load(open(self.fichier_cache_points))
        except Exception:
            return None
        # La signature manquait : une grille calculee sous un autre solveur,
        # un autre maillage ou d'autres bornes etait relue telle quelle.
        if not (d.get('complet') and d.get('n_total') == n_total
                and d.get('signature') == self.signature):
            return None
        self.tracer("[HF CUSTOM] cache complet charge (%d pts) -> 0 SOCP" % n_total)
        return d['g_vals']

    def _lire_cache_points_partiel(self, n_total):
        vide = [None] * n_total
        if not self.config_identique or not self.fichier_cache_points:
            return vide
        partiel = self.fichier_cache_points + '.partial'
        if not os.path.exists(partiel):
            return vide
        try:
            d = json.load(open(partiel))
        except Exception:
            return vide
        if d.get('n_total') == n_total and d.get('signature') == self.signature:
            return d['g_vals']
        return vide

    def _ecrire_cache_points(self, pts, g_vals, n_total):
        if not self.fichier_cache_points:
            return
        try:
            json.dump({'n_total': n_total, 'pts': pts.tolist(),
                       'g_vals': g_vals, 'complet': True,
                       'signature': self.signature},
                      open(self.fichier_cache_points, 'w'), indent=1)
            partiel = self.fichier_cache_points + '.partial'
            if os.path.exists(partiel):
                os.remove(partiel)
        except Exception:
            pass

    def _sauver_partiel_points(self, g_vals, n_total):
        if not self.fichier_cache_points:
            return
        try:
            json.dump({'n_total': n_total, 'g_vals': g_vals,
                       'signature': self.signature},
                      open(self.fichier_cache_points + '.partial', 'w'), indent=1)
        except Exception:
            pass

    def _calculer_points_manquants(self, pts, g_vals):
        n_total = len(pts)
        a_faire = [i for i in range(n_total) if g_vals[i] is None]
        n_sautes = n_total - len(a_faire)
        self.tracer("[HF CUSTOM] %d points, %d deja calcules, %d a faire"
                    % (n_total, n_sautes, len(a_faire)))
        if not a_faire:
            return g_vals

        liste = [list(pts[i]) for i in a_faire]
        if self.evaluer_lot is not None and len(liste) > 1:
            for k, valeur in enumerate(self.evaluer_lot(liste)):
                g_vals[a_faire[k]] = valeur
        else:
            self.marquer_phase("HF_CUSTOM")
            t_debut = time.perf_counter()
            for k, pt in enumerate(liste):
                g_val = self.evaluer(pt)[0]
                g_vals[a_faire[k]] = g_val
                self._sauver_partiel_points(g_vals, n_total)
                t_ecoule = time.perf_counter() - t_debut
                t_eta = (t_ecoule / (k + 1)) * (len(liste) - k - 1)
                self.tracer("  [HF CUSTOM %d/%d]  u=[%s]  g=%+.4f  ETA=%.1fmin"
                            % (n_sautes + k + 1, n_total,
                               ", ".join("%+.3f" % v for v in pt),
                               g_val, t_eta / 60))
        self.tracer("[HF CUSTOM] %d points calcules (%d skip)"
                    % (len(a_faire), n_sautes))
        return g_vals

    @staticmethod
    def _interpoler(pts, g_arr, marge, n_interp):
        """Une surface reguliere a partir de points epars.

        La grille etait construite DEUX FOIS dans la version d'origine -- une
        fois dans la branche « cache complet », une fois a la fin -- avec les
        memes constantes recopiees.
        """
        from scipy.interpolate import griddata
        ux = np.linspace(pts[:, 0].min() - marge, pts[:, 0].max() + marge, n_interp)
        uy = np.linspace(pts[:, 1].min() - marge, pts[:, 1].max() + marge, n_interp)
        UX, UY = np.meshgrid(ux, uy)
        return griddata(pts, g_arr, (UX, UY), method='linear'), UX, UY


    # ------------------------------------------------------------------ #
    # une coupe, par la voie la MOINS chere
    # ------------------------------------------------------------------ #
    def coupe(self, sd, fichier=None, finale=False):
        """Z sur une coupe. Cascade, du gratuit au couteux :

          1. la memoire      -- 0 appel
          2. le cache disque -- 0 appel (sous signature)
          3. la grille complete, si elle existe -- 0 appel (interpolation)
          4. le calcul       -- cote^2 appels solveur

        `finale=True` designe la seconde coupe -- celle choisie sur les
        facteurs d'importance a la fin du run. Elle a sa propre memoire et
        son propre fichier : deux coupes differentes ne doivent pas se
        recouvrir. Quand elles coincident, l'appelant passe `finale=False`
        et paie une seule fois -- c'est la garde qui a economise 29 heures
        sur le Moulin Blanc.
        """
        cle = "finale" if finale else "courante"
        memo = self.coupes[cle]
        if memo is not None:
            return np.array(memo["Z"])

        Z = self.lire_cache_2d(sd, fichier=fichier)
        if Z is not None:
            self.coupes[cle] = self._description(sd, self.cote, Z)
            return Z

        if self.complete is not None:
            self.tracer("[HF SLICE] extraction depuis grille full pour coupe "
                        "(%s,%s)" % (sd[0], sd[1]))
            Z = self.coupe_depuis_complete(sd)
            self.ecrire_cache_2d(Z, sd, fichier=fichier)
            self.coupes[cle] = self._description(sd, self.cote, Z)
            return Z

        Z, description = self.calculer_2d(self.points_de_coupe(sd),
                                          contexte="get_hf_slice",
                                          fichier=fichier, coupe=sd)
        self.coupes[cle] = description
        return Z


    # ------------------------------------------------------------------ #
    def fond_de_figure(self, coupe=None, fichier=None, finale=False):
        """Le fond de contour haute fidelite d'une figure, et CE QU'IL COUTE.

        Methode SEPAREE ET NOMMEE parce que c'est une ACTION, pas un detail
        de trace : jusqu'a `cote ** 2` appels au solveur. Sur le Moulin Blanc
        regle a 15, cela fait 225 appels, soit 29 heures -- pour un fond de
        figure. Appelee explicitement, elle se voit ; obtenue au fond d'un
        `print_*`, elle ne se voyait pas.

        Rendre None revient a tracer sans fond : les figures sortent quand
        meme, et sans un seul appel au solveur.

        LA GARDE DES 29 HEURES. Une coupe egale a `coupe_courante` n'a aucune
        raison d'etre recalculee dans un SECOND cache : c'est la meme grille.
        Elle l'etait, parce que la coupe finale etait servie par
        `hf_grid_cache_final.json` alors que la courante l'est par
        `hf_grid_cache.json` -- et les deux coupes valent `(0, 1, {})` des
        qu'il y a deux variables. Cout mesure : `cote^2` appels solveur en
        double.

        Retourne `(Z, UX, UY)`, ou None. Le maillage vient de la grille QUI A
        CALCULE Z, jamais d'ailleurs -- voir `test_110`.
        """
        sd = coupe if coupe is not None else (
            self.coupe_courante if self.coupe_courante is not None else (0, 1, {}))
        if sd == self.coupe_courante:
            fichier, finale = self.fichier_cache, False
        if self.points_libres is not None:
            return self.depuis_points_libres(self.points_libres)
        if not self.active:
            return None
        UX, UY = self.maillage_2d()
        return (self.coupe(sd,
                           fichier=fichier if fichier is not None
                           else self.fichier_cache,
                           finale=finale),
                UX, UY)

    def points_de_coupe(self, sd):
        """Les points du quadrillage 2D, en coordonnees COMPLETES.

        Une coupe fixe toutes les variables sauf deux ; le solveur, lui,
        attend un point entier.
        """
        idx_x, idx_y, fixes = sd
        UX, UY = self.maillage_2d()
        points = np.zeros((self.cote * self.cote, self.n_var))
        points[:, idx_x] = UX.ravel()
        points[:, idx_y] = UY.ravel()
        for idx, valeur in fixes.items():
            points[:, idx] = valeur
        return points

    # ------------------------------------------------------------------ #
    # la surface en trois dimensions
    # ------------------------------------------------------------------ #
    def surface_3d(self, fige=None, ecrire_recopiable=True):
        """La surface `g` sur un quadrillage, pour un trace en relief.

        `fige` est une grille deja calculee, recopiee dans le fichier d'etude
        sous la forme `{'params': (...), 'Z': [[...]]}` : elle evite de
        repayer les appels. C'est le seul cache qui vit dans le fichier
        d'etude plutot que sur le disque -- il sert a figer une figure pour
        un rapport.

        COUT : `cote^2` appels solveur si `fige` est None.

        Retourne `(U1, U2, Z)`.
        """
        if fige is not None:
            self.tracer("Cache hf_3d_grid_fixed disponible - pas d'appels solveur.")
            u1_min, u1_max, u2_min, u2_max, cote = fige['params']
            U1, U2 = np.meshgrid(np.linspace(u1_min, u1_max, cote),
                                 np.linspace(u2_min, u2_max, cote))
            return U1, U2, np.array(fige['Z'])

        U1, U2 = np.meshgrid(np.linspace(self.u1_min, self.u1_max, self.cote),
                             np.linspace(self.u2_min, self.u2_max, self.cote))
        points = np.column_stack([U1.ravel(), U2.ravel()])
        self.tracer("Evaluation HF grille %dx%d = %d appels solveur..."
                    % (self.cote, self.cote, self.cote ** 2))
        Z = np.array([self.evaluer(pt)[0] for pt in points]).reshape(self.cote,
                                                                     self.cote)
        if ecrire_recopiable:
            self.ecrire_recopiable(Z)
        return U1, U2, Z

    def ecrire_recopiable(self, Z):
        """La grille sous une forme a recopier dans un fichier d'etude.

        C'est la seule trace de `cote^2` appels solveur : sans elle, une
        grille calculee en 29 heures ne survit pas a la fermeture du terminal
        si le cache disque est efface.
        """
        self.tracer("\nhf_3d_grid_fixed = {")
        self.tracer("    'params': (%s, %s, %s, %s, %s),"
                    % (self.u1_min, self.u1_max, self.u2_min, self.u2_max,
                       self.cote))
        self.tracer("    'Z': [")
        for ligne in Z:
            self.tracer("        [%s]," % ", ".join("%.6f" % v for v in ligne))
        self.tracer("    ]")
        self.tracer("}")
