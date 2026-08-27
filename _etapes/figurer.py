r"""ACTION `figurer` : tout ce qui RESTITUE, rien qui CALCULE.

POURQUOI CE MODULE EXISTE
--------------------------
Le 26/08/2026, sept des onze fonctions `print_*` des scripts d'etude pouvaient
declencher un appel au solveur -- jusqu'a 225, soit 29 heures sur le Moulin
Blanc. Le chemin etait indirect, donc invisible a la lecture :

    print_planche_EFF -> _hf_from_custom_points -> run_HF
    print_3D_HF       -> run_HF                 (n_grid_hf^2 appels)
    print_results     -> run_HF                 (2 appels par mode FORM)

C'est ce qui m'a fait annoncer, a tort, que la grille haute fidelite arrivait
en DERNIER et qu'on pouvait l'interrompre sans rien perdre. Elle arrive AVANT
l'enrichissement. Une fonction nommee « print » n'eveille aucun soupcon.

LA REGLE DE CE FICHIER
-----------------------
Aucune fonction d'ici n'evalue l'etat limite. Ce n'est pas une intention :
`tests/test_93_figures_sans_solveur.py` construit le graphe d'appel du module
et calcule sa fermeture transitive. Une seule arete vers le solveur, meme a
travers trois intermediaires, fait echouer la suite.

Corollaire pratique : ce module prend en argument ce qui a deja ete calcule.
Il ne va pas le chercher.
"""

import openturns as ot


def _ecrire(message):
    """Sortie par defaut : vidangee, pour qu'un run suivi en direct ne
    reste pas muet pendant des heures dans le tampon."""
    print(message, flush=True)


def tracer_domaine_physique(dist, params_names, borne_min, borne_max,
                            ecrire=_ecrire):
    """Ce que les bornes de recherche valent EN UNITES PHYSIQUES.

    Personne ne lit `[-7.5, +7.5]` comme « de 8,9 a 461 MPa ». C'est pourtant
    ce que cela veut dire, et c'est ce qui a fait mourir un run de deux heures
    le 26/08/2026 : le rapport entre les deux nappes d'armatures atteignait 52
    a un coin du domaine, et Digital Structure a termine le processus.

    Renvoie la liste des couples (min, max) physiques, un par variable -- de
    quoi tester la fonction sans lire une sortie texte.

    ZERO appel solveur.
    """
    n_var = len(params_names)
    # T_inv est EXACTEMENT la transformation que l'evaluation de l'etat limite
    # emploie pour passer de u a x. En utiliser une autre ici ferait afficher
    # un domaine qui n'est pas celui qu'on evalue.
    #
    # La voie « naive » -- computeQuantile(Normal().computeCDF(u)) -- est de
    # surcroit FAUSSE dans la queue haute : `1 - p` y perd toute precision.
    # Mesure du 26/08/2026 sur Normal(235 ; 30,15) :
    #
    #     u = +6,0   ecart 2,7e-07
    #     u = +7,5   ecart 5,6e-03
    #     u = +8,5   ecart 11,3 MPa      (479,99 au lieu de 491,28)
    #
    # C'est le defaut corrige en phase 7 dans `_lib/transform.py`, et il a
    # failli etre reintroduit ici. T_inv : 5,7e-14 partout.
    T_inv = dist.getInverseIsoProbabilisticTransformation()
    bas_x = T_inv(ot.Point([borne_min] * n_var))
    haut_x = T_inv(ot.Point([borne_max] * n_var))

    ecrire("  DOMAINE DE RECHERCHE -- ce que les bornes valent physiquement")
    extremes = []
    for i, nom in enumerate(params_names):
        marginale = dist.getMarginal(i)
        lo, hi = sorted((float(bas_x[i]), float(haut_x[i])))
        extremes.append((lo, hi))
        alerte = ""
        if lo <= 0:
            alerte = "  <-- NEGATIF OU NUL"
        elif lo < 0.1 * float(marginale.getMean()[0]):
            alerte = "  <-- moins d'un dixieme de la moyenne"
        ecrire("    %-8s u in [%+.2f, %+.2f]  ->  [%.2f, %.2f]%s"
               % (nom, borne_min, borne_max, lo, hi, alerte))

    # Rapport le plus defavorable atteignable a un COIN du domaine : la
    # variable la plus haute contre la plus basse. C'est lui qui gouverne le
    # conditionnement des cones SOCP, pas la valeur absolue des bornes.
    if len(extremes) >= 2:
        bas = min(l for l, _ in extremes)
        haut = max(h for _, h in extremes)
        pire = haut / bas if bas > 0 else float('inf')
        ecrire("    rapport le plus defavorable aux coins : %.1f" % pire)
        if pire >= 5:
            ecrire("    ATTENTION : au-dela de ~5 les cones SOCP sont mal")
            ecrire("    conditionnes (docs/mesh/). A 52, le 26/08/2026,")
            ecrire("    Digital Structure a termine le processus.")
    ecrire("")
    return extremes


def resume_FORM(best_result, dist, params_names, ecrire=_ecrire):
    """Les resultats FORM, en clair.

    Cette moitie-la etait soudee a l'erreur FOSM, qui, elle, evalue l'etat
    limite EXACT en deux points. Une fonction nommee « print » lancait donc
    deux SOCP -- un quart d'heure sur le Moulin Blanc. Elles sont separees :
    le resume est ici, l'erreur FOSM reste du cote des actions qui coutent.

    Renvoie u*, dont l'appelant a besoin pour la suite.

    ZERO appel solveur.
    """
    u_star = best_result.getStandardSpaceDesignPoint()
    n_iter = best_result.getOptimizationResult().getIterationNumber()
    x_star = dist.getInverseIsoProbabilisticTransformation()(u_star)

    ecrire("n_iter FORM  = %d" % n_iter)
    for i, p in enumerate(params_names):
        ecrire("%s*          = %.4f" % (p, x_star[i]))
    ecrire("u*           = %s" % [round(v, 4) for v in u_star])
    ecrire("Imp.         = %s" % [round(v, 4) for v in best_result.getImportanceFactors()])
    ecrire("beta         = %.4f" % best_result.getHasoferReliabilityIndex())
    ecrire("Pf           = %.4e" % best_result.getEventProbability())
    return u_star


#: Les deux facons de cadrer une figure, telles qu'elles existaient.
CADRAGES = ("grille", "bornes_elargies")


def cadre_des_figures(mode, bornes_grille, eff_min, eff_max, marge=1.0):
    """Le rectangle dans lequel les figures sont tracees : (x0, x1, y0, y1).

    Deux facons de faire, qui coexistaient sans que rien ne le dise :

    * `"grille"` -- on cadre sur les bornes de la grille haute fidelite. La
      figure montre alors EXACTEMENT le domaine ou l'etat limite a ete evalue,
      ni plus ni moins.
    * `"bornes_elargies"` -- on cadre sur les bornes de recherche, elargies de
      `marge`. La figure deborde un peu du domaine enrichi, ce qui laisse voir
      ou l'algorithme s'est arrete de chercher.

    Aucune n'est meilleure ; ce sont deux lectures. Ce qui n'allait pas, c'est
    que le choix soit recopie dans quatre fonctions de trace par etude, sous
    forme d'expressions litterales.
    """
    if mode not in CADRAGES:
        raise ValueError("cadre_figures=%r inconnu (attendu : %s)"
                         % (mode, ", ".join(CADRAGES)))
    if mode == "grille":
        u1_min, u1_max, u2_min, u2_max = bornes_grille
        return u1_min, u1_max, u2_min, u2_max
    if len(eff_min) < 2:
        raise ValueError("cadre_figures='bornes_elargies' suppose au moins "
                         "deux variables ; l'etude en declare %d" % len(eff_min))
    return (eff_min[0] - marge, eff_max[0] + marge,
            eff_min[1] - marge, eff_max[1] + marge)


# --------------------------------------------------------------------------- #
# LE DECOR : ce qui ne change pas d'une figure a l'autre                       #
# --------------------------------------------------------------------------- #
import os

import numpy as np


class Decor:
    """Le cadre, la resolution, les noms, le dossier de sortie.

    Ces sept valeurs etaient des variables libres capturees par chaque
    fonction de trace -- vingt a vingt-cinq par fonction, recopiees dans les
    deux etudes. Les rassembler ne rend pas les figures plus belles ; cela
    rend possible de les sortir du script.

    ZERO appel solveur : le decor ne sait pas ce qu'est un etat limite.
    """

    def __init__(self, cadre, n_grid, params_names, modele, dossier,
                 horodatage, tracer=_ecrire):
        self.x0, self.x1, self.y0, self.y1 = cadre
        self.n_grid = n_grid
        self.params_names = params_names
        self.n_var = len(params_names)
        self.modele = modele
        self.dossier = dossier
        self.horodatage = horodatage
        self.tracer = tracer

    # ------------------------------------------------------------------ #
    def maillage(self):
        """Le quadrillage de trace, dans le cadre des figures."""
        return np.meshgrid(np.linspace(self.x0, self.x1, self.n_grid),
                           np.linspace(self.y0, self.y1, self.n_grid))

    def grille_de_coupe(self, coupe):
        """Les points du quadrillage, en coordonnees COMPLETES.

        Une coupe fixe toutes les variables sauf deux ; les points rendus ont
        donc `n_var` composantes, dont `n_var - 2` constantes. C'est ce que le
        metamodele attend -- lui ne sait pas qu'on regarde un plan.
        """
        idx_x, idx_y, fixes = coupe
        UX, UY = self.maillage()
        points = np.zeros((self.n_grid * self.n_grid, self.n_var))
        points[:, idx_x] = UX.ravel()
        points[:, idx_y] = UY.ravel()
        for idx, valeur in fixes.items():
            points[:, idx] = valeur
        return points

    def etiquettes(self, coupe):
        """`(abscisse, ordonnee, variables figees)` -- une figure sans ces
        trois-la ne dit pas ce qu'elle montre."""
        idx_x, idx_y, fixes = coupe
        figees = ("  " + "  ".join("%s=%.1f" % (self.params_names[k], v)
                                   for k, v in fixes.items())) if fixes else ""
        return ("u_%s" % self.params_names[idx_x],
                "u_%s" % self.params_names[idx_y], figees)

    def cadrer(self, ax):
        """Les memes bornes sur toutes les figures d'un run : sans cela, deux
        planches cote a cote ne se comparent pas."""
        ax.set_xlim(self.x0, self.x1)
        ax.set_ylim(self.y0, self.y1)

    def enregistrer(self, fig, nom, dpi=150):
        chemin = os.path.join(self.dossier, nom)
        fig.savefig(chemin, dpi=dpi, bbox_inches="tight")
        return chemin


# --------------------------------------------------------------------------- #
# LA PLANCHE D'ENRICHISSEMENT                                                  #
# --------------------------------------------------------------------------- #
def planche_EFF(decor, coupe, xt, xt_eff, Z_eff, Z_sigma, Z_g,
                fond_hf=None, sous_titre=""):
    """Trois vues cote a cote du meme etat : le critere EFF, l'ecart-type du
    metamodele, et l'etat limite qu'il croit voir.

    Les trois ensemble racontent l'enrichissement : EFF est grand la ou
    sigma est grand ET ou `g` est proche de zero. Une seule des trois ne
    permettrait pas de distinguer « le metamodele ne sait pas » de « il n'y a
    rien a savoir ici ».

    TOUT est recu deja calcule. Cette fonction obtenait autrefois le fond
    haute fidelite elle-meme, ce qui lui faisait declencher jusqu'a 225
    appels au solveur -- 29 heures sur le Moulin Blanc -- sous un nom qui dit
    « imprime ». C'est ce qui a fait croire que la grille arrivait EN DERNIER
    dans le run, alors qu'elle precede l'enrichissement d'une ligne.

    ZERO appel solveur.
    """
    import matplotlib.pyplot as plt

    idx_x, idx_y, _fixes = coupe
    n_ajoutes = len(xt_eff)
    UX, UY = decor.maillage()
    Z_vrai, UX_hf, UY_hf = fond_hf if fond_hf is not None else (None, None, None)
    xlabel, ylabel, _ = decor.etiquettes(coupe)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    # `xt` peut etre None -- en HF pur il n'y a pas de plan a montrer. Le
    # decor le prevoyait, le TITRE non : `len(None)` levait `TypeError`.
    fig.suptitle("%s - N=%d pts DOE  (%d ajoutes par EFF)%s"
                 % (decor.modele, 0 if xt is None else len(xt), n_ajoutes,
                    sous_titre), fontsize=10)

    def _decorer(ax):
        if Z_g is not None:
            ax.contour(UX, UY, Z_g, levels=[0], colors="cyan", linewidths=2,
                       linestyles="--")
        if Z_vrai is not None:
            ax.contour(UX_hf, UY_hf, Z_vrai, levels=[0], colors="red",
                       linewidths=2)
        if xt is not None:
            ax.scatter(xt[:, idx_x], xt[:, idx_y], c="white", s=40, zorder=5,
                       edgecolors="black", linewidths=0.8, label="DOE")
        if n_ajoutes > 0:
            ajoutes = np.array(xt_eff)
            ax.scatter(ajoutes[:, idx_x], ajoutes[:, idx_y], c="red", s=80,
                       zorder=6, marker="^", label="EFF (%d pts)" % n_ajoutes)
            # Le NUMERO de chaque point d'enrichissement : l'ordre dit ou
            # l'algorithme a cherche, et c'est la moitie de l'information.
            for i, pt in enumerate(ajoutes):
                ax.annotate(str(i + 1), (pt[idx_x], pt[idx_y]),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color="red", zorder=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        decor.cadrer(ax)
        ax.legend(loc="best", fontsize=9)

    cf1 = ax1.contourf(UX, UY, Z_eff, levels=20, cmap="viridis", alpha=0.85)
    plt.colorbar(cf1, ax=ax1, label="EFF")
    ax1.set_title("Critere EFF")
    _decorer(ax1)

    cf2 = ax2.contourf(UX, UY, Z_sigma, levels=20, cmap="plasma", alpha=0.85)
    plt.colorbar(cf2, ax=ax2, label="sigma (ecart-type surrogate)")
    ax2.set_title("Ecart-type surrogate (sigma)")
    _decorer(ax2)

    if Z_g is not None:
        cf3 = ax3.contourf(UX, UY, Z_g, levels=20, cmap="RdYlGn", alpha=0.6)
        plt.colorbar(cf3, ax=ax3, label="g surrogate")
        ax3.contour(UX, UY, Z_g, levels=[0], colors="blue", linewidths=2)
    ax3.set_title("g surrogate - etat limite")
    _decorer(ax3)

    plt.tight_layout()
    nom = "EFF_%dpoints_%s.png" % (n_ajoutes, decor.horodatage)
    decor.enregistrer(fig, nom)
    plt.close(fig)
    decor.tracer("  [EFF visu] -> %s" % nom)
    return nom


# --------------------------------------------------------------------------- #
# LA PLANCHE GLOBALE : l'enrichissement etape par etape                        #
# --------------------------------------------------------------------------- #
def planche_globale(decor, coupe, etapes, fond_hf=None, sous_titre=""):
    """Une ligne par etape de l'enrichissement, trois vues par ligne.

    C'est la figure qui montre le RAISONNEMENT de l'algorithme, et non son
    resultat : on y voit le critere EFF designer un point, l'ecart-type
    s'effondrer autour de lui a l'etape suivante, et la frontiere du
    metamodele se rapprocher de la vraie. Un enrichissement qui tourne en
    rond s'y lit d'un coup d'oeil, la ou les chiffres de convergence ne
    disent que « ca ne bouge plus ».

    `etapes` est une liste de dictionnaires DEJA CALCULES, portant `n_pts`,
    `xt`, `xt_eff`, `Z_eff`, `Z_sigma` et `Z_g`. Le reajustement du
    metamodele a chaque etape appartient a l'etude : c'est du calcul.

    ZERO appel solveur.
    """
    import matplotlib.pyplot as plt

    idx_x, idx_y, _fixes = coupe
    UX, UY = decor.maillage()
    Z_vrai, UX_hf, UY_hf = fond_hf if fond_hf is not None else (None, None, None)
    xlabel, ylabel, _ = decor.etiquettes(coupe)

    fig, axes = plt.subplots(len(etapes), 3, figsize=(21, 6 * len(etapes)))
    if len(etapes) == 1:
        axes = axes.reshape(1, 3)

    for ligne, etape in zip(axes, etapes):
        ax1, ax2, ax3 = ligne
        Z_g = etape["Z_g"]
        xt_k, xt_eff_k = etape["xt"], etape["xt_eff"]

        def _decorer(ax, _Z_g=Z_g, _xt=xt_k, _eff=xt_eff_k):
            ax.contour(UX, UY, _Z_g, levels=[0], colors="cyan", linewidths=2,
                       linestyles="--")
            if Z_vrai is not None:
                ax.contour(UX_hf, UY_hf, Z_vrai, levels=[0], colors="red",
                           linewidths=2)
            ax.scatter(_xt[:, idx_x], _xt[:, idx_y], c="white", s=40, zorder=5,
                       edgecolors="black", linewidths=0.8)
            if len(_eff) > 0:
                ajoutes = np.array(_eff)
                ax.scatter(ajoutes[:, idx_x], ajoutes[:, idx_y], c="red", s=80,
                           zorder=6, marker="^")
                for i, pt in enumerate(ajoutes):
                    ax.annotate(str(i + 1), (pt[idx_x], pt[idx_y]),
                                textcoords="offset points", xytext=(0, 8),
                                ha="center", fontsize=8, color="red", zorder=7)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            decor.cadrer(ax)

        cf1 = ax1.contourf(UX, UY, etape["Z_eff"], levels=20, cmap="viridis",
                           alpha=0.85)
        plt.colorbar(cf1, ax=ax1, label="EFF")
        ax1.set_title("EFF  N=%d  (%d pts EFF)" % (etape["n_pts"], len(xt_eff_k)))
        _decorer(ax1)

        cf2 = ax2.contourf(UX, UY, etape["Z_sigma"], levels=20, cmap="plasma",
                           alpha=0.85)
        plt.colorbar(cf2, ax=ax2, label="sigma")
        ax2.set_title("sigma  N=%d" % etape["n_pts"])
        _decorer(ax2)

        cf3 = ax3.contourf(UX, UY, Z_g, levels=20, cmap="RdYlGn", alpha=0.6)
        plt.colorbar(cf3, ax=ax3, label="g surrogate")
        ax3.contour(UX, UY, Z_g, levels=[0], colors="blue", linewidths=2)
        ax3.set_title("g surrogate  N=%d" % etape["n_pts"])
        _decorer(ax3)

    fig.suptitle("Evolution EFF - %s - %s" % (decor.modele, sous_titre),
                 fontsize=14, y=1.0)
    plt.tight_layout()
    nom = "globalplanche_EFF_%s.png" % decor.horodatage
    decor.enregistrer(fig, nom, dpi=100)
    plt.close(fig)
    decor.tracer("  [GLOBAL PLANCHE] -> %s" % nom)
    return nom
