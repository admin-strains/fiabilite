"""
Configuration d'une etude de fiabilite.

PHASE 4 du plan de nettoyage. Remplace les ~50 variables globales du bloc
`OPTIONS UTILISATEUR` des scripts AC par un objet valide, charge depuis un
fichier `.toml`.

CE QUE L'INVENTAIRE A MONTRE
----------------------------
Les deux scripts AC portaient 79 et 83 affectations de configuration.
Comparaison faite :

    77 communes, dont **65 de valeur identique**
    12 seulement diffferent vraiment
    14 des 65 ne sont pas de la configuration mais des accumulateurs d'etat
    14 autres sont des valeurs CALCULEES a partir des precedentes

Une etude ne fait donc pas soixante lignes de configuration : elle en fait
douze. Le reste appartient aux defauts, et les valeurs calculees deviennent
des proprietes -- elles ne peuvent plus se contredire.

DEUX SUBTILITES QUE LE SCHEMA REND EXPLICITES
---------------------------------------------
1. Les sept drapeaux `do_KRG`, `do_GEK`, `do_HF`, `do_PCKRG`, `do_GEPCK`,
   `do_PCK`, `do_old_GEPCK` encodaient UN SEUL choix, `modele`, en sept
   variables ecrites a la main -- quatorze lignes sur les deux scripts, qui
   pouvaient se contredire. Ce sont desormais des proprietes derivees.

2. `do_EFF` et `do_IS` etaient affectes DEUX FOIS : une premiere en haut du
   fichier comme choix de l'utilisateur, puis reecrits cinquante lignes plus
   loin par `do_IS = do_IS and modele != 'HF'`. Le choix etait donc
   silencieusement corrige. Les champs gardent l'intention de l'utilisateur ;
   `eff_actif` et `is_actif` donnent l'effet reel, et disent pourquoi.
"""

import os
from dataclasses import dataclass, field, fields
from typing import Optional, Tuple

MODELES = ("GEPCK", "PCK", "PCKRG", "KRG", "GEK", "HF", "old_GEPCK")
CRITERES_EFF = ("BB", "BS", "both", "at_least_one")


@dataclass(frozen=True)
class Configuration:
    """Parametres d'une etude. Immuable : un run ne modifie pas sa configuration."""

    # ----------------------------------------------------------------- etude
    modelname: str
    storage: str = r"C:\workspace\storage\admin\SF"
    #: racine des sorties (figures, journaux). None -> `<depot>/<etude>/output`
    dossier_sortie: Optional[str] = None

    # ------------------------------------------------------------ metamodele
    modele: str = "GEPCK"
    n0: int = 5                      #: taille du plan d'experiences initial
    max_degree: int = 2              #: degre max de la base PCE candidate
    max_of_maxdegree: int = 2        #: plafond autorise sur max_degree
    q: float = 0.75                  #: tri hyperbolique de la base candidate
    seuil_pce: float = 0.90          #: seuil de validation de l'erreur PCE
    reduc_PLS: int = 0               #: composantes PLS (GEK), 0 = desactive
    do_analytic_grad: bool = False

    # ------------------------------------------------------ enrichissement EFF
    do_EFF: bool = True              #: intention de l'utilisateur, cf. `eff_actif`
    epsilon_factor: float = 2.0      #: eps = epsilon_factor * sigma
    tol_EFF: float = 1e-3
    tol_BB: float = 0.05             #: |beta_sup - beta_inf| / beta
    tol_BS: float = 0.01             #: |beta - beta_precedent| / beta
    EFF_criteria: str = "BS"
    n_NLopt_EFF: int = 30            #: budget d'evaluations par recherche EFF
    n_max_EFF_points: int = 360      #: plafond de points ajoutes
    n_batch_EFF: int = 1             #: 1 = sequentiel, >1 = Kriging Believer
    eps_taylor: float = 0.0          #: PCK : points virtuels par Taylor ordre 1

    # -------------------------------------------------------------------- FORM
    n_max_FORM: int = 50
    tol_FORM: float = 0.05
    tol_all_modes: float = 0.9       #: distance DBSCAN entre deux modes
    tol_warmstart: float = 0.2
    do_multistart: bool = True
    do_warmstart: bool = False
    start_from_LHS: bool = False
    n_sp: int = 200                  #: taille du LHS de points de depart
    do_FORM_filter: bool = False     #: rejeter les u* hors bornes avant DBSCAN

    # ------------------------------------------------ tirage d'importance
    do_IS: bool = True               #: intention de l'utilisateur, cf. `is_actif`
    n_IS: int = 10_000
    cov_IS: float = 0.05

    # ------------------------------------------------------------- maillage
    global_size: float = 0.05        #: global_physical_size
    geo_min_approx: int = 4          #: geometric_approximation_min

    # ------------------------------------------------------------ execution
    n_workers_DOE: int = 6           #: appels SOCP du DOE en parallele
    config_is_identical: bool = True #: autorise la relecture des caches
    restart_enrich_only: bool = False
    save_history: bool = False       #: copie chaque dsmed (~8,8 Mo par point)

    # ------------------------------------------------------------- graphiques
    u1_min: float = -7.5
    u1_max: float = 7.5
    u2_min: float = -7.5
    u2_max: float = 7.5
    n_grid: int = 300
    n_grid_hf: int = 15
    print_HF: bool = True
    print_fullHF: bool = False
    print_DOE: bool = True
    print_3D: bool = False
    print_ana: bool = False
    print_Pf: bool = False
    print_grad_sp: bool = False
    print_EFF_progres: bool = True
    print_gepck_calls: bool = False
    do_custom_hf: bool = False

    # ------------------------------------------------------- resultats figes
    hf_2d_grid_fixed: Optional[list] = None
    hf_3d_grid_fixed: Optional[list] = None

    # =================================================================== #
    # Valeurs derivees : elles ne peuvent pas contredire les champs        #
    # =================================================================== #
    @property
    def do_GEPCK(self) -> bool:
        return self.modele == "GEPCK"

    @property
    def do_PCK(self) -> bool:
        return self.modele == "PCK"

    @property
    def do_PCKRG(self) -> bool:
        return self.modele == "PCKRG"

    @property
    def do_KRG(self) -> bool:
        return self.modele == "KRG"

    @property
    def do_GEK(self) -> bool:
        return self.modele == "GEK"

    @property
    def do_HF(self) -> bool:
        return self.modele == "HF"

    @property
    def do_old_GEPCK(self) -> bool:
        return self.modele == "old_GEPCK"

    @property
    def eff_actif(self) -> bool:
        """Enrichissement reellement pratique.

        En haute fidelite, chaque evaluation coute un SOCP : enrichir n'a pas
        de sens. Les scripts AC reecrivaient `do_EFF` pour cela, cinquante
        lignes apres l'avoir lu de l'utilisateur.
        """
        return self.do_EFF and not self.do_HF

    @property
    def is_actif(self) -> bool:
        """Tirage d'importance reellement pratique -- meme raison."""
        return self.do_IS and not self.do_HF

    @property
    def chemin_ds(self) -> str:
        return os.path.join(self.storage, self.modelname + ".ds")

    @property
    def bornes_u(self) -> Tuple[float, float, float, float]:
        return (self.u1_min, self.u1_max, self.u2_min, self.u2_max)

    def dossier_png_eff(self, timestamp: str) -> str:
        base = self.dossier_sortie or os.path.join(os.getcwd(), "output")
        return os.path.join(base, "png EFF", "png_EFF_%s" % timestamp)

    # =================================================================== #
    def valider(self) -> None:
        """Refuse une configuration incoherente, avec un message utilisable.

        Les scripts AC ne validaient rien : une faute de frappe sur `modele`
        mettait les sept drapeaux a False et le calcul partait sans
        metamodele, sans un mot.
        """
        problemes = []
        if self.modele not in MODELES:
            problemes.append("modele=%r inconnu (attendu : %s)"
                             % (self.modele, ", ".join(MODELES)))
        if self.EFF_criteria not in CRITERES_EFF:
            problemes.append("EFF_criteria=%r inconnu (attendu : %s)"
                             % (self.EFF_criteria, ", ".join(CRITERES_EFF)))
        if self.max_degree > self.max_of_maxdegree:
            problemes.append("max_degree=%d depasse max_of_maxdegree=%d"
                             % (self.max_degree, self.max_of_maxdegree))
        if self.n0 < 1:
            problemes.append("n0=%d : il faut au moins un point" % self.n0)
        if not 0.0 < self.q <= 1.0:
            problemes.append("q=%r doit etre dans ]0, 1]" % (self.q,))
        if self.n_workers_DOE < 1:
            problemes.append("n_workers_DOE=%d" % self.n_workers_DOE)
        if self.u1_min >= self.u1_max or self.u2_min >= self.u2_max:
            problemes.append("bornes de trace vides : %s" % (self.bornes_u,))
        for nom in ("tol_EFF", "tol_BB", "tol_BS", "tol_FORM", "cov_IS", "tol_all_modes"):
            v = getattr(self, nom)
            if v <= 0:
                problemes.append("%s=%r doit etre strictement positif" % (nom, v))
        if self.n_batch_EFF > 1 and self.eps_taylor > 0:
            problemes.append("n_batch_EFF>1 et eps_taylor>0 : combinaison non prevue")
        if problemes:
            raise ValueError("Configuration invalide :\n  - " + "\n  - ".join(problemes))

    def remplace(self, **modifications) -> "Configuration":
        """Copie modifiee, validee. L'objet reste immuable."""
        import dataclasses
        neuf = dataclasses.replace(self, **modifications)
        neuf.valider()
        return neuf

    def en_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ------------------------------------------------------------------------- #
# Chargement                                                                #
# ------------------------------------------------------------------------- #
def _lire_toml(chemin):
    try:
        import tomllib                     # Python >= 3.11
    except ImportError:
        import tomli as tomllib            # Python 3.10
    with open(chemin, "rb") as fh:
        return tomllib.load(fh)


def charger(chemin) -> Configuration:
    """Charge une etude depuis un `.toml` et valide le resultat.

    Le fichier ne contient que ce qui s'ecarte des defauts. Une cle inconnue
    est une erreur, pas un silence : c'est le cas d'une faute de frappe qui,
    dans les scripts AC, se traduisait par un parametre ignore.
    """
    donnees = _lire_toml(chemin)
    connus = {f.name for f in fields(Configuration)}
    inconnues = sorted(set(donnees) - connus)
    if inconnues:
        raise ValueError(
            "%s : cle(s) inconnue(s) %s\nAttendu parmi : %s"
            % (os.path.basename(chemin), inconnues, ", ".join(sorted(connus))))
    cfg = Configuration(**donnees)
    cfg.valider()
    return cfg
