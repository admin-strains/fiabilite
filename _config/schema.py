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
#: doit rester aligne sur `solver/fabrique.py:IMPLEMENTATIONS` -- verifie par test
SOLVEURS = ("digital_structure", "analytique")

#: Solveur lineaire du point interieur : nom -> valeur de `IPARM0[21]`.
#: Les valeurs viennent du commentaire des `InitSolver.py` :
#: « PT INT (1 = MKL PARDISO, 3 = MUMPS, 4 = CuDss) ».
#:
#: MKL Pardiso (valeur 1) et MyPardiso ne sont PAS exposes : Pardiso est
#: deprecie (Agnes, 26/08/2026). Les blocs `MKLPardiso_params` et
#: `MyPardiso_params` des `InitSolver.py` restent transmis au solveur, mais
#: aucune valeur d'`IPARM0[21]` proposee ici ne les selectionne.
SOLVEURS_LINEAIRES = {"mumps": 3, "cudss": 4}

#: L'indice IPARM0 qui porte ce choix pour l'approche cinematique. Le pendant
#: statique est `IPARM0[11]`, laisse tel quel : l'analyse a la rupture est
#: cinematique, et le solveur statique n'intervient pas dans la chaine.
IPARM0_SOLVEUR_LINEAIRE_CINEMATIQUE = 21

#: Champs qui rendent un cache de plan d'experiences INUTILISABLE. Voir
#: `Configuration.signature_solveur` et `_cache/doe.py:load_doe_cache`.
#:
#: Deux familles, et il faut les deux :
#:
#: * ce dont depend la VALEUR de `g` en un point -- modele, solveur, solveur
#:   lineaire, tailles de maille. Un point calcule autrement est faux ici.
#: * ce dont depend le CHOIX des points -- les bornes du domaine. Le plan est
#:   tire par `ot.Uniform(eff_bounds_min[i], eff_bounds_max[i])` : les points
#:   restent des evaluations correctes, mais ils ne couvrent plus le domaine
#:   demande. Ajoute le 26/08/2026 en bornant le domaine du Moulin Blanc, ou
#:   le cache complet de +/- 7,5 aurait ete relu tel quel a +/- 6,0.
#:
#: N'y figurent PAS les parametres qui font autre chose des memes points --
#: metamodele, enrichissement, FORM : ceux-la n'invalident rien.
CHAMPS_QUI_INVALIDENT_LE_CACHE = (
    "modelname", "storage", "solveur", "solveur_lineaire",
    "global_size", "geo_min_approx", "max_size",
    "eff_bound_min", "eff_bound_max",
)

#: Champs que PLUS AUCUN code vivant ne lit, et pourquoi. Mesure du 26/08/2026
#: sur les deux scripts AC : chaque nom n'y apparait que dans le bloc de
#: liaison, jamais en lecture ensuite (`historique/` mis a part).
#:
#: Ils sont conserves plutot que supprimes, parce que leur disparition est une
#: question a poser aux auteurs et non une decision d'outillage : AC et AC2
#: portaient une boucle de montee en degre (`while ... max_degree + 1 <=
#: max_of_maxdegree`) que AC3 a abandonnee. Etait-ce voulu ?
#:
#: En attendant, `valider()` refuse qu'on les regle a autre chose que leur
#: defaut. Un parametre sans effet qui accepte une valeur est un piege : il
#: laisse croire qu'on a change quelque chose.
SANS_EFFET = {
    "reduc_PLS": "composantes PLS du GEK -- le chemin GEK n'est plus cable dans AC3",
    "do_analytic_grad": "gradients analytiques du GEK -- meme raison",
    "max_of_maxdegree": "plafond de la montee en degre PCE, boucle presente dans "
                        "AC et AC2, absente de AC3",
    "seuil_pce": "seuil de validation de l'erreur PCE, lu par AC et AC2 seulement",
}


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

    # -------------------------------------------------------------- solveur
    #: quelle implementation evalue l'etat limite, cf. `solver/fabrique.py`.
    #: "digital_structure" = un SOCP par point, licence requise.
    #: "analytique" = la meme chaine sur un etat limite ferme, en secondes.
    solveur: str = "digital_structure"

    #: Solveur LINEAIRE du point interieur : "mumps" ou "cudss".
    #: None = ne rien imposer, laisser l'`InitSolver.py` de l'etude decider.
    #: Pardiso n'est pas propose : il est deprecie.
    #:
    #: POURQUOI CE CHAMP EXISTE -- constat du 26/08/2026
    #: Le choix vivait dans `InitSolver.py`, en clair mais sans que rien ne
    #: le remonte. Les deux etudes du depot avaient DIVERGE sans le dire :
    #:
    #:     pure_flexion/InitSolver.py    IPARM0[21] = 3   MUMPS
    #:     Moulinblanc/InitSolver.py     IPARM0[21] = 4   CuDss
    #:
    #: Or c'est exactement la ou se separent les deux reproductibilites
    #: mesurees : la flexion pure rejoue un point a 2,9e-11 pres, le Moulin
    #: Blanc a 7,7e-06 pres. Cela ne prouve rien -- les deux modeles n'ont ni
    #: la meme taille ni le meme conditionnement -- mais tant que le backend
    #: n'est pas un parametre visible, l'hypothese n'est meme pas testable.
    #:
    #: Le champ est de categorie ETUDE : il change les nombres.
    solveur_lineaire: Optional[str] = None

    #: Bornes du domaine de recherche, en ecarts-types de l'espace standard U.
    #: S'appliquent a TOUTES les variables. Elles gouvernent le plan
    #: d'experiences (tirage LHS), la recherche EFF, les points de depart FORM
    #: et le filtre `do_FORM_filter` -- donc les points qu'on DEMANDE AU
    #: SOLVEUR. Categorie ETUDE.
    #:
    #: POURQUOI CE CHAMP EXISTE -- crash du 26/08/2026
    #: Elles etaient codees en dur a +/- 7,5 dans les deux scripts AC (oubli
    #: de la phase 4b). Or l'espace standard n'a pas de sens physique : ce
    #: sont les LOIS qui le lui donnent, et une loi non bornee ne borne rien.
    #:
    #: Sur le Moulin Blanc, `fy ~ Normal(235 ; 30,15)`, non bornee en bas :
    #:
    #:     u = -7,5   ->  fy =   8,88 MPa   (plus faible que le beton)
    #:     u = -7,79  ->  fy =   0
    #:     u = -8,5   ->  fy = -21,27 MPa   (negative)
    #:
    #: Le point u = [+7,5 ; -7,5] donne fy1/fy2 = 461/8,9, soit un rapport de
    #: 52 entre les deux groupes d'aciers. `docs/mesh/` documente qu'un
    #: rapport >= ~5 mal-conditionne les cones SOCP. A 52, Digital Structure a
    #: TERMINE LE PROCESSUS -- sans exception Python, sans trace, en pleine
    #: iteration IPM, apres deux heures de calcul.
    #:
    #: Le defaut reste +/- 7,5 : c'est la valeur sous laquelle toutes les
    #: etudes ont tourne, et la flexion pure ne pose pas de probleme
    #: (`fy ~ Normal(550 ; 30,15)` vaut encore 324 MPa a u = -7,5). Une etude
    #: dont les lois ne sont pas bornees doit choisir sa valeur.
    eff_bound_min: float = -7.5
    eff_bound_max: float = 7.5

    #: Rejeter les points que le solveur declare non converges, au lieu de les
    #: laisser entrer dans le plan d'experiences.
    #:
    #: FAUX PAR DEFAUT, ET C'EST DELIBERE. Decision d'Agnes, 26/08/2026 : les
    #: criteres de convergence rendus par Digital Structure ne sont pas encore
    #: fiables -- un point sain peut etre marque non converge, et sans doute
    #: l'inverse. Les exclure aujourd'hui reviendrait a jeter des evaluations
    #: correctes, chacune coutant un SOCP.
    #:
    #: Ce que fait la chaine en attendant : elle SIGNALE chaque point suspect
    #: dans le journal, avec son statut et son alpha, et le garde. Rien n'est
    #: donc perdu et rien n'est masque -- ce qui n'etait pas le cas avant la
    #: phase 5, ou `Primal_bound` etait lu sans que `converged` ni
    #: `solver_status` n'apparaissent nulle part (zero occurrence dans les
    #: deux scripts).
    #:
    #: Le jour ou ces criteres seront fiables, basculer ce champ a `true` dans
    #: le fichier d'etude suffit : aucune modification de code.
    exclure_points_non_converges: bool = False

    # ------------------------------------------------------------- maillage
    global_size: float = 0.05        #: global_physical_size
    geo_min_approx: int = 4          #: geometric_approximation_min

    #: `max_size` du mailleur. None = suit `global_size`.
    #:
    #: Il valait 0.05 EN DUR dans les options recopiees de `run_one_SOL`, et
    #: plafonnait donc la taille des elements quelle que soit la valeur de
    #: `global_size`. Mesure du 26/08/2026 sur le Moulin Blanc : passer
    #: `global_size` de 0,05 a 0,15 ne retirait que 2,8 % des tetraedres
    #: (13 804 -> 13 418) et ne faisait RIEN gagner sur la duree.
    #:
    #: Le regler explicitement permet de dissocier la taille cible de la
    #: borne haute -- utile quand on veut un maillage grossier partout sauf
    #: la ou la geometrie l'impose.
    max_size: Optional[float] = None

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
        if self.solveur not in SOLVEURS:
            problemes.append("solveur=%r inconnu (attendu : %s)"
                             % (self.solveur, ", ".join(SOLVEURS)))
        if self.solveur_lineaire is not None \
                and self.solveur_lineaire not in SOLVEURS_LINEAIRES:
            problemes.append(
                "solveur_lineaire=%r inconnu (attendu : %s, ou omis pour "
                "laisser InitSolver.py decider)"
                % (self.solveur_lineaire, ", ".join(sorted(SOLVEURS_LINEAIRES))))
        if self.max_degree > self.max_of_maxdegree:
            problemes.append("max_degree=%d depasse max_of_maxdegree=%d"
                             % (self.max_degree, self.max_of_maxdegree))
        if self.n0 < 1:
            problemes.append("n0=%d : il faut au moins un point" % self.n0)
        if not 0.0 < self.q <= 1.0:
            problemes.append("q=%r doit etre dans ]0, 1]" % (self.q,))
        if self.n_workers_DOE < 1:
            problemes.append("n_workers_DOE=%d" % self.n_workers_DOE)
        if self.eff_bound_min >= self.eff_bound_max:
            problemes.append("eff_bound_min=%r >= eff_bound_max=%r : domaine vide"
                             % (self.eff_bound_min, self.eff_bound_max))
        if self.u1_min >= self.u1_max or self.u2_min >= self.u2_max:
            problemes.append("bornes de trace vides : %s" % (self.bornes_u,))
        for nom in ("tol_EFF", "tol_BB", "tol_BS", "tol_FORM", "cov_IS", "tol_all_modes"):
            v = getattr(self, nom)
            if v <= 0:
                problemes.append("%s=%r doit etre strictement positif" % (nom, v))
        if self.n_batch_EFF > 1 and self.eps_taylor > 0:
            problemes.append("n_batch_EFF>1 et eps_taylor>0 : combinaison non prevue")
        for nom, pourquoi in SANS_EFFET.items():
            defaut = _DEFAUTS[nom]
            if getattr(self, nom) != defaut:
                problemes.append(
                    "%s=%r : ce parametre n'est lu par AUCUN code vivant (%s). "
                    "Le regler ne changerait rien -- laisser %r, ou recabler le "
                    "chemin qui le lisait et retirer le nom de SANS_EFFET."
                    % (nom, getattr(self, nom), pourquoi, defaut))
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

    def signature_solveur(self) -> dict:
        """Ce qui rend un cache de plan d'experiences inutilisable.

        Jusqu'au 26/08/2026 le cache n'etait valide que sur `n0` : basculer le
        solveur lineaire de CuDss a MUMPS aurait rendu des points issus de
        l'autre backend, sans une ligne de journal pour le dire. Et borner le
        domaine aurait relu un plan tire sur l'ancien.

        Volontairement RESTREINT : le metamodele et l'enrichissement font
        autre chose des memes points, ils n'invalident rien. Voir
        `CHAMPS_QUI_INVALIDENT_LE_CACHE`.
        """
        return {nom: getattr(self, nom) for nom in CHAMPS_QUI_INVALIDENT_LE_CACHE}


# ------------------------------------------------------------------------- #
# Chargement                                                                #
# ------------------------------------------------------------------------- #
#: valeurs par defaut du schema, pour comparer sans instancier
_DEFAUTS = {f.name: f.default for f in fields(Configuration)}


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
    # Provenance : ce n'est pas un parametre de calcul, donc pas un champ (il
    # n'a rien a faire dans `en_dict()` ni dans une comparaison de deux
    # configurations). L'objet etant gele, il faut passer par object.
    object.__setattr__(cfg, "_origine", os.path.abspath(chemin))
    return cfg


# ------------------------------------------------------------------------- #
# Tracabilite d'un run                                                      #
# ------------------------------------------------------------------------- #
#: champs dont la valeur decide de ce qui est calcule, par opposition a ceux
#: qui ne decident que de ce qui est TRACE. La mesure du 25/08/2026 a montre
#: qu'une comparaison de deux runs ne vaut rien si leur configuration differe :
#: le resume met donc ces champs-la en premier.
# ------------------------------------------------------------------------- #
# A QUI APPARTIENT CHAQUE PARAMETRE                                          #
# ------------------------------------------------------------------------- #
#
# La question posee par Agnes le 26/08/2026 : « quelles options sont liees a
# un run utilisateur, et lesquelles sont internes au code ? » Elle ne l'etait
# pas. La phase 4 avait rassemble cinquante-trois affectations dans un fichier
# plat, sans dire laquelle definit l'ETUDE et laquelle decrit seulement la
# SESSION qui la calcule.
#
# L'incident qui l'a revele : `restart_enrich_only = true` figurait dans
# `studies/moulin_blanc.toml` parce que le script le portait. Or c'est un mode
# de session -- « reprends l'enrichissement la ou tu l'avais laisse » -- fige
# dans ce qui ressemble a une definition d'etude. Consequence : l'etude etait
# injouable sur tout poste depourvu du `restart_state.json`.
#
#   ETUDE      definit le probleme. Change le RESULTAT. Se transmet avec
#              l'etude, se cite dans une note de calcul.
#   SESSION    decrit comment CE run s'execute ici : parallelisme, caches,
#              reprise, archivage. Ne change pas le resultat, depend du poste
#              et du moment. N'a rien a faire dans la definition d'une etude,
#              et se surcharge par `--session`.
#   SORTIE     ce qui est TRACE. Ne change jamais le resultat -- mais peut
#              couter tres cher, cf. `COUTE_DES_APPELS_SOLVEUR`.
#   SANS_EFFET aucun code vivant ne les lit, cf. la constante du meme nom.
#
CATEGORIES = {
    # --- ETUDE : le probleme pose -------------------------------------------
    "modelname": "etude", "storage": "etude", "solveur": "etude",
    "solveur_lineaire": "etude",
    "eff_bound_min": "etude", "eff_bound_max": "etude",
    "modele": "etude", "n0": "etude", "max_degree": "etude", "q": "etude",
    "do_EFF": "etude", "epsilon_factor": "etude", "tol_EFF": "etude",
    "tol_BB": "etude", "tol_BS": "etude", "EFF_criteria": "etude",
    "n_NLopt_EFF": "etude", "n_max_EFF_points": "etude",
    "n_batch_EFF": "etude", "eps_taylor": "etude",
    "n_max_FORM": "etude", "tol_FORM": "etude", "tol_all_modes": "etude",
    "tol_warmstart": "etude", "do_multistart": "etude",
    "do_warmstart": "etude", "start_from_LHS": "etude", "n_sp": "etude",
    "do_FORM_filter": "etude",
    "do_IS": "etude", "n_IS": "etude", "cov_IS": "etude",
    "global_size": "etude", "geo_min_approx": "etude",
    "max_size": "etude",
    # decide quels points entrent au plan d'experiences : change le resultat
    "exclure_points_non_converges": "etude",

    # --- SESSION : comment ce run tourne ICI ---------------------------------
    "n_workers_DOE": "session",        # depend du nombre de coeurs du poste
    "config_is_identical": "session",  # autorise la relecture des caches
    "restart_enrich_only": "session",  # reprend un enrichissement en cours
    "save_history": "session",         # archivage, 8,8 Mo a 424 Mo par point
    "dossier_sortie": "session",       # ou vont les figures

    # --- SORTIE : ce qui est trace -------------------------------------------
    "u1_min": "sortie", "u1_max": "sortie", "u2_min": "sortie",
    "u2_max": "sortie", "n_grid": "sortie", "n_grid_hf": "sortie",
    "print_HF": "sortie", "print_fullHF": "sortie", "print_DOE": "sortie",
    "print_3D": "sortie", "print_ana": "sortie", "print_Pf": "sortie",
    "print_grad_sp": "sortie", "print_EFF_progres": "sortie",
    "print_gepck_calls": "sortie", "do_custom_hf": "sortie",
    "hf_2d_grid_fixed": "sortie", "hf_3d_grid_fixed": "sortie",

    # --- SANS EFFET ----------------------------------------------------------
    "reduc_PLS": "sans_effet", "do_analytic_grad": "sans_effet",
    "max_of_maxdegree": "sans_effet", "seuil_pce": "sans_effet",
}

#: Parametres de SORTIE qui declenchent des appels au solveur. Ils ne changent
#: pas le resultat, mais sur le Moulin Blanc un appel coute 466 s : une grille
#: 15x15 represente deux jours de calcul pour une figure.
COUTE_DES_APPELS_SOLVEUR = {
    "print_HF": "grille haute fidelite : n_grid_hf^2 appels",
    "print_fullHF": "grille complete : n_grid_hf^n_var appels",
    "n_grid_hf": "cote de la grille HF -- le cout est son carre",
    "do_custom_hf": "grille de points fournie par fichier",
    "print_Pf": "3 FORM+IS supplementaires a chaque iteration EFF",
}


def resume(cfg: "Configuration") -> str:
    """Configuration effective, en clair, pour l'en-tete du journal d'un run.

    Un journal qui ne porte pas sa configuration ne peut pas etre compare a un
    autre : c'est ce qui manquait aux runs de l'auteur, ou les cinquante
    parametres n'apparaissaient nulle part dans la sortie. Les valeurs
    derivees (`eff_actif`, `is_actif`) figurent aussi, parce qu'elles peuvent
    contredire l'intention de l'utilisateur -- en haute fidelite, un
    `do_EFF = True` n'enrichit rien.
    """
    # Le solveur lineaire est sorti du bloc et mis en tete : c'est le
    # parametre qu'on ne voyait PAS jusqu'au 26/08/2026, alors qu'il vaut
    # MUMPS dans une etude du depot et CuDss dans l'autre. Un
    # `solveur_lineaire=None` dans un bloc de cinquante valeurs ne se
    # remarque pas -- ici la ligne dit aussi ce que None veut dire.
    if cfg.solveur_lineaire is None:
        _lin = "(non impose -- valeur de IPARM0[21] dans l'InitSolver.py de l'etude)"
    else:
        _lin = "%s (IPARM0[21] = %d, impose par le fichier d'etude)" % (
            cfg.solveur_lineaire, SOLVEURS_LINEAIRES[cfg.solveur_lineaire])

    lignes = ["-" * 70,
              "CONFIGURATION : %s" % (getattr(cfg, "_origine", None) or "(defauts)"),
              "-" * 70,
              "  modele   %s" % cfg.chemin_ds,
              "  solveur  %s" % cfg.solveur,
              "  lineaire %s" % _lin]

    def _bloc(titre, noms):
        if not noms:
            return
        lignes.append("  %s" % titre)
        ligne = "   "
        for nom in noms:
            morceau = "%s=%r" % (nom, getattr(cfg, nom))
            if len(ligne) + len(morceau) > 68:
                lignes.append(ligne)
                ligne = "   "
            ligne += " " + morceau
        lignes.append(ligne)

    # L'ETUDE d'abord : c'est elle qui definit le resultat, et c'est elle
    # qu'on recopie dans une note de calcul.
    _bloc("ETUDE -- definit le resultat",
          [n for n in cfg.en_dict() if CATEGORIES.get(n) == "etude"
           and n not in ("modelname", "storage")])
    # La SESSION ensuite, separee : elle depend du poste et du moment, jamais
    # du probleme pose. Les confondre est ce qui avait rendu l'etude du Moulin
    # Blanc injouable ailleurs que sur le poste de son auteur.
    _bloc("SESSION -- ce run, sur ce poste",
          [n for n in cfg.en_dict() if CATEGORIES.get(n) == "session"])

    # Ce qui ne change pas le resultat mais coute des appels solveur : a
    # 466 s l'appel sur le Moulin Blanc, une grille 15x15 fait deux jours.
    couteux = [n for n in COUTE_DES_APPELS_SOLVEUR
               if getattr(cfg, n, None) not in (False, None)]
    if couteux:
        lignes.append("  SORTIES QUI COUTENT DES APPELS SOLVEUR")
        for nom in couteux:
            lignes.append("    %-16s %-8r %s"
                          % (nom, getattr(cfg, nom), COUTE_DES_APPELS_SOLVEUR[nom]))

    if cfg.do_EFF != cfg.eff_actif or cfg.do_IS != cfg.is_actif:
        lignes.append("  CORRIGE  modele=%s : eff_actif=%s  is_actif=%s"
                      % (cfg.modele, cfg.eff_actif, cfg.is_actif))
    lignes.append("-" * 70)
    return "\n".join(lignes)


def ecrire_trace(cfg: "Configuration", dossier: str) -> str:
    """Depose la configuration effective en JSON a cote des sorties du run.

    Complement du resume : lisible par `tools/comparer_runs.py`, qui peut
    ainsi refuser de comparer deux runs dont la configuration differe au lieu
    d'imputer l'ecart au code.
    """
    import json
    os.makedirs(dossier, exist_ok=True)
    cible = os.path.join(dossier, "configuration.json")
    donnees = cfg.en_dict()
    donnees["_origine"] = getattr(cfg, "_origine", None)
    donnees["_derive"] = {"eff_actif": cfg.eff_actif, "is_actif": cfg.is_actif,
                          "chemin_ds": cfg.chemin_ds}
    with open(cible, "w", encoding="utf-8") as fh:
        json.dump(donnees, fh, indent=1, sort_keys=True, ensure_ascii=False)
    return cible
