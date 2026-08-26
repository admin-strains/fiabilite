r"""
L'implementation Digital Structure du contrat `solver/interface.py`.

C'est le SEUL fichier du depot qui importe Digital Structure pour calculer.
Tout le reste -- plan d'experiences, metamodele, enrichissement, FORM, tirage
d'importance -- n'en depend plus.

CE QUI EST REPRIS VERBATIM, ET CE QUI NE L'EST PAS
---------------------------------------------------
Les dictionnaires d'options de maillage et de solveur sont recopies mot pour
mot depuis `run_one_SOL`, la plus complete des quatre copies. Un test les
compare a ceux que les scripts AC portaient a la revision precedente : ce sont
des reglages de calcul, pas du style, et une virgule qui bouge change le
resultat.

Deux differences volontaires, toutes deux motivees par une mesure :

1. **Une seule taille de maille.** `run_HF` codait en dur
   `global_physical_size = 0.05` et `geometric_approximation_min = "4"` la ou
   `run_one_SOL` lisait la configuration. Les deux alimentent pourtant le meme
   metamodele. Ici, les options viennent d'un seul endroit.

2. **L'etat de sante est rendu.** Les scripts lisaient `Primal_bound` sans
   regarder `converged` ni `solver_status` -- zero occurrence. A
   `global_physical_size = 0.018`, le solveur sort NUMERICAL_ERROR avec
   alpha = 1.5197 au lieu de ~1.3188 : un point faux entrait dans le plan
   d'experiences.

Ce module est le successeur direct de `tools/solve_one.py`, qui en a valide le
principe sur de vrais points (~10 s l'appel, modele de flexion pure).
"""

import json
import os
import re
import shutil
import time

from STRAINS.rupt.APIs.CetCAD_API import *          # noqa: F401,F403
from STRAINS.rupt.APIs import CetLOAD
from STRAINS.rupt.APIs.CetLOAD_API import *         # noqa: F401,F403
import STRAINS.rupt.core.CetMESH as CetMESH
from STRAINS.rupt.core import CetSOLV

from interface import Evaluation


def options_maillage(global_size, geo_min_approx, model_handle, max_size=None):
    """Options passees a `CetMESH.ANISO_MESH`. Recopiees de `run_one_SOL`.

    `global_size`, `geo_min_approx` et `max_size` viennent de la configuration
    de l'etude. Le reste est fige : ce sont les reglages sous lesquels toutes
    les etudes ont tourne, et rien ne les a jamais fait varier.

    CES TAILLES SONT RELATIVES, PAS EN METRES -- lecture du code, 26/08/2026
    -----------------------------------------------------------------------
    `physical_size_type` n'est pas dans ce dictionnaire. Le defaut du mailleur
    n'est PAS « ignorer la taille » : c'est `CmnMESH_PhysicalSizeTypeRelative`
    (`CetMESH_SessionAbstract.cpp:140`). En mode relatif, `RunCadSurf` suffixe
    la valeur d'un « r » avant de la passer a MeshGems (ligne 666) : c'est une
    FRACTION DE LA DIAGONALE de la boite englobante.

    Sur le Moulin Blanc, la boite mesure 96,2 x 14,1 x 12,7 m, soit une
    diagonale de 98,1 m. Donc :

        global_physical_size = 0,05  ->  4,90 m d'element
        max_size             = 0,05  ->  4,90 m
        min_size             = "-1"  ->  jamais transmis (garde `> 0`)

    Des elements de 4,90 m sur un tablier de 14 m de haut : la consigne est
    tres au-dessus de ce que la geometrie permet, elle ne mord sur rien. Le
    maillage est a son PLANCHER GEOMETRIQUE -- 13 804 tetraedres imposes par
    la topologie des faces et la carte de courbure, pas par la taille demandee.

    C'est ce qu'a montre la mesure du 26/08/2026 :

        global_size   geo_min   equivalent   tetraedres   duree
        0,05          4          4,90 m      13 804       454 s
        0,15          20        14,71 m      13 418       458 s
        0,30          35        29,42 m      13 092       455 s

    Aucun gain de temps, et les 5 % de tetraedres en moins viennent de
    `geo_min_approx` (4 -> 35), pas de la taille : les trois consignes sont
    inertes. On ne peut donc PAS faire plus grossier ici.

    RECTIFICATION : le commentaire precedent attribuait ce plateau a un
    `max_size` fige a 0,05 qui aurait plafonne les elements a 5 cm. C'etait
    faux -- il valait 4,90 m et etait aussi inerte que le reste. Seule
    l'observation tenait.

    `max_size` reste un parametre d'etude (il suit `global_size` par defaut,
    ce qui reproduit l'ancien comportement), mais ce n'est pas le levier.
    Pour piloter reellement la taille sur un grand modele, il faudrait passer
    `physical_size_type = "absolute"` et donner des metres -- changement de
    maillage, donc de resultat : a decider, pas a subir.
    """
    return {
        "cadSurfOptions": {"volume_gradation": 1.5, "gradation": 1.5, "anisotropic_ratio": 10},
        "tetraOptions": {"optimisation_level": "standard", "verbose": "10"},
        "global_physical_size": global_size,
        "max_size": global_size if max_size is None else max_size,
        "min_size": "-1",
        "gradation": 1.5,
        "volume_gradation": 1.5,
        "optimisation_level": "standard",
        "anisotropic_ratio": "10",
        "geometric_approximation_min": str(geo_min_approx),
        "geometric_approximation_max": "25",
        "geometric_approximation_on_edge": "false",
        "geometric_approximation_on_face": "true",
        "use_surface_proximity": "false",
        "surface_proximity_ratio": 0,
        "write_debug_files": "true",
        # Les cinq clefs suivantes ne sont lues que par
        # `CetMESH_SessionAnisoRemesh::SetOptions`, donc a partir de
        # l'iteration 1. A l'iteration 0 -- la seule que cette chaine
        # utilise -- c'est `CetMESH_SessionAnisoMesh` qui tourne, et il ne
        # surcharge pas `SetOptions` : ces valeurs tombent dans le vide,
        # sans avertissement. Gardees telles quelles pour rester fidele aux
        # scripts d'origine ; a retirer le jour ou on passe a un remaillage
        # adaptatif, ou elles reprendront leur sens.
        "approach": "kinematic",
        "is_iso": "true",
        "coeff_on_error": 0.01,
        "remesh_type": 1,
        "old_size_factor": 0.0,
        "model_handle": model_handle,
    }


def options_solveur(init_solver, model_handle, regions_sensibilite=None):
    """Options passees a `CetSOLV.SOLV`. Recopiees de `run_one_SOL`.

    `init_solver` est le contenu du `InitSolver.py` de l'etude, execute dans
    un espace de noms DEDIE. Les scripts AC l'executaient dans leurs globales
    (`exec(..., globals())`), ce qui y injectait cinq dictionnaires sans que
    rien ne l'indique -- l'auteur en doutait lui-meme en commentaire :
    « je ne suis pas sure que ca marche comme ca ».
    """
    ns = {}
    exec(init_solver, ns)                            # noqa: S102
    kwargs = {
        "scaling": 1,
        "write_debug_files": "true",
        "static_params": ns["static_params"],
        "cinematic_params": ns["cinematic_params"],
        "MKLPardiso_params": ns["MKLPardiso_params"],
        "MyPardiso_params": ns["MyPardiso_params"],
        "MUMPS_params": ns["MUMPS_params"],
        "FullLorentz": False,
        "LorentzToSdp": False,
        "SdpToLorentz": 0,
        "printIntPointSolutioEvolution": False,
        "trace_sur_point_integration": False,
        "calculate_error": "false",
        "max_nbOfDiv": 0,
        "customized_inc": [1],
        "tetra_discontinuities": False,
        "activated_plasticity": True,
        "welds_throat_limit": True,
        "approach": "kinematic",
    }
    if regions_sensibilite is not None:
        kwargs["sensitivity_analysis"] = "true"
        kwargs["sensitivity_regions"] = json.dumps(regions_sensibilite)
    kwargs["model_handle"] = model_handle
    return kwargs


def patch_params(path, **params):
    """Reecrit dsCad.txt et dsLoad.txt avec de nouvelles valeurs de parametres.

    Recopie verbatim de `AC3_pure_flexion.patch_params`. Ecrit EN PLACE dans le
    modele de l'utilisateur, comme l'original : c'est le mecanisme meme par
    lequel la geometrie est parametree.
    """
    for filename in ('dsCad.txt', 'dsLoad.txt'):
        fpath = os.path.join(path, filename)
        with open(fpath, 'r') as f:
            content = f.read()
        for name, value in params.items():
            content = re.sub(r'^' + name + r'\s*=.*$', f'{name}    = {value:.10f}',
                             content, count=1, flags=re.MULTILINE)
        with open(fpath, 'w') as f:
            f.write(content)


#: fichiers recopies par `archiver_sorties`
FICHIERS_SOCP = ("_0_PL_cin_out.msh", "_0_kine.dsmed", "_0_kine.dslog",
                 "_0_kine.dsmetares", "_0_stat.dsmed")

_catalogues_charges = [False]


def initialiser_catalogues(racine_ds=None):
    """INITCATALOG, une fois pour toutes.

    Les scripts AC l'appelaient dans leur en-tete, avec trois chemins absolus
    vers `C:\\workspace\\front`. C'est une operation du solveur, pas de la
    chaine de fiabilite : elle vit ici, et la racine vient de `launcher`.
    """
    if _catalogues_charges[0]:
        return
    if racine_ds is None:
        import launcher
        racine_ds = launcher.find_ds_root()
    base = os.path.join(racine_ds, "STRAINS", "common", "Catalog")
    lus = []
    for nom in ("CatalogTopo.json", "CatalogDimensions.json", "CatalogBolts.json"):
        with open(os.path.join(base, nom), "r") as fh:
            lus.append(fh.read())
    INITCATALOG(*lus)                                # noqa: F405
    _catalogues_charges[0] = True


class SolveurDS:
    """Evalue g = alpha - 1 par un calcul a la rupture sur Digital Structure."""

    def __init__(self, chemin_ds, dossier_etude, params_names, regions,
                 global_size=0.05, geo_min_approx=4, max_size=None,
                 analyse="Yield_analysis0", iteration=0,
                 archiver=False, verbeux=True):
        self.chemin_ds = chemin_ds
        self.dossier_etude = dossier_etude
        self.params_names = tuple(params_names)
        #: un dict `{'param': ..., 'region_key': ...}` par variable, dans l'ordre
        self.regions = list(regions)
        self.global_size = global_size
        self.geo_min_approx = geo_min_approx
        #: borne haute du mailleur. None = suit `global_size`. Voir
        #: `options_maillage` : c'est une taille RELATIVE, pas des metres.
        self.max_size = max_size
        self.analyse = analyse
        self.iteration = iteration
        self.archiver = archiver
        self.verbeux = verbeux
        self._appels = 0
        initialiser_catalogues()

    # ------------------------------------------------------------------ #
    @property
    def cout_par_appel(self) -> str:
        return "un SOCP complet : de la dizaine de secondes a plusieurs minutes"

    @property
    def nb_appels(self) -> int:
        """Nombre d'appels au solveur depuis la creation. C'est le cout reel
        d'une etude, et la seule grandeur que le budget d'enrichissement borne."""
        return self._appels

    # ------------------------------------------------------------------ #
    def _init_solver(self):
        with open(os.path.join(self.dossier_etude, "InitSolver.py"), "r") as fh:
            return fh.read()

    def _cle_vers_param(self, cle):
        """Mappe une cle de sensibilite Digital Structure vers un nom de
        variable. Correspondance EXACTE 'param:region_key', comme l'original."""
        for p, sens in zip(self.params_names, self.regions):
            if cle == sens['param'] + ':' + sens['region_key']:
                return p
        return None

    # ------------------------------------------------------------------ #
    def evaluer(self, valeurs, sensibilite=True, etiquette=None) -> Evaluation:
        path = self.chemin_ds
        patch_params(path, **{p: valeurs[p] for p in self.params_names})

        model = MODEL()                              # noqa: F405
        SET_CONTEXT(model, path)                     # noqa: F405

        with open(os.path.join(path, 'dsCad.txt'), 'r') as fh:
            exec(fh.read(), globals())               # noqa: S102
        model.Save(os.path.join(path, self.analyse + ".dscad"))
        erreurs = model.GETERRORS()
        if erreurs:
            print(erreurs)

        with open(os.path.join(path, 'dsLoad.txt'), 'r') as fh:
            script_load = fh.read()
        with CetLOAD.LOAD_MODEL(model, path):
            exec(script_load, globals())             # noqa: S102

        t0 = time.perf_counter()
        CetMESH.ANISO_MESH(self.analyse, self.iteration, path,
                           **options_maillage(self.global_size, self.geo_min_approx,
                                              model.GETHANDLEPTR(),
                                              max_size=self.max_size))
        t_mesh = time.perf_counter() - t0

        regions = self.regions if sensibilite else None
        t0 = time.perf_counter()
        CetSOLV.SOLV(self.analyse, self.iteration, path,
                     **options_solveur(self._init_solver(), model.GETHANDLEPTR(), regions))
        t_solv = time.perf_counter() - t0
        self._appels += 1

        return self._lire_resultat(path, sensibilite, etiquette, t_mesh, t_solv)

    # ------------------------------------------------------------------ #
    def _lire_resultat(self, path, sensibilite, etiquette, t_mesh, t_solv):
        with open(os.path.join(path, "%s_%d_kine.dsmetares"
                               % (self.analyse, self.iteration)), 'r') as fh:
            info = json.load(fh)["info"]

        alpha = info['Primal_bound'][0]
        dual = info.get('Dual_bound', [float('nan')])[0]
        statut = info.get('solver_status')
        converge = info.get('converged')
        diagnostic = {
            "solver_status": statut,
            "converged": converge,
            "solverIterations": info.get('solverIterations'),
            "numTetra": info.get('numTetra'),
            "systemSize": info.get('systemSize'),
            "Dual_bound": dual,
            "gap_relatif": abs(alpha - dual) / abs(alpha) if alpha else float('nan'),
            "t_mesh": t_mesh,
            "t_solv": t_solv,
            "etiquette": etiquette,
        }
        # `converged` et `solver_status` peuvent etre absents d'une version
        # ancienne du .dsmetares : dans ce cas on ne DECLARE PAS le point
        # malade, on dit seulement qu'on n'en sait rien.
        sain = True
        if converge is not None:
            sain = bool(converge)
        if statut is not None:
            sain = sain and statut == "OPTIMAL"

        grad_x = tuple([None] * len(self.params_names))
        if sensibilite and 'Sensitivity' in info:
            if self.verbeux:
                print(f"les sensibilites sont calculees pour les elements : "
                      f"{info['Sensitivity'].items()}")
            trouve = [None] * len(self.params_names)
            for cle, valeur in info['Sensitivity'].items():
                p = self._cle_vers_param(cle)
                if p is not None:
                    trouve[self.params_names.index(p)] = valeur
                if all(g is not None for g in trouve):
                    break
            grad_x = tuple(trouve)

        if self.archiver:
            self.archiver_sorties(path, etiquette or "appel_%03d" % self._appels)

        return Evaluation(g=alpha - 1.0, alpha=alpha, grad_x=grad_x,
                          sain=sain, diagnostic=diagnostic)

    # ------------------------------------------------------------------ #
    def archiver_sorties(self, path, etiquette):
        """Copie les sorties SOCP dans `SOCP_history/<etiquette>/`.

        Coute cher : ~8,8 Mo par point en flexion pure, ~424 Mo sur le Moulin
        Blanc -- ou l'historique avait atteint 135 Go. Pilote par
        `save_history` dans le fichier d'etude.
        """
        racine = os.environ.get("_DOE_MAIN_DS") or path
        cible = os.path.join(racine, "SOCP_history", etiquette)
        os.makedirs(cible, exist_ok=True)
        n, octets = 0, 0
        for suffixe in FICHIERS_SOCP:
            src = os.path.join(path, self.analyse + suffixe)
            if not os.path.exists(src):
                continue
            try:
                shutil.copy2(src, os.path.join(cible, os.path.basename(src)))
                n += 1
                octets += os.path.getsize(src)
            except Exception as exc:
                print(f"  [SOCP HISTORY] copy failed for {src} : {exc}", flush=True)
        print(f"  [SOCP HISTORY] {etiquette} : {n} fichiers sauves "
              f"({octets / 1024 / 1024:.1f} MB) dans {cible}", flush=True)
        return cible
