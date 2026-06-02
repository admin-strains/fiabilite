"""
CODE FIABILITE - VERSION AVEC DEFINITION DE FONCTIONS
"""
import os
import json
import shutil
import re

from STRAINS.rupt.APIs.CetCAD_API import *
from STRAINS.rupt.APIs import CetLOAD
from STRAINS.rupt.APIs.CetLOAD_API import *
import STRAINS.rupt.core.CetMESH as CetMESH
from STRAINS.rupt.core import CetSOLV as CetSOLV
from STRAINS.rupt.core import CetVISU as CetVISU, CetLIST as CetLIST
from STRAINS.rupt.APIs.CetNOTE_API import *
from STRAINS.rupt.APIs import CetNOTE


def getFile(nameFile):
    f = open(nameFile, 'r')
    res = f.read()
    f.close()
    return res


catalogTopo = getFile("C:\\workspace\\front\\STRAINS\\common\\Catalog\\CatalogTopo.json")
catalogDimensions = getFile("C:\\workspace\\front\\STRAINS\\common\\Catalog\\CatalogDimensions.json")
catalogBolt = getFile("C:\\workspace\\front\\STRAINS\\common\\Catalog\\CatalogBolts.json")
INITCATALOG(catalogTopo, catalogDimensions, catalogBolt)





import openturns as ot
import numpy as np
import autograd.numpy as anp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from smt.surrogate_models import GEKPLS
from scipy.optimize import brentq
import re
import math
from sklearn.cluster import DBSCAN


if __name__ == '__main__':
    modelname = "test_pure_flexion"

    print("=" * 70)
    print("CALCUL DE FIABILITE -- FLEXION PURE BETON")
    print("=" * 70)
    # --------------------------------------------------------------------------- #
    # OPTIONS UTILISATEUR                                                         #
    # --------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------- #
    # DEFINITION DU MODELE                                                        #
    do_KRG = False
    do_GEK = True
    do_HF = False
    try_pce = False

    params_names = ['fc','fy']
    n_var = len(params_names)
    n0 = 2
    fck, fyk=28, 550 #MPa
    cov_fck, cov_fyk = None, None

    # --------------------------------------------------------------------------- #
    # PARAMETRES FORM                                                             #
    n_max_FORM = 50
    tol_FORM = 1.0                                    # précision acceptée par FORM
    do_warm_start = False
    tol_warm_start = 0.0001                      # nécessité de faire le warm_start
    tol_all_modes = 0.01                              #comparaison entre deux modes
    do_multistart = False
    n_multistart = 1
    n0 = max(n0,n_multistart) # pour avoir n0>=n_multistart

    # --------------------------------------------------------------------------- #
    # PARAMETRES MODELE                                                           #
    # 1. GEK
    do_analytic_grad = False
    reduc_PLS = 0

    # 2. PCE                                                                
    do_pce = try_pce
    seuil_pce = 0.90                              # seuil de validation de l'erreur
    q = 0.75                                              # tri base poly candidats
    max_degree = 2                                       # degre max poly candidats
    min_max_degree = 1                                   # degre min poly candidats 

    # --------------------------------------------------------------------------- #
    # OPTIONS TEST
    print_DOE = True
    print_ana = True
    print_ana_hf_error = False
    do_visu_claude = True
    size_visu = 5
    n_grid_hf = 7
    u1_bornes = (-4, 7)
    u2_bornes = (-4, 7)
    print_pts = False
    # do_visu=True
    # do_GP_linear_test = True
    # do_GP_HF_test = True
    # do_pce_eval = False
    #  #NPO CHANGER POUR METTRE PARAMETRE
    
    # --------------------------------------------------------------------------- #
    # DEFINTION DE FONCTIONS                                                      #
    # --------------------------------------------------------------------------- #
   
    # --------------------------------------------------------------------------- #
    # FONCTION D'APPEL STRAINS ET DOE                                             #

    # --- DSCAD ET DSLOAD ---
    def patch_params(path, **params):
        """Reecrit dsCad.txt avec de nouvelles valeurs de parametres."""
        cad = os.path.join(path, 'dsCad.txt') #donne un nom au txt
        with open(cad, 'r') as f: #on stocke son contenu
            content = f.read()
        for name, value in params.items(): #on le modifie variable par variable pour celles dans la liste params
            content = re.sub(r'^' + name + r'\s*=.*$', f'{name}    = {value:.10f}', content, count=1, flags=re.MULTILINE)
        with open(cad, 'w') as f:
            f.write(content) #on l'écrit dans un fichier vide f (car 'w' donc vidé) - dsCad.txt est modifié
        # A COMPLETER AVEC COPIE COLLE DE CA AVEC DSLOAD QUAND ON AJOUTE LES LOADS MODIFIES.

    # --- DISTRIBUTIONS ---
    
    SIGMA_11, SIGMA_12, SIGMA_13 =  19.0, 22.0, 8.0 
    SIGMA = np.sqrt(SIGMA_11**2 + SIGMA_12**2 + SIGMA_13**2)  # ~30 MPa
    
    def loi_fy(fyk, cov=None):
        """
        Loi normale de fy calibree sur Eurocode (EN 1990 / EN 1992-1-1).
        fyk est le fractile 5% : mu_EC = fyk + 1.645 * sigma
        sigma issu du JCSS (SIGMA global) ou fourni via cov.

        Parameters
        ----------
        fyk : float
            Limite d'elasticite caracteristique [MPa] (fractile 5%).
        cov : float, optional
            Coefficient de variation. Si None, sigma = SIGMA (JCSS).

        Returns
        -------
        ot.Normal
            Distribution OpenTURNS.
        """
        if cov is not None:
            # on itere : mu = fyk / (1 - 1.645 * cov)
            mu_ec = fyk / (1.0 - 1.645 * cov)
            sig_ec = cov * mu_ec
        else:
            sig_ec = SIGMA
            mu_ec  = fyk + 1.645 * sig_ec

        dist = ot.Normal(mu_ec, sig_ec)
        dist.setName(f"fy_EC2_nom{int(fyk)}MPa")
        dist.setDescription(["fy [MPa]"])
        return dist
    
    def loi_fc(fck, cov=None):
        """
        Loi lognormale de fc calibree sur Eurocode 2.
        Moyenne : fcm = fck + 8 MPa (EN 1992-1-1 tableau 3.1)
        COV     : issu du tableau JCSS 3.1.2 (ou fourni manuellement).

        Parameters
        ----------
        fck : float
            Resistance caracteristique en compression [MPa] (ex. 35 pour C35).
        cov : float, optional
            Coefficient de variation. Si None, deduit de la classe JCSS la plus proche.

        Returns
        -------
        ot.LogNormal
            Loi ajustee de fc [MPa].
        """
        COV_TABLE = {"C15": 0.14, "C25": 0.12, "C35": 0.09, "C45": 0.07}
        classe = min(COV_TABLE, key=lambda c: abs(int(c[1:]) - fck))
        v = cov if cov is not None else COV_TABLE[classe]

        fcm = fck + 8.0
        sigma_ln = np.sqrt(np.log(1 + v**2))
        mu_ln    = np.log(fcm) - 0.5 * sigma_ln**2

        dist = ot.LogNormal(mu_ln, sigma_ln, 0.0)
        dist.setName(f"fc_EC2_C{int(fck)}")
        dist.setDescription(["fc [MPa]"])
        return dist

    def dist_jointe():
        dist = []
        if 'fc' in params_names:
            dist.append(loi_fc(fck, cov_fck)) 
        if 'fy' in params_names:
            dist.append(loi_fy(fyk, cov_fyk))
        #AJOUTER suite pour plus de variable 'if 'load' in params_names' etc.
        dist_X   = ot.JointDistribution(dist)
        return dist_X

    # --- APPELS STRAINS ---
    def run_one_SOL(modelname, SOL, params_names, sensitivity=False, with_sens_dict=None): 
        """Lance un calcul complet pour une valeur de FT donnee.
        Retourne la liste des solutions pour chaque jeu de variables dans SOL (liste de dictionnaire)"""
        path = "C:\\workspace\\storage\\admin\\SF\\" + modelname + ".ds"
        AnalysisName = 'Yield_analysis0'
        iteration = 0
        #MODIF 1 10/04 - on doit tout mettre dans params in SOL. TOUT.
        for i in range (len(SOL)): 
            patch_params(path, **SOL[i]) #à cette étape SOL ne contient que 'fc': ,'fy':
            model = MODEL() #ici model n'est pas encore rempli
            SET_CONTEXT(model, path)
            fileName = os.path.join(path, AnalysisName + ".dscad") #on crée le chemin du fichier disque .dscad lisible par C. C va tout faire et on renverra les info plus tard (.load)

            cadfile = open(path + '\\dsCad.txt', 'r')
            cadscript = cadfile.read() #on met dans cadscript les info de dsCad.txt
            exec(cadscript, globals()) # ici on modifie le modèle (C, cython) et donc les variables (on exécute le script de dsCad.txt ce qui modifie les variables - rien dans .dscad, tout dans var. en mémoire)
            model.Save(fileName) # ici on créé dscad et on enregistre les modifs des variables dans .dscad
            print(model.GETERRORS()) # est vide si pas de message d'erreur sur le logiciel

            loadfile = open(path + '\\dsLoad.txt', 'r')
            model.Load(fileName) #on remplit le modèle en lisant .dscad et ainsi l'utiliser avec LOAD_MODEL plus bas. 
            loadscript = loadfile.read() 
            with CetLOAD.LOAD_MODEL(model, path): #par with on appelle enter et exit et on force l'enregistrement par exit meme si erreur/ bug dans bloc.
                exec(loadscript, globals()) # pareil, on execute dsLoad et on enregistre dans var. mémoire

            Meshkwargs = { #définit la mesh - pas à comprendre ici car ne sera pas modifié. 
                "cadSurfOptions": {"volume_gradation": 1.5, "gradation": 1.5, "anisotropic_ratio": 10},
                "tetraOptions": {"optimisation_level": "standard", "verbose": "10"},
                "global_physical_size": 0.05,  # mesh fin pour bonne convergence
                "max_size": 0.05,
                "min_size": "-1",
                "gradation": 1.5,
                "volume_gradation": 1.5,
                "optimisation_level": "standard",
                "anisotropic_ratio": "10",
                "geometric_approximation_min": "4",
                "geometric_approximation_max": "25",
                "geometric_approximation_on_edge": "false",
                "geometric_approximation_on_face": "true",
                "use_surface_proximity": "false",
                "surface_proximity_ratio": 0,
                "approach": "kinematic",
                "write_debug_files": "true",
                "is_iso": "true",
                "coeff_on_error": 0.01,
                "remesh_type": 1,
                "old_size_factor": 0.0,
            }
            CetMESH.ANISO_MESH(AnalysisName, iteration, path, **Meshkwargs)

            kwargs = {"scaling": 1, "write_debug_files": "true"} # ci-dessous on définit dict kwargs en entrée de SOLV.
            exec(open(r"C:\_workingDir\_SF\test flexion\InitSolver.py").read(), globals()) #question pour Agnes : je ne suis pas sure que ca marche comme ca. 
            kwargs["static_params"] = static_params
            kwargs["cinematic_params"] = cinematic_params
            kwargs["MKLPardiso_params"] = MKLPardiso_params
            kwargs["MyPardiso_params"] = MyPardiso_params
            kwargs["MUMPS_params"] = MUMPS_params
            kwargs["FullLorentz"] = False
            kwargs["LorentzToSdp"] = False
            kwargs["SdpToLorentz"] = 0
            kwargs["printIntPointSolutioEvolution"] = False
            kwargs["trace_sur_point_integration"] = False
            kwargs["calculate_error"] = "false"
            kwargs["max_nbOfDiv"] = 0
            kwargs["customized_inc"] = [1]
            kwargs["tetra_discontinuities"] = False
            kwargs["activated_plasticity"] = True
            kwargs["welds_throat_limit"] = True
            kwargs["approach"] = "kinematic"

            if sensitivity:
                kwargs["sensitivity_analysis"] = "true"
                kwargs["sensitivity_regions"] = json.dumps([
                    {"param": "COMPRESSIVE_STRENGTH", "solids": ["Block1"]},
                    {"param": "YIELD_STRENGTH", "rebars": ["HA1","HA2","HA3","HA4"]},
                ]) #transformée en texte json (liste de caractères) pour être lisible par C++

            CetSOLV.SOLV(AnalysisName, iteration, path, **kwargs) #On relance le solveur avec le nouveau dsCad.

            # Lire le resultat
            metares_path = os.path.join(path, AnalysisName + "_0_kine.dsmetares") #on extrait l'addresse du fichier pour définir f
            with open(metares_path, 'r') as f: #f est le fichier créé par open, et on a with donc enter de fichier = donne accès au fichier (accès via f, toujours mettre as f) puis exit : ferme le fichier (qui reste lié à f)
                d = json.load(f) #chargement du fichier .dsmetares
            SOL[i]['g']=d['info']['Primal_bound'][0] -1
            for p in params_names:
                SOL[i][f'dg_{p}'] = None
            if sensitivity and 'Sensitivity' in d['info']:
                print(f"les sensibilités sont calculées pour les elements : {d['info']['Sensitivity'].items()}")
                for k, v in d['info']['Sensitivity'].items():
                    #je ne sais pas encore comment généraliser pour le code ci dessous donc je vais juste
                    #faire if 1, if 2, mais on devrait faire une double boucle, mais la question est comment
                    #on définit la liste des noms 'tensile_strength' etc. Voir dans dsCad. 
                    if 'COMPRESSIVE_STRENGTH' in k: 
                        SOL[i]['dg_fc']= v
                    if 'YIELD_STRENGTH' in k: 
                        SOL[i]['dg_fy']= v
                    if all(SOL[i].get(f'dg_{p}') is not None for p in params_names):
                        break    
        return SOL

    def run_HF(u):
        sensitivity = True
        n_var = len(u)
        dist_X = dist_jointe()
        T_inv = dist_X.getInverseIsoProbabilisticTransformation() 
        u_point = ot.Point(u)
        x_point = T_inv(u_point)
        path = "C:\\workspace\\storage\\admin\\SF\\" + modelname + ".ds"
        AnalysisName = 'Yield_analysis0'
        iteration = 0
        params={params_names[i]: x_point[i] for i in range(n_var)}
        patch_params(path, **params) #à cette étape SOL ne contient que 'fc': ,'fy':
        model = MODEL() #ici model n'est pas encore rempli
        SET_CONTEXT(model, path)
        fileName = os.path.join(path, AnalysisName + ".dscad") #on crée le chemin du fichier disque .dscad lisible par C. C va tout faire et on renverra les info plus tard (.load)

        cadfile = open(path + '\\dsCad.txt', 'r')
        cadscript = cadfile.read() #on met dans cadscript les info de dsCad.txt
        exec(cadscript, globals()) # ici on modifie le modèle (C, cython) et donc les variables (on exécute le script de dsCad.txt ce qui modifie les variables - rien dans .dscad, tout dans var. en mémoire)
        model.Save(fileName) # ici on créé dscad et on enregistre les modifs des variables dans .dscad
        print(model.GETERRORS()) # est vide si pas de message d'erreur sur le logiciel

        loadfile = open(path + '\\dsLoad.txt', 'r')
        model.Load(fileName) #on remplit le modèle en lisant .dscad et ainsi l'utiliser avec LOAD_MODEL plus bas. 
        loadscript = loadfile.read() 
        with CetLOAD.LOAD_MODEL(model, path): #par with on appelle enter et exit et on force l'enregistrement par exit meme si erreur/ bug dans bloc.
            exec(loadscript, globals()) # pareil, on execute dsLoad et on enregistre dans var. mémoire

        Meshkwargs = { #définit la mesh - pas à comprendre ici car ne sera pas modifié. 
            "cadSurfOptions": {"volume_gradation": 1.5, "gradation": 1.5, "anisotropic_ratio": 10},
            "tetraOptions": {"optimisation_level": "standard", "verbose": "10"},
            "global_physical_size": 0.05,  # mesh fin pour bonne convergence
            "max_size": 0.05,
            "min_size": "-1",
            "gradation": 1.5,
            "volume_gradation": 1.5,
            "optimisation_level": "standard",
            "anisotropic_ratio": "10",
            "geometric_approximation_min": "4",
            "geometric_approximation_max": "25",
            "geometric_approximation_on_edge": "false",
            "geometric_approximation_on_face": "true",
            "use_surface_proximity": "false",
            "surface_proximity_ratio": 0,
            "approach": "kinematic",
            "write_debug_files": "true",
            "is_iso": "true",
            "coeff_on_error": 0.01,
            "remesh_type": 1,
            "old_size_factor": 0.0,
        }
        CetMESH.ANISO_MESH(AnalysisName, iteration, path, **Meshkwargs)

        kwargs = {"scaling": 1, "write_debug_files": "true"} # ci-dessous on définit dict kwargs en entrée de SOLV.
        exec(open(r"C:\_workingDir\_SF\test flexion\InitSolver.py").read(), globals()) #question pour Agnes : je ne suis pas sure que ca marche comme ca. 
        kwargs["static_params"] = static_params
        kwargs["cinematic_params"] = cinematic_params
        kwargs["MKLPardiso_params"] = MKLPardiso_params
        kwargs["MyPardiso_params"] = MyPardiso_params
        kwargs["MUMPS_params"] = MUMPS_params
        kwargs["FullLorentz"] = False
        kwargs["LorentzToSdp"] = False
        kwargs["SdpToLorentz"] = 0
        kwargs["printIntPointSolutioEvolution"] = False
        kwargs["trace_sur_point_integration"] = False
        kwargs["calculate_error"] = "false"
        kwargs["max_nbOfDiv"] = 0
        kwargs["customized_inc"] = [1]
        kwargs["tetra_discontinuities"] = False
        kwargs["activated_plasticity"] = True
        kwargs["welds_throat_limit"] = True
        kwargs["approach"] = "kinematic"

        if sensitivity:
            kwargs["sensitivity_analysis"] = "true"
            kwargs["sensitivity_regions"] = json.dumps([
                {"param": "COMPRESSIVE_STRENGTH", "solids": ["Block1"]},
                {"param": "YIELD_STRENGTH", "rebars": ["HA1","HA2","HA3","HA4"]},
            ]) #transformée en texte json (liste de caractères) pour être lisible par C++

        CetSOLV.SOLV(AnalysisName, iteration, path, **kwargs) #On relance le solveur avec le nouveau dsCad.

        # Lire le resultat
        metares_path = os.path.join(path, AnalysisName + "_0_kine.dsmetares") #on extrait l'addresse du fichier pour définir f
        with open(metares_path, 'r') as f: #f est le fichier créé par open, et on a with donc enter de fichier = donne accès au fichier (accès via f, toujours mettre as f) puis exit : ferme le fichier (qui reste lié à f)
            d = json.load(f) #chargement du fichier .dsmetares
        g_HF=d['info']['Primal_bound'][0] -1
        grad_HF_X=[None]*n_var
        grad_HF_U=[None]*n_var
        if sensitivity and 'Sensitivity' in d['info']:
            print(f"les sensibilités sont calculées pour les elements : {d['info']['Sensitivity'].items()}")
            for k, v in d['info']['Sensitivity'].items():
                #je ne sais pas encore comment généraliser pour le code ci dessous donc je vais juste
                #faire if 1, if 2, mais on devrait faire une double boucle, mais la question est comment
                #on définit la liste des noms 'tensile_strength' etc. Voir dans dsCad. #faudrait un truc avec des clés et des asocciations officielels entre fc et compressive strength.... 
                if 'COMPRESSIVE_STRENGTH' in k: 
                    grad_HF_X[params_names.index('fc')] = v
                if 'YIELD_STRENGTH' in k: 
                    grad_HF_X[params_names.index('fy')] = v
                if all(grad_HF_X[i] is not None for i in range(n_var)):
                    break
            J_Tinv = T_inv.gradient(u)
            J_Tinv_T = J_Tinv.transpose()
            grad_HF_U = J_Tinv_T * ot.Point(grad_HF_X)
        if sensitivity and any(v is None for v in grad_HF_U):
            raise ValueError(f"run_HF : sensibilité demandée mais grad_HF_U contient None — vérifier que STRAINS a bien calculé les sensibilités. grad_HF_X={grad_HF_X}")
        return g_HF, grad_HF_U, grad_HF_X

    # --- DOE ---
    def build_DOE():
        dist = []
        if 'fc' in params_names:
            dist.append(loi_fc(fck, cov_fck))
        if 'fy' in params_names:
            dist.append(loi_fy(fyk, cov_fyk))
        dist_X   = ot.JointDistribution(dist) 
        T     = dist_X.getIsoProbabilisticTransformation() # on interroge dist_X et trouve la transfo n�cessaire puis l'applique ici
        T_inv = dist_X.getInverseIsoProbabilisticTransformation()
        dist_U = dist_X.getStandardDistribution()
        lhs    = ot.LHSExperiment(dist_U, n0)
        sa     = ot.SimulatedAnnealingLHS(lhs, ot.SpaceFillingMinDist())
        U_doe  = sa.generate()
        if print_DOE:
                print("U_doe_fixed = ot.Sample([")
                for i in range(U_doe.getSize()):
                    vals = []
                    for j in range(U_doe.getDimension()):
                        v = U_doe[i][j]
                        vals.append(f' {v:.16f}' if v >= 0 else f'{v:.16f}')
                    print(f"    [{', '.join(vals)}],")
                print("])", flush=True)
        X_doe  = T_inv(U_doe)
        SOL = [{} for _ in range(n0)] 
        for i in range(n0):
            for j in range(n_var):
                SOL[i][params_names[j]] = X_doe[i][j]
        SOL = run_one_SOL(modelname, SOL, params_names, sensitivity=True, with_sens_dict=None)
        xt = np.array(U_doe)
        yt = np.array([SOL[i]['g'] for i in range(n0)]).reshape(-1, 1)
        if print_DOE:
            print("yt_doe = [")
            for i in range(n0):
                print(f"    {yt[i][0]:.16f},")
            print("]", flush=True)
        all_grad = np.zeros((n0, n_var))
        for i in range (n0):
            J_Tinv = T_inv.gradient(U_doe[i])
            J_Tinv_T = J_Tinv.transpose()
            grad_X_g = ot.Point([SOL[i][f'dg_{p}'] for p in params_names])
            grad_U_g = J_Tinv_T * grad_X_g
            for j in range (n_var):
                all_grad[i][j]= grad_U_g[j]
                SOL[i][f'dg_u{j+1}'] = grad_U_g[j]
        return xt, yt, all_grad

    # --------------------------------------------------------------------------- #
    # FONCTION ANALYTIQUE DE REFERENCE                                            #
    class flexion_simple:
        # Initialisation ----------
        def __init__(self, Med, As, b, h, d, fc_otparams, fy_otparams, Es = 200000, ecu = 0.0035, eud = 0.045, gamma_c = 1.5, gamma_s = 1.15):
            # Données fixes
            self.Med = Med
            self.As = As
            self.b = b
            self.h = h
            self.d = d
            self.Es = Es
            self.ecu = ecu

            # Transformation isoprobabiliste (espace standard U → espace physique X)
            dist = []
            if 'fc' in params_names:
                dist.append(loi_fc(*fc_otparams))
            if 'fy' in params_names:
                dist.append(loi_fy(*fy_otparams))
            dist_X     = ot.JointDistribution(dist)
            self.T_inv = dist_X.getInverseIsoProbabilisticTransformation()
            self.T     = dist_X.getIsoProbabilisticTransformation()

            # Constantes pour la fonction de performance
            self.A = As * d / gamma_s
            self.B = -0.5 * As**2 / b * gamma_c/ gamma_s**2
            self.C = -Med
            self.Ap = 0.8 * b * d / (As * Es * ecu * gamma_c)
            self.Bp = 0.8 * b * d**2 / gamma_c

            # Constantes pour les conditions pivot A, plast
            self.A1 = As * Es * ecu / (0.8 * b * d) * gamma_c 
            self.A2 = Es * ecu * gamma_s
            self.A3 = ecu / (ecu + eud)

        # Définition de la fonction de performance ----------
        def f_plast(self, u):
            x_point = self.T_inv(ot.Point(u))
            x1, x2  = x_point[0], x_point[1]
            return self.A * x2 + self.B * x2**2 / x1 + self.C

        def f_nonplast(self, u):
            x1 = self.T_inv(ot.Point(u))[0]
            val = self.Bp * (-1 + (1 + 4 * self.Ap * x1)**0.5 - 0.1 * (-1 + (1 + 4 * self.Ap * x1)**0.5)**2 / (self.Ap**2 * x1)) + self.C
            return val

        def test_plast(self, u):
            x_point = self.T_inv(ot.Point(u))
            x1, x2  = x_point[0], x_point[1]
            alpha_init = (self.As * x2) / (0.8 * self.b * self.d * x1)
            est = (1 - alpha_init) / alpha_init * self.ecu
            return anp.where(est >= x2 / self.Es, 1.0, 0.0)

        def f(self, u): #fonction de performance (pas état limite)
            tp = self.test_plast(u)
            return tp * self.f_plast(u) + (1 - tp) * self.f_nonplast(u)
        
        def print_f(self):
            fig, ax = plt.subplots()

            u1_range = np.linspace(-10, 10, 100)
            u2_range = np.linspace(-10, 10, 100)
            U1, U2 = np.meshgrid(u1_range, u2_range)
            Z = np.array([[float(self.f([u1, u2])) for u1 in u1_range] for u2 in u2_range])
            ax.contour(U1, U2, Z, levels=[0], colors='blue')
            plt.show()

        # def gp_pivotB(self,u1):
        #     x_point = self.T_inv(ot.Point([u1,0.0]))
        #     x1 = x_point[0]
        #     a = self.B
        #     b = self.A*x1
        #     c = self.C*x1
        #     Delta = b**2 - 4*a*c
        #     return -(self.A*x1 + Delta**0.5)/(2*self.B)
            
        # def gnp_pivotB(self):
        #     a = self.Ap*self.Bp
        #     b = self.Ap*(2*self.Bp+self.C)-0.4*self.Bp
        #     c = 2*self.Ap*self.C
        #     Delta = b**2 - 4.0 * a * c
        #     u = (-b + Delta**0.5) / (2.0 * a)
        #     return u*(u+2)/(4*self.Ap)
        
        # def limite_pivotA(self):
        #     return (4 * self.B * self.C) / (self.A**2 - (self.A - (-2 * self.B) * self.A3 * self.A2 / self.A1)**2)

        # def print_ana(self, ax):
        #     # Médiane de chaque variable (T_inv([0,0]) = point médian en espace physique)
        #     origin = self.T_inv(ot.Point([0.0, 0.0]))
        #     x1_ref, x2_ref = origin[0], origin[1]

        #     def u1_de_x1(x1):
        #         return self.T(ot.Point([x1, x2_ref]))[0]

        #     def u2_de_x2(x2):
        #         return self.T(ot.Point([x1_ref, x2]))[1]

        #     # Frontières physiques x1 → espace standard u1
        #     x1_npB  = self.gnp_pivotB()
        #     x1_limA = self.limite_pivotA()
        #     u1_npB  = u1_de_x1(x1_npB)
        #     u1_limA = u1_de_x1(x1_limA)

        #     # --- Zone pivot B plastique : u1 ∈ [u1_npB, u1_limA] ---
        #     u1_B = np.linspace(u1_npB, u1_limA, 300)
        #     u2_B = []
        #     for u1 in u1_B:
        #         try:
        #             u2_B.append(u2_de_x2(self.gp_pivotB(u1)))
        #         except Exception:
        #             u2_B.append(np.nan)
        #     u2_B = np.array(u2_B)

        #     # --- Zone pivot A : u2 constante, u1 > u1_limA ---
        #     u2_cst = u2_de_x2(self.g_pivotA())
        #     xlim   = ax.get_xlim()
        #     u1_max = xlim[1] if xlim[1] > u1_limA + 0.5 else u1_limA + 4.0
        #     u1_A   = np.linspace(u1_limA, u1_max, 100)
        #     u2_A   = np.full_like(u1_A, u2_cst)

        #     # --- Tracé ---
        #     ax.axvline(u1_npB, color='green', linestyle='-.', linewidth=1.5)
        #     ax.plot(u1_B, u2_B, color='green', linestyle='-.', linewidth=2, label='g=0 ana')
        #     ax.plot(u1_A, u2_A, color='green', linestyle='-.', linewidth=2)

    
    # --------------------------------------------------------------------------- #
    # FONCTION ANALYTIQUE CORRIGEE                                               #
    class flexion_claude:
        """Version corrigée de flexion_simple — gnp_pivotB, limite_pivotA,
        gp_pivotB et print_ana déboguées (unités, signe, espace de retour)."""

        def __init__(self, Med, As, b, h, d, fc_otparams, fy_otparams,
                     Es=200000, ecu=0.0035, eud=0.045, gamma_c=1.5, gamma_s=1.15):
            self.Med = Med
            self.As  = As
            self.b   = b
            self.h   = h
            self.d   = d
            self.Es  = Es
            self.ecu = ecu
            self.fyk = fy_otparams[0]

            dist = []
            if 'fc' in params_names:
                dist.append(loi_fc(*fc_otparams))
            if 'fy' in params_names:
                dist.append(loi_fy(*fy_otparams))
            dist_X     = ot.JointDistribution(dist)
            self.T_inv = dist_X.getInverseIsoProbabilisticTransformation()
            self.T     = dist_X.getIsoProbabilisticTransformation()

            self.A  = As * d / gamma_s
            self.B  = -0.5 * As**2 / b * gamma_c / gamma_s**2
            self.C  = -Med
            self.A1 = As * Es * ecu / (0.8 * b * d) * gamma_c
            self.A2 = Es * ecu * gamma_s
            self.A3 = ecu / (ecu + eud)

        def gnp_pivotB(self):
            gamma_s = self.A2 / (self.Es * self.ecu)
            gamma_c = self.A1 * 0.8 * self.b * self.d / (self.As * self.Es * self.ecu)
            fyd = self.fyk / gamma_s
            eyd = fyd / self.Es
            return self.As * fyd * gamma_c * (self.ecu + eyd) / (0.8 * self.b * self.d * self.ecu)

        def limite_pivotA(self):
            gamma_c = self.A1 * 0.8 * self.b * self.d / (self.As * self.Es * self.ecu)
            x_limA  = self.d * self.A3
            return self.Med * gamma_c / (0.8 * self.b * x_limA * (self.d - 0.4 * x_limA))

        def gp_pivotB(self, u1):
            x_point = self.T_inv(ot.Point([u1, 0.0]))
            x1 = x_point[0]
            a  = self.B
            b  = self.A * x1
            c  = self.C * x1
            Delta = b**2 - 4 * a * c
            fy = -(self.A * x1 - Delta**0.5) / (2 * self.B)
            return self.T(ot.Point([x1, fy]))[1]

        def print_ana(self, ax):
            origin = self.T_inv(ot.Point([0.0, 0.0]))
            x1_ref, x2_ref = origin[0], origin[1]

            def u1_de_x1(x1):
                return self.T(ot.Point([x1, x2_ref]))[0]

            x1_npB  = self.gnp_pivotB()
            x1_limA = self.limite_pivotA()
            u1_npB  = u1_de_x1(x1_npB)
            u1_limA = u1_de_x1(x1_limA)

            # Zone pivot B plastique : u1 ∈ [u1_npB, u1_limA]
            u1_B = np.linspace(u1_npB, u1_limA, 300)
            u2_B = []
            for u1 in u1_B:
                try:
                    u2_B.append(self.gp_pivotB(u1))
                except Exception:
                    u2_B.append(np.nan)
            u2_B = np.array(u2_B)

            # Zone pivot A : u2 constant au-delà de u1_limA
            u2_cst = self.gp_pivotB(u1_limA)
            xlim   = ax.get_xlim()
            u1_max = xlim[1] if xlim[1] > u1_limA + 0.5 else u1_limA + 4.0
            u1_A   = np.linspace(u1_limA, u1_max, 100)
            u2_A   = np.full_like(u1_A, u2_cst)

            ax.axvline(u1_npB, color='green', linestyle='-.', linewidth=1.5)
            ax.plot(u1_B, u2_B, color='green', linestyle='-.', linewidth=2, label='g=0 ana')
            ax.plot(u1_A, u2_A, color='green', linestyle='-.', linewidth=2)

    def _parse(text, name):
        return float(re.search(rf'(?m)^\s*{re.escape(name)}\s*=\s*([\d.]+)', text).group(1))

    def calc_ana():
        # --- Lecture des données du modèle DS ---
        path = os.path.join(r'C:\workspace\storage\admin\SF', modelname + '.ds')
        with open(os.path.join(path, 'dsCad.txt'), 'r') as f:
            _cad = f.read()
        with open(os.path.join(path, 'dsLoad.txt'), 'r') as f:
            _load = f.read() 

        b   = _parse(_cad, 'b')    # largeur section (m)
        h   = _parse(_cad, 'h')    # hauteur section (m)
        L   = _parse(_cad, 'L')    # longueur poutre (m)
        phi = _parse(_cad, 'phi')  # diamètre armatures (mm)

        n_bars = len(re.findall(r'REBAR\(', _cad))
        As = n_bars * math.pi * (phi / 2e3) ** 2   # section totale acier (m²)

        # Hauteur utile : centroïde des armatures (3e coordonnée des POINT dans les lits)
        z_rebar = [float(v) for v in re.findall(
            r'pts\d+\.append\(POINT\([^,]+,\s*[^,]+,\s*([\d.]+)\)', _cad)]
        d = h/2 + sum(z_rebar) / len(z_rebar)      # distance fibre comprimée → centroïde aciers (m)

        # Charge appliquée (valeur absolue, en MN)
        F = abs(float(re.search(r"Z='(-?[\d.]+)'", _load).group(1)))   # MN

        Med = F * L   # moment en console (MN·m)

        calc = flexion_simple(Med=Med, As=As, b=b, h=h, d=d,
                            fc_otparams=(fck, cov_fck), fy_otparams=(fyk, cov_fyk))
        return calc

    def calc_ana_claude():
        path = os.path.join(r'C:\workspace\storage\admin\SF', modelname + '.ds')
        with open(os.path.join(path, 'dsCad.txt'), 'r') as f:
            _cad = f.read()
        with open(os.path.join(path, 'dsLoad.txt'), 'r') as f:
            _load = f.read()

        b   = _parse(_cad, 'b')
        h   = _parse(_cad, 'h')
        L   = _parse(_cad, 'L')
        phi = _parse(_cad, 'phi')

        n_bars = len(re.findall(r'REBAR\(', _cad))
        As = n_bars * math.pi * (phi / 2e3) ** 2

        z_rebar = [float(v) for v in re.findall(
            r'pts\d+\.append\(POINT\([^,]+,\s*[^,]+,\s*([\d.]+)\)', _cad)]
        d = h/2 + sum(z_rebar) / len(z_rebar)

        F = abs(float(re.search(r"Z='(-?[\d.]+)'", _load).group(1)))
        Med = F * L

        return flexion_claude(Med=Med, As=As, b=b, h=h, d=d,
                              fc_otparams=(fck, cov_fck), fy_otparams=(fyk, cov_fyk))

    # --------------------------------------------------------------------------- #
    # FONCTIONS LIEES AU MODELE HF                                                #

    # --- Wrapper OpenTURNS avec gradients analytiques ---
    class HFFunction(ot.OpenTURNSPythonFunction):
        def __init__(self):
            super().__init__(n_var, 1)
            self._cache_u    = None
            self._cache_g    = None
            self._cache_grad = None
            self.n_hf_calls  = 0  # compteur pour vérification

        def _run_if_needed(self, u):
            u_arr = np.array(u)
            if self._cache_u is None or not np.allclose(u_arr, self._cache_u, atol=1e-12):
                g, grad_U, _ = run_HF(u)
                self._cache_u    = u_arr.copy()
                self._cache_g    = float(g)
                self._cache_grad = [float(grad_U[i]) for i in range(n_var)]
                self.n_hf_calls += 1

        def _exec(self, u):
            self._run_if_needed(u)
            return [self._cache_g]

        def _gradient(self, u):
            self._run_if_needed(u)
            # Format OpenTURNS : (n_var, 1)
            return [[g] for g in self._cache_grad]

    # --------------------------------------------------------------------------- #
    # FONCTIONS LIEES AU MODELE PCE                                               #

    def build_metamodel_PCE(xt, y_hf):
        # 1. INITIALISATION : DOE ET DISTRIBUTION
        inputSample = ot.Sample(xt)
        outputSample = ot.Sample(y_hf)
        n0 = xt.shape[0]
        dist_X = dist_jointe()
        dist_U = dist_X.getStandardDistribution()

        # 2. BASE DE CANDIDATS : TYPE, ENUMERATION, DEGRE
        n_var = inputSample.getDimension()
        enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(n_var, q)
        basis = ot.OrthogonalProductPolynomialFactory([ot.HermiteFactory()] * n_var, enumerateFunction)
        basis_size = enumerateFunction.getBasisSizeFromTotalDegree(max_degree)
        basisStrategy = ot.FixedStrategy(basis, basis_size)

        # 3. PROPOSITION / PROJECTION / SELECTION
        selectionStrategy = ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut())
        projectionStrategy = ot.LeastSquaresStrategy(selectionStrategy) 

        # 4. RESULTAT
        algo = ot.FunctionalChaosAlgorithm(inputSample, outputSample, dist_U, basisStrategy, projectionStrategy)
        algo.run()
        result = algo.getResult()         
        metamodel = result.getMetaModel()
        return metamodel
       
    def build_residu(xt, y_hf, all_grad_hf, metamodel_PCE): #a appeler que dans le bloc do_pce, donc pas de do_pce en param.
        U_doe = ot.Sample(xt)                               # a modifier pour remettre le calcul de all_sensib_hf dans build_doe puis ici
        y_PCE = np.array(metamodel_PCE(U_doe))
        n_var = U_doe.getDimension()
        n0 = U_doe.getSize()
        dist_X = dist_jointe()
        T = dist_X.getIsoProbabilisticTransformation()
        T_inv = dist_X.getInverseIsoProbabilisticTransformation()
        all_grad_PCE = np.zeros((n0, n_var))
        # all_sensib_PCE = np.zeros((n0, n_var))
        for i in range(n0):
            grad_pce_u = metamodel_PCE.gradient(U_doe[i])       
            for j in range(n_var):
                all_grad_PCE[i, j] = grad_pce_u[j, 0]
            # sensibilités
            x_i = T_inv(ot.Sample([U_doe[i]]))[0]               
            J_T_i = T.gradient(x_i)                             
            grad_u_i = ot.Point(all_grad_PCE[i, :])             
            grad_x_i = J_T_i * grad_u_i                         
            # all_sensib_PCE[i, :] = np.array(grad_x_i)              
        return y_hf-y_PCE, all_grad_hf-all_grad_PCE
    
    def eval_PCE():
        return do_pce
    # --------------------------------------------------------------------------- #
    # FONCTIONS LIEES AU KRG                                                      #

    def build_metamodel_KRG(xt, yt):
        n_var = xt.shape[1]
        basis = ot.ConstantBasisFactory(n_var).build()
        # covarianceModel = ot.MaternModel([1.0] * n_var, 2.5)
        covarianceModel = ot.SquaredExponential([1.0] * n_var)
        algo_KRG = ot.KrigingAlgorithm(xt, yt, covarianceModel, basis)
        algo_KRG.run()
        result = algo_KRG.getResult()
        metamodel = result.getMetaModel()
        return metamodel

    # --------------------------------------------------------------------------- #
    # FONCTIONS LIEES AU GEK                                                      #

    # --- GEKPLS ---
    def build_metamodel_GEK(xt, yt, all_grad):
        xlimits = np.column_stack([xt.min(axis=0) - 1, xt.max(axis=0) + 1])

        sm = GEKPLS(
            n_comp=2,
            theta0=[1e-2, 1e-2],
            corr="squar_exp",
            poly="constant",
            xlimits=xlimits,
            print_global=False,
        )
        sm.set_training_values(xt, yt)
        for j in range(n_var):
            sm.set_training_derivatives(xt, all_grad[:, j].reshape(-1, 1), j)
        sm.train()
        return sm

    # --- Wrapper OpenTURNS avec gradients analytiques ---
    class GEKPLSFunction(ot.OpenTURNSPythonFunction):
        def __init__(self, surrogate):
            super().__init__(n_var, 1)
            self.sm = surrogate

        def _exec(self, u):
            return [self.sm.predict_values(np.array(u).reshape(1, -1)).item()]

        def _gradient(self, u):
            u_np = np.array(u).reshape(1, -1)
            return [[self.sm.predict_derivatives(u_np, kx).item()] for kx in range(n_var)]
    
    # --------------------------------------------------------------------------- #
    # FONCTIONS POUR FORM                                                         #

    def FORM_event(g_ot):
        # --- Événement de défaillance ---
        distribution = ot.ComposedDistribution([ot.Normal(0, 1)] * n_var)
        X = ot.RandomVector(distribution)
        Y = ot.CompositeRandomVector(g_ot, X)
        event = ot.ThresholdEvent(Y, ot.Less(), 0.0)
        return event

    

    def FORM_all_modes(starting_points, tol_all_modes):
        """
        Multi-start FORM + DBSCAN pour identifier les modes de défaillance.
        - Chaque cluster DBSCAN = un mode distinct.
        - u* isolés (label -1) = descentes mal convergées, ignorées.
        """
        all_u_star   = []   # u* de chaque run réussi
        all_results  = []   # FORMResult correspondant

        for sp in starting_points:
            try:
                solver = ot.AbdoRackwitz()
                solver.setStartingPoint(sp.tolist())
                form_i = ot.FORM(solver, event)
                form_i.run()
                r_i    = form_i.getResult()
                u_star = np.array(r_i.getStandardSpaceDesignPoint())
                all_u_star.append(u_star)
                all_results.append(r_i)
                print(f"  [sp={[round(v,3) for v in np.array(sp)]}, "
                    f"u*={[round(v,3) for v in u_star]}, "
                    f"beta={r_i.getHasoferReliabilityIndex():.4f}]", flush=True)
            except Exception as e:
                print(f"  [sp={[round(v,3) for v in np.array(sp)]}, "
                    f"ECHEC ({type(e).__name__})]", flush=True)

        if not all_u_star:
            return []

        # --- DBSCAN ---
        U_all  = np.array(all_u_star)          # shape (n_runs_ok, n_var)
        db     = DBSCAN(eps=tol_all_modes, min_samples=2).fit(U_all)
        labels = db.labels_

        n_noise = np.sum(labels == -1)
        if n_noise > 0:
            print(f"  {n_noise} descente(s) mal convergée(s) ignorée(s) (bruit DBSCAN)", flush=True)

        # --- Un mode par cluster : FORMResult avec beta minimal ---
        modes = []
        for lbl in sorted(set(labels) - {-1}):
            idx_cluster = [i for i, l in enumerate(labels) if l == lbl]
            best_i = min(idx_cluster,
                        key=lambda i: all_results[i].getHasoferReliabilityIndex())
            modes.append(all_results[best_i])

        modes.sort(key=lambda r: r.getHasoferReliabilityIndex())

        print(f"\n{len(modes)} mode(s) distinct(s) "
            f"(DBSCAN eps={tol_all_modes}, min_samples=2) :", flush=True)
        for i, m in enumerate(modes):
            u = [round(v, 3) for v in m.getStandardSpaceDesignPoint()]
            print(f"  mode {i+1} : beta={m.getHasoferReliabilityIndex():.4f}  "
                f"Pf={m.getEventProbability():.3e}  u*={u}", flush=True)

        return modes


    # --- Multi-start FORM depuis les points du DOE ---
    def FORM_multistart(starting_points):
        best_beta = np.inf
        best_result = None
        best_sp = None

        for sp in starting_points:
            try:
                solver = ot.AbdoRackwitz()
                solver.setStartingPoint(sp.tolist())
                solver.setMaximumIterationNumber(n_max_FORM)
                solver.setCheckStatus(False)
                solver.setMaximumConstraintError(tol_FORM)
                form_i = ot.FORM(solver, event)
                form_i.run()
                r_i = form_i.getResult()
                if r_i.getHasoferReliabilityIndex() < best_beta:
                    best_beta = r_i.getHasoferReliabilityIndex()
                    best_result = r_i
                    best_sp = sp
            except Exception as e:
                print(f"FORM exception: {type(e).__name__}: {e}")
        return best_result, best_sp

    # --------------------------------------------------------------------------- #
    # FONCTIONS RESULTATS/ AFFICHAGE                                              #  
    def print_results(best_result, g_ot_GEK, g_ot_KRG, g_ot_HF):
        u_star = best_result.getStandardSpaceDesignPoint()
        n_iter = best_result.getOptimizationResult().getIterationNumber()
        dist_X = dist_jointe()
        T_inv  = dist_X.getInverseIsoProbabilisticTransformation()
        x_star = T_inv(u_star)

        # --- Toujours affiché ---
        print(f"n_iter FORM  = {n_iter}", flush=True)
        for i, p in enumerate(params_names):
            print(f"{p}*          = {x_star[i]:.4f}", flush=True)
        print(f"u*           = {[round(v, 4) for v in u_star]}", flush=True)
        print(f"Imp.         = {[round(v, 4) for v in best_result.getImportanceFactors()]}", flush=True)
        print(f"beta         = {best_result.getHasoferReliabilityIndex():.4f}", flush=True)
        print(f"Pf           = {best_result.getEventProbability():.4e}", flush=True)

        # --- Bloc GEK ---
        if g_ot_GEK is not None:
            _, grad_HF_U_star, _ = run_HF(u_star)
            for i, p in enumerate(params_names):
                print(f"dg/du_{p} en u* (HF@u*GEK) = {grad_HF_U_star[i]:.6f}", flush=True)
            u0             = ot.Point([0.0] * n_var)
            g0_HF, grad_HF_U0, _ = run_HF(u0)
            u_FOSM         = grad_HF_U0 * (-g0_HF / grad_HF_U0.normSquare())
            print(f"u* FOSM (HF) = {[round(v, 4) for v in u_FOSM]}", flush=True)
            print(f"Erreur FOSM  = {(u_FOSM - u_star).norm() / u_star.norm():.4f}", flush=True)

        # --- Bloc KRG ---
        if g_ot_KRG is not None:
            _, grad_HF_U_star, _ = run_HF(u_star)
            for i, p in enumerate(params_names):
                print(f"dg/du_{p} en u* (HF@u*KRG) = {grad_HF_U_star[i]:.6f}", flush=True)
            u0             = ot.Point([0.0] * n_var)
            g0_HF, grad_HF_U0, _ = run_HF(u0)
            u_FOSM         = grad_HF_U0 * (-g0_HF / grad_HF_U0.normSquare())
            print(f"u* FOSM (HF) = {[round(v, 4) for v in u_FOSM]}", flush=True)
            print(f"Erreur FOSM  = {(u_FOSM - u_star).norm() / u_star.norm():.4f}", flush=True)

        # --- Bloc HF ---
        if g_ot_HF is not None:
            grad = g_ot_HF.gradient(u_star)
            for i, p in enumerate(params_names):
                print(f"dg/du_{p} en u* (HF)  = {grad[i, 0]:.6f}", flush=True)
            u0         = ot.Point([0.0] * n_var)
            g0         = g_ot_HF(u0)[0]
            grad_0     = g_ot_HF.gradient(u0)
            grad_0_vec = ot.Point([grad_0[i, 0] for i in range(n_var)])
            u_FOSM     = grad_0_vec * (-g0 / grad_0_vec.normSquare())
            print(f"u* FOSM (HF)  = {[round(v, 4) for v in u_FOSM]}", flush=True)
            print(f"Erreur FOSM   = {(u_FOSM - u_star).norm() / u_star.norm():.4f}", flush=True)

    def print_error_ana_hf(calc, n_scan=100):
        """
        Pour chaque u1, scanne u2 et trouve TOUS les zéros de f_ana = 0
        (détecte toutes les branches par changements de signe successifs).
        On calcule l'erreur relative sur la grille obtenue. 
        Utilise les globales : size_visu, fyk, cov_fyk, SIGMA.
        n_scan : résolution du scan u2 pour détecter les changements de signe.
        """
        if cov_fyk is None:
            u2_min = -(fyk / SIGMA + 1.645)
        else:
            u2_min = -1.0 / cov_fyk
        u2_low = max(-size_visu, u2_min)

        pts = []
        u1_scan = np.linspace(-size_visu, size_visu, n_scan)
        for u2 in np.linspace(size_visu, u2_low, 40):  # u2 decroissant, du haut vers le bas
            g_vals = [calc.f_ana([u1, u2]) for u1 in u1_scan]
            for i in range(len(u1_scan) - 1):
                if g_vals[i] * g_vals[i+1] < 0:
                    u1_star = brentq(
                        lambda u1: calc.f_ana([u1, u2]), u1_scan[i], u1_scan[i+1])
                    if abs(calc.f_ana([u1_star, u2])) < 1e-1:
                        pts.append([u1_star, u2])
        # --- Points HF sur la frontière analytique ---
        frontier_pts = np.array(pts) if pts else np.zeros((0, 2))
        if print_pts:
            print(f"Points sur la frontiere analytique : {len(frontier_pts)}", flush=True)
            for pt in frontier_pts:
                print(f"  u=({pt[0]:.4f}, {pt[1]:.4f})", flush=True)
            return None
        print(f"Points sur la frontière : {len(frontier_pts)}")
        if len(frontier_pts):
            print(f"  u1 : [{frontier_pts[:,0].min():.2f}, {frontier_pts[:,0].max():.2f}]")
            print(f"  u2 : [{frontier_pts[:,1].min():.2f}, {frontier_pts[:,1].max():.2f}]")

        # --- Sélection des points HF : sous-échantillonnage uniforme ---
        n_hf_target = 2 * n_grid_hf
        step = max(1, len(frontier_pts) // n_hf_target)
        error_grid = frontier_pts[::step]
        print(f"error_grid : {len(error_grid)} points (1 sur {step})")

        # --- Évaluation g_HF sur les points frontière ana ---
        g_HF_vals    = np.array([run_HF(pt)[0]          for pt in error_grid])
        f_ana_vals   = np.array([calc.f_ana(list(pt))   for pt in error_grid])   # ≈ 0 par construction

        print("u_grid    :", [list(np.round(pt, 4)) for pt in error_grid])
        print("g_HF_vals :", list(np.round(g_HF_vals, 6)))
        print("f_ana_vals:", list(np.round(f_ana_vals, 6)))

        err_abs      = np.abs(g_HF_vals - f_ana_vals)

        dist_X = dist_jointe()
        T_inv  = dist_X.getInverseIsoProbabilisticTransformation()
        print("--- Validation g_HF vs f_ana sur frontière ---")
        for i, pt in enumerate(error_grid):
            x = T_inv(ot.Point(list(pt)))
            print(f"  pt {i:2d} u=({pt[0]:6.2f},{pt[1]:6.2f})  "
                f"fc={x[0]:6.2f}  fy={x[1]:6.2f}  "
                f"g_HF={g_HF_vals[i]:+.4f}  f_ana={f_ana_vals[i]:+.4f}  "
                f"err_abs={err_abs[i]:.4f}")
        print(f"  → err_abs_moy = {err_abs.mean():.4f}  (biais moyen g_HF - f_ana sur la frontiere)")

    def print_visu(best_result, best_sp, xt, sm_GEK, g_ot_KRG, g_hf, modes):
        n_grid = 100
        u1 = np.linspace(*u1_bornes, n_grid)
        u2 = np.linspace(*u2_bornes, n_grid)
        U1, U2 = np.meshgrid(u1, u2)
        grid = np.column_stack([U1.ravel(), U2.ravel()])

        u_star = np.array(best_result.getStandardSpaceDesignPoint())

        fig, ax = plt.subplots(figsize=(7, 6))

        # --- Fond coloré : GEKPLS en priorité, sinon KRG ---
        if sm_GEK is not None:
            Z_surr = sm_GEK.predict_values(grid).reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_surr, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (surrogate GEKPLS)')
            ax.contour(U1, U2, Z_surr, levels=[0], colors='blue', linewidths=2)
        elif g_ot_KRG is not None:
            grid_ot = ot.Sample(grid.tolist())
            Z_krg = np.array(g_ot_KRG(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_krg, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (KRG)')

        # --- Contour KRG (toujours tracé si dispo, même si sm_GEK fournit le fond) ---
        if g_ot_KRG is not None:
            grid_ot = ot.Sample(grid.tolist())
            Z_krg = np.array(g_ot_KRG(grid_ot))[:, 0].reshape(n_grid, n_grid)
            ax.contour(U1, U2, Z_krg, levels=[0], colors='purple', linewidths=2, linestyles=':')

        # --- Contour HF grossier ---
        if g_hf is not None:
            u1_hf = np.linspace(*u1_bornes, n_grid_hf)
            u2_hf = np.linspace(*u2_bornes, n_grid_hf)
            U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
            grid_hf = np.column_stack([U1_hf.ravel(), U2_hf.ravel()])
            Z_true = np.array([g_hf(pt)[0] for pt in grid_hf]).reshape(n_grid_hf, n_grid_hf)
            ax.contour(U1_hf, U2_hf, Z_true, levels=[0], colors='red', linewidths=2, linestyles='--')


        # --- Points ---
        ax.scatter(xt[:, 0], xt[:, 1],    c='black',  s=30,  zorder=5, label='DOE')
        ax.scatter(0, 0,                   c='orange', s=100, zorder=6, marker='P', label='[0, 0]')
        ax.scatter(best_sp[0], best_sp[1], c='cyan',   s=100, zorder=7, marker='D', label='point de depart best')
        ax.scatter(u_star[0], u_star[1],   c='gold',   s=200, zorder=8, marker='*',
                label=f'u* mode1 beta={best_result.getHasoferReliabilityIndex():.3f}')
        if len(modes) > 0:
            for k, mode in enumerate(modes[1:], start=2):
                u_m = np.array(mode.getStandardSpaceDesignPoint())
                ax.scatter(u_m[0], u_m[1], c='magenta', s=200, zorder=8, marker='*',
                        label=f'u* mode{k} beta={mode.getHasoferReliabilityIndex():.3f}')

        # --- Légende contours ---
        legend_lines = []
        if sm_GEK is not None:
            legend_lines.append(Line2D([0], [0], color='blue',   linestyle='-',  linewidth=2, label='g=0 GEKPLS'))
        if g_ot_KRG is not None:
            legend_lines.append(Line2D([0], [0], color='purple', linestyle=':',  linewidth=2, label='g=0 KRG'))
        if g_hf is not None:
            legend_lines.append(Line2D([0], [0], color='red',    linestyle='--', linewidth=2, label='g=0 HF'))
        if calc is not None:
            legend_lines.append(Line2D([0], [0], color='green',  linestyle='-.', linewidth=2, label='g=0 ana'))

        ax.legend(handles=ax.legend().legend_handles + legend_lines)

        ax.set_xlabel('u1')
        ax.set_ylabel('u2')
        ax.set_title('FORM sur GEKPLS')
        plt.tight_layout()
        plt.show()

    def print_visu_claude(best_result, best_sp, xt, sm_GEK, g_ot_KRG, g_hf, modes, calc):
        n_grid = 100
        u1 = np.linspace(*u1_bornes, n_grid)
        u2 = np.linspace(*u2_bornes, n_grid)
        U1, U2 = np.meshgrid(u1, u2)
        grid = np.column_stack([U1.ravel(), U2.ravel()])

        u_star = np.array(best_result.getStandardSpaceDesignPoint())

        fig, ax = plt.subplots(figsize=(7, 6))

        # --- Fond coloré : GEKPLS en priorité, sinon KRG ---
        if sm_GEK is not None:
            Z_surr = sm_GEK.predict_values(grid).reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_surr, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (surrogate GEKPLS)')
            ax.contour(U1, U2, Z_surr, levels=[0], colors='blue', linewidths=2)
        elif g_ot_KRG is not None:
            grid_ot = ot.Sample(grid.tolist())
            Z_krg = np.array(g_ot_KRG(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_krg, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (KRG)')

        if g_ot_KRG is not None:
            grid_ot = ot.Sample(grid.tolist())
            Z_krg = np.array(g_ot_KRG(grid_ot))[:, 0].reshape(n_grid, n_grid)
            ax.contour(U1, U2, Z_krg, levels=[0], colors='purple', linewidths=2, linestyles=':')

        # --- Contour HF grossier ---
        if g_hf is not None:
            u1_hf = np.linspace(*u1_bornes, n_grid_hf)
            u2_hf = np.linspace(*u2_bornes, n_grid_hf)
            U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
            grid_hf = np.column_stack([U1_hf.ravel(), U2_hf.ravel()])
            Z_true = np.array([g_hf(pt)[0] for pt in grid_hf]).reshape(n_grid_hf, n_grid_hf)
            ax.contour(U1_hf, U2_hf, Z_true, levels=[0], colors='red', linewidths=2, linestyles='--')

        # --- Courbe analytique flexion_claude ---
        if calc is not None:
            calc.print_ana(ax)

        # --- Points ---
        ax.scatter(xt[:, 0], xt[:, 1],    c='black',  s=30,  zorder=5, label='DOE')
        ax.scatter(0, 0,                   c='orange', s=100, zorder=6, marker='P', label='[0, 0]')
        ax.scatter(best_sp[0], best_sp[1], c='cyan',   s=100, zorder=7, marker='D', label='point de depart best')
        ax.scatter(u_star[0], u_star[1],   c='gold',   s=200, zorder=8, marker='*',
                label=f'u* mode1 beta={best_result.getHasoferReliabilityIndex():.3f}')
        if len(modes) > 0:
            for k, mode in enumerate(modes[1:], start=2):
                u_m = np.array(mode.getStandardSpaceDesignPoint())
                ax.scatter(u_m[0], u_m[1], c='magenta', s=200, zorder=8, marker='*',
                        label=f'u* mode{k} beta={mode.getHasoferReliabilityIndex():.3f}')

        # --- Légende contours ---
        legend_lines = []
        if sm_GEK is not None:
            legend_lines.append(Line2D([0], [0], color='blue',   linestyle='-',  linewidth=2, label='g=0 GEKPLS'))
        if g_ot_KRG is not None:
            legend_lines.append(Line2D([0], [0], color='purple', linestyle=':',  linewidth=2, label='g=0 KRG'))
        if g_hf is not None:
            legend_lines.append(Line2D([0], [0], color='red',    linestyle='--', linewidth=2, label='g=0 HF'))
        if calc is not None:
            legend_lines.append(Line2D([0], [0], color='green',  linestyle='-.', linewidth=2, label='g=0 ana (claude)'))

        ax.legend(handles=ax.legend().legend_handles + legend_lines)

        ax.set_xlabel('u1')
        ax.set_ylabel('u2')
        ax.set_title('FORM sur GEKPLS — courbe analytique corrigée')
        plt.tight_layout()
        plt.show()


    """
    DEBUT DE CODE
    """
    sm_GEK = None
    sm_GEPCK = None
    g_ot_GEK = None
    g_ot_GEPCK = None
    g_ot_KRG = None
    g_ot_PCKRG = None
    g_ot_HF = None
    calc = None
    event = None
    
    
    if do_KRG and not try_pce:
        xt, yt, _ = build_DOE()
        g_ot_KRG = build_metamodel_KRG(xt, yt)
        event = FORM_event(g_ot_KRG)

    elif do_GEK and not try_pce: 
        xt, yt, all_grad = build_DOE()
        sm_GEK = build_metamodel_GEK(xt, yt, all_grad)
        g_ot_GEK = ot.Function(GEKPLSFunction(sm_GEK))
        event = FORM_event(g_ot_GEK)
    
    elif do_KRG and try_pce:
        xt, y_hf, all_grad_hf = build_DOE()
        g_ot_PCE = build_metamodel_PCE(xt, y_hf)
        yt, all_grad = build_residu(xt, y_hf, all_grad_hf, g_ot_PCE)
        g_ot_PCKRG = build_metamodel_KRG(xt, yt)
        event = FORM_event(g_ot_PCKRG)
    
    elif do_GEK and try_pce:
        xt, y_hf, all_grad_hf = build_DOE()
        g_ot_PCE = build_metamodel_PCE(xt, y_hf)
        yt, all_grad = build_residu(xt, y_hf, all_grad_hf, g_ot_PCE)
        sm_GEPCK = build_metamodel_GEK(xt, yt, all_grad)
        g_ot_GEPCK = ot.Function(GEKPLSFunction(sm_GEPCK))
        event = FORM_event(g_ot_GEPCK)

    elif do_HF:
        xt = np.empty((0, n_var))
        g_ot_HF = ot.Function(HFFunction())
        event = FORM_event(g_ot_HF)
    
    if event is None:
        print('Aucune branche active', flush=True)
        sys.exit(1)

    starting_points = np.vstack([xt, [[0.0, 0.0]]]) #pour l'instant cela signifie faire MS avant WS. 
    best_result, best_sp = FORM_multistart(starting_points)
    modes = FORM_all_modes(starting_points, tol_all_modes)
    if best_result is None:
        print('Aucun FORM ne marche.', flush=True)
        sys.exit(1)
    if len(modes)>1:
        print('On a trouvé plus de 1 mode! Les résultats du mode 2 sont:')
        print_results(modes[1], g_ot_GEK, g_ot_KRG, g_ot_HF)
        print('Les résultats du mode 1 sont : ')
    print_results(best_result, g_ot_GEK, g_ot_KRG, g_ot_HF)

    if do_visu_claude:
        calc_claude = calc_ana_claude()
        if not try_pce:
            print_visu_claude(best_result, best_sp, xt, sm_GEK, g_ot_KRG, run_HF, modes, calc_claude)
        else:
            print_visu_claude(best_result, best_sp, xt, sm_GEPCK, g_ot_PCKRG, run_HF, modes, calc_claude)
    else:
        if print_ana:
            calc = calc_ana()
            calc.print_f()
        if not try_pce:
            print_visu(best_result, best_sp, xt, sm_GEK, g_ot_KRG, run_HF, modes)
        else:
            print_visu(best_result, best_sp, xt, sm_GEPCK, g_ot_PCKRG, run_HF, modes)
        


    """
    DEBUT DANCIEN CODE COMMENTE
    """
    # --------------------------------------------------------------------------- #
    # DEBUT DE CODE                                                               #
    # --------------------------------------------------------------------------- #
    # if do_GP and try_pce and not do_GEK:
    # # --------------------------------------------------------------------------- #
    # # MODELE HYBRIDE PC-KRG                                                       #
    # # --------------------------------------------------------------------------- #
    # # 1. METAMODELE                                                               #
    #     xt, y_hf, all_grad_hf, all_sensib_hf = init_GP(modelname, params_names, n0, U_doe_fixed)
    #     # ici manquant : validation du PCE puis changement de do_pce si on ne veut pas faire GP hybride  
    #     if do_pce:
    #         metamodel_PCE = build_metamodel_PCE(modelname, params_names, xt, y_hf)
    #     y_PCE, all_grad_PCE, all_sensib_PCE = fill_PCE(modelname, params_names, xt, metamodel_PCE)
    #     yt, all_grad, all_sensib = fill_inputGP(y_hf, all_grad_hf, all_sensib_hf, y_PCE, all_grad_PCE, all_sensib_PCE, do_pce)
    #     metamodel_KRG = build_metamodel_KRG(xt, yt)
    #     if do_pce:
    #         metamodel = metamodel_KRG + metamodel_PCE
    #     else:
    #         metamodel = metamodel_KRG
    # # --------------------------------------------------------------------------- #
    # # 2. FORM                                                                     
    #     start_point = [0.0]*len(params_names)
    #     result = FORM_KRG(modelname, params_names, metamodel, start_point)                     
    #     U_warm = result.getPhysicalSpaceDesignPoint()
    #     if do_warm_start and metamodel(U_warm)[0] > tol_warm_start: 
    #         U_doe = ot.Sample(xt)
    #         U_doe.add(U_warm)
    #         print(f"Warm start lancé avec point de départ U={list(U_warm)}")
    #         xt_warm = np.array(U_warm)
    #         y_hf_warm, all_grad_hf_warm, all_sensib_hf_warm = run_HF(modelname, params_names, U_warm)
    #         xt = np.vstack([xt, xt_warm.reshape(1, -1)])
    #         y_hf       = np.vstack([y_hf, [[y_hf_warm]]])
    #         all_grad_hf = np.vstack([all_grad_hf, np.array(all_grad_hf_warm).reshape(1, -1)])
    #         all_sensib_hf = np.vstack([all_sensib_hf, np.array(all_sensib_hf_warm).reshape(1, -1)])
    #         # ici manquant : validation du PCE puis changement de do_pce si on ne veut pas faire GP hybride  
    #         if do_pce:
    #             metamodel_PCE = build_metamodel_PCE(modelname, params_names, xt, y_hf)
    #         y_PCE, all_grad_PCE, all_sensib_PCE = fill_PCE(modelname, params_names, xt, metamodel_PCE)
    #         yt, all_grad, all_sensib = fill_inputGP(y_hf, all_grad_hf, all_sensib_hf, y_PCE, all_grad_PCE, all_sensib_PCE, do_pce)
    #         metamodel_KRG = build_metamodel_KRG(xt, yt)
    #         if do_pce:
    #             metamodel = metamodel_KRG + metamodel_PCE
    #         else:
    #             metamodel = metamodel_KRG
    #         start_point = U_warm
    #         result = FORM_KRG(modelname, params_names, metamodel, start_point)
    #     n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM = resultats_GP(modelname, params_names, result, metamodel)
    # # --------------------------------------------------------------------------- #
    # # 3. AFFICHAGE                                                                #
    #     print_resultats(n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM)
    #     print_GP_tests(modelname, params_names, U_res, g_GP_res, metamodel)
    #     print_visu_HF_GP(metamodel, U_res, g_GP_res, size_visu, beta, Pf_FORM)
    
    # elif do_GP and not try_pce and not do_GEK:
    # # --------------------------------------------------------------------------- #
    # # MODELE KRG PUR                                                              #
    # # --------------------------------------------------------------------------- #
    # # 1. METAMODELE                                                               #
    #     xt, yt, all_grad, all_sensib = init_GP(modelname, params_names, n0, U_doe_fixed)
    #     metamodel = build_metamodel_KRG(xt, yt)
    # # --------------------------------------------------------------------------- #
    # # 2. FORM                                                                     #
    #     start_point = [0.0]*len(params_names)
    #     result = FORM_KRG(modelname, params_names, metamodel, start_point)
    #     U_warm = result.getPhysicalSpaceDesignPoint()
    #     if do_warm_start and metamodel(U_warm)[0] > tol_warm_start:
    #         U_doe = ot.Sample(xt) 
    #         U_doe.add(U_warm)
    #         print(f"Warm start lancé avec point de départ U={list(U_warm)}")
    #         xt_warm = np.array(U_warm)
    #         yt_warm, all_grad_warm, all_sensib_warm = run_HF(modelname, params_names, U_warm)
    #         xt = np.vstack([xt, xt_warm.reshape(1, -1)])
    #         yt       = np.vstack([yt, [[yt_warm]]])
    #         all_grad = np.vstack([all_grad, np.array(all_grad_warm).reshape(1, -1)])
    #         all_sensib = np.vstack([all_sensib, np.array(all_sensib_warm).reshape(1, -1)])
    #         metamodel = build_metamodel_KRG(xt, yt)
    #         start_point = U_warm
    #         result = FORM_KRG(modelname, params_names, metamodel, start_point)
    #     n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM = resultats_GP(modelname, params_names, result, metamodel)
    #     # if do_multi_start:
    #     #     U_doe_multistart = ot.Sample(xt)
    #     #     result_modes = [result] #revoir si devrait être défini hors de if
    #     #     while U_doe_multistart.getSize() > 0:
    #     #         result_modes, U_doe_multistart = FORM_multi_start(result_modes, U_doe_multistart)
    #     #     if len(result_modes)>1:
    #     #         print(f'Il y a plusieurs modes de défaillances.')
    #     #         i = 1
    #     #         for result in result_modes: 
    #     #             n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM = resultats_GP(modelname, params_names, result_modes[i], metamodel)
    #     #             print(f'\nRESULTATS DU MODE {i+1}')
    #     #             print_resultats(n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM)
        
    # # --------------------------------------------------------------------------- #
    # # 3. AFFICHAGE                                                                #
    #     print_resultats(n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM)
    #     print_GP_tests(modelname, params_names, U_res, g_GP_res, metamodel)
    #     print_visu_HF_GP(metamodel, U_res, g_GP_res, size_visu, beta, Pf_FORM)
    
    # elif do_GP and try_pce and do_GEK:
    # # --------------------------------------------------------------------------- #
    # # MODELE HYBRIDE GEPCK                                                        #
    # # --------------------------------------------------------------------------- #
    # # 1. METAMODELE                                                               #
    #     xt, y_hf, all_grad_hf, all_sensib_hf = init_GP(modelname, params_names, n0, U_doe_fixed)
    #     # ici manquant : validation du PCE puis changement de do_pce si on ne veut pas faire GP hybride  
    #     if do_pce:
    #         metamodel_PCE = build_metamodel_PCE(modelname, params_names, xt, y_hf)
    #     y_PCE, all_grad_PCE, all_sensib_PCE = fill_PCE(modelname, params_names, xt, metamodel_PCE)
    #     yt, all_grad, all_sensib = fill_inputGP(y_hf, all_grad_hf, all_sensib_hf, y_PCE, all_grad_PCE, all_sensib_PCE, do_pce)
    #     sm = build_metamodel_GEK(xt, yt, all_grad)
    #     metamodel = build_metamodel_total(sm, metamodel_PCE)
    # # --------------------------------------------------------------------------- #
    # # 2. FORM                                                                     # 
    #     start_point = [0.0]*len(params_names)
    #     result = FORM_GEK(modelname, params_names, metamodel, start_point)
    #     U_warm = result.getPhysicalSpaceDesignPoint()
    #     if do_warm_start and metamodel(U_warm)[0] > tol_warm_start:
    #         U_doe = ot.Sample(xt)
    #         U_doe.add(U_warm)
    #         print(f"Warm start lancé avec point de départ U={list(U_warm)}")
    #         xt_warm = np.array(U_warm)
    #         y_hf_warm, all_grad_hf_warm, all_sensib_hf_warm = run_HF(modelname, params_names, U_warm)
    #         xt = np.vstack([xt, xt_warm.reshape(1, -1)])
    #         y_hf       = np.vstack([y_hf, [[y_hf_warm]]])
    #         all_grad_hf = np.vstack([all_grad_hf, np.array(all_grad_hf_warm).reshape(1, -1)])
    #         all_sensib_hf = np.vstack([all_sensib_hf, np.array(all_sensib_hf_warm).reshape(1, -1)])
    #         if do_pce:
    #             metamodel_PCE = build_metamodel_PCE(modelname, params_names, xt, y_hf)
    #         y_PCE, all_grad_PCE, all_sensib_PCE = fill_PCE(modelname, params_names, xt, metamodel_PCE)
    #         yt, all_grad, all_sensib = fill_inputGP(y_hf, all_grad_hf, all_sensib_hf, y_PCE, all_grad_PCE, all_sensib_PCE, do_pce)
    #         sm = build_metamodel_GEK(xt, yt, all_grad)
    #         metamodel = build_metamodel_total(sm, metamodel_PCE)
    #         start_point = U_warm
    #         result = FORM_GEK(modelname, params_names, metamodel, start_point)
    #     n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM = resultats_GP(modelname, params_names, result, metamodel)
    # # --------------------------------------------------------------------------- #
    # # 3. AFFICHAGE                                                                #
    #     print_resultats(n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM)
    #     print_GP_tests(modelname, params_names, U_res, g_GP_res, metamodel)
    #     print_visu_HF_GP(metamodel, U_res, g_GP_res, size_visu, beta, Pf_FORM)

    # # 1. METAMODELE                                                               #
    #     # xt, yt, all_grad, all_sensib = init_GP(modelname, params_names, n0, U_doe_fixed)
    #     # sm = build_metamodel_GEK(xt, yt, all_grad)
    #     # metamodel = build_metamodel_total(sm)
    # # --------------------------------------------------------------------------- #
    # # 2. FORM                                                                     #
    #     # start_point = [0.0]*len(params_names)
    #     # result = FORM_GEK(modelname, params_names, metamodel, start_point)
    #     # U_warm = result.getPhysicalSpaceDesignPoint()
    #     # if do_warm_start and metamodel(U_warm)[0] > tol_warm_start:
    #     #     U_doe = ot.Sample(xt) 
    #     #     U_doe.add(U_warm)
    #     #     print(f"Warm start lancé avec point de départ U={list(U_warm)}")
    #     #     xt_warm = np.array(U_warm)
    #     #     yt_warm, all_grad_warm, all_sensib_warm = run_HF(modelname, params_names, U_warm)
    #     #     xt = np.vstack([xt, xt_warm.reshape(1, -1)])
    #     #     yt       = np.vstack([yt, [[yt_warm]]])
    #     #     all_grad = np.vstack([all_grad, np.array(all_grad_warm).reshape(1, -1)])
    #     #     all_sensib = np.vstack([all_sensib, np.array(all_sensib_warm).reshape(1, -1)])
    #     #     sm = build_metamodel_GEK(xt, yt, all_grad)
    #     #     metamodel = build_metamodel_total(sm)
    #     #     start_point = U_warm
    #     #     result = FORM_GEK(modelname, params_names, metamodel, start_point)
    #     # n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM = resultats_GP(modelname, params_names, result, metamodel)
    # # --------------------------------------------------------------------------- #
    # # 3. AFFICHAGE                                                                #
    #     # print_resultats(n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM)
    #     # print_GP_tests(modelname, params_names, U_res, g_GP_res, metamodel)
    #     # print_visu_HF_GP(metamodel, U_res, g_GP_res, size_visu, beta, Pf_FORM)

    # else:
    # # --------------------------------------------------------------------------- #
    # # MODELE HF                                                                   #
    # # --------------------------------------------------------------------------- #
    # # 1. MODELE                                                                   #
    #     hf_cache = build_hf_cache(modelname, params_names)
    # # --------------------------------------------------------------------------- #
    # # 2. FORM                                                                     #
    #     start_point = [0.0]*len(params_names)
    #     result = FORM_HF(modelname, params_names, hf_cache, start_point)
    #     if mode_number_goal>1:
    #         result_modes = [result]

    """
    LES ANCIENNES FONCTIONS RESULTATS COMMENTEES
    """
        # def resultats_GP(modelname, params_names, result, metamodel):
    #     n_iter = result.getOptimizationResult().getIterationNumber()
    #     U_res = result.getPhysicalSpaceDesignPoint()
    #     dist_X = dist_jointe(modelname, params_names)
    #     T_inv = dist_X.getInverseIsoProbabilisticTransformation()
    #     X_res = T_inv(U_res)
    #     if do_GEK:
    #         g_GP_res, grad_res = metamodel(U_res)
    #     else:
    #         g_GP_res = metamodel(U_res)[0]
    #         grad_res = metamodel.gradient(U_res) 
    #     importance = result.getImportanceFactors()
    #     beta = result.getHasoferReliabilityIndex()
    #     Pf_FORM = result.getEventProbability()
    #     return n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM
    
    # def print_resultats(n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM):
    #     n_var = U_res.getDimension()
    #     print(f"Nombre d'itérations FORM : {n_iter}")
    #     for i in range(n_var):
    #         print(f"  Design point U : u_{params_names[i]} = {U_res[i]:.4f}")
    #     for i in range(n_var):
    #         print(f"  Design point X : {params_names[i]} = {X_res[i]:.4f}")
    #     print(f"  g_GP_res   = {g_GP_res:.6f}")
    #     for i in range(n_var):
    #         print(f"  dg/du_{params_names[i]} en u* = {grad_res[i, 0]:.6f}")
    #     for i in range(n_var):
    #         print(f"  Importance factor {params_names[i]}: {importance[i]:.4f}")
    #     print(f"\nBeta FORM = {beta:.6f}", flush=True)
    #     print(f"Pf FORM   = {Pf_FORM:.6e}", flush=True)
    
    # def GP_linear_test(modelname, params_names, U_res):
    #     n_var = U_res.getDimension()
    #     u0 = ot.Point([0.0] * n_var)
    #     g0, grad_U_0, _ = run_HF(modelname, params_names, u0)
    #     norm_sq = grad_U_0.norm() ** 2
    #     u_FOSM = grad_U_0 * (-g0 / norm_sq)
    #     relative_error_FOSM = (u_FOSM - U_res).norm() / U_res.norm()
    #     return u_FOSM, relative_error_FOSM

    # def GP_HF_test(modelname, params_names, U_res, g_GP_res, metamodel): #renommer GP_test global une fois que tu sais ce que tu fais pour metamodel(GEK) car U_res doit etre convertit avant il me semble dans leur cas (pas un metamodel type ot pour ce cas)
    #     g_HF, _, _ = run_HF(modelname, params_names, U_res)
    #     relative_error_HF = abs(g_HF - g_GP_res) / abs(g_HF)
    #     return g_GP_res, g_HF, relative_error_HF       

    # def print_GP_tests(modelname, params_names, U_res, g_GP_res, metamodel): #pareil renommer, cest que le deuxieme bloc qu'il faut changer, sinon tu renomme et tu met un if dans cette fonction. ou tu arrive a adapter la précédente et pas de if. 
    #     if do_GP_linear_test:
    #         u_FOSM, relative_error_FOSM = GP_linear_test(modelname, params_names, U_res)
    #         print(f"\nTest linéarisation :")
    #         print(f"  u* FORM = {U_res}")
    #         print(f"  u* FOSM = {u_FOSM}")
    #         print(f"  Erreur relative entre u* FORM et u* FOSM : {relative_error_FOSM:.4f}")
    #     if do_GP_HF_test:
    #         g_GP, g_HF, relative_error_HF = GP_HF_test(modelname, params_names, U_res, g_GP_res, metamodel)
    #         print(f"\nTest GP au point de FORM :")
    #         print(f"  g* FORM = {g_HF:.6f}")
    #         print(f"  g* GP   = {g_GP:.6f}")
    #         print(f"  Erreur relative entre g* FORM et g* GP : {relative_error_HF:.4f}")
    
    # def print_visu_HF_GP(metamodel, U_res, g_GP_res, size_visu, beta, Pf_FORM):
    #     if do_visu:
    #         u_star = U_res
    #         n_fc = 8
    #         n_fy = 8  # 8×8 = 64 appels HF, nombre pair = pas de point central = pas de doublon avec u*
    #         u_fc_values = np.linspace( U_res[0] - size_visu,  U_res[0] + size_visu, n_fc)
    #         u_fy_values = np.linspace( U_res[1] - size_visu,  U_res[1] + size_visu, n_fy)

    #         G_HF = np.zeros((n_fy, n_fc))
    #         G_GP = np.zeros((n_fy, n_fc))
    #         for i, u_fy in enumerate(u_fy_values):
    #             for j, u_fc in enumerate(u_fc_values):
    #                 u_scan = ot.Point([u_fc, u_fy])
    #                 g_HF, _, _ = run_HF(modelname, params_names, u_scan)
    #                 G_HF[i, j] = g_HF
    #                 if do_GEK:
    #                     g_GP, _ = metamodel(u_scan)
    #                 else:
    #                     g_GP = metamodel(u_scan)[0]
    #                 G_GP[i, j] = g_GP

    #         from matplotlib.lines import Line2D
    #         plt.figure()
    #         plt.contour(u_fc_values, u_fy_values, G_HF, levels=[0], colors='r', linestyles='--', linewidths=2)
    #         plt.contour(u_fc_values, u_fy_values, G_GP, levels=[0], colors='b', linestyles='-',  linewidths=2)
    #         plt.plot(U_res[0], U_res[1], 'g*', markersize=14)
    #         plt.plot(0, 0, 'ko', markersize=8)
    #         legend_elements = [
    #             Line2D([0], [0], color='r', linestyle='--', linewidth=2, label='g_HF = 0'),
    #             Line2D([0], [0], color='b', linestyle='-',  linewidth=2, label='g_GP = 0'),
    #             Line2D([0], [0], color='g', marker='*', linestyle='', markersize=14, label=f'u* ({U_res[0]:.2f}, {U_res[1]:.2f})'),
    #             Line2D([0], [0], color='k', marker='o', linestyle='', markersize=8,  label='Moyenne (origine)'),
    #         ]
    #         plt.legend(handles=legend_elements)
    #         plt.xlabel('u_fc')
    #         plt.ylabel('u_fy')
    #         plt.title(f'Etat limite g=0 autour du point de conception, beta={beta:.3f}, Pf = {Pf_FORM:.3f}')
    #         plt.grid(True)
    #         plt.tight_layout()
    #         plt.savefig(r'C:\_workingDir\_SF\test flexion\etat_limite.png', dpi=150)
    #         plt.show()
    #         print("Visu sauvegardée : etat_limite.png")

    # --------------------------------------------------------------------------- #
    
    """
    LES ANCIENNES FONCTIONS FORM COMMENTEES
    """
        # A MODIFIER POUR AVOIR FORMAT EVENT
    # def FORM_HF(modelname, params_names, hf_cache, start_point):
    #     grad_call_count = [0]
    #     dist_X = dist_jointe(modelname, params_names)
    #     dist_U = dist_X.getStandardDistribution()
    #     n_var = len(params_names)
    #     def func(u):
    #         hf_cache.run_if_needed(u)
    #         return [hf_cache._last_g]
    #     def grad_func(u):
    #         grad_call_count[0] += 1
    #         print(f"[GRAD] appel #{grad_call_count[0]} en u={list(u)}", flush=True)
    #         hf_cache.run_if_needed(u)
    #         return [[v for v in hf_cache._last_grad]]
        
    #     myFunction = ot.PythonFunction(n_var, 1, func, gradient=grad_func)
    #     vect   = ot.RandomVector(dist_U)
    #     output = ot.CompositeRandomVector(myFunction, vect)
    #     event  = ot.ThresholdEvent(output, ot.Less(), 0.0)

    #     solver = ot.AbdoRackwitz()
    #     solver.setMaximumIterationNumber(n_max_FORM)
    #     solver.setCheckStatus(False)
    #     solver.setStartingPoint(start_point)

    #     algo = ot.FORM(solver, event)
    #     algo.run()
    #     result = algo.getResult()
    #     return result
        
    # def FORM_multi_start(result_modes, U_doe_multistart):
    #     norms = np.array([np.linalg.norm(np.array(u)) for u in U_doe_multistart])
    #     sorted_idx = np.argsort(norms)[::-1]  # indices ordre décroissant
    #     U_doe_multistart = ot.Sample([U_doe_multistart[int(i)] for i in sorted_idx])
    #     for n_FORM in range(n_multistart):
    #         solver.setStartingPoint(U_doe_multistart[n_FORM])
    #         algo = ot.FORM(solver, event)
    #         algo.run()
    #         u_new  = algo.getResult().getPhysicalSpaceDesignPoint()
    #         u_prev_list = [result.getPhysicalSpaceDesignPoint() for result in result_modes]
    #         if all((u_new - u_prev).norm() > tol_all_modes for u_prev in u_prev_list):
    #             result_modes.append(algo.getResult())
    #             U_doe_multistart = ot.Sample(np.delete(np.array(U_doe_multistart), n_FORM, axis=0))
    #             break
    #     return result_modes, U_doe_multistart

    """
    Le build total de gek
    """
        # ENLEVER TOTAL DU CODE POUR LINSTANT, LE TESTER SUR LAUTRE CODE PUIS LE REMETTRE CORRECTEMENT. 
    
    # def build_metamodel_total(sm, metamodel_PCE = None):
    #     n_var = len(params_names) 
    #     if metamodel_PCE is not None:
    #         def metamodel(u):
    #             u_np = np.array(u).reshape(1, -1)
    #             y_GEK = float(sm.predict_values(u_np)[0,0])
    #             grad_GEK = np.array([[float(sm.predict_derivatives(u_np, kx)[0, 0])] for kx in range(n_var)])
    #             u_sample_ot = ot.Sample(u_np)
    #             y_PCE = float(np.array(metamodel_PCE(u_sample_ot))[0,0])
    #             grad_PCE = np.array(metamodel_PCE.gradient(u))
    #             return y_GEK + y_PCE, grad_GEK + grad_PCE
    #     else: 
    #         def metamodel(u):
    #             u_np = np.array(u).reshape(1, -1)
    #             y_GEK = float(sm.predict_values(u_np)[0,0])
    #             grad_GEK = np.array([[float(sm.predict_derivatives(u_np, kx)[0, 0])] for kx in range(n_var)])
    #             return y_GEK, grad_GEK
    #     return metamodel

    """
    A supprimer
    """
    # def condition_pivotA(self, u1):
    #         x_point = self.T_inv(ot.Point([u1,0.0]))
    #         x1 = x_point[0]
    #         return self.A3*self.A2/self.A1 * x1
        
    # def condition_plast(self,u1):
    #     x_point = self.T_inv(ot.Point([u1,0.0]))
    #     x1 = x_point[0]
    #     return self.A2 * (-1 + (1 + 4 * x1 / self.A1)**0.5) / 2   


    # def g_ana(self,u1):
    #         x_point = self.T_inv(ot.Point([u1,0.0]))
    #         x1 = x_point[0]
    #         limite_pivotA = limite_pivotA(self) 
    #         if x1 > limite_pivotA:
    #             return g_pivotA(self)
    #         elif x1 < limite_pivotA and x1 > gnp_pivotB(self):
    #             return gp_pivotB(self,u1)
    #         else :
    #             # a completer





    # if print_ana:
    #     calc  = calc_ana()
    #     f_ana = calc.f_ana
    #     if print_ana_hf_error:
    #         print_error_ana_hf(calc, n_scan=100)