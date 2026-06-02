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
from scipy.stats import norm


def _parse(text, name):
    return float(re.search(rf'(?m)^\s*{re.escape(name)}\s*=\s*([\d.]+)', text).group(1))


if __name__ == '__main__':
    modelname = "test_pure_flexion"
    _path_ds = "C:\\workspace\\storage\\admin\\SF\\" + modelname + ".ds"
    with open(os.path.join(_path_ds, 'dsCad.txt'), 'r') as f:
        _cad_txt = f.read()

    print("=" * 70)
    print("CALCUL DE FIABILITE -- FLEXION PURE BETON")
    print("=" * 70)
    # --------------------------------------------------------------------------- #
    # OPTIONS UTILISATEUR                                                         #
    # --------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------- #
    # PARAMETRES DU MODELE                                                        #
    Es=200000
    ecu=0.0035
    eud=0.045
    gamma_c = _parse(_cad_txt, 'gamma_c')
    gamma_s = _parse(_cad_txt, 'gamma_s')
    n_rebars = len(re.findall(r'REBAR\(', _cad_txt))
    rebar_names = [f"HA{i+1}" for i in range(n_rebars)]
    gamma_c_fic = 1.0
    gamma_s_fic = 1.0

    params_names = ['fc','fy']
    n_var = len(params_names)
    n0 = 3
    fcm, fym = 48, 550 #MPa
    cov_fc, cov_fy = 0.12, None
    fc_otparams, fy_otparams = (fcm,cov_fc), (fym, cov_fy)

    # --------------------------------------------------------------------------- #
    # DEFINITION DU MODELE                                                        #
    do_KRG = False
    do_GEK = True
    do_HF = False
    try_pce = False
    do_eff = False

    # --------------------------------------------------------------------------- #
    # PARAMETRES FORM                                                             #
    n_max_FORM = 50
    do_multistart = True
    do_warmstart = False

    tol_FORM = 1.0                                    # précision acceptée par FORM
    tol_all_modes = 0.01                              #comparaison entre deux modes   
    tol_warmstart = 0.2                          # nécessité de faire le warm_start

    # --------------------------------------------------------------------------- #
    # PARAMETRES MODELE                                                           #
    # 1. GEK
    do_analytic_grad = False
    reduc_PLS = 0

    # 2. PCE                                                                
    do_pce = try_pce
    seuil_pce = 0.90                              # seuil de validation de l'erreur
    q = 0.75                                              # tri base poly candidats
    max_degree = 1                                       # degre max poly candidats
    # min_max_degree = 1                                 # degre min poly candidats 

    # 3. EFF
    epsilon_factor = 2                               # eps = epsilon_factor * sigma
    tol_EFF = 0.001                                           # critere d'arret EFF
    u1_eff_min, u1_eff_max = -10.0, 10.0
    u2_eff_min, u2_eff_max = -10.0, 10.0


    # --------------------------------------------------------------------------- #
    # PARAMETRES ET OPTIONS DE PRINT                                              #
    u1_max=10.0
    u2_max=10.0
    u1_min = -10.0
    u2_min=-10.0
    n_grid = 300
    n_grid_hf = 7
    print_ana = True
    print_HF = False
    print_DOE = True
    print_3D = False
    # -- temporaire ---
    print_grad_sp = False

    hf_3d_grid_fixed = {
        'params': (-10.0, 10.0, -10.0, 10.0, 7),
        'Z': [
            [-0.359874, -0.266997, -0.205356, -0.162941, -0.133274, -0.118529, -0.110761],
            [-0.259757, -0.080302,  0.041437,  0.123736,  0.180376,  0.218425,  0.237580],
            [-0.224965,  0.048410,  0.247795,  0.382025,  0.475826,  0.541447,  0.578053],
            [-0.196753,  0.115993,  0.413867,  0.617429,  0.752362,  0.844787,  0.910014],
            [-0.168402,  0.147183,  0.541008,  0.821024,  1.009377,  1.139576,  1.232379],
            [-0.140437,  0.175700,  0.624863,  0.998493,  1.249463,  1.421116,  1.540467],
            [-0.112244,  0.204017,  0.672091,  1.150387,  1.477693,  1.691754,  1.840008],
        ]
    }

    print_ana_hf_error = False
    print_pts = False

    # --- Résultats fixés du run HF 12/05 (gamma=1.0, F=0.74, n0=15) ---
    # Actifs uniquement en mode visu seule (tous do_* = False).
    if not do_HF and not do_GEK and not do_KRG:
        sol_modes_fixed = {
            # (sp_u1, sp_u2): (u*_u1, u*_u2)
            (-0.002,  1.332): (-5.306, -6.200),
            ( 0.610, -0.310): (-3.117, -7.349),
            (-0.571, -1.705): (-3.131, -7.347),
            ( 0.258,  0.306): (-4.655, -6.643),
            ( 0.121, -1.010): (-3.117, -7.345),
            ( 1.624, -0.531): (-3.046, -7.352),
            (-0.087, -0.172): (-4.721, -6.603),
            (-0.419,  0.597): (-5.290, -6.212),
            (-0.843, -0.005): (-5.341, -6.152),
            ( 0.868, -1.460): (-3.014, -7.363),
            (-1.681,  0.710): (-6.475, -4.986),
            ( 1.479,  0.239): (-3.098, -7.354),
            (-1.117, -0.696): (-5.200, -6.275),
            ( 0.745,  1.043): (-4.740, -6.571),
            (-0.694,  2.114): (-6.504, -4.966),
            ( 0.000,  0.000): (-4.776, -6.571),
        }
        best_sol_modes_fixed = {
            'A': {'sp': ( 0.868, -1.460), 'u*': (-3.014, -7.363)},
            'B': {'sp': ( 0.745,  1.043), 'u*': (-4.740, -6.571)},
            'C': {'sp': (-0.843, -0.005), 'u*': (-5.341, -6.152)},
            'D': {'sp': (-0.694,  2.114), 'u*': (-6.504, -4.966)},
        }
        # Gradients HF aux sp (run 1305_0937, 4 appels STRAINS)
        grad_sp_fixed = {
            'A': {'g': 0.550023, 'grad': [ 0.039500,  0.072312], 'neg_grad': [-0.039500, -0.072312]},
            'B': {'g': 0.722565, 'grad': [ 0.046148,  0.068963], 'neg_grad': [-0.046148, -0.068963]},
            'C': {'g': 0.573155, 'grad': [ 0.058216,  0.059250], 'neg_grad': [-0.058216, -0.059250]},
            'D': {'g': 0.704360, 'grad': [ 0.068462,  0.055309], 'neg_grad': [-0.068462, -0.055309]},
        }
    else:
        sol_modes_fixed = None
        best_sol_modes_fixed = None
        grad_sp_fixed = None
    
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
    
    def loi_fy(fym, cov=None):
        if cov is not None:
            sig_ec = cov * fym
        else:
            sig_ec = SIGMA

        dist = ot.Normal(fym, sig_ec)
        return dist
    
    def loi_fc(fcm, cov=None):
        COV_TABLE = {"C15": 0.14, "C25": 0.12, "C35": 0.09, "C45": 0.07}
        fck_eq = fcm - 8.0
        classe = min(COV_TABLE, key=lambda c: abs(int(c[1:]) - fck_eq))
        v = cov if cov is not None else COV_TABLE[classe]

        sigma_ln = np.sqrt(np.log(1 + v**2))
        mu_ln    = np.log(fcm) - 0.5 * sigma_ln**2

        dist = ot.LogNormal(mu_ln, sigma_ln, 0.0)
        return dist

    def dist_jointe():
        dist = []
        if 'fc' in params_names:
            dist.append(loi_fc(fcm, cov_fc)) 
        if 'fy' in params_names:
            dist.append(loi_fy(fym, cov_fy))
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
                    {"param": "YIELD_STRENGTH", "rebars": rebar_names},
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
                {"param": "YIELD_STRENGTH", "rebars": rebar_names},
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
            dist.append(loi_fc(fcm, cov_fc))
        if 'fy' in params_names:
            dist.append(loi_fy(fym, cov_fy))
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
        xt = np.array(U_doe)
        if not do_HF:
            SOL = [{} for _ in range(n0)] 
            for i in range(n0):
                for j in range(n_var):
                    SOL[i][params_names[j]] = X_doe[i][j]
            SOL = run_one_SOL(modelname, SOL, params_names, sensitivity=True, with_sens_dict=None)
            yt = np.array([SOL[i]['g'] for i in range(n0)]).reshape(-1, 1)
            all_grad = np.zeros((n0, n_var))
            for i in range (n0):
                J_Tinv = T_inv.gradient(U_doe[i])
                J_Tinv_T = J_Tinv.transpose()
                grad_X_g = ot.Point([SOL[i][f'dg_{p}'] for p in params_names])
                grad_U_g = J_Tinv_T * grad_X_g
                for j in range (n_var):
                    all_grad[i][j]= grad_U_g[j]
                    SOL[i][f'dg_u{j+1}'] = grad_U_g[j]
            if print_DOE:
                print("yt_doe = [")
                for i in range(n0):
                    print(f"    {yt[i][0]:.16f},")
                print("]", flush=True)
            return xt, yt, all_grad
        return xt

    # --------------------------------------------------------------------------- #
    # FONCTION ANALYTIQUE DE REFERENCE                                            #
    # --------------------------------------------------------------------------- #
    # FONCTION ANALYTIQUE CORRIGEE                                               #
    class flexion_claude:
        def __init__(self):

            # --- Lecture du dsCad et dsLoad ---
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

            # --- Définition de la transformation isoprobabiliste ---
            self.fym = fy_otparams[0]
            dist = []
            if 'fc' in params_names:
                dist.append(loi_fc(*fc_otparams))
            if 'fy' in params_names:
                dist.append(loi_fy(*fy_otparams))
            dist_X     = ot.JointDistribution(dist)
            self.T_inv = dist_X.getInverseIsoProbabilisticTransformation()
            self.T     = dist_X.getIsoProbabilisticTransformation()

            # --- Définiton des constantes pour le cas aciers plastifiés ---
            self.A  = As * d / gamma_s_fic
            self.B  = - As**2 * gamma_c_fic / (2 * b * gamma_s_fic**2)
            self.C  = -Med

            # --- Calcul de la limite plastique ---
            self.Ap = 0.8*d*b / (As*gamma_c_fic*Es*ecu)
            self.Bp = 0.8*b*d**2 / gamma_c_fic
            self.Cp = 2*self.Ap*self.C/self.Bp 
            ap = 1
            bp = self.Cp - 0.8
            cp = self.Cp - 0.2
            Delta_p = bp**2 - 4*ap*cp
            sol1_s = (-bp + Delta_p**0.5) / (2*ap)
            sol1_x1 = (sol1_s**2 - 1) / (4*self.Ap)
            self.u1_lim_plast = self.T(ot.Point([sol1_x1, 0.0]))[0]

            # --- Limite de plasticité ---
            self.A1 = As*gamma_c_fic*Es*ecu/(0.8*b*d)
            self.A2 = Es*ecu*gamma_s_fic

        def u2p_LS(self, u1):
            x_point = self.T_inv(ot.Point([u1, 0.0]))
            x1 = x_point[0]
            a  = self.B
            b  = self.A * x1
            c  = self.C * x1
            Delta = b**2 - 4 * a * c
            fy = (-b + Delta**0.5) / (2 * a)
            return self.T(ot.Point([0.0, fy]))[1]

        def g(self, u1, u2):
            x_point = self.T_inv(ot.Point([u1, u2]))
            x1 = x_point[0]
            x2 = x_point[1]
            x1_lim_plast_x2 = self.A1*x2*(self.A2+x2)/self.A2**2
            if x1 > x1_lim_plast_x2:
                return (self.A*x2+self.B*x2**2/x1+self.C)/(-self.C)
            else :
                s = (1 + 4*self.Ap*x1)**0.5
                return -1 - (s-1)/self.Cp + 0.8*(s-1)/(self.Cp*(s+1))
    
    def print_visu_ana():
        calc = flexion_claude()

        u1_lim = calc.u1_lim_plast
        u2_lim = calc.u2p_LS(u1_lim)

        # Branche plastifiée
        u1_grid = np.linspace(u1_lim, u1_max, n_grid)
        u2_grid = np.array([calc.u2p_LS(u) for u in u1_grid])

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(u1_grid, u2_grid, 'b-', lw=2,
                label=r'$u_2 = u_{2p,LS}(u_1)$  (aciers plastifiés)')
        ax.plot([u1_lim, u1_lim], [u2_lim, u2_max], 'r-', lw=2,
                label=r'$u_1 = u_{1,lim\,plast}$  (aciers non plastifiés)')
        ax.plot(u1_lim, u2_lim, 'ko', ms=6, zorder=5,
                label=f'Raccord ({u1_lim:.3f}, {u2_lim:.3f})')
        ax.plot(0, 0, 'g+', ms=12, mew=2, label='Origine')

        ax.axhline(0, color='gray', lw=0.4)
        ax.axvline(0, color='gray', lw=0.4)
        ax.set_xlabel(r'$u_1$  (espace standard, $f_c$)')
        ax.set_ylabel(r'$u_2$  (espace standard, $f_y$)')
        ax.set_title("Surface d'état-limite — flexion pivot B")
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(u1_min, u1_max)
        ax.set_ylim(u2_min, u2_max)
        plt.tight_layout()
        plt.show()
        return fig, ax

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
        n_active = result.getCoefficients().getSize()
        print(f"PCE construite : basis_size={basis_size}, coefficients actifs LARS={n_active}", flush=True)
        metamodel = result.getMetaModel()
        return metamodel

    def build_PCE(xt, y_hf, all_grad_hf, metamodel_PCE):
        U_doe = ot.Sample(xt)                               
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
        return y_PCE, all_grad_PCE

    class PCKRGFunction(ot.OpenTURNSPythonFunction):
        def __init__(self, g_pce, g_krg):
            super().__init__(n_var, 1)
            self.g_pce = g_pce
            self.g_krg = g_krg
        
        def _exec(self, u):
            return [self.g_pce(u)[0] + self.g_krg(u)[0]]

        def _exec_sample(self, U):
            U_ot = ot.Sample(U)
            Z_pce = np.array(self.g_pce(U_ot))[:, 0]
            Z_krg = np.array(self.g_krg(U_ot))[:, 0]
            return (Z_pce + Z_krg).reshape(-1, 1).tolist()

    class GEPCKFunction(ot.OpenTURNSPythonFunction):
        def __init__(self, g_pce, sm_gepck):
            super().__init__(n_var, 1)
            self.g_pce  = g_pce
            self.sm     = sm_gepck

        def _exec(self, u):
            y_pce = self.g_pce(ot.Point(list(u)))[0]
            y_gek = self.sm.predict_values(np.array(u).reshape(1, -1)).item()
            return [y_pce + y_gek]

        def _exec_sample(self, U):
            U_ot = ot.Sample(U)
            Z_pce = np.array(self.g_pce(U_ot))[:, 0]
            Z_gek = self.sm.predict_values(np.array(U))[:, 0]
            return (Z_pce + Z_gek).reshape(-1, 1).tolist()

        def _gradient(self, u):
            u_np     = np.array(u).reshape(1, -1)
            grad_pce = self.g_pce.gradient(ot.Point(list(u)))   # OT Matrix (n_var, 1)
            return [[grad_pce[i, 0] + self.sm.predict_derivatives(u_np, i).item()]
                    for i in range(n_var)]

    def eval_PCE():
        return do_pce
    # --------------------------------------------------------------------------- #
    # FONCTIONS LIEES AU KRG                                                      #

    def build_metamodel_KRG(xt, yt):
        n_var = xt.shape[1]
        basis = ot.ConstantBasisFactory(n_var).build()                  # tendance (beta0)
        # covarianceModel = ot.MaternModel([1.0] * n_var, 2.5)
        covarianceModel = ot.SquaredExponential([1.0] * n_var)          # k(x,x') 
        algo_KRG = ot.KrigingAlgorithm(xt, yt, covarianceModel, basis)
        algo_KRG.run()
        result = algo_KRG.getResult()
        metamodel = result.getMetaModel()
        return metamodel, result

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

        def _exec_sample(self, U):
            return self.sm.predict_values(np.array(U)).tolist()

        def _gradient(self, u):
            u_np = np.array(u).reshape(1, -1)
            return [[self.sm.predict_derivatives(u_np, kx).item()] for kx in range(n_var)]
    
    # --------------------------------------------------------------------------- #
    # FONCTIONS POUR FORM                                                         #
    def init_g_ot(g_ot, xt, yt, all_grad): # attention tester la nouvelle form_event version qui utilise celle la avant de tester EFF.
        """
        Cette fonction génère un DOE si xt n'est pas vide puis
        contruit un metamodele à partir de ces points. Dans le 
        cas HF, elle créé uniquement une fonction OT.
        """
        
        if do_KRG and not try_pce:
            if xt is None: xt, yt, all_grad = build_DOE()
            g_ot, result = build_metamodel_KRG(xt, yt)

        elif do_GEK and not try_pce:
            if xt is None: xt, yt, all_grad = build_DOE()
            sm_GEK = build_metamodel_GEK(xt, yt, all_grad) # variable intermédiaire
            g_ot = ot.Function(GEKPLSFunction(sm_GEK))

        elif do_KRG and try_pce:
            if xt is None: xt, y_hf, all_grad_hf = build_DOE()
            g_ot_PCE = build_metamodel_PCE(xt, y_hf)
            y_PCE, all_grad_PCE = build_PCE(xt, y_hf, all_grad_hf, g_ot_PCE)           # on calcule la composante PCE à partir des valeurs hf
            yr, all_grad_r = y_hf-y_PCE, all_grad_hf-all_grad_PCE                      # on construit le residu
            gr_ot_KRG, result_KRG = build_metamodel_KRG(xt, yr)                        # on construit le surrogate sur le residu
            g_ot = ot.Function(PCKRGFunction(g_ot_PCE, gr_ot_KRG))                     # on wrappe la somme du surrogate et du PCE
            yt, all_grad = y_hf, all_grad_hf # A REVOIR AVANT DE LANCER WARM START     # on stocke les valeurs hf pour si warmstart

        elif do_GEK and try_pce:
            if xt is None: xt, y_hf, all_grad_hf = build_DOE()
            g_ot_PCE = build_metamodel_PCE(xt, y_hf)
            y_PCE, all_grad_PCE = build_PCE(xt, y_hf, all_grad_hf, g_ot_PCE) 
            yr, all_grad_r = y_hf-y_PCE, all_grad_hf-all_grad_PCE 
            smr_GEK = build_metamodel_GEK(xt, yr, all_grad_r)
            g_ot  = ot.Function(GEPCKFunction(g_ot_PCE, smr_GEK))   
            yt, all_grad = y_hf, all_grad_hf # A REVOIR AVANT DE LANCER WARM START

        elif do_HF:
            if xt is None: xt = build_DOE()
            g_ot = ot.Function(HFFunction())
            yt, all_grad = None, None
        
        return g_ot, xt, yt, all_grad
    
    def init_FORM(g_ot, xt, yt, all_grad):
        """
        Cette fonction calcule l'évenement FORM pour
        le métamodèle donné. 
        Si xt n'est pas vide, il est initialisé avec 
        l'appel à init_g_ot. La fonction retourne 
        l'evenement, le metamodele, et xt, yt, grad. 
        """
        # --- Événement de défaillance ---
        distribution = ot.JointDistribution([ot.Normal(0, 1)] * n_var)
        X = ot.RandomVector(distribution)
        g_ot, xt, yt, all_grad = init_g_ot(g_ot, xt, yt, all_grad)
        Y = ot.CompositeRandomVector(g_ot, X) if g_ot is not None else None      
        event = ot.ThresholdEvent(Y, ot.Less(), 0.0) if Y is not None else None
        return event, g_ot, xt, yt, all_grad

    # --- Multi-start FORM depuis les points du DOE ---
    def FORM_all_modes(starting_points, tol_all_modes, event):
        """
        Multi-start FORM + DBSCAN pour identifier les modes de défaillance.
        - Chaque cluster DBSCAN = un mode distinct.
        - u* isolés (label -1) = descentes mal convergées, ignorées.
        """
        all_u_star   = []   # u* de chaque run réussi
        all_results  = []   # FORMResult correspondant
        all_sp       = []   # point de départ correspondant
        n_total = len(starting_points)

        for k, sp in enumerate(starting_points):
            print(f"  FORM {k+1}/{n_total}...", flush=True)
            try:
                solver = ot.AbdoRackwitz()
                solver.setStartingPoint(sp.tolist())
                solver.setMaximumIterationNumber(n_max_FORM)
                solver.setCheckStatus(False)
                solver.setMaximumConstraintError(tol_FORM)
                form_i = ot.FORM(solver, event)
                form_i.run()
                r_i    = form_i.getResult()
                u_star = np.array(r_i.getStandardSpaceDesignPoint())
                all_u_star.append(u_star)
                all_results.append(r_i)
                all_sp.append(sp)
                print(f"  [sp={[round(v,3) for v in np.array(sp)]}, "
                    f"u*={[round(v,3) for v in u_star]}, "
                    f"beta={r_i.getHasoferReliabilityIndex():.4f}]", flush=True)
            except Exception as e:
                print(f"  [sp={[round(v,3) for v in np.array(sp)]}, "
                    f"ECHEC ({type(e).__name__})]", flush=True)

        if not all_u_star:
            return [], []

        # --- Cas 1 point : pas de DBSCAN ---
        if len(all_u_star) == 1:
            print(f"\n1 mode(s) distinct(s) (1 seul point de depart, pas de DBSCAN) :", flush=True)
            u = [round(v, 3) for v in all_results[0].getStandardSpaceDesignPoint()]
            print(f"  mode 1 : beta={all_results[0].getHasoferReliabilityIndex():.4f}  "
                  f"Pf={all_results[0].getEventProbability():.3e}  u*={u}", flush=True)
            return [all_results[0]], [all_sp[0]]

        # --- DBSCAN ---
        U_all  = np.array(all_u_star)          # shape (n_runs_ok, n_var)
        db     = DBSCAN(eps=tol_all_modes, min_samples=2).fit(U_all)
        labels = db.labels_

        n_noise = np.sum(labels == -1)
        if n_noise > 0:
            print(f"  {n_noise} descente(s) mal convergée(s) ignorée(s) (bruit DBSCAN)", flush=True)

        # --- Un mode par cluster : FORMResult avec beta minimal ---
        modes     = []
        best_sps  = []
        for lbl in sorted(set(labels) - {-1}):
            idx_cluster = [i for i, l in enumerate(labels) if l == lbl]
            best_i = min(idx_cluster,
                        key=lambda i: all_results[i].getHasoferReliabilityIndex())
            modes.append(all_results[best_i])
            best_sps.append(all_sp[best_i])

        order = sorted(range(len(modes)), key=lambda i: modes[i].getHasoferReliabilityIndex())
        modes    = [modes[i]    for i in order]
        best_sps = [best_sps[i] for i in order]

        print(f"\n{len(modes)} mode(s) distinct(s) "
            f"(DBSCAN eps={tol_all_modes}, min_samples=2) :", flush=True)
        for i, m in enumerate(modes):
            u = [round(v, 3) for v in m.getStandardSpaceDesignPoint()]
            print(f"  mode {i+1} : beta={m.getHasoferReliabilityIndex():.4f}  "
                f"Pf={m.getEventProbability():.3e}  u*={u}", flush=True)

        return modes, best_sps
    
    # --- Warm-start FORM depuis les points du DOE ---
    def FORM_warm_start(modes, best_sps, sm_GEK, sm_GEPCK, g_ot_KRG, g_ot_GEK, g_ot_PCKRG, g_ot_GEPCK, g_ot_HF):
        if len(modes)>0:
            u_star = modes[0].getStandardSpaceDesignPoint()
            g_ot = g_ot_GEK or g_ot_KRG or g_ot_HF
            g_val = g_ot(ot.Point(u_star))[0] if g_ot is not None else None
    
            if g_val is not None and abs(g_val) > tol_warmstart:
                # -- on fait warm start uniquement si on est au dessus de 0.2, sinon, on accepte le résultat. --
                xt_ws = np.vstack([xt, [np.array(u_star)]])
                yt_ws = np.vstack([yt, [[g_val]]])
                if do_GEK or do_HF:
                    grad_ot  = g_ot.gradient(ot.Point(u_star))
                    grad_val = np.array([[grad_ot[i, 0] for i in range(n_var)]])
                    all_grad_ws = np.vstack([all_grad, grad_val])
                else:
                    all_grad_ws = all_grad
                event, g_ot, xt_ws, yt_ws, all_grad_ws = init_FORM(g_ot, xt_ws, yt_ws, all_grad_ws)
                starting_points = np.vstack([xt_ws, [[0.0, 0.0]]]) if do_multistart else np.array([[0.0, 0.0]])
                modes, best_sps = FORM_all_modes(starting_points, tol_all_modes, event)
        return modes, best_sps

    # --------------------------------------------------------------------------- #
    # FONCTIONS D'ENRICHISSEMENT DU PLAN D'EXPERIENCE (EFF)                       #
    class EFFFunction(ot.OpenTURNSPythonFunction):
        def __init__(self, result):
            super().__init__(n_var, 1)
            self.result = result

        def _exec(self, u):
            u_ot    = ot.Point(list(u))
            sigmaG  = np.sqrt(self.result.getConditionalMarginalVariance(u_ot))
            muG     = self.result.getMetaModel()(u_ot)[0]
            epsilon = epsilon_factor * sigmaG
            t1 = -muG / sigmaG
            t2 = (epsilon + muG) / sigmaG
            t3 = (epsilon - muG) / sigmaG
            return [2*muG*norm.cdf(t1) - (epsilon+muG)*norm.cdf(-t2) + (epsilon-muG)*norm.cdf(t3) + sigmaG*(norm.pdf(t2)-norm.pdf(t3))]

    
    #ARRETEE ICI : changer form_event qui utilise cette focntion voir ne plus utiliser form_event jsp ,finir de colmpleter cette fonction (ta enleve des truc en haut) et ensuite tester en commentant les fonction en cours de modif avec petit n0 et faire relire par claude, puis enfin ajouter lutilisation de cette fonction dans EFF pour le reevaluer. 

    def run_EFF(result):
        # --- Trouver u = argmax(EFF) ---
        f = ot.Function(EFFFunction(result))
        bounds = ot.Interval([u1_eff_min, u2_eff_min], [u1_eff_max, u2_eff_max])
        problem = ot.OptimizationProblem(f, ot.Function(), ot.Function(), bounds)
        problem.setMinimization(False)   # maximisation — il n'y a PAS de setMaximization(), seulement setMinimization(bool)
        algo = ot.NLopt(problem, "GN_DIRECT")          
        algo.setStartingPoint([0.0] * n_var)          
        algo.setMaximumCallsNumber(N)
        algo.run()
        u_opt = algo.getResult().getOptimalPoint()
        g_HF_u_opt = 








        # --- Trouver x = argmax ---
        # --- Recréer le metamodel (peut etre le mettre en entrée du coup?) avec DOE+x ---
        # --- Recalculer EFF et comparer au critère, on sort de la boucle while, renvoie le metamodel, une fois que tout est good, metamodel sera un objet ot (voir si pb avec celle wrapée vs non wrapée et si faut ajouter un exec sample ou quoi) ---







    # --------------------------------------------------------------------------- #
    # FONCTIONS RESULTATS/ AFFICHAGE                                              #  
    def print_results(best_result, g_ot_GEK, g_ot_KRG, g_ot_PCKRG, g_ot_GEPCK, g_ot_HF):
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

        # --- Bloc PCKRG ---
        if g_ot_PCKRG is not None:
            _, grad_HF_U_star, _ = run_HF(u_star)
            for i, p in enumerate(params_names):
                print(f"dg/du_{p} en u* (HF@u*PCKRG) = {grad_HF_U_star[i]:.6f}", flush=True)
            u0             = ot.Point([0.0] * n_var)
            g0_HF, grad_HF_U0, _ = run_HF(u0)
            u_FOSM         = grad_HF_U0 * (-g0_HF / grad_HF_U0.normSquare())
            print(f"u* FOSM (HF) = {[round(v, 4) for v in u_FOSM]}", flush=True)
            print(f"Erreur FOSM  = {(u_FOSM - u_star).norm() / u_star.norm():.4f}", flush=True)

        # --- Bloc GEPCK ---
        if g_ot_GEPCK is not None:
            _, grad_HF_U_star, _ = run_HF(u_star)
            for i, p in enumerate(params_names):
                print(f"dg/du_{p} en u* (HF@u*GEPCK) = {grad_HF_U_star[i]:.6f}", flush=True)
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

    def print_visu(best_result, best_sp, xt, g_ot_GEK, g_ot_KRG, g_ot_PCKRG, g_ot_GEPCK, modes):
        
        u1 = np.linspace(u1_min, u1_max, n_grid)
        u2 = np.linspace(u2_min, u2_max, n_grid)
        U1, U2 = np.meshgrid(u1, u2)
        grid = np.column_stack([U1.ravel(), U2.ravel()])

        fig, ax = plt.subplots(figsize=(7, 6))

        # --- Fond coloré : GEK en priorité, sinon KRG ---
        if g_ot_GEK is not None:
            grid_ot = ot.Sample(grid.tolist())
            Z_gek = np.array(g_ot_GEK(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_gek, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (GEK)')
            ax.contour(U1, U2, Z_gek, levels=[0], colors='blue', linewidths=2)
        elif g_ot_KRG is not None:
            grid_ot = ot.Sample(grid.tolist())
            Z_krg = np.array(g_ot_KRG(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_krg, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (KRG)')
        elif g_ot_PCKRG is not None:
            grid_ot = ot.Sample(grid.tolist())
            Z_pckrg = np.array(g_ot_PCKRG(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_pckrg, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (PCKRG)')
            ax.contour(U1, U2, Z_pckrg, levels=[0], colors='blue', linewidths=2)
        elif g_ot_GEPCK is not None:
            grid_ot = ot.Sample(grid.tolist())
            Z_gepck = np.array(g_ot_GEPCK(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_gepck, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (GEPCK)')
            ax.contour(U1, U2, Z_gepck, levels=[0], colors='blue', linewidths=2)

        # --- Contour KRG ---
        if g_ot_KRG is not None:
            grid_ot = ot.Sample(grid.tolist())
            Z_krg = np.array(g_ot_KRG(grid_ot))[:, 0].reshape(n_grid, n_grid)
            ax.contour(U1, U2, Z_krg, levels=[0], colors='purple', linewidths=2, linestyles=':')

        # --- Contour HF grossier ---
        if print_HF:
            u1_hf = np.linspace(u1_min, u1_max, n_grid_hf)
            u2_hf = np.linspace(u2_min, u2_max, n_grid_hf)
            U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
            grid_hf = np.column_stack([U1_hf.ravel(), U2_hf.ravel()])
            Z_true = np.array([run_HF(pt)[0] for pt in grid_hf]).reshape(n_grid_hf, n_grid_hf)
            ax.contour(U1_hf, U2_hf, Z_true, levels=[0], colors='red', linewidths=2, linestyles='--')

        # --- LS analytique (depuis flexion_claude) ---
        if print_ana:
            calc = flexion_claude()
            u1_lim_a = calc.u1_lim_plast
            u2_lim_a = calc.u2p_LS(u1_lim_a)
            u1_g_a = np.linspace(u1_lim_a, u1_max, n_grid)
            u2_g_a = np.array([calc.u2p_LS(u) for u in u1_g_a])
            ax.plot(u1_g_a, u2_g_a, color='green', linestyle='-.', linewidth=2)
            ax.plot([u1_lim_a, u1_lim_a], [u2_lim_a, u2_max],
                    color='green', linestyle='-.', linewidth=2)
            ax.plot(u1_lim_a, u2_lim_a, 'ko', ms=6, zorder=6)

        # --- Points ---
        if xt is not None:
            ax.scatter(xt[:, 0], xt[:, 1], c='black', s=30, zorder=5, label='DOE')

        ax.scatter(0, 0, c='orange', s=100, zorder=6, marker='P', label='[0, 0]')

        if best_sp is not None:
            ax.scatter(best_sp[0], best_sp[1], c='cyan', s=100, zorder=7, marker='D',
                    label='point de depart best')

        if best_result is not None:
            u_star = np.array(best_result.getStandardSpaceDesignPoint())
            ax.scatter(u_star[0], u_star[1], c='gold', s=200, zorder=8, marker='*',
                    label=f'u* mode1 beta={best_result.getHasoferReliabilityIndex():.3f}')

        if len(modes) > 0:
            for k, mode in enumerate(modes[1:], start=2):
                u_m = np.array(mode.getStandardSpaceDesignPoint())
                ax.scatter(u_m[0], u_m[1], c='magenta', s=200, zorder=8, marker='*',
                        label=f'u* mode{k} beta={mode.getHasoferReliabilityIndex():.3f}')

        # --- Points fixes (run HF précédent) ---
        if best_sol_modes_fixed is not None:
            colors_fixed = ['blue', 'red', 'green', 'gold']
            for col, (lbl, data) in zip(colors_fixed, best_sol_modes_fixed.items()):
                ustar_f = data['u*']
                sp_f    = data['sp']
                ax.scatter(ustar_f[0], ustar_f[1], c=col, s=200, zorder=9, marker='*',
                           label=f'u* {lbl}')
                ax.scatter(sp_f[0], sp_f[1], c=col, s=100, zorder=9, marker='x',
                           linewidths=2, label=f'sp {lbl}')
                if grad_sp_fixed is not None and lbl in grad_sp_fixed:
                    ng = np.array(grad_sp_fixed[lbl]['neg_grad'])
                    ng = ng / np.linalg.norm(ng) * 1.5
                    ax.quiver(sp_f[0], sp_f[1], ng[0], ng[1], color=col,
                              angles='xy', scale_units='xy', scale=1.0, width=0.005)

        # --- Légende contours ---
        legend_lines = []
        if sm_GEK is not None:
            legend_lines.append(Line2D([0], [0], color='blue',   linestyle='-',  linewidth=2, label='g=0 GEKPLS'))
        if g_ot_KRG is not None:
            legend_lines.append(Line2D([0], [0], color='purple', linestyle=':',  linewidth=2, label='g=0 KRG'))
        if print_HF:
            legend_lines.append(Line2D([0], [0], color='red',    linestyle='--', linewidth=2, label='g=0 HF'))
        if print_ana:
            legend_lines.append(Line2D([0], [0], color='green',  linestyle='-.', linewidth=2, label='g=0 ana'))

        ax.legend(handles=ax.legend().legend_handles + legend_lines)

        ax.set_xlabel('u1')
        ax.set_ylabel('u2')
        ax.set_xlim(u1_min, u1_max)
        ax.set_ylim(u2_min, u2_max)
        ax.set_title('FORM et etat limite g=0')
        plt.tight_layout()
        plt.show()

    def print_3D_HF():
        if hf_3d_grid_fixed is not None:
            print("Cache hf_3d_grid_fixed disponible — pas d'appels STRAINS.", flush=True)
            u1_min_c, u1_max_c, u2_min_c, u2_max_c, n_c = hf_3d_grid_fixed['params']
            u1_hf = np.linspace(u1_min_c, u1_max_c, n_c)
            u2_hf = np.linspace(u2_min_c, u2_max_c, n_c)
            U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
            Z = np.array(hf_3d_grid_fixed['Z'])
        else:
            u1_hf = np.linspace(u1_min, u1_max, n_grid_hf)
            u2_hf = np.linspace(u2_min, u2_max, n_grid_hf)
            U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
            grid_hf = np.column_stack([U1_hf.ravel(), U2_hf.ravel()])
            print(f"Evaluation HF grille {n_grid_hf}x{n_grid_hf} = {n_grid_hf**2} appels STRAINS...", flush=True)
            Z_flat = [run_HF(pt)[0] for pt in grid_hf]
            Z = np.array(Z_flat).reshape(n_grid_hf, n_grid_hf)

        # --- Impression copy-pastable ---
        print(f"\nhf_3d_grid_fixed = {{", flush=True)
        print(f"    'params': ({u1_min}, {u1_max}, {u2_min}, {u2_max}, {n_grid_hf}),", flush=True)
        print(f"    'Z': [", flush=True)
        for row in Z:
            vals = ', '.join(f'{v:.6f}' for v in row)
            print(f"        [{vals}],", flush=True)
        print(f"    ]", flush=True)
        print(f"}}", flush=True)

        # --- Plot 3D ---
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(U1_hf, U2_hf, Z, color='red', alpha=0.3, label='g_HF')
        ax.contour(U1_hf, U2_hf, Z, levels=[0], colors='red', linewidths=2,
                   zdir='z', offset=float(Z.min()))
        ax.contour(U1_hf, U2_hf, Z, levels=[0], colors='darkred', linewidths=2)

        # --- Surface analytique g_ana (flexion_claude) ---
        if print_ana:
            calc = flexion_claude()
            u1_a = np.linspace(u1_min, u1_max, n_grid)
            u2_a = np.linspace(u2_min, u2_max, n_grid)
            U1_a, U2_a = np.meshgrid(u1_a, u2_a)
            Z_ana = np.array([calc.g(u1, u2)
                              for u1, u2 in zip(U1_a.ravel(), U2_a.ravel())]
                             ).reshape(n_grid, n_grid)
            ax.plot_surface(U1_a, U2_a, Z_ana, color='blue', alpha=0.3, label='g_ana')
            ax.contour(U1_a, U2_a, Z_ana, levels=[0], colors='green', linewidths=2,
                       zdir='z', offset=float(Z.min()))
            ax.contour(U1_a, U2_a, Z_ana, levels=[0], colors='green', linewidths=2)

        if best_sol_modes_fixed is not None:
            for col, (lbl, data) in zip(['blue', 'red', 'green', 'gold'],
                                         best_sol_modes_fixed.items()):
                u1_f, u2_f = data['u*']
                u1_s, u2_s = data['sp']
                ax.scatter(u1_f, u2_f, 0.0, c=col, s=200, marker='*', label=f'u* {lbl}')
                ax.scatter(u1_s, u2_s, 0.0, c=col, s=100, marker='x', linewidths=2, label=f'sp {lbl}')
                if grad_sp_fixed is not None:
                    ng = grad_sp_fixed[lbl]['neg_grad']
                    ax.quiver(u1_s, u2_s, 0.0, ng[0], ng[1], 0.0,
                              color=col, length=3.0, normalize=True, arrow_length_ratio=0.3)
        ax.set_xlabel('u1 (fc)')
        ax.set_ylabel('u2 (fy)')
        ax.set_zlabel('g_HF')
        ax.set_title(f'Surface g_HF — {n_grid_hf}x{n_grid_hf} pts HF')
        ax.legend()
        plt.tight_layout()
        plt.show()

    """
    DEBUT DE CODE
    """
    # u2 = test_calc_claude(22.83)
    # print(f"u2 = {u2}", flush=True)

    event, sm_GEK, sm_GEPCK, g_ot_GEK, g_ot_GEPCK, g_ot_KRG, g_ot_PCKRG, g_ot_HF, xt, yt, all_grad = [None] * 11

    if print_3D:
        print_3D_HF()
        sys.exit(0)

    if print_grad_sp:
        print("=== -grad(g) aux points de depart sp A/B/C/D ===", flush=True)
        for lbl, data in best_sol_modes_fixed.items():
            sp = list(data['sp'])
            g_sp, grad_sp, _ = run_HF(sp)
            neg_grad = [-v for v in grad_sp]
            print(f"Mode {lbl} : sp={sp}", flush=True)
            print(f"  g_HF(sp)  = {g_sp:.6f}", flush=True)
            print(f"  grad(sp)  = [{grad_sp[0]:.6f}, {grad_sp[1]:.6f}]", flush=True)
            print(f"  -grad(sp) = [{neg_grad[0]:.6f}, {neg_grad[1]:.6f}]", flush=True)
        sys.exit(0)

    event, g_ot, xt, yt, all_grad = init_FORM(g_ot, xt, yt, all_grad)
    
    if event is None:
        if best_sol_modes_fixed is not None:
            print_visu(None, None, None, None, None, None, None, [])
            sys.exit(0)
        print('Aucune branche active', flush=True)
        sys.exit(1)
    
    if do_warmstart:
        starting_points = np.array([[0.0, 0.0]])
        modes, best_sps = FORM_all_modes(starting_points, tol_all_modes, event) #FORM simple avec event créé
        modes, best_sps = FORM_warm_start(modes, best_sps, sm_GEK, sm_GEPCK, g_ot_KRG, g_ot_GEK, g_ot_PCKRG, g_ot_GEPCK, g_ot_HF) #warm_start puis FORM multistart avec event warm
    else:
        starting_points = np.vstack([xt, [[0.0, 0.0]]]) if do_multistart else np.array([[0.0, 0.0]])
        modes, best_sps = FORM_all_modes(starting_points, tol_all_modes, event)

    best_result = modes[0] if modes else None
    best_sp     = best_sps[0] if best_sps else None
    if best_result is None:
        print('Aucun FORM ne marche.', flush=True)
        sys.exit(1)
    if len(modes)>1:
        print('On a trouvé plus de 1 mode! Les résultats du mode 2 sont:')
        print_results(modes[1], g_ot_GEK, g_ot_KRG, g_ot_PCKRG, g_ot_GEPCK, g_ot_HF)
        print('Les résultats du mode 1 sont : ')
    print_results(best_result, g_ot_GEK, g_ot_KRG, g_ot_PCKRG, g_ot_GEPCK, g_ot_HF)
    print_visu(best_result, best_sp, xt, g_ot_GEK, g_ot_KRG, g_ot_PCKRG, g_ot_GEPCK, modes)
        
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
    Anciens test (obsolete)
    """
    # def print_error_ana_hf(calc, n_scan=100):
    #     """
    #     Pour chaque u1, scanne u2 et trouve TOUS les zéros de f_ana = 0
    #     (détecte toutes les branches par changements de signe successifs).
    #     On calcule l'erreur relative sur la grille obtenue. 
    #     Utilise les globales : size_visu, fym, cov_fy, SIGMA.
    #     n_scan : résolution du scan u2 pour détecter les changements de signe.
    #     """
    #     if cov_fy is None:
    #         u2_min = -(fym / SIGMA + 1.645)
    #     else:
    #         u2_min = -1.0 / cov_fy
    #     u2_low = max(-size_visu, u2_min)

    #     pts = []
    #     u1_scan = np.linspace(-size_visu, size_visu, n_scan)
    #     for u2 in np.linspace(size_visu, u2_low, 40):  # u2 decroissant, du haut vers le bas
    #         g_vals = [calc.f_ana([u1, u2]) for u1 in u1_scan]
    #         for i in range(len(u1_scan) - 1):
    #             if g_vals[i] * g_vals[i+1] < 0:
    #                 u1_star = brentq(
    #                     lambda u1: calc.f_ana([u1, u2]), u1_scan[i], u1_scan[i+1])
    #                 if abs(calc.f_ana([u1_star, u2])) < 1e-1:
    #                     pts.append([u1_star, u2])
    #     # --- Points HF sur la frontière analytique ---
    #     frontier_pts = np.array(pts) if pts else np.zeros((0, 2))
    #     if print_pts:
    #         print(f"Points sur la frontiere analytique : {len(frontier_pts)}", flush=True)
    #         for pt in frontier_pts:
    #             print(f"  u=({pt[0]:.4f}, {pt[1]:.4f})", flush=True)
    #         return None
    #     print(f"Points sur la frontière : {len(frontier_pts)}")
    #     if len(frontier_pts):
    #         print(f"  u1 : [{frontier_pts[:,0].min():.2f}, {frontier_pts[:,0].max():.2f}]")
    #         print(f"  u2 : [{frontier_pts[:,1].min():.2f}, {frontier_pts[:,1].max():.2f}]")

    #     # --- Sélection des points HF : sous-échantillonnage uniforme ---
    #     n_hf_target = 2 * n_grid_hf
    #     step = max(1, len(frontier_pts) // n_hf_target)
    #     error_grid = frontier_pts[::step]
    #     print(f"error_grid : {len(error_grid)} points (1 sur {step})")

    #     # --- Évaluation g_HF sur les points frontière ana ---
    #     g_HF_vals    = np.array([run_HF(pt)[0]          for pt in error_grid])
    #     f_ana_vals   = np.array([calc.f_ana(list(pt))   for pt in error_grid])   # ≈ 0 par construction

    #     print("u_grid    :", [list(np.round(pt, 4)) for pt in error_grid])
    #     print("g_HF_vals :", list(np.round(g_HF_vals, 6)))
    #     print("f_ana_vals:", list(np.round(f_ana_vals, 6)))

    #     err_abs      = np.abs(g_HF_vals - f_ana_vals)

    #     dist_X = dist_jointe()
    #     T_inv  = dist_X.getInverseIsoProbabilisticTransformation()
    #     print("--- Validation g_HF vs f_ana sur frontière ---")
    #     for i, pt in enumerate(error_grid):
    #         x = T_inv(ot.Point(list(pt)))
    #         print(f"  pt {i:2d} u=({pt[0]:6.2f},{pt[1]:6.2f})  "
    #             f"fc={x[0]:6.2f}  fy={x[1]:6.2f}  "
    #             f"g_HF={g_HF_vals[i]:+.4f}  f_ana={f_ana_vals[i]:+.4f}  "
    #             f"err_abs={err_abs[i]:.4f}")
    #     print(f"  → err_abs_moy = {err_abs.mean():.4f}  (biais moyen g_HF - f_ana sur la frontiere)")
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
    #             return u2p_LS(self,u1)
    #         else :
    #             # a completer





    # if print_ana:
    #     calc  = calc_ana()
    #     f_ana = calc.f_ana
    #     if print_ana_hf_error:
    #         print_error_ana_hf(calc, n_scan=100)


    # def print_visu_claude(best_result, best_sp, xt, sm_GEK, g_ot_KRG, g_hf, modes, calc):
    #     n_grid = 100
    #     u1 = np.linspace(*u1_bornes, n_grid)
    #     u2 = np.linspace(*u2_bornes, n_grid)
    #     U1, U2 = np.meshgrid(u1, u2)
    #     grid = np.column_stack([U1.ravel(), U2.ravel()])

    #     u_star = np.array(best_result.getStandardSpaceDesignPoint())

    #     fig, ax = plt.subplots(figsize=(10, 10))

    #     # --- Fond coloré : GEKPLS en priorité, sinon KRG ---
    #     if sm_GEK is not None:
    #         Z_surr = sm_GEK.predict_values(grid).reshape(n_grid, n_grid)
    #         cf = ax.contourf(U1, U2, Z_surr, levels=20, cmap='RdYlGn', alpha=0.6)
    #         plt.colorbar(cf, ax=ax, label='g (surrogate GEKPLS)')
    #         ax.contour(U1, U2, Z_surr, levels=[0], colors='blue', linewidths=2)
    #     elif g_ot_KRG is not None:
    #         grid_ot = ot.Sample(grid.tolist())
    #         Z_krg = np.array(g_ot_KRG(grid_ot))[:, 0].reshape(n_grid, n_grid)
    #         cf = ax.contourf(U1, U2, Z_krg, levels=20, cmap='RdYlGn', alpha=0.6)
    #         plt.colorbar(cf, ax=ax, label='g (KRG)')

    #     if g_ot_KRG is not None:
    #         grid_ot = ot.Sample(grid.tolist())
    #         Z_krg = np.array(g_ot_KRG(grid_ot))[:, 0].reshape(n_grid, n_grid)
    #         ax.contour(U1, U2, Z_krg, levels=[0], colors='purple', linewidths=2, linestyles=':')

    #     # --- Contour HF grossier ---
    #     if g_hf is not None:
    #         u1_hf = np.linspace(*u1_bornes, n_grid_hf)
    #         u2_hf = np.linspace(*u2_bornes, n_grid_hf)
    #         U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
    #         grid_hf = np.column_stack([U1_hf.ravel(), U2_hf.ravel()])
    #         Z_true = np.array([g_hf(pt)[0] for pt in grid_hf]).reshape(n_grid_hf, n_grid_hf)
    #         ax.contour(U1_hf, U2_hf, Z_true, levels=[0], colors='red', linewidths=2, linestyles='--')

    #     # --- Courbe analytique flexion_claude ---
    #     if calc is not None:
    #         ax.set_xlim(-30, 30)
    #         ax.set_ylim(-30, 30)
    #         calc.print_ana(ax)

    #     # --- Points ---
    #     ax.scatter(xt[:, 0], xt[:, 1],    c='black',  s=30,  zorder=5, label='DOE')
    #     ax.scatter(0, 0,                   c='orange', s=100, zorder=6, marker='P', label='[0, 0]')
    #     ax.scatter(best_sp[0], best_sp[1], c='cyan',   s=100, zorder=7, marker='D', label='point de depart best')
    #     ax.scatter(u_star[0], u_star[1],   c='gold',   s=200, zorder=8, marker='*',
    #             label=f'u* mode1 beta={best_result.getHasoferReliabilityIndex():.3f}')
    #     if len(modes) > 0:
    #         for k, mode in enumerate(modes[1:], start=2):
    #             u_m = np.array(mode.getStandardSpaceDesignPoint())
    #             ax.scatter(u_m[0], u_m[1], c='magenta', s=200, zorder=8, marker='*',
    #                     label=f'u* mode{k} beta={mode.getHasoferReliabilityIndex():.3f}')

    #     # --- Légende contours ---
    #     legend_lines = []
    #     if sm_GEK is not None:
    #         legend_lines.append(Line2D([0], [0], color='blue',   linestyle='-',  linewidth=2, label='g=0 GEKPLS'))
    #     if g_ot_KRG is not None:
    #         legend_lines.append(Line2D([0], [0], color='purple', linestyle=':',  linewidth=2, label='g=0 KRG'))
    #     if g_hf is not None:
    #         legend_lines.append(Line2D([0], [0], color='red',    linestyle='--', linewidth=2, label='g=0 HF'))
    #     if calc is not None:
    #         legend_lines.append(Line2D([0], [0], color='green',  linestyle='-.', linewidth=2, label='g=0 ana (claude)'))

    #     ax.legend(handles=ax.legend().legend_handles + legend_lines)

    #     ax.set_xlabel('u1')
    #     ax.set_ylabel('u2')
    #     ax.set_title('FORM sur GEKPLS — courbe analytique corrigée')
    #     plt.tight_layout()
    #     plt.show()

        # def calc_ana_claude():
    #     path = os.path.join(r'C:\workspace\storage\admin\SF', modelname + '.ds')
    #     with open(os.path.join(path, 'dsCad.txt'), 'r') as f:
    #         _cad = f.read()
    #     with open(os.path.join(path, 'dsLoad.txt'), 'r') as f:
    #         _load = f.read()

    #     b   = _parse(_cad, 'b')
    #     h   = _parse(_cad, 'h')
    #     L   = _parse(_cad, 'L')
    #     phi = _parse(_cad, 'phi')

    #     n_bars = len(re.findall(r'REBAR\(', _cad))
    #     As = n_bars * math.pi * (phi / 2e3) ** 2

    #     z_rebar = [float(v) for v in re.findall(
    #         r'pts\d+\.append\(POINT\([^,]+,\s*[^,]+,\s*([\d.]+)\)', _cad)]
    #     d = h/2 + sum(z_rebar) / len(z_rebar)

    #     F = abs(float(re.search(r"Z='(-?[\d.]+)'", _load).group(1)))
    #     Med = F * L

    #     return flexion_claude(Med=Med, As=As, b=b, h=h, d=d,
    #                           fc_otparams=(fcm, cov_fc), fy_otparams=(fym, cov_fy))
    
    # def test_calc_claude(x1):
    #     calc = calc_ana_claude()
    #     u1 = calc.T(ot.Point([x1, 0.0]))[0]
    #     return calc.u2p_LS(u1)


    # def FORM_multistart(starting_points):
    #     best_beta = np.inf
    #     best_result = None
    #     best_sp = None

    #     for sp in starting_points:
    #         try:
    #             solver = ot.AbdoRackwitz()
    #             solver.setStartingPoint(sp.tolist())
    #             solver.setMaximumIterationNumber(n_max_FORM)
    #             solver.setCheckStatus(False)
    #             solver.setMaximumConstraintError(tol_FORM)
    #             form_i = ot.FORM(solver, event)
    #             form_i.run()
    #             r_i = form_i.getResult()
    #             if r_i.getHasoferReliabilityIndex() < best_beta:
    #                 best_beta = r_i.getHasoferReliabilityIndex()
    #                 best_result = r_i
    #                 best_sp = sp
    #         except Exception as e:
    #             print(f"FORM exception: {type(e).__name__}: {e}")
    #     return best_result, best_sp


    # def build_residu(xt, y_hf, all_grad_hf, metamodel_PCE): #a appeler que dans le bloc do_pce, donc pas de do_pce en param.
    #         U_doe = ot.Sample(xt)                               # a modifier pour remettre le calcul de all_sensib_hf dans build_doe puis ici
    #         y_PCE = np.array(metamodel_PCE(U_doe))
    #         n_var = U_doe.getDimension()
    #         n0 = U_doe.getSize()
    #         dist_X = dist_jointe()
    #         T = dist_X.getIsoProbabilisticTransformation()
    #         T_inv = dist_X.getInverseIsoProbabilisticTransformation()
    #         all_grad_PCE = np.zeros((n0, n_var))
    #         # all_sensib_PCE = np.zeros((n0, n_var))
    #         for i in range(n0):
    #             grad_pce_u = metamodel_PCE.gradient(U_doe[i])       
    #             for j in range(n_var):
    #                 all_grad_PCE[i, j] = grad_pce_u[j, 0]
    #             # sensibilités
    #             x_i = T_inv(ot.Sample([U_doe[i]]))[0]               
    #             J_T_i = T.gradient(x_i)                             
    #             grad_u_i = ot.Point(all_grad_PCE[i, :])             
    #             grad_x_i = J_T_i * grad_u_i                         
    #             # all_sensib_PCE[i, :] = np.array(grad_x_i)              
    #         return y_hf-y_PCE, all_grad_hf-all_grad_PCE




    # def init_FORM(sm_GEK, sm_GEPCK, g_ot_KRG, g_ot_GEK, g_ot_PCKRG, g_ot_GEPCK, g_ot_HF, xt, yt, all_grad):
    #     # --- Événement de défaillance ---
    #     distribution = ot.JointDistribution([ot.Normal(0, 1)] * n_var)
    #     X = ot.RandomVector(distribution)
    #     Y = None

    #     if do_KRG and not try_pce:
    #         if xt is None: xt, yt, all_grad = build_DOE()
    #         g_ot_KRG, result_KRG = build_metamodel_KRG(xt, yt)
    #         Y = ot.CompositeRandomVector(g_ot_KRG, X)

    #     elif do_GEK and not try_pce:
    #         if xt is None: xt, yt, all_grad = build_DOE()
    #         sm_GEK = build_metamodel_GEK(xt, yt, all_grad)
    #         g_ot_GEK = ot.Function(GEKPLSFunction(sm_GEK))
    #         Y = ot.CompositeRandomVector(g_ot_GEK, X)

    #     elif do_KRG and try_pce:
    #         if xt is None: xt, y_hf, all_grad_hf = build_DOE()
    #         g_ot_PCE = build_metamodel_PCE(xt, y_hf)
    #         y_PCE, all_grad_PCE = build_PCE(xt, y_hf, all_grad_hf, g_ot_PCE)           # on calcule la composante PCE à partir des valeurs hf
    #         yr, all_grad_r = y_hf-y_PCE, all_grad_hf-all_grad_PCE                      # on construit le residu
    #         gr_ot_KRG, result_KRG = build_metamodel_KRG(xt, yr)                                    # on construit le surrogate sur le residu
    #         g_ot_PCKRG = ot.Function(PCKRGFunction(g_ot_PCE, gr_ot_KRG))               # on wrappe la somme du surrogate et du PCE
    #         yt, all_grad = y_hf, all_grad_hf # A REVOIR AVANT DE LANCER WARM START     # on stocke les valeurs hf pour si warmstart
    #         Y = ot.CompositeRandomVector(g_ot_PCKRG, X)

    #     elif do_GEK and try_pce:
    #         if xt is None: xt, y_hf, all_grad_hf = build_DOE()
    #         g_ot_PCE = build_metamodel_PCE(xt, y_hf)
    #         y_PCE, all_grad_PCE = build_PCE(xt, y_hf, all_grad_hf, g_ot_PCE) 
    #         yr, all_grad_r = y_hf-y_PCE, all_grad_hf-all_grad_PCE 
    #         smr_GEK = build_metamodel_GEK(xt, yr, all_grad_r)
    #         g_ot_GEPCK  = ot.Function(GEPCKFunction(g_ot_PCE, smr_GEK))   
    #         yt, all_grad = y_hf, all_grad_hf # A REVOIR AVANT DE LANCER WARM START
    #         Y = ot.CompositeRandomVector(g_ot_GEPCK, X)

    #     elif do_HF:
    #         if xt is None: xt = build_DOE()
    #         g_ot_HF = ot.Function(HFFunction())
    #         Y = ot.CompositeRandomVector(g_ot_HF, X)
    #         yt, all_grad = None, None
        
    #     event = ot.ThresholdEvent(Y, ot.Less(), 0.0) if Y is not None else None
    #     return event, sm_GEK, sm_GEPCK, g_ot_KRG, g_ot_GEK, g_ot_PCKRG, g_ot_GEPCK, g_ot_HF, xt, yt, all_grad
    """
    options supprimées
    """
    # do_visu=True
    # do_GP_linear_test = True
    # do_GP_HF_test = True
    # do_pce_eval = False
    #  #NPO CHANGER POUR METTRE PARAMETRE
