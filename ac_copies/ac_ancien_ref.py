"""
Test de validation de la sensibilite IPM en traction pure beton.

Solution analytique :
  Pour un bloc en traction uniaxiale uniforme, la charge limite est
  F_max = FT * A ou A = b*h.
  Donc alpha+ = FT * A / F_ref ou F_ref est la charge appliquee unitaire.
  D'ou : d(alpha+)/d(FT) = alpha+ / FT (lineaire en FT, ratio constant).

On compare 3 valeurs :
  1. Adjoint    : d(alpha+)/d(FT) calcule par SensSolve
  2. Diff finie : (alpha+(FT+h) - alpha+(FT-h)) / (2h)  -- centree, O(h^2)
  3. Analytique : alpha+ / FT
"""
"""
Je pense qu'on devrait donner une liste de variables autorisées à modifier pour les 'questions utilisateur'
Ou alors, il demande l'étude d'une variable et on répond non si elle n'est pas dans notre liste pré-définie 
sur laquelle on a calculé les dérivées/ les distributions et on sait faire l'étude. 
Pour l'instant je vais créer une classe qu'on utilisera; peut être qu'on peut mettre dans cette classe pour mettre run_one dedans? jsp 
Ou alors je créé juste un dictionnaire, et on le remplit avec les éléments, et on cherche dans ce dictionnaire. 
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

def run_one_SOL(modelname, SOL, params_names, sensitivity=False, with_sens_dict=None): #on entre les valeurs de fc et fy dans params.
    """Lance un calcul complet pour une valeur de FT donnee.
    Retourne la liste des solutions pour chaque jeu de variables dans SOL (liste de dictionnaire)"""
    path = "C:\\workspace\\storage\\semia\\" + modelname + ".ds"
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
                {"param": "YIELD_STRENGTH", "rebars": ["HA1", "HA2", "HA3", "HA4"]},
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

def run_HF(modelname, u, params_names,T_inv, sensitivity=True, with_sens_dict=None):
    n_var = len(u)
    u_point = ot.Point(u)
    x_point = T_inv(u_point)
    """Lance un calcul complet pour une valeur de X donnée. 
    Retourne la fonction de performance évaluée en X (liste de dictionnaire)"""
    path = "C:\\workspace\\storage\\semia\\" + modelname + ".ds"
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
            {"param": "YIELD_STRENGTH", "rebars": ["HA1", "HA2", "HA3", "HA4"]},
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
    return g_HF, grad_HF_U


import openturns as ot
import numpy as np
from smt.surrogate_models import GEKPLS
from config.jcss_fy import loi_fy
from config.jcss_fc import loi_fc

"""
Note importante pour Agnes et Xavier: 
Dans la suite du code, je défini do_pce, do_GP, et do_GEK pour qu'on puisse choisir le type
de modèle, mais j'utilise dans les boucles if les memes noms de variables (ex: myFunction et result)
Il ne devrait pas y avoir de probleme mais si vous pensez que c'est mieux de les nommer différemment
je change ça (risque d'utilisation de la mauvaise variable si j'ai oublié de mettre dans le bloc if)
Note pour moi : eventuellement renommer les fonction au dessus pour dire qu'une crée un dictionnaire genre fill_sol ou quoi. L'autre devrait s'appeler run_HF. 
Ca se trouve ton truc est pas du tout optimal et tu peux utiliser celle la dans run_one_SOL en appelant les params_names... tout ca a revoir plus tard. 
"""
if __name__ == '__main__':
    modelname = "test_pure_flexion"

    print("=" * 70)
    print("VALIDATION SENSIBILITE -- FLEXION PURE BETON")
    print("=" * 70)

    # --------------------------------------------------------------------------- #
    # OPTIONS FORM                                                                #
    n_max_FORM = 50
    tol_FORM = 0.2
    do_warm_start = False
    tol_warm_start = 0.0001
    n_multi_start = 1
    # --------------------------------------------------------------------------- #
    # OPTIONS MODELE                                                              #
    # 1. GP 
    do_GP = True
    n0 = max(15,n_multi_start) # pour avoir n0>=n_multi_start

    # 2. GEK
    do_GEK = True
    do_GEK_analytic_grad = True
    reduc_PLS = 0

    # 3. PCE                                                                
    try_pce = False
    do_pce = False
    seuil_pce = 0.90
    min_max_degree = 1
    # --------------------------------------------------------------------------- #   
    # OPTIONS TEST 
    do_visu=False
    do_linear_test = True
    do_GP_test = True
    do_pce_eval = True
    # --------------------------------------------------------------------------- #
    

    # --------------------------------------------------------------------------- #
    # ETAPE 0 - CONFIGURATION ET TRANSFORMATION ISOPROBABILISTE                   #
    # --------------------------------------------------------------------------- #
    # 1. Définition des paramètres et donc des colonnes de RESULTAT et de sa taille et après on remplira avec LHS dans chaque colonne
    params_names = ['fc','fy'] #peut être le résultat de l'output utilisateur (on lui demande ce qu'il veut changer) et ensuite peut etre utilisé pour : la creation du dictionnaire et en particulier le nom de chaque mot du dict, et aussi ci dessous pour générer le DOE, pour générer les dist etc.)
    n_var = len(params_names)
    # 2. Densité jointe espace X
    #OPTIMISATION POSSIBLE / la partie ci dessous peut eventuellement etre optimisée pour faire une boucle sur les paramètres ou alors on ajoute 30 autres 'if'.
    with open("C:\\workspace\\storage\\semia\\" + modelname + ".ds\\dsCad.txt", 'r') as f:
        d = float(re.search(r'^phi\s*=\s*([\d.]+)', f.read(), re.MULTILINE).group(1))
    dist = []
    if 'fc' in params_names:
        dist.append(loi_fc("C35", t=28, tau=1e6)) #je les nommes aussi fc et fy par précaution mais je pense que je peux les nommer dist_fc, dist_fy. 
    if 'fy' in params_names:
        dist.append(loi_fy(d=d, fy=500))
    #AJOUTER suite pour plus de variable 'if 'load' in params_names' etc.
    dist_X   = ot.JointDistribution(dist) #ca peut etre utile ici. On créé la distribution jointe de tout le monde. Peut etre qu'on peut créer un dictionnaire de pdf (proba density function) associée à chaque fc, fy etc. )

    # 3. Transformation isoprobabiliste et densit� jointe espace U -----#
    T     = dist_X.getIsoProbabilisticTransformation() # on interroge dist_X et trouve la transfo n�cessaire puis l'applique ici
    T_inv = dist_X.getInverseIsoProbabilisticTransformation()
    dist_U = dist_X.getStandardDistribution()


    # --------------------------------------------------------------------------- #
    # ETAPE 1 - CONSTRUCTION DU DOE INTIAL                                        #
    # --------------------------------------------------------------------------- #
    # 1. TIRAGE DANS ESPACE U
    # lhs    = ot.LHSExperiment(dist_U, n0)
    # sa     = ot.SimulatedAnnealingLHS(lhs, ot.SpaceFillingMinDist())
    # U_doe  = sa.generate()
    U_doe = ot.Sample([
        [ 1.0272625484832025,  0.3251235065050853],
        [ 0.2588934150948534, -1.6856336900013655],
        [-0.7900915845657982,  1.8047217395005692],
        [-0.0301755082064849,  1.3223984111477798],
        [-1.8073810055112547, -1.1012751718677385],
        [-0.2377471223963969, -0.4914312425631510],
        [ 0.7216266145109314,  1.0830320538875535],
        [ 0.4776729449462015, -0.2656508781535193],
        [-0.8730465106774573,  0.6497494474356423],
        [-1.1677174906609287,  0.0310652111349381],
        [ 1.1194425579629474, -0.7943643305093363],
        [ 0.1857520921586401,  0.4724170659386679],
        [-0.5669380193636159, -1.4858232340964801],
        [ 2.9454553139272623, -0.1582987245612891],
        [-0.2947626989079067,  0.1355018527305618],
    ])

    print(f"DOE U : {[list(U_doe[i]) for i in range(n0)]}")

    # 2. CALCULS HF DANS ESPACE X
    X_doe  = T_inv(U_doe)
    SOL = [{} for _ in range(n0)] #on ajoute ensuite fc, fy avec plan DOE. Il faut dabord transformer Xdoe ou alors extraire jsp
    for i in range(n0):
        for j in range(n_var):
            SOL[i][params_names[j]] = X_doe[i][j]
    
    if do_GP:
        SOL = run_one_SOL(modelname, SOL, params_names, sensitivity=True, with_sens_dict=None)
        g_ref, dg_adj_fc, dg_adj_fy = [SOL[i]['g'] for i in range(n0)], [SOL[i]['dg_fc'] for i in range(n0)], [SOL[i]['dg_fy'] for i in range(n0)]

        print("\n" + "=" * 70)
        print("RESULTATS")
        print("=" * 70)
        print(f"alpha+(params={params_names}) = {g_ref}")
        print(f"")
        print(f"  Adjoint     : d(g+)/d(FC) = {dg_adj_fc}")
        print(f"  Adjoint     : d(g+)/d(FY) = {dg_adj_fy}")
        print(f"")
        SOL_save = SOL
    
    # 3. GRADIENT PAR RAPPORT A U
        all_grad_U_g = np.zeros((n0, n_var))
        for i in range (n0):
            J_Tinv = T_inv.gradient(U_doe[i])
            J_Tinv_T = J_Tinv.transpose()
            grad_X_g = ot.Point([SOL[i][f'dg_{p}'] for p in params_names])
            grad_U_g = J_Tinv_T * grad_X_g
            for j in range (n_var):
                all_grad_U_g[i][j]= grad_U_g[j]
                SOL[i][f'dg_u{j+1}'] = grad_U_g[j]
        print(f"  Adjoint     : d(g+)/d(U) = {all_grad_U_g}")

    # --------------------------------------------------------------------------- #
    # ETAPE 3.1 - CONSTRUCTION DU PCE                                             #
    # --------------------------------------------------------------------------- #
    def result_PCE(inputSample, outputSample, dist_U, q, max_degree, seuil_pce, min_max_degree = min_max_degree):
        # 1. INITIALISATION (BASE DE CANDIDATS : TYPE, ENUMERATION, DEGRE)
        n_var = inputSample.getDimension()
        enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(n_var, q)
        basis = ot.OrthogonalProductPolynomialFactory([ot.HermiteFactory()] * n_var, enumerateFunction)
        basis_size = enumerateFunction.getBasisSizeFromTotalDegree(max_degree)
        basisStrategy = ot.FixedStrategy(basis, basis_size)

        # 2. PROPOSITION / PROJECTION / SELECTION
        selectionStrategy = ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut())
        projectionStrategy = ot.LeastSquaresStrategy(selectionStrategy) 

        # 3. RESULTAT
        algo = ot.FunctionalChaosAlgorithm(inputSample, outputSample, dist_U, basisStrategy, projectionStrategy)
        algo.run()
        result = algo.getResult()         
        return result
    
    # def compute_q2_loo(inputSample, outputSample, dist_U, q, max_degree, seuil_pce):
    #     n = inputSample.getSize()
    #     y = outputSample[:, 0]
    #     y_mean = y.computeMean()[0]

    #     sse = 0.0
    #     sst = sum((y[i][0] - y_mean) ** 2 for i in range(n))

    #     for train_idx, test_idx in ot.LeaveOneOutSplitter(n):
    #         i = int(test_idx[0])
    #         result_trained = result_PCE(inputSample[train_idx],
    #                             outputSample[train_idx],
    #                             dist_U, q, max_degree, seuil_pce)
    #         y_pred = result_trained.getMetaModel()(inputSample[test_idx])[0, 0]
    #         sse += (y[i][0] - y_pred) ** 2

    #     return 1.0 - sse / sst

    if try_pce:
        inputSample = U_doe
        outputSample = ot.Sample([[SOL[i]['g']] for i in range(n0)])
        q = 0.75
        max_degree = 2
        result = result_PCE(inputSample, outputSample, dist_U, q, max_degree, seuil_pce , min_max_degree = min_max_degree)
        VALIDATION = {}
        # q2_loo = compute_q2_loo(inputSample, outputSample, dist_U, q, max_degree, seuil_pce)
        # while q2_loo < seuil_pce and max_degree > min_max_degree:
        #     max_degree -= 1
        #     result = result_PCE(inputSample, outputSample, dist_U, q, max_degree, seuil_pce , min_max_degree = min_max_degree)
        #     q2_loo = compute_q2_loo(inputSample, outputSample, dist_U, q, max_degree, seuil_pce)
        # if q2_loo > seuil_pce:
        if try_pce:
        # 4. VALIDATION DOE
            # VALIDATION['PCE']=True
            # print(f"Le PCE est de bonne qualité avec un degré maximum de {max_degree}.")
            do_pce = True
            metamodel_pce = result.getMetaModel()
            y_pce = np.array(metamodel_pce(U_doe))

            all_grad_PCE = np.zeros((n0, n_var))
            for i in range(n0):
                # On récupère le point i dans l'espace U
                point_U = U_doe[i]
                grad_pce_u = metamodel_pce.gradient(point_U)

                for j in range(n_var):
                    all_grad_PCE[i, j] = grad_pce_u[j, 0]
        # else:
        #     VALIDATION['PCE']=False
        #     print("Attention : Q2 faible. Le métamodèle est fait avec GP pur.")
        #     do_pce = False

    # --------------------------------------------------------------------------- #
    # ETAPE 3.2 - CONSTRUCTION DU MÉTAMODÈLE HYBRIDE                              #
    # --------------------------------------------------------------------------- #

    if do_GP:
        xt = np.array(U_doe)
        y_hf = np.array([SOL[i]['g'] for i in range(n0)]).reshape(-1, 1)
        yt = y_hf
        all_grad = all_grad_U_g
        if do_pce:
            yt-= y_pce
            all_grad -= all_grad_PCE
        
        if do_GEK:
            n_comp = n_var - reduc_PLS
            sm = GEKPLS(
                theta0=[1e-2] * n_comp,
                n_comp=n_comp,
                corr='squar_exp',
                poly='constant',
                print_global=False
            )
            sm.set_training_values(xt, yt)
            for j in range(n_var):
                sm.set_training_derivatives(xt, all_grad[:, j].reshape(-1, 1), j)
            sm.train()

            def metamodel_GEK(u, do_pce=do_pce): 
                """
                u: point ou échantillon dans l'espace standard U
                Retourne le metamodel évalué en u (hybride uniquement si do_pce=True)
                """
                u_np = np.array(u).reshape(1, -1)  
                val = float(sm.predict_values(u_np)[0,0])
                grad = np.array([[float(sm.predict_derivatives(u_np, kx)[0, 0])] for kx in range(n_var)])
                if do_pce:
                    u_sample_ot = ot.Sample(u_np)
                    u_point_ot = ot.Point(u_np[0])
                    val += float(np.array(metamodel_pce(u_sample_ot))[0,0])
                    grad += np.array(metamodel_pce.gradient(u_point_ot))
                return val, grad   

            def g_GEK(u):
                val, _ = metamodel_GEK(u, do_pce=do_pce)
                return [val]
            def grad_g_GEK(u):
                _,grad = metamodel_GEK(u, do_pce=do_pce)
                return grad
        else:
            basis = ot.ConstantBasisFactory(n_var).build()
            # covarianceModel = ot.MaternModel([1.0] * n_var, 2.5)
            covarianceModel = ot.SquaredExponential([1.0] * n_var)
            algo_KRG = ot.KrigingAlgorithm(xt, yt, covarianceModel, basis)
            algo_KRG.run()
            result = algo_KRG.getResult()
            metamodel_KRG = result.getMetaModel()
            if do_pce:
                metamodel_KRG+=metamodel_pce
    else:
        class HFCache:
            def __init__(self):
                self._last_u = None
                self._last_g = None
                self._last_grad = None

            def run_if_needed(self, u):
                u_list = list(u)
                if self._last_u is None or u_list != self._last_u:
                    self._last_g, self._last_grad = run_HF(
                        modelname, u, params_names, T_inv
                    )
                    self._last_u = u_list
        
        hf_cache = HFCache()

        def func(u):
            hf_cache.run_if_needed(u)
            return [hf_cache._last_g]

        grad_call_count = [0]

        def grad_func(u):
            grad_call_count[0] += 1
            print(f"[GRAD] appel #{grad_call_count[0]} en u={list(u)}", flush=True)
            hf_cache.run_if_needed(u)
            return [[v for v in hf_cache._last_grad]]

    if do_GP and do_GEK:
        # ---------- validation gradient analytique vs FD ---------- #
        import numpy as np

        u_test = np.array([-1.2, -3.0])          # point hors DOE, à changer si besoin
        h = 1e-4                                  # pas FD centré

        grad_ana = grad_g_GEK(u_test)            # shape (n_var, 1)

        grad_fd = []
        for i in range(n_var):
            e = np.zeros(n_var)
            e[i] = h
            gp = g_GEK(u_test + e)[0]
            gm = g_GEK(u_test - e)[0]
            grad_fd.append((gp - gm) / (2 * h))

        print("\n=== Validation gradient GEK ===")
        print(f"Point test u = {u_test.tolist()}")
        print(f"{'Var':<6} {'Analytique':>14} {'FD centré':>14} {'Err rel':>12}")
        for i in range(n_var):
            ana = grad_ana[i][0]
            fd  = grad_fd[i]
            err = abs(ana - fd) / (abs(fd) + 1e-12)
            print(f"u_{i:<4} {ana:>14.6e} {fd:>14.6e} {err:>11.2%}")
        print("================================\n")
        
    # --------------------------------------------------------------------------- #
    # ETAPE 4 - FORM SUR LE METAMODELE                                            #
    # --------------------------------------------------------------------------- #

    # Wrapper OT : prend un ot.Point u, retourne [g(u)]
    # Le gradient est calculé automatiquement par différences finies par OT

    if do_GP:
        if do_GEK:
            if do_GEK_analytic_grad:
                myFunction = ot.PythonFunction(
                    n_var,
                    1,
                    g_GEK,
                    gradient=grad_g_GEK
                )
            else:
                myFunction = ot.PythonFunction(
                    n_var,
                    1,
                    g_GEK
                )
            vect   = ot.RandomVector(dist_U)
            output = ot.CompositeRandomVector(myFunction, vect)
            event  = ot.ThresholdEvent(output, ot.Less(), 0.0)  # g < 0 = failure
            solver = ot.AbdoRackwitz()
            solver.setMaximumIterationNumber(n_max_FORM)
            solver.setCheckStatus(False)
            solver.setMaximumConstraintError(tol_FORM)
            solver.setStartingPoint([0.0] * n_var)  # point de depart = moyenne en U
            algo = ot.FORM(solver, event)
            algo.run()
            result = algo.getResult()

            U_warm = result.getPhysicalSpaceDesignPoint()
            if do_warm_start and float(metamodel_GEK(U_warm)[0]) > tol_warm_start: 
            # 1. mise a jour de xt, yt et all_grad
                U_doe.add(U_warm)
                print(f"Warm start lancé avec point de départ U={list(U_warm)}")
                xt = np.array(U_doe)
                y_to_add, all_grad_to_add = run_HF(modelname, U_warm, params_names, T_inv, sensitivity=True)
                y_hf = np.vstack([y_hf, [[y_to_add]]])
                yt = y_hf
                all_grad_U_g = np.vstack([all_grad_U_g, [all_grad_to_add]])  
                all_grad = all_grad_U_g  
                if do_pce:
                    yt-= y_pce
                    all_grad -= all_grad_PCE
            # 2. mise à jour du metamodel
                n_comp = n_var
                sm = GEKPLS(
                    theta0=[1e-2] * n_comp,
                    n_comp=n_comp,
                    corr='squar_exp',
                    poly='constant',
                    print_global=False
                )
                sm.set_training_values(xt, yt)
                for j in range(n_var):
                    sm.set_training_derivatives(xt, all_grad[:, j].reshape(-1, 1), j)
                sm.train()
            # 3. mise à jour du result FORM - FORM fait avec warm_start
                if do_GEK_analytic_grad:
                    myFunction = ot.PythonFunction(n_var, 1, g_GEK, gradient=grad_g_GEK)
                else:
                    myFunction = ot.PythonFunction(n_var, 1, g_GEK)
                output = ot.CompositeRandomVector(myFunction, vect)
                event  = ot.ThresholdEvent(output, ot.Less(), 0.0)  # g < 0 = failure
                solver = ot.AbdoRackwitz()
                solver.setMaximumIterationNumber(n_max_FORM)
                solver.setCheckStatus(False)
                solver.setMaximumConstraintError(tol_FORM)
                solver.setStartingPoint(U_warm)  
                algo = ot.FORM(solver, event)
                algo.run()
                result = algo.getResult()
        
        else:
            vect = ot.RandomVector(dist_U)
            output = ot.CompositeRandomVector(metamodel_KRG, vect)
            event = ot.ThresholdEvent(output, ot.Less(), 0.0)
            solver = ot.AbdoRackwitz()
            solver.setMaximumIterationNumber(n_max_FORM)
            solver.setCheckStatus(False)
            solver.setMaximumConstraintError(tol_FORM)
            solver.setStartingPoint([0.0] * n_var)
            algo = ot.FORM(solver, event)
            algo.run()
            result = algo.getResult()

            U_warm = result.getPhysicalSpaceDesignPoint()
            if do_warm_start and float(metamodel_KRG(U_warm)[0]) > tol_warm_start: 
            # 1. mise a jour de xt et yt
                U_doe.add(U_warm)
                print(f"Warm start lancé avec point de départ U={list(U_warm)}")
                xt = np.array(U_doe)
                y_to_add, _ = run_HF(modelname, U_warm, params_names, T_inv, sensitivity=False)
                y_hf = np.vstack([y_hf, [[y_to_add]]])                
                yt = y_hf
                if do_pce:
                    yt-= y_pce
            # 2. mise à jour du metamodel
                basis = ot.ConstantBasisFactory(n_var).build()
                covarianceModel = ot.MaternModel([1.0] * n_var, 2.5)
                algo_KRG = ot.KrigingAlgorithm(xt, yt, covarianceModel, basis)
                algo_KRG.run()
                result_KRG = algo_KRG.getResult()
                metamodel_KRG = result_KRG.getMetaModel()
            # 3. mise à jour du result FORM - FORM fait avec warm_start
                output = ot.CompositeRandomVector(metamodel_KRG, vect)
                event = ot.ThresholdEvent(output, ot.Less(), 0.0)
                solver = ot.AbdoRackwitz()
                solver.setMaximumIterationNumber(n_max_FORM)
                solver.setCheckStatus(False)
                solver.setMaximumConstraintError(tol_FORM)
                solver.setStartingPoint(U_warm)
                algo = ot.FORM(solver, event)
                algo.run()
                result = algo.getResult()

    else:
        myFunction = ot.PythonFunction(
            n_var,
            1,
            func,
            gradient=grad_func
        )

        vect   = ot.RandomVector(dist_U)
        output = ot.CompositeRandomVector(myFunction, vect)
        event  = ot.ThresholdEvent(output, ot.Less(), 0.0)

        solver = ot.AbdoRackwitz()
        solver.setMaximumIterationNumber(n_max_FORM)
        solver.setCheckStatus(False)
        solver.setStartingPoint([0.0] * n_var)

        algo = ot.FORM(solver, event)
        algo.run()
        result = algo.getResult()
        result_modes = [result]

        if n_multi_start >1:
            """
            On lance n_multi_start<=n0 autres fois le FORM. On s'arrête dès
            qu'on a trouvé un point différent du précédent pour KRG à
            adapter car le point sera différent par imprécision, choix
            tol).
            """
            norms = np.array([np.linalg.norm(np.array(U_doe[i])) for i in range(n0)])
            sorted_idx = np.argsort(norms)[::-1]  # indices ordre décroissant
            U_doe_sorted = ot.Sample([U_doe[int(i)] for i in sorted_idx])
            for n_FORM in range(n_multi_start):
                solver.setStartingPoint(U_doe_sorted[n_FORM])
                algo = ot.FORM(solver, event)
                algo.run()
                u_new  = algo.getResult().getPhysicalSpaceDesignPoint()
                u_prev = result_modes[-1].getPhysicalSpaceDesignPoint()
                if (u_new - u_prev).norm() > 1e-3:
                    result_modes.append(algo.getResult())
                    break
            #choix du mode à étudier
            mode_number = 0 #on affiche le résultat du mode trouvé en premier.
            result = result_modes[mode_number]
  
    opt_result = result.getOptimizationResult()
    n_iter = opt_result.getIterationNumber()
    print(f"Nombre d'itérations FORM : {n_iter}")
    U_res = result.getPhysicalSpaceDesignPoint()
    X_res = T_inv(U_res)
    for i in range(n_var):
        print(f"Design point in physical space for {params_names[i]}: {X_res[i]:.4f}")
    if do_GP:
        if do_GEK:
            importance = result.getImportanceFactors()
            for i in range(n_var):
                print(f"  Importance factor {params_names[i]}: {importance[i]:.4f}")
        else:
            grad_star = metamodel_KRG.gradient(U_res)
            for i in range(n_var):
                print(f"  dg/du_{params_names[i]} en u* = {grad_star[i, 0]:.6f}")
            importance = result.getImportanceFactors()
            for i in range(n_var):
                print(f"  Importance factor {params_names[i]}: {importance[i]:.4f}")
    else:
        grad_star = myFunction.gradient(U_res)
        for i in range(n_var):
            print(f"  dg/du_{params_names[i]} en u* = {grad_star[i, 0]:.6f}")
        importance = result.getImportanceFactors()
        for i in range(n_var):
            print(f"  Importance factor {params_names[i]}: {importance[i]:.4f}")
    beta    = result.getHasoferReliabilityIndex()
    Pf_FORM = result.getEventProbability()
    print()
    print(f"\nBeta FORM = {beta:.6f}", flush=True)
    print(f"Pf FORM   = {Pf_FORM:.6e}", flush=True)

    # --------------------------------------------------------------------------- #
    # ETAPE 5 - AFFICHAGE DES RESULTATS ET TESTS                                  #
    # --------------------------------------------------------------------------- #
    if do_linear_test:
        u_star = U_res
        u0 = ot.Point([0.0] * n_var)
        g0, grad_U_0 = run_HF(modelname, u0, params_names, T_inv, sensitivity=True)
        norm_sq = grad_U_0.norm() ** 2
        u_star_FOSM = grad_U_0 * (-g0 / norm_sq)
        relative_error = (u_star_FOSM - u_star).norm() / u_star.norm()
        print(f"\nTest linéarisation :")
        print(f"  u* FORM = {u_star}", flush=True)
        print(f"  u* FOSM = {u_star_FOSM}", flush=True)
        print(f"  Erreur relative entre u* FORM et u* FOSM : {relative_error:.4f}", flush=True)

    if try_pce and do_pce_eval:
        def compute_q2_loo(inputSample, outputSample, dist_U, q, max_degree, seuil_pce):
            n = inputSample.getSize()
            y = outputSample[:, 0]
            y_mean = y.computeMean()[0]

            sse = 0.0
            sst = sum((y[i][0] - y_mean) ** 2 for i in range(n))

            for train_idx, test_idx in ot.LeaveOneOutSplitter(n):
                i = int(test_idx[0])
                result_trained = result_PCE(inputSample[train_idx],
                                    outputSample[train_idx],
                                    dist_U, q, max_degree, seuil_pce)
                y_pred = result_trained.getMetaModel()(inputSample[test_idx])[0, 0]
                sse += (y[i][0] - y_pred) ** 2

            return 1.0 - sse / sst
        
        inputSample = U_doe
        outputSample = ot.Sample([[SOL[i]['g']] for i in range(n0)])
        q = 0.75
        max_degree = 2
        q2_loo = compute_q2_loo(inputSample, outputSample, dist_U, q, max_degree, seuil_pce)
        print(f"  g* HF  = {q2_loo:.3f}")

    if do_GP_test and do_GP:
        if do_GEK:
            g_GEK_star, _ = metamodel_GEK(U_res)
            g_HF, _ = run_HF(modelname, U_res, params_names, T_inv, sensitivity=False)
            print(f"\nTest GEK au point de FORM :")
            print(f"  g* HF  = {g_HF:.6f}")
            print(f"  g* GEK = {g_GEK_star:.6f}")
            print(f"  Erreur relative : {abs(g_HF - g_GEK_star) / (abs(g_HF) + 1e-12):.4f}")
            u_ref_HF = ot.Point([-0.1595, -0.9398])
            g_meta_ref, _ = metamodel_GEK(u_ref_HF)
            g_HF_ref = -6.1e-05
            print(f"\nTest au point u*_HF = {list(u_ref_HF)} :")
            print(f"  g_GEK(u*_HF) = {g_meta_ref:.6f}  (g_HF ≈ 0 par définition)")
            print(f"  g_HF      = {g_HF_ref:.6f}")
            print(f"  Erreur relative : {abs(g_meta_ref - g_HF_ref) / (abs(g_HF_ref) + 1e-12):.4f}")

        else:
            # Test du GP : on évalue le metamodel en u* et on compare à l'appel HF en u*
            g_GP = metamodel_KRG(U_res)[0]
            g_HF, _ = run_HF(modelname, U_res, params_names, T_inv, sensitivity=False)
            print(f"\nTest GP au point de FORM :")
            print(f"  g* FORM = {g_HF:.6f}")
            print(f"  g* GP   = {g_GP:.6f}")
            print(f"  Erreur relative entre g* FORM et g* GP : {abs(g_HF - g_GP) / abs(g_HF):.4f}")
    if do_visu:
        import matplotlib.pyplot as plt
        u_star = U_res
        g_ustar = hf_instance._last_g
        grad_ustar_U = hf_instance._last_grad
        n_fc = 8
        n_fy = 8  # 8×8 = 64 appels HF, nombre pair = pas de point central = pas de doublon avec u*
        u_fc_values = np.linspace(u_star[0] - 1.5, u_star[0] + 1.5, n_fc)
        u_fy_values = np.linspace(u_star[1] - 1.5, u_star[1] + 1.5, n_fy)

        G = np.zeros((n_fy, n_fc))
        for i, u_fy in enumerate(u_fy_values):
            for j, u_fc in enumerate(u_fc_values):
                u_scan = ot.Point([u_fc, u_fy])
                g_val, _ = run_HF(modelname, u_scan, params_names, T_inv, sensitivity=False)
                G[i, j] = g_val

        plt.figure()
        plt.contour(u_fc_values, u_fy_values, G, levels=[0], colors='r', linestyles='--', linewidths=2)
        plt.plot(u_star[0], u_star[1], 'g*', markersize=14, label=f'u* ({u_star[0]:.2f}, {u_star[1]:.2f})')
        plt.plot(0, 0, 'ko', markersize=8, label='Moyenne (origine)')
        plt.xlabel('u_fc')
        plt.ylabel('u_fy')
        plt.title('État limite g=0 autour du point de conception')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(r'C:\_workingDir\_SF\test flexion\etat_limite.png', dpi=150)
        plt.show()
        print("Visu sauvegardée : etat_limite.png")
    
    if do_linear_test:
        u_star = U_res
        u0 = ot.Point([0.0] * n_var)
        g0, grad_U_0 = run_HF(modelname, u0, params_names, T_inv, sensitivity=True)
        norm_sq = grad_U_0.norm() ** 2
        u_star_FOSM = grad_U_0 * (-g0 / norm_sq)
        relative_error = (u_star_FOSM - u_star).norm() / u_star.norm()
        print(f"\nTest linéarisation :")
        print(f"  u* FORM = {u_star}", flush=True)
        print(f"  u* FOSM = {u_star_FOSM}", flush=True)
        print(f"  Erreur relative entre u* FORM et u* FOSM : {relative_error:.4f}", flush=True)



        # class HFModel(ot.OpenTURNSPythonFunction):
        #     def __init__(self):
        #         super().__init__(n_var, 1)
        #         self._last_u = None
        #         self._last_g = None
        #         self._last_grad = None

        #     def _run_if_needed(self, u):
        #         u_list = list(u)
        #         if self._last_u is None or u_list != self._last_u:
        #             self._last_g, self._last_grad = run_HF(
        #                 modelname, u, params_names, T_inv
        #             )
        #             self._last_u = u_list

        #     def _exec(self, u):
        #         self._run_if_needed(u)
        #         return [self._last_g]
        
        # class HFGradient(ot.GradientImplementation):
        #     def __init__(self, hf_model):
        #         super().__init__()
        #         self.hf = hf_model

        #     def getInputDimension(self):
        #         return n_var

        #     def getOutputDimension(self):
        #         return 1

        #     def gradient(self, u):
        #         self.hf._run_if_needed(u)
        #         return ot.Matrix([[v] for v in self.hf._last_grad])
        # hf_impl = HFModel()
        # hf_grad = HFGradient(hf_impl)

        # myFunction = ot.Function(hf_impl)
        # myFunction.setGradient(hf_grad)

        # vect   = ot.RandomVector(dist_U)
        # output = ot.CompositeRandomVector(myFunction, vect)
        # event  = ot.ThresholdEvent(output, ot.Less(), 0.0)

        # solver = ot.AbdoRackwitz()
        # solver.setMaximumIterationNumber(n_max_FORM)
        # solver.setStartingPoint([0.0] * n_var)

        # algo = ot.FORM(solver, event)
        # algo.run()
        # result = algo.getResult()