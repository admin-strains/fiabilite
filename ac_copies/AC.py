import os
os.add_dll_directory("C:\\workspace\\front\\STRAINS\\common\\Dll")
os.add_dll_directory("C:\\workspace\\front\\01_3RDPARTY\\03_meshgems\\lib\\Win10_64_VC17")
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin")
from STRAINS.rupt.APIs.CetCAD_API import *
from STRAINS.rupt.APIs import CetLOAD
from STRAINS.rupt.APIs.CetLOAD_API import *
import STRAINS.rupt.core.CetMESH as CetMESH
from STRAINS.rupt.core import CetSOLV as CetSOLV
from STRAINS.rupt.core import CetVISU as CetVISU, CetLIST as CetLIST
from STRAINS.rupt.APIs import CetNOTE
from STRAINS.rupt.APIs.CetNOTE_API import *
from STRAINS.rupt.APIs.CetList_API import *

import json
import os
import shutil
import uuid
from math import *
import time

def getFile(nameFile):
    f = open(nameFile,'r')
    res = f.read()
    f.close()
    return res

catalogTopo = getFile( "C:\\workspace\\front\\STRAINS\\common\\Catalog\\CatalogTopo.json" )
catalogDimensions = getFile( "C:\\workspace\\front\\STRAINS\\common\\Catalog\\CatalogDimensions.json" )
catalogBolt = getFile("C:\\workspace\\front\\STRAINS\\common\\Catalog\\CatalogBolts.json" )
INITCATALOG(catalogTopo, catalogDimensions, catalogBolt)
catalogDimensions = json.loads( catalogDimensions )
catalogDimensionsClient = catalogDimensions
catalogDimensionsTopo = dict()


def export_dict_to_json(dico, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dico, f, indent=4, ensure_ascii=False)


def saveDocx(rootPath, folderName, noteType, _temp_dir):
    shutil.make_archive(os.path.join(rootPath, folderName), 'zip', os.path.join(_temp_dir, folderName))
    docxName = os.path.join(rootPath, folderName + "_" + noteType + ".docx")
    if os.path.exists(docxName):
        os.remove(docxName)
    os.rename(os.path.join(rootPath, folderName + ".zip"), docxName)
    dirPath = os.path.join(_temp_dir, folderName)
    shutil.rmtree(dirPath, ignore_errors = True)

def _dslog(path, analysis_name, iteration, job_type, message, mode ="a"):
    log_path = os.path.join(path, analysis_name+'_'+str(iteration)+'_'+job_type+'.dslog')
    with open(log_path, mode) as f:
        f.write(message+"\n")

def gestioncad(copyfrom = '') :
    fichier_this = os.path.join(path, "$this.dscad")
    fichier_analyse = os.path.join(path, AnalysisName+".dscad")
    if copyfrom != '' :
        fichier_this  = os.path.join(path, copyfrom+".dscad")
    if not os.path.isfile(fichier_this): print(f"Erreur : le fichier source est introuvable : {fichier_this} , lancer le cad !")
    else:
        if not os.path.isfile(fichier_analyse):
            with open(fichier_this, 'rb') as src, open(fichier_analyse, 'wb') as dst:
                dst.write(src.read())
            print(f"{fichier_analyse} n'existait pas. Copie effectuée.")
        else:
            time_this = os.path.getmtime(fichier_this)
            time_analyse = os.path.getmtime(fichier_analyse)
            if time_this > time_analyse:
                with open(fichier_this, 'rb') as src, open(fichier_analyse, 'wb') as dst:
                    dst.write(src.read())
                print(f"{fichier_this} est plus récent. analyse.dscad mis à jour.")
            else:
                print("analyse.dscad est à jour. Aucune action effectuée.")
    return()


modelname = "test_pure_flexion"
path = "C:\\workspace\\storage\\admin\\SF\\" + modelname + ".ds"
_temp_dir = "C:\\workspace\\storage\\_temp\\" + modelname
path_note = "C:\\workspace\\storage\\_note\\" + modelname

NLGEOM = True

List_cas = ['analysis0']

suff_approach = ''       # '' → Yield_analysis0
kine_ou_stat = 'kinematic'
approach = suff_approach + kine_ou_stat   # 'kinematic'

for iteration in range(0,1):

    for icas in range(0, len(List_cas)):
        if suff_approach == '':
            AnalysisName = "Yield_"
        elif suff_approach == 'EP_':
            AnalysisName = "Elastoplastic_"
        elif suff_approach == 'elastic_':
            AnalysisName = "Elastic_"
        if icas == 0:
            AnalysisName = AnalysisName + List_cas[icas]
            Analysis0 = ""
            dependent = False
        if icas != 0:
            Analysis0 = AnalysisName + List_cas[icas-1]
            AnalysisName = AnalysisName + List_cas[icas]
            restart_analysis = Analysis0
            dependent = True

        if iteration == 0:
            exescad = 0
            exesload = 1
        else:
            exescad = 0
            exesload = 0

        imesh = 0
        copymeshkine = 1
        copymeshstat = 1
        copymeshkwargs = 1
        if icas == 0:
            imesh = 0
            copymeshkine = 0
            copymeshstat = 0
            copymeshkwargs = 0

        isolv = 1
        ivisumesh = 0
        ivisures = 1

        ilist = 0
        ierror = 0

        nb_ana = 1

        if AnalysisName[0] == "E" and suff_approach == "": print("potentielle erreur dans le choix de suff_approach")

        fileName = os.path.join(path, AnalysisName+".dscad")
        model = MODEL()
        if(exescad):
            SET_CONTEXT(model, path)
            cadfile = open(path + '\\dsCad.txt', 'r')
            cadscript = cadfile.read()
            exec(cadscript)
            for i_ana in range(nb_ana):
                fileNameloc = os.path.join(path, AnalysisName+".dscad")
                model.Save(fileNameloc)
                fileNameloc = os.path.join(path, "$this.dscad")
                model.Save(fileNameloc)
            print(model.GETERRORS())

        if(exescad == 0):
            if icas != 0: gestioncad(copyfrom = Analysis0)
            else: gestioncad()

        if(exesload):
            loadfile = open(path + '\\dsLoad.txt', 'r')
            model.Load(fileName)
            loadscript = loadfile.read()
            with CetLOAD.LOAD_MODEL(model, path):
                exec(loadscript)

        if(imesh):
            chemin_mesh_viewres = os.path.join(path, AnalysisName+"_"+str(iteration)+"_mesh.dsviewres")
            if os.path.isfile(chemin_mesh_viewres):
                os.remove(chemin_mesh_viewres)

            Meshkwargs = {}
            Meshkwargs["cadSurfOptions"] = {}
            Meshkwargs["cadSurfOptions"]["volume_gradation"] = 1.5
            Meshkwargs["cadSurfOptions"]["gradation"] = 1.5
            Meshkwargs["cadSurfOptions"]["anisotropic_ratio"] = 10
            Meshkwargs["tetraOptions"] = {}
            Meshkwargs["tetraOptions"]["optimisation_level"] = "standard"
            Meshkwargs["tetraOptions"]["verbose"] = "10"
            Meshkwargs["global_physical_size"] = 0.02
            Meshkwargs["min_size"] = '-1'
            Meshkwargs["gradation"] = 1.5
            Meshkwargs["volume_gradation"] = 1.5
            Meshkwargs["optimisation_level"] = "standard"
            Meshkwargs["anisotropic_ratio"] = "10"
            Meshkwargs["geometric_approximation_min"] = "20"
            Meshkwargs["geometric_approximation_max"] = "25"
            Meshkwargs["geometric_approximation_on_edge"] = "false"
            Meshkwargs["geometric_approximation_on_face"] = "true"
            Meshkwargs["use_surface_proximity"] = "false"
            Meshkwargs["surface_proximity_ratio"] = 0
            Meshkwargs["approach"] = approach
            Meshkwargs["write_debug_files"] = "true"
            Meshkwargs["is_csl_law"] = "false"
            Meshkwargs["OptiMesh"] = "true"
            Meshkwargs["approach"] = "kinematic"
            Meshkwargs["is_iso"] = "true"
            Meshkwargs["coeff_on_error"] = 0.01
            if(iteration >= 1):
                Meshkwargs["coeff_on_courb"] = 0.2
                Meshkwargs["pct_remeshed"] = 0.08
            Meshkwargs["remesh_type"] = 1
            Meshkwargs["old_size_factor"] = 0.0
            Meshkwargs["write_debug_files"] = "true"
            export_dict_to_json(Meshkwargs, path+'/'+AnalysisName+"_"+str(iteration)+"_Meshkwargs.json")
            CetMESH.ANISO_MESH(AnalysisName, iteration, path, **Meshkwargs)

        if(imesh == 0):
            fichier_kinemed1 = os.path.join(path, Analysis0+"_"+str(iteration)+"_kine.dsmed")
            fichier_statmed1 = os.path.join(path, Analysis0+"_"+str(iteration)+"_stat.dsmed")
            fichier_kinemed2 = os.path.join(path, AnalysisName+"_"+str(iteration)+"_kine.dsmed")
            fichier_statmed2 = os.path.join(path, AnalysisName+"_"+str(iteration)+"_stat.dsmed")
            if copymeshkine:
                with open(fichier_kinemed1, 'rb') as src, open(fichier_kinemed2, 'wb') as dst:
                    dst.write(src.read())
                    print('copie kine : '+ Analysis0+"_"+str(iteration)+"_kine.dsmed" + ' remplace '+ AnalysisName+"_"+str(iteration)+"_kine.dsmed")
            if copymeshstat:
                with open(fichier_statmed1, 'rb') as src, open(fichier_statmed2, 'wb') as dst:
                    dst.write(src.read())
                    print('copie stat : '+ Analysis0+"_"+str(iteration)+"_stat.dsmed" + ' remplace '+ AnalysisName+"_"+str(iteration)+"_stat.dsmed")
            if copymeshkwargs:
                fichier_meshkwargs1 = os.path.join(path, Analysis0+"_"+str(iteration)+"_Meshkwargs.json")
                fichier_meshkwargs2 = os.path.join(path, AnalysisName+"_"+str(iteration)+"_Meshkwargs.json")
                with open(fichier_meshkwargs1, 'rb') as src, open(fichier_meshkwargs2, 'wb') as dst:
                    dst.write(src.read())

        kwargs = {"scaling": 1, "write_debug_files": "true"}

        _initsolver_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "InitSolver.py")
        exec(open(_initsolver_path).read())
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
        kwargs["tetra_discontinuities"] = False
        kwargs["activated_plasticity"] = True
        kwargs["welds_throat_limit"] = True
        kwargs["hybrid_elements"] = False

        if(isolv):
            if kine_ou_stat != 'kinematic' and kine_ou_stat != "static": print("pb dans la definition de kine_ou_stat")
            if suff_approach != 'EP_' and suff_approach != "" and suff_approach != "elastic_": print("pb dans la definition de suff_approach")
            if kine_ou_stat == 'kinematic':
                chemin_dep_viewres = os.path.join(path, AnalysisName+"_"+str(iteration)+"_Displacement.dsviewres")
                if os.path.isfile(chemin_dep_viewres):
                    os.remove(chemin_dep_viewres)
                chemin_strains_viewres = os.path.join(path, AnalysisName+"_"+str(iteration)+"_Strain.dsviewres")
                if os.path.isfile(chemin_strains_viewres):
                    os.remove(chemin_strains_viewres)
            if kine_ou_stat == 'static':
                chemin_sigma_viewres = os.path.join(path, AnalysisName+"_"+str(iteration)+"_Stress.dsviewres")
                if os.path.isfile(chemin_sigma_viewres):
                    os.remove(chemin_sigma_viewres)

            kwargs["geometric_nonlinearities"] = NLGEOM
            kwargs["approach"] = approach
            kwargs["bolts_material_treatement"] = True
            kwargs["activated_plasticity"] = True
            kwargs["GeneralizedStrainsDegree"] = 2
            print("bolts material treatment : ", kwargs["bolts_material_treatement"])
            if dependent:
                if kine_ou_stat == "kinematic":
                    CetSOLV.SOLV(AnalysisName, iteration, path, restart_analysis+"_0_kine", **kwargs)
                else:
                    CetSOLV.SOLV(AnalysisName, iteration, path, restart_analysis+"_0_stat", **kwargs)
            else:
                CetSOLV.SOLV(AnalysisName, iteration, path, **kwargs)

            print("fin du calcul pour l'analyse" + AnalysisName)

        if(ivisumesh):
            avantvisumesh = time.time()
            CetVISU.RenderResult(AnalysisName, 0, "mesh", path)
            apresvisumesh = time.time()
            print(f"Durée de la generation du mesh.dsviewres : {apresvisumesh - avantvisumesh:.4f} secondes")

        if ivisures == 1:
            if kine_ou_stat == "kinematic":
                avantvisukine = time.time()
                CetVISU.RenderResult(AnalysisName, 0, "kine", path)
                apresvisukine = time.time()
                print(f"Durée de la generation des kine.dsviewres : {apresvisukine - avantvisukine:.4f} secondes")
            if kine_ou_stat == "static":
                avantvisu = time.time()
                CetVISU.RenderResult(AnalysisName, 0, "stat", path)
                apresvisu = time.time()
                print(f"Durée de la generation des stat.dsviewres : {apresvisu - avantvisu:.4f} secondes")

        if(ierror):
            ErrorKwargs = {}
            ErrorKwargs["write_debug_files"] = "true"
            ErrorKwargs["run_debug"] = "true"
            CetSOLV.CALCULATE_ERRORS(AnalysisName, iteration, path, **ErrorKwargs)