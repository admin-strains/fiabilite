# -*- coding: utf-8 -*-
static_params = [
    {"value": 0,        "table": "IPARM0","indices": [10]},          # MOSEK (1=YES)
    {"value": 3,        "table": "IPARM0","indices": [11]},          # PT INT (1 = MKL PARDISO, 3 = MUMPS)
    {"value": 0,        "table": "IPARM0","indices": [12]},          # WriteLog Pt Int
    {"value": 300,       "table": "IPARM0","indices": [13]},          # Max iterations number
    {"value": 2,       "table": "IPARM0","indices": [14]},          # 0 = Aucune condensation Pt Int activée ; 1 = sans glams mais gvar ; 2 = gvar et glam
    {"value": 100,        "table": "IPARM0","indices": [15]},          # Frequence Fact Symbolique (tout solveur)
    {"value": 0,        "table": "IPARM0","indices": [16]},          # Dessin de la connectivité de la matrice  
    {"value": 0.90,    "table": "DPARM0","indices": [10]},          # Cone border coef
    {"value": 1.0e-4,   "table": "DPARM0","indices": [11]},          # Tol. abs Alpha
    {"value": 1.0e-2,   "table": "DPARM0","indices": [12]},          # Tol. rel. |Pobj - Dobj|
    {"value": 1.0e-12,   "table": "DPARM0","indices": [13,14,15,16]}  # Tol. rel. Res
]                                                            
cinematic_params = [
    {"value": 0,        "table": "IPARM0","indices": [20]},          # 1=Rankine HV MOSEK, 2= Rankine MA MOSEK, 
    {"value": 4,        "table": "IPARM0","indices": [21]},          # PT INT (1 = MKL PARDISO, 3 = MUMPS)
    {"value": 0,        "table": "IPARM0","indices": [22]},          # WriteLog Pt Int
    {"value": 300,      "table": "IPARM0","indices": [23]},          # Max iterations number
    {"value":2,        "table": "IPARM0","indices": [24]},          # 0 = Aucune condensation Pt Int activée ; 1 = sans glams mais gvar ; 2 = gvar et glam
    {"value": 100,        "table": "IPARM0","indices": [25]},          # Frequence Fact Symbolique (tout solveur)
    {"value": 0,        "table": "IPARM0","indices": [26]},          # Dessin de la connectivité de la matrice    
    {"value": 0.90,     "table": "DPARM0","indices": [20]},          # Cone border coef
    {"value": 1.0e-4,   "table": "DPARM0","indices": [21]},          # Tol. abs Alpha
    {"value": 1e-4,   "table": "DPARM0","indices": [22]},          # Tol. rel. |Pobj - Dobj|
    {"value": 1.0e-12,  "table": "DPARM0","indices": [23,24,25,26]},  # Tol. rel. Res 
    {"value": 1.0,  "table": "DPARM0","indices": [28]}  # special init = 1 CVX, =0 old       
]
MKLPardiso_params = [     
    {"value": 3, "indices": [101]}, # METIS (=2) or OMP METIS (3)
    {"value": 0, "indices": [103]}, # Preconditioned CGS/CG (0* - 1 ind. - 2 sdp)
    {"value": 0, "indices": [107]}, # Max of number of iterative refinement steps 
    {"value": 8, "indices": [109]}, # Tol sur Eps pivoting (8*)
    {"value": 1, "indices": [110]}, # Scaling vectors (0* ou 1)
    {"value": 1, "indices": [112]}, # Weighted matching (0* ou 1)
    {"value": 0, "indices": [120]}, # 1*1 and 2*2 BK piv (3 = 1 + optimized)
    {"value": 0, "indices": [123]}, # Par fact : 0* or 1 (OpenMP if 8 cores)
    {"value": 0, "indices": [133]}  # CNR mode (jamais avec OpenMP dans iparm[1]
]
MyPardiso_params = [     
    {"value": 2, "indices": [201]}, # 1 = MDA , 2* = METIS v4.1 , 3 = METIS v5.1
    {"value":16, "indices": [202]}, # Number of threads OMP
    {"value": 0, "indices": [207]}, # Number of iterative refinement (0*)    
    {"value": 8, "indices": [209]}, # Tol sur Eps pivoting (8*)
    {"value": 0, "indices": [210]}, # Scaling vectors (0*, 1 pour IPM si Matching = 1 ou 2)
    {"value": 0, "indices": [212]}, # Matching (0* ou 1 ou 2)
    {"value": 1, "indices": [220]}, # For sym indef mat with 1*1 and 2*2 Bunch-Kaufman pivoting (1*)
    {"value": 1, "indices": [223]}, # Parallel factorization control : 0 (old) or 1* (new)    
    {"value": 1, "indices": [224]}, # Parallel backward and forward solve (1* = parallel)
    {"value": 1, "indices": [227]}, # Parallel METIS (0*)
    {"value": 0, "indices": [231]}, # 0* = Direct Solver ; 1 = multi-recursive iterative method
    {"value": 1, "indices": [233]}  # Identical Solution independant of NProc 
]
MUMPS_params = [                    
    {"value": 6   ,"indices": [300]}, # Output stream for error messages
    {"value": 6   ,"indices": [301]}, # Output stream for diagnostic
    {"value": 6   ,"indices": [302]}, # Output stream for global information
    {"value": 1   ,"indices": [303]}, # Verbosity (1=err, 2=err+warnings, 3=all)
    {"value": 7   ,"indices": [305]}, # Permut (7 = auto)
    {"value": 7   ,"indices": [306]}, # Ordering (7=auto)
    {"value": 77  ,"indices": [307]}, # Scaling (77 = auto)
    {"value": 0   ,"indices": [309]}, # Iterative refinement (0*)
    {"value": 0   ,"indices": [310]}, # Error Statistics (1 = all)
    {"value": 0   ,"indices": [312]}, # ScaLAPACK on the root node
    {"value": 100 ,"indices": [313]}, # Increase of the working space
    {"value": 0   ,"indices": [322]}, # Size alloue pour MUMPS
    {"value": 0   ,"indices": [323]}, # Detection of null pivot    
    {"value": 0   ,"indices": [332]}  # compute the determinant
]