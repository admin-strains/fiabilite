# Plan : Arborescence fonctionnelle du PCK (PC-Kriging) dans UQLab 2.2.0

## Source MATLAB

```
C:\Users\semia.frikha\Downloads\UQLab_Rel2.2.0\modules\uq_model\builtin\uq_metamodel\
```

## Instructions de codage (appliquées à chaque branche)

> 1. Lire le code MATLAB source mot à mot.
> 2. Créer le fichier Python correspondant (`branche*.py`) en traduisant fidèlement chaque fonction.
> 3. Vérifier la fidélité en relisant le MATLAB et en comparant chaque ligne.
> 4. Écrire un fichier de test (`test_branche*.py`) et valider que tous les tests passent.
> 5. Mettre à jour ce fichier MD avec les résultats.

> **Règle absolue — lors de la lecture de ce fichier l'utilisatrice exige aussi que :**  
> Tous les fichiers Python existants (`branche1.py` à `branche5.py`, `branche_lars.py`) soient lus **systématiquement** à chaque mise à jour du contexte, pas seulement avant de proposer du code.  
> Ne jamais réécrire une fonction déjà codée — identifier ce qui existe et ne proposer que le delta réel.

## Explication de la bibliothèque Python PCK

Tout passe par **un seul fichier d'entrée** : `branche1.py`.  
Les autres (`branche2` à `branche5`, `branche_lars`) sont des dépendances internes — ne pas les importer directement.

---

### Fichiers à avoir dans le même dossier

```
branche1.py        ← point d'entrée public
branche2.py        ← parsing options (B2)
branche3.py        ← fit Kriging + PCE (B3)
branche4.py        ← prédiction (B4)
branche5.py        ← fonctions atomiques (B5)
branche_lars.py    ← algorithme LARS
```

---

### 1. Format des données d'entrée

```python
# Marginals : liste de M dicts, un par dimension d'entrée
marginals = [
    {'Type': 'Uniform',   'Parameters': [-1.0, 1.0]},   # dim 1
    {'Type': 'Gaussian',  'Parameters': [0.0, 1.0]},    # dim 2  (mu, sigma)
    {'Type': 'Lognormal', 'Parameters': [0.0, 0.3]},    # dim 3  (mu_ln, sigma_ln)
]

# Copule : Independent dans presque tous les cas
import numpy as np
copula = {'Type': 'Independent', 'Parameters': np.eye(len(marginals))}
```

---

### 2. Générer un DOE (TEMPORAIRE)

```python
from branche1 import generate_doe   # fonction temporaire de test

X = generate_doe(
    N=30,               # nombre de points
    marginals=marginals,
    method='lhs',       # 'lhs' (Latin Hypercube) ou 'mc' (Monte Carlo pur)
    seed=42
)
# X : ndarray (N, M)
```

---

### 3. Créer le métamodèle

```python
from branche1 import fit_pck

options = {
    'Mode': 'sequential',             # 'sequential' (défaut) ou 'optimal'
    'PCE':  {'Degree': [1, 2, 3],    # degrés polynomiaux à tester
             'Method': 'LARS'},       # 'LARS' ou 'OMP'
}

fitted_model = fit_pck(X, Y, options, marginals, copula)
# Y : ndarray (N,) pour 1 sortie  ou  (N, Nout) pour plusieurs sorties
```

**Options Kriging avancées (facultatif) :**

```python
options = {
    'Mode': 'sequential',
    'PCE':  {'Degree': [1, 2, 3, 4], 'Method': 'LARS'},
    'Kriging': {
        'Corr': {
            'Family':    'matern-5_2',   # défaut ; autres : 'gaussian', 'exponential'
            'Type':      'separable',
            'Isotropic': False,
        },
        'Optim': {
            'Method': 'gradbased',       # 'gradbased' (L-BFGS-B) ou 'de' (évolution diff.)
            'Bounds': np.array([[0.01]*M, [100.0]*M]),  # bornes sur theta
        },
        'EstimMethod': 'ml',             # 'ml' (maximum de vraisemblance) ou 'cv'
    },
}
```

---

### 4. Appeler le métamodèle (prédiction)

```python
from branche1 import predict_pck

X_test = ...   # ndarray (N_test, M)

# Moyenne seule
YMu = predict_pck(fitted_model, X_test)
# YMu : (N_test, Nout)

# Moyenne + variance prédictive
YMu, YVar = predict_pck(fitted_model, X_test, return_var=True)
# YVar : (N_test, Nout)  — variance sigma^2 (pas écart-type)

# Moyenne + variance + matrice de covariance complète
YMu, YVar, YCov = predict_pck(fitted_model, X_test, return_var=True, return_cov=True)
# YCov : (N_test, N_test, Nout)
```

---

### 5. Lire les résultats du fit

```python
# Erreur LOO par sortie (Leave-One-Out normalisée, in [0,1] si bon modèle)
for oo in range(fitted_model['Nout']):
    print(f'Output {oo}  LOO = {fitted_model["Error"][oo]["LOO"]:.4e}')

# Nombre de polynômes retenus par sortie
print(fitted_model['NumberOfPoly'])   # liste de Nout entiers

# Hyperparamètres Kriging par sortie
for oo, k in enumerate(fitted_model['Kriging']):
    print(f'Output {oo}  theta={k["theta"]}  sigma2={k["sigmaSQ"]:.4e}')
```

---

### 6. Utiliser le métamodèle comme fonction limite (FORM / fiabilité)

Pour FORM, le métamodèle remplace le modèle mécanique coûteux.  
La fonction limite est : `g(X) = seuil - Y_pred(X)` (défaillance si g < 0).

```python
def limit_state(X_row, fitted_model, threshold):
    """
    Évalue g(x) = threshold - f_hat(x) en un point ou un batch.
    Retourne scalaire si X_row est (M,), vecteur si (N, M).
    """
    X_2d  = np.atleast_2d(X_row)
    Y_hat = predict_pck(fitted_model, X_2d)   # (N, Nout)
    g     = threshold - Y_hat[:, 0]            # (N,)
    return float(g[0]) if X_2d.shape[0] == 1 else g

# Exemple d'appel dans une boucle FORM/SORM/Monte-Carlo
g_values = limit_state(X_mc_sample, fitted_model, threshold=0.0)
P_f_hat  = np.mean(g_values < 0)
```

Pour obtenir le gradient numérique (nécessaire à FORM/SORM) :

```python
def grad_limit_state(x, fitted_model, threshold, eps=1e-5):
    """Gradient par différences finies centrées."""
    x  = np.array(x, dtype=float)
    M  = len(x)
    g0 = limit_state(x, fitted_model, threshold)
    grad = np.zeros(M)
    for i in range(M):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps;  xm[i] -= eps
        grad[i] = (limit_state(xp, fitted_model, threshold)
                 - limit_state(xm, fitted_model, threshold)) / (2 * eps)
    return grad
```

---

### 7. Workflow complet type

```python
import numpy as np
from branche1 import generate_doe, fit_pck, predict_pck

# 1. Décrire l'espace d'entrée
M         = 2
marginals = [{'Type': 'Uniform', 'Parameters': [-1.0, 1.0]}] * M
copula    = {'Type': 'Independent', 'Parameters': np.eye(M)}

# 2. Générer le DOE et évaluer le modèle réel (coûteux)
X_doe = generate_doe(30, marginals, method='lhs', seed=0)
Y_doe = mon_modele_couteux(X_doe)   # à fournir

# 3. Construire le métamodèle PCK
opts = {'Mode': 'sequential', 'PCE': {'Degree': [1, 2, 3], 'Method': 'LARS'}}
fm   = fit_pck(X_doe, Y_doe, opts, marginals, copula)

# 4. Vérifier la qualité
print(f'LOO = {fm["Error"][0]["LOO"]:.4e}')   # < 0.05 = bon

# 5. Utiliser le métamodèle (ex. Monte-Carlo sur le substitut)
X_mc  = generate_doe(100_000, marginals, method='mc', seed=1)
Y_hat = predict_pck(fm, X_mc)
```

---

## Contexte

Le module PCK d'UQLab (Polynomial Chaos-Kriging) est une hybridation de deux métamodèles :
- **PCE (Polynomial Chaos Expansion)** fournit un trend polynomial sparse (via LARS/OMP)  
- **Kriging (Processus Gaussien)** capture les résidus autour de ce trend

L'objectif est de cartographier toutes les fonctions MATLAB impliquées, de la racine d'appel jusqu'aux feuilles atomiques.

---

## ARBORESCENCE FONCTIONNELLE COMPLÈTE

### TRONC — Interface utilisateur

```
uq_createModel(OPTIONS)
   [core/interfaces/CLI/uq_createModel.m]
   Entrée unique de l'utilisateur. OPTIONS.MetaType = 'PCK'
```

---

### BRANCHE 1 — Dispatch central

```
uq_initialize_uq_metamodel(current_model)
   [modules/uq_model/builtin/uq_metamodel/uq_initialize_uq_metamodel.m]
   Aiguille selon OPTIONS.MetaType → case 'pck' → BRANCHE 2
   Puis appelle uq_calculateMetamodel() → case 'pck' → BRANCHE 3
```

---

### BRANCHE 2 — Initialisation PCK

```
uq_PCK_initialize(current_model)
   [PCK/uq_PCK_initialize.m]
   │
   ├── uq_getInput()
   │      [core] Récupère l'objet probabiliste Input courant
   │
   ├── uq_process_option(Options, 'Input', ...)
   │      [lib] Parse et valide les options ; utilisé pour CHAQUE paramètre
   │      (Mode, IgnoreDependence, PCE, Kriging, CombCrit, PolyIndices…)
   │
   └── uq_copyObj(current_model.Internal.Input)
          [core] Crée une copie de l'Input pour le remplacer par une copule
          indépendante si IgnoreDependence = true
```

**Logique décisionnelle dans uq_PCK_initialize :**
- **TrendMethod = 'user'** si l'utilisateur fournit `PolyIndices` + `PolyTypes`
- **TrendMethod = 'pce'**  si PCE.Method ∈ {LARS, OMP} (défaut)
- **Mode = 'sequential'** (défaut) ou **'optimal'**
- **CombCrit = 'rel_loo'** (défaut) uniquement si Mode='optimal'

---

### BRANCHE 3 — Calcul des coefficients (cœur hybride)

```
uq_PCK_calculate_coefficients(module)
   [PCK/uq_PCK_calculate_coefficients.m]
```

#### Étape A — Design expérimental

```
   ├── uq_getModel(module)
   │      [core] Récupère le modèle courant depuis la session
   │
   ├── uq_getExpDesignSample(current_model)
   │      [lib/ExpDesign] Génère les points X du design (LHS, MC, user…)
   │
   ├── uq_eval_ExpDesign(current_model, X)
   │      [lib/ExpDesign] Évalue le modèle de calcul en X → Y
   │
   └── uq_remove_constants_from_input(Input, '-private')
          [lib] Supprime les dimensions constantes de l'espace d'entrée
```

#### Étape B — Calcul du trend polynomial (deux cas)

**Cas TrendMethod = 'pce' :**
```
   └── uq_createModel(popts)   [popts.MetaType='PCE', popts.Method='LARS'|'OMP']
          │  Déclenche tout le sous-arbre PCE (BRANCHE 3a ci-dessous)
          │
          └── Extrait idxranking{oo} = myPCE.Internal.PCE(oo).LARS.lars_idx
                 ou                    myPCE.Internal.PCE(oo).OMP.omp_idx
                 (classement des polynômes par importance décroissante)
```

**Cas TrendMethod = 'user' :**
```
   └── uq_createModel(popts)   [popts.Method='Custom', PolyIndices fournis]
          idxranking{oo} = 1:nbPolynomes  (ordre utilisateur)
```

#### Étape C — Espace auxiliaire

```
   └── uq_createInput(myPCE.Internal.ED_Input, '-private')
          [core] Crée l'espace probabiliste auxiliaire dans lequel
          les polynômes sont définis (espace isoprobabiliste)
```

#### Étape D — Composition Kriging + trend polynomial (deux modes)

**Mode 'sequential' — un seul Kriging avec TOUS les polynômes :**
```
   └── POUR chaque output oo:
          │
          ├── POUR chaque polynôme ii dans idxranking{oo}:
          │      └── Construit popts2.PCE(ii) : copie de myPCE.PCE(oo)
          │             tous coefficients = 0 sauf coefficients(idxranking(ii)) = 1
          │
          ├── uq_createModel(popts2, '-private')
          │      [MetaType='PCE', Method='Custom']
          │      → crée myPIP : PCE vectorielle (Npoly sorties)
          │        chaque sortie = un polynôme de base isolé
          │
          ├── kopts.Trend.Handle = @(X,dummy) uq_evalModel(myPIP, X)
          │      Le trend du Kriging est une FONCTION des polynômes PCE
          │
          └── uq_createModel(kopts, '-private')
                 [MetaType='Kriging']
                 → BRANCHE 3b : tout le sous-arbre Kriging
```

**Mode 'optimal' — cherche le meilleur sous-ensemble de polynômes :**
```
   └── POUR chaque output oo:
          └── POUR ii = 1 à length(idxranking{oo}):
                 │  (test successif : 1 poly, puis 2 polys, … jusqu'à N)
                 ├── Construit popts2 (polynômes 1..ii)
                 ├── uq_createModel(popts2) → myPIP
                 ├── kopts.Trend.Handle = @(X,dummy) uq_evalModel(myPIP, X)
                 ├── uq_createModel(kopts) → myKriging(ii)  [BRANCHE 3b]
                 └── CompCrit(ii) = myKriging(ii).Error.LOO
                       (ou critère custom si CombCrit='fh')
          Sélectionne ii* = argmin(CompCrit)
          → myPCKrigingoo = myKriging(ii*)
```

#### Étape E — Stockage résultats

```
   ├── uq_addprop(current_model, 'PCK', myPCKriging.Kriging)
   │      [core] Ajoute la propriété PCK à l'objet
   │
   └── Remplit :
          current_model.Internal.Kriging(oo)     [modèles Kriging par output]
          current_model.Internal.AuxSpace         [espace auxiliaire polynomial]
          current_model.Internal.NumberOfPoly(oo) [nb polynômes retenus]
          current_model.ExpDesign.U               [coordonnées dans espace aux.]
          current_model.Error(oo).LOO             [erreur LOO]
```

---

### BRANCHE 3a — Sous-arbre PCE (calcul du trend)

```
uq_PCE_initialize(current_model)
   [PCE/uq_PCE_initialize.m]
   ├── uq_process_option() × N   [parsing de toutes les options PCE]
   ├── uq_getInput()
   └── Détermine la base polynomiale (Degree, PolyTypes, TruncStrategy)

uq_PCE_calculate_coefficients(current_model)
   [PCE/uq_PCE_calculate_coefficients.m]
   │
   ├── (Méthode LARS)
   │      uq_PCE_lars(Psi, Y, options)
   │         [PCE/PolyCoeff/uq_PCE_lars.m]
   │         Least Angle Regression sur la base polynomiale
   │         Retourne : coefficients sparse + lars_idx (classement)
   │
   ├── (Méthode OMP)
   │      uq_PCE_omp(Psi, Y, options)
   │         [PCE/PolyCoeff/uq_PCE_omp.m]
   │         Orthogonal Matching Pursuit
   │         Retourne : coefficients sparse + omp_idx (classement)
   │
   ├── uq_PCE_loo_error(current_model, oo)
   │      [PCE/uq_PCE_loo_error.m]
   │      Calcule l'erreur LOO de la PCE (via correction analytique LARS)
   │
   └── Construction de la base :
          uq_eval_legendre(P, U)   [lib/Poly] Polynômes de Legendre
          uq_eval_hermite(P, U)    [lib/Poly] Polynômes d'Hermite
          uq_PCE_create_Psi(indices, univ_p_val)  [PCE] Assemble Ψ
```

---

### BRANCHE 3b — Sous-arbre Kriging (GP résiduel)

```
uq_Kriging_initialize(current_model)
   [Kriging/uq_Kriging_initialize.m]
   │
   ├── uq_Kriging_init_Input(current_model)
   │      [Kriging/init/] Gère l'espace d'entrée et les constantes
   │
   ├── uq_Kriging_init_Trend(current_model)
   │      [Kriging/init/] Configure le type de trend
   │      (ici : type = 'custom', Handle = @PCE vectorielle)
   │
   ├── uq_Kriging_init_Corr(current_model)
   │      [Kriging/init/] Choisit et initialise la fonction de corrélation
   │      (Matérn 5/2 par défaut ; Gaussian, Exponential, etc.)
   │
   ├── uq_Kriging_init_Optim(current_model)
   │      [Kriging/init/] Borne et point initial pour l'optimisation de θ
   │
   ├── uq_Kriging_init_EstimMethod(current_model)
   │      [Kriging/init/] Méthode d'estimation : 'ML' (défaut) ou 'CV'
   │
   ├── uq_Kriging_init_Scaling(current_model)
   │      [Kriging/init/] Normalisation de l'espace (ici = AuxSpace PCK)
   │
   ├── uq_Kriging_init_Regression(current_model)
   │      [Kriging/init/] Active la régression (bruit) si demandée
   │
   ├── uq_Kriging_init_KeepCache(current_model)
   │      [Kriging/init/] Active le cache des matrices auxiliaires
   │
   └── uq_Kriging_init_GRF(current_model)
          [Kriging/init/] Random Field (si trajectoires demandées)

uq_Kriging_calculate(current_model)
   [Kriging/uq_Kriging_calculate.m]
   │
   ├── uq_Kriging_initialize_optimizer(current_model)
   │      [Kriging/init/uq_Kriging_initialize_optimizer.m]
   │      Prépare les options d'optimisation (bounds, initials, solver)
   │
   ├── evalF_handle(X, current_model)
   │      = Trend.Handle = @(X,m) uq_evalModel(myPIP, X)
   │      (PCE vectorielle définie dans BRANCHE 3)
   │
   ├── uq_Kriging_optimizer(X, Y, optim_options, current_model)
   │      [Kriging/optimizer/uq_Kriging_optimizer.m]
   │      Optimise les hyperparamètres θ (portées de corrélation)
   │      │
   │      ├── (méthode ML)
   │      │      uq_Kriging_eval_J_of_theta_ML(theta, current_model)
   │      │         [Kriging/optimizer/] Calcule la log-vraisemblance négative
   │      │         → appelle evalR_handle(X, X, theta, CorrOptions)
   │      │              (la fonction de corrélation choisie, ex. Matérn5/2)
   │      │
   │      ├── (méthode CV)
   │      │      uq_Kriging_eval_J_of_theta_CV(theta, current_model)
   │      │         [Kriging/optimizer/] Calcule le critère de cross-validation
   │      │
   │      └── Solveurs d'optimisation :
   │             uq_optim_de(J, theta0, lb, ub, opts)
   │                [Kriging/optimizer/] Differential Evolution
   │             uq_optim_sade(J, theta0, lb, ub, opts)
   │                [Kriging/optimizer/] Self-Adaptive DE
   │             fmincon / fminbnd
   │                [MATLAB built-in] Optimisation locale
   │
   ├── calc_R(current_model, X, oo)          [locale dans calculate]
   │      evalR_handle(X, X, theta, CorrOptions)
   │      → Matrice de corrélation R (N×N) aux points d'entraînement
   │
   ├── uq_Kriging_calc_auxMatrices(R, F, Y, runCase)
   │      [Kriging/calc/uq_Kriging_calc_auxMatrices.m]
   │      Calcule : Cholesky de R, R⁻¹, F^T R⁻¹, F^T R⁻¹ F
   │
   ├── calc_Beta(current_model, auxMatrices, oo)   [locale dans calculate]
   │      uq_Kriging_calc_beta(F, trendType, Y, method, auxMatrices)
   │         [Kriging/calc/] GLS : β = (F^T R⁻¹ F)⁻¹ F^T R⁻¹ Y
   │
   ├── calc_SigmaSQML / calc_SigmaSQCV             [locales dans calculate]
   │      uq_Kriging_calc_sigmaSq(parameters, method)
   │         [Kriging/calc/] Estime σ² du processus gaussien
   │
   └── uq_Kriging_calc_KFold(randIdx, Y, F, auxMatrices)
          [Kriging/calc/uq_Kriging_calc_KFold.m]
          Leave-One-Out cross-validation → Error.LOO
```

---

### BRANCHE 4 — Évaluation (prédiction)

```
uq_evalModel(myPCK, X_test)
   [core/interfaces/CLI/uq_evalModel.m]
   └── uq_eval_uq_metamodel(current_model, X)
          [modules/uq_model/builtin/uq_metamodel/uq_eval_uq_metamodel.m]
          └── case 'pck' → uq_PCK_eval(current_model, X)

uq_PCK_eval(current_model, X)
   [PCK/uq_PCK_eval.m]
   └── POUR chaque output oo:
          Kriging_oo = current_model.Internal.Kriging(oo)
          └── uq_Kriging_eval(Kriging_oo, X)
                 [Kriging/eval/uq_Kriging_eval.m]
                 │
                 ├── uq_GeneralIsopTransform(X, Input.Marginals, Input.Copula,
                 │       Scaling.Marginals, Scaling.Copula)
                 │      [lib] Transforme X → U (espace auxiliaire polynomial)
                 │
                 ├── evalF_handle(U0, current_model)
                 │      = @(X,m) uq_evalModel(myPIP, X)
                 │      Évalue le trend polynomial aux points de prédiction
                 │
                 ├── evalR_handle(U0, U, theta, CrossCorOpts)
                 │      Évalue la corrélation croisée r₀ (prédiction ↔ entraînement)
                 │
                 ├── uq_Kriging_calc_auxMatrices(R, F, Y, 'default')
                 │      [si cache vide] Recalcule les matrices auxiliaires
                 │
                 ├── uq_Kriging_calc_DiagOfCongruent(r0, R)
                 │      [Kriging/calc/] Calcule diag(r0 R⁻¹ r0^T) → variance D1
                 │
                 └── Retourne [YMu, YSigma2, YCov]
                        YMu = f0·β + r0·R⁻¹·(Y - F·β)   [prédiction]
                        YSigma2 = σ²·(1 - D1 + D2)        [variance prédictive]
```

---

### BRANCHE 5 — Fonctions du trend polynomial (auxiliaires PCK)

Ces fonctions sont utilisées pour évaluer la matrice F du trend polynomial
dans le contexte Kriging (appelées via le handle `evalF_handle`).

```
uq_PCK_eval_F(X, polyindices, PolyTypes, Input)
   [PCK/uq_PCK_eval_F.m]
   │
   ├── uq_poly_marginals(PolyTypes)
   │      [lib/PCE] Retourne les marginales canoniques (Uniform→Legendre, etc.)
   │
   ├── uq_GeneralIsopTransform(X, Input.Marginals, Input.Copula,
   │       PolyMarginals, PolyCopula)
   │      [lib] Transforme vers l'espace probabiliste des polynômes
   │
   ├── uq_PCK_eval_unipoly(Upce, polyindices, PolyTypes)
   │      [PCK/uq_PCK_eval_unipoly.m]
   │      │   Évalue les polynômes univariés par dimension
   │      ├── uq_eval_legendre(P, U(:,i))
   │      │      [lib/Poly] Polynômes de Legendre jusqu'au degré P
   │      └── uq_eval_hermite(P, U(:,i))
   │             [lib/Poly] Polynômes de Hermite jusqu'au degré P
   │
   └── uq_PCE_create_Psi(polyindices, univ_p_val)
          [PCE] Produit tensoriel des polynômes univariés → base multivariée Ψ
```

---

### BRANCHE 6 — Affichage et impression

```
uq_PCK_display(PCKRGModel, outArray, varargin)
   [PCK/uq_PCK_display.m]
   ├── uq_evalModel(myPCK, Xplot)     [évalue sur grille fine]
   ├── uq_GeneralIsopTransform(...)   [pour les transformées]
   ├── uq_figure()                    [lib/graphics] Crée figure UQLab
   ├── uq_plot(...)                   [lib/graphics] Courbe moyenne
   └── uq_plotConfidence(...)         [lib/graphics] Bande de confiance

uq_PCK_print(PCKRGModel, outArray, varargin)
   [PCK/uq_PCK_print.m]
   ├── uq_sprintf_mat(M, fmt)         [lib] Formate une matrice en string
   └── add_leadingChars(str, chars)   [locale] Ajoute préfixe à chaque ligne
```

---

## RÉSUMÉ VISUEL DE L'ARBORESCENCE

```
uq_createModel('PCK')
└── uq_initialize_uq_metamodel          [dispatch]
    ├── uq_PCK_initialize               [BRANCHE 2 : init options]
    │   ├── uq_process_option           [×N : parse chaque param]
    │   ├── uq_getInput                 
    │   └── uq_copyObj                  [si IgnoreDependence]
    │
    └── uq_PCK_calculate_coefficients   [BRANCHE 3 : cœur hybride]
        ├── uq_getExpDesignSample       [génère X]
        ├── uq_eval_ExpDesign           [évalue Y]
        ├── uq_remove_constants_from_input
        │
        ├── uq_createModel(PCE)         [BRANCHE 3a : trend sparse]
        │   ├── uq_PCE_initialize
        │   └── uq_PCE_calculate_coefficients
        │       ├── uq_PCE_lars         [LARS regression]
        │       │   └── uq_PCE_loo_error
        │       ├── uq_PCE_omp          [OMP regression]
        │       ├── uq_eval_legendre    [base polynomiale]
        │       ├── uq_eval_hermite
        │       └── uq_PCE_create_Psi  [assemblage base Ψ]
        │
        ├── uq_createInput(AuxSpace)    [espace auxiliaire]
        │
        └── uq_createModel(Kriging)     [BRANCHE 3b : GP résiduel]
            ├── uq_Kriging_initialize
            │   ├── uq_Kriging_init_Input
            │   ├── uq_Kriging_init_Trend   [handle = PCE vectorielle]
            │   ├── uq_Kriging_init_Corr    [Matérn5/2 par défaut]
            │   ├── uq_Kriging_init_Optim
            │   ├── uq_Kriging_init_Scaling [= AuxSpace]
            │   └── uq_Kriging_init_EstimMethod  [ML ou CV]
            │
            └── uq_Kriging_calculate
                ├── uq_Kriging_initialize_optimizer
                ├── evalF_handle = uq_evalModel(PCE_vectorielle)
                ├── uq_Kriging_optimizer        [optimise θ]
                │   ├── uq_Kriging_eval_J_of_theta_ML
                │   ├── uq_Kriging_eval_J_of_theta_CV
                │   ├── uq_optim_de             [Differential Evolution]
                │   └── uq_optim_sade           [Self-Adaptive DE]
                ├── uq_Kriging_calc_auxMatrices [chol R, R⁻¹, F^T R⁻¹ F]
                ├── uq_Kriging_calc_beta        [GLS : β]
                ├── uq_Kriging_calc_sigmaSq     [σ²]
                └── uq_Kriging_calc_KFold       [LOO error]

uq_evalModel(myPCK, X)
└── uq_PCK_eval                         [BRANCHE 4 : prédiction]
    └── uq_Kriging_eval                 [délègue au GP interne]
        ├── uq_GeneralIsopTransform     [X → U espace aux.]
        ├── evalF_handle → uq_evalModel(PCE_vectorielle)
        ├── evalR_handle                [corrélation croisée r₀]
        ├── uq_Kriging_calc_auxMatrices [si pas de cache]
        └── uq_Kriging_calc_DiagOfCongruent  [variance D1]

(Fonctions auxiliaires du trend)
uq_PCK_eval_F                           [BRANCHE 5]
├── uq_poly_marginals
├── uq_GeneralIsopTransform
├── uq_PCK_eval_unipoly
│   ├── uq_eval_legendre
│   └── uq_eval_hermite
└── uq_PCE_create_Psi

(Visualisation)
uq_PCK_display                          [BRANCHE 6]
uq_PCK_print
```

---

## POINTS ARCHITECTURAUX CLÉS

1. **Le trend est une closure** : `kopts.Trend.Handle = @(X,dummy) uq_evalModel(myPIP, X)`  
   Le Kriging ne "sait" pas qu'il travaille avec des polynômes PCE — il reçoit juste une fonction.

2. **L'espace auxiliaire** est l'espace probabiliste canonique de la base PCE (Uniform[-1,1] pour Legendre, Normal(0,1) pour Hermite). C'est dans cet espace que le Kriging est calibré (variable `Scaling = ED_Input`).

3. **Mode optimal = boucle imbriquée** : N Kriging sont entraînés (1 poly, 2 polys, … N polys), et on garde celui avec le LOO minimal. Coût = N × coût(Kriging).

4. **uq_Kriging_eval est réutilisé tel quel** dans PCK. Le PCK n'a pas son propre moteur de prédiction — il instancie un Kriging complet et délègue l'évaluation.

5. **LOO analytique** : Le Kriging calcule son erreur LOO par la formule de Dubrule (pas par cross-validation explicite), via `uq_Kriging_calc_KFold` avec N folds.

---

## RELATIONS D'APPEL ENTRE BRANCHES

```
B1 (dispatch)
├── appelle B2 (init options)
└── appelle B3 (calculate)
         ├── appelle B3a (LARS → idxranking)
         │       B3a utilise B5 pour évaluer Ψ (legendre, hermite, create_Psi)
         │
         └── appelle B3b (Kriging) avec closure(myPIP)
                 B3b utilise B5 via le handle : isop_transform, corr_matern52
                 B3b utilise B5 via la closure : legendre, hermite, create_Psi

B4 (eval)
└── appelle B3b (Kriging eval) directement
        B3b utilise B5 : isop_transform, evalF_handle→myPIP, corr croisée
```

- **B5** est la seule branche appelée par tout le monde sans appeler personne.
- **B3** est la seule branche qui appelle à la fois B3a et B3b.
- **B4** court-circuite B3 — il appelle B3b directement, pas B3.
- **B3b ne peut pas exister sans la closure que B3 lui fabrique.** B3b et B3 forment une unité inséparable.

---

## ORDRE STRATÉGIQUE DE CODAGE (bottom-up)

### Étape 1 — B5 : feuilles atomiques

Aucune dépendance interne. Rien ne peut tourner sans elles.

```
eval_legendre(P, U)
eval_hermite(P, U)
isop_transform(X, marginals_in, copula_in, marginals_out, copula_out)
create_Psi(indices, univ_p_val)
corr_matern52(X1, X2, theta, opts)   [+ autres fonctions de corrélation]
```

### Étape 2 — Cœur de B3a : LARS seul ✅ FAIT

Dépend de B5 uniquement (pour construire Ψ).
Testable sur données synthétiques : entrée = Ψ et y, sortie = `idxranking`.

### Étape 3 — B2 : uq_PCK_initialize (parsing d'options) ✅ FAIT

Zéro dépendance mathématique. Lit les options utilisateur et remplit
`current_model['Internal']` avec la configuration validée.
B3 consomme ce dict de config pour savoir quel mode utiliser.

### Étape 4 — B3 : PCK_calculate (première unité PCK complète) ✅ FAIT

C'est ici que tout s'assemble. Les fonctions Kriging internes (auxMatrices,
beta_GLS, sigmaSq, LOO, optimizer) sont des utilitaires privés de cette étape,
pas des modules exposés séparément. La boucle qui construit myPIP → closure →
optimize θ → β → LOO appartient entièrement à cette étape.

> **Kriging seul n'existe pas dans UQLab en dehors de PCK.**
> Il n'est jamais instancié sans un `Trend.Handle` fourni par B3.
> Ne pas créer de module Kriging standalone en Python.

### Étape 5 — B4 : PCK_eval

Boucle triviale sur B3b eval. Testable dès que l'étape 4 tourne.

### Étape 6 — B1 : entry point

`fit_pck(X, y, options)` qui appelle B2 puis B3. Se code en dernier.

### Dépendances circulaires : clarification

Il n'y a **pas** de dépendance circulaire dans l'algorithme PCK de UQLab.

- LARS sélectionne les polynômes à inclure (OLS pur, indépendant de θ).
- Le trend **handle** fourni à Kriging retourne F (la matrice de base évaluée en X), pas F·β.
- β est calculé **à l'intérieur** de Kriging en post-traitement de θ : β = (F^T R(θ)⁻¹ F)⁻¹ F^T R(θ)⁻¹ y.
- Les coefficients β (amplitude des polynômes) sont donc bien produits par l'optimisation θ,
  mais le handle lui-même n'en dépend pas.

Dans le mode **optimal** (boucle sur le nombre de polynômes), PCE et Kriging co-évoluent
à travers les itérations : θ est ré-optimisé à chaque ajout d'un polynôme.
Cette co-évolution se produit au niveau de la boucle orchestratrice de B3,
pas à l'intérieur d'un seul appel Kriging.

---

## IMPLÉMENTATION PYTHON — BRANCHE 5

### Fichier

`C:\_workingDir\_SF\test flexion\branche5.py`

Traduction mot-à-mot du code MATLAB UQLab 2.2.0. Toutes les fonctions de B5
sont implémentées dans l'ordre de dépendance.

### Fonctions traduites

| # | Fonction Python | Source MATLAB | Notes |
|---|---|---|---|
| 1 | `uq_poly_rec_coeffs(n_max, polytype)` | `PCE/PolyBasis/uq_poly_rec_coeffs.m` | Coefficients analytiques (Gautschi 2004). Familles : Legendre, Hermite, Laguerre, Jacobi, Fourier, Zero. Retourne `[AB_matrix, bounds]`. |
| 2 | `uq_eval_rec_rule(X, AB, nonrecursive)` | `PCE/PolyBasis/uq_eval_rec_rule.m` | Récurrence à 3 termes. Retourne matrice `N×(ORDER+1)`. Décalage d'indexing MATLAB→Python documenté. |
| 3 | `uq_eval_legendre(ORDER, X)` | `PCE/PolyBasis/uq_eval_legendre.m` | Wrapper. Vérifie X ∈ (-1,1). |
| 4 | `uq_eval_hermite(ORDER, X)` | `PCE/PolyBasis/uq_eval_hermite.m` | Wrapper. Pas de contrainte de domaine. |
| 5 | `uq_PCK_eval_unipoly(U, polyindices, PolyTypes)` | `PCK/uq_PCK_eval_unipoly.m` | Tenseur `(N, M, P+1)`. `np.asarray(polyindices)` ajouté pour cas sparse (MATLAB `full()`). |
| 6 | `uq_PCE_create_Psi(Indices, univ_p_val)` | `PCE/PolyCoeff/Regression/uq_PCE_create_Psi.m` | Produit tensoriel. `np.asarray(Indices)` + `try/except` (MATLAB `warning`). |
| 7 | `uq_GeneralIsopTransform(X, Xm, Xc, Ym, Yc)` | `uq_input/uq_GeneralIsopTransform.m` | Cascade : bloc indépendant → Nataf (stub) → Rosenblatt (stub). `_uq_IsopTransform` implémenté pour : Uniform, Gaussian, Lognormal, Gumbel, Exponential. |
| 8 | `uq_assemble_Kernel(h, K_family, K_type)` | `lib/uq_kernel/uq_assemble_Kernel.m` | Toutes familles stationnaires. Séparable : `prod(axis=1)`. |
| 9 | `uq_eval_Kernel(X1, X2, theta, options)` | `lib/uq_kernel/uq_eval_Kernel.m` | Gram (triangle inférieur + symétrie), non-Gram, nugget, isotropique/anisotropique, callable custom. |

### Décision architecturale clé

Le handle trend transmis à Kriging retourne **F** (matrice de base `N×P`),
pas `F·β`. β est calculé **à l'intérieur** de Kriging en post-traitement de θ.
Le handle ne dépend pas de θ. Donc Kriging n'existe pas sans PCK : en Python,
**ne pas créer de module Kriging standalone**.

### Bugs corrigés par rapport à la première traduction

1. `uq_PCK_eval_unipoly` : `np.asarray(polyindices)` manquant — plantait sur indices sparse.
2. `uq_PCE_create_Psi` : `np.asarray(Indices)` + `try/except` manquants.

### Résultats des tests (26/05/2026)

Toutes les assertions passent avec erreur numérique ≤ 2.22e-16 (précision machine) :

```
uq_poly_rec_coeffs    : coefficients Legendre et Hermite exacts
uq_eval_legendre      : P0=1, P1=sqrt(3)*x, P2=sqrt(5)/2*(3x²-1)
uq_eval_hermite       : H0=1, H1=x, H2=(x²-1)/sqrt(2)
uq_PCK_eval_unipoly   : shape (N,M,P+1), degrés 0 et 1
uq_PCE_create_Psi     : termes constant, linéaires, quadratique
uq_GeneralIsopTransform: U[0,1]→U[-1,1], Gauss(2,3)→N(0,1)
uq_assemble_Kernel    : Matérn-5/2(0)=1, Gaussien
uq_eval_Kernel        : Gram (symétrie, diag=1), non-Gram,
                        nugget (diag=1+eps), anisotropique 2D
```

### Prochaine étape après B5

Étape 2 du plan de codage : **LARS** — algorithme OLS seul, entrée `(Psi, y)`,
sortie `idxranking`. Dépend uniquement de B5 (Psi calculée par les fonctions ci-dessus).

---

## IMPLÉMENTATION PYTHON — BRANCHE 2

### Fichier

`C:\_workingDir\_SF\test flexion\branche2.py`

Traduction mot-à-mot du code MATLAB UQLab 2.2.0. Deux fonctions.

### Fonctions traduites

| # | Fonction Python | Source MATLAB | Notes |
|---|---|---|---|
| 1 | `uq_process_option(AllOptions, OptionName, Default, AllowedClasses, EmptyAsMissing)` | `core/uq_process_option.m` | Lookup case-insensitif. Flags Missing/Invalid/Disabled. Merge struct champ-par-champ si AllowedClasses='struct'. Pour 'uq_input' (objet MATLAB, `isstruct()=False`) : assignation directe sans merge. |
| 2 | `uq_PCK_initialize(current_model, global_input)` | `PCK/uq_PCK_initialize.m` | Remplit `current_model['Internal']` depuis `current_model['Options']`. |

### Logique de uq_PCK_initialize — 4 décisions

1. **TrendMethod** :
   - Ni PCE ni PolyIndices → `'pce'` + DEFAULTPCE `{MetaType:'PCE', Degree:[1,2,3], Method:'LARS'}`
   - PolyIndices sans PCE → `'user'` (stocke PolyIndices + PolyTypes)
   - PCE sans PolyIndices → `'pce'` (merge avec DEFAULTPCE, vérifie Method ∈ {lars,omp})
   - Les deux → erreur

2. **Mode** : `'sequential'` (défaut) ou `'optimal'`

3. **CombCrit** : uniquement si Mode=`'optimal'`, défaut `'rel_loo'`

4. **IgnoreDependence** : si True, deepcopy de l'Input + remplace copule par Indépendante

5. **Kriging.Optim.Bounds** : réduit aux colonnes `nonConst` si constantes présentes

### Particularité clé de uq_process_option

En MATLAB, le merge struct ne s'applique qu'aux vrais structs (`isstruct()=True`).
Les objets `uq_input` ont `isstruct()=False` → assignation directe.
En Python les deux sont des `dict` : le guard `'struct' in AllowedClasses` reproduit cette distinction.

### Résultats des tests (26/05/2026)

50 PASS / 0 FAIL. Tous les cas couverts :

```
Section 1 : uq_process_option (19 tests)
  - missing, found, case-insensitif, invalid type, struct merge,
    double clé, EmptyAsMissing, logical/double list, uq_input

Section 2 : defaults                    (9 tests)
Section 3 : mode=optimal                (3 tests)
Section 4 : CombCrit custom             (1 test)
Section 5 : PCE options override        (4 tests)
Section 6 : TrendMethod=user            (3 tests)
Section 7 : 5 cas d'erreur              (5 tests)
Section 8 : IgnoreDependence            (4 tests)
Section 9 : Kriging.Optim.Bounds ajusté (2 tests)
```

### Prochaine étape

Étape 3 du plan de codage : **B2** (fait). Puis **B3 PCK_calculate**.

---

## IMPLÉMENTATION PYTHON — LARS (Étape 2)

### Fichier

`C:\_workingDir\_SF\test flexion\branche_lars.py`

Traduction mot-à-mot du code MATLAB UQLab 2.2.0. Quatre fonctions + wrapper.

### Fonctions traduites

| # | Fonction Python | Source MATLAB | Notes |
|---|---|---|---|
| 1 | `uq_PCE_loo_error(Psi, M, Y, coefficients, modified_flag, modi_diag)` | `PCE/uq_PCE_loo_error.m` | Toujours le chemin nargout==3 (Python retourne toujours 3 valeurs). Correction T uniquement si modified_flag=True. NaN→Inf pour problèmes sous-déterminés. |
| 2 | `uq_blockwise_inverse(Ainv, B, C, D)` | `lib/uq_matrix_utils/uq_blockwise_inverse.m` | Prend **Ainv = A⁻¹**, pas A. Complément de Schur. Cas scalaire D et cas matriciel. |
| 3 | `uq_PCE_OLS_regression(Psi, Y, options)` | `PCE/PolyCoeff/Regression/uq_PCE_OLS_regression.m` | rcond > 1e-12 → solve (rapide), sinon pinv (stable). |
| 4 | `uq_lar(Psi, Y, options)` | `lib/uq_regression/LAR/uq_lar.m` | Algorithme LARS complet. |
| 5 | `uq_PCE_lars(Psi, Y, ...)` | `PCE/PolyCoeff/Regression/uq_PCE_lars.m` | Wrapper fin autour de uq_lar. |

### Algorithme uq_lar — points clés

1. **Centering** : si colonne constante détectée (`diff=0`), Ψ et Y centrés. Constante toujours incluse dans le résultat final.
2. **Normalisation** : stddev=1 sur les colonnes (optionnelle, active par défaut).
3. **Mise à jour rang-1** : `uq_blockwise_inverse(M, x, x', r)` pour éviter une ré-inversion complète à chaque itération.
4. **Gamma** : formule d'Efron et al. 2004 — min des valeurs positives du vecteur tmp.
5. **LOO dans la boucle** : avec `hybrid_loo=True` (défaut), coefficients OLS (M*Ψ'*Y) ; sinon coefficients LARS courants.
6. **Early stop** : si `loo_scores[k-mm-1] <= refscore` pour mm=max(round(nvars*0.1), 100) itérations.
7. **Hybrid LARS** : à la fin, les coefficients sont recalculés par OLS sur la base sélectionnée.

### Convention d'index MATLAB → Python

| MATLAB (1-indexé) | Python (0-indexé) |
|---|---|
| `coeff_array(k+1, a_coeff)` | `coeff_array[k, a_arr]` |
| `a_scores(k+1)` | `a_scores[k]` |
| `loo_scores(k-mm)` | `loo_scores[k-mm-1]` |
| `a_coeff(1:(k-1))` | `a_coeff[:k_best]` |
| `constindices(1)` | `constindices[0]` |
| `nz_idx = abs(coeff_array(k,:))>0` | `nz_idx = abs(coeff_array[k_best,:])>0` |

### Bug corrigé lors de la vérification

Le test 1a utilisait `A` (la matrice, pas son inverse) comme premier argument de `uq_blockwise_inverse`. L'API attend `Ainv = A⁻¹`. Bug dans le test, pas dans l'implémentation. Corrigé en passant `np.linalg.inv(A)`.

### Résultats des tests (26/05/2026)

42 PASS / 0 FAIL :

```
Section 1 : uq_blockwise_inverse       (3 tests) — 2x2, 3x3, symétrie
Section 2 : uq_PCE_loo_error           (6 tests) — zero LOO, None coeff, T, varY=0, modi_diag, keys
Section 3 : uq_PCE_OLS_regression      (6 tests) — précision, LOO, ill-conditioned
Section 4 : uq_lar P==1                (4 tests) — cas trivial
Section 5 : uq_lar sparse recovery     (7 tests) — récupère cols {0,5,12} sur P=20
Section 6 : uq_lar constant regressor  (5 tests) — centering + intercept
Section 7 : uq_lar normalize=False     (2 tests)
Section 8 : uq_lar early stop          (2 tests)
Section 9 : uq_PCE_lars wrapper        (2 tests)
Section 10: hybrid_lars=False          (3 tests)
```

### Prochaine étape

Étape 4 du plan : **B3 PCK_calculate** ✅ FAIT. Prochaine étape : **B4 PCK_eval** (prédiction).

---

## IMPLÉMENTATION PYTHON — B3 (PCK_calculate)

### Fichier

`C:\_workingDir\_SF\test flexion\branche3.py`

Traduction mot-à-mot du code MATLAB UQLab 2.2.0. Dix couches (layers).

### Fonctions traduites

| # | Fonction Python | Source MATLAB | Notes |
|---|---|---|---|
| 1 | `uq_Kriging_calc_DiagOfCongruent(A, B)` | `Kriging/calc/uq_Kriging_calc_DiagOfCongruent.m` | `diag(A B^{-1} A^T)`. Condition : `rcond(B) > eps`. |
| 2 | `uq_Kriging_calc_auxMatrices(R, F, Y, runCase)` | `Kriging/calc/uq_Kriging_calc_auxMatrices.m` | 3 runCases : default (FTRinv block), ml_optimization (QR block), ml_estimation (les deux). `cholR` = upper-tri (convention MATLAB = numpy.cholesky().T). `None` remplace MATLAB `nan`. |
| 3 | `uq_Kriging_calc_beta(F, trendType, Y, method, auxMatrices)` | `Kriging/calc/uq_Kriging_calc_beta.m` | QR : `beta = G^{-1} Q1^T Ytilde`. Standard : `(F^T R^{-1} F)^{-1} F^T R^{-1} Y`. Simple : `ones`. |
| 4 | `uq_Kriging_calc_sigmaSq(KrgParameters, estimMethod)` | `Kriging/calc/uq_Kriging_calc_sigmaSq.m` | 5 variantes : ml_chol, ml_nochol, ml_bypass_chol, ml_bypass_nochol, cv. |
| 5 | `uq_Kriging_calc_KFold(randIdx, Y, F, auxMatrices)` | `Kriging/calc/uq_Kriging_calc_KFold.m` | LOO (N folds) via formule de Dubrule : `yPredMu = Y - (1/diag(B1)) * (B1@Y)`. K-fold générique. |
| 6 | `uq_Kriging_eval_J_of_theta_ML(theta, KrgModelParameters)` | `Kriging/optimizer/uq_Kriging_eval_J_of_theta_ML.m` | Objectif ML : `J = 0.5*(N*log(2pi*sigma^2) + log|R| + N)`. |
| 7 | `kriging_optimize_theta(KrgModelParameters, theta0, bounds, method)` | `Kriging/optimizer/uq_Kriging_optimizer.m` | scipy L-BFGS-B (gradbased), differential_evolution (de), ou évaluation seule (none). |
| 8 | `fit_kriging_pck(U, Y, F_handle, CorrOptions, ...)` | `Kriging/uq_Kriging_calculate.m` | Équivalent fonctionnel pour le contexte PCK. Optimise theta, calcule beta, sigma², LOO. |
| 9 | Utilitaires PCE | `PCE + PCK` | `pce_multi_indices`, `pce_eval_design_matrix`, `poly_type_from_marginal`, `aux_marginal_from_poly_type`. |
| 10 | `uq_PCK_calculate_coefficients(X, Y, pck_config, ...)` | `PCK/uq_PCK_calculate_coefficients.m` | Orchestrateur B3 : LARS → trend handle → Kriging. Modes sequential et optimal. |

### Bugs corrigés lors de la vérification MATLAB

1. **`pce_multi_indices`** : stack initialisé avec `remaining=0` au lieu de `remaining=max_degree` — ne produisait que le multi-indice nul `(0,...,0)`. Corrigé en `stack = [([], max_degree)]`.
2. **`uq_Kriging_eval_J_of_theta_ML`** : chemin nochol (Cholesky échoue) — `kp` ne contenait pas `Y` et `F`, nécessaires pour `ml_nochol`. Corrigé en ajoutant `kp['Y'] = Y` et `kp['F'] = F`.
3. **`uq_Kriging_calc_DiagOfCongruent`** : condition `matrix_rank` redondante ; simplifié en `1/cond(B) > eps` (fidèle au MATLAB `rcond(B) > eps`).

### Convention Cholesky

MATLAB `chol(R)` = upper-tri `L` telle que `L^T L = R`.
Python : `cholR = np.linalg.cholesky(R).T` (upper-tri). `None` = Cholesky échoue.
Tous les backsolves adaptés en conséquence : `solve(cholR.T, ...)` pour solve triangulaire inférieur, `solve(cholR, ...)` pour supérieur.

### Architecture de make_trend_handle

Remplace `@(X,dummy) uq_evalModel(myPIP, X)` de MATLAB. Closure Python :
```python
def make_trend_handle(selected_idx, Indices, poly_types, ...):
    Idx_sel = Indices[np.array(selected_idx), :]   # (P_sel, Mred)
    def F_handle(U):
        uv  = uq_PCK_eval_unipoly(U, Idx_sel, p_types)
        F   = uq_PCE_create_Psi(Idx_sel, uv)
        return F                                   # (N, P_sel)
    return F_handle
```

### Résultats des tests (26/05/2026)

75 PASS / 0 FAIL :

```
Section 1  : uq_Kriging_calc_DiagOfCongruent  (4 tests)
Section 2  : uq_Kriging_calc_auxMatrices       (10 tests) — 3 runCases, keys, orthogonalite Q1
Section 3  : uq_Kriging_calc_beta              (4 tests)  — QR, standard, simple, fallback
Section 4  : uq_Kriging_calc_sigmaSq           (6 tests)  — ml_chol ~= ml_bypass_chol ~= ml_nochol
Section 5  : uq_Kriging_calc_KFold             (7 tests)  — LOO in (0,1), K=3
Section 6  : uq_Kriging_eval_J_of_theta_ML     (5 tests)  — finite J, J change avec theta
Section 7  : kriging_optimize_theta            (4 tests)  — theta in bounds, J non-degrade
Section 8  : PCE utilities                    (12 tests)  — pce_multi_indices (M=1,2,3 D=2,4), Psi shape
Section 9  : fit_kriging_pck                   (7 tests)  — LOO < 0.5, trend prediction
Section 10 : uq_PCK_calculate sequential      (8 tests)
Section 11 : uq_PCK_calculate optimal         (3 tests)
Section 12 : uq_PCK_calculate multi-output    (5 tests)
```

### Prochaine etape

Etape 5 : **B4 PCK_eval** ✅ FAIT.
Etape 6 : **B1** — point d'entree `fit_pck(X, y, options)` qui appelle B2 puis B3.

---

## IMPLÉMENTATION PYTHON — B4 (PCK_eval)

### Fichiers

- `C:\_workingDir\_SF\test flexion\branche4.py` — implémentation
- `C:\_workingDir\_SF\test flexion\test_branche4.py` — tests

### Sources MATLAB lues

- `PCK/uq_PCK_eval.m`
- `Kriging/eval/uq_Kriging_eval.m`

### Fonctions traduites

| # | Fonction Python | Source MATLAB | Notes |
|---|---|---|---|
| 1 | `uq_Kriging_eval_one_output(kriging_oo, U_test, U_train, Y_train, F_train, CorrOptions, return_var, return_cov)` | `Kriging/eval/uq_Kriging_eval.m` (corps de boucle oo) | Trois chemins : mean only / mean+var / mean+var+cov. Nugget forcé à 0 pour r0. |
| 2 | `uq_PCK_eval(fitted_model, X_test, return_var, return_cov)` | `PCK/uq_PCK_eval.m` | Boucle sur Nout sorties. Appelle `uq_GeneralIsopTransform` pour X→U avec `red_cop = Independent`. |

### Architecture

- **F_handle** stocké dans le dict retourné par `fit_kriging_pck` (ajout B3) : `kriging_oo['F_handle'](U_test)` → `(N_test, P)`.
- **Nugget = 0** pour la corrélation croisée r0 (prédiction ↔ entraînement) : `CrossCorOpts['Nugget'] = 0.0`.
- **Rinv depuis cholR** : `solve(cholR, solve(cholR.T, eye(N)))`.
- **Copule réduite** : `red_cop = {'Type': 'Independent', 'Parameters': eye(Mred)}` (constantes déjà supprimées).
- **Propriété d'interpolation** : à un point d'entraînement i, `r0 = R[i,:]` → D1=1, u0=0, D2=0, YSigma2=0.

### Résultats des tests (26/05/2026)

41 PASS / 0 FAIL :

```
Section 1 : uq_Kriging_eval_one_output (mean only)           (3 tests)
Section 2 : uq_Kriging_eval_one_output (mean + variance)     (4 tests)
Section 3 : uq_Kriging_eval_one_output (mean + covariance)   (5 tests) — symétrie, PSD
Section 4 : Kriging interpolation property                    (2 tests) — YSig2~0 aux points train
Section 5 : uq_PCK_eval single output, 3 modes               (8 tests)
Section 6 : uq_PCK_eval interpolation                        (2 tests)
Section 7 : uq_PCK_eval modèle 2D                            (6 tests)
Section 8 : uq_PCK_eval multi-output (Nout=2)                (7 tests)
Section 9 : Qualité prédiction (RMSE, variance)              (4 tests)
```

### Prochaine etape

Etape 6 : **B1** ✅ FAIT.

---

## IMPLÉMENTATION PYTHON — B1 (entry point)

### Fichiers

- `C:\_workingDir\_SF\test flexion\branche1.py` — implémentation
- `C:\_workingDir\_SF\test flexion\test_branche1.py` — tests

### Sources MATLAB lues

- `modules/uq_model/builtin/uq_metamodel/uq_initialize_uq_metamodel.m`
- `modules/uq_model/builtin/uq_metamodel/uq_eval_uq_metamodel.m`

### Fonctions traduites

| # | Fonction Python | Source MATLAB | Notes |
|---|---|---|---|
| 1 | `fit_pck(X, Y, options, marginals, copula)` | `uq_initialize_uq_metamodel.m` (chemin PCK + 'user' ED) + `uq_calculateMetamodel` | Appelle B2 (`uq_PCK_initialize`) puis B3 (`uq_PCK_calculate_coefficients`). |
| 2 | `predict_pck(fitted_model, X_test, return_var, return_cov)` | `uq_eval_uq_metamodel.m` (case 'pck') | Wrapper mince autour de B4 (`uq_PCK_eval`). |

### Architecture

- **B1 est une colle** entre les options utilisateur et B2+B3 : construit le dict `current_model`, appelle B2 pour parser les options, extrait `pck_config` de `current_model['Internal']`, passe à B3.
- **Kriging tuning** extrait de `options['Kriging']` : `Optim.Bounds` → `theta_bounds`, `Optim.Method` → `optim_method`, `EstimMethod` → `estim_method`, `Corr` → `CorrOptions`.
- **`predict_pck`** = `uq_evalModel(myPCK, X)` en MATLAB : un seul appel à `uq_PCK_eval`.

### Résultats des tests (26/05/2026)

51 PASS / 0 FAIL :

```
Section 1  : Default options (sequential, LARS 1-2-3)        (7 tests)
Section 2  : Mode=optimal                                     (3 tests)
Section 3  : Custom PCE degree                                (3 tests)
Section 4  : Custom Kriging correlation (Gaussian)            (4 tests)
Section 5  : TrendMethod=user (PolyIndices manuels)           (4 tests)
Section 6  : IgnoreDependence                                 (3 tests)
Section 7  : Multi-output (Nout=2)                            (8 tests)
Section 8  : Input 2D                                         (4 tests)
Section 9  : predict_pck -- 3 modes retour                    (7 tests)
Section 10 : Interpolation aux points d'entrainement          (2 tests)
Section 11 : predict_pck == uq_PCK_eval (identite)           (2 tests)
Section 12 : Cas d'erreur                                     (4 tests)
```

### Fonction DOE temporaire (27/05/2026)

Ajout d'une fonction `generate_doe(N, marginals, method, seed)` **temporaire** dans `branche1.py`
(commentée comme telle). Utilise `scipy.stats.qmc.LatinHypercube` pour LHS ou MC pur.
Marginals supportés : Uniform, Gaussian/Normal, Lognormal.
À supprimer quand les tests seront terminés.

### Démonstration visuelle (27/05/2026)

Script : `demo_pck.py`  
Figure : `demo_pck.png`

Fonction test : **`f(x) = sin(3πx) · exp(−x²)`** sur `[-2, 2]`  
DOE : N=20 points LHS, degrés PCE 1-4, mode sequential.

Résultat :
- LOO = 1.04e+00 > 1 → sous-apprentissage attendu : la fonction a ~3 oscillations complètes,
  N=20 est insuffisant pour les capturer toutes. L'IC 95% est large dans les zones sans points.
- La tendance générale est bien capturée ; les pics mal résolus correspondent aux zones peu peuplées.
- Le modèle interpole exactement aux 20 points du DOE (propriété Kriging).

---

## GEK — Kriging avec information de gradient (Bouhlel & Martins 2019)

Source : *Gradient-enhanced kriging for high-dimensional problems*, Engineering with Computers 35:157–173.  
Section pertinente : **Section 3.2 — Direct gradient-enhanced kriging** (Eqs. 19–26).

---

### Contexte et motivation

Le Kriging standard n'utilise que les valeurs `y(x^(i))` au DOE.  
Quand le gradient `∂y/∂x` est disponible (calcul analytique, différences finies, méthode adjointe),
on peut l'intégrer dans le modèle pour améliorer la précision — potentiellement divisée par 2 à 3
pour le même coût de calcul (le gradient coûte environ autant qu'une évaluation si l'on a un code adjoint).

Deux formulations existent :
- **Indirecte** (Section 3.1) : ajoute des points fictifs autour de chaque point du DOE via une approximation Taylor. Aucune modification du code Kriging, mais matrice de corrélation mal conditionnée.
- **Directe** (Section 3.2) : augmente le vecteur Y avec les dérivées et construit une matrice R̃ en 4 blocs. C'est cette méthode qui est décrite ici.

---

### Vecteur Y augmenté (Eq. 19)

Taille : **n(d+1) × 1**, où n = nombre de points du DOE, d = dimension de l'entrée.

```
y = [y(x^(1)), ..., y(x^(n)),
     ∂y(x^(1))/∂x_1, ..., ∂y(x^(1))/∂x_d,
     ...
     ∂y(x^(n))/∂x_1, ..., ∂y(x^(n))/∂x_d]^T
```

Les n premières entrées sont les **valeurs** de la fonction.  
Les nd entrées suivantes sont les **dérivées partielles** dans chaque direction, pour chaque point.

Le vecteur `1` de la vraisemblance devient (Eq. 20) :
```
1 = [1,...,1,  0,...,0]^T
     n uns     nd zéros
```
(les dérivées ne contribuent pas au terme de moyenne constante μ)

---

### Matrice de corrélation R̃ en 4 blocs (Eq. 21)

Taille : **n(d+1) × n(d+1)**. Structure par blocs :

```
R̃ = [ Bloc 1  |  Bloc 2 ]
     [ Bloc 3  |  Bloc 4 ]
```

| Bloc | Lignes | Colonnes | Contenu |
|------|--------|----------|---------|
| 1 | n | n | `r(x^(i), x^(j))` — corrélation valeur/valeur |
| 2 | n | nd | `∂r(x^(i),x^(j)) / ∂x^(j)` — corrélation valeur/dérivée |
| 3 | nd | n | `∂r(x^(i),x^(j)) / ∂x^(i)` — corrélation dérivée/valeur |
| 4 | nd | nd | `∂²r(x^(i),x^(j)) / ∂x^(i)∂x^(j)` — corrélation dérivée/dérivée |

**R̃ n'est PAS symétrique** (l'article le dit explicitement, p. 162).  
Raison : le Bloc 3 = −(Bloc 2)^T car `∂r/∂x^(i) = −∂r/∂x^(j)` (les signes s'inversent quand on dérive par rapport à l'autre argument).

---

### Formules explicites pour le noyau Gaussien (Eqs. 22–24)

Noyau Gaussien séparable (Eq. 4 de l'article) :
```
r(x^(i), x^(j)) = ∏_k  exp(−θ_k (x_k^(i) − x_k^(j))²)
```

**Notation** : `δ_k = x_k^(i) − x_k^(j)`,  `r = r(x^(i), x^(j))`.

#### Bloc 2 — ∂r / ∂x_k^(j)  (Eq. 23)

Dériver r par rapport à `x_k^(j)`, sachant que `∂δ_k/∂x_k^(j) = −1` :

```
∂r / ∂x_k^(j)  =  +2θ_k δ_k · r
```

C'est un vecteur colonne de taille d (un terme par direction k).

#### Bloc 3 — ∂r / ∂x_k^(i)  (Eq. 22)

Dériver r par rapport à `x_k^(i)`, sachant que `∂δ_k/∂x_k^(i) = +1` :

```
∂r / ∂x_k^(i)  =  −2θ_k δ_k · r
```

Signe opposé au Bloc 2, donc Bloc 3 = −(Bloc 2)^T.

#### Bloc 4 — ∂²r / (∂x_k^(i) ∂x_l^(j))  (Eq. 24)

Dériver le Bloc 3 par rapport à `x_l^(j)`, en distinguant k = l et k ≠ l :

**Cas k ≠ l** (termes hors-diagonale de la sous-matrice d×d) :
```
∂²r / (∂x_k^(i) ∂x_l^(j))  =  −4θ_k θ_l δ_k δ_l · r
```

**Cas k = l** (termes diagonaux) :
```
∂²r / (∂x_k^(i) ∂x_k^(j))  =  2θ_k (1 − 2θ_k δ_k²) · r
```

La différence vient du terme `∂δ_k/∂x_k^(j) = −1` qui n'apparaît que lorsqu'on dérive deux fois selon la même direction.

L'article (Eq. 24) écrit la formule générale `−4θ_kθ_l δ_k δ_l r` pour tous k,l, mais cela ne couvre que les termes hors-diagonale. Les termes diagonaux ont un terme supplémentaire `+2θ_k r` qui vient du δ_{kl} de Kronecker.

---

### Formule de prédiction (Eq. 25)

Une fois θ estimé, la prédiction en un point x non observé est :

```
ŷ(x) = μ̂ + r̃_{xX}^T · R̃^{-1} · (y − 1 μ̂)
```

Le vecteur de corrélation croisée r̃_{xX} (Eq. 26) entre x et l'ensemble du DOE augmenté :

```
r̃_{xX} = [r_{xx^(1)}, ..., r_{xx^(n)},
            ∂r_{x^(1)x}/∂x^(1), ..., ∂r_{x^(n)x}/∂x^(n)]^T
```

Taille : n(d+1) × 1.

---

### Remarque sur le noyau Matérn 5/2 (noyau par défaut dans notre code)

Notre code (`branche5.py`, `uq_eval_Kernel`) utilise le noyau **Matérn 5/2** par défaut, pas le noyau Gaussien.  
La **structure en 4 blocs de R̃ est identique** ; seules les formules des dérivées changent.

Le noyau Matérn 5/2 séparable s'écrit :
```
k_k(h)  =  (1 + √(5θ_k)|h| + 5θ_k h²/3) · exp(−√(5θ_k)|h|)
```
avec `h = δ_k = x_k^(i) − x_k^(j)`.

Les dérivées de ce noyau par rapport à `h` sont plus longues à calculer que pour le Gaussien
(il faut distinguer h > 0 et h < 0 à cause du `|h|`), mais le principe reste le même :
- Bloc 2 : `∂k/∂x_k^(j) = −∂k/∂h · (∂h/∂x_k^(j)) = +∂k/∂h`
- Bloc 3 : `∂k/∂x_k^(i) = −∂k/∂h`  (signe opposé)
- Bloc 4 diagonal (k=l) : `∂²k/∂x_k^(i)∂x_k^(j) = −∂²k/∂h²`
- Bloc 4 hors-diagonale (k≠l) : `(∂k_k/∂x_k^(i)) · (∂k_l/∂x_l^(j)) / r`  [produit des dérivées univariées]

Ces formules doivent être dérivées analytiquement avant implémentation.

---

### Résumé de ce qui change par rapport au Kriging standard

| Élément | Kriging standard | GEK direct |
|---------|-----------------|------------|
| Vecteur Y | n × 1 (valeurs) | n(d+1) × 1 (valeurs + dérivées) |
| Matrice R | n × n, symétrique | n(d+1) × n(d+1), **non symétrique** |
| Vecteur 1 | n × 1 (tous uns) | n(d+1) × 1 (n uns puis nd zéros) |
| r̃_{xX} | n × 1 | n(d+1) × 1 |
| Formule de prédiction | identique (Eq. 9) | identique (Eq. 25) |
| Coût inversion R | O(n³) | O((n(d+1))³) |

La formule de prédiction `ŷ = μ̂ + r̃^T R̃^{-1}(y − 1μ̂)` est **structurellement identique** à celle du Kriging standard (Eq. 9) — seules les dimensions changent.

---

## GEPCK — Polynomial-Chaos–Kriging avec gradient (Zuhal et al. 2021)

Source : *Polynomial-Chaos–Kriging with Gradient Information for Surrogate Modeling in Aerodynamic Design*,  
AIAA Journal, https://doi.org/10.2514/1.J059905.  
Sections pertinentes : **II.C** (GEK), **II.D** (GEPCE), **III.B** (GEPCK), **III.C** (LOOCV), Eqs. 6–14.

---

### Vecteur ẏ augmenté (Eq. 6) — taille n(m+1) × 1 — ordre dimension-major

```
ẏ = { y(x^(1)), y(x^(2)), ..., y(x^(n)),
      ∂y(x^(1))/∂x_1^(1), ..., ∂y(x^(n))/∂x_1^(n),
      ...,
      ∂y(x^(1))/∂x_m^(1), ..., ∂y(x^(n))/∂x_m^(n) }^T
```

**Ordre** : d'abord les n valeurs, puis les n dérivées par rapport à x₁ pour tous les points, ..., puis les n dérivées par rapport à xₘ pour tous les points.

Différence avec Bouhlel 2019 (Eq. 19) : Bouhlel groupe toutes les directions pour le point i avant de passer au point i+1 (ordre point-major). Zuhal groupe tous les points pour la direction k avant de passer à la direction k+1 (ordre dimension-major).

---

### Matrice de corrélation augmentée R̃ (Eq. 7) — taille n(m+1) × n(m+1) — (m+1)×(m+1) blocs de taille n×n

```
      valeurs      ∂/∂x_1^i           ∂/∂x_2^i          ...   ∂/∂x_m^i
     ┌──────────┬──────────────────┬──────────────────┬─────┬──────────────────┐
val  │    R     │  ∂R/∂x_1^i       │  ∂R/∂x_2^i       │ ... │  ∂R/∂x_m^i       │
     ├──────────┼──────────────────┼──────────────────┼─────┼──────────────────┤
∂x_1 │∂R/∂x_1^j│∂²R/(∂x_1^i∂x_1^j)│∂²R/(∂x_1^i∂x_2^j)│ ... │∂²R/(∂x_1^i∂x_m^j)│
     ├──────────┼──────────────────┼──────────────────┼─────┼──────────────────┤
∂x_2 │∂R/∂x_2^j│∂²R/(∂x_1^j∂x_2^i)│∂²R/(∂x_2^i∂x_2^j)│ ... │∂²R/(∂x_2^i∂x_m^j)│
     ├──────────┼──────────────────┼──────────────────┼─────┼──────────────────┤
 ... │   ...    │       ...        │       ...        │ ... │       ...        │
     ├──────────┼──────────────────┼──────────────────┼─────┼──────────────────┤
∂x_m │∂R/∂x_m^j│∂²R/(∂x_1^j∂x_m^i)│∂²R/(∂x_2^j∂x_m^i)│ ... │∂²R/(∂x_m^i∂x_m^j)│
     └──────────┴──────────────────┴──────────────────┴─────┴──────────────────┘
```

**Convention de notation (Zuhal)** :
- `x^i` = premier argument de `k(x^i, x^j)` → ligne de la matrice R
- `x^j` = second argument de `k(x^i, x^j)` → colonne de la matrice R
- `∂R/∂x_k^i` : dérivée par rapport à la k-ème composante du **premier** argument (entrées de ligne)
- `∂R/∂x_k^j` : dérivée par rapport à la k-ème composante du **second** argument (entrées de colonne)

L'article utilise le noyau Gaussien (Eq. 1) :
```
k(x^i, x^j; θ) = exp(−|x^i − x^j|² / (2θ²))
```
avec un vecteur de longueurs de corrélation `θ = {θ₁, θ₂, ..., θₘ}` pour les problèmes multidimensionnels.

---

### Vecteur de corrélation augmenté ṙ(x) (page 4, avant Eq. 8)

```
ṙ(x) = { r(x), ∂r(x)/∂x_1, ..., ∂r(x)/∂x_m }^T
```

où `r(x) = [k(x, x^(1)), ..., k(x, x^(n))]^T` (taille n × 1)  
et `∂r(x)/∂x_l = [∂k(x, x^(1))/∂x_l, ..., ∂k(x, x^(n))/∂x_l]^T` (taille n × 1 pour chaque l).  
`ṙ(x)` a donc la taille n(m+1) × 1.

---

### Prédiction GEK trend constant (Eq. 8)

```
ŷ = μ̂ + ṙ(x)^T R̃^{−1} (ẏ − 1μ̂)
```

---

### Vecteur polynomial augmenté Ψ̃ (Eq. 9) — partie GEPCE / trend de GEPCK

```
Ψ̃ = [ Ψ(x),  ∂Ψ(x)/∂x_1,  ...,  ∂Ψ(x)/∂x_m ]^T
```

où `Ψ(x) = {Ψ₀(x), Ψ₁(x), ..., Ψ_{P−1}(x)}^T` est le vecteur de la base polynomiale PCE.  
L'article utilise des **polynômes de Legendre** sur le domaine borné (Table 1 fournit les polynômes jusqu'à l'ordre 4 et leurs dérivées).

La matrice F̃ (gradient-enhanced information matrix) est définie juste après Eq. 9 :
```
F̃ = { Ψ̃(x^(1)), Ψ̃(x^(2)), ..., Ψ̃(x^(n)) }^T
```

---

### Erreur LOOCV pour GEPCK (Eq. 12)

```
e_LOO = Σ_{i=1}^{n+nm} ε²_i / (n + nm)
```

Le dénominateur est `n + nm` (et non `n` comme dans PCK), car l'erreur est calculée sur l'ensemble des informations augmentées (valeurs et gradients).

---

### Matrice B pour le calcul rapide du LOOCV dans GEPCK (Eq. 14)

```
B = [ σ²R̃   F̃  ]
    [ F̃^T    0  ]
```

Formule d'évaluation LOOCV au point i (Eq. 13) :
```
ŷ^{−i}(x^(i)) = − Σ_{j=1}^{n+nm} (B_{ij}/B_{ii}) · y_j + y^(i)
```

La somme porte sur `j = 1` à `n+nm`, cohérent avec la taille de ẏ.

---

### Résumé des différences PCK → GEPCK

| Élément | PCK (sans gradient) | GEPCK |
|---------|--------------------|----|
| Vecteur réponse | y, taille n×1 | ẏ, taille n(m+1)×1 |
| Matrice de corrélation | R, taille n×n | R̃, taille n(m+1)×n(m+1), (m+1)×(m+1) blocs n×n |
| Vecteur trend | Ψ(x), taille P×1 | Ψ̃(x), taille P(m+1)×1 |
| Matrice F | F = {Ψ(x^(1)),...}^T | F̃ = {Ψ̃(x^(1)),...}^T |
| LOOCV dénominateur | n | n + nm |
| Matrice B | [σ²R, F; F^T, 0] | [σ²R̃, F̃; F̃^T, 0] |
| Nb. infos disponibles | n | n + nm = n(m+1) |

---

## Construction de la matrice R dans le code Python (branches 3 et 5)

### Code de base : `uq_eval_Kernel` — branche5.py:777

C'est la seule fonction qui construit R. Elle prend `(X1, X2, theta, options)` et retourne une matrice N1×N2.

Calcul concret aux lignes 880–889 (noyaux stationnaires séparables) :
```python
h = np.abs((X1[idx1[zidx], :] - X2[idx2[zidx], :]) / theta.ravel())
K_flat[zidx.ravel()] = uq_assemble_Kernel(h, K_family, K_type)
```
puis symétrisée + diagonale + nugget aux lignes 941–953.
Pour Matern-5/2, `uq_assemble_Kernel` est à la ligne branche5.py:741.

### Deux points d'appel dans branche3.py

**1. Pendant l'optimisation** — branche3.py:359, dans `uq_Kriging_eval_J_of_theta_ML` :
```python
R = evalR_handle(X, X, theta, CorrOptions)
```
Appelé à chaque itération du solveur L-BFGS-B (ou DE) pour calculer la log-vraisemblance.
C'est la boucle coûteuse — R est reconstruite à chaque theta testé.

**2. Après optimisation** — branche3.py:504, dans `fit_kriging_pck` :
```python
R = evalR(U, U, theta_opt, CorrOptions)
```
Une seule fois, au theta optimal. C'est le R final, stocké dans `fitted['R']` et passé à
`uq_Kriging_calc_auxMatrices` pour calculer Cholesky, β, σ², LOO.

### Chaîne d'appel complète

```
fit_pck  (branche1.py:99)
  └── uq_PCK_calculate_coefficients  (branche3.py:653)
        └── fit_kriging_pck  (branche3.py:460)
              ├── kriging_optimize_theta  (branche3.py:416)
              │     └── uq_Kriging_eval_J_of_theta_ML  (branche3.py:341)
              │           └── uq_eval_Kernel(X, X, theta, ...)  ← R pendant opti
              │
              └── uq_eval_Kernel(U, U, theta_opt, ...)  ← R finale (ligne 504)
                    └── uq_Kriging_calc_auxMatrices(R, F, Y, ...)  (ligne 512)
```

Note : `U` (espace auxiliaire polynomial, Hermite/Legendre) est passé à `uq_eval_Kernel`,
pas `X` brut — c'est la transformation isoprobabiliste de X vers l'espace canonique,
faite dans `pce_eval_design_matrix` (branche3.py:753).

---

## Modification : warm start theta dans le mode 'optimal' (branche3.py:831)

### Comportement d'origine

Dans la boucle `'optimal'` de `uq_PCK_calculate_coefficients`, chaque appel à `fit_kriging_pck`
partait du **même point initial** `theta0.copy()` (moyenne géométrique des bornes), quelle que
soit l'itération `ii`.

### Modification appliquée

Ajout d'une variable `theta_current` initialisée à `theta0` avant la boucle, mise à jour
à chaque itération avec le theta optimal de l'itération précédente :

```python
theta_current = theta0.copy()   # warm start : réinitialisé par output précédent

for ii in range(1, len(idx_ranked) + 1):
    ...
    fitted = fit_kriging_pck(
        U_train, Y[:, oo], F_handle,
        CorrOptions, theta_bounds, theta_current,   # ← au lieu de theta0.copy()
        ...)

    theta_current = fitted['theta']   # warm start pour l'itération suivante
    ...
```

`fitted['theta']` est retourné à la ligne branche3.py:547 par `fit_kriging_pck`.

### Justification

Quand on passe de `ii` à `ii+1` polynômes dans le trend, les hyperparamètres du Kriging
ne changent pas drastiquement. Repartir du theta optimal précédent accélère la convergence
de L-BFGS-B à chaque nouvelle itération (stratégie analogue au BFGS séquentiel décrit dans
Zuhal 2021 pour GEPCK-LAR, section III.B.1).

---

## Modification : initialisation GA (DE+BFGS) sur trend constant au début du mode 'optimal' (branche3.py:831)

### Objectif

Avant d'entrer dans la boucle `for ii`, initialiser `theta_current` via un fit Kriging avec
**trend constant** (Ψ₀=1, F=ones(N,1)) optimisé par Differential Evolution + L-BFGS-B (polish=True
par défaut dans scipy). Ceci remplace l'initialisation par moyenne géométrique des bornes.

### Code ajouté (avant la boucle for ii)

```python
# Initialisation GA+BFGS sur trend constant (Zuhal 2021 Section III.B.1)
F_constant = lambda U: np.ones((U.shape[0], 1))
fitted_constant = fit_kriging_pck(
    U_train, Y[:, oo], F_constant,
    CorrOptions, theta_bounds, theta0.copy(),
    estim_method=estim_method,
    optim_method='de')
theta_current = fitted_constant['theta']
```

`optim_method='de'` → appelle `kriging_optimize_theta(..., method='de')` → `differential_evolution`
avec `polish=True` (défaut scipy) → DE global suivi d'un polissage L-BFGS-B.
Le `theta0.copy()` passé est **ignoré** par DE (DE cherche globalement sans point de départ).

### Problème identifié en 1D

Pour `f(x) = sin(3πx)exp(-x²)` (fonction oscillatoire), le theta optimal pour le trend constant
est très différent du theta optimal pour les trends polynomiaux :
- Trend constant : GP doit capturer TOUTES les oscillations → theta petit (~0.01)
- Trend polynomial : GP modélise les résidus lisses → theta plus grand nécessaire

Démarre L-BFGS-B (dans la boucle) depuis un theta trop petit → convergence vers mauvais minimum
local → LOO=0.64 (dégradé vs moyenne géométrique). Le mode 2D (fonction banane, plus lisse) est
moins affecté.

### Statut

Modification commitée. Comportement 1D dégradé en mode 'optimal' — investigation en cours.
En attendant, le mode 1D est repassé en 'sequential' dans demo_pck.py (non affecté par le bloc GA).

---

## Réflexions architecture GEK — matrice R augmentée

### Où sont définis les noyaux (branche5.py)

- **`uq_assemble_Kernel`** (ligne 707) : formules explicites du noyau en fonction de la distance
  normalisée `h = |xi - xj| / theta`. Exemple Matérn-5/2 séparable :
  `K = ∏_l (1 + √5·h_l + 5/3·h_l²) · exp(-√5·h_l)`
- **`uq_eval_Kernel`** (ligne 777) : fonction mère appelée via `CorrOptions['Handle']`.
  Calcule les distances h depuis X1, X2, theta, puis appelle `uq_assemble_Kernel`.

### Dérivées du noyau k(x, x')

`k(x, x')` a deux arguments distincts. On peut dériver par rapport à la l-ième composante
de l'un **ou** de l'autre :

- `∂k/∂x^l`  : dérivée par rapport à la l-ième composante du **premier** point `x`
- `∂k/∂x'^l` : dérivée par rapport à la l-ième composante du **second** point `x'`

Pour un noyau stationnaire `k(x, x') = φ(x - x')` :
```
∂k/∂x^l = -∂k/∂x'^l
```
car `∂(x-x')/∂x^l = +1` et `∂(x-x')/∂x'^l = -1`.

### Blocs de la matrice R augmentée GEK (N(1+M) × N(1+M))

| Bloc | Contenu | Dérivée |
|---|---|---|
| (réponse, réponse) | k(xi, xj) | aucune |
| (réponse, gradient_l) | ∂k(xi,xj)/∂xj^l | par rapport au **2e argument** |
| (gradient_l, réponse) | ∂k(xi,xj)/∂xi^l | par rapport au **1er argument** |
| (gradient_l, gradient_m) | ∂²k(xi,xj)/∂xi^l∂xj^m | par rapport aux **deux** |

### Architecture proposée pour la fonction noyau dérivée

Deux vecteurs binaires `d1` et `d2` (un par argument de k) indiquant les dimensions dérivées :

```
d1 = [0, 1, 0]  →  dériver k par rapport à x^1  (1er argument)
d2 = [0, 0, 1]  →  dériver k par rapport à x'^2 (2e argument)
```

`kernel_deriv_factory(family, d1, d2)` retourne une fonction `K(X1, X2, theta)` qui
calcule le bloc correspondant. Au plus un "1" dans chaque vecteur pour GEK.

Pour les noyaux séparables, la dérivée se factorise dimension par dimension :
- Une dimension dérivée → remplacer `k_l(h_l)` par `dk_l/dh_l · (±1/theta_l)`
- Deux dimensions dérivées (l≠m) → remplacer `k_l · k_m` par
  `dk_l/dh_l · dk_m/dh_m · (−1/theta_l theta_m)`
- Deux fois la même dimension → remplacer `k_l` par `−d²k_l/dh_l² · (1/theta_l²)`

---

## Design de la factory function de dérivée de noyau

### Interface proposée

```
kernel_deriv_factory(base_kernel, der, der_prime) → f(X1, X2, theta)
```

- `der = i` : dériver par rapport à la i-ème composante du **premier** argument x (0-indexé), ou `None`
- `der_prime = j` : dériver par rapport à la j-ème composante du **second** argument x' (0-indexé), ou `None`
- Retourne une fonction `f(X1, X2, theta) → matrice n×n`

Quatre cas :
- `[None, None]` → k(x, x') tel quel
- `[i, None]`    → ∂k/∂x_i
- `[None, j]`    → ∂k/∂x'_j
- `[i, j]`       → ∂²k/(∂x_i ∂x'_j)

### Formules pour Matérn 5/2 séparable

Noyau 1D : `k_l(h) = (1 + √5h + 5h²/3)·exp(−√5h)` avec `h = |x_l−x'_l|/θ_l`

```
dk_l/dh   = −(5h/3)(1+√5h)·exp(−√5h)
d²k_l/dh² = (5/3)(5h²−√5h−1)·exp(−√5h)
```

Règle de chaîne : `∂h_l/∂x_l = +sign(δ_l)/θ_l`,  `∂h_l/∂x'_l = −sign(δ_l)/θ_l`
Simplification : `sign(δ_l)·h_l = (x_l−x'_l)/θ_l`

**Cas [i, j] avec i ≠ j (en x et x') :**
```
∂²k/(∂x_i ∂x'_j) = −(25/(9θ_i²θ_j²))·(x_i−x'_i)(x_j−x'_j)
                    ·(1+√5h_i)(1+√5h_j)·exp(−√5(h_i+h_j))
                    ·∏_{l≠i,j} k_l(h_l)
```

**Cas [i, i] :**
```
∂²k/(∂x_i ∂x'_i) = (5/3θ_i²)·(1+√5h_i−5h_i²)·exp(−√5h_i)·∏_{l≠i} k_l(h_l)
```

### Construction de uq_augmented_kernel

`uq_augmented_kernel(X1, X2, theta, base_kernel, m)` boucle sur (row_block, col_block) ∈ {0..m}² :
- `der       = row_block − 1` si `row_block > 0` sinon `None`
- `der_prime = col_block − 1` si `col_block > 0` sinon `None`
- Appelle `kernel_deriv_factory(base_kernel, der, der_prime)(X1, X2, theta)`
- Insère le bloc n×n à la position correspondante dans R̃ n(m+1)×n(m+1)
- **Ne pas exploiter la symétrie** de R̃ (car Bloc(k,0) = −Bloc(0,k)^T)



---

## Noyau utilisé pour GEK/GEPCK

**Zuhal 2021** utilise le **noyau Gaussien** (squared exponential) en forme séparable (produit 1D), car C∞ → toutes les dérivées existent. Matérn 5/2 est mentionné comme travail futur.

**Règle d'implémentation pour GEK dans notre code :**
- Si `Type='separable'` (Matérn ou Gaussien) → factory fonctionne directement
- Si `Type='ellipsoidal'` et option GEK demandée → **forcer `Type='separable'`** automatiquement et afficher une notification (ex. sur le graphe ou en log) : *"GEK : Type='ellipsoidal' non supporté, passage automatique à 'separable'"*
- Gaussien reste Gaussien (séparable par nature, aucun changement)

**Vérification UQLab** (code MATLAB `uq_eval_Kernel.m`) :
- `separable` : `h` est une matrice N×M, `prod(..., 2)` → produit sur les dimensions → produit de k_l(h_l)
- `ellipsoidal` : `h = pdist2(..., 'seuclidean', theta')` → distance euclidienne scalaire → vrai Matérn isotropique non séparable → non différentiable dimension par dimension

---

## Implémentation de kernel_deriv_factory (branche5.py)

### Noyaux supportés pour GEK

Deux noyaux uniquement : `'gaussian'` et `'matern-5_2'` (séparables). Tout autre noyau lève une erreur.

### Formules — Noyau Gaussien

`k(x, x') = ∏_l exp(-δ_l²/(2θ_l²))`,  δ_l = x_l − x'_l

| [der, der_prime] | Formule |
|---|---|
| [None, None] | k(x, x') |
| [i, None] | −δ_i/θ_i² · k |
| [None, j] | +δ_j/θ_j² · k |
| [i, i] | (1 − δ_i²/θ_i²)/θ_i² · k |
| [i, j] i≠j | −δ_i·δ_j/(θ_i²·θ_j²) · k |

Tous les cas sont de la forme **coefficient · k(x,x')** — le noyau entier factorise.

### Formules — Noyau Matérn 5/2

`k(x, x') = ∏_l k_l`,  k_l = (1 + a_l + a_l²/3)·exp(−a_l),  a_l = √5|δ_l|/θ_l

| [der, der_prime] | Formule |
|---|---|
| [None, None] | ∏_l k_l |
| [i, None] | −(5/3θ_i²)·δ_i·(1+a_i)·exp(−a_i) · ∏_{l≠i} k_l |
| [None, j] | +(5/3θ_j²)·δ_j·(1+a_j)·exp(−a_j) · ∏_{l≠j} k_l |
| [i, i] | (5/3θ_i²)·(1+a_i−a_i²)·exp(−a_i) · ∏_{l≠i} k_l |
| [i, j] i≠j | −(25/9θ_i²θ_j²)·δ_i·δ_j·(1+a_i)(1+a_j)·exp(−a_i−a_j) · ∏_{l≠i,j} k_l |

### Helpers

**`_prod_excl(K_uni)`** : K_uni (n1,n2,M) → K_excl (n1,n2,M) où K_excl[:,:,l] = ∏_{m≠l} K_uni[:,:,m]. Algorithme prefix/suffix O(M) sans division.

Pour le cas [i,j] i≠j, ∏_{l≠i,j} k_l = K_excl[:,:,i] / K_uni[:,:,j] (valide car k_l > 0 toujours).

### Architecture

```
kernel_deriv_factory(family, der, der_prime)
  ├── 'gaussian'   → f(X1,X2,theta) : appelle uq_eval_Kernel pour k, puis coefficient·k
  ├── 'matern-5_2' → f(X1,X2,theta) : calcule a=√5|δ|/θ, K_uni, _prod_excl, puis formule
  └── autre        → raise ValueError
```

Les callables retournés par la factory prennent **(X1, X2, theta)** — valeurs réelles, pas h.
`uq_eval_Kernel` est réutilisé pour k dans le cas Gaussien. Pour Matérn, K_uni est calculé directement depuis a (évite un appel par dimension).

### Utilisation prévue

`uq_augmented_kernel(X1, X2, theta, family, m)` boucle sur (row_block, col_block) ∈ {0..m}², appelle `kernel_deriv_factory(family, row_block-1 or None, col_block-1 or None)(X1, X2, theta)` et assemble R̃ n(m+1)×n(m+1). **Ne pas exploiter la symétrie** de R̃ (Bloc(k,0) = −Bloc(0,k)^T).

---

## RÉSUMÉ SESSION POST-COMPACTAGE (28/05/2026)

### Ce qui a été implémenté et testé : 3 fonctions GEK dans branche5.py

#### Fichier : `branche5.py`, lignes 964–1109

Trois fonctions ajoutées à la fin du fichier, après `uq_eval_Kernel` :

**`_prod_excl(K_uni)`** (ligne 964)
- Entrée : `K_uni` (n1, n2, M)
- Sortie : `K_excl` (n1, n2, M) où `K_excl[:,:,l] = ∏_{m≠l} K_uni[:,:,m]`
- Algorithme prefix/suffix O(M) sans division : tableaux `pre` et `suf` de taille (n1,n2,M+1), résultat = `pre[:,:,:M] * suf[:,:,1:]`
- Utilisée dans `kernel_deriv_factory` pour Matérn 5/2 cas [i,j] i≠j : `∏_{l≠i,j} k_l = K_excl[:,:,i] / K_uni[:,:,j]` (valide car k_l > 0 toujours)

**`kernel_deriv_factory(family, der, der_prime)`** (ligne 979)
- Entrée : `family` ('gaussian' ou 'matern-5_2'), `der` (int ou None), `der_prime` (int ou None)
- Sortie : fonction `f(X1, X2, theta) → (n1, n2)`
- Quatre cas selon (der, der_prime) :

| [der, der_prime] | Gaussien | Matérn 5/2 |
|---|---|---|
| [None, None] | k via uq_eval_Kernel | K_uni.prod(axis=2) |
| [i, None] | −δ_i/θ_i² · k | −(5/3θ_i²)·δ_i·(1+a_i)·exp(−a_i)·K_excl[:,:,i] |
| [None, j] | +δ_j/θ_j² · k | +(5/3θ_j²)·δ_j·(1+a_j)·exp(−a_j)·K_excl[:,:,j] |
| [i, i] | (1−δ_i²/θ_i²)/θ_i² · k | (5/3θ_i²)·(1+a_i−a_i²)·exp(−a_i)·K_excl[:,:,i] |
| [i, j] i≠j | −δ_i·δ_j/(θ_i²θ_j²)·k | −(25/9θ_i²θ_j²)·δ_i·δ_j·(1+a_i)(1+a_j)·exp(−a_i−a_j)·K_excl[:,:,i]/K_uni[:,:,j] |

- Variables : `delta = X1[:,np.newaxis,:] - X2[np.newaxis,:,:]` (n1,n2,M) ; `a = √5·|delta|/theta` (n1,n2,M) ; `K_uni = (1+a+a²/3)·exp(-a)` (n1,n2,M)
- Tout autre family → `raise ValueError`

**`uq_assemble_global_Kernel(X, theta, family)`** (ligne 1058)
- Entrée : `X` (n,m), `theta` (m,), `family` str
- Sortie : `R_tilde` (n(m+1), n(m+1))
- Double boucle rb, cb ∈ {0..m} : `der = rb-1 si rb>0 sinon None`, `dp = cb-1 si cb>0 sinon None`
- Place `kernel_deriv_factory(family, der, dp)(X, X, theta)` dans le bloc `[rb*n:(rb+1)*n, cb*n:(cb+1)*n]`

#### Propriété mathématique clé — R̃ EST SYMÉTRIQUE

Contrairement à la formulation Bouhlel 2019 (4 blocs, non symétrique), la formulation dimension-major de Zuhal 2021 produit une matrice **symétrique**.

Preuve : Block(0,k)[j,i] = ∂k(xj,xi)/∂xi_{k-1} = −∂k(xj,xi)/∂xj_{k-1} = −∂k(xi,xj)/∂xi_{k-1} = Block(k,0)[i,j]  
→ Block(0,k)^T = Block(k,0) pour tout k.  
Vérifié numériquement : max|R̃ − R̃^T| = 0 à précision machine.  
Valeurs propres positives (min ≈ 1e-3 pour DOE 5pts 2D, θ=[0.5,0.5]).

Note : la docstring de `uq_assemble_global_Kernel` contient la preuve complète.

#### Tests : `test_gek_kernel.py` — 39 PASS / 0 FAIL

Sections :
1. `_prod_excl` (4 tests) : M=1 produit vide=1, M=2 swap, M=3 chaque slice = produit des autres, cohérence K*K_excl=total
2. `kernel_deriv_factory` structurel (8 tests) : [None,None]=uq_eval_Kernel, antisymétrie [i,None]=−[None,i], diagonale nulle à xi=xj, [i,i] symétrique
3. `kernel_deriv_factory` FD (16 tests) : [i,None] vs DF/X1, [None,j] vs DF/X2, [i,i] vs DF de [i,None]/X2, [0,1] vs DF de [None,1]/X1 — pour gaussian et matern-5_2
4. `uq_assemble_global_Kernel` (11 tests) : shape, Bloc(0,0)=uq_eval_Kernel, symétrie, Block(k,0)=−Block(0,k), PSD, ValueError, 1D

---

### Prochaine étape : `poly_deriv_handle_factory` (NON ENCORE CODÉE)

#### Objectif

Créer l'équivalent de `kernel_deriv_factory` pour les polynômes PCE.  
Factory : `poly_deriv_handle_factory(Idx_sel, poly_types, der)` → `F_der_handle(U) → (N, P_sel)`  
où chaque colonne k = `∂Ψ_k(U)/∂U_der`.

Utilisée pour construire la matrice F̃ augmentée de GEPCK (Zuhal 2021 Eq. 9) :
```
F̃ = [ F(U)            ]   ← lignes 0..n-1     : Ψ_k(U)
    [ ∂F/∂U_0(U)      ]   ← lignes n..2n-1    : ∂Ψ_k/∂U_0
    [ ...              ]
    [ ∂F/∂U_{m-1}(U)  ]   ← lignes mn..(m+1)n-1
```

#### Pourquoi ça marche (règle du produit sur base séparable)

`Ψ_k(U) = ∏_i φ_{α_ki}(U_i)` — chaque facteur ne dépend que de U_i.

Donc `∂φ_i(U_i)/∂U_der = 0` pour i≠der, et la règle du produit donne UN SEUL terme :

```
∂Ψ_k/∂U_der = φ'_{α_k,der}(U_der) · ∏_{i≠der} φ_{α_ki}(U_i)
```

`uq_PCE_create_Psi` fait le bon produit automatiquement si on remplace `uv[:, der, :]` par les dérivées.

#### Architecture concrète

**Étape 1 — créer `uq_eval_hermite_deriv(ORDER, X)` dans branche5.py**
- Même signature et shape que `uq_eval_hermite` : (N, ORDER+1)
- Colonne k = H'_k(x) pour les N points scalaires x
- Formule : `H'_0 = 0`, `H'_k(x) = √k · H_{k-1}(x)` pour k≥1

```python
def uq_eval_hermite_deriv(ORDER, X):
    X = np.asarray(X, dtype=float).reshape(-1)
    out = np.zeros((X.shape[0], ORDER + 1))
    if ORDER >= 1:
        H = uq_eval_hermite(ORDER, X)
        for k in range(1, ORDER + 1):
            out[:, k] = np.sqrt(k) * H[:, k - 1]
    return out
```

**Étape 2 — créer `uq_eval_legendre_deriv(ORDER, X)` dans branche5.py**
- Même signature et shape : (N, ORDER+1)
- Colonne k = L'_k(x) pour les N points scalaires x (L_k = polynôme de Legendre orthonormal)
- **Normalisation confirmée numériquement** : `uq_eval_legendre(2,[1.0])` → `[1, √3, √5]`
  donc `L_n = √(2n+1) · P_n` (orthonormal par rapport à la densité uniforme 1/2 sur [-1,1])
  **PAS** `√((2n+1)/2) · P_n` (qui serait pour la densité 1)
- Recurrence : `P'_0=0, P'_1=1, P'_n=(2n−1)·P_{n-1}+P'_{n-2}`, puis `L'_n = √(2n+1) · P'_n`
- Valeurs clés : `L'_0=0`, `L'_1=√3` (constant), `L'_2=3√5·x`

```python
def uq_eval_legendre_deriv(ORDER, X):
    X = np.asarray(X, dtype=float).reshape(-1)
    N = X.shape[0]
    out = np.zeros((N, ORDER + 1))
    if ORDER == 0:
        return out
    L  = uq_eval_legendre(ORDER, X)
    ns = np.arange(ORDER + 1, dtype=float)
    c  = np.sqrt(2 * ns + 1)              # L_n = sqrt(2n+1)*P_n
    P  = L / c[np.newaxis, :]             # récupère P_n non-normalisé
    dP = np.zeros((N, ORDER + 1))
    dP[:, 1] = 1.0
    for n in range(2, ORDER + 1):
        dP[:, n] = (2 * n - 1) * P[:, n - 1] + dP[:, n - 2]
    out = c[np.newaxis, :] * dP
    return out
```

**Étape 3 — écrire `make_trend_handle_deriv(selected_idx, Indices, poly_types, der)` dans branche3.py**

Même structure que `make_trend_handle`, paramètre `der` en plus :

```python
def make_trend_handle_deriv(selected_idx, Indices, poly_types, der):
    Idx_sel = Indices[np.array(selected_idx), :]
    p_types = poly_types

    def F_der_handle(U):
        uv = uq_PCK_eval_unipoly(U, Idx_sel, p_types)   # (N, M, P+1)
        P  = uv.shape[2] - 1
        pt = p_types[der].lower()
        if pt == 'hermite':
            uv[:, der, :] = uq_eval_hermite_deriv(P, U[:, der])
        elif pt == 'legendre':
            uv[:, der, :] = uq_eval_legendre_deriv(P, U[:, der])
        Psi = uq_PCE_create_Psi(Idx_sel, uv)
        # uq_PCE_create_Psi saute les degrés 0 (ligne : aa = Indices[:,mm] > 0).
        # Pour la dérivée, φ'_0 = 0 (dérivée d'une constante).
        # Ces colonnes ne sont pas touchées par uq_PCE_create_Psi → annuler explicitement.
        Psi[:, Idx_sel[:, der] == 0] = 0.0
        return Psi

    return F_der_handle
```

**Pourquoi `Psi[:, Idx_sel[:, der] == 0] = 0.0`** :  
`uq_PCE_create_Psi` ne lit jamais `uv[:, der, 0]` (skip degré 0, ligne `aa = Indices[:,mm] > 0`).  
Or `φ'_0 = 0` (dérivée d'une constante). Ces colonnes restent à leur valeur initiale 1 au lieu de 0.

#### Vérification effectuée (test_make_trend_deriv.py) — **4/4 PASS**

Base Hermite `[[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]]`, H_0=1, H_1=x, H_2=(x²-1)/√2 :
- `der=0` : colonnes attendues `[0, 1, 0, √2·U0, U1, 0]` → **analytique PASS + FD PASS**
- `der=1` : colonnes attendues `[0, 0, 1, 0, U0, √2·U1]` → **analytique PASS + FD PASS**

Base Legendre `[[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]]`, L_0=1, L_1=√3·x, L_2=√5·(3x²-1)/2 :
- `der=0` : colonnes attendues `[0, √3, 0, 3√5·V0, 3·V1, 0]` → **analytique PASS + FD PASS**
- `der=1` : colonnes attendues `[0, 0, √3, 0, 3·V0, 3√5·V1]` → **analytique PASS + FD PASS**

Bugs identifiés et corrigés :
1. **Degrés 0 non annulés** : `uq_PCE_create_Psi` skipppe `uv[:, der, 0]` (ligne `aa = Indices[:,mm] > 0`) → les colonnes où `Idx_sel[:, der]==0` restaient à 1. Fix : `Psi[:, Idx_sel[:, der] == 0] = 0.0`
2. **Mauvaise constante Legendre** : `c_n = √((2n+1)/2)` (faux) → `c_n = √(2n+1)` (correct, vérifié numériquement)

---

## RÉSUMÉ SESSION 28/05/2026 (suite)

### Modifications branche3.py

#### 1. `make_trend_handle` — nettoyage signature
Les 3 arguments inutilisés `orig_marg`, `orig_cop`, `aux_marg` supprimés. Absents du MATLAB original (`@(X,dummy) uq_evalModel(myPIP, X)`). La transformation X→U est faite en amont dans `pce_eval_design_matrix` — le handle reçoit U directement.

Signature avant : `make_trend_handle(selected_idx, Indices, poly_types, orig_marg, orig_cop, aux_marg)`  
Signature après : `make_trend_handle(selected_idx, Indices, poly_types)`

3 call sites mis à jour (make_trend_global_handle + mode sequential + mode optimal).

#### 2. `make_trend_global_handle` — ajoutée (nested dans `uq_PCK_calculate_coefficients`)

```python
def make_trend_global_handle(selected_idx, Indices, poly_types):
    Idx_sel = Indices[np.array(selected_idx), :]
    M_loc   = Idx_sel.shape[1]
    F0 = make_trend_handle(selected_idx, Indices, poly_types)
    Fk = [make_trend_handle_deriv(selected_idx, Indices, poly_types, k)
          for k in range(M_loc)]
    def F_global(U):
        blocks = [F0(U)] + [fk(U) for fk in Fk]
        return np.vstack(blocks)
    return F_global
```

Retourne F̃(U) de shape `(N*(M+1), P_sel)` — matrice augmentée GEPCK (Zuhal 2021, Eq. 9).  
Logique identique à la version dans `test_make_trend_deriv.py` (déjà testée 4/4 PASS dans ce fichier).

### Modifications branche5.py

#### 3. `uq_assemble_global_Kernel` — généralisée `(X, theta, family)` → `(X1, X2, theta, family)`

Nécessaire pour la cross-corrélation r̃₀ en prédiction (X1=X_test ≠ X2=X_train).  
Blocs passent de `(n, n)` à `(n1, n2)`. `f(X, X, theta)` → `f(X1, X2, theta)`.

Cas Gram (X1==X2) : matrice symétrique et SDP (inchangé).  
Cas non-Gram (X1≠X2) : cross-corrélation rectangulaire `(n1*(m+1), n2*(m+1))`.

#### 4. `uq_eval_global_Kernel(X1, X2, theta, options)` — nouvelle fonction

Equivalent de `uq_eval_Kernel` pour GEPCK. Même organisation :

```
PCK :   CorrOptions['Handle'] = uq_eval_Kernel
          └── uq_eval_Kernel(X1, X2, theta, options)
                ├── gère nugget, isotropique, Gram
                └── appelle uq_assemble_Kernel(h, family, type)

GEPCK : CorrOptions['Handle'] = uq_eval_global_Kernel
          └── uq_eval_global_Kernel(X1, X2, theta, options)
                ├── gère nugget, Gram
                └── appelle uq_assemble_global_Kernel(X1, X2, theta, family)
```

Options lues : `'Family'` (obligatoire), `'Nugget'` (défaut 0.0).  
Nugget ajouté uniquement si Gram et nugget ≠ 0.

#### Tests — 39/39 PASS (test_gek_kernel.py, 28/05/2026)

Aucune régression. Les appels dans test_gek_kernel.py utilisaient déjà la forme 4-args.

### Architecture des fonctions make_ (branche3.py, toutes nested dans uq_PCK_calculate_coefficients)

| Fonction | Args | Rôle |
|---|---|---|
| `make_trend_handle(sel, Idx, pt)` | 3 | F(U) → (N, P_sel) |
| `make_trend_handle_deriv(sel, Idx, pt, der)` | 4 | ∂F/∂U_der(U) → (N, P_sel) |
| `make_trend_global_handle(sel, Idx, pt)` | 3 | F̃(U) → (N*(M+1), P_sel) |
| `fit_kriging_gepck(U, Y_aug, F_global_handle, CorrOptions, ...)` | — | Clone de `fit_kriging_pck` pour GEPCK |

### `fit_kriging_gepck` — ligne 561 branche3.py

Clone direct de `fit_kriging_pck`. Différences :

| | `fit_kriging_pck` | `fit_kriging_gepck` |
|---|---|---|
| Réponse | `Y` (N,) | `Y_aug` (N*(M+1),) |
| Trend | `F_handle(U)` → (N, P_sel) | `F_global_handle(U)` → (N*(M+1), P_sel) |
| Corrélation | `uq_eval_Kernel` → R (N, N) | `uq_eval_global_Kernel` → R̃ (N*(M+1), N*(M+1)) |
| N dans vraisemblance | `U.shape[0]` | `len(Y_aug)` = N*(M+1) |
| varY pour LOO | `np.var(Y)` | `np.var(Y_aug[:N])` — N valeurs seules |
| Clés retournées | `'F'`, `'R'`, `'F_handle'` | `'F_tilde'`, `'R_tilde'`, `'F_global_handle'` |

Tout le reste est partagé : `kriging_optimize_theta`, `uq_Kriging_calc_auxMatrices`, `uq_Kriging_calc_beta`, `uq_Kriging_calc_sigmaSq`, `uq_Kriging_calc_KFold`.

### Convention de dérivation dans `uq_assemble_global_Kernel` et `uq_eval_global_Kernel`

**Convention actuelle (Zuhal 2021)** — `der` dérive le 1er argument, `dp` le 2e :

```
bloc(rb, cb) = kernel_deriv_factory(family, der=cb-1, dp=rb-1)(X1, X2, theta)
  bloc(0, k) = dk/dX1_{k-1}  →  Cov(dy/dx_{k-1}(X1), y(X2))
  bloc(k, 0) = dk/dX2_{k-1}  →  Cov(y(X1), dy/dx_{k-1}(X2))
```

`r̃₀` dans le cas non-Gram : `der=cb-1, dp=None` — = premier bloc-ligne de R̃ → interpolation exacte (T5 PASS).

**Convention précédente (covariance GP / Morris 1993)** — `der` et `dp` échangés :

```
bloc(rb, cb) = kernel_deriv_factory(family, der=rb-1, dp=cb-1)(X1, X2, theta)
```

Les deux conventions donnent R̃ symétrique, SDP et BLUP exact — à condition que r̃₀ soit calculée avec la même convention. Le switch vers Zuhal est fait pour coller à l'article. Tests : 39/39 PASS (`test_gek_kernel.py`) + 5/5 PASS (`test_eval_global_kernel.py`).

