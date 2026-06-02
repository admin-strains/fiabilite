# Résumé de session — Implémentation PC-Kriging Python
**Date :** 26 mai 2026
**Objectif de la session :** Coder PC-Kriging (Polynomial Chaos Kriging) en Python fidèle à UQLab MATLAB, à tester sur une fonction simple, pour l'intégrer ensuite dans AC_pure_flexion.py (fiabilité flexion BA).

---

## 0. Protocole de mise à jour mémoire

Ce fichier est le point d'entrée unique pour la session PC-Kriging.

**Étape 1 — Ce fichier**
Donne le contexte complet : ce qui a été codé, les formules mathématiques, l'état d'avancement, les bugs rencontrés, ce qui reste à faire.

**Étape 2 — Les fichiers Python créés**
```
C:\_workingDir\_SF\test flexion\
├── kernels.py           ← fonctions de corrélation
├── pce_trend.py         ← trend PCE via chaospy + LARS
├── pck.py               ← classe PCKriging principale
└── AC_pckrg_claude.py   ← fichier principal avec tests progressifs
```

**Étape 3 — Source de référence**
UQLab MATLAB (lire si besoin de vérifier une formule) :
```
C:\Users\semia.frikha\Downloads\UQLab_Rel2.2.0\
├── modules\uq_model\builtin\uq_metamodel\PCK\
│   ├── uq_PCK_calculate_coefficients.m  ← orchestration
│   ├── uq_PCK_eval.m                    ← prédiction
│   └── uq_PCK_initialize.m              ← options
├── modules\uq_model\builtin\uq_metamodel\Kriging\
│   ├── eval\uq_Kriging_eval.m           ← formules BLUP
│   ├── calc\uq_Kriging_calc_auxMatrices.m
│   ├── calc\uq_Kriging_calc_beta.m
│   ├── calc\uq_Kriging_calc_sigmaSq.m
│   └── optimizer\uq_Kriging_eval_J_of_theta_ML.m ← MLE
└── lib\uq_kernel\uq_eval_Kernel.m       ← kernel
```

**Article de référence :** Moustapha et al. (2022), *Active learning for structural reliability: Survey, general framework and benchmark*, Structural Safety.
→ PC-Kriging = meilleur surrogate dans le benchmark (PCK + SuS + U + stopping critère combined)

---

## 1. Contexte et motivation

### Pourquoi PC-Kriging ?
Dans AC_pure_flexion.py, on utilise actuellement KRG (Kriging classique) et GEK (Gradient-Enhanced Kriging). D'après le benchmark Moustapha 2022 :
- **PC-Kriging** (PCK) est systématiquement supérieur au Kriging classique, surtout en dimensions moyennes (M < 20)
- PCK est disponible dans UQLab MATLAB mais pas en Python directement
- L'objectif est d'avoir une version Python standalone pour l'intégrer dans le workflow de fiabilité

### PC-Kriging = ?
```
Y(x) = f_PCE(x)   [trend polynomial chaos]
     + Z(x)        [processus Gaussien de moyenne nulle]

f_PCE = Σ aα · Ψα(X)   (base orthogonale × coefficients sparse via LARS)
Z     = corrélation par kernel Matérn 5/2
```

---

## 2. Formules mathématiques implémentées

Toutes vérifiées dans le source MATLAB de UQLab.

### 2.1 Kernel (kernels.py)
Distance ellipsoïdale (défaut UQLab) :
```
h_ij = sqrt(Σ_k ((x1_ik - x2_jk) / θ_k)²)   ← seuclidean distance
```
Matérn 5/2 (défaut UQLab) :
```
K(h) = (1 + √5·h + 5/3·h²) · exp(-√5·h)
R = K(X_train, X_train) + nugget·I    (nugget = 1e-4)
r₀ = K(X_test, X_train)               (nugget = 0 ici, fidèle MATLAB)
```

### 2.2 Matrices auxiliaires (pck.py — _compute_aux_matrices)
```
L = chol(R)  (upper triangular, R = Lᵀ L)
Ỹ = L⁻ᵀ · Y
F̃ = L⁻ᵀ · F
[Q1, G] = qr(F̃)   (QR économe)
FᵀR⁻¹ = solve_triangular(L, F.T, lower=False)  (2 triangulaires)
FᵀR⁻¹F = FᵀR⁻¹ · F
```
Fallback si Cholesky échoue : pseudo-inverse `Rinv = pinv(R)`.

### 2.3 Coefficients trend β (pck.py — _compute_beta)
Via QR (numeriquement stable, fidèle uq_Kriging_calc_beta.m) :
```
β = G⁻¹ · Q1ᵀ · Ỹ
```
Fallback OLS :
```
β = (FᵀR⁻¹F)⁻¹ · FᵀR⁻¹ · Y
```

### 2.4 Variance GP σ² (pck.py — _compute_sigma_sq)
Via bypass QR (fidèle uq_Kriging_calc_sigmaSq.m, calc_sigmaSqMLBypass) :
```
z = Ỹ - Q1·Q1ᵀ·Ỹ
σ² = (1/N) · ||z||²
```

### 2.5 MLE (pck.py — _mle_objective)
```
J(θ) = 0.5 · (N·log(2π·σ²) + log|R| + N)
```
Optimisation : `scipy.optimize.minimize`, L-BFGS-B, sur `log(θ)` (bornes [-4.6, 4.6]).
Multi-départ : n_optim_starts = 10 par défaut.

### 2.6 Prédiction BLUP (pck.py — predict)
```
Ŷ(x₀) = f₀·β + r₀ · R⁻¹ · (Y - F·β)

D1[i] = ||L⁻¹ r₀[i,:]||²               ← diag(r₀ R⁻¹ r₀ᵀ)
u₀    = FᵀR⁻¹·r₀ᵀ - f₀ᵀ                ← (n_poly, N_test)
D2[i] = u₀[:,i]ᵀ (FᵀR⁻¹F)⁻¹ u₀[:,i]   ← diag via lstsq
Var(x₀) = σ² · (1 - D1 + D2)
```

### 2.7 LOO error (pck.py — _compute_loo)
```
h_ii = ||Q1[i,:]||²    ← diag de la hat matrix
e_loo[i] = (Y[i] - F[i,:]·β) / (1 - h_ii)
LOO = mean(e_loo²) / Var(Y)
```

### 2.8 PCE Trend (pce_trend.py)
1. `chaospy.generate_expansion(degree, joint, normed=True)` → base orthonormale
2. Évaluation : `expansion(*X.T)` → Ψ (N, P)
3. LARS (`sklearn.linear_model.Lars`, fit_path=True)
4. À chaque étape LARS : refit OLS sur l'ensemble actif → calcule LOO analytique
5. Early stopping : si LOO ne s'améliore pas sur 2 étapes → arrêt
6. Ranking par |coefficient| décroissant → utilisé par sequential/optimal

---

## 3. Architecture des fichiers Python

### kernels.py
```python
eval_kernel(X1, X2, theta, family='matern-5_2', nugget=1e-4) → np.ndarray (N1, N2)
_apply_kernel(h, family)  # Matérn 5/2 | 3/2 | Gaussian | Exponential
```

### pce_trend.py
```python
class PCETrend:
    __init__(distributions, degree=range(1,4))
    fit(X, Y)              # calibre LARS, sélectionne polynômes, ranking
    eval(X)                # retourne Ψ(X) complet (N, P_total)
    eval_active(X)         # retourne Ψ réduit aux actifs (N, n_active)
    eval_subset(X, n_poly) # retourne les n_poly premiers du ranking (N, n_poly)
    n_polynomials          # property : nb polynômes actifs
    ranked_indices         # liste des indices dans le ranking LARS
    loo_error              # LOO error du meilleur degré retenu
```

### pck.py
```python
class PCKriging:
    __init__(mode='sequential', pce_degree=range(1,4),
             corr_family='matern-5_2', nugget=1e-4, n_optim_starts=5)
    fit(X, Y, distributions)       # calibre PCE trend + Kriging
    predict(X_test, return_std=False) → Y_mean [, Y_std]
    _fit_sequential(...)           # 1 calibration Kriging (tous les poly)
    _fit_optimal(...)              # minimise LOO sur sous-ensembles croissants
    _calibrate_kriging(...)        # optimisation MLE multi-départ
    loo_error                      # LOO final du modèle retenu
```

**Format distributions :**
```python
distributions = [
    {'type': 'uniform', 'parameters': [a, b]},
    {'type': 'gaussian', 'parameters': [mu, sigma]},
    {'type': 'lognormal', 'parameters': [mu_log, sigma_log]},
]
```

### AC_pckrg_claude.py (fichier principal)
```python
test_kernels()               # TEST 1 : vérifie K(x,x)=1, symétrie, Matérn 5/2 à la main
test_pce_trend()             # TEST 2 : LARS sur polynôme exact de degré 2
test_pck_xsinx(mode)         # TEST 3 : f(x)=x·sin(x), vérifie interpolation exacte
test_comparison_seq_opt()    # TEST 4 : Sequential vs Optimal, figures comparatives
```

---

## 4. Dépendances requises

```
numpy
scipy
scikit-learn   (LARS via sklearn.linear_model.Lars)
chaospy        ← BASE POLYNOMIALE ORTHOGONALE — PROBLÈME (voir section 5)
matplotlib
```

---

## 5. Bug en cours — chaospy non installable

### Symptôme
```
pip install chaospy
→ ERROR: Failed building wheel for numpoly
→ error: Unable to find a compatible Visual Studio installation.
```

### Cause
`numpoly` (dépendance de `chaospy`) contient du code Cython (`.pyx`) qui nécessite un compilateur C (Visual Studio Build Tools) pour être compilé sur Windows. Ce compilateur n'est pas installé sur la machine.

### Solutions à tester en priorité (session suivante)
**Option A (la plus simple) :** Installer Visual Studio Build Tools
```
winget install Microsoft.VisualStudio.2022.BuildTools
```
Puis relancer `pip install chaospy`.

**Option B :** Chercher un wheel précompilé pour Windows Python 3.10
```
pip install chaospy --find-links https://pypi.org/simple/chaospy/
# ou tenter avec conda :
conda install -c conda-forge chaospy
```

**Option C (si les options A/B échouent) :** Remplacer chaospy par une implémentation pure-numpy des polynômes orthogonaux dans `pce_trend.py`.
- Pour Uniform[a,b] → Legendre : récurrence à 3 termes connue
- Pour Gaussian → Hermite physiciste : récurrence connue
- Voir `uq_eval_legendre.m` et `uq_eval_hermite.m` dans UQLab pour les formules exactes
- C'est faisable (~50 lignes) et élimine toute dépendance externe pour les distributions courantes (Uniform, Gaussian)

**Option D :** `orthopy` ou `quadpy` — bibliothèques Python pour polynômes orthogonaux, généralement sans extension C.
```
pip install orthopy
```

### État au moment de l'arrêt de la session
- Le code est écrit et correct mathématiquement
- Le seul blocage est l'installation de chaospy
- Aucune ligne de code n'a été exécutée (impossible sans chaospy)

---

## 6. Ce qui reste à faire

### Immédiat (session suivante)
1. **Régler le problème chaospy** (voir options A/B/C/D section 5)
2. **Lancer AC_pckrg_claude.py** → vérifier les 4 tests
3. **Débugger si nécessaire** (les formules sont correctes mais une erreur de signe ou d'ordre de factorisation est possible)

### Vérifications à faire une fois le code tournant
- [ ] TEST 1 kernel : K(x,x)=1 pour tous les kernels sans nugget
- [ ] TEST 2 PCE : LOO < 0.01 sur un polynôme de degré 2 exact
- [ ] TEST 3 PCK sequential : variance ≈ 0 aux points d'entraînement (interpolation)
- [ ] TEST 3 PCK optimal : idem
- [ ] TEST 4 : Sequential vs Optimal donnent des LOO comparables sur x·sin(x)

### Intégration dans AC_pure_flexion.py (après validation)
4. Créer une branche `do_PCK` dans AC_pure_flexion.py analogue à `do_KRG`
5. Le surrogate PCK.predict(X, return_std=True) remplace l'appel Kriging dans FORM
6. Les distributions à passer : `fc` ~ LogNormal, `fy` ~ Normal (voir section 1 du global_resume_session_2404.md)

---

## 7. Choix d'implémentation et justifications

| Choix | Raison |
|-------|--------|
| `scipy.optimize.minimize` L-BFGS-B sur `log(θ)` | Fidèle à UQLab. log(θ) évite les contraintes de positivité. |
| Multi-départ (n_optim_starts) | UQLab utilise HGA (Genetic Algorithm). L-BFGS-B + multi-départ est une approche locale robuste mais suffisante pour commencer. |
| nugget = 0 sur r₀ (cross-corrélation) | Fidèle au code MATLAB : `CrossCorOpts.Nugget = 0` explicitement. |
| LARS + OLS hybrid (refit OLS sur active set) | UQLab `LARS.HybridLoo = true` par défaut. Plus précis que les coefficients LARS purs. |
| LOO early stopping sur 2 étapes consécutives | UQLab `LARS.LarsEarlyStop = true`. |
| `eval_subset(X, n_poly)` pour le mode optimal | Permet de tester des sous-ensembles croissants [poly_1], [poly_1, poly_2], ... sans recalculer la base. |
| Cholesky upper triangular (`np.linalg.cholesky().T`) | UQLab : `cholR` est upper triangular (R = cholR' * cholR). Attention : numpy retourne lower par défaut → `.T` obligatoire. |

---

## 8. Extraits de code MATLAB de référence clés

### Formule MLE (uq_Kriging_eval_J_of_theta_ML.m)
```matlab
J = 0.5 * (N * log(2*pi*sigmaSq) + logDetR + N);
```

### Formule sigma² bypass (uq_Kriging_calc_sigmaSq.m)
```matlab
z = Ytilde - Q1*transpose(Q1)*Ytilde;
sigmaSq = 1/N * (transpose(z) * z);
```

### Variance prédiction (uq_Kriging_eval.m)
```matlab
D1 = uq_Kriging_calc_DiagOfCongruent(r0, R);  % diag(r0 R^{-1} r0')
u0 = FTRinv * r0.' - f0.';
D2 = uq_Kriging_calc_DiagOfCongruent(transpose(u0), FTRinvF);
YSigmaOO = sigmaSQ * (ones(size(D1)) - D1 + D2);
```

### Mode optimal (uq_PCK_calculate_coefficients.m)
```matlab
for ii = 1:length(idxranking{oo})
    % ajouter le ii-ème polynôme au trend
    kopts.Trend.Handle = @(X,dummy) uq_evalModel(myPIP, X);
    myKriging(ii) = uq_createModel(kopts, '-private');
    CompCrit(ii) = myKriging(ii).Error.LOO;
end
[~, iidx] = min(CompCrit);
myPCKrigingoo = myKriging(iidx);
```

### Nugget = 0 sur cross-corrélation (uq_Kriging_eval.m)
```matlab
CrossCorOpts = GPCorrOptions;
CrossCorOpts.Nugget = 0;  % force nugget to 0
r0 = evalR_handle(U0, U, theta, CrossCorOpts);
```

---

## 9. Contexte plus large (lien avec AC_pure_flexion.py)

Ce travail s'inscrit dans le projet de fiabilité structurale de la poutre BA en flexion pure :
- Modèle HF = STRAINS (code FEM propriétaire, ~1s/appel)
- FORM = méthode d'analyse de fiabilité (iHLRF)
- Surrogate = remplace les appels STRAINS pour accélérer FORM
- Actuellement dans AC_pure_flexion.py : KRG (Kriging via openturns) et GEK (Gradient-Enhanced Kriging)
- **PC-Kriging viendra s'ajouter** comme nouvelle branche `do_PCK`

Voir `global_resume_session_2404.md` pour tout le contexte sur AC_pure_flexion.py.

---

## 10. Pour démarrer la session suivante

```
Lire ce fichier.
Lancer : python AC_pckrg_claude.py
Si erreur chaospy → appliquer Option A ou B de la section 5.
Si l'Option C est choisie → modifier pce_trend.py pour remplacer chaospy par récurrence pure-numpy.
```

**Commande de test rapide une fois chaospy installé :**
```bash
cd "C:\_workingDir\_SF\test flexion"
python AC_pckrg_claude.py
```

Résultats attendus :
- TEST 1 : toutes les lignes "OK"
- TEST 2 : LOO < 0.1
- TEST 3 : max std aux points d'entraînement < 0.1, figure PNG générée
- TEST 4 : figure comparative Sequential vs Optimal générée
