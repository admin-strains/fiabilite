# GEPCK — Architecture et plan d'implémentation

**Créé le 28/05/2026 — à lire en priorité dans toute nouvelle session GEPCK.**  
Source principale : Zuhal et al. 2021, AIAA Journal, https://doi.org/10.2514/1.J059905

---

## 1. Structure des fichiers (copie essentielle du MD principal)

```
branche1.py        ← point d'entrée public (fit_pck, predict_pck)
branche2.py        ← parsing options (uq_PCK_initialize, uq_process_option)
branche3.py        ← fit Kriging + PCE (cœur B3)
branche4.py        ← prédiction (uq_PCK_eval)
branche5.py        ← fonctions atomiques (polynômes, noyaux, isop_transform)
branche_lars.py    ← algorithme LARS (uq_lar, uq_PCE_lars, uq_PCE_loo_error)
```

Tout passe par **branche1.py** comme point d'entrée public. Les autres ne s'importent pas directement.

---

## 2. Architecture PCK complète (rappel)

```
fit_pck(X, Y, options, marginals, copula)          [branche1.py:99]
└── uq_PCK_initialize(current_model, global_input)  [branche2.py]
└── uq_PCK_calculate_coefficients(X, Y, pck_config, ...)  [branche3.py:654]
    │
    ├── pce_eval_design_matrix(X, ...)              → U_train (espace isoprobabiliste)
    │
    ├── [LARS] uq_PCE_lars + branche_lars           → idxranking (classement polynômes)
    │     └── uq_PCK_eval_unipoly                   [branche5.py:340]
    │     └── uq_PCE_create_Psi                     [branche5.py:388]
    │
    ├── make_trend_handle(sel, Idx, poly_types)     [branche3.py:786]
    │     → F(U) → (N, P_sel)
    │
    └── fit_kriging_pck(U, Y, F_handle, CorrOptions, ...)  [branche3.py:461]
          ├── uq_Kriging_eval_J_of_theta_ML(theta, kp)  [branche3.py:342]
          │     └── uq_eval_Kernel(U,U,θ,opts)       [branche5.py:852]
          │           └── uq_assemble_Kernel(h,family,type)  [branche5.py:782]
          ├── kriging_optimize_theta(kp, θ0, bounds, method)  [branche3.py:417]
          ├── uq_eval_Kernel(U,U,θ_opt,opts)          → R finale (N,N)
          ├── uq_Kriging_calc_auxMatrices(R, F, Y, ...)  [branche3.py:130]
          ├── uq_Kriging_calc_beta(F, ...)            [branche3.py:167]
          ├── uq_Kriging_calc_sigmaSq(...)            [branche3.py:212]
          └── uq_Kriging_calc_KFold(...)              [branche3.py:301]
                → LOO (dénominateur = N)

predict_pck(fitted_model, X_test)                  [branche1.py:227]
└── uq_PCK_eval(fitted_model, X_test, ...)         [branche4.py:157]
    └── uq_Kriging_eval_one_output(kriging_oo, U_test, U_train, ...)  [branche4.py:49]
          ├── uq_GeneralIsopTransform(X_test → U_test)   [branche5.py:616]
          ├── F_handle(U_test)                       → f_test (N_test, P_sel)
          ├── uq_eval_Kernel(U_test, U_train, θ)     → r₀ (N_test, N_train)
          ├── uq_Kriging_calc_auxMatrices            [si pas de cache]
          └── uq_Kriging_calc_DiagOfCongruent(r₀, R)  [branche3.py:48]
```

---

## 3. Fonctions implémentées dans chaque branche (état 28/05/2026)

### branche5.py

| Ligne | Fonction | Rôle |
|---|---|---|
| 31 | `uq_poly_rec_coeffs(n_max, polytype)` | Coefficients récurrence polynômes orthogonaux |
| 148 | `uq_eval_rec_rule(X, AB, nonrecursive)` | Évalue polynômes via récurrence 3 termes |
| 210 | `uq_eval_legendre(ORDER, X)` | Polynômes de Legendre orthonormaux (N, ORDER+1) |
| 240 | `uq_eval_legendre_deriv(ORDER, X)` | **GEK** Dérivées de Legendre (N, ORDER+1) |
| 286 | `uq_eval_hermite(ORDER, X)` | Polynômes de Hermite orthonormaux (N, ORDER+1) |
| 311 | `uq_eval_hermite_deriv(ORDER, X)` | **GEK** Dérivées de Hermite (N, ORDER+1) |
| 340 | `uq_PCK_eval_unipoly(U, polyindices, PolyTypes)` | Tenseur univarié (N, M, P+1) |
| 388 | `uq_PCE_create_Psi(Indices, univ_p_val)` | Produit tensoriel → base Ψ (N, P_sel) |
| 456 | `_uq_find_nonconstant_marginals(marginals)` | Indices non-constants |
| 465 | `_uq_isIndependenceCopula(copula)` | Test copule indépendante |
| 470 | `_uq_IndepCopula(M)` | Crée copule indépendante |
| 479 | `_uq_IsopTransform(X, src_m, tgt_m)` | Transformée marginale source→cible |
| 501 | `_marginal_cdf(x, m)` | CDF d'une marginale |
| 525 | `_marginal_icdf(u, m)` | CDF inverse d'une marginale |
| 549–564 | Stubs Nataf/Rosenblatt | Non implémentés |
| 569 | `_uq_BlockGeneralIsopTransform(...)` | Transformée bloc par bloc |
| 616 | `uq_GeneralIsopTransform(X, Xm, Xc, Ym, Yc)` | Transformée isoprobabiliste générale |
| 782 | `uq_assemble_Kernel(h, K_family, K_type)` | Valeurs noyau depuis distances normalisées |
| 852 | `uq_eval_Kernel(X1, X2, theta, options)` | Matrice corrélation R (N1, N2) |
| 1039 | `_prod_excl(K_uni)` | **GEK** Produits excluant chaque dimension |
| 1054 | `kernel_deriv_factory(family, der, der_prime)` | **GEK** Factory dérivée noyau → f(X1,X2,θ) |
| 1133 | `uq_assemble_global_Kernel(X1, X2, theta, family)` | **GEK** Matrice R̃ augmentée (N1*(M+1), N2*(M+1)) |
| 1180 | `uq_eval_global_Kernel(X1, X2, theta, options)` | **GEK** Clone augmenté de uq_eval_Kernel |

### branche3.py

| Ligne | Fonction | Rôle |
|---|---|---|
| 48 | `uq_Kriging_calc_DiagOfCongruent(A, B)` | diag(A B⁻¹ Aᵀ) |
| 69–128 | `_calc_CholR`, `_calc_FTRinv`, `_calc_AuxMatricesQR`, `_calc_FTRinvF_inv` | Helpers internes auxMatrices |
| 130 | `uq_Kriging_calc_auxMatrices(R, F, Y, runCase)` | Cholesky R, R⁻¹, FᵀR⁻¹F, etc. |
| 167 | `uq_Kriging_calc_beta(F, trendType, Y, method, auxMatrices)` | GLS : β = (FᵀR⁻¹F)⁻¹ FᵀR⁻¹Y |
| 212 | `uq_Kriging_calc_sigmaSq(KrgParameters, estimMethod)` | Estime σ² |
| 260 | `uq_Kriging_helper_create_randIdx(N, K)` | Indices K-fold |
| 277 | `_calc_B1(Y, F, auxMatrices)` | Matrice B1 formule Dubrule |
| 301 | `uq_Kriging_calc_KFold(randIdx, Y, F, auxMatrices)` | LOO (formule Dubrule) |
| 342 | `uq_Kriging_eval_J_of_theta_ML(theta, kp)` | Objectif ML : J = 0.5*(N*log(2π σ²) + log\|R\| + N) |
| 417 | `kriging_optimize_theta(kp, theta0, bounds, method)` | Optimise θ (L-BFGS-B / DE) |
| 461 | `fit_kriging_pck(U, Y, F_handle, CorrOptions, ...)` | Fit Kriging complet pour PCK |
| 565 | `poly_type_from_marginal(marginal)` | Marginal UQLab → famille polynomiale |
| 581 | `aux_marginal_from_poly_type(poly_type)` | Famille poly → marginale canonique |
| 596 | `pce_multi_indices(M, max_degree)` | Multi-indices PCE total order |
| 626 | `pce_eval_design_matrix(X, marginals, copula, ...)` | Matrice Ψ + U_train |
| 654 | `uq_PCK_calculate_coefficients(X, Y, pck_config, ...)` | **Orchestrateur B3 PCK** |
| 786 | `make_trend_handle(sel, Idx, pt)` | Fermeture F(U) → (N, P_sel) |
| 803 | `make_trend_handle_deriv(sel, Idx, pt, der)` | **GEK** Fermeture ∂F/∂U_der(U) → (N, P_sel) |
| 831 | `make_trend_global_handle(sel, Idx, pt)` | **GEK** Fermeture F̃(U) → (N*(M+1), P_sel) |

### branche4.py

| Ligne | Fonction | Rôle |
|---|---|---|
| 32 | `_verify_YSigma2(v)` | Plafonne variance négative à 0 |
| 49 | `uq_Kriging_eval_one_output(kriging_oo, U_test, U_train, ...)` | Prédiction Kriging 1 sortie |
| 157 | `uq_PCK_eval(fitted_model, X_test, ...)` | Prédiction PCK (boucle sur Nout) |

### branche1.py

| Ligne | Fonction | Rôle |
|---|---|---|
| 53 | `generate_doe(N, marginals, method, seed)` | Générateur DOE temporaire (LHS/MC) |
| 99 | `fit_pck(X, Y, options, marginals, copula)` | **Point d'entrée public PCK** |
| 227 | `predict_pck(fitted_model, X_test, ...)` | Prédiction PCK |

---

## 4. Différences mathématiques PCK → GEPCK (Zuhal 2021)

| Élément | PCK | GEPCK |
|---|---|---|
| Vecteur réponse | y, shape (N, 1) | ẏ, shape (N*(M+1), 1) : N valeurs + N*M gradients |
| Ordre de ẏ | — | dimension-major : [y(x^1..n), ∂y/∂x_1(x^1..n), ..., ∂y/∂x_M(x^1..n)] |
| Matrice corrélation entraînement | R, shape (N,N), symétrique | R̃, shape (N*(M+1), N*(M+1)), **symétrique** (Zuhal, pas Bouhlel) |
| Trend matrix entraînement | F = F(U), shape (N, P_sel) | F̃ = F̃(U), shape (N*(M+1), P_sel) |
| Trend vector prédiction | f(x*), shape (P_sel,) | f(x*), shape (P_sel,) **— identique, NON augmenté** |
| Vecteur cross-corr prédiction | r₀(x*), shape (N,) | r̃₀(x*), shape (N*(M+1),) — inclut ∂k/∂x*_l |
| LOOCV dénominateur | N | N*(M+1) (automatique si len(Y_aug) utilisé) |
| Matrice B LOOCV | [σ²R, F ; Fᵀ, 0] | [σ²R̃, F̃ ; F̃ᵀ, 0] |
| Noyau corrélation | `uq_eval_Kernel` | `uq_eval_global_Kernel` |
| Noyaux supportés | tous | **gaussian et matern-5_2 uniquement** (séparables) |

### Propriété mathématique clé — R̃ est symétrique

Contrairement à Bouhlel 2019 (4 blocs non symétrique), la convention dimension-major de Zuhal 2021 donne une matrice R̃ symétrique. Preuve :  
`Block(k,0)[j,i] = ∂k(xj,xi)/∂xi_{k-1} = −∂k(xi,xj)/∂xi_{k-1} = Block(0,k)[i,j]`  
→ `Block(0,k)ᵀ = Block(k,0)` pour tout k.  
Vérifié numériquement : max|R̃ − R̃ᵀ| = 0 à précision machine.

### Formules dérivées de noyau (branche5.py kernel_deriv_factory)

```
Notation : δ_l = x_l − x'_l,  h_l = |δ_l|/θ_l,  a_l = √5·h_l

Noyau Gaussien k = ∏_l exp(-δ_l²/(2θ_l²)) :
  [None, None] → k
  [i, None]    → −δ_i/θ_i² · k
  [None, j]    → +δ_j/θ_j² · k
  [i, i]       → (1 − δ_i²/θ_i²)/θ_i² · k
  [i, j] i≠j   → −δ_i·δ_j/(θ_i²θ_j²) · k

Noyau Matérn 5/2 k = ∏_l k_l,  k_l = (1+a_l+a_l²/3)·exp(−a_l) :
  [None, None] → ∏ k_l
  [i, None]    → −(5/3θ_i²)·δ_i·(1+a_i)·exp(−a_i) · K_excl[:,i]
  [None, j]    → +(5/3θ_j²)·δ_j·(1+a_j)·exp(−a_j) · K_excl[:,j]
  [i, i]       → (5/3θ_i²)·(1+a_i−a_i²)·exp(−a_i) · K_excl[:,i]
  [i, j] i≠j   → −(25/9θ_i²θ_j²)·δ_i·δ_j·(1+a_i)(1+a_j)·exp(−a_i−a_j)·K_excl[:,i]/K_uni[:,j]

Convention der/der_prime :
  der       = composante du PREMIER argument (= x, point test ou ligne)
  der_prime = composante du SECOND argument (= x', point train ou colonne)
```

### Formules dérivées de polynômes (branche5.py)

```
Hermite orthonormal : H_k = polynôme de degré k, H_0=1, H_1=x, H_2=(x²-1)/√2
  H'_0 = 0
  H'_k(x) = √k · H_{k-1}(x)  pour k ≥ 1

Legendre orthonormal : L_n = √(2n+1) · P_n  (NB: normalisation UQLab, PAS √((2n+1)/2))
  L_n = √(2n+1) · P_n
  L'_0 = 0,  L'_1 = √3 (constante),  L'_2 = 3√5·x
  Récurrence : P'_0=0, P'_1=1, P'_n = (2n−1)·P_{n-1} + P'_{n-2}
  Puis L'_n = √(2n+1) · P'_n
```

---

## 5. Inventaire complet — statut de chaque fonction pour GEPCK

### Légende
- **Partagée** : fonction PCK réutilisée telle quelle en GEPCK (même signature, même code)
- **✅ FAIT** : nouvelle fonction GEPCK déjà implémentée et testée
- **❌ MANQUANTE** : à implémenter

### branche5.py

| Fonction | Statut GEPCK |
|---|---|
| `uq_poly_rec_coeffs` | Partagée — aucun clone |
| `uq_eval_rec_rule` | Partagée — aucun clone |
| `uq_eval_legendre` | Partagée |
| `uq_eval_hermite` | Partagée |
| `uq_eval_legendre_deriv` | **✅ FAIT** |
| `uq_eval_hermite_deriv` | **✅ FAIT** |
| `uq_PCK_eval_unipoly` | Partagée |
| `uq_PCE_create_Psi` | Partagée |
| Toutes les fonctions isop_transform | Partagées |
| `uq_assemble_Kernel` | Partagée (PCK seulement, pas appelée en GEPCK) |
| `uq_eval_Kernel` | Partagée (PCK seulement) |
| `_prod_excl` | **✅ FAIT** (helper kernel_deriv_factory) |
| `kernel_deriv_factory(family, der, der_prime)` | **✅ FAIT** |
| `uq_assemble_global_Kernel(X1, X2, theta, family)` | **✅ FAIT** |
| `uq_eval_global_Kernel(X1, X2, theta, options)` | **✅ FAIT** |
| `uq_eval_gepck_cross_corr(U_test, U_train, theta, options)` | **❌ MANQUANTE** — nouvelle (pas de clone PCK) |

### branche3.py

| Fonction | Statut GEPCK |
|---|---|
| `uq_Kriging_calc_DiagOfCongruent` | Partagée |
| `uq_Kriging_calc_auxMatrices` | Partagée — fonctionne sur R̃, F̃, ẏ de toute taille |
| `uq_Kriging_calc_beta` | Partagée |
| `uq_Kriging_calc_sigmaSq` | Partagée |
| `uq_Kriging_calc_KFold` | Partagée — dénominateur = `len(Y_aug)` = N*(M+1) automatique |
| `uq_Kriging_eval_J_of_theta_ML` | Partagée — N dérivé de `R.shape[0]` automatique |
| `kriging_optimize_theta` | Partagée |
| `fit_kriging_pck` | PCK seulement |
| `poly_type_from_marginal` | Partagée |
| `aux_marginal_from_poly_type` | Partagée |
| `pce_multi_indices` | Partagée |
| `pce_eval_design_matrix` | Partagée |
| `uq_PCK_calculate_coefficients` | PCK seulement |
| `make_trend_handle` | Partagée (imbriquée, appelée aussi dans make_trend_global_handle) |
| `make_trend_handle_deriv` | **✅ FAIT** |
| `make_trend_global_handle` | **✅ FAIT** |
| `assemble_Y_aug(Y, dYdU)` | **❌ MANQUANTE** — nouvelle (pas de clone PCK) |
| `fit_kriging_gepck(U, Y_aug, F_global_handle, CorrOptions, ...)` | **❌ MANQUANTE** — clone de `fit_kriging_pck` |
| `uq_GEPCK_calculate_coefficients(X, Y, dYdX, pck_config, ...)` | **❌ MANQUANTE** — clone de `uq_PCK_calculate_coefficients` |

### branche4.py

| Fonction | Statut GEPCK |
|---|---|
| `_verify_YSigma2` | Partagée |
| `uq_Kriging_eval_one_output` | PCK seulement |
| `uq_PCK_eval` | PCK seulement |
| `uq_GEPCK_eval_one_output(gepck_oo, U_test, U_train, Y_aug, F_aug_train, ...)` | **❌ MANQUANTE** — clone de `uq_Kriging_eval_one_output` |
| `uq_GEPCK_eval(fitted_model, X_test, ...)` | **❌ MANQUANTE** — clone de `uq_PCK_eval` |

### branche1.py

| Fonction | Statut GEPCK |
|---|---|
| `generate_doe` | Partagée (temporaire) |
| `fit_pck` | PCK seulement |
| `predict_pck` | PCK seulement |
| `fit_gepck(X, Y, dYdX, options, marginals, copula)` | **❌ MANQUANTE** — clone de `fit_pck` |
| `predict_gepck(fitted_model, X_test, ...)` | **❌ MANQUANTE** — clone de `predict_pck` |

---

## 6. Récapitulatif — 8 fonctions restant à créer

| # | Fonction | Fichier | Analogue PCK | Complexité |
|---|---|---|---|---|
| 1 | `uq_eval_gepck_cross_corr(U_test, U_train, theta, options)` | branche5.py | (nouvelle) | Faible |
| 2 | `assemble_Y_aug(Y, dYdU)` | branche3.py | (nouvelle) | Triviale |
| 3 | `fit_kriging_gepck(U, Y_aug, F_global_handle, CorrOptions, ...)` | branche3.py | `fit_kriging_pck` | Moyenne |
| 4 | `uq_GEPCK_calculate_coefficients(X, Y, dYdX, pck_config, ...)` | branche3.py | `uq_PCK_calculate_coefficients` | Élevée |
| 5 | `uq_GEPCK_eval_one_output(gepck_oo, U_test, U_train, ...)` | branche4.py | `uq_Kriging_eval_one_output` | Moyenne |
| 6 | `uq_GEPCK_eval(fitted_model, X_test, ...)` | branche4.py | `uq_PCK_eval` | Faible |
| 7 | `fit_gepck(X, Y, dYdX, options, marginals, copula)` | branche1.py | `fit_pck` | Faible |
| 8 | `predict_gepck(fitted_model, X_test, ...)` | branche1.py | `predict_pck` | Triviale |

---

## 7. Arborescence GEPCK complète (pipeline cible)

```
fit_gepck(X, Y, dYdX, options, marginals, copula)            [B1 — ❌ clone de fit_pck]
│  X      : (N, M)   — points d'entraînement (espace physique)
│  Y      : (N,)     — valeurs de la fonction
│  dYdX   : (N, M)   — gradients ∂y/∂x (espace physique, transformés en ∂y/∂u)
│
└── uq_PCK_initialize                                          [B2 — partagée]
└── uq_GEPCK_calculate_coefficients(X, Y, dYdX, pck_config, ...)  [B3 — ❌ clone de uq_PCK_calculate_coefficients]
    │
    ├── pce_eval_design_matrix(X, ...)                         [B3 — partagée]
    │     → U_train (N, M), marginals_aux, copula_aux
    │
    ├── assemble_Y_aug(Y, dYdU)                               [B3 — ❌ nouvelle]
    │     dYdU = dYdX transformé via jacobien isop (à faire dans l'orchestrateur)
    │     → ẏ shape (N*(M+1),)
    │     Ordre : [y(x^1..n), ∂y/∂u_1(x^1..n), ..., ∂y/∂u_M(x^1..n)]
    │
    ├── [LARS] uq_PCE_lars + branche_lars                     [partagées]
    │     → idxranking (classement polynômes)
    │
    ├── make_trend_global_handle(sel, Idx, poly_types)         [B3 — ✅]
    │     → F_global(U) → F̃ shape (N*(M+1), P_sel)  (Zuhal Eq. 9)
    │     ├── make_trend_handle(sel, Idx, pt)                  [B3 — partagée]
    │     │     → F(U) → (N, P_sel)
    │     └── make_trend_handle_deriv(sel, Idx, pt, der)       [B3 — ✅]
    │           → ∂F/∂U_der(U) → (N, P_sel)
    │           ├── uq_eval_legendre_deriv(ORDER, X)           [B5 — ✅]
    │           └── uq_eval_hermite_deriv(ORDER, X)            [B5 — ✅]
    │
    └── fit_kriging_gepck(U_train, ẏ, F_global_handle,         [B3 — ❌ clone de fit_kriging_pck]
                          CorrOptions_gepck, theta_bounds, theta0, ...)
          │  Différences vs fit_kriging_pck :
          │    - F_train = F_global_handle(U) → shape (N*(M+1), P_sel)
          │    - CorrOptions_gepck['Handle'] = uq_eval_global_Kernel
          │    - R̃ de taille (N*(M+1), N*(M+1))
          │    - ẏ de taille (N*(M+1),)
          │    - tout le reste (Cholesky, GLS, σ², LOO) est partagé
          │
          ├── uq_Kriging_eval_J_of_theta_ML(theta, kp)         [B3 — partagée]
          │     └── uq_eval_global_Kernel(U,U,θ,opts)          [B5 — ✅]
          │           └── uq_assemble_global_Kernel(X1,X2,θ,family)  [B5 — ✅]
          │                 └── kernel_deriv_factory(family,der,dp)   [B5 — ✅]
          │                       └── _prod_excl(K_uni)               [B5 — ✅]
          ├── kriging_optimize_theta(kp, θ0, bounds, method)   [B3 — partagée]
          ├── uq_eval_global_Kernel(U,U,θ_opt,opts)            → R̃ finale
          ├── uq_Kriging_calc_auxMatrices(R̃, F̃, ẏ, ...)       [B3 — partagée]
          ├── uq_Kriging_calc_beta(F̃, ...)                     [B3 — partagée]
          ├── uq_Kriging_calc_sigmaSq(...
          )                      [B3 — partagée]
          └── uq_Kriging_calc_KFold(...)                        [B3 — partagée]
                → LOO sur N*(M+1) points (dénominateur automatique)

predict_gepck(fitted_model, X_test)                           [B1 — ❌ clone de predict_pck]
└── uq_GEPCK_eval(fitted_model, X_test, ...)                  [B4 — ❌ clone de uq_PCK_eval]
    └── uq_GEPCK_eval_one_output(gepck_oo, U_test, U_train, Y_aug,
                                  F_aug_train, CorrOptions, ...)   [B4 — ❌ clone de uq_Kriging_eval_one_output]
          │
          │  Différences clés vs uq_Kriging_eval_one_output :
          │    1. r̃₀ est AUGMENTÉ (inclut ∂k/∂x*_l) ← principal changement
          │    2. f_test = standard F_handle(U_test) NON augmenté ← IMPORTANT
          │    3. R̃, ẏ, F̃_train sont augmentés
          │
          ├── uq_GeneralIsopTransform(X_test → U_test)         [B5 — partagée]
          │
          ├── gepck_oo['F_handle_standard'](U_test)            [B3 — partagée]
          │     → f_test shape (N_test, P_sel)   ← trend STANDARD, pas augmenté
          │
          ├── uq_eval_gepck_cross_corr(U_test, U_train, θ, opts)  [B5 — ❌ nouvelle]
          │     → R̃₀ shape (N_train*(M+1), N_test)
          │     Définition : R̃₀[:,j][k*N_train:(k+1)*N_train]
          │       = kernel_deriv_factory(family, k-1 if k>0 else None, None)
          │             (U_test[[j]], U_train, θ).ravel()
          │     └── kernel_deriv_factory(family, der, None)     [B5 — ✅]
          │           der = None (k(x*,x)), 0 (∂k/∂x*_1), ..., M-1 (∂k/∂x*_M)
          │
          ├── uq_Kriging_calc_auxMatrices(R̃, F̃, ẏ, ...)       [B3 — partagée, si pas de cache]
          └── uq_Kriging_calc_DiagOfCongruent(R̃₀, R̃)           [B3 — partagée]
```

---

## 8. Notes d'implémentation critiques

### 8.1 assemble_Y_aug — transformation des gradients
Les gradients `dYdX` sont fournis dans l'espace physique X. Il faut les transformer en ∂y/∂u (espace isoprobabiliste). Pour des marginales indépendantes, le jacobien est diagonal :  
`∂y/∂u_l = ∂y/∂x_l · (∂x_l/∂u_l)` où `∂x_l/∂u_l` est l'inverse de la densité transformée.  
**Simplification** : si l'espace U est canonique (Legendre → Uniform[-1,1], Hermite → Normal(0,1)), le jacobien est la dérivée du quantile : `∂x/∂u = σ` pour Gaussien, `(b-a)/2` pour Uniform[a,b].

### 8.2 fit_kriging_gepck — structure quasi-identique à fit_kriging_pck
Les seuls changements par rapport à `fit_kriging_pck` :
1. `CorrOptions['Handle'] = uq_eval_global_Kernel` (au lieu de `uq_eval_Kernel`)
2. `F_train = F_global_handle(U_train)` → shape (N*(M+1), P_sel)
3. `Y` est `Y_aug` shape (N*(M+1),)
4. Tout le reste (optimisation θ, GLS, σ², LOO) est strictement partagé

### 8.3 uq_eval_gepck_cross_corr — signature et logique
```python
def uq_eval_gepck_cross_corr(U_test, U_train, theta, options):
    """
    Retourne R̃₀ de shape (N_train*(M+1), N_test).
    Colonne j = r̃₀(x*_j) = vecteur corrélation augmenté test point j vs tous train points.
    Blocs : k=0 → k(x*_j, x^(i)), k=1..M → ∂k(x*_j, x^(i))/∂x*_{k-1}
    """
    family = options['Family']
    M = U_train.shape[1]
    N_train = U_train.shape[0]
    N_test  = U_test.shape[0]
    R0 = np.empty((N_train * (M+1), N_test))
    for k in range(M + 1):
        der = k - 1 if k > 0 else None
        f   = kernel_deriv_factory(family, der, None)
        # f(U_test, U_train, theta) → (N_test, N_train), .T → (N_train, N_test)
        R0[k*N_train:(k+1)*N_train, :] = f(U_test, U_train, theta).T
    return R0
```

### 8.4 Prédiction — f(x*) est NON augmenté
Point critique : pendant l'entraînement, on utilise F̃(U) (augmentée, N*(M+1) × P_sel) pour estimer β. Mais en prédiction, le trend au point test est f(x*) = Ψ(x*) standard (P_sel,). La formule est :
```
ŷ(x*) = f(x*)ᵀ β + r̃₀(x*)ᵀ R̃⁻¹ (ẏ − F̃ β)
```
L'information de gradient entre dans la prédiction **uniquement** via r̃₀ (cross-corrélation augmentée), pas via le trend f.

### 8.5 Tests à écrire
- `test_assemble_Y_aug.py` : vérifier shapes et ordre dimension-major
- `test_gepck_cross_corr.py` : vérifier FD vs analytique pour r̃₀
- `test_fit_gepck.py` : LOO < 0.5 sur fonction test 2D avec gradient exact
- Vérifier la propriété d'interpolation de GEPCK (ŷ(x^i) = y(x^i))

---

## 9. Ordre de codage (bottom-up, feuilles → tronc)

Même logique que le PCK : B5 (feuilles atomiques) → B3 (fit) → B4 (prédiction) → B1 (point d'entrée). Chaque étape est testable indépendamment avant de passer à la suivante.

**Étape 1 — B5 : `uq_eval_gepck_cross_corr`**
Feuille atomique. Dépend uniquement de `kernel_deriv_factory` (déjà fait). Testable seule par FD :
`r̃₀(x* + ε·eₖ) − r̃₀(x* − ε·eₖ) / 2ε` doit correspondre aux blocs dérivés analytiques.

**Étape 2 — B3 utilitaire : `assemble_Y_aug`**
Triviale, aucune dépendance GEPCK. Assemble ẏ = [Y ; dY/dU₀ ; ... ; dY/dU_{M-1}] depuis Y (N,) et
dYdU (N, M). Testable sur données synthétiques en vérifiant shape et ordre des blocs.

**Étape 3 — B3 cœur : `fit_kriging_gepck`**
Dépend des étapes 1 et 2 + toutes les fonctions `uq_Kriging_calc_*` partagées + `uq_eval_global_Kernel`
(déjà fait). Clone direct de `fit_kriging_pck` avec F̃, R̃, ẏ augmentés.
Testable en standalone sur une fonction 2D avec gradient exact (LOO < 0.5, ŷ(x^i) = y(x^i)).

**Étape 4 — B3 orchestrateur : `uq_GEPCK_calculate_coefficients`**
Dépend des étapes 2 et 3 + `make_trend_global_handle` (déjà fait) + LARS partagé.
Clone de `uq_PCK_calculate_coefficients` : mêmes modes sequential/optimal, remplace `fit_kriging_pck`
par `fit_kriging_gepck` et `make_trend_handle` par `make_trend_global_handle`.

**Étape 5 — B4 prédiction : `uq_GEPCK_eval_one_output`**
Dépend de l'étape 1 pour r̃₀, et du modèle produit en étape 4.
Clone de `uq_Kriging_eval_one_output` : seuls changements = r₀ → r̃₀ via `uq_eval_gepck_cross_corr`,
et f_test = F_handle standard (non augmenté).

**Étape 6 — B4 boucle : `uq_GEPCK_eval`**
Dépend de l'étape 5. Clone trivial de `uq_PCK_eval` : boucle sur Nout sorties,
appelle `uq_GEPCK_eval_one_output`.

**Étape 7 — B1 points d'entrée : `fit_gepck` + `predict_gepck`**
Dépendent de tout. Colles minces : `fit_gepck` appelle B2 (partagé) puis étape 4 ;
`predict_gepck` appelle étape 6. Testable avec un test end-to-end sur fonction analytique.
