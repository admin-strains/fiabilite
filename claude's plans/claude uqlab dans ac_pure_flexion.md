# Plan : Intégration branche1 dans AC_pure_flexion.py — affichage g analytique

## Nouvelle règle de travail (demandée lors de cette session)

Créer et maintenir `C:\_workingDir\_SF\test flexion\mix_fonctionnement_global.md` :
- Fichier synthétique rempli **progressivement** au fil du travail
- But : éviter de tout relire après un autocompactage
- Contient : état des fichiers clés, ce qui est fait, ce qui reste à faire, décisions architecturales
- À mettre à jour après chaque modification significative

---

## Contexte

`AC_pure_flexion.py` est un code de fiabilité (FORM) pour flexion pure béton armé.
Il contient une classe `flexion_claude` qui définit la fonction de performance analytique
`g(u1, u2)` en espace standard (U ~ N(0,1)).

Objectif de cette tâche : **appeler `fit_pck` / `predict_pck` (branche1) pour construire un
métamodèle PCK de `g`, et afficher la courbe g=0 du PCK superposée à la courbe analytique.**
C'est uniquement la première étape d'affichage — pas encore de FORM.

---

## Ce que j'ai lu et compris

**Fichier principal** : `C:\_workingDir\_SF\test flexion\AC_pure_flexion.py`

### La fonction g

`flexion_claude.g(u1, u2)` — lignes 673-682 :
- Prend `(u1, u2)` en **espace standard** (coordonnées normales standard)
- Appelle en interne `self.T_inv(ot.Point([u1, u2]))` pour revenir à l'espace physique (fc, fy)
- Deux branches : aciers plastifiés (`x1 > x1_lim_plast_x2`) et non plastifiés
- Retourne un scalaire (positif = sûr, négatif = défaillance)
- Nécessite les fichiers `dsCad.txt` et `dsLoad.txt` dans `C:\workspace\storage\admin\SF\test_pure_flexion.ds\`

### Variables aléatoires

| Variable | Loi | Paramètres dans le code |
|---|---|---|
| fc (résistance béton) | LogNormal | `fcm=48`, `cov_fc=0.12` → `sigma_ln=sqrt(log(1.0144))`, `mu_ln=log(48)-0.5*sigma_ln^2` |
| fy (limite élastique acier) | Normale | `fym=550`, `SIGMA=sqrt(19^2+22^2+8^2)` ≈ 29.34 MPa |

### Espace de travail

Tout le code existant travaille en **espace standard U** (Normal(0,1) × Normal(0,1)) :
- `init_FORM` crée `ot.JointDistribution([ot.Normal(0,1)] * n_var)` (ligne 1037)
- Les métamodèles reçoivent (u1, u2) en entrée
- La fonction `g(u1, u2)` prend directement du standard

→ **Le PCK sera entraîné en espace U avec des marginales Gaussian(0,1).**
  `fit_pck` avec Gaussian(0,1) utilise les polynômes de Hermite, cohérent avec les variables normales standard.

### Paramètres de visualisation existants (lignes 133-137)
```
u1_min=-10, u1_max=10, u2_min=-10, u2_max=10, n_grid=300
```

### Fonction de visualisation existante
`print_visu_ana()` (lignes 684-714) : trace la courbe g=0 parametriquement via `u2p_LS(u1)`.
Pas de contour 2D — on ajoutera un contour PCK sur le même type de figure.

---

## Plan d'implémentation

### Fichier à modifier
`C:\_workingDir\_SF\test flexion\AC_pure_flexion.py`

### 4 modifications, dans l'ordre

---

#### Modification 1 — Import branche1 (dans le bloc imports, vers ligne 30)

```python
import sys as _sys
_sys.path.insert(0, r'C:\_workingDir\_SF\test flexion')
from branche1 import fit_pck, predict_pck, generate_doe
```

Juste après les imports existants (après `from math import comb`).

---

#### Modification 2 — Nouvelle option dans le bloc OPTIONS (vers ligne 79)

Ajouter après la ligne `do_GEPCK = ...` :

```python
do_PCK_B1 = True if modele == 'PCK_B1' else False
```

Et ajouter `'PCK_B1'` en commentaire dans la liste des valeurs possibles de `modele`.

---

#### Modification 3 — Nouvelle fonction `print_visu_pck_b1` (après `print_visu_ana`, vers ligne 715)

```python
def print_visu_pck_b1(N_doe=40, pce_degree=None, seed=42):
    """
    Construit un metamodele PCK (branche1) sur g_analytique et affiche
    la courbe g=0 PCK superposee a la courbe analytique.
    
    Travaille en espace standard U ~ N(0,1) x N(0,1), coherent avec init_FORM.
    """
    import warnings

    # --- Marginales en espace U : Normal(0,1) x Normal(0,1) ---
    marginals_u = [{'Type': 'Gaussian', 'Parameters': [0.0, 1.0]}] * n_var
    copula_u    = {'Type': 'Independent', 'Parameters': np.eye(n_var)}
    opts_pck    = {
        'Mode': 'sequential',
        'PCE':  {'Degree': pce_degree or [1, 2, 3, 4], 'Method': 'LARS'},
    }

    # --- DOE en espace U ---
    X_doe = generate_doe(N_doe, marginals_u, method='lhs', seed=seed)  # (N_doe, 2)

    # --- Evaluer g analytique au DOE ---
    calc  = flexion_claude()
    Y_doe = np.array([calc.g(float(u[0]), float(u[1])) for u in X_doe])

    # --- Fit PCK ---
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fm = fit_pck(X_doe, Y_doe, opts_pck, marginals_u, copula_u)

    loo = fm['Error'][0]['LOO']
    print(f'[PCK_B1] N_doe={N_doe}  LOO={loo:.4e}  '
          f'n_poly={fm["NumberOfPoly"][0]}', flush=True)

    # --- Grille fine en espace U ---
    # On utilise une grille reduite pour la visu (100x100 suffisant)
    ng = min(n_grid, 150)
    u1g = np.linspace(u1_min, u1_max, ng)
    u2g = np.linspace(u2_min, u2_max, ng)
    U1, U2 = np.meshgrid(u1g, u2g)
    X_grid = np.column_stack([U1.ravel(), U2.ravel()])  # (ng^2, 2)

    # Analytique sur grille
    Z_ana = np.array([calc.g(float(u1), float(u2))
                      for u1, u2 in X_grid]).reshape(ng, ng)

    # PCK sur grille
    YMu_grid = predict_pck(fm, X_grid)[:, 0].reshape(ng, ng)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 7))

    # Fond : signe de g analytique (domaine de defaillance)
    ax.contourf(U1, U2, Z_ana, levels=[-1e10, 0], colors=['#ffcccc'], alpha=0.4)
    ax.contourf(U1, U2, Z_ana, levels=[0, 1e10],  colors=['#cce5ff'], alpha=0.4)

    # Courbe g=0 analytique
    cs_ana = ax.contour(U1, U2, Z_ana,    levels=[0], colors='k',      linewidths=2.0)
    # Courbe g=0 PCK
    cs_pck = ax.contour(U1, U2, YMu_grid, levels=[0], colors='royalblue',
                        linewidths=2.0, linestyles='--')

    # Points du DOE
    ax.plot(X_doe[:, 0], X_doe[:, 1], 'r+', ms=8, mew=1.5,
            label=f'DOE ($N={N_doe}$)')
    ax.plot(0, 0, 'g+', ms=12, mew=2, label='Origine')

    # Légende manuelle pour les contours
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='k',          lw=2,   label='$g=0$ analytique'),
        Line2D([0], [0], color='royalblue',   lw=2,   linestyle='--',
               label=f'$g=0$ PCK ($N={N_doe}$, LOO={loo:.1e})'),
        Line2D([0], [0], color='r',           lw=0, marker='+', ms=8,
               markeredgewidth=1.5, label=f'DOE ($N={N_doe}$)'),
    ]
    ax.legend(handles=handles, loc='best', fontsize=9)

    ax.set_xlabel(r'$u_1$  (espace standard, $f_c$)')
    ax.set_ylabel(r'$u_2$  (espace standard, $f_y$)')
    ax.set_title("Surface d'état-limite — flexion pivot B\n"
                 r"$g=0$ analytique (noir) vs PCK branche1 (bleu)")
    ax.axhline(0, color='gray', lw=0.4)
    ax.axvline(0, color='gray', lw=0.4)
    ax.set_xlim(u1_min, u1_max)
    ax.set_ylim(u2_min, u2_max)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig, ax, fm
```

---

#### Modification 4 — Appel dans le bloc d'exécution (dans le `if do_PCK_B1:` block)

À ajouter dans la section d'exécution du code, avant l'appel `init_FORM`.
Chercher l'endroit où les autres `do_*` sont actionnés, ajouter :

```python
    if do_PCK_B1:
        fig_pck, ax_pck, fm_pck_b1 = print_visu_pck_b1(
            N_doe=40,
            pce_degree=[1, 2, 3, 4],
            seed=42,
        )
```

---

## Points de vigilance

1. **`flexion_claude()` lit des fichiers disque** — doit tourner dans l'environnement STRAINS avec `C:\workspace\storage\admin\SF\test_pure_flexion.ds\` accessible.

2. **`generate_doe` est marquée TEMPORAIRE** dans branche1.py — c'est intentionnel pour l'instant.

3. **Grille 150×150** au lieu de 300×300 pour la visu PCK — 22,500 points × 2 appels (analytique + PCK) reste léger. Si la figure est trop grossière, monter à 200.

4. **`flexion_claude.g` est scalaire** — la list comprehension sur la grille est lente (22,500 appels OpenTURNS). Si trop lent, réduire la grille ou vectoriser via un batch OT.

5. **LOO attendu** : pour N=40 et une fonction non triviale (deux branches), LOO ~0.01-0.10 est un bon signe. Si LOO >> 0.1, augmenter N_doe ou le degré PCE.

---

## Vérification

Après exécution avec `modele = 'PCK_B1'` :
- La figure apparaît avec 2 courbes g=0 (noire = analytique, bleue = PCK)
- Le print terminal affiche `[PCK_B1] N_doe=40  LOO=...  n_poly=...`
- Les deux courbes g=0 doivent coïncider visuellement (écart visible = LOO trop élevé)
- Le `do_PCK_B1 = True` ne touche rien à l'existant : aucun autre bloc `do_*` n'est modifié

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
