# Plan — Maîtrise des options KrigingAlgorithm OpenTURNS

## Contexte

Le krigeage pur (KRG, `build_metamodel_KRG`) donne des résultats dégradés sur la géométrie actuelle
(b=h=0.8m, F=0.74 MN, u*≈[-4.8,-6.6], beta_HF≈8.12) sans avoir besoin de la composante PCE.
La cause principale est documentée en session 15/05 : **artefact de rebound** — le surrogate forme
une "île" g<0 fermée au lieu d'une région semi-infinie comme la vraie fonction.

---

## 1. Pourquoi le KRG seul échoue — la formule

```
g_hat(x) = beta0 + r(x)^T R^-1 (Y - beta0)
```

- `r(x)` = vecteur de corrélations entre `x` et les points DOE.
- SquaredExponential : `r(x) -> 0` quand `x` s'éloigne du DOE (corrélation exponentielle).
- Résultat : la prédiction revient vers `beta0` partout hors du DOE.
- `beta0` = estimateur GLS = moyenne(g_DOE) **> 0** car la majorité des points LHS tombent
  en zone sûre (failure region petite par rapport à [-10,10]²).
- Le surrogate "invente" une zone g<0 locale proche du DOE, mais revient vers `beta0 > 0`
  au-delà → la vraie frontière semi-infinie n'est pas capturée.

---

## 2. Catalogue complet des options KrigingAlgorithm

### 2.1 Trend basis — **Levier principal contre le rebound**

| Option | Code | Comportement hors DOE |
|--------|------|-----------------------|
| `ConstantBasisFactory` (actuel) | `ot.ConstantBasisFactory(n_var).build()` | Revient vers `beta0 = mean(g_DOE) > 0` |
| `LinearBasisFactory` | `ot.LinearBasisFactory(n_var).build()` | Extrapole avec `beta0 + beta1*u1 + beta2*u2` — capture la pente descendante vers la failure region |
| `QuadraticBasisFactory` | `ot.QuadraticBasisFactory(n_var).build()` | Extrapole avec courbure — risque d'overfitting sur petit DOE |

**Mécanique** : Avec `LinearBasisFactory`, la régression GLS sur le DOE estime `beta1, beta2`.
Si la fonction g décroît linéairement vers u* (pente négative vers (-5,-7)), la régression le
capture et l'extrapolation pointe vers g<0 correctement. Le rebound disparaît ou s'atténue
fortement.

Déjà identifié dans le global (session 15/05, l.701 du script) comme alternative non testée.

---

### 2.2 Modèle de covariance — noyau

| Option | Code | Effet |
|--------|------|-------|
| `SquaredExponential` (actuel) | `ot.SquaredExponential([1.0]*n_var)` | Infiniment lisse, corrélation décroît très vite → rebound fort |
| `MaternModel ν=2.5` (commenté) | `ot.MaternModel([1.0]*n_var, 2.5)` | Processus 2× dérivable, corrélation décroît moins vite → rebound moins marqué |
| `MaternModel ν=1.5` | `ot.MaternModel([1.0]*n_var, 1.5)` | Processus 1× dérivable, encore moins de rebound |
| `IsotropicCovarianceModel` | `ot.IsotropicCovarianceModel(ot.SquaredExponential(), n_var)` | 1 seul paramètre theta (au lieu de 2) → optimisation plus stable, moins d'artefacts anisotropes |

**Recommandation** : `MaternModel ν=2.5` ou `ν=1.5` est adapté si la fonction STRAINS est seulement
2× dérivable (ce qui est plausible pour une limite d'analyse plastique). SquaredExponential impose
une régularité infinie qui est probablement excessive.

**Isotropie** : le global (session 15/05) note "isotrope souvent meilleur" car le modèle anisotrope
peut trouver theta₁=50, theta₂=350 → artefacts. `IsotropicCovarianceModel` force theta₁=theta₂ → 
1 param au lieu de 2 → optimisation plus robuste.

---

### 2.3 Paramètre scale θ — bornes d'optimisation

**Diagnostic essentiel** : afficher le theta optimisé après `run()` :
```python
result.getCovarianceModel().getScale()    # theta [u_fc, u_fy]
result.getCovarianceModel().getAmplitude() # sigma
```

| Valeur de theta | Signification | Conséquence |
|-----------------|---------------|-------------|
| theta << 1 | Corrélation très locale | Rebound très proche du DOE → failure region non capturée |
| theta ~ 1-5 | Corrélation moyenne | Compromise |
| theta >> 5 | Corrélation longue portée | Modèle très lisse → extrapolation vers failure region possible |

**Bornes** : OT utilise des bornes par défaut très larges mais l'optimiseur TNC peut rester bloqué
sur un minimum local avec un theta petit. Fixer des bornes explicites :
```python
lower = ot.Point([0.01] * n_var)
upper = ot.Point([100.0] * n_var)
algo_KRG.setOptimizationBounds(ot.Interval(lower, upper))
```

---

### 2.4 Algorithme d'optimisation des hyperparamètres

| Option | Code | Caractéristiques |
|--------|------|-----------------|
| TNC (défaut) | `ot.TNC()` | Local, sans dérivées, bornes. Risque minima locaux |
| Cobyla | `ot.Cobyla()` | Local, sans dérivées, contraintes. Alternative à TNC |
| NLopt DIRECT (global) | `ot.NLopt('GN_DIRECT')` | Exploration globale des hyperparamètres. Coûteux |
| **MultiStart** | `ot.MultiStart(ot.TNC(), starting_sample)` | Lance TNC depuis N points de départ, retourne le meilleur |

**MultiStart** est le levier le plus efficace pour éviter les minima locaux sur l'optimisation de theta.
Exemple :
```python
starting_sample = ot.LHSExperiment(
    ot.JointDistribution([ot.Uniform(0.01, 100.0)] * n_var),
    16
).generate()
algo_KRG.setOptimizationAlgorithm(ot.MultiStart(ot.TNC(), starting_sample))
```

---

### 2.5 Nugget et bruit

| Option | Code | Effet |
|--------|------|-------|
| Nugget = 0 (défaut) | — | Krigeage **interpolant** — exact aux points DOE |
| Nugget activé | `covariance_model.activateNuggetFactor(True)` puis passer à KrigingAlgorithm | Krigeage **régression** — lisse les données, meilleure généralisation |
| Bruit hétérogène | `algo_KRG.setNoise([sigma2]*len(yt))` | Idem mais niveau de bruit par point |

**Ici** : STRAINS est déterministe → pas de bruit de mesure → **nugget = 0 est correct**.
Ne pas activer le nugget, cela ne ferait que dégrader la précision aux points DOE.

Exception : si la matrice R est mal conditionnée (theta très petit, points DOE très proches),
un nugget numérique minimal (1e-6) améliore la stabilité.

---

### 2.6 setOptimizeParameters

```python
algo_KRG.setOptimizeParameters(True)   # défaut — theta est optimisé
algo_KRG.setOptimizeParameters(False)  # theta reste à [1.0]*n_var
```

Toujours `True` — ne jamais fixer theta à sa valeur initiale sauf pour diagnostic.

---

## 3. Diagnostic recommandé avant modification

**Étape 0 — lire le theta actuel** (ajouter dans `build_metamodel_KRG`) :
```python
cov_opt = result.getCovarianceModel()
print(f"  theta_opt = {list(cov_opt.getScale())}", flush=True)
print(f"  sigma_opt = {list(cov_opt.getAmplitude())}", flush=True)
```

Interpréter :
- theta petit (< 1) → rebound très serré → LinearBasis ou bornes larges nécessaires
- theta grand (> 10) → modèle lisse → rebound tardif, mais LinearBasis reste conseillée
- theta très dissemblables (ex: 0.1 vs 50) → anisotropie artefactuelle → passer à IsotropicCovarianceModel

---

## 4. Ordre d'essai recommandé

| Priorité | Modification | Code | Diagnostic attendu |
|----------|-------------|------|--------------------|
| **1** | Afficher theta optimisé | `cov_opt.getScale()` dans `build_metamodel_KRG` | Comprendre la cause |
| **2** | `LinearBasisFactory` | Remplacer `ConstantBasisFactory` | Élimine le rebound si pente est capturée |
| **3** | `MaternModel ν=2.5` | Remplacer `SquaredExponential` | Corrélation plus longue portée |
| **4** | Bornes theta `[0.01, 100]` | `setOptimizationBounds(Interval([0.01]*n_var, [100.0]*n_var))` | Assure que theta peut être grand |
| **5** | `MultiStart` sur l'optimisation theta | `setOptimizationAlgorithm(MultiStart(TNC(), 16pts LHS))` | Évite minima locaux dans l'optim theta |
| **6** | `IsotropicCovarianceModel` | `ot.IsotropicCovarianceModel(ot.MaternModel(1, 1, 2.5), n_var)` | Réduction nb params si theta dissemblables |

Priorité 2 (LinearBasis) est le candidat le plus fort : **c'est la seule option qui change
le comportement d'extrapolation vers infinity**, indépendamment de theta.

---

## 5. Fichier à modifier

**`C:\_workingDir\_SF\test flexion\AC_pure_flexion.py`**
Fonction `build_metamodel_KRG` (lignes 849–858) — seul point à modifier.

Code actuel :
```python
def build_metamodel_KRG(xt, yt):
    n_var = xt.shape[1]
    basis = ot.ConstantBasisFactory(n_var).build()
    # covarianceModel = ot.MaternModel([1.0] * n_var, 2.5)
    covarianceModel = ot.SquaredExponential([1.0] * n_var)
    algo_KRG = ot.KrigingAlgorithm(xt, yt, covarianceModel, basis)
    algo_KRG.run()
    result = algo_KRG.getResult()
    metamodel = result.getMetaModel()
    return metamodel, result
```

Code cible (à décliner selon les tests) :
```python
def build_metamodel_KRG(xt, yt):
    n_var = xt.shape[1]

    # Levier 2 : Linear basis (extrapole correctement hors DOE)
    basis = ot.LinearBasisFactory(n_var).build()

    # Levier 3 : Matern nu=2.5 (corrélation moins agressive que SquaredExp)
    covarianceModel = ot.MaternModel([1.0] * n_var, 2.5)
    # ou : ot.IsotropicCovarianceModel(ot.MaternModel(1, 1, 2.5), n_var) pour isotropie

    algo_KRG = ot.KrigingAlgorithm(xt, yt, covarianceModel, basis)
    algo_KRG.setOptimizeParameters(True)

    # Levier 4 : Bornes theta
    algo_KRG.setOptimizationBounds(
        ot.Interval([0.01] * n_var, [100.0] * n_var)
    )

    # Levier 5 : MultiStart (16 pts LHS sur [0.01, 100]^n_var)
    starting_sample = ot.LHSExperiment(
        ot.JointDistribution([ot.Uniform(0.01, 100.0)] * n_var), 16
    ).generate()
    algo_KRG.setOptimizationAlgorithm(ot.MultiStart(ot.TNC(), starting_sample))

    algo_KRG.run()
    result = algo_KRG.getResult()

    # Levier 1 (diagnostic) : afficher theta optimisé
    cov_opt = result.getCovarianceModel()
    print(f"  KRG theta_opt = {list(cov_opt.getScale())}", flush=True)
    print(f"  KRG sigma_opt = {list(cov_opt.getAmplitude())}", flush=True)

    return result.getMetaModel(), result
```

---

## 6. Vérification

1. Lancer avec `do_KRG=True`, `n0=5`, `do_EFF=False` — noter beta et theta_opt.
2. Comparer beta vs beta_HF (référence `sp=[0,0]` → 8.1235).
3. Si theta_opt est petit (< 1) : les bornes et MultiStart n'ont pas suffi → activer LinearBasis.
4. Utiliser `print_HF=True` (visu) pour voir si la frontière du surrogate KRG rejoint bien la frontière HF.
5. Comparer `output_1905_0856.txt` (KRG n0=5, no EFF, beta=7.2981) comme baseline.
