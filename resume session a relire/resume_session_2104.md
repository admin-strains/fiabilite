# Résumé de session — PCE + GEK/KRG Flexion Pure BA
**Date :** 21 avril 2026
**Objectif :** Réécrire le bloc PCE from scratch (FunctionalChaosAlgorithm OpenTURNS), l'intégrer proprement dans le workflow PCE+KRG/GEK, corriger les incohérences du code, et lancer un premier run PCE-KRG.

---

## 1. Sujets abordés et explications détaillées

### 1a. FunctionalChaosAlgorithm (OpenTURNS) — API complète

**Constructeur :**
```python
FunctionalChaosAlgorithm(inputSample, outputSample, distribution, adaptiveStrategy, projectionStrategy)
```

**Méthodes clés :**
- `algo.run()` → lance le calcul
- `algo.getResult()` → retourne `FunctionalChaosResult`
- `result.getMetaModel()` → métamodèle callable
- `result.getCoefficients()` → coefficients α_k

**FunctionalChaosResult :**
- `getMetaModel()` → callable, entrée `ot.Sample` ou `ot.Point`
- `getMetaModel().gradient(ot.Point)` → `Matrix` (n_var, 1)
- `getMetaModel()(ot.Sample)` → `ot.Sample` (n_points, 1)

**FunctionalChaosSobolIndices :**
```python
chaosSI = ot.FunctionalChaosSobolIndices(result)
s1 = chaosSI.getSobolIndex(j)       # effet direct variable j
st = chaosSI.getSobolTotalIndex(j)  # effet total variable j
```
Indices calculés analytiquement depuis les coefficients — sans simulation supplémentaire.

**FunctionalChaosValidation :**
```python
# Version rapide (analytique, hat matrix) — NE FONCTIONNE PAS avec LARS
validation = ot.FunctionalChaosValidation(result)

# Version force brute LOO — FONCTIONNE toujours, y compris avec LARS
validation = ot.FunctionalChaosValidation(result, ot.LeaveOneOutSplitter(n0))

q2_loo = validation.computeR2Score()[0]
```

---

### 1b. Espace U vs espace X

- **Espace X :** variables physiques (fc en MPa, fy en MPa) — distributions non-standard (lognormale, normale)
- **Espace U :** transformation isoprobabiliste — chaque variable devient N(0,1)
- Le PCE est construit en **espace U** : les polynômes d'Hermite sont orthogonaux par rapport à N(0,1)
- `T_inv = dist_X.getInverseIsoProbabilisticTransformation()` : U → X
- `T_inv.gradient(u_point)` : jacobien J de T_inv au point u

---

### 1c. Base de polynômes d'Hermite

Les polynômes d'Hermite (probabilistes) :
```
H₀(u) = 1
H₁(u) = u
H₂(u) = u² - 1
```
Orthogonaux par rapport à la mesure N(0,1).

Pour n_var=2, chaque terme candidat est un **produit tensoriel** :
```
ψ₀ = H₀(u₁)·H₀(u₂) = 1
ψ₁ = H₁(u₁)·H₀(u₂) = u₁
ψ₂ = H₀(u₁)·H₁(u₂) = u₂
ψ₃ = H₂(u₁)·H₀(u₂) = u₁²-1
ψ₄ = H₀(u₁)·H₂(u₂) = u₂²-1
```

---

### 1d. Norme hyperbolique et HyperbolicAnisotropicEnumerateFunction

`enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(n_var, q)` avec q=0.75.

Un terme de degrés (p₁, p₂) est inclus si :
```
||(p₁, p₂)||_q = (p₁^q + p₂^q)^(1/q) ≤ max_degree
```
Pour q=0.75, max_degree=2 : le terme croisé (1,1) donne `(1+1)^(1/0.75) ≈ 2.52 > 2` → **exclu**.
Résultat : 5 termes candidats {(0,0),(1,0),(0,1),(2,0),(0,2)}, pas de terme croisé.

---

### 1e. LARS (Least Angle Regression) — fonctionnement détaillé

LARS construit une **séquence de modèles emboîtés** (chaque modèle contient le précédent) en ajoutant un polynôme à chaque étape.

**Algorithme :**
1. Modèle 0 : g ≈ α₀ = ḡ (constante). Résidu initial = g(uᵢ) - ḡ.
2. À chaque étape, cherche le polynôme ψₖ du catalogue qui maximise |corr(ψₖ, résidu courant)|.
3. Ajoute ce terme, recalcule α par moindres carrés sur les n0 points, calcule nouveau résidu.
4. Répète jusqu'à la limite algébrique : nombre de termes < nombre de points (sinon système sous-déterminé).

**Corrélation :**
```
corr(ψₖ, r) = Σᵢ ψₖ(uᵢ)·rᵢ / (||ψₖ|| · ||r||)
```
`ψₖ` évalué sur le DOE = vecteur de taille n0. `r` = vecteur résidu de taille n0.

**Limite algébrique :** avec n0=5 points, LOO ne peut utiliser que 4 points → max 4 termes possibles. LARS ne propose jamais autant de termes que de points.

**L'ordre d'emboîtage n'est pas fixé à l'avance** — il dépend des données. Un terme de degré 2 peut être ajouté avant un terme de degré 1 si plus corrélé au résidu.

---

### 1f. CorrectedLeaveOneOut — fonctionnement détaillé

LOO calcule pour chaque modèle de la séquence LARS :
```
Q² = 1 - Σᵢ (g(uᵢ) - ĝ₋ᵢ(uᵢ))² / Σᵢ (g(uᵢ) - ḡ)²
```
où ĝ₋ᵢ = modèle entraîné sans le point i, évalué en uᵢ.

**Astuce "Corrected" (hat matrix) :**
La formule normale nécessiterait n0 ré-entraînements. L'astuce analytique utilise la **hat matrix** H = Ψ(ΨᵀΨ)⁻¹Ψᵀ :
```
g(uᵢ) - ĝ₋ᵢ(uᵢ) = (g(uᵢ) - ĝ(uᵢ)) / (1 - Hᵢᵢ)
```
`Hᵢᵢ` = influence du point i sur sa propre prédiction. `Hᵢᵢ = 1` → point totalement auto-influent.
Cette astuce ne fonctionne que pour la **régression moindres carrés linéaire**.

**Utilisation dans LARS :** LARS génère toute la séquence, calcule Q² LOO pour chaque modèle via CorrectedLOO, retient le modèle au maximum du Q².

**Important :** CorrectedLOO est utilisé par LARS en **interne** lors de la sélection. `FunctionalChaosValidation(result)` seul essaie de refaire cette LOO analytique **depuis l'extérieur** sur le résultat final — mais OpenTURNS refuse car il détecte que LARS a fait de la sélection → erreur.

---

### 1g. Matrice Ψ et régression moindres carrés

Ψ[i,k] = ψₖ(uᵢ) : matrice (n0 × n_termes). Chaque ligne = un point DOE, chaque colonne = un polynôme évalué.

Minimiser `||y - Ψα||²` → équations normales → `α = (ΨᵀΨ)⁻¹Ψᵀy`.

`ĝ = Ψα = Hy` où H = Ψ(ΨᵀΨ)⁻¹Ψᵀ est la hat matrix.

Appeler ça une "projection" car ĝ est la **projection orthogonale** de y sur l'espace engendré par les colonnes de Ψ. Condition de projection : `Ψᵀ(y - Ψα) = 0` = équations normales.

---

### 1h. Indices de Sobol

S1_k = part de Var(g) expliquée par la variable k **seule** (effet direct).
ST_k = part de Var(g) due à k y compris interactions avec les autres variables (effet total).

Propriétés : `Σ S1_k ≤ 1`, `Σ ST_k ≥ 1`. `ST_k > S1_k` → k interagit avec d'autres variables.

Pour un PCE, ces indices sont calculés analytiquement depuis les coefficients α (variance partielle = somme des α² des termes contenant k).

---

### 1i. try_pce vs do_pce — séparation des responsabilités

```
try_pce = intention utilisateur ("tenter le PCE ?")
do_pce  = résultat calculé ("le PCE est-il assez bon ?")
```
Sans cette séparation, on serait obligé d'écraser `do_pce` après l'avoir fixé, ce qui est confus.

---

### 1j. Formats numpy / OpenTURNS

| Format | Shape | Usage |
|---|---|---|
| `ot.Point([u1, u2])` | (n_var,) | Un seul point — pour `gradient()` |
| `ot.Sample([[u1,u2]])` | (1, n_var) | Batch d'un point — pour `__call__()` |
| `np.array(u).reshape(1,-1)` | (1, n_var) | Pour `sm.predict_values()` SMT |
| `np.atleast_2d(u)` | ≥ (1, n_var) | Garantit 2D mais pas la shape exacte |
| `u_np[0]` | (n_var,) | Extrait la première ligne d'un array 2D |

`metamodel_pce(u_ot)` attend `ot.Sample` (vectorisé).
`metamodel_pce.gradient(u_ot)` attend `ot.Point` (un seul point à la fois).

---

## 2. Modifications du code AC_pure_flexion.py

### 2a. Section OPTIONS (lignes 292-314) — nouvelle structure

```python
# OPTIONS FORM
n_max_FORM = 50
tol_FORM = 0.2          # nouvelle variable globale (anciennement hardcodé)
do_warm_start = False
tol_warm_start = 0.0001
n_multi_start = 1

# OPTIONS MODELE
do_GP = True
n0 = max(5, n_multi_start)
do_GEK = False          # False = KRG, True = GEK
do_GEK_analytic_grad = True
reduc_PLS = 0

# PCE
try_pce = True
do_pce = False          # sera mis à True si Q² > seuil
seuil_pce = 0.90
min_max_degree = 1
```

### 2b. Bloc ETAPE 3.1 — PCE réécrit from scratch (lignes 392-445)

**Avant :** bloc désactivé avec `do_pce = False`, utilisant `basisStrategy` comme nom de variable.

**Après :**
```python
def result_PCE(inputSample, outputSample, dist_U, q, max_degree, seuil_pce,
               propos=ot.LARS(), select=ot.CorrectedLeaveOneOut(),
               min_max_degree=min_max_degree):
    n_var = inputSample.getDimension()
    enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(n_var, q)
    basis = ot.OrthogonalProductPolynomialFactory([ot.HermiteFactory()] * n_var, enumerateFunction)
    basis_size = enumerateFunction.getBasisSizeFromTotalDegree(max_degree)
    adaptiveStrategy = ot.FixedStrategy(basis, basis_size)
    selectionStrategy = ot.LeastSquaresMetaModelSelectionFactory(propos, select)
    projectionStrategy = ot.LeastSquaresStrategy(selectionStrategy)
    algo = ot.FunctionalChaosAlgorithm(inputSample, outputSample, dist_U, adaptiveStrategy, projectionStrategy)
    algo.run()
    return algo.getResult()

if try_pce:
    inputSample = U_doe
    outputSample = ot.Sample([[SOL[i]['g']] for i in range(n0)])
    q = 0.75
    max_degree = 2
    result = result_PCE(inputSample, outputSample, dist_U, q, max_degree, seuil_pce,
                        propos=ot.LARS(), select=ot.CorrectedLeaveOneOut(),
                        min_max_degree=min_max_degree)
    VALIDATION = {}
    validation = ot.FunctionalChaosValidation(result, ot.LeaveOneOutSplitter(n0))  # ← FIX bug
    q2_loo = validation.computeR2Score()[0]
    while q2_loo < seuil_pce and max_degree > min_max_degree:
        max_degree -= 1
        result = result_PCE(inputSample, outputSample, dist_U, q, max_degree, seuil_pce,
                            propos=ot.LARS(), select=ot.CorrectedLeaveOneOut(),
                            min_max_degree=min_max_degree)
        validation = ot.FunctionalChaosValidation(result, ot.LeaveOneOutSplitter(n0))
        q2_loo = validation.computeR2Score()[0]
    if q2_loo > seuil_pce:
        VALIDATION['PCE'] = True
        print(f"Le PCE est de bonne qualité avec un degré maximum de {max_degree}.")
        do_pce = True
        metamodel_pce = result.getMetaModel()
        y_pce = np.array(metamodel_pce(U_doe))            # (n0, 1)
        all_grad_PCE = np.zeros((n0, n_var))
        for i in range(n0):
            grad_pce_u = metamodel_pce.gradient(U_doe[i]) # U_doe[i] = ot.Point
            for j in range(n_var):
                all_grad_PCE[i, j] = grad_pce_u[j, 0]
    else:
        VALIDATION['PCE'] = False
        print("Attention : Q2 faible. Le métamodèle est fait avec GP pur.")
        do_pce = False
```

### 2c. metamodel_GEK — réécriture complète (lignes 477-497)

**Avant :** retournait seulement `val`, `g_GEK` et `grad_g_GEK` séparées.

**Après :** `metamodel_GEK` retourne `(val, grad)` ensemble :
```python
def metamodel_GEK(u, do_pce=do_pce):
    u_np = np.array(u).reshape(1, -1)       # shape (1, n_var) pour SMT
    val = float(sm.predict_values(u_np)[0, 0])
    grad = np.array([[float(sm.predict_derivatives(u_np, kx)[0, 0])] for kx in range(n_var)])
    if do_pce:
        u_sample_ot = ot.Sample(u_np)       # pour metamodel_pce()
        u_point_ot = ot.Point(u_np[0])      # pour metamodel_pce.gradient()
        val += float(np.array(metamodel_pce(u_sample_ot))[0, 0])
        grad += np.array(metamodel_pce.gradient(u_point_ot))
    return val, grad

def g_GEK(u):
    val, _ = metamodel_GEK(u, do_pce=do_pce)
    return [val]       # ← liste obligatoire pour ot.PythonFunction

def grad_g_GEK(u):
    _, grad = metamodel_GEK(u, do_pce=do_pce)
    return grad        # shape (n_var, 1)
```

### 2d. tol_FORM utilisé partout (lignes 587, 630, 643)

Remplace les valeurs hardcodées `1e-2` ou `5e-2` :
```python
solver.setMaximumConstraintError(tol_FORM)
```

---

## 3. Problèmes rencontrés et solutions

### Problème 1 — `do_pce` NameError si `try_pce = False`
**Symptôme :** `do_pce` n'était défini qu'à l'intérieur du bloc `if try_pce:`. Si `try_pce=False`, ligne 440 (`if do_pce:`) lève `NameError`.
**Solution retenue :** Ajouter `do_pce = False` dans la section OPTIONS (ligne 311), au même niveau que `try_pce`.

---

### Problème 2 — `metamodel_pce`, `y_pce`, `all_grad_PCE` jamais calculés
**Symptôme :** La fonction `result_PCE` retournait uniquement `result`, sans extraire le métamodèle ni les prédictions/gradients sur le DOE. L'ETAPE 3.2 utilisait `y_pce` et `all_grad_PCE` qui n'existaient pas.
**Solution retenue :** Extraire ces trois variables dans le bloc `if q2_loo > seuil_pce:`.

---

### Problème 3 — `g_GEK` retournait un `float` au lieu d'une liste
**Symptôme :** `g_GEK(u)[0]` plantait (indexation sur un float). `ot.PythonFunction(n_var, 1, g_GEK)` attend `[float]`.
**Solution retenue :** `return [val]` avec crochets.

---

### Problème 4 — `grad_g_GEK` ne réajoutait pas le gradient PCE
**Symptôme :** Même avec `do_pce=True`, le gradient du métamodèle hybride n'incluait que la partie GEK.
**Solution retenue :** Dans `metamodel_GEK`, ajouter `grad += np.array(metamodel_pce.gradient(u_point_ot))` dans le bloc `if do_pce:`.

---

### Problème 5 — Shape mismatch dans le warm start GEK (à corriger plus tard)
**Symptôme :** Après ajout d'un point warm start, `y_hf` et `all_grad_U_g` ont n0+1 lignes mais `y_pce` et `all_grad_PCE` en ont toujours n0. `yt -= y_pce` planterait.
**Solution non encore implémentée** (reportée car `do_warm_start=False` pour les tests) :
```python
if do_pce:
    u_warm_np = np.array(U_warm).reshape(1, -1)
    y_pce = np.vstack([y_pce, np.array(metamodel_pce(ot.Sample(u_warm_np)))])
    grad_pce_new = np.array(metamodel_pce.gradient(ot.Point(u_warm_np[0]))).T  # (1, n_var)
    all_grad_PCE = np.vstack([all_grad_PCE, grad_pce_new])
    yt -= y_pce
    all_grad -= all_grad_PCE
```

---

### Problème 6 — Shape mismatch dans le warm start KRG (à corriger plus tard)
**Symptôme :** Même problème que ci-dessus pour le bloc KRG. De plus, le nouveau `metamodel_KRG` reconstruit après warm start n'inclut pas le PCE.
**Solution non encore implémentée :** Même fix pour `y_pce` + ajouter `if do_pce: metamodel_KRG += metamodel_pce` après reconstruction.

---

### Problème 7 — `FunctionalChaosValidation` incompatible avec LARS (bug run)
**Symptôme :**
```
TypeError: InvalidArgumentException : Cannot perform fast cross-validation
with a polynomial chaos expansion involving model selection
```
**Cause :** `FunctionalChaosValidation(result)` sans deuxième argument utilise l'astuce analytique (hat matrix H). Celle-ci n'est valide que sans sélection de termes. Quand LARS a sélectionné un sous-ensemble, OpenTURNS refuse.
**Note :** LARS utilise bien CorrectedLOO en interne lors de la sélection — le bug est uniquement sur l'objet `FunctionalChaosValidation` externe.
**Solution retenue :**
```python
validation = ot.FunctionalChaosValidation(result, ot.LeaveOneOutSplitter(n0))
```
Cela fait n0 vrais ré-entraînements (brute force LOO) valides dans tous les cas.

---

## 4. État du code au moment de l'arrêt de session

```python
# OPTIONS FORM
n_max_FORM = 50
tol_FORM = 0.2
do_warm_start = False
tol_warm_start = 0.0001
n_multi_start = 1

# OPTIONS MODELE
do_GP = True
n0 = max(5, n_multi_start)   # = 5
do_GEK = False               # → mode KRG
do_GEK_analytic_grad = True
reduc_PLS = 0

# PCE
try_pce = True
do_pce = False
seuil_pce = 0.90
min_max_degree = 1
```

DOE : 5 points fixes dans espace U (lignes 336-342 de AC_pure_flexion.py).

**Dernier run :** crashé sur le bug FunctionalChaosValidation (Problème 7). Fix identifié, non encore appliqué.

---

## 5. Prochaines étapes (après session du matin 21/04)

1. ~~**Corriger le bug FunctionalChaosValidation**~~ → **ERREUR dans ce résumé** : la solution `ot.FunctionalChaosValidation(result, ot.LeaveOneOutSplitter(n0))` CRASH AUSSI (voir session après-midi 21/04 ci-dessous).
2. **Relancer** le code PCE+KRG et vérifier Q² LOO, β, Pf.
3. **Créer les fichiers résultats** selon le même schéma que les sessions précédentes : `resultats_PCE_KRG.md` et `comparaison_HF_KRG_PCE.md`.
4. **Implémenter le fix warm start PCE** (Problèmes 5 et 6) une fois le run de base validé.

---

## 6. Fichiers clés

| Fichier | Rôle | État |
|---|---|---|
| `AC_pure_flexion.py` | Script principal | PCE réécrit, FunctionalChaosValidation à corriger |
| `launcher.py` | Lance avec DLL STRAINS | Inchangé |
| `out_run_PCE_KRG.txt` | Sortie du dernier run | Crash ligne 418, 5 runs STRAINS OK |
| `resume_session_1704_aprem.md` | Session HF FORM | Complet |
| `resume_session_2004.md` | Session KRG FORM | Complet |
| `resume_session_2104.md` | Cette session | Complet |

---
---

# Suite de session — Après-midi 21 avril 2026 : Correction LOO LARS

## Contexte de reprise

La session du matin avait identifié `ot.FunctionalChaosValidation(result, ot.LeaveOneOutSplitter(n0))` comme solution au crash. **Cette solution est fausse** : elle crash aussi. La session après-midi a cherché la vraie solution et l'a trouvée.

---

## A. Problèmes rencontrés et solutions

### Problème A1 — `FunctionalChaosValidation` crash MÊME avec `LeaveOneOutSplitter`

**Symptôme exact :**
```
TypeError: InvalidArgumentException : Cannot perform fast cross-validation
with a polynomial chaos expansion involving model selection
```
Levée quel que soit le deuxième argument du constructeur (`LeaveOneOutSplitter`, `KFoldSplitter`, rien du tout). Le crash se produit **à la construction de l'objet**, pas à l'appel de `computeR2Score()`.

**Cause réelle (après recherche dans le code source OT 1.26) :**
- L'objet `FunctionalChaosValidation` appelle en C++ `result.involvesModelSelection()` dans son constructeur
- Si LARS a été utilisé, cette méthode retourne `True`
- OT 1.26 lève immédiatement l'exception, **sans aller plus loin**, quel que soit le splitter
- Ce check est dans du code C++ compilé, non contournable par héritage Python

**Source :**
- `C:\python3\lib\site-packages\openturns\analytical.py` (wrapper Python)
- Comportement C++ sous-jacent dans `FunctionalChaosValidation.cxx`

**Solutions tentées et abandonnées :**

| Tentative | Résultat | Pourquoi ça échoue |
|---|---|---|
| `FunctionalChaosValidation(result, LeaveOneOutSplitter(n0))` | Crash C++ au constructeur | `involvesModelSelection()` check avant toute autre logique |
| `FunctionalChaosValidation(result, KFoldSplitter(n0, 3))` | Crash C++ au constructeur | Même vérification |
| Sous-classer `FunctionalChaosValidation` en Python | Impossible | Constructeur C++ inaccessible |
| `try/except` autour du constructeur | Catch l'exception mais pas de résultat | L'objet n'est pas construit, aucun résultat disponible |
| `ResourceMap.SetAsBool("FunctionalChaosValidation-ModelSelection", True)` | **Fonctionne techniquement** | Utilise la formule analytique biaisée (voir ci-dessous) — **rejetée par l'utilisatrice** |

**Solution retenue : LOO manuel (voir ci-dessous)**

---

### Problème A2 — `getErrorHistory()` : MSE, pas Q²

**Contexte :** Exploration d'une alternative à `FunctionalChaosValidation` via la méthode `getErrorHistory()` sur le résultat LARS.

**Test réalisé sur le run PCE (n0=5, max_degree=2) :**
```python
result.getProjectionStrategy().getErrorHistory()
# Retourne : [1.875, 3.23e-30, 7.65e-29]
```

**Ce que contient `getErrorHistory()` :**
- Ce sont des **MSE** (Mean Squared Error) calculés à chaque étape LARS pour le modèle de cette étape
- Formule : `MSE_k = Σᵢ (eᵢ^(k) / (1 - Hᵢᵢ^(k)))² / n` (hat matrix, formule analytique)
- Valeurs non bornées entre 0 et 1 — peuvent dépasser 1 ou être négatives si données mal conditionnées
- **Pas des Q²** → on ne peut pas directement comparer à `seuil_pce = 0.90`

**Conversion possible mais biaisée :**
```python
# Q² dérivable MAIS avec le même biais analytique
q2_approx = 1 - min(result.getProjectionStrategy().getErrorHistory()) / outputSample.computeVariance()[0]
```
Cette formule retourne bien un Q² dans [−∞, 1], mais il est **optimiste** (même biais post-sélection qu'avec `FunctionalChaosValidation`) → **non retenu**.

---

### Problème A3 — Biais post-sélection : pourquoi la LOO analytique est optimiste avec LARS

**Question de l'utilisatrice :** "Je ne comprends pas pourquoi OpenTURNS refuse, et en quoi c'est biaisé."

**Explication complète (construite progressivement en session) :**

La LOO analytique utilise la hat matrix H = Ψ(ΨᵀΨ)⁻¹Ψᵀ. Elle suppose que Ψ est **fixe** (indépendante des données). L'erreur de prédiction LOO est alors :
```
ĝ₋ᵢ(uᵢ) = ĝ(uᵢ) - eᵢ / (1 - Hᵢᵢ)   avec eᵢ = g(uᵢ) - ĝ(uᵢ)
```

**Le problème avec LARS :** LARS construit Ψ en regardant les données y = {g(uᵢ)}. Ψ = Ψ(y). Donc :
1. Ψ a été choisi pour **maximiser la corrélation avec y**
2. Les résidus eᵢ = g(uᵢ) - ĝ(uᵢ) sont **artificiellement petits** (LARS a optimisé pour y)
3. La formule analytique ne "sait" pas que Ψ a été adapté aux données → sous-estime les erreurs LOO → Q² trop optimiste

C'est ce qu'on appelle le **biais de post-sélection** : quand le modèle (ici la base Ψ) a été sélectionné en regardant les données, les métriques calculées sur ces mêmes données sont biaisées à la hausse.

OpenTURNS refuse parce qu'il a conscience de ce biais et protège l'utilisateur de statistiques trompeuses.

---

### Problème A4 — `ResourceMap.SetAsBool("FunctionalChaosValidation-ModelSelection", True)` : solution biaisée

**Ce qu'elle fait exactement :**
- Contourne le check `involvesModelSelection()` dans le constructeur C++
- Calcule Q² avec la formule hat matrix (CorrectedLOO) **sur le modèle final sélectionné par LARS**
- Retourne un Q² optimiste (biais post-sélection non corrigé)

**Pourquoi rejetée par l'utilisatrice :** Donne un score trompeur et contredit le principe de validation non biaisée recherché.

---

## B. Sujets théoriques approfondis

### B1. Pratique industrielle de validation LOO pour LARS

**Blatman & Sudret (2011) — article fondateur :**
- La LOO analytique (CorrectedLeaveOneOut) est utilisée **pendant LARS** comme critère de sélection/arrêt des termes
- Ce n'est PAS une métrique de validation finale — c'est un outil de construction interne
- Pour la validation finale, l'article s'appuie sur des ensembles de validation indépendants

**UQLab (ETH Zürich) — recommandations explicites :**
- Si assez de données → **holdout set** indépendant non vu pendant l'entraînement
- Si données limitées (comme notre cas n0=5) → **double validation croisée** = re-entraîner LARS à chaque fold, valider sur le fold laissé de côté
- C'est exactement le LOO manuel = notre `compute_q2_loo`

**OpenTURNS documentation officielle (1.26) :**
> "If a model selection method is used (such as LARS), then the fast CV method can produce an optimistic estimated error."

OT refuse plutôt que de donner un score trompeur.

**Conclusion :** La boucle LOO manuelle avec re-entraînement par fold = "double cross-validation" = standard rigoureux dans la communauté UQLab/fiabilité.

---

### B2. `MetaModelValidation` — classe mère, sans restriction LARS

`FunctionalChaosValidation` hérite de `MetaModelValidation`. La classe mère prend directement `(outputSample, predictions)` sans jamais toucher au résultat PCE ni à LARS.

**API :**
```python
val = ot.MetaModelValidation(outputSample, predictions)
q2 = val.computeR2Score()[0]   # Q² dans [−∞, 1], 1 = fit parfait
```

- `outputSample` : `ot.Sample` de shape (n, 1) — valeurs réelles
- `predictions` : `ot.Sample` de shape (n, 1) — prédictions du métamodèle
- `computeR2Score()` : retourne `ot.Point` de taille 1

**Q² calculé :**
```
Q² = 1 - Σᵢ (yᵢ - ŷᵢ)² / Σᵢ (yᵢ - ȳ)²
```
Aucune restriction sur la façon dont `ŷᵢ` a été produit → aucun problème avec LARS.

---

### B3. `LeaveOneOutSplitter` — itérabilité en Python

```python
splitter = ot.LeaveOneOutSplitter(n)
for indicesTrain, indicesTest in splitter:
    # indicesTrain : ot.Indices de taille n-1
    # indicesTest  : ot.Indices de taille 1
    subSample = inputSample[indicesTrain]   # syntaxe officielle OT
```

- `inputSample[indicesTrain]` : retourne un `ot.Sample`
- `int(indicesTest[k])` pour convertir en entier Python
- L'objet est **consommé** après une itération — recréer si besoin de plusieurs passes

---

### B4. `result_PCE` avec n-1 points — comportement LARS à petite taille

Avec n0=5 et LOO, chaque fold entraîne le PCE sur 4 points :
- LARS sélectionne au maximum 3 termes (s'arrête avant n_points termes)
- max_degree=2, n_var=2 → 5 termes candidats, LARS en garde 3 max
- La sélection des termes peut varier d'un fold à l'autre (normal)

---

## C. Solution retenue — LOO manuel `compute_q2_loo`

### C1. Code exact (approuvé, non encore appliqué au 21/04)

**Fonction helper à insérer entre `result_PCE` et `if try_pce:` :**

```python
    def compute_q2_loo(inputSample, outputSample, dist_U, q, max_degree, seuil_pce):
        n = inputSample.getSize()
        splitter = ot.LeaveOneOutSplitter(n)
        g_loo_list = [None] * n
        for indicesTrain, indicesTest in splitter:
            result_i = result_PCE(inputSample[indicesTrain], outputSample[indicesTrain],
                                  dist_U, q, max_degree, seuil_pce)
            pred = result_i.getMetaModel()(inputSample[indicesTest])
            for k in range(len(indicesTest)):
                g_loo_list[int(indicesTest[k])] = pred[k, 0]
        g_loo_pred = ot.Sample([[v] for v in g_loo_list])
        return ot.MetaModelValidation(outputSample, g_loo_pred).computeR2Score()[0]
```

**Deux remplacements dans le bloc `if try_pce:` (lignes 418-419 et 423-424) :**

```python
# AVANT (crash)
validation = ot.FunctionalChaosValidation(result, ot.LeaveOneOutSplitter(n0))
q2_loo = validation.computeR2Score()[0]

# APRÈS (correct, non biaisé)
q2_loo = compute_q2_loo(inputSample, outputSample, dist_U, q, max_degree, seuil_pce)
```

### C2. Pourquoi ce code est non biaisé

À chaque fold, LARS reçoit seulement les n-1 points d'entraînement. Il construit Ψ sans voir le point de test → la prédiction sur ce point est honnête. Le Q² final est calculé sur les n prédictions LOO, ce qui correspond à la vraie performance de généralisation du processus LARS.

### C3. Coût computationnel

n0 folds × 1 run PCE par fold = 5 runs PCE supplémentaires. Très rapide (pas de STRAINS), négligeable.

---

## D. État du code au 21/04 après-midi

- `AC_pure_flexion.py` : PCE réécrit correctement, validation encore cassée (lignes 418-424 toujours avec `FunctionalChaosValidation`)
- Modification approuvée mais **pas encore appliquée** (l'utilisatrice veut comprendre avant)
- Sauvegardée dans mémoire Claude : `memory/project_pce_loo_fix_pending.md`

**Prochaine étape immédiate :** Appliquer le fix `compute_q2_loo` (lignes 411-424), relancer via `launcher.py`.

---

## E. Ce qu'OpenTURNS 1.26 permet et ne permet pas avec LARS

| Action | Possible ? | Raison |
|---|---|---|
| Construire PCE avec LARS | Oui | `FunctionalChaosAlgorithm` |
| `FunctionalChaosValidation` sans argument | Non | Crash C++, check `involvesModelSelection()` |
| `FunctionalChaosValidation(result, splitter)` | Non | Même check au constructeur |
| `ResourceMap("FunctionalChaosValidation-ModelSelection", True)` | Oui technique | Mais biaisé — déconseillé |
| `getErrorHistory()` pour Q² | Non direct | Retourne MSE non borné, pas Q² |
| LOO manuel + `MetaModelValidation` | Oui | Aucune restriction, non biaisé |
| Indices de Sobol analytiques | Oui | `FunctionalChaosSobolIndices(result)` |
| Gradient du métamodèle PCE | Oui | `result.getMetaModel().gradient(ot.Point)` |
