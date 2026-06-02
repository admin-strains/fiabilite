# Contexte technique — GEKPLS (SMT) + OpenTURNS FORM

Document de contexte destiné à Claude Code pour poursuivre le travail sur `AC_pure_flexion.py`.
Rédigé à partir de la lecture directe des documentations officielles SMT et OpenTURNS.

---

## 1. Sources lues

| Ressource | URL |
|---|---|
| GEKPLS SMT v2.5.0 | https://smt.readthedocs.io/en/v2.5.0/_src_docs/surrogate_models/gekpls.html |
| GEKPLS source RST GitHub | https://github.com/SMTorg/smt/blob/master/doc/_src_docs/surrogate_models/gekpls.rst |
| SMT API générale | https://smt.readthedocs.io/en/stable/_src_docs/surrogate_models.html |
| Issue SMT #174 (contrainte 1D) | https://github.com/SMTorg/smt/issues/174 |
| Issue SMT #186 (bug `predict_derivatives`) | https://github.com/SMTorg/smt/issues/186 |
| OpenTURNS FORM | https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.FORM.html |
| OpenTURNS FORMResult | https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.FORMResult.html |
| OpenTURNS AbdoRackwitz | https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.AbdoRackwitz.html |
| OpenTURNS OptimizationAlgorithm | https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.OptimizationAlgorithm.html |

---

## 2. GEKPLS (SMT) — Résumé

### 2.1 Principe

GEKPLS = **Gradient-Enhanced Kriging + Partial Least Squares**. Extension du krigeage qui exploite les gradients sans exploser la taille de la matrice de corrélation.

**Mécanisme** :
1. Autour de chaque point d'échantillonnage, génère des points d'approximation par **Taylor ordre 1** (utilisant le gradient).
2. Applique PLS plusieurs fois localement → coefficients d'influence de chaque variable.
3. Moyenne des coefficients PLS → influence globale.
4. Seulement `m` (= `extra_points`) points Taylor ajoutés par point, retenant les `m` plus hauts coefficients de la 1re CP PLS.

**Noyau gaussien** :

$$k(x^{(i)}, x^{(j)}) = \sigma \prod_{l=1}^{n_x} \prod_{k=1}^{h} \exp\left(-\theta_k \left(w_l^{(k)} x_l^{(i)} - w_l^{(k)} x_l^{(j)}\right)^2\right)$$

Réduction du nombre d'hyperparamètres de `nx` à `h` avec `nx >> h`.

### 2.2 Options constructeur `GEKPLS(...)`

| Option | Type | Rôle |
|---|---|---|
| `design_space` | `DesignSpace` | API recommandée (remplace `xlimits` dans versions récentes) |
| `theta0` | `list[float]` de longueur `n_comp` | Hyperparamètres initiaux |
| `n_comp` | `int` | Nombre de composantes PLS (dimension réduite `h`) |
| `extra_points` | `int ∈ [1, nx]` | Nombre de points Taylor par point d'échantillonnage |
| `delta_x` | `float` (ex. `1e-2`) | Pas du développement de Taylor |
| `poly` | `'constant'` / `'linear'` / `'quadratic'` | Fonction de régression de la tendance |
| `corr` | `'squar_exp'` uniquement | Noyau lisse imposé (gradients) |
| `pow_exp_power` | `float ∈ (0, 2]` | Paramètre noyau puissance-exp |
| `print_training/prediction/problem/solver` | `bool` | Toggles d'affichage |
| `categorical_kernel`, `hierarchical_kernel` | `MixIntKernelType` | Pour mixte, **incompatible avec dérivées** |

### 2.3 API

```python
sm = GEKPLS(design_space=..., theta0=..., n_comp=..., extra_points=..., delta_x=...)
sm.set_training_values(xt, yt)                     # f(x)
for i in range(ndim):
    sm.set_training_derivatives(xt, dy_dxi, i)     # ∂f/∂x_i
sm.train()
y    = sm.predict_values(x)                        # OK, fiable
dydx = sm.predict_derivatives(x, kx=i)             # ⚠️ voir §2.5
```

### 2.4 Contraintes et limitations

- **Dimension minimale = 2** (issue #174). `IndexError` dans `ge_compute_pls` pour `ndim=1`. Contrainte d'implémentation.
- **Corrélation restreinte à `squar_exp`** (différentiabilité du noyau).
- **Variables mixtes incompatibles avec les dérivées**.
- **Sortie scalaire uniquement** — 1 surrogat par sortie.

### 2.5 ⚠️ Bug connu sur `predict_derivatives` (Issue SMT #186)

**Important** : la méthode `predict_derivatives(x, kx=i)` **existe** sur GEKPLS (héritée de `KrgBased`) mais a un **historique de bugs** :

- Avec `poly='linear'` : gradient **totalement faux** (ex. `[7.29, 8.72]` au lieu de `[0.76, 9.28]` sur un test sphere 2D).
- Avec `poly='constant'` (défaut) : résultats corrects sur ce même test.
- Un autre utilisateur rapportait un `IndexError` sur `krg_based.py:348` (`self.X_std[kx]`) selon la configuration `n_comp`.
- Bug marqué "Fixed by #188 or #200" → corrigé dans versions récentes mais **à valider sur la version installée**.

**Conséquence pratique** : ne pas injecter aveuglément `predict_derivatives` comme gradient analytique dans un optimiseur FORM. **Toujours valider contre différences finies** d'abord (voir §5.3).

### 2.6 Positionnement vs full GEK

| Aspect | GEKPLS | Full GEK (matrice augmentée) |
|---|---|---|
| Utilisation gradients | Points Taylor fictifs + PLS | Injection directe dans K augmentée |
| Taille matrice | ~ `nt(1+extra_points)` | `nt(1+nx)` |
| Hyperparamètres | `h` (après PLS) | `nx` |
| Conditionnement | Amélioré par PLS | Ill-conditioning connu |
| Adapté aux gradients adjoints bon marché | Sous-exploite l'info | **Exploite toute l'info** |

**Dans le projet Semia** : GEKPLS a été abandonné au profit de **full GEK avec matrice de covariance augmentée**, justifié par la disponibilité de gradients adjoints peu coûteux. La logique de wrapping OpenTURNS reste la même, seules les fonctions de prédiction changent.

---

## 3. OpenTURNS FORM — Résumé

### 3.1 Principe

FORM = **First Order Reliability Method**. Linéarise la fonction d'état limite `G(U, d)` au **point de conception P*** — point sur la surface `G(U, d) = 0` le plus proche de l'origine de l'espace standard (après transformation iso-probabiliste type Nataf).

Probabilité de défaillance exacte sur l'hyperplan tangent grâce à l'invariance par rotation de `f_U` :

$$P_f = \Phi(-\beta)$$

où β = indice de Hasofer-Lind = `‖U*‖`.

### 3.2 Constructeur

```python
ot.FORM(nearestPointAlgorithm, event)
```

- `nearestPointAlgorithm` : `OptimizationAlgorithm` (AbdoRackwitz, Cobyla, TNC, SQP…)
- `event` : `RandomVector` (typiquement `ThresholdEvent`)

### 3.3 Exemple canonique

```python
myFunction    = ot.PythonFunction(d, 1, g_py)  # ou SymbolicFunction
distribution  = ot.JointDistribution(marginals, copula)
vect          = ot.RandomVector(distribution)
output        = ot.CompositeRandomVector(myFunction, vect)
event         = ot.ThresholdEvent(output, ot.Less(), 0.0)

solver = ot.AbdoRackwitz()
solver.setStartingPoint(distribution.getMean())  # espace physique
solver.setMaximumIterationNumber(100)            # ← c'est n_max_FORM
algo = ot.FORM(solver, event)
algo.run()
result = algo.getResult()
```

**Note de version** : dans OpenTURNS ≥ 1.22, `setStartingPoint` est posé **sur le solveur**, pas sur FORM directement. L'ancienne signature `ot.FORM(solver, event, start)` est dépréciée.

### 3.4 Classes voisines

- `SORM` : approximation du 2nd ordre
- `MultiFORM` : plusieurs points de conception (modes de défaillance multiples)
- `SystemFORM` : événements système (IntersectionEvent, UnionEvent)
- `StrongMaximumTest` : vérification *a posteriori* de l'unicité du point de conception

---

## 4. `FORMResult` — Accès aux sorties

| Accesseur | Retour |
|---|---|
| `getEventProbability()` | `P_f` (FORM) |
| `getHasoferReliabilityIndex()` | `β_HL = ‖U*‖` |
| `getGeneralisedReliabilityIndex()` | `β_g = ±β_HL` selon origine dans domaine de défaillance |
| `getStandardSpaceDesignPoint()` | `U*` |
| `getPhysicalSpaceDesignPoint()` | `X*` (iso-probabiliste inverse) |
| `getImportanceFactors()` | Facteurs α² |
| `getIsStandardPointOriginInFailureSpace()` | bool |
| `getOptimizationResult()` | **clé pour diagnostiquer non-convergence** |
| `getEventProbabilitySensitivity()` | Sensibilités aux paramètres de distribution |
| `drawImportanceFactors()`, `drawEventProbabilitySensitivity()` | Graphes |

### Diagnostic de convergence

```python
opt_res = result.getOptimizationResult()
print("Iterations   :", opt_res.getIterationNumber())
print("Calls        :", opt_res.getCallsNumber())
print("Abs error    :", opt_res.getAbsoluteError())
print("Rel error    :", opt_res.getRelativeError())
print("Residual err :", opt_res.getResidualError())
print("Constr error :", opt_res.getConstraintError())
```

Si `IterationNumber == MaximumIterationNumber` → le solveur a terminé **sur la borne d'itérations**, pas par convergence.

---

## 5. `AbdoRackwitz` — Paramètres et réglage

Algorithme SQP spécialisé pour `NearestPointProblem` (minimiser `‖U‖²` sous `G(U) = 0`). Hérite de `OptimizationAlgorithm`.

### 5.1 Paramètres

| Paramètre | Accesseur | Rôle |
|---|---|---|
| Point de départ | `setStartingPoint(Point)` | **Obligatoire**, en espace physique |
| Itérations max | `setMaximumIterationNumber(n)` | **= `n_max_FORM` dans le code Semia** |
| Appels objectif max | `setMaximumCallsNumber(n)` | Borne évaluations de G |
| Erreur absolue max | `setMaximumAbsoluteError(ε)` | `‖U_{k+1} − U_k‖` |
| Erreur relative max | `setMaximumRelativeError(ε)` | Critère relatif |
| Erreur résiduelle max | `setMaximumResidualError(ε)` | `|G(U)|` |
| Erreur contrainte max | `setMaximumConstraintError(ε)` | Violation de contrainte |
| Durée max | `setMaximumDuration(s)` | Secondes |
| `omega` | `setOmega(ω)` | Facteur d'Armijo (line search) |
| `tau` | `setTau(τ)` | Décroissance multiplicative du pas |
| `checkStatus` | `setCheckStatus(bool)` | Si `False`, pas d'exception en non-convergence |

### 5.2 Pathologie identifiée dans `AC_pure_flexion.py`

**`n_max_FORM = 10` insuffisant**. Valeur par défaut OpenTURNS ≈ 100. Avec couplage à un solveur externe (STRAINS/DS) et éventuellement un surrogat bruité, 10 itérations sont systématiquement insuffisantes.

**Recommandation** : `setMaximumIterationNumber(100)` minimum pour le run, puis ajuster selon diagnostic.

### 5.3 Protocole de debug recommandé

1. **Augmenter `n_max_FORM` à 100**.
2. **Activer `setCheckStatus(False)`** en phase debug pour récupérer un candidat même non-convergé.
3. **Inspecter `getOptimizationResult()`** après chaque run.
4. **Comparer les critères d'erreur** (absolute/residual) aux tolérances.
5. **Tester un solveur alternatif** si G est bruitée : `ot.Cobyla()` ou `ot.SQP()`.
6. **Point de départ** : `distribution.getMean()` ou médianes, plus robuste que `[0,0,...]`.

---

## 6. Wrapper GEKPLS → `openturns.Function`

OpenTURNS attend un objet `Function`. Le pont est `ot.PythonFunction(nIn, nOut, func, ...)`.

### 6.1 Version sans gradient analytique (recommandée par défaut pour GEKPLS)

**Compte tenu du bug #186**, version à adopter tant que la validation du gradient n'est pas faite :

```python
import numpy as np
import openturns as ot

def g_py(X):
    x = np.atleast_2d(np.asarray(X, dtype=float))
    y = sm.predict_values(x)
    return [float(y[0, 0])]

def g_py_sample(X):
    x = np.asarray(X, dtype=float)
    return sm.predict_values(x)

myFunction = ot.PythonFunction(d, 1, func=g_py, func_sample=g_py_sample)
# PAS de gradient → OpenTURNS fait des différences finies (d+1 appels par itération)
```

Coût surrogat = ms, différences finies non pénalisantes.

### 6.2 Version avec gradient analytique (UNIQUEMENT après validation)

**À n'utiliser qu'après avoir validé que `predict_derivatives` donne des valeurs correctes sur la version SMT installée** :

```python
def g_grad(X):
    x = np.atleast_2d(np.asarray(X, dtype=float))
    J = np.zeros((d, 1))  # convention OT : (nInput, nOutput)
    for i in range(d):
        J[i, 0] = sm.predict_derivatives(x, kx=i)[0, 0]
    return ot.Matrix(J)

myFunction = ot.PythonFunction(d, 1, func=g_py, func_sample=g_py_sample, gradient=g_grad)
```

### 6.3 Test de validation préalable OBLIGATOIRE

```python
# À exécuter une fois avant d'utiliser predict_derivatives en production
import numpy as np

x0 = np.atleast_2d(np.array([mu_fc, mu_fy]))  # point physique typique
eps = 1e-4

grad_sm = np.array([sm.predict_derivatives(x0, kx=i)[0, 0] for i in range(d)])

grad_fd = np.zeros(d)
y0 = sm.predict_values(x0)[0, 0]
for i in range(d):
    xp = x0.copy(); xp[0, i] += eps
    grad_fd[i] = (sm.predict_values(xp)[0, 0] - y0) / eps

err_rel = np.abs(grad_sm - grad_fd) / (np.abs(grad_fd) + 1e-12)
print("predict_derivatives :", grad_sm)
print("FD                  :", grad_fd)
print("erreur relative     :", err_rel)

# CRITÈRE : err_rel < 1e-3 partout → gradient fiable
#           sinon → NE PAS utiliser gradient analytique, rester en FD OpenTURNS
```

### 6.4 Points d'attention GEKPLS → OT

1. **Espace de travail du surrogat** : si GEKPLS est entraîné en espace **physique**, aucune transformation supplémentaire à faire (OT appelle `g_py` avec valeurs physiques, la transfo iso-probabiliste vers U est interne à FORM). Si entraîné en espace normalisé, **dénormaliser X avant `predict_values`**.

2. **Extrapolation hors domaine d'entraînement** : l'optimiseur FORM peut sortir des `xlimits` du DOE. GEKPLS extrapole silencieusement et devient erratique → contribue probablement à la non-convergence observée. Options :
   - Laisser FORM diverger et diagnostiquer si `X*` sort du DOE → élargir le DOE.
   - Clipping dans `g_py` (fausse le gradient au bord, dégrade la convergence).

3. **Cohérence des gradients** : `set_training_derivatives` + `predict_derivatives` travaillent en dérivées **physiques** `∂y/∂x_i`. Pas de mise à l'échelle supplémentaire côté OT.

4. **Contrainte `ndim ≥ 2`** : valable si `d` tombe à 1 dans un cas dégénéré.

---

## 7. Intégration complète — Squelette pour `AC_pure_flexion.py`

```python
import numpy as np
import openturns as ot
from smt.surrogate_models import GEKPLS, DesignSpace

# --- GEKPLS entraîné en amont sur [fc, fy] ---
# sm : instance GEKPLS trainée
# d  : nombre de variables (ex. 2 pour fc, fy)
# xlimits : array (d, 2) des bornes physiques du DOE

# --- Fonction d'état limite ---
def g_py(X):
    x = np.atleast_2d(np.asarray(X, dtype=float))
    return [float(sm.predict_values(x)[0, 0])]

def g_py_sample(X):
    x = np.asarray(X, dtype=float)
    return sm.predict_values(x)

myFunction = ot.PythonFunction(d, 1, func=g_py, func_sample=g_py_sample)

# --- Distribution jointe (JCSS) ---
dist_fc = ot.LogNormalMuSigma(mu_fc, sigma_fc, 0.0).getDistribution()
dist_fy = ot.Normal(mu_fy, sigma_fy)
marginals = [dist_fc, dist_fy]
copula    = ot.IndependentCopula(d)
distribution = ot.JointDistribution(marginals, copula)

# --- Événement de défaillance ---
vect   = ot.RandomVector(distribution)
output = ot.CompositeRandomVector(myFunction, vect)
event  = ot.ThresholdEvent(output, ot.Less(), 0.0)  # g(X) <= 0 = défaillance

# --- Solveur ---
solver = ot.AbdoRackwitz()
solver.setStartingPoint(distribution.getMean())
solver.setMaximumIterationNumber(100)          # au lieu de 10
solver.setMaximumAbsoluteError(1e-6)
solver.setMaximumResidualError(1e-6)
solver.setCheckStatus(False)                   # phase debug

# --- FORM ---
algo = ot.FORM(solver, event)
algo.run()
result = algo.getResult()

# --- Diagnostic ---
opt = result.getOptimizationResult()
print(f"Pf             = {result.getEventProbability():.3e}")
print(f"beta_HL        = {result.getHasoferReliabilityIndex():.3f}")
print(f"U*             = {result.getStandardSpaceDesignPoint()}")
print(f"X*             = {result.getPhysicalSpaceDesignPoint()}")
print(f"Iter           = {opt.getIterationNumber()}/{solver.getMaximumIterationNumber()}")
print(f"Calls          = {opt.getCallsNumber()}")
print(f"Abs err final  = {opt.getAbsoluteError()}")
print(f"Res err final  = {opt.getResidualError()}")
```

---

## 8. Points d'action identifiés

1. **[Prioritaire] Porter `n_max_FORM` de 10 à 100** dans `AC_pure_flexion.py`.
2. **Activer le diagnostic `getOptimizationResult()`** systématiquement après chaque run FORM.
3. **Ne pas passer `gradient=g_grad`** à `PythonFunction` tant que le test §6.3 n'a pas été fait sur la version SMT installée.
4. **Vérifier que le point de conception `X*` reste dans les `xlimits` du DOE GEKPLS** — sinon élargir le DOE ou changer de surrogat.
5. **Pour full GEK (solution retenue à long terme)** : coder manuellement le gradient analytique via la matrice de covariance augmentée — la structure du wrapper OpenTURNS ci-dessus reste valable.
6. **Considérer `MultiFORM`** si plusieurs modes de défaillance sont suspectés pour AC en flexion pure.

---

## 9. Préférences utilisateur à respecter

- Réponses **strictement neutres**, factuelles et directes.
- **Pas de compliments, pas d'encouragements, pas de gratifications**.
- Ne pas commencer par donner raison à l'utilisateur.
- Langue de travail : français, terminologie EN génie structural européen.
- Environnement : Windows via RDP, VS Code + Claude Code + Git Bash + TortoiseGit, codebase `C:\_workingDir\_SF\`.
