# Résumé de session — FORM HF Flexion Pure BA
**Date :** 17 avril 2026  
**Objectif :** Lancer FORM (AbdoRackwitz) avec appels directs au code STRAINS (HF) pour une poutre BA en flexion pure (phi=16mm), calibrer F pour atteindre β≈6, et s'assurer que le gradient analytique de STRAINS est bien utilisé.

---

## 1. Architecture du problème

- **Fonction de performance :** g = α⁺ − 1, où α⁺ est le multiplicateur de charge limite calculé par STRAINS
- **Variables aléatoires :** fc (béton, lognormale JCSS), fy (acier phi=16mm, lognormale JCSS)
- **Espace standard U :** transformation isoprobabiliste T : (fc, fy) → (u_fc, u_fy)
- **FORM :** algorithme AbdoRackwitz d'OpenTURNS, minimise ‖u‖ sous contrainte g(u)=0
- **β = ‖u*‖**, Pf = Φ(−β)

---

## 2. Problème 1 — Gradient FD au lieu du gradient analytique STRAINS

### Symptôme
FORM faisait 5 appels STRAINS par itération (schéma différences finies) au lieu d'1 seul (gradient analytique).

### Cause
Le wrapper OT initial utilisait `ot.PythonFunction(n, 1, func)` sans gradient → OT calcule automatiquement le gradient par FD (5 appels pour n=2 variables).

### Solutions tentées et abandonnées
- Sous-classer `ot.OpenTURNSPythonFunction` et overrider `_gradient()` → **impossible** : le vtable C++ ignore les overrides Python
- Utiliser `ot.MemoizeFunction` + `setGradient()` → `setGradient` non exposé sur OpenTURNSPythonFunction

### Solution retenue
```python
# Paramètre gradient= de ot.PythonFunction
myFunction = ot.PythonFunction(n_var, 1, func, gradient=grad_func)
```

**Règle de forme du gradient (critique) :**
- OT attend shape `[1][n]` : `[[dg/du1, dg/du2]]`
- Erreur silencieuse si on donne `[[dg/du1], [dg/du2]]` (shape n×1) → OT fallback FD sans warning

```python
def grad_func(u):
    hf_cache.run_if_needed(u)
    return [[v for v in hf_cache._last_grad]]  # ← shape [1][n] CORRECT
    # return [[v] for v in hf_cache._last_grad]  # ← shape [n][1] FAUX, FD silencieux
```

---

## 3. Problème 2 — Double appel STRAINS (func + grad_func au même point)

### Symptôme
AbdoRackwitz appelle `func(u)` puis `grad_func(u)` au même point → 2 runs STRAINS inutiles.

### Solution retenue — Pattern HFCache
```python
class HFCache:
    def __init__(self):
        self._last_u = None
        self._last_g = None
        self._last_grad = None

    def run_if_needed(self, u):
        u_list = list(u)
        if self._last_u is None or u_list != self._last_u:
            self._last_g, self._last_grad = run_HF(modelname, u, params_names, T_inv)
            self._last_u = u_list

hf_cache = HFCache()

def func(u):
    hf_cache.run_if_needed(u)
    return [hf_cache._last_g]

def grad_func(u):
    hf_cache.run_if_needed(u)
    return [[v for v in hf_cache._last_grad]]
```

---

## 4. Problème 3 — FORM RuntimeError : design point not on limit state

### Symptôme
```
RuntimeError: Obtained design point is not on the limit state:
its image = 1.3e-05 > limit state tolerance 1e-05
```
AbdoRackwitz trouve u* avec g(u*)≈1.3e-05 (quasi-zéro), mais FORM rejette le résultat car g > seuil 1e-05 (codé en dur en C++, correspond à `solver.getMaximumConstraintError()`).

### Compréhension du mécanisme
- FORM lit `solver.getMaximumConstraintError()` (défaut=1e-05) comme seuil de validation finale
- Cette vérification est dans `_analytical.FORM_run()` — code C++ compilé, non modifiable
- Le check est **indépendant** de la convergence du solver

### Solutions tentées et abandonnées

| Tentative | Résultat | Pourquoi ça échoue |
|---|---|---|
| Augmenter `n_max_FORM` (23→25→35) | g=1.3e-05, RuntimeError | L'algo converge très lentement, g n'atteint jamais <1e-05 |
| `setMaximumResidualError(2e-5)` | g=1.42e-05, RuntimeError | Stoppe l'algo plus tôt (pas le bon critère) |
| `setMaximumConstraintError(2e-5)` | g=5.18e-05, RuntimeError | Relâche le critère d'arrêt du solver mais un autre critère (ResidualError) se déclenche avant, donnant un g pire |
| `try/except` + `algo.getResult()` | Résultat vide, crash C++ | FORM ne stocke PAS son résultat avant de lever l'exception |
| `try/except` + `solver.getResult()` | Résultat vide, crash C++ | FORM utilise une copie C++ interne du solver, pas l'objet Python |

### Solution retenue — `solver.setCheckStatus(False)`

**Découverte :** AbdoRackwitz (hérité de `OptimizationAlgorithmImplementation`) expose :
```python
solver.setCheckStatus(False)
# Doc: "If set to False, run() will not throw an exception if the algorithm
#       does not fully converge and will allow one to still find a feasible candidate."
```

Lorsque `checkStatus=False`, FORM ne lève pas le RuntimeError final, et `algo.getResult()` est pleinement accessible avec β, Pf, importances, u* complets.

**Code final du bloc FORM :**
```python
solver = ot.AbdoRackwitz()
solver.setMaximumIterationNumber(n_max_FORM)
solver.setCheckStatus(False)
solver.setStartingPoint([0.0] * n_var)

algo = ot.FORM(solver, event)
algo.run()
result = algo.getResult()
result_modes = [result]
```

---

## 5. Ce qu'on a appris sur OpenTURNS

### AbdoRackwitz — critères d'arrêt (tous indépendants, OR)

| Critère | Formule | Défaut | Rôle |
|---|---|---|---|
| `MaximumConstraintError` | `‖g(xₙ)‖` | 1e-05 | Valeur de g au point courant — aussi utilisé par FORM comme seuil de validation |
| `MaximumAbsoluteError` | `‖xₙ₊₁−xₙ‖` | 1e-07 | Déplacement entre itérés |
| `MaximumRelativeError` | `‖xₙ₊₁−xₙ‖/‖xₙ₊₁‖` | 1e-07 | Déplacement relatif |
| `MaximumResidualError` | `‖f(xₙ₊₁)−f(xₙ)‖/‖f(xₙ₊₁)‖` | 1e-07 | Variation de la fonction objectif |
| `MaximumIterationNumber` | n_iter ≥ n_max | 100 | Garde-fou |

Le solver s'arrête dès que **l'un quelconque** de ces critères est satisfait.

### FORM — méthodes clés
```python
result = algo.getResult()                        # FORMResult
beta   = result.getHasoferReliabilityIndex()     # β = ‖u*‖
Pf     = result.getEventProbability()            # Φ(−β)
U_res  = result.getPhysicalSpaceDesignPoint()    # u* (espace U dans notre cas)
importance = result.getImportanceFactors()       # (αᵢ)² = (u*ᵢ/β)²
opt_result = result.getOptimizationResult()      # OptimizationResult du solver
n_iter = opt_result.getIterationNumber()         # nombre d'itérations effectuées
```

### Comportement de FORM après RuntimeError
- `algo.getResult()` → résultat **vide** (FORM ne stocke pas avant de lever)
- `solver.getResult()` → résultat **vide** (FORM utilise une copie C++ interne du solver, pas l'objet Python)
- Seul `setCheckStatus(False)` permet de récupérer le résultat complet

### Sources documentaires consultées
- https://openturns.github.io/openturns/1.26/user_manual/_generated/openturns.AbdoRackwitz.html
- https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.FORM.html
- https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.Analytical.html
- Source Python : `C:\python3\lib\site-packages\openturns\optim.py` (lignes 1771–1970)
- Source Python : `C:\python3\lib\site-packages\openturns\analytical.py`

### Sorties Python noyées dans STRAINS
Quand on redirige vers un fichier (`> out.txt`), les prints Python sont **bufférisés** et peuvent être enfouis dans 80 000+ lignes de sortie STRAINS. Utiliser `grep -n` sur le fichier de sortie pour retrouver les résultats.

---

## 6. Modifications du code AC_pure_flexion.py (hors résolution de problèmes)

### 6a. Compteur d'appels au gradient
```python
grad_call_count = [0]

def grad_func(u):
    grad_call_count[0] += 1
    print(f"[GRAD] appel #{grad_call_count[0]} en u={list(u)}", flush=True)
    hf_cache.run_if_needed(u)
    return [[v for v in hf_cache._last_grad]]
```
Permet de confirmer que le gradient analytique est bien appelé et de suivre la convergence (u évolue vers u*).

### 6b. Affichage du nombre d'itérations FORM
```python
opt_result = result.getOptimizationResult()
n_iter = opt_result.getIterationNumber()
print(f"Nombre d'itérations FORM : {n_iter}")
```

### 6c. n_max_FORM paramétrable
```python
n_max_FORM = 40  # augmenté progressivement : 6 → 12 → 23 → 25 → 35 → 40
```

---

## 7. Résultats FORM obtenus (phi=16mm)

| F (MN) | β | Pf | n_iter FORM | u* |
|---|---|---|---|---|
| 0.235 | 0.952 | 1.71e-01 | 1 | [-0.159, -0.938] |
| 0.230 | 1.518 | 6.44e-02 | 1 | [-0.254, -1.497] |
| 0.225 | 2.084 | 1.86e-02 | 1 | [-0.346, -2.056] |
| 0.220 | 2.654 | 3.98e-03 | 15 | [-0.347, -2.631] |
| 0.215 | 3.222 | 6.36e-04 | 17 | [-0.440, -3.192] |
| 0.210 | 3.784 | 7.73e-05 | 21 | [-0.526, -3.747] |
| 0.195 | 5.484 | 2.07e-08 | 18 | [-0.829, -5.422] |

Objectif β≈6 → prochain run F=0.190 MN.

---

## 8. Fichiers clés

| Fichier | Rôle |
|---|---|
| `AC_pure_flexion.py` | Script principal FORM HF |
| `launcher.py` | Lance AC_pure_flexion.py avec les DLL STRAINS |
| `dsLoad.txt` | Force appliquée (à modifier entre runs) |
| `resultats_HF_run2.md` | Tableau complet des résultats |
| `plan_1e5_a_reprendre.md` | Plan des options A/B/C pour le pb tolérance 1e-05 (résolu par C=setCheckStatus) |
