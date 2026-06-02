# Plan — Correction du bloc GEK pour FORM OpenTURNS

## Contexte

Le bloc `if do_GEK:` dans `AC_pure_flexion.py` construit un métamodèle GEKPLS (SMT) entraîné
avec gradients HF, puis l'expose à OpenTURNS via `ot.PythonFunction`. Objectif : passer à
`do_GEK=True` et lancer FORM avec le métamodèle GEK.

Analyse du code source SMT installé (`C:\python3\lib\site-packages\smt\`) et comparaison
avec le bloc HF (session 17/04) révèle 3 bugs bloquants et 2 manques structurels.

---

## Fichier à modifier

`C:\_workingDir\_SF\test flexion\AC_pure_flexion.py`

- Bloc GEKPLS training : lignes ~439–468 (`if do_GEK:`)
- Bloc FORM GEK : lignes ~515–525 (`if do_GP: if do_GEK:`)
- Bloc affichage post-FORM : lignes ~591–605 (`if do_GP: if do_GEK==False:`)

---

## Découverte clé : predict_derivatives EXISTE sur GEKPLS

Source : `krg_based.py` ligne 227 — KrgBased (grand-parent de GEKPLS) déclare
`supports["derivatives"] = True`. La méthode `_predict_derivatives(x, kx)` est implémentée
et applique correctement la règle de chaîne PLS via `componentwise_distance_PLS(return_derivative=True)`
qui retourne shape `(n_eval*nt, nx)` dans l'espace des variables originales (pas dans l'espace PLS réduit).

```
sm.predict_derivatives(u_np, kx)  →  ndarray(n_eval, 1)  — dérivée de g w.r.t. variable kx
```

---

## Bug 1 — `corr='matern52'` invalide pour GEKPLS [BLOQUANT]

GEKPLS redéclare `corr` avec `values=("abs_exp", "squar_exp")` seulement
(source : `gekpls.py` lignes 20–28). `corr='matern52'` lève ValueError à la construction.

```python
# AVANT
sm = GEKPLS(corr='matern52', ...)   # ValueError à la construction

# APRÈS
sm = GEKPLS(corr='squar_exp', ...)  # seule valeur valide proche de Matérn
```

**Note :** `squar_exp` (gaussien) ≠ Matérn 5/2 — à noter comme différence méthodologique
entre GEK et KRG lors de la comparaison des résultats.

---

## Bug 2 — Gradient non passé à OT → FD silencieux [même problème que HF session 17/04]

```python
# AVANT
myFunction = ot.PythonFunction(n_var, 1, g_GEK)   # FD : 3 appels/iter, gaspille l'avantage GEK

# APRÈS : ajouter grad_GEK juste après la définition de g_GEK (~ligne 468)
def grad_GEK(u):
    u_np = np.array(u).reshape(1, -1)
    grad = [float(sm.predict_derivatives(u_np, kx)[0, 0]) for kx in range(n_var)]
    return [grad]   # shape [1][n_var] — règle de forme identique au bloc HF

myFunction = ot.PythonFunction(n_var, 1, g_GEK, gradient=grad_GEK)
```

**Note :** Pas de HFCache nécessaire — `predict_derivatives` est analytique (pas d'appel STRAINS).

---

## Bug 3 — Solver Cobyla + paramètres manquants

```python
# AVANT
solver = ot.Cobyla()
solver.setStartingPoint([0.0] * n_var)
# manque : setMaximumIterationNumber, setCheckStatus(False)

# APRÈS
solver = ot.AbdoRackwitz()
solver.setMaximumIterationNumber(n_max_FORM)
solver.setCheckStatus(False)
solver.setStartingPoint([0.0] * n_var)
```

---

## Manque 4 — Affichage post-FORM absent pour do_GEK=True

```python
# AVANT : bloc ~ligne 591
if do_GP:
    if do_GEK==False:      # rien pour do_GEK=True → gradient et importance non affichés
        grad_star = metamodel_KRG.gradient(U_res)
        ...

# APRÈS
if do_GP:
    if do_GEK:
        grad_star_list = [float(sm.predict_derivatives(
                              np.array(U_res).reshape(1,-1), kx)[0, 0])
                          for kx in range(n_var)]
        for i in range(n_var):
            print(f"  dg/du_{params_names[i]} en u* = {grad_star_list[i]:.6f}")
        importance = result.getImportanceFactors()
        for i in range(n_var):
            print(f"  Importance factor {params_names[i]}: {importance[i]:.4f}")
    else:   # KRG
        grad_star = metamodel_KRG.gradient(U_res)
        for i in range(n_var):
            print(f"  dg/du_{params_names[i]} en u* = {grad_star[i, 0]:.6f}")
        importance = result.getImportanceFactors()
        for i in range(n_var):
            print(f"  Importance factor {params_names[i]}: {importance[i]:.4f}")
```

---

## Ordre d'implémentation

1. Corriger `corr='matern52'` → `'squar_exp'` dans le bloc training GEK
2. Ajouter `grad_GEK` juste après `g_GEK`
3. Passer `gradient=grad_GEK` à `ot.PythonFunction`
4. Remplacer `Cobyla` par `AbdoRackwitz` + ajouter `setMaximumIterationNumber` et `setCheckStatus(False)`
5. Corriger le bloc affichage post-FORM

## Vérification

1. Lancer avec `do_GEK=True`, `do_GP=True`, F=0.235 (β≈0.95, cas simple)
2. Pas de ValueError à l'initialisation GEKPLS → Bug 1 corrigé
3. Gradient analytique utilisé (pas de FD) → vérifier n_iter ~ même ordre que KRG
4. Comparer β_GEK vs β_HF et β_KRG
5. Si RuntimeError → ajouter `setMaximumConstraintError(1e-2)` comme pour KRG
