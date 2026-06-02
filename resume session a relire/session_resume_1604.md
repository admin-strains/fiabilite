# Résumé de session — AC_pure_flexion.py
## Date : 2026-04-16

---

## 1. Contexte général

Workflow de fiabilité FORM sur une poutre BA en flexion pure (STRAINS, analyse limite cinématique).
- **Fichier principal** : `C:\_workingDir\_SF\test flexion\AC_pure_flexion.py`
- **Modèle STRAINS** : `C:\workspace\storage\semia\test_pure_flexion.ds\`
- **Lanceur** : `C:\_workingDir\_SF\test flexion\launcher.py` (fait `exec(open('AC_pure_flexion.py').read(), {'__name__': '__main__'})`)
- **Variables aléatoires** : fc (béton C35, loi JCSS) et fy (acier, loi JCSS, phi dépendant)
- **Fonction de performance** : g = α⁺ − 1 où α⁺ est le multiplicateur de charge limite
- **Distributions** : `from config.jcss_fy import loi_fy` et `from config.jcss_fc import loi_fc`
  - fc moyen ≈ 31.67 MPa (C35)
  - fy moyen ≈ 613.39 MPa (phi=16mm ou 32mm, fy_nominal=500)

---

## 2. Architecture du code (état actuel)

### Flags de contrôle (lignes ~319-325)
```python
do_GEK = False
do_GP = False
do_pce = False
n_start = 1
n0 = max(25, n_start)
n_max_FORM = 10   # ajouté cette session
```

### Étapes principales
- **Étape 0** : distributions + transformation isoprobabiliste T, T_inv, dist_U
- **Étape 1** : DOE LHS dans espace U (n0=25 points) — run_one_SOL seulement si `do_GP=True`
- **Étape 3.1** : PCE (désactivé, `do_pce=False`)
- **Étape 3.2** : Métamodèle (GEK ou KRG si `do_GP=True`, sinon `HFModel`)
- **Étape 4** : FORM (AbdoRackwitz ou Cobyla selon modèle)
- **Étape 5** : Visu (`do_visu=False`) + test linéarisation FOSM (`do_linear_test=True`)

---

## 3. Fonctions clés

### `patch_params(path, **params)` (ligne 50)
Réécrit dsCad.txt avec regex `re.sub` pour chaque paramètre.

### `run_one_SOL(modelname, SOL, params_names, sensitivity, ...)` (ligne 61)
Lance STRAINS pour une liste de points SOL. Remplit `SOL[i]['g']`, `SOL[i]['dg_fc']`, `SOL[i]['dg_fy']`.
Utilisé seulement dans `if do_GP:`.

### `run_HF(modelname, u, params_names, T_inv, sensitivity=True, ...)` (ligne 161)
Lance 1 calcul STRAINS pour un point u (espace U).
- Convertit u → x via T_inv
- Patch dsCad.txt
- Lance mesh + STRAINS
- Lit `.dsmetares`
- Retourne `(g_HF, grad_HF_U)` où grad_HF_U est un `ot.Point` (gradient en espace U)
- **Clés sensitivity_regions** : `{"param": "COMPRESSIVE_STRENGTH", "solids": ["Block1"]}` et `{"param": "YIELD_STRENGTH", "rebars": ["HA1","HA2","HA3","HA4"]}`
- **Garde-fou ajouté cette session** (ligne 264) : lève `ValueError` si `sensitivity=True` mais `grad_HF_U` contient `None`

### `HFModel(ot.OpenTURNSPythonFunction)` (ligne 477)
Classe OT pour FORM HF direct. Cache le dernier appel pour éviter double run STRAINS :
- `_last_u`, `_last_g`, `_last_grad`
- `_run_if_needed(u)` : compare `list(u) != list(self._last_u)`
- `_exec(u)` → retourne `[self._last_g]`
- `_gradient(u)` → retourne `ot.Matrix([[v] for v in self._last_grad])`

---

## 4. Bloc FORM HF (lignes 526-558)

```python
hf_instance = HFModel()
myFunction = ot.Function(hf_instance)
vect   = ot.RandomVector(dist_U)
output = ot.CompositeRandomVector(myFunction, vect)
event  = ot.ThresholdEvent(output, ot.Less(), 0.0)
solver = ot.AbdoRackwitz()
solver.setMaximumIterationNumber(n_max_FORM)   # ajouté cette session
solver.setStartingPoint([0.0] * n_var)
algo = ot.FORM(solver, event)
algo.run()
result = algo.getResult()
```

**Note importante** : le FORM est construit avec `dist_U` (distribution standard).
Donc `result.getPhysicalSpaceDesignPoint()` retourne u* (espace U standard), PAS x*.
Et `X_res = T_inv(U_res)` convertit correctement u* → x* (ligne 563).

### Multi-start (lignes 539-558, actif seulement si n_start > 1)
```python
if n_start > 1:
    norms = np.array([np.linalg.norm(np.array(U_doe[i])) for i in range(n0)])
    sorted_idx = np.argsort(norms)[::-1]
    U_doe_sorted = ot.Sample([U_doe[int(i)] for i in sorted_idx])
    result_modes = [result]
    for n_FORM in range(n_start):
        solver.setStartingPoint(U_doe_sorted[n_FORM])
        algo = ot.FORM(solver, event)
        algo.run()
        u_new  = algo.getResult().getPhysicalSpaceDesignPoint()
        u_prev = result_modes[-1].getPhysicalSpaceDesignPoint()
        if (u_new - u_prev).norm() > 1e-3:
            result_modes.append(algo.getResult())
            break
```

---

## 5. Test linéarisation FOSM (lignes 623-633)

```python
if do_linear_test:
    u_star = U_res
    u0 = ot.Point([0.0] * n_var)
    g0, grad_U_0 = run_HF(modelname, u0, params_names, T_inv, sensitivity=True)
    norm_sq = grad_U_0.norm() ** 2
    u_star_FOSM = grad_U_0 * (-g0 / norm_sq)
    relative_error = (u_star_FOSM - u_star).norm() / u_star.norm()
    print(f"  u* FORM = {u_star}")
    print(f"  u* FOSM = {u_star_FOSM}")
    print(f"  Erreur relative : {relative_error:.4f}")
```

Principe : linéarisation de g depuis u=0 (nominal). u*_FOSM = −g₀ · ∇g_U(0) / ‖∇g_U(0)‖²
À comparer avec u* FORM pour valider la linéarité de g.

---

## 6. Problèmes rencontrés et solutions

### PB1 : `dg/dfy = 0` pour tous les points DOE → FORM ne converge pas
**Cause** : mauvaise clé `"solids"` au lieu de `"rebars"` pour les armatures dans `sensitivity_regions`
**Solution retenue** : corriger ligne 135 et 237 : `{"param": "YIELD_STRENGTH", "rebars": ["HA1","HA2","HA3","HA4"]}`

### PB2 : `getHasoferLindbergReliabilityIndex` AttributeError
**Solution** : renommer en `getHasoferReliabilityIndex`

### PB3 : `grad_HF_U` NameError (était `grad_U`)
**Solution** : renommer correctement la variable

### PB4 : DOE (run_one_SOL) se lance même quand `do_GP=False`
**Cause** : lignes 424-430 (`xt`, `y_hf`, `all_grad`) hors du bloc `if do_GP:`
**Solution retenue** (cette session) : déplacer ces lignes à l'intérieur de `if do_GP:` (fait)

### PB5 : docstring mal indentée dans `if n_start > 1:` → SyntaxError
**Solution** : indenter le `"""..."""` de 4 espaces supplémentaires (fait)

### PB6 : AbdoRackwitz fait 1001 appels HF sans converger
**Contexte** : avec phi=16mm + F=0.6 MN, comportement anormal
**Causes possibles** :
  1. `grad_HF_U` contient None si STRAINS ne calcule pas les sensibilités → gradient corrompu silencieusement
  2. Configuration incompatible (g(0)<0 + gradient mal orienté)
**Solutions apportées** :
  - `solver.setMaximumIterationNumber(n_max_FORM)` avec `n_max_FORM=10` (fait)
  - `raise ValueError` si grad_HF_U contient None (fait, ligne 264)
**Statut** : PB non résolu — run interrompu, cause exacte inconnue

### PB7 : section 0.3×0.3 → encore plus lent (406 sessions STRAINS)
**Cause** : section agrandie temporairement, F=1.0 MN → comportement encore pire
**Solution** : revenu à 0.2×0.2, phi=16mm, F=0.6 MN

### PB8 : Bash tool — `python launcher.py` rejeté systématiquement
**Contexte** : fonctionne normalement depuis le début de session, puis bloqué
**Investigations** : 
  - `echo "test"` et `python --version` fonctionnent
  - `settings.json` global vide `{}`, aucun hook projet
  - L'utilisateur dit approuver mais rien ne se passe
**Statut** : NON RÉSOLU — raison à investiguer dans nouvelle session

---

## 7. Configuration actuelle des fichiers modèle

### `dsCad.txt` (dernière valeur patched par STRAINS, à ignorer pour fc/fy)
```
phi = 16.0   ← changé cette session (était 32.0)
b   = 0.2    ← revenu à 0.2 (avait été mis à 0.3 temporairement)
h   = 0.2
L   = 5.0
ft  = 3.5
E   = 35.0
```
(fy et fc sont des valeurs patchées par le dernier run STRAINS, pas les nominaux)

### `dsLoad.txt`
```
Z='-0.6'    ← F = 0.6 MN (changé plusieurs fois cette session)
```

---

## 8. Résultats obtenus

### Run 1 — phi=32mm, F=0.235 MN (session précédente)
| β | Pf | fc* | fy* | Importance fc | Importance fy |
|---|---|---|---|---|---|
| 16.22 | 1.89e-59 | 30.01 MPa | 148.11 MPa | 0.16% | 99.84% |

### Run 2 — phi=32mm, F=0.47 MN
| β | Pf | fc* | fy* |
|---|---|---|---|
| 9.477 | 1.31e-21 | 30.86 MPa | 351.20 MPa |
- FOSM erreur relative : 0.09% → g quasi-linéaire ✓

### Run 3 — phi=32mm, F=0.6 MN
| β | Pf | fc* | fy* |
|---|---|---|---|
| 5.744 | 4.62e-09 | 31.36 MPa | 463.64 MPa |
- FOSM erreur relative : 0.14% → g quasi-linéaire ✓
- **→ Run de référence pour beta≈6 avec phi=32mm**

---

## 9. À faire dans la nouvelle session

1. **Résoudre le pb Bash** : investiguer pourquoi `python launcher.py` est bloqué
2. **Relancer avec phi=16mm + F=0.6 MN** et vérifier si le ValueError de grad_HF_U est levé
3. Si ValueError : STRAINS ne calcule pas les sensibilités pour phi=16mm dans certains régimes → à investiguer
4. Si pas de ValueError mais toujours 1000+ itérations : autre cause à chercher
5. Calibrer F pour beta≈6 avec phi=16mm (estimation : F << 0.235 MN car phi=16mm a capacité bien plus faible)

---

## 10. Syntaxe utile rappel

```python
# Norme d'un ot.Point
p.norm()

# Produit scalaire ot.Point
p * scalar

# Design point FORM (avec dist_U comme distribution → "physical" = U space)
result.getPhysicalSpaceDesignPoint()   # retourne u* (ot.Point)

# Limiter AbdoRackwitz
solver.setMaximumIterationNumber(10)
solver.setMaximumCallsNumber(50)

# Gradient de T_inv (jacobien)
J = T_inv.gradient(u_point)   # retourne ot.Matrix
```
