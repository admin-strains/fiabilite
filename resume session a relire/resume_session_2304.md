# Résumé de session — Audit complet PCE-KRG + corrections bugs
**Date :** 22-23 avril 2026  
**Objectif :** Suite de la session 2204 après autocompactage du contexte. Relecture des résumés précédents, poursuite des corrections de bugs dans AC_pure_flexion.py, audit complet des blocs PCE-KRG et KRG pur, audit des fonctions helper associées.

---

## 1. Sujets théoriques abordés

### 1a. Réponse à la question en suspens : `metamodel_KRG + metamodel_PCE`

**Question :** Est-ce que `metamodel_KRG + metamodel_PCE` est valide en OT ?

**Réponse :** Oui. Les deux sont des `ot.Function` (duck-typed). OT surcharge l'opérateur `+` pour créer une nouvelle Function qui somme les sorties. Les deux retournent un scalaire → dimensions compatibles.

**Attention `+=`** : dans une session précédente, `metamodel_KRG += metamodel_pce` causait un segfault. Utiliser `metamodel = metamodel_KRG + metamodel_PCE` (nouvelle variable) est plus sûr.

---

### 1b. Aliasing numpy — `yt = y_hf` ne crée PAS une copie

**Principe :** En numpy, `b = a` crée un **alias** (référence) sur le même tableau mémoire. `b -= x` est équivalent à `np.subtract(b, x, out=b)` → modifie les données en place → `a` est aussi modifié.

**Démonstration :**
```python
a = np.array([[1.], [2.]])
b = a
b -= np.array([[1.], [1.]])
print(a)  # [[0.], [1.]]  ← a est modifié !
```

**Règle :** Pour une copie indépendante : `b = a.copy()`. Cette règle s'applique à tout numpy array.

---

### 1c. `.reshape(1,-1)` nécessite un numpy array

`ot.Point` et les listes Python n'ont pas de méthode `.reshape()`. Avant d'appeler `.reshape(1,-1)`, il faut convertir : `np.array(obj).reshape(1, -1)`.

`run_HF` retourne `(g_HF, grad_HF_U, grad_HF_X)` où :
- `grad_HF_U` = OT Point (résultat de `J_Tinv_T * ot.Point(grad_HF_X)`)
- `grad_HF_X` = liste Python (`[None]*n_var` remplie de floats)

Ni l'un ni l'autre ne sont des numpy arrays → `.reshape()` échoue directement dessus.

---

### 1d. Argument par défaut Python évalué à la définition

```python
def f(x=[0.0] * n_var):
```
`n_var` est évalué **quand Python exécute la ligne `def`**, pas à l'appel. Si `n_var` n'est pas défini dans le scope global à ce moment → **NameError à la définition**, le script s'arrête avant toute exécution.

Conséquence : `def FORM_KRG(..., start_point=[0.0]*n_var, ...)` → crash si `n_var` n'est pas une globale.

---

### 1e. Sensibilités PCE via jacobien de T

Le PCE est construit en espace U → `metamodel_PCE.gradient(u)` donne `∇_U g`.

Pour obtenir `∇_X g` (sensibilités en espace physique) via T (T : X→U) :
```
∇_X g = J_T · ∇_U g
```
où `J_T = T.gradient(x)` est la matrice OT (n_var, n_var) avec `J_T[i,j] = dU_j/dX_i` (convention OT = transposée de la convention mathématique standard). Pas de transposée ni d'inversion nécessaire avec la convention OT.

**En code :**
```python
x_i = T_inv(ot.Sample([U_doe[i]]))[0]    # u → x
J_T_i = T.gradient(x_i)                  # OT Matrix (n_var, n_var)
grad_u_i = ot.Point(all_grad_PCE[i, :])  # OT Point (n_var,)
grad_x_i = J_T_i * grad_u_i             # OT Point (n_var,)  ← pas de transposée
all_sensib_PCE[i, :] = np.array(grad_x_i)
```

`T_inv` (U→X) est nécessaire pour convertir le point u en x avant d'évaluer T.gradient.

---

### 1f. Mise à jour de `xt` après ajout d'un point warm start

`xt` est numpy `(n0, n_var)`. `U_warm` est OT Point → `np.array(U_warm)` donne `(n_var,)` (1D).

Option simple (recommandée pour faible coût) :
```python
U_doe.add(U_warm)
xt = np.array(U_doe)   # recompute depuis OT Sample mis à jour
```

Option efficace (stack) :
```python
xt = np.vstack([xt, np.array(U_warm).reshape(1, -1)])
```
`.reshape(1,-1)` nécessaire car `np.array(OT_Point)` donne `(n_var,)` et `vstack` attend des tableaux 2D.

---

### 1g. `ot.Point(all_grad_PCE[i, :])` — pourquoi ça marche

`all_grad_PCE[i, :]` est un slice numpy → tableau 1D de shape `(n_var,)`. `ot.Point` accepte n'importe quelle séquence 1D → conversion directe. OK.

À distinguer de `ot.Point(all_grad_PCE)` avec `all_grad_PCE` de shape `(n0, n_var)` → échec ou mauvais résultat (tableau 2D).

---

## 2. Problèmes identifiés et solutions

### Problème 1 — `init_GP` ligne 285 : NameError au return
**Symptôme :** `return xt, y_hf, all_grad_hf, all_sensib_hf` — variables locales s'appellent `all_grad` et `all_sensib`.  
**Solution retenue :** Renommer les variables locales en `all_grad_hf` et `all_sensib_hf` dans le corps de la fonction (lignes 280-284).  
**Code corrigé :**
```python
all_grad_hf = np.zeros((n0, n_var))
all_sensib_hf = np.zeros((n0, n_var))
for i in range(n0):
    for j in range(n_var):
        all_grad_hf[i][j] = SOL_U[i][f'dg_u{j+1}']
        all_sensib_hf[i][j] = SOL_U[i][f'dg_{params_names[j]}']
return xt, y_hf, all_grad_hf, all_sensib_hf
```

---

### Problème 2 — `dist_jointe` appelée avec 3 args (TypeError)
**Symptôme :** `dist_jointe(modelname, params_names, n0)` dans `build_metamodel_PCE` (ligne 292) et `fill_PCE` (ligne 318). Signature réelle : `dist_jointe(modelname, params_names)` (2 params).  
**Solution retenue :** Supprimer `n0` dans les deux appels.

---

### Problème 3 — `FORM_KRG` ligne 386 : `metamodel_KRG` au lieu de `metamodel`
**Symptôme :** `output = ot.CompositeRandomVector(metamodel_KRG, vect)` — le paramètre s'appelle `metamodel`. Pour KRG pur, `metamodel_KRG` est indéfini → NameError.  
**Solution retenue :** `output = ot.CompositeRandomVector(metamodel, vect)`.

---

### Problème 4 — `U_doe` indéfini dans les warm start (NameError)
**Symptôme :** `U_doe.add(U_warm)` dans les blocs warm start. `init_GP` retourne `xt` (numpy) pas `U_doe` (OT Sample). `U_doe` n'existe pas dans le scope appelant.  
**Solution retenue :** Ajouter `U_doe = ot.Sample(xt)` juste avant `U_doe.add(U_warm)`.  
**Corrigé pour PCE-KRG** (ligne 452). **Non encore corrigé pour KRG pur** (ligne 483 — bug restant).

---

### Problème 5 — `.reshape(1,-1)` sur OT Point / liste (AttributeError)
**Symptôme :** Dans les warm start, `all_grad_hf_warm.reshape(1,-1)` (OT Point) et `all_sensib_hf_warm.reshape(1,-1)` (liste Python) → AttributeError.  
**Solution retenue :** Envelopper avec `np.array()` :
```python
np.array(all_grad_hf_warm).reshape(1, -1)
np.array(all_sensib_hf_warm).reshape(1, -1)
```
**Corrigé pour PCE-KRG** (lignes 459-460). **Non encore corrigé pour KRG pur** (lignes 489-490 — bug restant).

---

### Problème 6 — `n_var` non défini globalement (NameError dans plusieurs fonctions)
**Symptôme :** `build_metamodel_KRG`, `FORM_KRG`, `init_GP` utilisent `n_var` comme variable libre. Pas de `n_var = len(params_names)` dans le scope global entre lignes 186-521.  
**Solution retenue :** Ajouter `n_var` localement dans chaque fonction :
- `init_GP` : `n_var = len(params_names)` en début de fonction
- `build_metamodel_KRG` : `n_var = xt.shape[1]` en début de fonction
- `FORM_KRG` : `n_var = len(params_names)` en début de fonction

---

### Problème 7 — `fill_inputGP` aliasing numpy (corruption silencieuse)
**Symptôme :** `yt = y_hf` crée un alias. `yt -= y_PCE` (in-place) modifie `y_hf` en place. Conséquence : dans le warm start PCE-KRG, `y_hf` contenait des résiduels (y_hf - y_PCE) au lieu des valeurs HF → `build_metamodel_PCE(xt, y_hf)` construisait un PCE depuis des résiduels. Même problème pour `all_grad_hf` et `all_sensib_hf`.  
**Solution retenue :** Utiliser `.copy()` :
```python
yt = y_hf.copy()
all_grad = all_grad_hf.copy()
all_sensib = all_sensib_hf.copy()
```

---

### Problème 8 — `build_metamodel_GEK` : `metamodel_pce` → `metamodel_PCE`
**Symptôme :** Lignes 368-369 : `metamodel_pce(...)` et `metamodel_pce.gradient(...)` → NameError si `do_pce=True`.  
**Solution retenue :** Renommer en `metamodel_PCE`. Non encore corrigé (GEK non encore implémenté).

---

### Problème 9 — `FORM_KRG` default arg `start_point=[0.0]*n_var` (NameError à la définition)
**Symptôme :** `def FORM_KRG(..., start_point=[0.0]*n_var, ...)` — `n_var` évalué à la définition de la fonction, or `n_var` n'est pas défini globalement → **NameError quand Python exécute la ligne `def`**, avant même d'atteindre les blocs PCE-KRG ou KRG pur.  
**Solution proposée :** `start_point=None` avec `if start_point is not None else [0.0]*n_var` dans le corps.  
**Solution retenue par l'utilisatrice :** `start_point` devient un **paramètre obligatoire** (sans valeur par défaut). Le point de départ est défini explicitement avant chaque appel dans les blocs appelants :
```python
# Signature
def FORM_KRG(modelname, params_names, metamodel, start_point, n_max_FORM=n_max_FORM, tol_FORM=tol_FORM):
    n_var = len(params_names)
    ...
    solver.setStartingPoint(start_point)

# Dans les blocs PCE-KRG et KRG pur, avant l'appel initial :
start_point = [0.0] * len(params_names)
result = FORM_KRG(modelname, params_names, metamodel, start_point)

# Avant l'appel warm start :
start_point = U_warm
result = FORM_KRG(modelname, params_names, metamodel, start_point)
```

---

### Problème 10 — `fill_PCE` ne retournait pas `all_sensib_PCE` (perte de données)
**Symptôme :** `return y_PCE, all_grad_PCE` — `all_sensib_PCE` calculée mais non retournée.  
**Solution retenue :** `return y_PCE, all_grad_PCE, all_sensib_PCE`.  
Appel dans ETAPE 1 mis à jour : `y_PCE, all_grad_PCE, all_sensib_PCE = fill_PCE(...)`.

---

### Problème 11 — `fill_PCE` retournait `y_PCE` (NameError) et `fill_inputGP` utilisait `y_pce` (NameError)
**Symptôme :** Variable locale nommée `y_pce` (minuscule) mais return disait `y_PCE` et `fill_inputGP` utilisait `y_pce`.  
**Solution retenue :** Uniformiser en `y_PCE` (majuscule) — renommer la variable locale dans `fill_PCE` en `y_PCE`.

---

### Problème 12 — `fill_PCE` : `T_inv` non défini localement
**Symptôme :** `T_inv` utilisé à ligne 328 (`x_i = T_inv(...)`) mais seul `T` était défini localement.  
**Solution retenue :** Ajouter `T_inv = dist_X.getInverseIsoProbabilisticTransformation()` après `T = dist_X.getIsoProbabilisticTransformation()`.

---

### Problème 13 — Warm start PCE-KRG : FORM ne bénéficiait pas du point de départ U_warm
**Symptôme :** `FORM_KRG` démarrait toujours depuis `[0]*n_var`, ignorant `U_warm`.  
**Solution retenue :** Ajouter `start_point` comme paramètre à `FORM_KRG`, passer `U_warm` à l'appel warm start :
```python
result = FORM_KRG(modelname, params_names, metamodel, start_point=U_warm)
```

---

## 3. Modifications effectuées au code AC_pure_flexion.py

### 3a. `init_GP` — nommage correct + n_var local
```python
def init_GP(modelname, params_names, n0, U_doe_fixed=None, SOL_calc=None, do_pce=do_pce):
    n_var = len(params_names)           # ← ajouté
    ...
    all_grad_hf = np.zeros((n0, n_var)) # ← renommé (était all_grad)
    all_sensib_hf = np.zeros((n0, n_var)) # ← renommé (était all_sensib)
    for i in range(n0):
        for j in range(n_var):
            all_grad_hf[i][j] = SOL_U[i][f'dg_u{j+1}']
            all_sensib_hf[i][j] = SOL_U[i][f'dg_{params_names[j]}']
    return xt, y_hf, all_grad_hf, all_sensib_hf  # ← noms corrigés
```

### 3b. `build_metamodel_PCE` — suppression n0 dans dist_jointe
```python
dist_X = dist_jointe(modelname, params_names)   # ← était dist_jointe(..., n0)
```

### 3c. `fill_PCE` — réécriture complète avec sensibilités
```python
def fill_PCE(modelname, params_names, xt, metamodel_PCE):
    U_doe = ot.Sample(xt)
    y_PCE = np.array(metamodel_PCE(U_doe))          # ← y_PCE (majuscule)
    n_var = U_doe.getDimension()
    n0 = U_doe.getSize()
    dist_X = dist_jointe(modelname, params_names)    # ← plus de n0
    T = dist_X.getIsoProbabilisticTransformation()
    T_inv = dist_X.getInverseIsoProbabilisticTransformation()  # ← ajouté
    all_grad_PCE = np.zeros((n0, n_var))
    all_sensib_PCE = np.zeros((n0, n_var))           # ← ajouté
    for i in range(n0):
        grad_pce_u = metamodel_PCE.gradient(U_doe[i])
        for j in range(n_var):
            all_grad_PCE[i, j] = grad_pce_u[j, 0]
        x_i = T_inv(ot.Sample([U_doe[i]]))[0]
        J_T_i = T.gradient(x_i)
        grad_u_i = ot.Point(all_grad_PCE[i, :])
        grad_x_i = J_T_i * grad_u_i
        all_sensib_PCE[i, :] = np.array(grad_x_i)
    return y_PCE, all_grad_PCE, all_sensib_PCE       # ← all_sensib_PCE ajouté
```

### 3d. `fill_inputGP` — ajout .copy() (fix aliasing)
```python
def fill_inputGP(y_hf, all_grad_hf, all_sensib_hf, y_PCE, all_grad_PCE, all_sensib_PCE, do_pce=do_pce):
    yt = y_hf.copy()              # ← .copy() ajouté
    all_grad = all_grad_hf.copy() # ← .copy() ajouté
    all_sensib = all_sensib_hf.copy()  # ← .copy() ajouté
    if do_pce:
        yt -= y_PCE
        all_grad -= all_grad_PCE
        all_sensib -= all_sensib_PCE
    return yt, all_grad, all_sensib
```

### 3e. `build_metamodel_KRG` — n_var local
```python
def build_metamodel_KRG(xt, yt):
    n_var = xt.shape[1]   # ← ajouté
    basis = ot.ConstantBasisFactory(n_var).build()
    ...
```

### 3f. `FORM_KRG` — metamodel corrigé + n_var local + start_point paramètre obligatoire
```python
def FORM_KRG(modelname, params_names, metamodel, start_point, n_max_FORM=n_max_FORM, tol_FORM=tol_FORM):
    n_var = len(params_names)                        # ← ajouté
    dist_X = dist_jointe(modelname, params_names)
    dist_U = dist_X.getStandardDistribution()
    vect = ot.RandomVector(dist_U)
    output = ot.CompositeRandomVector(metamodel, vect)  # ← était metamodel_KRG
    event = ot.ThresholdEvent(output, ot.Less(), 0.0)
    solver = ot.AbdoRackwitz()
    solver.setMaximumIterationNumber(n_max_FORM)
    solver.setCheckStatus(False)
    solver.setMaximumConstraintError(tol_FORM)
    solver.setStartingPoint(start_point)             # ← paramètre obligatoire passé explicitement
    algo = ot.FORM(solver, event)
    algo.run()
    result = algo.getResult()
    return result
```

Appel initial dans les blocs (PCE-KRG ligne 449, KRG pur ligne 482) :
```python
start_point = [0.0] * len(params_names)
result = FORM_KRG(modelname, params_names, metamodel, start_point)
```

Appel warm start (PCE-KRG ligne 470-471, KRG pur ligne 496-497) :
```python
start_point = U_warm   # OT Point — setStartingPoint accepte OT Point
result = FORM_KRG(modelname, params_names, metamodel, start_point)
```

### 3g. Bloc PCE-KRG warm start — U_doe + np.array() wrapping
```python
if do_warm_start and metamodel(U_warm)[0] > tol_warm_start:
    U_doe = ot.Sample(xt)         # ← ajouté (était absent)
    U_doe.add(U_warm)
    ...
    all_grad_hf = np.vstack([all_grad_hf, np.array(all_grad_hf_warm).reshape(1, -1)])   # ← np.array() ajouté
    all_sensib_hf = np.vstack([all_sensib_hf, np.array(all_sensib_hf_warm).reshape(1, -1)])  # ← np.array() ajouté
    ...
    result = FORM_KRG(modelname, params_names, metamodel, start_point=U_warm)  # ← start_point ajouté
```

---

## 4. État du code au 23/04

### Ce qui est corrigé — tous les bugs résolus
- `init_GP` : return correct (`all_grad_hf`, `all_sensib_hf`), n_var local ✓
- `build_metamodel_PCE` : dist_jointe 2 args ✓
- `fill_PCE` : y_PCE majuscule, T_inv local, all_sensib_PCE retourné ✓
- `fill_inputGP` : .copy() sur les 3 variables ✓
- `build_metamodel_KRG` : n_var local ✓
- `FORM_KRG` : `metamodel` (pas `metamodel_KRG`) ✓, n_var local ✓, start_point paramètre obligatoire ✓
- PCE-KRG warm start : U_doe défini (ligne 453), np.array() wrapping (lignes 460-461) ✓
- KRG pur warm start : U_doe défini (ligne 486), np.array() wrapping (lignes 493-494) ✓

### Bugs restants (non bloquants)
1. `build_metamodel_GEK` lignes 369-370 : `metamodel_pce` → `metamodel_PCE` (GEK non encore implémenté, non bloquant)

### Prêt à tester
PCE-KRG (`try_pce=True`, `do_GEK=False`, `do_warm_start=False`) et KRG pur (`try_pce=False`, `do_GEK=False`, `do_warm_start=False`) sont prêts à tourner.

---

## 5. Fichiers clés

| Fichier | Rôle | État |
|---|---|---|
| `AC_pure_flexion.py` | Script principal | PCE-KRG réécrit, 1 bug bloquant restant (FORM_KRG default arg) |
| `launcher.py` | Lance avec DLL STRAINS | Inchangé |
| `comparaison_KRG_PCEKRG.md` | Résultats PCE-KRG vs KRG pur (DOE fixé n0=15) | β_HF=3.784, PCE-KRG β=3.779 (0.1% erreur), KRG pur β=5.067 (33.9% erreur) |
| `resume_session_1704_aprem.md` | Session HF FORM | Complet |
| `resume_session_2004.md` | Session KRG FORM | Complet |
| `resume_session_2104.md` | Session PCE construction + LOO | Complet |
| `resume_session_2204.md` | Session audit GEK + refactoring | Complet |
| `resume_session_2304.md` | Cette session | Complet |

---

## 6. Architecture des fonctions helper (état final)

```
init_GP(modelname, params_names, n0, U_doe_fixed=None, SOL_calc=None, do_pce=do_pce)
  → (xt, y_hf, all_grad_hf, all_sensib_hf)
  → appelle tirage_DOE ou utilise U_doe_fixed
  → appelle fill_sol ou utilise SOL_calc

fill_sol(modelname, params_names, U_doe)
  → SOL_U (liste de dicts avec 'g', 'dg_u1', 'dg_u2', 'dg_fc', 'dg_fy')
  → appelle run_HF(modelname, params_names, u) pour chaque point

build_metamodel_PCE(modelname, params_names, xt, y_hf, q, max_degree, seuil_pce, min_max_degree)
  → metamodel_PCE (ot.Function, PCE via LARS+CorrectedLOO)

fill_PCE(modelname, params_names, xt, metamodel_PCE)
  → (y_PCE, all_grad_PCE, all_sensib_PCE)
  → y_PCE : np array (n0, 1)
  → all_grad_PCE : np array (n0, n_var) — gradient en espace U
  → all_sensib_PCE : np array (n0, n_var) — sensibilités en espace X via J_T

fill_inputGP(y_hf, all_grad_hf, all_sensib_hf, y_PCE, all_grad_PCE, all_sensib_PCE, do_pce)
  → (yt, all_grad, all_sensib)
  → si do_pce : yt = y_hf - y_PCE (résiduels), all_grad -= all_grad_PCE, all_sensib -= all_sensib_PCE
  → si not do_pce : copies directes (pas de soustraction)
  → IMPORTANT : utilise .copy() pour éviter l'aliasing numpy

build_metamodel_KRG(xt, yt)
  → metamodel_KRG (ot.Function, KRG SquaredExponential + base constante)
  → yt peut être y_hf (KRG pur) ou résiduels PCE (PCE-KRG)

FORM_KRG(modelname, params_names, metamodel, start_point=None, n_max_FORM, tol_FORM)
  → result (ot.FORMResult)
  → metamodel = metamodel_KRG (KRG pur) ou metamodel_KRG + metamodel_PCE (PCE-KRG)
```

**Flux PCE-KRG :**
1. `init_GP` → `xt, y_hf, all_grad_hf, all_sensib_hf`
2. `build_metamodel_PCE(xt, y_hf)` → `metamodel_PCE`
3. `fill_PCE(xt, metamodel_PCE)` → `y_PCE, all_grad_PCE, all_sensib_PCE`
4. `fill_inputGP(y_hf, ..., y_PCE, ...)` → `yt=y_hf-y_PCE, all_grad, all_sensib`
5. `build_metamodel_KRG(xt, yt)` → `metamodel_KRG` (sur résiduels)
6. `metamodel = metamodel_KRG + metamodel_PCE` (si do_pce)
7. `FORM_KRG(metamodel)` → result, β, Pf, u*

**Flux KRG pur :**
1. `init_GP` → `xt, yt, all_grad, all_sensib`
2. `build_metamodel_KRG(xt, yt)` → `metamodel`
3. `FORM_KRG(metamodel)` → result

---

## PARTIE 2 — Session après-midi 23/04 : Tests KRG pur + architecture GEK

---

## 7. Sujets théoriques abordés (partie 2)

### 7a. UnicodeDecodeError cp1252 — caractère `<-` (fleche gauche)

Le launcher lit `AC_pure_flexion.py` avec l'encodage Windows par défaut `cp1252`. Le caractère `←` (U+2190, UTF-8 : `\xe2\x86\x90`) est invalide en cp1252 → `UnicodeDecodeError: 'cp1252' codec can't decode byte 0x90`. Fix : remplacer `←` par `--` dans les commentaires.

**Règle :** Eviter tout caractère Unicode non-ASCII dans les commentaires Python lancés via launcher Windows.

---

### 7b. IndentationError — `else:` avec seulement un commentaire (pas de `pass`)

Un bloc `else:` ne contenant que des commentaires (pas de code, pas de `pass`) provoque `IndentationError`. Toujours ajouter `pass` si le bloc est vide ou que du code commenté.

---

### 7c. Mécanisme warm start — limite de détection

Le warm start teste `metamodel(U_warm)[0] > tol_warm_start`. Si FORM a convergé sur le metamodel (g_meta ≈ 0), la condition est fausse même si le metamodel est globalement faux. Le warm start ne peut pas détecter une "mauvaise convergence" du metamodel — il détecte seulement que g_meta(u*) est encore positif.

---

### 7d. `ot.Sample` depuis `ot.Point` — forme correcte

`ot.Sample(ot.Point)` crée un Sample de taille `(n_var, 1)` (chaque coordonnée devient un échantillon 1D) — **incorrect** pour une évaluation de metamodel. La forme correcte pour un seul point :
```python
ot.Sample([u])                         # liste Python wrapping ot.Point
ot.Sample(np.array(u).reshape(1, -1))  # depuis numpy
```
`metamodel_PCE.gradient(u)` accepte directement un `ot.Point` — pas de conversion nécessaire.

---

### 7e. Forme du gradient dans `FORM_GEK` — convention OT PythonFunction

OT `PythonFunction` avec `gradient=grad_func` exige que `grad_func(u)` retourne shape `(1, n_var)` = `[[dg/du1, dg/du2]]`. La convention OT Matrix est transposée de la convention mathématique standard.

`build_metamodel_total` retourne `grad` de shape `(n_var, 1)`. Il faut donc faire `return grad.T` dans `grad_g_GEK` pour passer de `(n_var, 1)` à `(1, n_var)`.

---

### 7f. Formes des gradients PCE et GEK — compatibilité addition

- `grad_GEK = np.array([[float(sm.predict_derivatives(u_np, kx)[0, 0])] for kx in range(n_var)])` → shape `(n_var, 1)`
- `grad_PCE = np.array(metamodel_PCE.gradient(u))` → OT Matrix `(n_var, 1)` converti en numpy `(n_var, 1)`

Même shape → addition directe `grad_GEK + grad_PCE` valide.

---

### 7g. Définition de fonction locale dans une fonction

`def f():` à l'intérieur de `def g():` crée une fonction locale, invisible hors de `g`. Valide Python, recommandé quand la fonction n'est utilisée qu'en un seul endroit. Peut utiliser les variables de la closure de `g`.

---

### 7h. Variables libres vs paramètres dans les fonctions

Flags définis une seule fois sans jamais être réassignés (`do_GP`, `do_GEK`, `do_linear_test`, `do_GP_linear_test`, `do_GP_HF_test`, etc.) → sûrs en tant que variables libres dans les fonctions imbriquées. Variables qui peuvent être réassignées (`result`, `metamodel`) → passer explicitement en paramètre.

---

### 7i. `n_iter = result.getOptimizationResult().getIterationNumber()`

Retourne le nombre d'itérations du solver AbdoRackwitz pendant FORM, **pas** le nombre d'appels HF (STRAINS). Pendant FORM sur metamodel, les appels HF = 0.

---

### 7j. Argument par défaut `do_analytic_grad=do_analytic_grad`

Dans `def FORM_GEK(..., do_analytic_grad=do_analytic_grad)`, la valeur par défaut est évaluée au moment où Python exécute la ligne `def`. Fonctionne car la globale `do_analytic_grad` est définie avant ce `def` et n'est jamais réassignée.

---

### 7k. Test 1 vs Test 2 (décision)

- **Test 1** : `g_HF(u*_meta)` — vérifie si u* méta est un vrai point de défaillance. Implémenté dans `GP_HF_test`.
- **Test 2** : `g_meta(u*_HF)` — vérifie si le metamodel est correct au u* HF (hardcodé, nécessite un run HF préalable). Décision : **Test 2 abandonné**, Test 1 suffisant.

---

## 8. Corrections de bugs (partie 2)

### Bug 1 — UnicodeDecodeError (cp1252 byte 0x90)
**Cause :** Flèche `←` dans un commentaire de `fill_PCE` encodée en UTF-8 incompatible cp1252.
**Fix :** Remplacé par `--`.

### Bug 2 — IndentationError dans `tirage_DOE`
**Cause :** 5 espaces au lieu de 4 (indentation incorrecte ligne 245).
**Fix :** Suppression de l'espace surnuméraire.

### Bug 3 — IndentationError `else:` avec seulement un commentaire
**Cause :** Bloc `else:` (ligne 724) contenant uniquement un commentaire, sans `pass`.
**Fix :** L'utilisatrice a commenté l'ensemble du bloc.

### Bug 4 — `resultats_KRG` → `resultats_GP` renommage + `g_GP_res`
**Cause :** `resultats_KRG` ne gérait pas GEK et ne retournait pas `g_GP_res`.
**Fix :** Renommée en `resultats_GP`, ajout de `g_GP_res` en retour et en paramètre de `print_resultats`.

### Bug 5 — `g_GP` indéfini dans `GP_HF_test` (ligne 522)
**Cause :** `abs(g_HF - g_GP)` utilise `g_GP` non défini ; la variable correcte est `g_GP_res`.
**Statut :** **Non corrigé** — à faire.

### Bug 6 — Double `return metamodel` dans `build_metamodel_total` (ligne 423)
**Cause :** Ligne 423 est du code mort après le `return` de ligne 419.
**Statut :** **Non corrigé** — à faire (supprimer ligne 423).

### Bug 7 — `n_var` non défini dans `build_metamodel_total`
**Cause :** `n_var` utilisé comme variable libre dans les closures internes, mais non défini globalement.
**Statut :** **Non corrigé** — à faire : ajouter `n_var = len(params_names)` ou passer en paramètre.

---

## 9. Nouvelles fonctions ajoutées (partie 2)

### `resultats_GP(modelname, params_names, result, metamodel)`
```python
def resultats_GP(modelname, params_names, result, metamodel):
    n_iter = result.getOptimizationResult().getIterationNumber()
    U_res = result.getPhysicalSpaceDesignPoint()
    dist_X = dist_jointe(modelname, params_names)
    T_inv = dist_X.getInverseIsoProbabilisticTransformation()
    X_res = T_inv(U_res)
    if do_GEK:
        g_GP_res, grad_res = metamodel(U_res)
    else:
        g_GP_res = metamodel(U_res)[0]
        grad_res = metamodel.gradient(U_res)
    importance = result.getImportanceFactors()
    beta = result.getHasoferReliabilityIndex()
    Pf_FORM = result.getEventProbability()
    return n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM
```
**Note :** `T_inv` utilisé pour convertir U_res → X_res via la transformation inverse (KRG et GEK vivent en espace U).

---

### `print_resultats(n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM)`
```python
def print_resultats(n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM):
    n_var = U_res.getDimension()
    print(f"Nombre d'iterations FORM : {n_iter}")
    for i in range(n_var):
        print(f"  Design point U : u_{params_names[i]} = {U_res[i]:.4f}")
    for i in range(n_var):
        print(f"  Design point X : {params_names[i]} = {X_res[i]:.4f}")
    print(f"  g_GP_res   = {g_GP_res:.6f}")
    for i in range(n_var):
        print(f"  dg/du_{params_names[i]} en u* = {grad_res[i, 0]:.6f}")
    for i in range(n_var):
        print(f"  Importance factor {params_names[i]}: {importance[i]:.4f}")
    print(f"\nBeta FORM = {beta:.6f}", flush=True)
    print(f"Pf FORM   = {Pf_FORM:.6e}", flush=True)
```

---

### `GP_linear_test(modelname, params_names, U_res)`
Teste si g est bien linéaire : compare u* FORM avec u* FOSM (linéarisation depuis l'origine).
```python
def GP_linear_test(modelname, params_names, U_res):
    n_var = U_res.getDimension()
    u0 = ot.Point([0.0] * n_var)
    g0, grad_U_0, _ = run_HF(modelname, params_names, u0)
    norm_sq = grad_U_0.norm() ** 2
    u_FOSM = grad_U_0 * (-g0 / norm_sq)
    relative_error_FOSM = (u_FOSM - U_res).norm() / U_res.norm()
    return u_FOSM, relative_error_FOSM
```

---

### `GP_HF_test(modelname, params_names, U_res, g_GP_res, metamodel)`
Vérifie que u* metamodel est un vrai point de défaillance (Test 1).
```python
def GP_HF_test(modelname, params_names, U_res, g_GP_res, metamodel):
    g_HF, _, _ = run_HF(modelname, params_names, U_res)
    relative_error_HF = abs(g_HF - g_GP_res) / abs(g_HF)  # NB: g_GP dans version actuelle = BUG
    return g_GP_res, g_HF, relative_error_HF
```
**BUG RESTANT :** Version actuelle utilise `g_GP` (indéfini) au lieu de `g_GP_res`.

---

### `print_GP_tests(modelname, params_names, U_res, g_GP_res, metamodel)`
```python
def print_GP_tests(modelname, params_names, U_res, g_GP_res, metamodel):
    if do_GP_linear_test:
        u_FOSM, relative_error_FOSM = GP_linear_test(modelname, params_names, U_res)
        print(f"\nTest linearisation :")
        print(f"  u* FORM = {U_res}")
        print(f"  u* FOSM = {u_FOSM}")
        print(f"  Erreur relative entre u* FORM et u* FOSM : {relative_error_FOSM:.4f}")
    if do_GP_HF_test:
        g_GP, g_HF, relative_error_HF = GP_HF_test(modelname, params_names, U_res, g_GP_res, metamodel)
        print(f"\nTest GP au point de FORM :")
        print(f"  g* FORM = {g_HF:.6f}")
        print(f"  g* GP   = {g_GP:.6f}")
        print(f"  Erreur relative entre g* FORM et g* GP : {relative_error_HF:.4f}")
```

---

### `build_metamodel_total(metamodel_PCE, sm, do_pce)`
Construit la closure GEK (ou GEK pur sans PCE) retournant `(valeur, gradient)`.
```python
def build_metamodel_total(metamodel_PCE, sm, do_pce):
    # BUG : n_var doit etre defini ici (n_var = ?)
    if do_pce:
        def metamodel(u):
            u_np = np.array(u).reshape(1, -1)
            y_GEK = float(sm.predict_values(u_np)[0, 0])
            grad_GEK = np.array([[float(sm.predict_derivatives(u_np, kx)[0, 0])] for kx in range(n_var)])
            u_sample_ot = ot.Sample([u])     # forme correcte pour metamodel_PCE
            y_PCE = float(np.array(metamodel_PCE(u_sample_ot))[0, 0])
            grad_PCE = np.array(metamodel_PCE.gradient(u))  # shape (n_var, 1)
            return y_GEK + y_PCE, grad_GEK + grad_PCE
    else:
        def metamodel(u):
            u_np = np.array(u).reshape(1, -1)
            y_GEK = float(sm.predict_values(u_np)[0, 0])
            grad_GEK = np.array([[float(sm.predict_derivatives(u_np, kx)[0, 0])] for kx in range(n_var)])
            return y_GEK, grad_GEK
    return metamodel   # ligne 419 — correct (ligne 423 = code mort a supprimer)
```

---

### `FORM_GEK(modelname, params_names, metamodel, start_point)`
```python
def FORM_GEK(modelname, params_names, metamodel, start_point, n_max_FORM=n_max_FORM, tol_FORM=tol_FORM, do_analytic_grad=do_analytic_grad):
    dist_X = dist_jointe(modelname, params_names)
    dist_U = dist_X.getStandardDistribution()
    n_var = len(params_names)
    def g_GEK(u):           # u est un ot.Point (convention OT PythonFunction)
        val, _ = metamodel(u)
        return [val]
    def grad_g_GEK(u):
        _, grad = metamodel(u)
        return grad.T       # (n_var,1) -> (1,n_var) convention OT
    if do_analytic_grad:
        myFunction = ot.PythonFunction(n_var, 1, g_GEK, gradient=grad_g_GEK)
    else:
        myFunction = ot.PythonFunction(n_var, 1, g_GEK)
    vect   = ot.RandomVector(dist_U)
    output = ot.CompositeRandomVector(myFunction, vect)
    event  = ot.ThresholdEvent(output, ot.Less(), 0.0)
    solver = ot.AbdoRackwitz()
    solver.setMaximumIterationNumber(n_max_FORM)
    solver.setCheckStatus(False)
    solver.setMaximumConstraintError(tol_FORM)
    solver.setStartingPoint(start_point)
    algo = ot.FORM(solver, event)
    algo.run()
    result = algo.getResult()
    return result
```

---

## 10. Résultats KRG pur — run3 (F=0.210, n0=15, DOE fixé)

| Paramètre | Valeur |
|---|---|
| n points DOE | 15 (fixé) |
| fc* (MPa) | 26.73 |
| fy* (MPa) | 475.11 |
| u* | [-1.756, -4.750] |
| beta FORM | 5.065 |
| Pf FORM | 2.05e-07 |
| g_HF(u*) | -4.96e-02 |
| g_KRG(u*) | +9.3e-05 |
| Erreur relative g | 100.2% |
| u* FOSM | [-0.643, -3.728] |
| Erreur FOSM | 29.84% |
| Ecart beta vs HF | +33.9% |

**Conclusion :** DOE fixé améliore KRG pur (run2 LHS aléatoire : 50.3% → run3 DOE fixé : 33.9%). g_HF(u*)=-0.05 confirme surface limite metamodel mal placée. Motivation PCE-KRG confirmée.

---

## 11. Résultats F=0.195 (warm start)

Run au point F=0.195 (beta_HF ≈ 5.48, plus difficile) pour valider le mécanisme warm start.
- Warm start déclenché (g_meta(u*) > tol), STRAINS appelé pour nouveau point.
- FORM divergé vers point physiquement aberrant (u_fc=+5.09, beta=8.20) — n0=15 insuffisant a beta≈5.48.
- **Décision :** Pas ajouté au .md de résultats, juste validation du déclenchement warm start.
- Après test : dsLoad.txt remis a F=0.210 (Z='-0.210').

---

## 12. DOE fixé utilisé (n0=15, U-space)

```python
U_doe_fixed = ot.Sample([
    [ 1.0272625484832025,  0.3251235065050853],
    [ 0.2588934150948534, -1.6856336900013655],
    [-0.7900915845657982,  1.8047217395005692],
    [-0.0301755082064849,  1.3223984111477798],
    [-1.8073810055112547, -1.1012751718677385],
    [-0.2377471223963969, -0.4914312425631510],
    [ 0.7216266145109314,  1.0830320538875535],
    [ 0.4776729449462016, -0.2656508781535193],
    [-0.8730465106774573,  0.6497494474356423],
    [-1.1677174906609287,  0.0310652111349381],
    [ 1.1194425579629474, -0.7943643305093363],
    [ 0.1857520921586401,  0.4724170659386679],
    [-0.5669380193636159, -1.4858232340964800],
    [ 2.9454553139272623, -0.1582987245612891],
    [-0.2947626989079067,  0.1355018527305618],
])
```

---

## 13. Bugs restants au fin de session 2304 (partie 2)

| # | Fichier / ligne | Description | Priorité |
|---|---|---|---|
| 1 | `GP_HF_test` ligne 522 | `g_GP` → `g_GP_res` (NameError) | **URGENT** |
| 2 | `build_metamodel_total` ligne 423 | Double `return metamodel` — code mort | Mineur |
| 3 | `build_metamodel_total` | `n_var` non defini — NameError | **URGENT** |
| 4 | `dsLoad.txt` | Remettre Z='-0.210' apres test F=0.195 | **FAIT** |
| 5 | `build_metamodel_GEK` lignes 369-370 | `metamodel_pce` → `metamodel_PCE` | Non bloquant (GEK non lancé) |

---

## 14. Etat du code fin de session 2304 — OPTIONS actives

```python
do_warm_start = True
do_GEK = False          # KRG pur
try_pce = False         # KRG pur
do_GP_linear_test = True
do_GP_HF_test = True
```

---

## 15. Prochaines étapes

1. Corriger les 3 bugs urgents (#1, #3 ci-dessus + vérifier n_var dans build_metamodel_total)
2. Implémenter le bloc GEK complet dans les if/elif appelants
3. Tester GEK a F=0.210 et comparer avec KRG pur et PCE-KRG
4. Remplir `comparaison_KRG_PCEKRG.md` avec les résultats GEK

---

## PARTIE 3 — Session fin 23/04 : Audit GEPCK + GEK pur + run GEK pur run1

---

## 16. Sujets théoriques abordés (partie 3)

### 16a. Argument par defaut avant argument sans defaut — SyntaxError

`def f(a=None, b):` est une SyntaxError en Python 3. Python interdit un argument AVEC valeur par defaut avant un argument SANS valeur par defaut. Le fichier ne se charge pas du tout (erreur a la compilation).

**Regle :** les arguments avec valeur par defaut (`=`) doivent toujours etre APRES les arguments obligatoires.

---

### 16b. Argument `metamodel_PCE=None` — lien entre do_pce et presence du metamodel

Nouvelle logique adoptee : au lieu de passer `do_pce` comme parametre a `build_metamodel_total`, on passe `metamodel_PCE` directement. La presence (`is not None`) ou absence (`None`) de `metamodel_PCE` determine si le bloc PCE est ajoute. Cela elimine la redundance `do_pce` comme parametre.

Signature finale : `def build_metamodel_total(sm, metamodel_PCE=None):`
- `metamodel_PCE=None` → branche `else` (GEK pur)
- `metamodel_PCE=<objet OT>` → branche `if` (GEK + PCE)

Les blocs appellants creenent `metamodel_PCE` uniquement dans `if do_pce:` et le passent a `build_metamodel_total(sm, metamodel_PCE)`.

---

### 16c. GEK pur — instabilite avec n0=15

GEK pur avec n0=15 DOE fixe donne β=2.118 (erreur -44% vs HF β=3.784). FORM n'a pas converge sur la surface g=0 du metamodele (g_GP_res=+0.074 en u*). Le point de defaillance trouve (u_fc=+0.316) a une fc superieure a la moyenne, ce qui est physiquement absurde — indique un metamodele mal conditionne.

Le test GP (erreur=0.01%) montre que le GEK est localement precis en u*, mais que FORM a trouve le mauvais u*. KRG pur est meilleur sur ce DOE (β=5.065, erreur +34% mais au moins dans le bon sens).

---

### 16d. do_warm_start = False lors du run GEK pur run1

Le warm start etait desactive (`do_warm_start=False`) lors du run GEK pur run1. Le g_GP_res=0.074 >> tol_warm_start=0.0001 aurait du declencher le warm start si `do_warm_start=True`, mais le test n'a pas ete fait.

---

## 17. Audit GEPCK — bugs identifies et corriges (partie 3)

### Bug 1 — `start_point` non defini dans le bloc GEPCK (NameError)

**Symptome :** `FORM_GEK(modelname, params_names, metamodel, start_point)` appele sans que `start_point` soit defini dans le scope GEPCK.

**Cause :** Dans les blocs PCE-KRG (ligne 558) et KRG pur (ligne 596), `start_point = [0.0]*len(params_names)` est defini avant l'appel FORM. Cette ligne etait absente du bloc GEPCK.

**Fix applique :** Ajout de `start_point = [0.0]*len(params_names)` avant la ligne `result = FORM_GEK(...)` dans le bloc GEPCK.

---

### Bug 2 — `n_var` non defini dans `build_metamodel_total` (NameError)

**Symptome :** Les closures internes utilisent `n_var` dans `for kx in range(n_var)`, mais `n_var` n'est jamais defini ni localement dans `build_metamodel_total` ni globalement dans `__main__`.

**Cause :** `n_var` est defini localement dans d'autres fonctions (`init_GP`, `FORM_KRG`, etc.) mais pas dans le scope `__main__`. La closure cherche `n_var` via LEGB : Local → Enclosing → Global → NameError.

**Fix applique :** Ajout de `n_var = len(params_names)` en tete de `build_metamodel_total`. `params_names` est accessible car defini dans `__main__` avant la `def`.

---

### Bug 3 — `g_GP` non defini dans `GP_HF_test` (NameError)

**Symptome :** `abs(g_HF - g_GP) / abs(g_HF)` — `g_GP` non defini. La variable correcte est `g_GP_res` (parametre de la fonction).

**Fix applique :** `abs(g_HF - g_GP_res) / abs(g_HF)`.

---

### Bug 4 — Double `return metamodel` dans `build_metamodel_total`

**Symptome :** Deux `return metamodel` consecutifs — le second etait du code mort apres le premier `return`.

**Fix applique :** Suppression du second `return metamodel` redondant.

---

### Bug 5 — SyntaxError signature `build_metamodel_total(metamodel_PCE=None, sm)`

**Symptome :** Apres le refactoring de l'utilisatrice, la signature `def build_metamodel_total(metamodel_PCE=None, sm):` levait une SyntaxError — argument avec defaut avant argument sans defaut.

**Fix applique :** Inversion en `def build_metamodel_total(sm, metamodel_PCE=None):`.

Consequence : les appels GEPCK `build_metamodel_total(metamodel_PCE, sm)` devaient aussi etre mis a jour en `build_metamodel_total(sm, metamodel_PCE)`.

---

### Bug 6 — `metamodel_PCE` non defini dans le warm start GEK pur (NameError)

**Symptome :** Dans le warm start du bloc GEK pur, `build_metamodel_total(sm, metamodel_PCE)` utilisait `metamodel_PCE` qui n'est jamais defini dans le scope GEK pur (pas de PCE).

**Fix applique :** `build_metamodel_total(sm)` — `metamodel_PCE` prend la valeur par defaut `None`.

---

### Bug 7 — IndentationError `else:` avec seulement un commentaire (ancien code)

**Symptome :** Dans le bloc `do_run_ancien`, un `else:` ne contenait qu'un commentaire et aucun code ni `pass` → `IndentationError: expected an indented block`.

**Fix applique :** Ajout de `pass` dans ce `else:`. Puis l'utilisatrice a prefere commenter entierement le bloc `do_run_ancien` pour proprete.

---

### Bug 8 — dsLoad.txt a F=0.195 (oublie du test F=0.195 de la partie 2)

**Symptome :** dsLoad.txt avait Z='-0.195' au lieu de Z='-0.210' suite au test de la partie 2 de session.

**Fix applique :** Remis a Z='-0.210' avant le run GEK pur.

---

## 18. Modifications effectuees au code (partie 3)

### 18a. `build_metamodel_total` — version finale

```python
def build_metamodel_total(sm, metamodel_PCE=None):
    n_var = len(params_names)      # fix bug 2 : n_var local
    if metamodel_PCE is not None:  # fix bug 5 : signature corrigee, logique do_pce remplacee
        def metamodel(u):
            u_np = np.array(u).reshape(1, -1)
            y_GEK = float(sm.predict_values(u_np)[0, 0])
            grad_GEK = np.array([[float(sm.predict_derivatives(u_np, kx)[0, 0])] for kx in range(n_var)])
            u_sample_ot = ot.Sample(u_np)
            y_PCE = float(np.array(metamodel_PCE(u_sample_ot))[0, 0])
            grad_PCE = np.array(metamodel_PCE.gradient(u))
            return y_GEK + y_PCE, grad_GEK + grad_PCE
    else:
        def metamodel(u):
            u_np = np.array(u).reshape(1, -1)
            y_GEK = float(sm.predict_values(u_np)[0, 0])
            grad_GEK = np.array([[float(sm.predict_derivatives(u_np, kx)[0, 0])] for kx in range(n_var)])
            return y_GEK, grad_GEK
    return metamodel   # fix bug 4 : un seul return
```

---

### 18b. `GP_HF_test` — fix g_GP → g_GP_res

```python
# AVANT
relative_error_HF = abs(g_HF - g_GP) / abs(g_HF)
# APRES
relative_error_HF = abs(g_HF - g_GP_res) / abs(g_HF)
```

---

### 18c. Bloc GEPCK — ajout start_point avant FORM

```python
# AVANT (ligne 632 avant fix)
result = FORM_GEK(modelname, params_names, metamodel, start_point)  # start_point undefined

# APRES
start_point = [0.0]*len(params_names)
result = FORM_GEK(modelname, params_names, metamodel, start_point)
```

---

### 18d. Appels `build_metamodel_total` — ordre arguments corrige

```python
# GEPCK (lignes 624, 645)
metamodel = build_metamodel_total(sm, metamodel_PCE)   # sm en premier

# GEK pur (ligne 661 initial + 678 warm start)
metamodel = build_metamodel_total(sm)                   # metamodel_PCE=None par defaut
```

---

### 18e. Blocs PCE-KRG et GEPCK — `metamodel_PCE` dans `if do_pce:`

Refactoring de l'utilisatrice : `metamodel_PCE` n'est cree que si `do_pce=True` :
```python
if do_pce:
    metamodel_PCE = build_metamodel_PCE(modelname, params_names, xt, y_hf)
y_PCE, all_grad_PCE, all_sensib_PCE = fill_PCE(modelname, params_names, xt, metamodel_PCE)
```
Note : si `do_pce` devenait `False` apres une validation PCE future, `metamodel_PCE` serait indefini avant `fill_PCE` — latent, non bloquant actuellement car `do_pce=try_pce=True` a l'entree de ces blocs.

---

### 18f. Bloc `do_run_ancien` — commente

L'ensemble du bloc `if do_run_ancien:` (ancien code de debug) a ete commente par l'utilisatrice pour eviter les IndentationError residuels.

---

## 19. Trace complete du flux GEK pur verifie

| Etape | Code | Statut |
|---|---|---|
| `init_GP(modelname, params_names, n0, U_doe_fixed)` | 4 vals retournees | OK |
| `build_metamodel_GEK(xt, yt, all_grad)` | xt (n0,n_var), yt (n0,1), all_grad (n0,n_var) | OK |
| `build_metamodel_total(sm)` | metamodel_PCE=None → branche else | OK |
| `start_point = [0.0]*len(params_names)` | defini avant FORM | OK |
| `FORM_GEK(modelname, params_names, metamodel, start_point)` | closure g_GEK(u) + do_analytic_grad=False | OK |
| Warm start check `metamodel(U_warm)[0]` | tuple[0] = y_GEK (float) | OK |
| Warm start : `run_HF(...)` → 3 vals | yt_warm, all_grad_warm, all_sensib_warm | OK |
| Warm start : `build_metamodel_total(sm)` | metamodel_PCE=None, pas de PCE | OK |
| `resultats_GP(...)` → do_GEK=True | `g_GP_res, grad_res = metamodel(U_res)` | OK |
| `print_resultats(...)` | `grad_res[i,0]` sur numpy (n_var,1) | OK |
| `print_GP_tests(...)` → `GP_HF_test(...)` | `g_GP_res` fixe | OK |

---

## 20. Resultats GEK pur run1 (F=0.210, n0=15, DOE fixe, do_warm_start=False)

| Parametre | Valeur |
|---|---|
| n points DOE | 15 (fixe) |
| n iter FORM | 37 |
| fc* (MPa) | 33.26 |
| fy* (MPa) | 555.17 |
| u* | [+0.316, -2.095] |
| g_GP_res | +0.074 |
| Importance fc | 2.22% |
| Importance fy | 97.78% |
| beta FORM | 2.118 |
| Pf FORM | 1.71e-02 |
| Test GP erreur | 0.01% |
| Test FOSM erreur | 89.4% |

**Comparaison :**

| | HF | KRG pur run3 | GEK pur run1 |
|---|---|---|---|
| beta | 3.784 | 5.065 | 2.118 |
| Erreur | 0% | +33.9% | -44.0% |
| u* | [-0.53, -3.75] | [-1.76, -4.75] | [+0.32, -2.09] |
| g_meta(u*) | ~0 | +9.3e-05 | +0.074 |
| Test GP erreur | -- | 100.2% | 0.01% |

**Conclusion :** GEK pur run1 est moins bon que KRG pur sur ce DOE. FORM n'a pas converge sur la surface g=0 du metamodele (g=0.074 en u*). Le point de defaillance (u_fc>0) est physiquement aberrant. Le GEK est localement precis (test GP 0.01%) mais mal conditionne globalement. Le warm start etait desactive.

---

## 21. Principe de remplissage des fichiers .md (rappel)

| Fichier | Contenu | Quand ecrire |
|---|---|---|
| `resume_session_DDMM.md` | Resumes de session : sujets, bugs, solutions, code cite, resultats | Fin de session ou apres autocompactage |
| `comparaison_KRG_PCEKRG.md` | Tableau comparatif KRG / PCE-KRG / GEK / GEPCK sur memes conditions | Apres chaque nouveau run valide |
| `resultats_KRG_runN.md` | Resultats detailles d'un run individuel | Apres run important |
| `MEMORY.md` + fichiers memory | Info persistante cross-session : feedback, profil user, projet, references | Quand une info doit survivre a la suppression de session |

**Regles absolues :**
- Ne jamais supprimer un fichier sans confirmation explicite de l'utilisatrice
- Les `.md` de resultats ne sont jamais ecrases — on append ou on cree un nouveau fichier runN+1
- `resume_session_DDMM.md` s'append par parties (PARTIE 1, PARTIE 2, etc.) dans la meme session si autocompactage

---

## 22. Etat du code fin de session 2304 partie 3

```python
do_GEK = True
try_pce = False     # GEK pur
do_pce = False
do_warm_start = False
do_GP_linear_test = True
do_GP_HF_test = True
```

**Tous les bugs de la liste partie 2 sont corriges.** `do_run_ancien` commente.

---

## 23. Prochaines etapes

1. Relancer GEK pur avec `do_warm_start=True` pour voir si le warm start corrige β
2. Analyser pourquoi FORM GEK converge a g=0.074 (non-convergence sur la surface)
3. Tester GEPCK (do_GEK=True, try_pce=True) et comparer avec KRG pur et HF
4. Remplir `comparaison_KRG_PCEKRG.md` avec colonne GEK pur et GEPCK
