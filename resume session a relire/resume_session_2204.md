# Résumé de session — Débogage LOO + Run PCE-KRG
**Date :** 22 avril 2026
**Objectif :** Comprendre le fonctionnement de `compute_q2_loo`, corriger le segfault, et obtenir un premier run PCE+KRG complet.

---

## 1. Sujets théoriques abordés

### 1a. Double validation croisée (double CV) — pourquoi et comment

**Problème de départ :** `FunctionalChaosValidation` crash en OT 1.26 avec LARS (voir résumé 2104). Il faut une validation LOO non biaisée.

**Double CV :** Pour valider un modèle construit avec LARS (sélection de base adaptive), on ne peut pas utiliser la LOO analytique (hat matrix) car la base Ψ a été construite en regardant toutes les données → biais post-sélection. La seule approche honnête est de re-entraîner LARS à chaque fold.

**Procédure concrète avec n0=5 :**

| Fold | Points d'entraînement | Point de test |
|---|---|---|
| 1 | u2, u3, u4, u5 | u1 |
| 2 | u1, u3, u4, u5 | u2 |
| 3 | u1, u2, u4, u5 | u3 |
| 4 | u1, u2, u3, u5 | u4 |
| 5 | u1, u2, u3, u4 | u5 |

Pour chaque fold i :
1. LARS tourne sur les 4 points d'entraînement → construit Ψᵢ* et calcule coefficients α̂ᵢ
2. On évalue le polynôme Ψᵢ* au point de test uᵢ : `ĝᵢ(uᵢ) = Σₖ α̂ᵢₖ · ψₖ(uᵢ)` (substitution numérique simple, pas de STRAINS)
3. On stocke la prédiction LOO : `g_loo_list[i] = ĝᵢ(uᵢ)`

À la fin : Q² calculé sur les 5 paires (g(uᵢ) réel, ĝᵢ(uᵢ) prédit) :
```
Q² = 1 - Σᵢ (g(uᵢ) - ĝᵢ(uᵢ))² / Σᵢ (g(uᵢ) - ḡ)²
```

**Ce que valide le Q² LOO :** pas Ψ* directement, mais la **procédure** "appliquer LARS sur n0-1 points". Si Q² bon → on fait confiance que LARS sur n0 points (Ψ*) généralise bien.

**Pourquoi on ne peut pas valider Ψ* directement :** Ψ* a été construit en voyant tous les n0 points, y compris chaque point qu'on voudrait utiliser comme test. Le score serait biaisé (trop optimiste).

**"Double" :** deux niveaux imbriqués — sélection de Ψ par LARS (niveau interne), validation par fold (niveau externe).

---

### 1b. Segfault — définition et comportement

Un segfault (segmentation fault) = crash au niveau OS. Du code C++ accède à une zone mémoire interdite. Python n'a pas le temps de lever une exception ni de flusher la sortie. Le processus est tué directement par l'OS. C'est pour ça qu'on ne voyait que "LAUNCHER START" dans la sortie.

**Technique de diagnostic :** ajouter des prints avec `flush=True` pour forcer l'écriture avant le crash. Permet de localiser la dernière ligne exécutée avant la mort du processus.

---

### 1c. Argument par défaut mutable en Python — `propos=ot.LARS()`

```python
def result_PCE(..., propos=ot.LARS(), select=ot.CorrectedLeaveOneOut(), ...):
```

`ot.LARS()` est créé **une seule fois** quand Python lit la définition de la fonction (au chargement du script), pas à chaque appel. Tous les appels sans `propos` explicite partagent **le même objet L en mémoire**. Si LARS modifie son état interne pendant l'exécution, les appels LOO suivants reçoivent un objet potentiellement corrompu.

Les appels avec `propos=ot.LARS()` **explicite** créent un nouvel objet à chaque appel → sûrs.

**Statut :** risque théorique identifié par la doc OT 1.26, mais non confirmé comme cause du segfault (les 5 folds LOO se terminaient correctement dans les tests diagnostiques).

---

### 1d. Syntaxes OT vérifiées par recherche doc 1.26

| Syntaxe | Verdict | Notes |
|---|---|---|
| `inputSample[indicesTrain]` avec `ot.Indices` | **Sûre** | Syntaxe officielle montrée dans doc `LeaveOneOutSplitter` |
| `sample[0, 0]` sur `ot.Sample` | **Sûre** | Accès scalaire ligne/colonne valide |
| `computeR2Score()[0]` | **Sûre** | Retourne `ot.Point`, `[0]` valide |
| `FunctionalChaosAlgorithm` 4 points + basis_size=5 | **Sûre en pratique** | Risque théorique (sous-déterminé), mais les 5 folds ont tourné sans crash |
| `propos=ot.LARS()` en défaut | **Risque théorique** | Objet mutable partagé, fix recommandé |

---

## 2. Problèmes rencontrés et solutions

### Problème 1 — Segfault lors de l'exécution de `compute_q2_loo`

**Symptôme :** Exit code 139 (Segmentation fault), seul "LAUNCHER START" visible en sortie.

**Diagnostic :** Ajout de prints avec `flush=True` à chaque étape.

**Localisation progressive :**
1. Premier diagnostic → crash avant tout print LOO
2. Ajout prints au début de `compute_q2_loo` → print visible "entree compute_q2_loo", "n=5", "splitter cree"
3. Tous les 5 folds LOO se terminent : "fold 0 PCE OK", ..., "fold 4 PCE OK"
4. Q² LOO calculé : `q2_loo=0.3735...` (max_degree=2), puis `q2_loo=...` (max_degree=1 → > 0.90)
5. Print "Le PCE est de bonne qualité avec max_degree=1" visible
6. Print "[KRG] debut etape 3.2" visible
7. Print "[KRG] yt=..., do_pce=True" visible
8. Print "[KRG] yt apres soustraction PCE=..." visible (valeurs ~1e-4)
9. **Crash après ici** → dans `algo_KRG.run()` ou `metamodel_KRG += metamodel_pce`

**Conclusion :** Le segfault N'EST PAS dans `compute_q2_loo`. Il est dans le bloc ETAPE 3.2 (construction KRG), probablement dans `algo_KRG.run()` avec `yt` résiduel très petit (~1e-4) ou dans `metamodel_KRG += metamodel_pce`.

**Solution retenue :** L'utilisatrice a commenté/modifié le bloc de validation pour contourner temporairement. Le code tourne sans segfault avec `if True:` à la place de `if q2_loo > seuil_pce:`.

**Cause exacte du segfault KRG non encore confirmée.** Hypothèses :
- `algo_KRG.run()` avec `yt` très petit (~1e-4) → instabilité numérique dans l'optimisation de vraisemblance
- `metamodel_KRG += metamodel_pce` → problème de références C++ (non confirmé, rétabli à `result = algo_KRG.getResult()` après test)

---

### Problème 2 — `NameError: name 'VALIDATION' is not defined`

**Symptôme :** Après suppression du bloc de validation, `VALIDATION` référencé ailleurs → erreur Python propre (exit code 1, pas segfault).

**Solution :** L'utilisatrice a ajusté le code pour retirer les références à `VALIDATION`.

---

## 3. Modifications du code AC_pure_flexion.py

### 3a. `compute_q2_loo` — fonction ajoutée (lignes 411-420)

```python
    def compute_q2_loo(inputSample, outputSample, dist_U, q, max_degree, seuil_pce):
        n = inputSample.getSize()
        g_loo_list = [None] * n
        for indicesTrain, indicesTest in ot.LeaveOneOutSplitter(n):
            i = int(indicesTest[0])
            result_i = result_PCE(inputSample[indicesTrain], outputSample[indicesTrain],
                                  dist_U, q, max_degree, seuil_pce)
            g_loo_list[i] = result_i.getMetaModel()(inputSample[indicesTest])[0, 0]
        g_loo_pred = ot.Sample([[v] for v in g_loo_list])
        return ot.MetaModelValidation(outputSample, g_loo_pred).computeR2Score()[0]
```

**Note sur le design :** boucle sur `k` de l'ancienne version retirée (inutile en LOO — `indicesTest` a toujours 1 seul élément). `i = int(indicesTest[0])` suffit.

### 3b. Bloc `if try_pce:` — état actuel (partiellement commenté)

```python
    if try_pce:
        inputSample = U_doe
        outputSample = ot.Sample([[SOL[i]['g']] for i in range(n0)])
        q = 0.75
        max_degree = 2
        result = result_PCE(inputSample, outputSample, dist_U, q, max_degree, seuil_pce,
                            propos=ot.LARS(), select=ot.CorrectedLeaveOneOut(), min_max_degree=min_max_degree)
        VALIDATION = {}
        # q2_loo = compute_q2_loo(...)  ← COMMENTÉ (contournement segfault)
        while q2_loo < seuil_pce and max_degree > min_max_degree:  # ← boucle présente mais q2_loo non défini
            max_degree -= 1
            result = result_PCE(...)
            q2_loo = compute_q2_loo(...)
        # if q2_loo > seuil_pce:
        if True:                          ← WORKAROUND temporaire
            do_pce = True
            metamodel_pce = result.getMetaModel()
            y_pce = np.array(metamodel_pce(U_doe))
            all_grad_PCE = np.zeros((n0, n_var))
            for i in range(n0):
                grad_pce_u = metamodel_pce.gradient(U_doe[i])
                for j in range(n_var):
                    all_grad_PCE[i, j] = grad_pce_u[j, 0]
        # else: ... do_pce = False  ← COMMENTÉ
```

### 3c. `result_KRG` — renommage testé puis annulé

Ligne 510 : `result = algo_KRG.getResult()` → testé `result_KRG = algo_KRG.getResult()` puis rétabli à `result`. Ne change pas le comportement (segfault persistait de toute façon dans le bloc de validation).

---

## 4. Résultats obtenus

### Run PCE+KRG réussi (validation bypassée avec `if True:`)

| Paramètre | Valeur |
|---|---|
| F (MN) | 0.210 |
| β FORM | 3.790 |
| Pf FORM | 7.54e-05 |
| n_iter FORM | 15 |
| u* | [-0.616, -3.739] |
| do_pce | True (hardcodé) |
| max_degree retenu | 1 (Q² LOO deg2=0.374 < 0.90, Q² LOO deg1 > 0.90) |

**Comparaison avec HF FORM :** β_HF(F=0.210)=3.784, Pf_HF=7.73e-05 → PCE+KRG très cohérent.

**Anomalie à investiguer :** "Erreur relative entre g* FORM et g* GP : 1.0000" (100%) — probablement dû au PCE qui absorbe presque toute la variance, le KRG résiduel étant ~0.

---

## 5. État du code au 22/04

### Ce qui fonctionne
- `result_PCE` : construit le PCE avec LARS+CorrectedLOO → OK
- `compute_q2_loo` : définie et fonctionnelle → mais **commentée** aux lignes d'appel
- Run PCE+KRG complet → β=3.79 cohérent avec HF

### Ce qui reste à faire
1. **Réactiver `compute_q2_loo`** : décommenter la ligne `q2_loo = compute_q2_loo(...)` + restaurer `if q2_loo > seuil_pce:` + rétablir les branches `do_pce = True/False` proprement
2. **Corriger l'argument par défaut mutable** dans `result_PCE` : créer `ot.LARS()` et `ot.CorrectedLeaveOneOut()` à l'intérieur du corps de la fonction, supprimer `propos` et `select` de la signature
3. **Investiguer l'erreur relative 100%** au test GP au point de FORM
4. **Comprendre la cause exacte du segfault KRG** (hypothèse : `yt` résiduel ~1e-4 trop petit pour l'optimisation KRG, ou conflit entre `result` PCE et KRG en mémoire)

---

## 6. Fichiers clés

| Fichier | Rôle | État |
|---|---|---|
| `AC_pure_flexion.py` | Script principal | compute_q2_loo définie mais appels commentés, if True workaround |
| `launcher.py` | Lance avec DLL STRAINS | Inchangé |
| `out_run_PCE_KRG2.txt` | Sortie du dernier run réussi | β=3.79, Pf=7.54e-05 |
| `resume_session_1704_aprem.md` | Session HF FORM | Complet |
| `resume_session_2004.md` | Session KRG FORM | Complet |
| `resume_session_2104.md` | Session PCE construction + LOO théorie | Complet (mis à jour) |
| `resume_session_2204.md` | Cette session | Complet |

---
---

# Résumé de session — Audit architecture GEK + Refactoring fonctions
**Date :** 22 avril 2026 (suite de session, même journée)
**Objectif :** Auditer l'architecture GEK après refactoring utilisatrice, identifier et corriger les incohérences, refactoriser les fonctions `tirage_DOE`, `fill_sol`, `init_GP`.

---

## 1. Sujets théoriques abordés

### 1a. Variables libres (closures) Python et warm start

`metamodel_GEK` (renommée `build_metamodel_GEK`) utilise `sm` comme variable libre — Python résout les noms libres **à l'appel**, pas à la définition. Donc quand le warm start fait `sm = train_GEK(...)`, le rebinding est automatiquement vu par `build_metamodel_GEK` à l'appel suivant. **Condition** : `sm` doit exister avant le premier appel — il manquait dans la version initiale.

### 1b. Arguments par défaut Python — évaluation au moment de la définition

Les valeurs par défaut sont évaluées **une seule fois** quand Python lit la `def`. Donc `def f(a=x):` capture la valeur de `x` au moment de la définition — les modifications ultérieures de `x` ne sont pas vues. **Exception** : si `x` est défini juste avant la `def` et ne change pas avant, `do_analytic_grad=do_analytic_grad` fonctionne et donne la bonne valeur.

### 1c. SyntaxError Python 3 — argument sans défaut après argument avec défaut

`def f(a=1, b):` est une `SyntaxError` à la compilation (lecture du fichier). Python refuse de parser une signature où un argument sans `=valeur` suit un argument avec `=valeur`. Cette erreur empêche le fichier de se charger, quelle que soit la valeur de la variable globale correspondante.

### 1d. `enumerate` sur `ot.Sample`

`enumerate` est un builtin Python qui fonctionne sur tout itérable. `ot.Sample` étant itérable (`for u in U_doe` fonctionne), `enumerate(U_doe)` fonctionne sans vérification supplémentaire.

### 1e. Dimensions d'un `ot.Sample`

`ot.Sample` est une matrice `n0 × n_var` :
- `getSize()` → n0 (nombre de lignes / points)
- `getDimension()` → n_var (nombre de colonnes / variables par point)
- `for u in U_doe` itère sur les **lignes** → `u` est un `ot.Point` de dimension n_var

### 1f. Pattern "builder vs évaluateur" pour les métamodèles

`build_metamodel_KRG(xt, yt, do_pce)` est un **builder** — il entraîne et retourne un objet OT. `metamodel_KRG = build_metamodel_KRG(xt, yt)` stocke l'objet OT. L'**évaluateur** est ensuite `metamodel_KRG(u)` (appel sur l'objet OT). Il ne faut pas confondre les deux.

### 1g. Argument par défaut `do_analytic_grad=do_analytic_grad` — pourquoi ça marche

`do_analytic_grad = False` est défini dans le scope `__main__` **avant** la `def FORM_GEK(...)`. Python évalue `do_analytic_grad` dans la signature au moment de lire la `def` → trouve la variable globale = False → la capture. Correct. La valeur est figée à la définition, mais puisqu'elle ne change pas, c'est identique à toujours la passer explicitement.

### 1h. Reconstruction de `U_doe` depuis `xt`

`xt` est un numpy array `(n0, n_var)`. Pour retrouver un `ot.Sample` : `U_doe = ot.Sample(xt)`.

### 1i. Pattern `if guard: return` vs `else:`

Quand un `if` se termine par `return`, le `else` est inutile — le code suivant n'est exécuté que si la condition était fausse. Exemple :
```python
if U_doe_fixed is not None:
    return U_doe_fixed
# ici, U_doe_fixed est nécessairement None — pas besoin de else:
dist = []
...
```

### 1j. `yt` vs `g_ref` pour l'affichage

Si `do_pce=False` : `yt = y_hf` → équivalent à `g_ref`, mais format numpy array `(n0,1)` au lieu de liste. Si `do_pce=True` : `yt = y_hf - y_pce` → ce sont les **résiduels** PCE, pas les g bruts. Donc `yt` ne remplace `g_ref` que si `do_pce=False`.

---

## 2. Problèmes identifiés et solutions

### Problème 1 — `run_HF` retourne 3 valeurs, plusieurs appels n'en dépackent que 2

**Symptôme :** `ValueError: too many values to unpack` aux lignes 563, 736, 794, 810 (et 836 dans `do_visu`).

**Cause :** `run_HF` a été modifié pour retourner `(g_HF, grad_HF_U, grad_HF_X)` mais les anciens appels utilisaient `g, grad = run_HF(...)`.

**Solution retenue :** Remplacer par dépackage à 3 valeurs :
```python
g_HF, grad_U, _ = run_HF(...)      # quand grad_HF_X non nécessaire
g_HF, _, _ = run_HF(...)           # quand seul g_HF nécessaire
self._last_g, self._last_grad, _ = run_HF(...)  # HFCache
```

**Lignes corrigées :** 563, 736, 801, 817, 836.

---

### Problème 2 — `build_metamodel_KRG` utilisé comme évaluateur (TypeError)

**Symptôme :** Appels `build_metamodel_KRG(U_warm)[0]` et `build_metamodel_KRG(U_res)[0]` → `TypeError` car `build_metamodel_KRG` attend `(xt, yt, do_pce)`.

**Cause :** Renommage de `metamodel_KRG` (objet OT) en `build_metamodel_KRG` (fonction builder), sans mettre à jour les appels d'évaluation.

**Solution retenue :**
- `metamodel_KRG = build_metamodel_KRG(xt, yt, do_pce=do_pce)` stocke l'objet OT (ligne 551)
- Les évaluations utilisent `metamodel_KRG(U_warm)[0]` et `metamodel_KRG(U_res)[0]`
- Le warm start appelle `metamodel_KRG = build_metamodel_KRG(xt, yt, do_pce=do_pce)` pour re-entraîner

**Lignes corrigées :** 646 (warm start check), 816 (do_GP_test).

---

### Problème 3 — `train_GEK` : `inputSample.getDimension()` hors scope (NameError)

**Symptôme :** `NameError: name 'inputSample' is not defined` si `try_pce=False`.

**Cause :** `inputSample` est définie uniquement dans le bloc `if do_GP and try_pce:`. Hors de ce bloc, elle n'existe pas.

**Solution retenue :** `n_var = xt.shape[1]` — `xt` est toujours défini avant l'appel à `train_GEK`.

**Ligne corrigée :** 503 (anciennement 500).

---

### Problème 4 — `FORM_GEK` : SyntaxError argument sans défaut après arguments avec défaut

**Symptôme :** `SyntaxError: non-default argument follows default argument` — le fichier ne se charge pas.

**Cause :** `def FORM_GEK(..., n_max_FORM=n_max_FORM, tol_FORM=tol_FORM, do_analytic_grad):` — Python 3 interdit un argument sans `=valeur` après des arguments avec `=valeur`.

**Solutions proposées :**
1. `do_analytic_grad=False` (valeur par défaut explicite)
2. `do_analytic_grad=do_analytic_grad` (capture la variable globale au moment de la définition)
3. Placer `do_analytic_grad` avant `n_max_FORM`

**Solution retenue :** `do_analytic_grad=do_analytic_grad` — la variable globale est définie avant la `def`, donc Python la capture correctement. L'argument est toujours passé explicitement à chaque appel.

**Ligne corrigée :** 588.

---

### Problème 5 — `do_visu` : `hf_instance` renommé en `hf_cache` (NameError)

**Symptôme :** `NameError: name 'hf_instance' is not defined`.

**Cause :** La classe `HFCache` a été renommée et son instance s'appelle `hf_cache`, mais les références dans `do_visu` n'ont pas été mises à jour.

**Solution retenue :** Remplacer `hf_instance._last_g` et `hf_instance._last_grad` par `hf_cache._last_g` et `hf_cache._last_grad`.

**Lignes corrigées :** 825, 826.

---

### Problème 6 — `init_GP` appelé sans `U_doe_fixed` ni `SOL_calc` → double calcul HF (15 appels inutiles)

**Symptôme :** ETAPE 0 calcule déjà `SOL_U` via `fill_sol`, puis `init_GP` est appelé sans ces données → régénère un nouveau DOE aléatoire + 15 nouveaux appels HF.

**Cause :** `xt, yt, all_grad = init_GP(modelname, params_names, n0)` sans arguments optionnels.

**Solutions proposées :**
1. `init_GP(..., U_doe_fixed=U_doe, SOL_calc=SOL_U)` pour éviter tout recalcul
2. Fusionner ETAPE 0 et ETAPE 3.2 en un seul appel `init_GP`

**Solution retenue :** Refactoring plus profond (voir section 3).

---

### Problème 7 — Warm start : `init_GP` sans `U_doe_fixed` → DOE étendu perdu

**Symptôme :** Après `U_doe.add(U_warm)`, le warm start appelle `init_GP` sans `U_doe_fixed` → génère un nouveau DOE aléatoire de n0 points, perdant le point `U_warm` ajouté.

**Cause :** `xt, yt, all_grad = init_GP(modelname, params_names, n0)` — `U_doe` (étendu) non passé.

**Solution retenue :** À faire (warm start non encore corrigé en fin de session) :
```python
U_doe.add(U_warm)
xt, yt, all_grad = init_GP(modelname, params_names, U_doe.getSize(), U_doe_fixed=U_doe)
```

---

### Problème 8 — `sm` jamais créé avant le premier appel GEK

**Symptôme :** `NameError: name 'sm' is not defined` si `do_GEK=True` et validation gradient active.

**Cause :** `g_GEK` et `grad_g_GEK` appellent `build_metamodel_GEK` qui utilise `sm` comme variable libre, mais `sm = train_GEK(...)` n'était pas appelé avant la validation gradient ni avant FORM.

**Solution retenue :** `sm = train_GEK(xt, yt, all_grad, reduc_PLS=reduc_PLS)` ajouté ligne 634 avant FORM, et ligne 748 avant validation gradient (après FORM). La validation gradient a été déplacée après l'ETAPE 4 (après β/Pf).

---

### Problème 9 — `tirage_DOE` appelé puis immédiatement écrasé (appel inutile)

**Symptôme :** `U_doe = tirage_DOE(...)` générait un DOE aléatoire immédiatement remplacé par le DOE fixé.

**Solutions proposées :**
1. Passer `U_doe_fixed` comme 4ème argument à `tirage_DOE`
2. Supprimer l'appel à `tirage_DOE` et utiliser `U_doe_fixed` directement
3. Laisser `tirage_DOE` en 3 paramètres et ne l'appeler que si pas de DOE fixé

**Solution retenue :** Revert — `tirage_DOE` reste une fonction pure à 3 paramètres. La logique "fixé ou aléatoire" est entièrement gérée par `init_GP` via `U_doe_fixed=None`.

---

### Problème 10 — `g_ref`, `dg_adj_fc`, `dg_adj_fy` non définis après refactoring ETAPE 0

**Symptôme :** Prints lignes 347-350 referencent `g_ref`, `dg_adj_fc`, `dg_adj_fy` qui n'existent plus après le refactoring vers `init_GP`.

**Cause :** Ces variables étaient extraites de `SOL_U` dans l'ancienne ETAPE 0 ; elles ne sont plus définies dans la nouvelle architecture.

**Solution :** Non appliquée en fin de session — à corriger en reconstruisant depuis `yt` (si `do_pce=False`) ou en faisant retourner `U_doe` par `init_GP` et en recréant `g_ref = [SOL_U[i]['g'] for i in range(n0)]`.

---

### Problème 11 — `fill_sol` rappelé après `init_GP` (double calcul HF résiduel)

**Symptôme :** Ligne 335 : `SOL_U = fill_sol(...)` appelé alors que `init_GP` (avec `SOL_calc=None`) a déjà lancé `fill_sol` → 30 appels HF au lieu de 15.

**Cause :** Refactoring incomplet — ancienne logique non supprimée.

**Solution :** Non appliquée en fin de session.

---

## 3. Modifications effectuées au code

### 3a. `run_HF` — signature inchangée, appels mis à jour

`run_HF` retourne `(g_HF, grad_HF_U, grad_HF_X)`. Tous les appels mis à jour :
```python
# HFCache (ligne 563)
self._last_g, self._last_grad, _ = run_HF(modelname, u, params_names, T_inv)

# do_linear_test (ligne 736)
g0, grad_U_0, _ = run_HF(modelname, u0, params_names, T_inv, sensitivity=True)

# do_GP_test GEK (ligne 801)
g_HF, _, _ = run_HF(modelname, U_res, params_names, T_inv, sensitivity=False)

# do_GP_test KRG (ligne 817)
g_HF, _, _ = run_HF(modelname, U_res, params_names, T_inv, sensitivity=False)

# do_visu (ligne 836)
g_val, _, _ = run_HF(modelname, u_scan, params_names, T_inv, sensitivity=False)
```

### 3b. `train_GEK` — correction scope

```python
# AVANT
n_var = inputSample.getDimension()  # NameError si try_pce=False

# APRÈS
n_var = xt.shape[1]
```

### 3c. `FORM_GEK` — correction SyntaxError

```python
# AVANT
def FORM_GEK(dist_U, g_GEK, grad_g_GEK, n_max_FORM=n_max_FORM, tol_FORM=tol_FORM, do_analytic_grad):

# APRÈS
def FORM_GEK(dist_U, g_GEK, grad_g_GEK, n_max_FORM=n_max_FORM, tol_FORM=tol_FORM, do_analytic_grad=do_analytic_grad):
```

### 3d. `do_visu` — correction `hf_instance` → `hf_cache`

```python
# AVANT
g_ustar = hf_instance._last_g
grad_ustar_U = hf_instance._last_grad

# APRÈS
g_ustar = hf_cache._last_g
grad_ustar_U = hf_cache._last_grad
```

### 3e. Warm start KRG — correction évaluateur

```python
# AVANT
if do_warm_start and build_metamodel_KRG(U_warm)[0] > tol_warm_start:  # TypeError

# APRÈS
if do_warm_start and metamodel_KRG(U_warm)[0] > tol_warm_start:  # utilise objet OT stocké
```

### 3f. Test GP KRG — correction évaluateur

```python
# AVANT
g_GP = build_metamodel_KRG(U_res)[0]  # TypeError

# APRÈS
g_GP = metamodel_KRG(U_res)[0]
```

### 3g. `tirage_DOE` — pattern `if guard: return` (suppression `else:`)

```python
def tirage_DOE(modelname, params_names, n0):
    # 3 paramètres, pas de U_doe_fixed — logique "fixé ou aléatoire" dans init_GP
    dist = []
    ...
    return U_doe
```

### 3h. `init_GP` — ajout `SOL_calc=None`

```python
def init_GP(modelname, params_names, n0, U_doe_fixed=None, SOL_calc=None):
    U_doe = tirage_DOE(modelname, params_names, n0, U_doe_fixed)  # ← TEMPORAIRE, revert prévu
    if SOL_calc is not None:
        SOL_U = SOL_calc
    else:
        SOL_U = fill_sol(modelname, U_doe, params_names, T_inv, sensitivity=True)
    xt = np.array(U_doe)
    y_hf = np.array([SOL_U[i]['g'] for i in range(n0)]).reshape(-1, 1)
    yt = y_hf
    all_grad = np.zeros((n0, n_var))
    for i in range(n0):
        for j in range(n_var):
            all_grad[i][j] = SOL_U[i][f'dg_u{j+1}']
    if do_pce:
        yt -= y_pce
        all_grad -= all_grad_PCE
    return xt, yt, all_grad
```

**Note :** `tirage_DOE` rappelé avec 4 arguments (temporairement) — à revert. `n_var`, `do_pce`, `y_pce`, `all_grad_PCE` sont des variables libres du scope englobant.

### 3i. `fill_sol` — bug ligne 278 corrigé (session précédente, rappel)

```python
# AVANT
SOL_U['g'][i] = g_HF  # ← TypeError : SOL_U est une liste

# APRÈS
SOL_U[i]['g'] = g_HF
```

---

## 4. État du code en fin de session

### Corrigé
- `run_HF` dépackage à 3 valeurs : toutes les lignes ✓
- `build_metamodel_KRG` vs `metamodel_KRG` : évaluateurs corrigés ✓
- `train_GEK` : `xt.shape[1]` ✓
- `FORM_GEK` : `do_analytic_grad=do_analytic_grad` ✓
- `hf_cache` dans `do_visu` ✓

### Reste à faire
1. **Warm start `init_GP`** sans `U_doe_fixed` → DOE étendu perdu (lignes 640, 649)
2. **ETAPE 0 refactoring** : supprimer double appel `fill_sol` (ligne 335), recréer `g_ref`/`dg_adj_fc`/`dg_adj_fy`
3. **`init_GP` ligne 333** : retourne 4 valeurs (`all_sensib`) mais fonction en retourne 3 → crash
4. **Blocs à supprimer** : lignes 52–152 (`run_one_SOL`), 394–403 (ancien SOL), 863–901 (HFModel/HFGradient)
5. **`compute_q2_loo`** : toujours commentée dans le bloc `if try_pce:`
6. **`VALIDATION = {}`** inutilisé (ligne 450), `if try_pce:` redondant (ligne 457)

---

## 5. Blocs à supprimer — confirmés sans risque

| Lignes | Contenu |
|--------|---------|
| 52–152 | `run_one_SOL` commenté — remplacé par `fill_sol` + `run_HF` |
| 394–403 | Ancien bloc SOL commenté — remplacé dans ETAPE 0 |
| 863–901 | `HFModel`/`HFGradient` commentés dans `do_visu` — approche abandonnée |
