# Resume session 24 avril 2026 — Refactoring FORM_multi_start

## Note generale — Nouveau dossier de resultats

A partir du 24/04, toutes les analyses de resultats repartent de zero.
Les nouveaux fichiers .md de resultats et comparaison sont enregistres dans :
`C:\_workingDir\_SF\test flexion\nouveau résultats et comparaison\`

Les anciens fichiers dans `résultats et comparaison\` ne sont pas modifies — ils servent de reference historique.

---

## Partie 1 — Refactoring FORM_multi_start

## Contexte
Refactoring de la fonction `FORM_multi_start` dans `AC_pure_flexion.py`.
Objectif : permettre la recherche de plusieurs modes de defaillance en relancant FORM depuis plusieurs points de depart du DOE, tries par norme decroissante.

---

## Version finale de la fonction (etat fin de session)

```python
def FORM_multi_start(result_modes, U_doe_multistart):
    norms = np.array([np.linalg.norm(np.array(u)) for u in U_doe_multistart])
    sorted_idx = np.argsort(norms)[::-1]
    U_doe_multistart = ot.Sample([U_doe_multistart[int(i)] for i in sorted_idx])
    for n_FORM in range(n_multi_start):
        solver.setStartingPoint(U_doe_multistart[n_FORM])
        algo = ot.FORM(solver, event)
        algo.run()
        u_new  = algo.getResult().getPhysicalSpaceDesignPoint()
        u_prev_list = [result.getPhysicalSpaceDesignPoint() for result in result_modes]
        if all((u_new - u_prev).norm() > tol_multistart for u_prev in u_prev_list):
            result_modes.append(algo.getResult())
            U_doe_multistart = ot.Sample(np.delete(np.array(U_doe_multistart), n_FORM, axis=0))
            break
    return result_modes, U_doe_multistart
```

Appelee dans le bloc appelant :
```python
if do_multi_start:
    U_doe_multistart = ot.Sample(xt)
    result_modes = [result]
    while U_doe_multistart.getSize() > 0:
        result_modes, U_doe_multistart = FORM_multi_start(result_modes, U_doe_multistart)
    if len(result_modes) > 1:
        print(f'Il y a plusieurs modes de defaillances.')
        i = 1
        for result in result_modes:
            n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM = resultats_GP(modelname, params_names, result_modes[i], metamodel)
            print(f'\nRESULTATS DU MODE {i+1}')
            print_resultats(n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM)
```

---

## Problemes identifies (non encore corriges)

### P1 — Boucle infinie (BLOQUANT)
**Description :** Si aucun des `n_multi_start` points du `for` ne mene a un nouveau mode (tous convergent vers un mode deja connu), le `for` se termine sans `break` et `U_doe_multistart` est retourne inchange → `getSize()` ne diminue jamais → boucle `while` infinie.

**Fix propose :**
```python
        for n_FORM in range(n_multi_start):
            ...
            if all((u_new - u_prev).norm() > tol_multistart for u_prev in u_prev_list):
                result_modes.append(algo.getResult())
                U_doe_multistart = ot.Sample(np.delete(np.array(U_doe_multistart), n_FORM, axis=0))
                break
        else:  # for termine sans break = aucun nouveau mode trouve parmi les n_multi_start points
            n_del = min(n_multi_start, U_doe_multistart.getSize())
            U_doe_multistart = ot.Sample(np.delete(np.array(U_doe_multistart), list(range(n_del)), axis=0))
```

### P3 — `result` du scope externe (fragile)
**Description :** `result_modes = [result]` utilise `result` defini dans le scope externe (le dernier resultat FORM principal). Si la fonction est appelee avant qu'un `result` existe → `NameError`.

**Fix propose :** Passer `result` en parametre du bloc appelant :
```python
result_modes = [result]  # s'assurer que result est defini juste avant ce bloc
```
Ou initialiser `result_modes` en dehors du `if do_multi_start:` juste apres le premier appel FORM.

### P4 — `i` non incremente dans la boucle d'affichage (BUG LOGIQUE)
**Description :** `i = 1` est fixe, jamais incremente. `result_modes[i]` affiche toujours le mode 2 (index 1), jamais les autres modes.

**Fix propose :** Utiliser `enumerate` :
```python
        for i, res in enumerate(result_modes):
            n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM = resultats_GP(modelname, params_names, res, metamodel)
            print(f'\nRESULTATS DU MODE {i+1}')
            print_resultats(n_iter, U_res, X_res, g_GP_res, grad_res, importance, beta, Pf_FORM)
```

### P5 — Variable de boucle `result` ecrase le scope externe
**Description :** `for result in result_modes` utilise `result` comme variable de boucle, ce qui ecrase le `result` du scope externe (utilise en P3 pour initialiser `result_modes`).

**Fix :** Renomme la variable de boucle, par exemple `for res in result_modes:` (deja corrige dans le fix P4 avec `enumerate`).

---

## Notes complementaires

- `solver` et `event` utilises dans `FORM_multi_start` viennent du scope externe — coherent avec l'architecture actuelle (fonctions imbriquees dans `__main__`), mais couplage fort.
- `tol_multistart` doit etre ajoute dans les options utilisateur (lignes ~199-231), exemple : `tol_multistart = 1e-3`.
- Le re-tri de `U_doe_multistart` a chaque iteration `while` est inefficace (le sample est deja trie) mais non bloquant.
- `ot.Sample` n'a pas de methode de suppression directe — utiliser `np.delete` ou list comprehension.
- Pour creer un `ot.Sample` vide par defaut : `ot.Sample(0, 0)` (suffisant pour le test `getSize() > 0`).

---

## Partie 2 — Analyse GEK pur : non-determinisme et comparaison ancienne/nouvelle version

### Runs effectues (GEK pur, F=0.210 MN, n0=15, do_warm_start=False, do_analytic_grad=True)

| Output | Version code | DOE | beta | Erreur vs HF | g_meta(u*) |
|---|---|---|---|---|---|
| output_2404_1050 | Nouvelle | LHS OT (=DOE fixe en pratique) | 3.509 | -7.3% | 0.015 |
| output_2404_1111 | Nouvelle | LHS OT (idem) | 2.154 | -43.1% | 0.070 |
| output_2404_1134 | Nouvelle | LHS OT (idem) | 1.487 | -60.7% | 0.098 |
| output_2404_1205 | **Ancienne** | DOE hardcode (memes 15 points) | **3.774** | **-0.26%** | **~0** |

### Observation cle : graine OT fixe

OpenTURNS initialise son PRNG global avec la **meme graine fixe a chaque demarrage de processus Python**. Donc `tirage_DOE` (LHS + SimulatedAnnealing) produit toujours la meme sequence, que `U_doe_fixed=None` ou non. Les trois runs "aleatoires" de la nouvelle version ont tous utilise le meme DOE (verifie en comparant les valeurs imprimees par `print_DOE`).

**Consequence :** avec ou sans `U_doe_fixed`, le comportement est identique en l'absence de re-seeding. Pour un vrai DOE aleatoire : `ot.RandomGenerator.SetSeed(int(time.time()))` avant `tirage_DOE`.

### Observation cle : non-determinisme de GEKPLS

Meme DOE → meme donnees d'entrainement → GEKPLS donne des hyperparametres theta differents a chaque entrainement (non-determinisme de l'optimisation L-BFGS-B : parallelisme BLAS, etat interne). Les metamodeles obtenus sont visuellement identiques (courbes g=0 superposees dans la visu), mais le **champ de gradient** differe suffisamment pour que FORM (gradient-based) converge vers des points tres differents. Betas observes : 3.509, 2.154, 1.487 sur le meme DOE.

### Comparaison ancienne vs nouvelle version du code

La version ancienne (structure pre-refactoring) donne beta=3.774 (-0.26%) tres stable sur le meme DOE. Differences identifiees :

| Element | Ancienne version | Nouvelle version |
|---|---|---|
| `run_HF` signature | `(modelname, u, params_names, T_inv)` | `(modelname, params_names, u)` |
| `run_HF` retour | `(g_HF, grad_HF_U)` — 2 valeurs | `(g_HF, grad_HF_U, grad_HF_X)` — 3 valeurs |
| `T_inv` | Calcule une fois globalement dans `__main__` | Recalcule dans chaque fonction |
| `metamodel_GEK` | Closure locale `def metamodel_GEK(u, do_pce=...)` | Closure `metamodel` retournee par `build_metamodel_total` |
| `grad_g_GEK` | Retourne `grad` (shape `(n_var,1)`) | Retourne `grad.T` (shape `(1,n_var)`) |
| DOE | Hardcode directement (LHS commente) | `U_doe_fixed` parametre + `tirage_DOE` |
| Structure | Monolithique dans `__main__` | Refactoring en fonctions separees |
| `do_linear_test` | Bloc duplique (appele deux fois) | Une seule fois via `print_GP_tests` |
| `do_visu` | False (ancienne visu HF uniquement) | True, avec comparaison HF+GP |

**Hypothese sur la difference de performance :** le `grad_g_GEK` de l'ancienne version retourne `grad` (shape `(n_var,1)`) au lieu de `(1,n_var)` attendu par OT PythonFunction. OT pourrait utiliser les differences finies (FD) en fallback si la shape est incorrecte — ce qui donnerait un gradient plus stable. **Non confirme** : tester avec `do_analytic_grad=False` dans la nouvelle version pour verifier.

### Bug corrige dans l'ancienne version pour la faire tourner

Ligne 834 : `float(metamodel_GEK(np.array(U_res).reshape(1, -1))[0, 0])` → `TypeError` car `metamodel_GEK` retourne un tuple `(val, grad)` et `tuple[0,0]` est invalide.
Fix applique : `g_GEK_star, _ = metamodel_GEK(U_res)` (et idem ligne 841 pour `g_meta_ref`).
