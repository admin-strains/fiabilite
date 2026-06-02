# Résumé global de session — Fiabilité flexion pure BA
**Couvre :** Sessions du 16/04 au 23/04/2026 + état du code actuel

---

## 1. Contexte du problème

Workflow de fiabilité FORM sur surrogate pour une poutre BA en flexion pure (analyse limite cinématique STRAINS).

- **Variables aléatoires :** `fc` (béton C35, loi JCSS log-normale, moy≈31.67 MPa) et `fy` (acier phi=16mm, fy_nom=500 MPa, loi JCSS normale)
- **Fonction de performance :** `g = α+ − 1` où α+ = multiplicateur de charge limite kinématique STRAINS
- **Charge calibrée :** F=0.210 MN → β_HF = **3.784**, Pf = 7.73e-05 (référence absolue)
- **DOE fixé :** n0=15 points U-space (lignes 235-251 dans AC_pure_flexion.py)

**Fichiers modèle STRAINS :**
- `C:\workspace\storage\semia\test_pure_flexion.ds\dsLoad.txt` — charge (Z='-0.210' actuellement)
- `C:\workspace\storage\semia\test_pure_flexion.ds\dsCad.txt` — géométrie + matériaux
- `C:\_workingDir\_SF\fiabilite\config\jcss_fy.py` — loi fy(d, fy_nominal)
- `C:\_workingDir\_SF\fiabilite\config\jcss_fc.py` — loi fc("C35", t, tau)

---

## 2. Architecture du code (état actuel)

**Fichiers :**
- `AC_pure_flexion.py` — script principal, toutes branches métamodèle
- `launcher.py` — à toujours utiliser pour lancer (configure DLL STRAINS avant import)

**Branches contrôlées par 2 flags (lignes 646-803) :**

| `do_GEK` | `try_pce` | Branche active | Métamodèle |
|---|---|---|---|
| False | False | **KRG pur** | KRG(g_HF) |
| False | True | **PCE-KRG** | KRG(résidus PCE) + PCE |
| True | False | **GEK pur** | GEK(g_HF) |
| True | True | **GEPCK** | GEK(résidus PCE) + PCE |

**État actuel des options (lignes 199-231) :**
- `do_GEK=True, try_pce=True` → branche **GEPCK** active
- `do_warm_start=True`, `tol_warm_start=1e-4`
- `n0=15`, DOE fixé (U_doe_fixed hardcodé)
- `n_max_FORM=50`, `tol_FORM=0.2` (tolérance relâchée)
- `do_visu=True` → 64 appels HF supplémentaires (8×8 grid)
- `do_GP_linear_test=True`, `do_GP_HF_test=True`

**Pipeline complet :**
```
init_GP(n0, U_doe_fixed)          → (xt, y_hf, all_grad_hf, all_sensib_hf)  [n0 appels STRAINS]
build_metamodel_PCE(xt, y_hf)     → metamodel_PCE  [si try_pce]
fill_PCE(xt, metamodel_PCE)       → (y_PCE, all_grad_PCE, all_sensib_PCE)
fill_inputGP(...)                  → (yt=y_hf−y_PCE, all_grad, all_sensib)   [.copy() anti-aliasing]
build_metamodel_KRG(xt, yt)       → metamodel_KRG  [si KRG/PCE-KRG]
build_metamodel_GEK(xt, yt, all_grad) → sm (GEKPLS)  [si GEK/GEPCK]
build_metamodel_total(sm, metamodel_PCE=None) → metamodel (closure (val,grad))
FORM_{KRG|GEK}(metamodel, start_point) → result
[si warm_start et metamodel(U_warm)[0] > tol] → 1 appel STRAINS + rebuild métamodèle
resultats_GP + print_resultats + print_GP_tests + print_visu_HF_GP
```

**Retours run_HF :** `(g_HF: float, grad_HF_U: ot.Point, grad_HF_X: list)` — toujours dépackager à 3 valeurs.

---

## 3. Historique des sessions

### Session 16/04 — Mise en place FORM HF
- **Bug critique résolu :** `dg/dfy = 0` → mauvaise clé `"solids"` pour armatures → corrigé en `"rebars"` (ligne 148 AC_pure_flexion.py)
- Calibration de la charge : F=0.235 MN → β=16 (trop fort) → F=0.60 MN → β=5.74 → F=0.210 MN → β=3.784
- Architecture HFCache + AbdoRackwitz validée
- Test linéarisation FOSM : g quasi-linéaire (erreur FOSM < 0.5%)

### Session 17/04 après-midi — Débogage FORM HF
- `run_HF` retourne maintenant 3 valeurs : `(g_HF, grad_HF_U, grad_HF_X)`
- Garde-fou : ValueError si `grad_HF_U` contient None

### Session 20/04 — FORM KRG + impact DOE
- KRG pur validé sur F=0.235 (β_HF≈0.95) : excellent avec n0≥25, bon avec n0=16, dégradé avec n0=8
- Sur F=0.210 (β_HF≈3.78) : KRG nécessite n0=60 pour erreur 8.6%, n0=15 → erreur 50%
- Stratégie enrichissement n0=15+1 point → erreur 0.6% (proof of concept warm start)
- Résultats complets : `comparaison_HF_KRG.md`, `resultats_KRG_run2.md`

### Session 21/04 matin — PCE (LARS, théorie, construction)
- PCE construit avec `FunctionalChaosAlgorithm` (LARS + CorrectedLeaveOneOut)
- **Bug bloquant :** `FunctionalChaosValidation` crash en OT 1.26 avec LARS quelle que soit l'option (check C++ `involvesModelSelection()`)
- Solution identifiée : LOO manuel `compute_q2_loo` (double CV non biaisé)
- Séparation `try_pce` (intention) vs `do_pce` (résultat après validation)

### Session 21/04 après-midi — Fix LOO
- `compute_q2_loo` implémentée via `ot.MetaModelValidation`
- Segfault lors de `algo_KRG.run()` avec yt résiduel ~1e-4 → contourné par `if True:` workaround
- **Premier run PCE+KRG réussi :** β=3.790 (cohérent avec β_HF=3.784, erreur 0.16%)

### Session 22/04 — Refactoring + audit GEK
- Refactoring complet en fonctions : `init_GP`, `fill_sol`, `tirage_DOE`, `build_metamodel_*`, `FORM_*`, `resultats_GP`, `print_*`
- Nombreux bugs corrigés : aliasing numpy (.copy()), n_var locaux, évaluateur vs builder KRG, dépackage run_HF, SyntaxError FORM_GEK, hf_cache renommage
- Warm start PCE-KRG et KRG pur : U_doe reconstruit depuis xt, np.array() wrapping

### Session 23/04 matin — Audit PCE-KRG + KRG pur
- `fill_PCE` corrigée : retourne all_sensib_PCE, T_inv local, y_PCE majuscule
- `fill_inputGP` : .copy() sur les 3 variables (bug aliasing critique)
- `FORM_KRG` : metamodel (pas metamodel_KRG), start_point paramètre obligatoire
- **Run KRG pur run3 (DOE fixé n0=15) :** β=5.065 (+33.9% vs HF), g_HF(u*)=-0.050

### Session 23/04 après-midi — Audit GEPCK + run GEK pur
- `build_metamodel_total` corrigée : n_var local, signature (sm, metamodel_PCE=None), 1 seul return
- `GP_HF_test` : g_GP → g_GP_res
- **Run GEK pur run1 (DOE fixé n0=15, do_warm_start=False) :** β=2.118 (-44% vs HF), g_GP_res=0.074 (FORM non convergé sur surface limite), instabilité globale GEK avec n0=15

---

## 4. Résultats (F=0.210 MN, n0=15, DOE fixé)

| Méthode | β | Erreur β vs HF | g_meta(u*) | g_HF(u*) | n_iter |
|---|---|---|---|---|---|
| **HF référence** | **3.784** | 0% | ≈0 | ≈0 | 21 |
| PCE-KRG (session 2204) | 3.790 | +0.16% | ≈0 | ? | 15 |
| PCE-KRG (DOE fixé) | **3.779** | **−0.1%** | ≈0 | N/A | 15 |
| KRG pur (run3, DOE fixé) | 5.065 | +33.9% | +9.3e-05 | −4.96e-02 | 24 |
| GEK pur (run1, no WS) | 2.118 | −44.0% | +0.074 | ~+0.074 | 37 |
| GEK+WS | 1.914 | −49.4% | +0.113 | +0.113 | 1 |
| PCE-KRG+WS | 1.644 | −56.6% | +0.091 | +0.091 | 1 |
| GEPCK+WS | 0.991 | −73.8% | +0.119 | +0.119 | 1 |

**Référence HF pour d'autres charges (phi=16mm) :** `resultats_HF_run2.md`

---

## 5. Problèmes actifs

### Bug A — RuntimeError FORM (BLOQUANT, dernier run)
```
RuntimeError: Obtained design point is not on the limit state:
image = 5.18173e-05, incompatible with threshold 0, tolerance 2e-05
```
`tol_FORM=0.2` → solver converge à g≈0.2, mais OT vérifie le résultat final avec tolérance interne 2e-05. `setCheckStatus(False)` est sur le solver, pas sur ce check final OT.
**Fix :** entourer `algo.run()` d'un `try/except RuntimeError` dans `FORM_KRG` et `FORM_GEK`.

### Bug B — compute_q2_loo définie mais non appelée
`do_pce = try_pce` hardcodé → PCE toujours appliqué sans validation Q2. `compute_q2_loo` est dans le code mais les appels sont commentés. Non bloquant pour les runs actuels.

### Bug C — `metamodel_pce` → `metamodel_PCE` dans build_metamodel_GEK (lignes 369-370)
Potentiellement bloquant si do_pce=True + do_GEK=True (branche GEPCK). A vérifier.

### Bug D — Branche HF (`else:`) : `mode_number_goal` non défini
NameError si do_GP=False. Non bloquant car do_GP=True dans tous les tests actuels.

### Observation — do_visu=True = 64 appels STRAINS supplémentaires
8×8 grid autour de u* → très lent. Mettre do_visu=False pendant les tests.

---

## 6. Ce qui reste à tester

Objectif : tester les 4 branches × 2 (avec/sans warm start) à F=0.210 MN, n0=15 DOE fixé.

**Avant tout :** corriger Bug A (try/except RuntimeError) + mettre do_visu=False.

**Ordre :** KRG pur → PCE-KRG → GEK pur → GEPCK (du plus stable au moins stable).

Pour chaque run, noter : β, Pf, u*, n_iter, g_meta(u*), g_HF(u*), warm_start déclenché O/N.

---

## 7. Points techniques à mémoriser

- **Espaces :** tout en espace U (N(0,1)). `result.getPhysicalSpaceDesignPoint()` = u*, pas x*.
- **Gradient GEK pour OT :** la closure `build_metamodel_total` retourne `(val:float, grad:np(nv,1))`. Dans `FORM_GEK`, `grad_g_GEK(u)` fait `return grad.T` → shape `(1,nv)` attendu par OT PythonFunction.
- **Aliasing numpy :** `yt = y_hf` sans `.copy()` → modifications de yt modifient y_hf → corrigé dans fill_inputGP.
- **do_analytic_grad=False (défaut GEK) :** FORM utilise différences finies → 3 appels métamodèle/itération. Mettre `True` pour le gradient analytique de la closure.
- **LARS + OT 1.26 :** `FunctionalChaosValidation` incontournable (crash C++). Seule solution : LOO manuel `compute_q2_loo`.
- **Encodage cp1252 :** pas de caractères Unicode non-ASCII dans les commentaires Python.
- **Warm start :** détecte seulement `g_meta(u*)>tol`, pas une convergence globalement fausse du métamodèle.

---

## 8. Références résumés de session

| Fichier | Date | Points clés |
|---|---|---|
| `RESUME_SESSION.md` | ~16/04 | Bug rebars, calibration charge, architecture initiale |
| `session_resume.md` | 16/04 | FORM HF, multi-start, syntaxe OT |
| `resume_session_1704_aprem.md` | 17/04 | HFCache, garde-fou grad_HF_U |
| `resume_session_2004.md` | 20/04 | KRG impact DOE, warm start proof of concept |
| `resume_session_2104.md` | 21/04 | PCE théorie complète, bug FunctionalChaosValidation, fix LOO |
| `resume_session_2204.md` | 22/04 | Refactoring fonctions, audit GEK, bugs aliasing/builder/dépackage |
| `resume_session_2304.md` | 23/04 | Run KRG/GEK pur, build_metamodel_total, audit GEPCK |
