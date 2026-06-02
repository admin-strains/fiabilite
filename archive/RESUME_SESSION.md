# Résumé de session — Fiabilité flexion pure BA

## Objectif
Workflow de fiabilité (FORM sur surrogate) pour un bloc BA en flexion pure.
Variables aléatoires : `fc` (résistance compression), `fy` (limite élastique armatures).

---

## Bug critique résolu : `dg/dfy = 0`

**Cause** : mauvaise clé dans `sensitivity_regions` pour les armatures.

```python
# AVANT (faux — solids ne s'applique pas aux rebars)
{"param": "YIELD_STRENGTH", "solids": ["HA1", "HA2", "HA3", "HA4"]}

# APRÈS (correct)
{"param": "YIELD_STRENGTH", "rebars": ["HA1", "HA2", "HA3", "HA4"]}
```

**Preuve** : `C:\workspace\front\04_TESTS\Python\test_Sensitivity.py` ligne 323 utilise `"rebars"` pour les armatures.  
**Résultat** : `dg_adj_fy ≈ 0.000289` (non nul, confirmé en post-traitement).

---

## Architecture du workflow (`AC_pure_flexion.py`)

```
ETAPE 0  : Config + transformation isoprobabiliste X → U
ETAPE 1  : DOE LHS (n0=25) dans U, appels HF STRAINS avec sensibilité adjointe
ETAPE 3.1: PCE (LARS) — désactivé (do_pce=False) — instable avec OT 1.26 + LARS
ETAPE 3.2: Surrogate KRG (OT KrigingAlgorithm) ou GEK (SMT GEKPLS)
ETAPE 4  : FORM sur le surrogate
```

---

## Choix du surrogate : OT KrigingAlgorithm (pas GEKPLS)

| | GEKPLS (SMT) | KrigingAlgorithm (OT) |
|---|---|---|
| Gradient analytique pour FORM | Non (`predict_derivatives` n'existe pas) | Oui (via `getMetaModel()`) |
| Intégration OT native | Non — black box | Oui — `CompositeRandomVector` direct |
| Gradient interface OT | Cassé (Matrix 4×n au lieu de Point) | Natif |
| Gradients en entrée (GEK) | Oui (`set_training_derivatives`) | Non |

**Conclusion** : Pour FORM, `KrigingAlgorithm` OT est la seule option viable car il fournit un gradient analytique utilisable directement par `AbdoRackwitz`.

---

## Calibration de la charge (`dsLoad.txt`)

| `Z` (charge) | Comportement | Décision |
|---|---|---|
| `-1.0` | Tous `g < 0` → Pf=1 | Trop fort |
| `-0.1` | `g_moy ≈ 1.5`, beta ≈ 16 | Trop faible |
| `-0.22` | beta ≈ 3.3, hors LHS (±2.58σ) | Trop faible |
| `-0.235` | g straddlent 0 (4 pts < 0) | **Actuel** |

---

## État actuel du code

- `dsLoad.txt` : `Z='-0.235'`
- `do_GEK = False` → utilise `OT KrigingAlgorithm`
- `do_pce = False`
- Échelle covariance : `[1.0] * n_var` (corrigé, était `[1e-2]`)
- Solveur FORM branche KRG : `AbdoRackwitz` (corrigé, était `Cobyla`)

---

## Problème restant

**FORM ne converge pas** : avec F=0.235 et Cobyla, le solver trouvait `min g = 0.0288` (pas sur la surface limite).  
Cause probable : soit le KRG n'interpolait pas correctement les pts < 0 (échelle 1e-2 trop petite), soit Cobyla sans gradient diverge.  
**Fix en cours** : échelle → 1.0 + AbdoRackwitz.

---

## Problème PCE (non bloquant)

`ot.FunctionalChaosValidation(result).computeR2Score()` crash avec LARS (OT 1.26).  
Raison : LOO analytique suppose base fixe, mais LARS sélectionne des termes différents.  
**Contournement** : `do_pce = False` pour l'instant.

---

## Fichiers clés

| Fichier | Rôle |
|---|---|
| `C:\_workingDir\_SF\test flexion\AC_pure_flexion.py` | Script principal fiabilité |
| `C:\_workingDir\_SF\test flexion\launcher.py` | Lance AC_pure_flexion.py avec les bon paths DLL |
| `C:\workspace\storage\semia\test_pure_flexion.ds\dsCad.txt` | Géométrie + matériaux |
| `C:\workspace\storage\semia\test_pure_flexion.ds\dsLoad.txt` | Chargement (Z à calibrer) |
| `C:\_workingDir\_SF\fiabilite\config\jcss_fy.py` | Loi statistique fy |
| `C:\_workingDir\_SF\fiabilite\config\jcss_fc.py` | Loi statistique fc |

---

## Prochaines étapes

1. Vérifier que FORM converge avec AbdoRackwitz + échelle 1.0
2. Valider beta obtenu vs estimation analytique
3. Réactiver PCE (fix LOO : refitter OLS sur base LARS réduite)
4. Tester GEK path (`do_GEK=True`) une fois KRG validé
5. Ajouter boucle active learning (EFF) pour enrichissement près de la surface limite
6. Ajouter Importance Sampling
