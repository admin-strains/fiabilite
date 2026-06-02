# Comparaison KRG Matérn 5/2 vs KRG SquaredExponential — Flexion pure BA (phi=16mm)

> **Objectif :** évaluer l'impact du noyau de covariance sur la précision du FORM KRG.
> **Noyaux comparés :** Matérn 5/2 (`ot.MaternModel([1.0]*n_var, 2.5)`) vs Squared Exponential (`ot.SquaredExponential([1.0]*n_var, [1.0])`)
> **Référence :** résultat FORM HF (appels directs STRAINS).
> **g_HF(u\*)** : valeur HF au point de design trouvé ; **g_KRG(u\*)** : valeur KRG au même point (≈ 0 par construction FORM).
> **Erreur β** : |β_modèle − β_HF| / β_HF.
> **Motivation :** le GEK (GEKPLS SMT) utilise `squar_exp` (gaussien C∞). On teste si le noyau seul explique les différences de résultats observées entre KRG et GEK.

---

## Tableau 1 — F = 0.235 MN (β_HF ≈ 0.95)

| | **HF** | **KRG Matérn n0=25** | **KRG squar_exp n0=25** |
|---|---|---|---|
| u* | [-0.159, -0.938] | [-0.1595, -0.9398] | [-0.1604, -0.9400] |
| β | 0.952 | 0.9532 | 0.9536 |
| g_HF(u*) | ≈ 0 | -6.1e-05 | -7.6e-05 |
| g_KRG(u*) | ≈ 0 | -9.0e-06 | -5.0e-06 |
| Erreur relative β | 0% (réf.) | 0.1% | 0.16% |
| n_iter FORM | 1 | 11 | 11 |

**Conclusion F=0.235, n0=25 :** résultats quasi-identiques (écart β : 0.04 pt). DOE différents entre les deux runs — comparaison non rigoureuse.

---

## Tableau 2 — F = 0.235 MN, n0=10, **DOE identique fixé**

> DOE fixé (hardcodé) pour comparer uniquement l'effet du noyau, toutes choses égales par ailleurs.
> DOE U : [[-0.377, 1.482], [-1.946, 0.810], [0.138, -0.461], [0.445, 0.890], [0.853, 0.131], [-0.582, -1.169], [-0.188, 0.329], [-1.136, -0.246], [1.490, -0.657], [0.536, -2.088]]

| | **HF** | **KRG squar_exp n0=10** | **KRG Matérn 5/2 n0=10** |
|---|---|---|---|
| u* | [-0.159, -0.938] | [-0.156, -0.937] | [-0.160, -0.931] |
| β | 0.952 | 0.9492 | 0.9451 |
| g_HF(u*) | ≈ 0 | +9.3e-05 | +2.44e-04 |
| g_KRG(u*) | ≈ 0 | -2.0e-06 | +1.0e-06 |
| Erreur relative β | 0% (réf.) | **0.29%** | 0.73% |
| n_iter FORM | 1 | 11 | 11 |

**Conclusion DOE fixé :** sur le même jeu de données, squar_exp est légèrement plus précis que Matérn 5/2 (0.29% vs 0.73%). g_HF(u*) plus proche de 0 pour squar_exp → surface limite mieux localisée. La différence reste faible — les deux kernels sont adaptés pour ce β. Le noyau ne peut pas expliquer l'échec du GEK (β=0.69).

---

## Tableau 3 — F = 0.210 MN (β_HF ≈ 3.78), n0=20, **DOE identique fixé + warm start**

> DOE fixé (hardcodé, 20 pts) — même jeu de données pour les deux noyaux.
> Warm start activé dans les deux cas (tol_warm_start ajustée pour déclencher l'enrichissement).

| | **HF** | **KRG Matérn 5/2 n0=20+1** | **KRG squar_exp n0=20+1** |
|---|---|---|---|
| u* | [-0.526, -3.747] | [-0.643, -3.625] | [-0.538, -3.754] |
| β | 3.784 | 3.682 | **3.792** |
| g_HF(u*) | ≈ 0 | +4.5e-03 | -3.3e-04 |
| g_KRG(u*) | ≈ 0 | +2.5e-03 | +4.2e-03 |
| Erreur relative β | 0% (réf.) | 2.7% | **0.2%** |


---
