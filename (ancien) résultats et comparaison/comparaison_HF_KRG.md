# Comparaison FORM HF vs FORM KRG — Flexion pure BA (phi=16mm)

> **Objectif :** évaluer l'impact du nombre de points du DOE (n0) sur la précision du métamodèle KRG.
> **Référence :** résultat FORM HF (appels directs STRAINS).
> **g_HF(u\*)** : valeur HF au point de design trouvé ; **g_KRG(u\*)** : valeur KRG au même point (≈ 0 par construction).
> **Erreur β** : |β_modèle − β_HF| / β_HF.

---

## Tableau 1 — F = 0.235 MN (β_HF ≈ 0.95)

> ³ GEK étape 1 (Cobyla + FD, squar_exp, n0=25) : Cobyla arrêté à MaximumConstraintError=1e-2 sans atteindre g=0. u* n'est PAS sur la surface limite (g_HF(u*)=0.01 > 0). β=0.686 non physique. La précision g_GEK/g_HF=0.17% traduit uniquement la qualité du DOE (u* proche de l'origine, bien couvert par les 25 pts LHS) — pas la qualité intrinsèque du noyau squar_exp. Solver à remplacer par AbdoRackwitz (étape 2).

| | **HF** | **KRG n0=25** | **KRG n0=10** | **KRG n0=5** | **GEK n0=25 étape1** ³ |
|---|---|---|---|---|---|
| u* | [-0.159, -0.938] | [-0.1595, -0.9398] | [-0.1604, -0.9314] | [-0.1033, -0.8557] | [-0.125, -0.675] |
| β | 0.952 | 0.9532 | 0.9451 | 0.8620 | 0.686 |
| g_HF(u*) | ≈ 0 | -6.1e-05 | +2.44e-04 | +3.42e-03 | **+9.99e-03** |
| g_métamodèle(u*) | ≈ 0 | -9.0e-06 | +1.0e-06 | ≈ 0 | +1.00e-02 |
| Erreur g_méta/g_HF | — | ~85% (non sign.) | — | — | **0.17%** (non sign.) ³ |
| Erreur relative β | 0% (réf.) | 0.1% | 0.72% | 9.5% | 28% |
| n_iter FORM | 1 | 11 | 11 | 12 | 0 |

---

## Tableau 2 — F = 0.225 MN (β_HF ≈ 2.08)

| | **HF** | **KRG n0=25** | **KRG n0=16** | **KRG n0=8** |
|---|---|---|---|---|
| u* | [-0.346, -2.056] | [-0.3471, -2.0698] | [-0.3545, -2.0376] | [-0.7057, -2.1580] |
| β | 2.084 | 2.0987 | 2.0682 | 2.2704 |
| g_HF(u*) | ≈ 0 | -5.58e-04 | +6.43e-04 | -5.93e-03 |
| g_KRG(u*) | ≈ 0 | ≈ 0 | ≈ 0 | +1.0e-06 |
| Erreur relative β | 0% (réf.) | 0.72% | 0.77% | 8.9% |
| n_iter FORM | 1 | 14 | 14 | 16 |

---

## Tableau 3 — F = 0.210 MN (β_HF ≈ 3.78)

> n0=15 : tolérance relâchée (MaximumConstraintError=1e-2), gradient nul au point trouvé — résultat sans signification physique.

| | **HF** | **KRG n0=60** | **KRG n0=40** | **KRG n0=25** | **KRG n0=20** | **KRG n0=15** ¹ | **KRG n0=15+1** ² |
|---|---|---|---|---|---|---|---|
| u* | [-0.526, -3.747] | [-0.8350, -4.0249] | [-0.7637, -4.1931] | [-0.8966, -4.4234] | [-0.7820, -4.8545] | [-1.6540, -5.4416] | [-0.7364, -3.6886] |
| β | 3.784 | 4.1106 | 4.2621 | 4.5134 | 4.9171 | 5.6874 | **3.7614** |
| Pf | 7.73e-05 | 1.97e-05 | 1.01e-05 | 3.19e-06 | 4.39e-07 | 6.45e-09 | **8.45e-05** |
| g_HF(u*) | ≈ 0 | -1.35e-02 | -2.00e-02 | -3.04e-02 | -4.76e-02 | -7.76e-02 | **+1.24e-03** |
| g_KRG(u*) | ≈ 0 | ≈ 0 | +9.0e-06 | ≈ 0 | +1.0e-06 | +8.3e-03 ¹ | +3.4e-04 |
| Erreur relative β | 0% (réf.) | 8.6% | 12.6% | 19.3% | 29.9% | 50.3% | **0.6%** |
| n_iter FORM | 21 | 19 | 19 | 21 | 23 | 51 | **2** |

¹ Tolérance relâchée (MaximumConstraintError=1e-2) — gradient nul, résultat non physique.  
² DOE LHS 15 pts + u*_n15 comme point d'enrichissement, warm start depuis u*_n15.
