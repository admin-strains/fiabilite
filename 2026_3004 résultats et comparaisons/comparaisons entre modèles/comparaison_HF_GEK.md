# Comparaison HF vs GEK — Nouvelle géométrie — F=0.1 MN

**Géométrie :** b=0.5m, h=0.8m, L=5m, φ=16mm, 2 lits 3HA16, Block1 (ft1=0.1) + Block2 (ft2=3.5)
**Distributions JCSS :** fck=40 MPa, fyk=500 MPa

## F = 0.1 MN (beta_HF ≈ 0.39)

| Paramètre | HF référence | GEK n0=35 |
|---|---|---|
| Output | output_3004_1203.txt | output_3004_1447.txt |
| n points DOE | — | 35 |
| n iter FORM | 6 | 1 |
| n_appels HF (FORM) | 1 | 0 |
| fc* (MPa) | 47.8138 | 47.8323 |
| fy* (MPa) | 561.3999 | 560.9651 |
| u* [u_fc, u_fy] | [0.0016, 0.3915] | [0.0059, 0.3771] |
| dg/du_fc en u* | 0.000220 | 0.000226 |
| dg/du_fy en u* | 0.051833 | 0.051794 |
| Importance fc (%) | 0% | 0.02% |
| Importance fy (%) | 100% | 99.98% |
| β (FORM) | 0.3915 | 0.3771 |
| Pf (FORM) | 6.5229e-01 | 6.4696e-01 |
| g_meta(u*) | N/A | N/A |
| g_HF(u*) | ~0 | N/A |
| u* FOSM | [0.0015, 0.3933] | [0.0015, 0.3941] |
| Erreur FOSM | 0.46% | 4.65% |
| Écart β vs HF | — | -0.0144 (-3.68%) |
| do_warm_start | N | N |

## Observations

- **Accord très bon sur X-space :** fc* et fy* quasi identiques (écart <0.01% sur fc, <0.08% sur fy).
- **Accord bon sur gradients :** dg/du_fc et dg/du_fy identiques à 3% près.
- **Écart beta = -3.68% :** le GEK sous-estime légèrement beta (0.3771 vs 0.3915). Acceptable pour n0=35.
- **u* GEK légèrement éloigné du HF :** [0.0059, 0.3771] vs [0.0016, 0.3915] — même direction, norme légèrement inférieure.
- **Erreur FOSM GEK (4.65%) 10× celle du HF (0.46%) :** la surface de limite du GEK est légèrement plus courbée que la vraie.
- **Conclusion :** GEK n0=35 reproduit correctement le comportement global (fy dominant, beta ~0.39). La calibration de F reste la priorité (beta cible 3–4).
