# Tests amélioration KRG — 19/05/2026

**Config commune :** b=h=0.8m, 3 lits 8HA32, phi=32mm, F=0.74 MN, gamma_c=gamma_s=1.0, fcm=48 MPa, fym=550 MPa, cov_fc=0.12, n0=5, do_EFF=False, do_multistart=True, do_warmstart=False, tol_FORM=1.0.

**Objectif :** identifier quel levier KrigingAlgorithm corrige l'artefact rebound (île g<0 fermée au lieu d'une région semi-infinie).

**Référence HF** (sp=[0,0]) : beta=8.1235, u*=[-4.776, -6.571]

---

## Tableau de comparaison

| Métrique | HF ref (sp=[0,0]) | sans variante (1905) | v0 — Constant (1905) | v1 — Linear (1905) |
|---|---|---|---|---|
| Basis | — | ConstantBasis | ConstantBasis | **LinearBasis** |
| Noyau | — | SquaredExp | SquaredExp | SquaredExp |
| Bounds theta | — | non | non | non |
| MultiStart theta | — | non | non | non |
| n points DOE | — | 5 | 5 | 5 |
| theta_opt | — | non capturé | non capturé | non capturé |
| n_iter FORM | 7 (1205 ref) | 43 | — | **5** |
| fc* (MPa) | 32.83 (1205) | 36.33 | — | 28.83 |
| fy* (MPa) | 328.55 (1205) | 456.03 | — | 370.11 |
| u* [u_fc, u_fy] | [-4.776, -6.571] | [-2.271, -3.117] | [-2.27, -3.12] | **[-4.203, -5.967]** |
| Importance fc (%) | 15.3% (1205) | N/A | — | N/A |
| Importance fy (%) | 84.7% (1205) | N/A | — | N/A |
| dg/du_fc en u* | 0.025 (1205) | 0.047292 | — | 0.039345 |
| dg/du_fy en u* | 0.072 (1205) | 0.060260 | — | 0.059047 |
| **beta (FORM)** | **8.1235** | **3.8564** | **3.856** | **7.2981** |
| Pf (FORM) | — | 5.753e-05 | — | 1.459e-13 |
| u* FOSM (HF) | [-4.149, -6.377] | [-4.149, -6.377] | — | [-4.149, -6.377] |
| Erreur FOSM | — | 97.57% | — | **5.68%** |
| do_warmstart | N | N | N | N |
| Nb modes trouvés | — | 1 | 1 | 1 |
| Output | — | output_KRG_n5_baseline.txt | output_KRG_v0.txt | output_KRG_v1_linear.txt |
| **Écart vs HF** | — | **-52.5%** | **-52.5%** | **-10.2%** |

---

## Détail sans variante — ConstantBasisFactory, TNC seul (original)

- Output : `output_KRG_n5_baseline.txt`
- **beta = 3.8564**, Pf = 5.753e-05, u* = [-2.271, -3.117]
- n_iter FORM = 43 — toutes les descentes convergent vers le même u* (1 mode)
- fc* = 36.33 MPa, fy* = 456.03 MPa
- dg/du_fc = 0.047292, dg/du_fy = 0.060260
- u* FOSM = [-4.149, -6.377], Erreur FOSM = 97.57%
- theta_opt = non capturé (fonction originale sans print)
- Observation : rebound sévère, îlot g<0 fermé, u* faux

---

## Détail v0 — ConstantBasisFactory, TNC seul

- Output : `output_KRG_v0.txt`
- **beta = 3.856**, u* = [-2.27, -3.12]
- theta_opt = non capturé (buffer perdu à la mort du process)
- Identique à "sans variante" mathématiquement — même résultat confirmé
- Note instabilité : output_1905_0856 (même config, run précédent) avait donné beta=7.2981 → TNC sans bornes, résultats non reproductibles

---

## Détail v1 — LinearBasisFactory, TNC seul

- Output : `output_KRG_v1_linear.txt`
- **beta = 7.2981**, Pf = 1.459e-13, u* = [-4.203, -5.967]
- n_iter FORM = 5 (vs 43 pour ConstantBasis — 8x moins d'itérations)
- fc* = 28.83 MPa, fy* = 370.11 MPa
- dg/du_fc = 0.039345, dg/du_fy = 0.059047
- u* FOSM = [-4.149, -6.377], Erreur FOSM = 5.68% (vs 97.57% ConstantBasis)
- theta_opt = non capturé
- Écart vs HF = -10.2% (vs -52.5% ConstantBasis)
- **Conclusion : LinearBasisFactory corrige très significativement le rebound**

---

## Référence comparaison (session 13/05, fichier comparaison_GEK_KRG_1305.md)

| Run | n0 | beta | u* | Écart HF |
|---|---|---|---|---|
| KRG n0=10 (1305_1726) | 10+WS | 5.679 | [-4.055, -3.976] | -28.8% |
| KRG n0=18 (1305_1743) | 18+WS | 3.052 | [0.573, -2.997] | -61.8% |
| PCKRG n0=5+4EFF (1905_0901) | 9 | 7.9652 | [-6.104, -5.117] | -0.17% |

---

## Détail idée theta fixe — ConstantBasis + theta=10 gelé

- Output : `output_KRG_constant_theta10.txt`
- theta confirmé gelé : `KRG theta=[10.0, 10.0]  sigma=[1.0]`
- **ÉCHEC TOTAL** : 6 descentes FORM convergent hors domaine (u* à [-12,4], [8,-12], [-14,0]...), beta=11-15
- DBSCAN : 0 modes retenus — "Aucun FORM ne marche"
- Pas de visu matplotlib (skippée si 0 modes)
- **Cause** : `setOptimizeParameters(False)` gèle aussi sigma à 1.0, pas seulement theta. Les vraies valeurs g_DOE ≈ 0.4–0.8 → amplitude mal calibrée → surface quasi-plate avec gradients incohérents → FORM diverge hors domaine

---

## Détail idée theta fixe — ConstantBasis + theta=5 gelé

- Output : `output_KRG_constant_theta5.txt`
- Run en cours

---

## Conclusions

- **ConstantBasis (sans variante / v0) :** beta=3.856, écart -52.5%, Erreur FOSM=97.6% — rebound sévère, résultat non reproductible d'un run à l'autre
- **LinearBasis (v1) :** beta=7.2981, écart -10.2%, Erreur FOSM=5.7% — rebound corrigé, u* beaucoup plus proche du HF, FORM converge en 5 itérations
- **ConstantBasis + theta=10 gelé :** échec total — sigma=1.0 gelé aussi → surface mal calibrée → FORM diverge
- **LinearBasisFactory est le levier principal** : change l'extrapolation hors DOE de "retour vers constante positive" vers "continuation de la pente vers g<0"
- **Theta fixe (Option A) à éviter** : gèle aussi sigma → utiliser Option B (borne inférieure theta) à la place
