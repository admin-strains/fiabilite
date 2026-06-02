# Comparaison GEK vs KRG — Flexion pure BA (phi=16mm)

> **Objectif :** évaluer la performance de GEK (GEKPLS SMT) vs KRG (OpenTURNS) pour le FORM, à DOE identique.
> **Conditions communes :** F=0.210 MN (β_HF≈3.78), noyau squar_exp, solver AbdoRackwitz, warm start activé, DOE fixé identique hardcodé.
> **Référence :** FORM HF (appels directs STRAINS), β_HF=3.784.
> **Gradient GEK :** calculé par différences finies (FD) par OpenTURNS — étape 2 du plan d'activation GEK.

---

## Tableau 1 — n0=20, squar_exp, warm start, DOE fixé

| | **HF** | **KRG squar_exp n0=20+1** | **GEK FD n0=20+1** |
|---|---|---|---|
| u* | [-0.526, -3.747] | [-0.538, -3.754] | [-1.400, -3.546] |
| β | 3.784 | **3.792** | 3.813 |
| g_HF(u*) | ≈ 0 | -3.3e-04 | +2.8e-03 |
| g_méta(u*) | ≈ 0 | +4.2e-03 | +3.0e-03 |
| Erreur relative β | 0% (réf.) | **0.2%** | 0.8% |
| u* warm start déclencheur | — | — | [-0.338, -3.308] |

**Conclusion :** à n0=20 avec warm start et DOE identique, KRG squar_exp est plus précis que GEK FD (0.2% vs 0.8%). Les deux convergent vers un résultat physique proche de la référence. L'écart reste faible — le warm start corrige efficacement les deux métamodèles sur ce cas. La différence principale est la localisation de u* : KRG trouve [-0.538, -3.754] très proche du vrai [-0.526, -3.747], GEK trouve [-1.400, -3.546] plus décalé en u_fc.

---

## Tableau 2 — n0=5, squar_exp, warm start, DOE fixé identique

> DOE fixé (5 pts) : [[0.323, 1.084], [-0.449, 0.443], [0.929, -0.037], [-1.364, -0.590], [0.249, -1.437]]

| | **HF** | **KRG squar_exp n0=5+1** | **GEK FD n0=5+1** |
|---|---|---|---|
| u* | [-0.526, -3.747] | — | [-0.466, -3.755] |
| β | 3.784 | **✗ RuntimeError** | **3.784** |
| g_HF(u*) | ≈ 0 | — | +3.0e-06 |
| g_méta(u*) | ≈ 0 | — | +1.0e-06 |
| Erreur relative β | 0% (réf.) | — | **0.0%** |
| u* warm start déclencheur | — | — | [-0.474, -3.806] |

**Conclusion :** à n0=5, KRG squar_exp échoue (gradient nul, DOE trop sparse pour couvrir la queue β≈3.78) là où GEK converge parfaitement (erreur 0.0%). L'avantage de GEK vient des gradients adjoints STRAINS utilisés à l'entraînement : chaque point DOE apporte n_var+1=3 informations au lieu de 1 pour KRG. À ce n0 très faible, cette richesse est décisive.

---
