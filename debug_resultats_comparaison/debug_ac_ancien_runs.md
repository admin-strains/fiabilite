# Debug GEK — ac_ancien_ref.py (Tests A et B)
**Date :** 24 avril 2026
**Source :** `ac_ancien_ref.py` copie dans `AC_pure_flexion.py`
**Config :** do_GEK=True, do_GEK_analytic_grad=False (FD explicite), do_warm_start=False, n0=15, DOE fixe hardcode, F=0.210 MN
**Outputs :** output_2404_1524.txt (run 1), output_2404_1536.txt (run 2), output_2404_1541.txt (run 3)
**Reference HF :** beta=3.784, u*=[-0.526, -3.747]

---

## Tableau comparatif des 3 runs

| Parametre | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| fc* (MPa) | 33.154 | 33.287 | 30.346 |
| fy* (MPa) | 576.148 | 551.329 | 529.471 |
| u* [u_fc, u_fy] | [+0.286, -1.399] | [+0.324, -2.222] | [-0.553, -2.947] |
| Importance fc | 4.02% | 2.09% | 3.40% |
| Importance fy | 95.98% | 97.91% | 96.60% |
| **beta (FORM)** | **1.428** | **2.246** | **2.999** |
| Pf (FORM) | 7.661e-02 | 1.235e-02 | 1.355e-03 |
| n_iter FORM | 18 | 32 | 23 |
| g_HF(u*) | 0.1030 | 0.0691 | 0.0333 |
| g_GEK(u*) | 0.1030 | 0.0691 | 0.0327 |
| u* FOSM | [-0.643, -3.728] | [-0.643, -3.728] | [-0.643, -3.728] |
| Erreur FOSM | 175.6% | 79.7% | 26.2% |
| Ecart beta vs HF | -2.356 (-62.3%) | -1.538 (-40.6%) | -0.785 (-20.7%) |
| Gradient err u_fc | 41% | 144% | **860%** |
| Gradient err u_fy | 22% | 40% | 53% |

---

## Observations

### Non-determinisme confirme
Les 3 runs utilisent le meme DOE fixe et le meme code → seul le training GEKPLS (L-BFGS-B non-deterministe) differe. Betas obtenus : 1.428 / 2.246 / 2.999 → ecart de 110% entre run 1 et run 3. Aucun run n'atteint 3.774 (valeur historique de l'ancien code avec do_GEK_analytic_grad=True).

### Gradient GEKPLS tres mauvais
La validation gradient (bloc avant FORM, point u=[-1.2,-3.0]) montre des erreurs enormes :
- Run 1 : 41% et 22%
- Run 2 : 144% et 40%  
- Run 3 : **860%** et 53% (gradient u_fc a SIGNE INVERSE de la FD !)

Ces erreurs signifient que pour ces trainings GEKPLS, `predict_derivatives` est completement faux. La valeur `predict_values` reste coherente (g_GEK ≈ g_HF au point u* dans les 3 runs) mais les gradients sont inutilisables. FORM, guide par ces gradients (via FD sur predict_values ou directement via predict_derivatives), converge vers de mauvais points.

### u* physiquement faux (runs 1 et 2)
Run 1 et 2 : u_fc > 0 → fc au-dela de la moyenne → cote sur → defaillance dans le mauvais sens. Run 3 : u_fc < 0 (correct) mais trop faible en norme (beta=2.999 au lieu de 3.784).

---

## Conclusion Test A

**L'ancien code avec FD explicite est NON-DETERMINISTE et donne de mauvais resultats.** Le beta=3.774 historique (run unique de l'ancien code avec do_GEK_analytic_grad=True) etait un run chanceux. Le non-determinisme GEKPLS est la cause principale de l'instabilite, independamment du mode gradient (FD ou analytique).

**Prochain test (Test B) :** ac_ancien_ref.py, FD, n0=20 — voir partie 2 ci-dessous.

---

## Partie 2 — Test B.1 : ac_ancien_ref.py, FD, n0=20

**Config :** do_GEK=True, do_GEK_analytic_grad=False (FD), do_warm_start=False, n0=20, F=0.210 MN
**Run 1 :** DOE aleatoire LHS — runs 2 et 3 : DOE fixe (hardcode apres run 1)

### DOE utilise (runs 2 et 3)

```python
U_doe = ot.Sample([
    [ 0.1724259008776929,  0.02126916531407232],
    [ 0.7730726578349874, -0.11022816812156475],
    [ 0.5600784079159713,  0.4030783887642334],
    [-1.0303664509409196,  1.9321687806923271],
    [-2.2885465390848676, -0.20037884810151527],
    [ 1.0433459230268767,  1.1765090675540273],
    [ 0.26508747566031765,  0.9551009704807997],
    [ 0.9566965600165915, -1.9364918802084576],
    [ 3.3900408127386714,  0.6239523233537078],
    [ 1.5640345893341396, -0.7247802568841013],
    [-0.2855596718323952,  0.24414447736367606],
    [ 0.09067231286194077, -1.27281785101642],
    [-1.267189339905397,  0.35906055166314166],
    [ 0.4888845723253038, -0.6251126047456709],
    [-0.41774800192353856, -1.3103402564435584],
    [-0.06634024071502326, -0.4955137444808357],
    [-1.3195709857820606, -0.9094700391064136],
    [-0.24974253472143296,  1.5036742080221814],
    [-0.5278817646262698, -0.2938143331206941],
    [-0.6905110539329726,  0.7512096086122947],
])
```

### Tableau comparatif des runs

| Parametre | Run 1 (DOE aleat.) | Run 2 (DOE fixe) | Run 3 (DOE fixe) |
|---|---|---|---|
| Output | output_2404_1552.txt | output_2404_1754.txt | output_2404_1803.txt |
| fc* (MPa) | 25.474 | 29.741 | 30.407 |
| fy* (MPa) | 511.322 | 558.868 | 539.947 |
| u* [u_fc, u_fy] | [-2.213, -3.549] | [-0.744, -1.972] | [-0.534, -2.600] |
| Importance fc | 28.0% | 12.46% | 4.05% |
| Importance fy | 72.0% | 87.54% | 95.95% |
| **beta (FORM)** | **4.183** | **2.108** | **2.654** |
| Pf (FORM) | 1.440e-05 | 1.751e-02 | 3.975e-03 |
| n_iter FORM | 16 | 29 | 31 |
| g_HF(u*) | -0.002079 | 0.072486 | 0.047808 |
| g_GEK(u*) | -0.001985 | 0.072165 | 0.047491 |
| Erreur FOSM | 37.8% | 83.43% | 42.71% |
| Ecart beta vs HF | +0.399 (+10.5%) | -1.676 (-44.3%) | -1.130 (-29.9%) |
| Gradient err u_fc | 51.1% | 50.62% | 65.87% |
| Gradient err u_fy | 9.3% | 8.52% | 23.47% |

---

## Observations Test B.1

### Non-determinisme confirme a n0=20

Meme DOE fixe (runs 2 et 3) → betas tres differents : 2.108 vs 2.654 (ecart 26%). Le non-determinisme GEKPLS persiste a n0=20.

### Run 1 (DOE aleatoire) exceptionnellement bon

Run 1 obtient beta=4.183 (+10.5% vs HF=3.784) avec g_HF(u*)=-0.002 (presque sur la surface limite) — ce run avait un bon DOE aleatoire ET un bon training GEKPLS. Les runs 2 et 3 sur le meme DOE fixe donnent 2.108 et 2.654 (mauvais) → c'est le training GEKPLS non-deterministe qui fait varier le resultat, pas le DOE lui-meme.

### Gradient GEKPLS encore mauvais

Erreurs gradient u_fc : 50.6% / 65.9% pour runs 2 et 3 (vs 51.1% pour run 1). Gradient u_fy un peu moins mauvais (8.5-23.5%). u* converge dans la bonne direction (u_fc < 0) mais avec une norme trop faible (beta sous-estime de 30-44%).

### Comparaison n0=15 vs n0=20

n0=20 ne stabilise pas le GEKPLS : betas 2.108-4.183 (n0=20) vs 1.428-2.999 (n0=15). La dispersion reste grande. Le meilleur run n0=20 (4.183) est meilleur que le meilleur run n0=15 (2.999), mais c'est du a un DOE aleatoire chanceux, pas a une stabilisation structurelle.

## Conclusion Test B.1

**NON-DETERMINISTE confirme a n0=20.** n0=20 n'apporte pas de stabilisation systematique. Le beta=4.183 du run 1 etait un concours de circonstances (bon DOE + bon training). La cause racine reste le non-determinisme L-BFGS-B dans GEKPLS → champ de gradient `predict_derivatives` tres variable → FORM mal guide.

---

## Partie 3 — Test B.2 : ac_ancien_ref.py, FD, n0=25

**Config :** do_GEK=True, do_GEK_analytic_grad=False (FD), do_warm_start=False, n0=25, F=0.210 MN
**Run 1 :** DOE aleatoire LHS — runs 2 et 3 : DOE fixe (hardcode apres run 1)

### DOE utilise (runs 2 et 3)

```python
U_doe = ot.Sample([
    [ 2.454187017123209,   0.08220678832885409],
    [-0.04504671036992964, -0.6328301439347265],
    [-0.16905619080659504,  2.3155177186100167],
    [-3.0150330449309357,  -0.8955715188949352],
    [ 1.1562893063919488,  -0.4471346756160089],
    [-1.101965529756189,   -1.7847316246580256],
    [ 0.43687448503337856,  0.7111231798185268],
    [-0.40316282672990783, -1.215015534228918 ],
    [ 0.9638538811269464,   1.1706334266429532],
    [ 0.053002912303317565, 0.32161893098854344],
    [ 0.6453082472671939,  -1.617626701960161 ],
    [-0.9587502634091257,   0.4028581779674414],
    [-0.3093069069423424,   0.6978483391653028],
    [-0.6750260195009847,   1.5642706949591916],
    [ 1.3625545817536366,  -1.0496745223924995],
    [ 1.5225396576611896,   0.5366444863441309],
    [ 0.55470727076279,    -0.7489739892416397],
    [ 0.8406231994796461,   0.15409624135952626],
    [ 0.24287567336678678,  1.378868193946595 ],
    [ 0.3320815464098168,  -0.30899009915187625],
    [-0.5224065327879118,  -0.5683714992978386],
    [-1.2562546674839232,   0.9115204516163219],
    [-0.7387459712022774,  -0.1337816380546613],
    [-1.6003641148720698,   0.0038926603638018897],
    [-0.14814394390412508, -0.16382912813241024],
])
```

### Tableau comparatif des runs

| Parametre | Run 1 (DOE aleat.) | Run 2 (DOE fixe) | Run 3 (DOE fixe) |
|---|---|---|---|
| Output | output_2404_1818.txt | output_2404_1826.txt | output_2404_1834.txt |
| fc* (MPa) | 30.001 | 32.154 | 31.822 |
| fy* (MPa) | 542.829 | 545.253 | 580.240 |
| u* [u_fc, u_fy] | [-0.662, -2.504] | [-0.004, -2.424] | [-0.102, -1.263] |
| Importance fc | 6.52% | **0.00%** | 0.65% |
| Importance fy | 93.48% | **100.00%** | 99.35% |
| **beta (FORM)** | **2.590** | **2.424** | **1.268** |
| Pf (FORM) | 4.796e-03 | 7.677e-03 | 1.025e-01 |
| n_iter FORM | 25 | 24 | 21 |
| g_HF(u*) | 0.051019 | 0.058451 | 0.105943 |
| g_GEK(u*) | 0.050892 | 0.058682 | 0.105992 |
| Erreur FOSM | 47.26% | 59.92% | 199.07% |
| Ecart beta vs HF | -1.194 (-31.6%) | -1.361 (-36.0%) | -2.516 (-66.5%) |
| Gradient err u_fc | **152.3%** (signe inverse) | 28.18% | **651.2%** |
| Gradient err u_fy | 7.51% | 0.97% | 27.60% |

---

## Observations et Conclusion Test B.2

### Non-determinisme confirme a n0=25, pire qu'a n0=20

Meme DOE fixe (runs 2 et 3) → betas : 2.424 vs 1.268 (ecart 91%). Le run 3 est le pire de toute la serie (beta=1.268, ecart -66.5% vs HF). Augmenter n0 de 20 a 25 n'ameliore pas la stabilite — au contraire.

### Gradient u_fc completement faux dans runs 1 et 3

Run 1 : erreur 152% (signe inverse). Run 3 : erreur 651% → FORM converge vers un point loin de la surface limite (g=0.106). Run 2 : gradient u_fc acceptable (28%) mais u_fc≈0 → importance fc=0% → point de defaillance physiquement faux (fc au niveau de la moyenne).

### Bilan n0=15 / n0=20 / n0=25

| n0 | Betas obtenus | Min | Max | Ecart max-min |
|---|---|---|---|---|
| 15 (Test A) | 1.428 / 2.246 / 2.999 | 1.428 | 2.999 | 1.571 |
| 20 (Test B.1) | 4.183 / 2.108 / 2.654 | 2.108 | 4.183 | 2.075 |
| 25 (Test B.2) | 2.590 / 2.424 / 1.268 | 1.268 | 2.590 | 1.322 |

Augmenter n0 ne stabilise pas GEKPLS. La dispersion reste enorme et le meilleur run (n0=20, run1=4.183) etait un concours de circonstances. La cause racine est le non-determinisme L-BFGS-B → theta differents → paysage de `predict_values` different loin des points d'entrainement → gradient FD utilise par FORM variable → convergence vers des points tres differents. Note : en mode FD, `predict_derivatives` n'est jamais appele par FORM — le bloc de validation gradient est uniquement diagnostique.

---

## Partie 4 — Test B.3 : ac_ancien_ref.py, FD, n0=25, do_warm_start=True

**Config :** do_GEK=True, do_GEK_analytic_grad=False (FD), do_warm_start=True, n0=25, F=0.210 MN
**Run 1 :** DOE aleatoire LHS — runs 2 et 3 : meme DOE fixe que Test B.2 (hardcode)
**Objectif :** voir si warm start stabilise FORM malgre le non-determinisme GEKPLS.

### DOE utilise (tous les runs)

Meme DOE fixe que Test B.2 (DOE non remis en aleatoire entre les tests) :

```python
U_doe = ot.Sample([
    [ 2.454187017123209,   0.08220678832885409],
    [-0.04504671036992964, -0.6328301439347265],
    [-0.16905619080659504,  2.3155177186100167],
    [-3.0150330449309357,  -0.8955715188949352],
    [ 1.1562893063919488,  -0.4471346756160089],
    [-1.101965529756189,   -1.7847316246580256],
    [ 0.43687448503337856,  0.7111231798185268],
    [-0.40316282672990783, -1.215015534228918 ],
    [ 0.9638538811269464,   1.1706334266429532],
    [ 0.053002912303317565, 0.32161893098854344],
    [ 0.6453082472671939,  -1.617626701960161 ],
    [-0.9587502634091257,   0.4028581779674414],
    [-0.3093069069423424,   0.6978483391653028],
    [-0.6750260195009847,   1.5642706949591916],
    [ 1.3625545817536366,  -1.0496745223924995],
    [ 1.5225396576611896,   0.5366444863441309],
    [ 0.55470727076279,    -0.7489739892416397],
    [ 0.8406231994796461,   0.15409624135952626],
    [ 0.24287567336678678,  1.378868193946595 ],
    [ 0.3320815464098168,  -0.30899009915187625],
    [-0.5224065327879118,  -0.5683714992978386],
    [-1.2562546674839232,   0.9115204516163219],
    [-0.7387459712022774,  -0.1337816380546613],
    [-1.6003641148720698,   0.0038926603638018897],
    [-0.14814394390412508, -0.16382912813241024],
])
```

### Tableau comparatif des runs

| Parametre | Run 1 (DOE fixe B.2) | Run 2 (DOE fixe B.2) | Run 3 (DOE fixe B.2) |
|---|---|---|---|
| Output | output_2404_1843.txt | output_2404_1852.txt | output_2404_1900.txt |
| fc* (MPa) | 33.466 | 37.042 | 30.544 |
| fy* (MPa) | 612.675 | 524.187 | 610.253 |
| u* [u_fc, u_fy] | [+0.375, -0.188] | [+1.338, -3.123] | [-0.491, -0.268] |
| Importance fc | 80.00% | 15.52% | 77.07% |
| Importance fy | 20.00% | 84.48% | 22.93% |
| **beta (FORM)** | **0.420** | **3.397** | **0.560** |
| Pf (FORM) | 3.374e-01 | 3.402e-04 | 2.879e-01 |
| n_iter FORM | 1 (apres WS) | 1 (apres WS) | 1 (apres WS) |
| Warm start declenche | Oui, U=[-0.780,-1.299] | Oui, U=[-0.232,-3.795] | Oui, U=[-0.148,-1.740] |
| g_HF(u*) | 0.153973 | 0.039787 | 0.144520 |
| g_GEK(u*) | 0.153954 | 0.039700 | 0.144575 |
| Erreur FOSM | 877.89% | 60.99% | 618.86% |
| Ecart beta vs HF | -3.364 (-88.9%) | -0.387 (-10.2%) | -3.224 (-85.2%) |
| Gradient err u_fc | 62.0% | 22.47% | 11.85% |
| Gradient err u_fy | 18.66% | 17.79% | 3.04% |

---

## Observations et Conclusion Test B.3

### Warm start amplifie le non-determinisme

Le warm start est declenche dans les 3 runs (g_meta(u*) > tol_warm_start apres le 1er FORM). Le point de warm start varie fortement selon le run : U=[-0.780,-1.299] / [-0.232,-3.795] / [-0.148,-1.740] → cela reflete la convergence non-deterministe du 1er FORM. Le 2eme FORM (depuis le point WS) donne alors des betas radicalement differents : 0.420 / 3.397 / 0.560. Le warm start ne corrige pas le non-determinisme — il ajoute une variable aleatoire supplementaire (le point de depart du 2eme FORM).

### n_iter=1 apres warm start — signature d'un metamodele localement faux

Dans les 3 runs, FORM converge en 1 seule iteration apres le warm start. Cela signifie que g_meta(u_WS) est tres proche de 0 selon le metamodele (condition d'arret atteinte immediatement) alors que g_HF(u*) est loin de 0 (0.144, 0.040, 0.145). Le metamodele interpole correctement les points d'entrainement mais extrapole mal en dehors — FORM arrive dans une zone ou le metamodele dit "surface limite" alors que la vraie g est loin de 0.

### Run 2 seul donne un resultat acceptable (3.397)

Le point WS du run 2 ([-0.232,-3.795]) etait proche de la vraie region de defaillance → le 2eme FORM a converge dans la bonne direction. Mais u_fc=+1.338 > 0 reste physiquement faux. C'est encore un concours de circonstances, pas une stabilisation.

## Conclusion Test B.3

**Le warm start aggrave le comportement plutot qu'il ne le corrige.** Il est declenche systematiquement (le 1er FORM converge toujours loin de la surface limite avec ce DOE), et le 2eme FORM converge en 1 iteration vers un faux minimum local du metamodele. La cause racine n'est pas resolue : le paysage de `predict_values` genere par GEKPLS est non-deterministe et souvent faux loin des points d'entrainement.

---

## Partie 5 — Test B.4 : ac_ancien_ref.py, FD, n0=40, do_warm_start=True

**Config :** do_GEK=True, do_GEK_analytic_grad=False (FD), do_warm_start=True, n0=40, F=0.210 MN
**Run 1 :** DOE aleatoire LHS — runs 2 et 3 : DOE fixe (hardcode apres run 1)
**Objectif :** voir si n0=40 + warm start stabilise FORM.

### DOE utilise (runs 2 et 3)

```python
U_doe = ot.Sample([
    [-0.09642876374824510,  -0.004935576597932493],
    [ 0.75472945113387680,   1.3288102470801681  ],
    [ 2.82508716426242300,   0.9093149471718069  ],
    [-0.44455136192165756,  -0.2547597649679175  ],
    [-0.32768818347430350,  -1.936849449618994   ],
    [-1.53309278706591720,  -0.2156278362085319  ],
    [-0.30031841983430135,   0.25799511301284095 ],
    [-0.14329062917323124,  -0.39740904521480913 ],
    [ 0.84466949796827610,  -1.1939583023296974  ],
    [ 0.52318691738158460,  -1.9639140965004946  ],
    [ 0.15106911215120086,   1.1486629471658547  ],
    [ 0.66066804609256320,   0.5538333098360521  ],
    [-0.58382928686348280,   0.6344616932174629  ],
    [-0.83461779336763290,  -0.8656521290782975  ],
    [-0.85910934947789320,  -0.4864137492786775  ],
    [-1.22584554469584560,   0.1885448543996336  ],
    [ 0.56197517638781300,  -0.3229871524890924  ],
    [ 1.95045489840018260,   0.054754609915340625],
    [ 0.29109936483832527,   0.42963081654255036 ],
    [ 1.00069400380617070,   0.23673112747026992 ],
    [ 0.12218306622978949,  -0.1867329015687322  ],
    [ 1.41135714686575200,  -0.5839764746534565  ],
    [-1.29168234291926080,  -0.7742002671149933  ],
    [-0.48765342866193984,  -0.6961856594579741  ],
    [-1.91664873008099780,   0.4976907986609022  ],
    [ 1.45490760645954880,   1.5000385743003386  ],
    [-0.24284156097087240,   1.2336365278261192  ],
    [ 0.21878563484297714,  -1.5286297373429822  ],
    [ 0.39196517860322544,  -0.09169398239921221 ],
    [ 0.00209692176901697,   0.34076008510051997 ],
    [-1.05691472582633320,   1.0169450069611647  ],
    [-0.66677177600587310,   1.7135884421078893  ],
    [ 0.33869052175799910,   1.980788676867951   ],
    [ 1.25089689345483260,  -0.9598725758109781  ],
    [-0.73917137768866040,   0.11002489151366933 ],
    [ 0.78756347048760560,  -0.6731058959139757  ],
    [-0.97558980701798480,  -1.4091552138410595  ],
    [-2.00118991831035680,  -1.074812508578901   ],
    [-0.04618092359015204,   0.727957118016397   ],
    [ 1.13241562225754320,   0.8365195001961125  ],
])
```

### Tableau comparatif des runs

| Parametre | Run 1 (DOE aleat.) | Run 2 (DOE fixe) | Run 3 (DOE fixe) |
|---|---|---|---|
| Output | output_2404_1911.txt | — | — |
| fc* (MPa) | 30.474 | — | — |
| fy* (MPa) | 614.262 | — | — |
| u* [u_fc, u_fy] | [-0.513, -0.135] | — | — |
| Importance fc | 93.52% | — | — |
| Importance fy | 6.48% | — | — |
| **beta (FORM)** | **0.531** | — | — |
| Pf (FORM) | 2.979e-01 | — | — |
| n_iter FORM | 1 (apres WS) | — | — |
| Warm start declenche | Oui, U=[-0.176,-0.579] | — | — |
| g_HF(u*) | 0.149878 | — | — |
| g_GEK(u*) | 0.149970 | — | — |
| Erreur FOSM | 677.74% | — | — |
| Ecart beta vs HF | -3.253 (-86.0%) | — | — |
| Gradient err u_fc | 59.68% | — | — |
| Gradient err u_fy | 42.87% | — | — |

**Test B.4 arrete apres run 1** : meme pathologie que B.3 (WS declenche, n_iter=1 apres WS, g_HF=0.150). Runs 2 et 3 non effectues.

---

## Partie 6 — Test C : ac_ancien_ref.py, KRG (do_GEK=False), n0=25, do_warm_start=True

**Config :** do_GEK=False (KRG pur), do_warm_start=True, n0=25, F=0.210 MN
**Run 1 :** DOE aleatoire LHS — runs 2 et 3 : DOE fixe (hardcode apres run 1)
**Objectif :** verifier si KRG est aussi non-deterministe que GEK ou si le probleme est specifique a GEKPLS.

### DOE utilise (tous les runs)

Identique au DOE Test B.2 (OT PRNG deterministe, meme graine, n0=25) — voir Partie 3.

### Tableau comparatif des runs

| Parametre | Run 1 (DOE aleat.=B.2) | Run 2 (DOE fixe) | Run 3 (DOE fixe) |
|---|---|---|---|
| Output | output_2404_1924.txt | output_2404_1932.txt | output_2404_1939.txt |
| fc* (MPa) | 30.201 | 33.052 | 33.032 |
| fy* (MPa) | 505.294 | 508.115 | 507.889 |
| u* [u_fc, u_fy] | [-0.598, -3.749] | [+0.257, -3.656] | [+0.251, -3.663] |
| Importance fc | 2.48% | 0.49% | 0.47% |
| Importance fy | 97.52% | 99.51% | 99.53% |
| **beta (FORM)** | **3.797** | **3.665** | **3.672** |
| Pf (FORM) | 7.331e-05 | 1.238e-04 | 1.204e-04 |
| n_iter FORM | 1 (apres WS) | 15 | 15 |
| Warm start declenche | Oui, U=[0.249,-3.664] | Non | Non |
| g_KRG(u*) | 0.001086 | -0.000241 | 0.000073 |
| Erreur FOSM | 1.30% | 24.65% | 24.43% |
| Ecart beta vs HF | +0.013 (+0.3%) | -0.119 (-3.2%) | -0.112 (-3.0%) |

---

## Observations et Conclusion Test C — KRG

### KRG est deterministe : runs 2 et 3 quasi-identiques

Runs 2 et 3 sur le meme DOE fixe → betas 3.665 / 3.672 (ecart 0.2%), u* quasi-identiques ([+0.257,-3.656] vs [+0.251,-3.663]), n_iter=15 pour les deux. L'entrainement KRG (optimisation des hyperparametres) est deterministe — contrairement a GEKPLS (L-BFGS-B non-deterministe).

### Run 1 different a cause du warm start

Run 1 : warm start declenche → 1 point HF supplementaire ajoute au DOE (n0+1=26 points) → metamodele legerement different → beta=3.797 au lieu de ~3.67. Sans warm start, KRG sur ce DOE donne ~3.67 de maniere reproductible.

### KRG nettement plus stable que GEK sur le meme DOE

Sur le meme DOE n0=25 (Test B.2 vs Test C) :
| Metamodele | Betas obtenus | Ecart max-min |
|---|---|---|
| GEK (Test B.2) | 2.590 / 2.424 / 1.268 | 1.322 |
| KRG (Test C) | 3.797 / 3.665 / 3.672 | 0.132 |

Le non-determinisme est specifique a GEKPLS. KRG donne des resultats stables et proches de la reference HF (erreur 3%) avec le meme DOE.

## Conclusion Test C

**KRG est deterministe et stable.** Le probleme de non-determinisme est bien specifique a GEKPLS (non-determinisme L-BFGS-B dans l'optimisation des hyperparametres theta). La question suivante est : pourquoi le GEK du nouveau code est-il instable alors que le nouveau code utilise aussi KRG dans d'autres branches ? → Test D : tester ac_nouveau_copie.py en mode FD pour voir si la refactorisation a introduit un probleme specifique au GEK.

---

## Partie 7 — Test E : ac_ancien_ref.py, GEK FD, n0=25, do_warm_start=True, F=0.235 MN

**Config :** do_GEK=True, do_GEK_analytic_grad=False (FD), do_warm_start=True, n0=25, **F=0.235 MN** (beta_HF≈0.95), dsLoad.txt Z='-0.235'
**Run 1 :** DOE aleatoire LHS — runs 2 et 3 : DOE fixe (hardcode apres run 1)
**Objectif :** verifier si le non-determinisme GEKPLS persiste a F=0.235 (zone de defaillance proche de l'origine).

### DOE utilise (tous les runs)

Identique aux Tests B.2 et C (OT PRNG deterministe, meme graine, n0=25) — voir Partie 3.

### Tableau comparatif des runs

| Parametre | Run 1 (DOE fixe B.2) | Run 2 (DOE fixe B.2) | Run 3 (DOE fixe B.2) |
|---|---|---|---|
| Output | output_2404_1958.txt | output_2404_2007.txt | output_2404_2014.txt |
| fc* (MPa) | 31.525 | 31.882 | 32.038 |
| fy* (MPa) | 612.714 | 612.117 | 613.131 |
| u* [u_fc, u_fy] | [-0.191, -0.186] | [-0.085, -0.206] | [-0.038, -0.173] |
| Importance fc | 51.32% | 14.40% | 4.73% |
| Importance fy | 48.68% | 85.60% | 95.27% |
| **beta (FORM)** | **0.267** | **0.223** | **0.177** |
| Pf (FORM) | 3.947e-01 | 4.118e-01 | 4.298e-01 |
| n_iter FORM | 15 (apres WS) | 4 (apres WS) | 8 (apres WS) |
| Warm start declenche | Oui, U=[0.034,-0.151] | Oui, U=[0.025,-0.165] | Oui, U=[0.036,-0.068] |
| g_HF(u*) | 0.027665 | 0.027634 | 0.029178 |
| g_GEK(u*) | 0.027693 | 0.027627 | 0.029163 |
| Erreur FOSM | 282.13% | 330.90% | 439.49% |
| Ecart beta vs HF | -0.683 (-71.9%) | -0.727 (-76.5%) | -0.773 (-81.4%) |
| Gradient err u_fc | 88.74% | **122.3%** (signe inv.) | **123.8%** (signe inv.) |
| Gradient err u_fy | 8.26% | 26.99% | 17.59% |

---

## Observations et Conclusion Test E

### Non-determinisme confirme a F=0.235

Les 3 runs donnent des betas differents (0.267 / 0.223 / 0.177) sur le meme DOE — le non-determinisme GEKPLS persiste independamment de la valeur de F et de la difficulte du probleme (beta_HF≈0.95 vs 3.784).

### Gradient u_fc avec signe inverse dans les 3 runs

Erreurs 89% / 122% / 124% — le signe de la derivee selon fc est faux dans les 3 trainings. FORM converge vers des points proches de l'origine (norme ~0.17-0.27) au lieu de chercher la vraie surface limite.

### Betas plus resserres mais tous faux

Ecart max-min = 0.09 (vs 1.32 en Test B.2) — FORM converge toujours vers la meme mauvaise region (proche de l'origine) quel que soit le training GEKPLS. Cela s'explique par la proximite du vrai u* de la reference a l'origine (beta_HF=0.95 → u* proche du centre) : le faux minimum local du metamodele et le vrai minimum sont dans la meme region, mais le metamodele "colle" au mauvais endroit.

## Conclusion Test E

**Le non-determinisme GEKPLS est independant de F.** A F=0.235 (beta facile ≈0.95), GEK donne 0.177-0.267 (erreur ~75-81%). Le probleme vient bien du training GEKPLS lui-meme, pas des specificites du probleme a beta eleve.

---

## Partie 8 — Test F : ac_ancien_ref.py, GEK FD, n0=25, tol_FORM=0.05, F=0.235 MN

**Config :** do_GEK=True, do_GEK_analytic_grad=False (FD), do_warm_start=True, n0=25, **tol_FORM=0.05**, F=0.235 MN, DOE fixe B.2 (tous les runs)
**Objectif :** voir si une tolerance FORM plus stricte ameliore la convergence malgre le mauvais gradient GEKPLS.

### Tableau comparatif des runs

| Parametre | Run 1 (DOE fixe B.2) | Run 2 (DOE fixe B.2) | Run 3 (DOE fixe B.2) |
|---|---|---|---|
| Output | output_2404_2025.txt | output_2404_2033.txt | — |
| fc* (MPa) | 32.301 | 32.278 | — |
| fy* (MPa) | 604.167 | 614.354 | — |
| u* [u_fc, u_fy] | [+0.039, -0.470] | [+0.033, -0.132] | — |
| Importance fc | 0.69% | 5.74% | — |
| Importance fy | 99.31% | 94.26% | — |
| **beta (FORM)** | **0.471** | **0.136** | — |
| Pf (FORM) | 3.186e-01 | 4.459e-01 | — |
| n_iter FORM | 19 (apres WS) | 4 (apres WS) | — |
| Warm start declenche | Oui, U=[-0.031,-0.146] | Oui, U=[-0.083,-0.063] | — |
| g_HF(u*) | 0.018647 | 0.031142 | — |
| g_GEK(u*) | 0.018632 | 0.031107 | — |
| Erreur FOSM | 108.35% | 610.99% | — |
| Ecart beta vs HF | -0.479 (-50.4%) | -0.814 (-85.7%) | — |
| Gradient err u_fc | 81.77% | 30.91% | — |
| Gradient err u_fy | 18.38% | 51.42% | — |

---

## Partie 9 — Test G : ac_ancien_ref.py, GEK FD, n0=25, tol_FORM=0.001, F=0.235 MN

**Config :** do_GEK=True, do_GEK_analytic_grad=False (FD), do_warm_start=True, n0=25, **tol_FORM=0.001**, F=0.235 MN, DOE fixe B.2 (1 run)
**Objectif :** voir si une tolerance FORM tres stricte ameliore la convergence malgre le mauvais gradient GEKPLS.
**RESULTAT : CRASH** — RuntimeError OT : g_meta(u*)=0.00860 > tol_FORM=0.001. Aucun resultat recuperable. Output : output_2404_2135.txt.

### Tableau comparatif des runs

| Parametre | Run 1 (DOE fixe B.2) |
|---|---|
| Output | — |
| fc* (MPa) | — |
| fy* (MPa) | — |
| u* [u_fc, u_fy] | — |
| Importance fc | — |
| Importance fy | — |
| **beta (FORM)** | — |
| Pf (FORM) | — |
| n_iter FORM | — |
| Warm start declenche | — |
| g_HF(u*) | — |
| g_GEK(u*) | — |
| Erreur FOSM | — |
| Ecart beta vs HF | — |
| Gradient err u_fc | — |
| Gradient err u_fy | — |

---

## Partie 10 — Test H : ac_ancien_ref.py + fix BLAS single-thread, GEK FD, n0=25, tol_FORM=0.01, F=0.235 MN

**Config :** do_GEK=True, do_GEK_analytic_grad=False (FD), do_warm_start=True, n0=25, **tol_FORM=0.01**, F=0.235 MN, DOE fixe B.2 (3 runs)
**Fix BLAS :** `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` ajoutes en tete de AC_pure_flexion.py
**Objectif :** verifier si forcer BLAS en single-thread stabilise le training GEKPLS → betas identiques entre 3 runs.

### Tableau comparatif des runs

| Parametre | Run 1 (DOE fixe B.2) | Run 2 (DOE fixe B.2) | Run 3 (DOE fixe B.2) |
|---|---|---|---|
| Output | output_2404_2200.txt | output_2404_2210.txt | output_2404_2220.txt |
| fc* (MPa) | 32.402 | 32.402 | 32.402 |
| fy* (MPa) | 613.594 | 613.594 | 613.594 |
| u* [u_fc, u_fy] | [+0.069, -0.157] | [+0.069, -0.157] | [+0.069, -0.157] |
| Importance fc | 16.10% | 16.10% | 16.10% |
| Importance fy | 83.90% | 83.90% | 83.90% |
| **beta (FORM)** | **0.172** | **0.172** | **0.172** |
| Pf (FORM) | 4.319e-01 | 4.319e-01 | 4.319e-01 |
| n_iter FORM | 6 (apres WS) | 6 (apres WS) | 6 (apres WS) |
| Warm start declenche | Oui, U=[-0.047,-0.132] | Oui, U=[-0.047,-0.132] | Oui, U=[-0.047,-0.132] |
| g_HF(u*) | 0.030445 | 0.030445 | 0.030445 |
| g_GEK(u*) | 0.030415 | 0.030415 | 0.030415 |
| Erreur FOSM | 475.36% | 475.36% | 475.36% |
| Ecart beta vs HF | -0.778 (-81.9%) | -0.778 (-81.9%) | -0.778 (-81.9%) |
| Gradient err u_fc | 62.00% | 62.00% | 62.00% |
| Gradient err u_fy | 51.92% | 51.92% | 51.92% |

---

## Partie 11 — Test I : ac_ancien_ref.py + fix BLAS + n_comp=1, GEK FD, n0=25, tol_FORM=0.05, F=0.235 MN

**Config :** do_GEK=True, do_GEK_analytic_grad=False (FD), do_warm_start=True, n0=25, tol_FORM=0.05, **reduc_PLS=1 (n_comp=1)**, F=0.235 MN, DOE fixe B.2 (3 runs), fix BLAS single-thread actif
**Objectif :** n_comp=1 reduit theta a 1 parametre → paysage log-vraisemblance plus simple → meilleur theta → meilleur gradient GEK → FORM converge vers u*.

### Tableau comparatif des runs

| Parametre | Run 1 (DOE fixe B.2) | Run 2 (DOE fixe B.2) | Run 3 (DOE fixe B.2) |
|---|---|---|---|
| Output | — | — | — |
| fc* (MPa) | — | — | — |
| fy* (MPa) | — | — | — |
| u* [u_fc, u_fy] | — | — | — |
| Importance fc | — | — | — |
| Importance fy | — | — | — |
| **beta (FORM)** | — | — | — |
| Pf (FORM) | — | — | — |
| n_iter FORM | — | — | — |
| Warm start declenche | — | — | — |
| g_HF(u*) | — | — | — |
| g_GEK(u*) | — | — | — |
| Erreur FOSM | — | — | — |
| Ecart beta vs HF | — | — | — |
| Gradient err u_fc | — | — | — |
| Gradient err u_fy | — | — | — |

---

### Conclusion Test H

**BLAS single-thread = training parfaitement reproductible.** Les 3 runs sont bit-for-bit identiques : meme beta, meme u*, meme warm start trigger, memes erreurs gradient.

**Diagnostic confirme :** le non-determinisme observe dans les tests A-F etait exclusivement cause par le parallelisme BLAS dans L-BFGS-B. Avec 1 thread, L-BFGS-B converge vers le meme theta a chaque run.

**Probleme residuel :** le theta trouve de maniere reproductible donne un mauvais gradient (62%/52% d'erreur) → FORM converge vers un faux zero pres de l'origine (beta=0.172 au lieu de 0.95). Le fix BLAS resout le non-determinisme mais pas la qualite du training GEKPLS sur ce DOE.
