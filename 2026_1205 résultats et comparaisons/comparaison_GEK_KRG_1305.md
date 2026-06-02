# Comparaison GEK / KRG vs HF reference — 13/05/2026

**Config commune :** b=h=0.8m, 3 lits 8HA32, phi=32mm, F=0.74 MN, gamma_c=gamma_s=1.0, fcm=48 MPa, fym=550 MPa, cov_fc=0.12, do_warmstart=True, do_multistart=True, tol_FORM=1.0, tol_warmstart=0.2.

---

## Tableau de comparaison

| Metrique | HF ref (1205) | GEK n0=6 (1305_1704) | GEK n0=10 (1305_1712) | KRG n0=10 (1305_1726) | KRG n0=18 (1305_1743) | PCKRG n0=12 (1505_0931) | GEPCK n0=12 (1505_0946) | PCKRG n0=5+4EFF (1905_0901) |
|---|---|---|---|---|---|---|---|---|
| n points DOE | 15 | 6+1 (WS) | 10+1 (WS) | 10+1 (WS) | 18+1 (WS) | 12 | 12 | 5+4 (EFF) |
| n_iter FORM | 7 | 46 | 44 | 40 | 41 | 11 | 11 | 14 |
| n_appels HF (FORM) | n_iter | 1 (WS) | 1 (WS) | 1 (WS) | 1 (WS) | 0 | 0 | 0 |
| fc* (MPa) | 32.83 | 18.52 | 30.13 | 29.35 | 51.04 | 21.02 | 21.02 | 22.97 |
| fy* (MPa) | 328.55 | 517.58 | 465.61 | 430.14 | 459.64 | 396.77 | 396.77 | 395.72 |
| u* [u_fc, u_fy] | [-3.117, -7.345] | [-7.905, -1.075] | [-3.835, -2.799] | [-4.055, -3.976] | [0.573, -2.997] | [-6.848, -5.082] | [-6.848, -5.082] | [-6.104, -5.117] |
| dg/du_fc en u* | 0.025218 | 0.098668 | 0.055551 | 0.048118 | 0.033171 | N/A (*) | N/A (*) | 0.051979 |
| dg/du_fy en u* | 0.071830 | 0.008637 | 0.050556 | 0.053987 | 0.074931 | N/A (*) | N/A (*) | 0.044101 |
| Importance fc (%) | 15.3% | 98.2% | 65.2% | 51.0% | 3.5% | 64.5% | 64.5% | 58.7% |
| Importance fy (%) | 84.7% | 1.8% | 34.8% | 49.0% | 96.5% | 35.5% | 35.5% | 41.3% |
| beta (FORM) | **7.9788** | 7.9776 | 4.7476 | 5.6790 | 3.0515 | 8.5277 | 8.5277 | **7.9652** |
| Pf (FORM) | 7.39e-16 | 7.46e-16 | 1.03e-06 | 6.78e-09 | 1.14e-03 | 7.47e-18 | 7.47e-18 | 8.25e-16 |
| g_meta(u*) | ~0 | N/A | N/A | N/A | N/A | N/A (*) | N/A (*) | N/A |
| g_HF(u*) | ~0 | N/A | N/A | N/A | N/A | N/A (*) | N/A (*) | N/A |
| u* FOSM | [-4.149, -6.377] | [-4.149, -6.377] | [-4.149, -6.377] | [-4.149, -6.377] | [-4.149, -6.377] | N/A (*) | N/A (*) | [-4.149, -6.377] |
| Erreur FOSM | 17.7% | 81.4% | 75.7% | 42.3% | 190.3% | N/A (*) | N/A (*) | 29.2% |
| Ecart beta vs HF | — | -0.0015 (-0.01%) | -3.231 (-40.5%) | -2.300 (-28.8%) | -4.927 (-61.8%) | +0.549 (+6.88%) | +0.549 (+6.88%) | **-0.014 (-0.17%)** |
| do_warmstart | N | O | O | O | O | N | N | N |
| Nb modes trouves | 1 (DBSCAN) | 1 (1 pt) | 2 | 2 | 2 | 1 (2/13, 11 bruit) | 1 (2/13, 11 bruit) | 1 |

(*) print_results appelé avec ancienne signature (sans g_ot_PCKRG) — disponible à partir du prochain run.

---

## Resultats detailles par run

### HF reference — output_1205_2032.txt

- do_HF=True, n0=15, do_multistart=True, do_warmstart=False
- **beta = 7.9788**, Pf = 7.39e-16
- u* = [-3.117, -7.345]
- fc* = 32.83 MPa, fy* = 328.55 MPa
- Imp. fc/fy = 15.3% / 84.7%
- n_iter = 7, 1 mode DBSCAN
- dg/du_fc = 0.025218, dg/du_fy = 0.071830
- u* FOSM = [-4.149, -6.377], Erreur FOSM = 17.7%

### GEK + warm start n0=6 — output_1305_1704.txt

- do_GEK=True, n0=6, do_warmstart=True, do_multistart=True
- Warm start declenche : FORM initial [0,0] → beta=7.693, u*=[-7.585,-1.286]
- Apres WS : 1 mode (1 seul u* = warm start point comme sp unique)
- **beta = 7.9776**, Pf = 7.46e-16
- u* = [-7.905, -1.075]  ← direction FAUSSE (vs HF u*=[-3.1,-7.3])
- fc* = 18.52 MPa (fc tres faible, hors zone physique)
- fy* = 517.58 MPa
- Imp. fc/fy = 98.2% / 1.8%  ← inverses vs HF (85% fy)
- n_iter = 46
- dg/du_fc = 0.098668, dg/du_fy = 0.008637
- u* FOSM = [-4.149, -6.377], Erreur FOSM = 81.4%
- **Note :** beta numeriquement proche du HF mais mode completement faux — artefact rebound KRG/GEK, le surrogate avec n0=6 (tous g>0) cree une "ile" g<0 en direction u1<<0.

### GEK + warm start n0=10 — output_1305_1712.txt

- do_GEK=True, n0=10, do_warmstart=True, do_multistart=True
- Warm start declenche : FORM initial [0,0] → beta=6.273, u*=[1.0,-6.193]
- Apres WS : 2 modes trouves
- **Mode 1 (best) : beta = 4.7476**, Pf = 1.03e-06
- u* = [-3.835, -2.799]
- fc* = 30.13 MPa, fy* = 465.61 MPa
- Imp. fc/fy = 65.2% / 34.8%
- n_iter = 44
- dg/du_fc = 0.055551, dg/du_fy = 0.050556
- u* FOSM = [-4.149, -6.377], Erreur FOSM = 75.7%
- Mode 2 : beta=6.234, u*=[0.697,-6.195], Imp. fc/fy=1.25%/98.75%, n_iter=51, Erreur FOSM=77.8%

### KRG + warm start n0=10 — output_1305_1726.txt

- do_KRG=True, n0=10, do_warmstart=True, do_multistart=True
- Warm start declenche : FORM initial [0,0] → beta=6.398, u*=[1.254,-6.274]
- Apres WS : 2 modes trouves
- **Mode 1 (best) : beta = 5.6790**, Pf = 6.78e-09
- u* = [-4.055, -3.976]
- fc* = 29.35 MPa, fy* = 430.14 MPa
- Imp. fc/fy = 51.0% / 49.0%
- n_iter = 40
- dg/du_fc = 0.048118, dg/du_fy = 0.053987
- u* FOSM = [-4.149, -6.377], Erreur FOSM = 42.3%
- Mode 2 : beta=7.720, u*=[0.824,-7.675], Imp. fc/fy=1.1%/98.9%, n_iter=51, Erreur FOSM=66.6%
- **Note :** mode 2 KRG (beta=7.72) tres proche du HF (7.98, -3.2%). KRG meilleur que GEK a n0=10.

### KRG + warm start n0=18 — output_1305_1743.txt

- do_KRG=True, n0=18, do_warmstart=True, do_multistart=True
- Warm start declenche : FORM initial [0,0] → beta=3.171, u*=[-2.682,-1.692]
- Apres WS : 2 modes trouves
- **Mode 1 (best) : beta = 3.0515**, Pf = 1.14e-03
- u* = [0.573, -2.997]
- fc* = 51.04 MPa, fy* = 459.64 MPa
- Imp. fc/fy = 3.5% / 96.5%
- n_iter = 41
- dg/du_fc = 0.033171, dg/du_fy = 0.074931
- u* FOSM = [-4.149, -6.377], Erreur FOSM = 190.3%
- Mode 2 : beta=3.172, u*=[-2.678,-1.700], Imp. fc/fy=71.3%/28.7%, n_iter=40, Erreur FOSM=154.6%

### PCKRG no warmstart n0=12 — output_1505_0931.txt

- do_KRG=True, try_pce=True, n0=12, do_warmstart=False, do_multistart=True
- 13 descentes FORM (12 DOE + [0,0])
- 11/13 descentes ignorees (bruit DBSCAN eps=0.01) — betas disperses 8.27 a 9.00
- Seules 2 descentes convergent au meme u* (distance≈0.009 < eps) → 1 mode retenu
- **beta = 8.5277**, Pf = 7.47e-18
- u* = [-6.8477, -5.0823] ← direction fausse (vs HF [-3.1, -7.3])
- fc* = 21.02 MPa (tres faible), fy* = 396.77 MPa
- Imp. fc/fy = 64.5% / 35.5% ← inverses vs HF (15% / 85%)
- n_iter = 11
- dg/du, FOSM, g_meta, g_HF : N/A (ancienne signature print_results)

### GEPCK no warmstart n0=12 — output_1505_0946.txt

- do_GEK=True, try_pce=True, n0=12, do_warmstart=False, do_multistart=True
- Meme DOE que PCKRG (graine OT fixe -> LHS identique) -> resultats identiques
- 11/13 descentes ignorees (bruit DBSCAN), 1 mode retenu
- **beta = 8.5277**, Pf = 7.47e-18
- u* = [-6.8477, -5.0824], fc* = 21.02 MPa, fy* = 396.77 MPa
- Imp. fc/fy = 64.5% / 35.5%, n_iter = 11
- dg/du, FOSM, g_meta, g_HF : N/A (ancienne signature print_results)

### GEPCK no warmstart n0=3 — output_1505_1117.txt

- do_GEK=True, try_pce=True, n0=3, do_warmstart=False, do_multistart=True, max_degree=1
- **PCE construite : basis_size=3, coefficients actifs LARS=1** (constante seulement — LARS rejette u1/u2 avec 3 pts)
- 2/4 descentes ECHEC (RuntimeError), 2 convergent au meme u*
- 1 mode retenu (DBSCAN eps=0.01)
- **beta = 1.5364**, Pf = 6.222e-02
- u* = [-1.1968, -0.9634] — direction completement fausse (vs HF ref [-4.776, -6.571])
- fc* = 41.30 MPa, fy* = 520.96 MPa
- Imp. fc/fy = 60.7% / 39.3%
- n_iter = 30
- dg/du_fc (HF@u*GEPCK) = 0.054537, dg/du_fy = 0.059878
- u* FOSM (HF) = [-4.149, -6.377]
- **Note :** beta=1.5364 identique aux runs max_degree=1 et max_degree=2 precedents — confirme que la PCE est toujours degeneree a une constante (LARS 1 coeff) quel que soit le degre candidat avec n0=3.

---

## Tableau comparatif GEPCK vs PCKRG — n0 variable, no warmstart (session 15/05)

**Reference HF commune (sp=[0,0]) :** beta=8.1235, u*=[-4.776, -6.571]

| n0 | GEPCK beta | GEPCK u* | PCKRG beta | PCKRG u* | Ecart GEPCK | Ecart PCKRG |
|---|---|---|---|---|---|---|
| 3 | **1.5364** | [-1.197, -0.963] | **ECHEC** | — | -81% | — |
| 6 | 7.6856 | [-4.68, -6.10] | 7.6856 | [-4.68, -6.10] | -5.4% | -5.4% |
| 8 | 7.4445 | [-5.73, -4.75] | TBD | TBD | -8.4% | TBD |
| 12 | 8.5277 | [-6.85, -5.08] | 8.5277 | [-6.85, -5.08] | +5.0% | +5.0% |

**Observations :**
- n0=6/12 : GEPCK = PCKRG (graine OT fixee => LHS identique => meme DOE => meme resultat)
- n0=3 : GEPCK converge (beta=1.5364 faux mais converge) alors que PCKRG echoue totalement (0 FORM reussi) — differentiation visible. Explication : GEKPLS utilise 3x2=6 valeurs supplementaires (gradients) pour contraindre le residu, KRG uniquement les 3 valeurs scalaires.
- n0=3 GEPCK : PCE degeneree (constante, LARS=1 coeff) => avantage GEPCK vient UNIQUEMENT de GEKPLS sur le residu, pas de la PCE.
- n0=8 PCKRG : a relancer.

---

## PCKRG n0=5+4EFF — output_1905_0901.txt (session 19/05)

**Config :** PCKRG, n0=5 DOE initial, do_EFF=True, update_degree dynamique (max_of_maxdegree=2), do_warmstart=False, do_multistart=True.
**EFF :** 4 points ajoutes (max_degree 1→2 au 4e point, DOE=8 > n0_min(2)=7), puis convergence EFF.

- n points DOE total : 9 (5 initial + 4 EFF)
- n_iter FORM : 14
- fc* = 22.97 MPa, fy* = 395.72 MPa
- u* = [-6.104, -5.117], beta = **7.9652**, Pf = 8.25e-16
- Importance fc/fy = 58.7% / 41.3%
- dg/du_fc = 0.051979, dg/du_fy = 0.044101
- u* FOSM = [-4.149, -6.377], Erreur FOSM = 29.2%
- Ecart vs HF ref (7.9788) = **-0.17%** — meilleur resultat de toute la comparaison (hors HF)

**Comparaison directe PCKRG n0=12 no EFF vs PCKRG n0=5+4EFF :**
| | PCKRG n0=12 no EFF | PCKRG n0=5+4EFF |
|---|---|---|
| n appels HF total | 12 | **9** |
| beta | 8.5277 (+6.88%) | **7.9652 (-0.17%)** |
| u* | [-6.848, -5.082] | [-6.104, -5.117] |
EFF + update_degree dynamique : 3 appels de moins, erreur 40x plus faible.

---

## Observations generales

- **Artefact rebound surrogate :** tous les surrogates (GEK et KRG) avec DOE aleatoire LHS en zone g>0 creent une "ile" g<0 artificielle. Le surrogate revient vers beta0>0 (moyenne DOE) loin du DOE. La zone rouge est une bulle fermee, pas la vraie region semi-infinie.
- **GEK n0=6 :** beta numeriquement correct (7.98) mais mode faux (u* a [-7.9,-1.1] au lieu de [-3.1,-7.3]). La convergence est fortuite, dans la mauvaise direction.
- **Augmenter n0 ne stabilise pas :** KRG n0=18 (beta=3.05) pire que KRG n0=10 (beta=5.68). Le DOE LHS aleatoire peut tomber n'importe ou.
- **KRG mode 2 a n0=10 :** beta=7.72, erreur -3.2% vs HF. Le surrogate capture un mode proche du vrai par chance.
- **Erreur FOSM tres elevee (42-190%) :** la surface limite est fortement non-lineaire dans la zone exploree par les surrogates → hypothese FOSM invalide ici.
- **Conclusion :** DOE aleatoire LHS insuffisant pour cette geometrie. Besoin d'un DOE oriente vers la zone de defaillance (importance sampling, DOE adaptatif, ou DOE fixe couvrant u*).
