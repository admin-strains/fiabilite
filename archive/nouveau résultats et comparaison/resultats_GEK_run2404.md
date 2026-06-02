# Resultats FORM GEK pur — Run 2404
**Date :** 24 avril 2026
**Configuration :** GEK pur (do_GEK=True, try_pce=False), n0=15, DOE aleatoire LHS, F=0.210 MN, do_warm_start=False, do_analytic_grad=True, noyau squar_exp, solver AbdoRackwitz
**Outputs :** `output/output_2404_1050.txt` (run 1), `output/output_2404_1111.txt` (run 2)
**Reference HF :** beta=3.784, Pf=7.73e-05, u*=[-0.526, -3.747], n_iter=21

---

### F = 0.210 MN (beta_HF = 3.784)

| Parametre | Run 1 (DOE aleatoire #1) | Run 2 (DOE aleatoire #2) |
|---|---|---|
| n points DOE | 15 (aleatoire LHS) | 15 (aleatoire LHS) |
| fc* (MPa) | 28.1505 | 29.6271 |
| fy* (MPa) | 519.6437 | 557.7960 |
| u* | [-1.2653, -3.2733] | [-0.7804, -2.0079] |
| dg/du_fc en u* | 0.006241 | 0.006570 |
| dg/du_fy en u* | 0.041636 | 0.041590 |
| Importance fc | 13.00% | 13.12% |
| Importance fy | 87.00% | 86.88% |
| beta (FORM) | 3.509353 | 2.154206 |
| Pf (FORM) | 2.245987e-04 | 1.561200e-02 |
| n_appels HF (FORM) | 0 | 0 |
| n_iter FORM | 22 | 32 |
| g_HF(u*) | 0.014977 | 0.070800 |
| g_meta(u*) | 0.014548 | 0.070311 |
| Erreur relative g | 2.86% | 0.69% |
| u* FOSM | [-0.643, -3.728] | [-0.643, -3.728] |
| Erreur FOSM | 21.96% | 80.11% |
| Ecart beta vs HF | -0.275 (-7.3%) | -1.630 (-43.1%) |
| do_warm_start | Non | Non |

---

### Notes

- Warnings sklearn PLS : "y residual is constant at iteration 1" (plusieurs occurrences dans les deux runs) — non bloquant, comportement GEK connu avec n0 faible.
- **Run 1 :** g_HF(u*)=0.015, g_meta(u*)=0.015. FORM non converge sur la surface limite exacte mais erreur relative faible (2.86%). Le point u* est legerement dans le domaine sur.
- **Run 2 :** g_HF(u*)=0.071, g_meta(u*)=0.070. Ecart au seuil g=0 beaucoup plus important. u* tres eloigne de la reference HF [-0.526,-3.747]. GEK instable sur ce DOE.
- **Variabilite inter-DOE :** beta=3.509 vs 2.154 pour deux realisations aleatoires du meme LHS n0=15 — ecart de 63% entre les deux runs. Confirme l'instabilite de GEK pur a n0=15.
- Comparaison avec GEK pur DOE fixe run1 (ancien dossier) : beta=2.118 (-44%) → meme ordre de grandeur que run 2 aleatoire.
- do_warm_start=False : si WS avait ete active, il aurait declenche dans les deux runs (g_meta >> tol=0.0001).
