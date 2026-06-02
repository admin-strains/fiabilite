# Résultats run HF direct — 12/05/2026

## Configuration
- do_HF = True, do_GEK = False
- n0 = 15 (LHS) + [0,0] = 16 points de départ
- gamma_c = 1.0, gamma_s = 1.0 (dsCad)
- gamma_c_fic = 1.0, gamma_s_fic = 1.0
- F = 0.74 MN, b = h = 0.8m, 3 lits 8HA32
- fcm = 48 MPa, fym = 550 MPa, cov_fc = 0.12
- tol_all_modes = 0.01, n_max_FORM = 50

## Résultat FORM (meilleur mode)
- beta = 7.9788
- Pf   = 7.3874e-16
- u*   = [-3.1166, -7.3449]
- fc*  = 32.8318 MPa
- fy*  = 328.5525 MPa
- Imp. = [0.1526, 0.8474]  (fc=15%, fy=85%)
- n_iter FORM = 7
- u* FOSM (HF) = [-4.1491, -6.3773]
- Erreur FOSM  = 17.74%

## Points de départ et u* obtenus (FORM_all_modes)

| sp | u* | beta |
|---|---|---|
| [-0.002,  1.332] | [-5.306, -6.200] | 8.1601 |
| [ 0.610, -0.310] | [-3.117, -7.349] | 7.9831 |
| [-0.571, -1.705] | [-3.131, -7.347] | 7.9859 |
| [ 0.258,  0.306] | [-4.655, -6.643] | 8.1121 |
| [ 0.121, -1.010] | [-3.117, -7.345] | 7.9788 |
| [ 1.624, -0.531] | [-3.046, -7.352] | 7.9582 |
| [-0.087, -0.172] | [-4.721, -6.603] | 8.1174 |
| [-0.419,  0.597] | [-5.290, -6.212] | 8.1589 |
| [-0.843, -0.005] | [-5.341, -6.152] | 8.1471 |
| [ 0.868, -1.460] | [-3.014, -7.363] | 7.9564 |
| [-1.681,  0.710] | [-6.475, -4.986] | 8.1724 |
| [ 1.479,  0.239] | [-3.098, -7.354] | 7.9800 |
| [-1.117, -0.696] | [-5.200, -6.275] | 8.1494 |
| [ 0.745,  1.043] | [-4.740, -6.571] | 8.1026 |
| [-0.694,  2.114] | [-6.504, -4.966] | 8.1830 |
| [ 0.000,  0.000] | [-4.776, -6.571] | 8.1235 |

## Observations
- 1 mode retenu par DBSCAN (eps=0.01, min_samples=2) autour de u*≈[-3.0, -7.35]
- Plusieurs familles de convergence visibles : [-3,-7.35], [-4.7,-6.6], [-5.3,-6.2], [-6.5,-5.0]
- Tous les u* en quadrant u1<0, u2<0 (branche horizontale pivot B)
- beta élevé (≈8) cohérent avec gamma=1.0 (pas de coefficients de sécurité)
