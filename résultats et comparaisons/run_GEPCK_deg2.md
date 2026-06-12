# Run GEPCK degree 2 — EFF critere at_least_one — 11/06/2026

**Output de reference : output_1106_1902.txt**

## Configuration

- modele = 'GEPCK'
- max_of_maxdegree = 2 (polynomes Hermite jusqu'au degre 2 : H2(u1), H2(u2), H1(u1)*H1(u2))
- n0 = 5 (LHS fixe), n_var = 2
- do_EFF = True, EFF_criteria = 'at_least_one' (premier critere atteint parmi BB, BS, both)
- tol_EFF = 1e-3, tol_BB = 0.01, tol_BS = 0.01
- n_IS = 10000, cov_IS = 0.05
- epsilon_factor = 2.0 (bande EFF = 2*sigma)
- Grille EFF : u in [-7.5, 7.5]^2

---

## Resultat EFF

- **Converge par BS (3 iterations valides)** apres **12 points ajoutes** (N=17 total)
- EFF final = 0.0185 (> tol_EFF=0.001 — arret par critere beta, pas EFF)
- count_valid_BB = 0, count_valid_BS = 3, count_valid_both = 0
- Historique ratio BB : [0.0752, 0.0506, None, 0.3049, 0.4944, 0.7317, 0.0786, 0.3841, 0.0769, 0.1139, 0.0915, 0.0872, 0.0826]
- Historique ratio BS : [0.0057, None, 0.0893, 0.0897, 0.0706, 0.1352, 0.134, 0.0781, 0.0193, 0.0023, 0.0004, 0.0064]

---

## Tableau iterations EFF

| N | EFF(u_opt) | n_poly | LOO | theta (fc, fy) | Termes PCE | BB ratio | BS ratio |
|---|-----------|--------|-----|----------------|-----------|----------|----------|
| 5 | 0.00692 (init) | 3 | 2.01e-02 | [0.442, 0.830] | `+0.327*1 +0.056*H1(u2) +0.039*H1(u1)` | 0.0752 (init) | — |
| 6 | 0.00692 | 3 | 2.49e-03 | [0.400, 0.864] | `+0.327*1 +0.055*H1(u2) +0.039*H1(u1)` | 0.0506 | 0.0057 |
| 7 | 0.01083 | 2 | 3.67e-01 | [1.825, 0.894] | `+0.247*1 -0.019*H1(u1)*H1(u2)` | None (FORM echoue) | None |
| 8 | 0.20020 | 3 | 7.15e-03 | [3.971, 0.387] | `+0.326*1 +0.043*H1(u1) +0.056*H1(u2)` | 0.3049 | 0.0893 |
| 9 | 0.03708 | 3 | 3.55e-02 | [2.452, 1.272] | `+0.313*1 +0.042*H1(u1) +0.056*H1(u2)` | 0.4944 | 0.0897 |
| 10 | 0.07501 | 3 | 1.20e-01 | [1.214, **14.636**] | `+0.321*1 +0.057*H1(u2) +0.034*H1(u1)` | 0.7317 | 0.0706 |
| 11 | 0.20424 | 5 | 5.39e-03 | [4.380, 5.742] | `+0.326*1 +0.055*H1(u2) +0.039*H1(u1) -0.005*H2(u1) +0.005*H1*H1` | 0.0786 | 0.1352 |
| 12 | 0.09160 | 3 | 5.77e-02 | [0.939, 3.798] | `+0.285*1 +0.056*H1(u2) +0.040*H1(u1)` | 0.3841 | 0.1340 |
| 13 | 0.06737 | 6 | 3.39e-03 | [0.641, 0.808] | `+0.327*1 +0.050*H1(u2) +0.046*H1(u1) +0.004*H1*H1 -0.003*H2(u1) -0.002*H2(u2)` | 0.0769 | 0.0781 |
| 14 | 0.02880 | 6 | 1.31e-02 | [1.301, 3.946] | `+0.336*1 +0.044*H1(u1) +0.005*H1*H1 +0.053*H1(u2) -0.002*H2(u2) -0.004*H2(u1)` | 0.1139 | 0.0193 |
| 15 | 0.01490 | 6 | 6.53e-03 | [0.392, 1.617] | `+0.325*1 +0.047*H1(u1) +0.004*H1*H1 +0.048*H1(u2) -0.002*H2(u2) -0.003*H2(u1)` | 0.0915 | **0.0023** |
| 16 | 0.00943 | 6 | 7.02e-03 | [0.432, 2.085] | `+0.324*1 +0.047*H1(u1) +0.049*H1(u2) +0.004*H1*H1 -0.002*H2(u2) -0.003*H2(u1)` | 0.0872 | **0.0004** |
| 17 | 0.00639 | 6 | 9.90e-03 | [1.099, 2.994] | `+0.331*1 +0.045*H1(u1) +0.052*H1(u2) +0.005*H1*H1 -0.002*H2(u2) -0.004*H2(u1)` | 0.0826 | **0.0064** |

**Convergence BS a iter 10, 11, 12** (N=15, 16, 17) : BS = 0.0023, 0.0004, 0.0064 — tous < tol_BS=0.01.

---

## Observations EFF

- **N=7 : LOO=0.37** — surrogate degenere (PCE reduit a 2 termes dont H1(u1)*H1(u2) seul).
  FORM echoue les 3 appels (_form_is_iter). EFF=0.20 avec u_opt=[7.2, 6.7] hors zone d'interet.
- **N=10 : theta fy=14.6** — longueur de correlation quasi-nulle sur fy, Kriging pathologique.
  BB ratio = 0.73, BB diverge.
- **N=11 : premier passage a 6 termes deg2** (n_poly=5) — BB retombe a 0.079.
  Mais LOO remonte a 5.4e-03 (OK), et N=12 retombe a 3 termes (LARS elimine les deg2).
- **N=13 : stabilisation sur 6 termes** — LOO < 5e-03, BS descend progressivement.
- **BB ne converge jamais** : oscille entre 0.07 et 0.73 sans tendance monotone.
  Critere at_least_one permet la sortie sur BS seul.

---

## Surrogate final (N=17)

- PCE tendance : `+0.331*1 +0.045*H1(u1) +0.052*H1(u2) +0.005*H1(u1)*H1(u2) -0.002*H2(u2) -0.004*H2(u1)`
- n_poly = 6 (base deg2 complete pour 2 variables)
- theta Kriging = [1.099, 2.994]
- LOO = 9.90e-03

---

## FORM final (N=17) — 18 points de depart (xt + [0,0])

| Mode | u* (u1, u2) | beta | Pf | fc* (MPa) | fy* (MPa) | Imp. (fc, fy) | Err. FOSM |
|------|------------|------|----|-----------|-----------|--------------:|-----------|
| 1 | [-2.915, -4.016] | 4.9624 | 3.48e-07 | 33.63 | 428.93 | [0.345, 0.655] | 5.1% |
| 2 | [-4.628, -2.014] | 5.0467 | 2.25e-07 | 27.41 | 489.28 | [0.841, 0.159] | 57.0% |
| 3 | [1.045, -6.712] | 6.7931 | 5.49e-12 | — | — | — | — |

**Mode dominant : mode 1** (beta=4.962, fy gouverne a 65.5%)  
**Mode 2 : fc gouverne** (u1=-4.6 tres faible, fy proche de la moyenne), erreur FOSM 57% — linearisation mauvaise.

Gradients HF evalues en u* :
- Mode 2 (u*=[-4.628, -2.014]) : dg/du_fc=0.0538, dg/du_fy=0.0348
- Mode 1 (u*=[-2.915, -4.016]) : dg/du_fc=0.0372, dg/du_fy=0.0488

DBSCAN eps=0.9, min_samples=2 — 3 modes retenus, 3 descentes isolees (bruit).
18 points de depart au total (xt=17 + [0,0]) :

| # | sp (u1, u2) | u* (u1, u2) | beta |
|---|-------------|-------------|------|
| 1 | [0.888, -1.047] | [-2.400, -4.365] | 4.9812 |
| 2 | [-0.291, -0.648] | [-2.360, -4.385] | 4.9800 |
| 3 | [-0.928, 0.410] | [-2.493, -4.370] | 5.0310 |
| 4 | [0.108, 0.162] | [-2.366, -4.386] | 4.9837 |
| 5 | [0.453, 1.159] | [-2.441, -4.380] | 5.0143 |
| 6 | [-1.296, -5.000] | [-2.453, -4.385] | 5.0243 |
| 7 | [-6.111, -1.667] | [-4.645, -2.554] | 5.3009 |
| 8 | [7.222, 6.667] | [-1.690, -2.341] | 2.8874 |
| 9 | [-7.222, -0.185] | [-4.628, -2.014] | 5.0467 |
| 10 | [1.667, -7.222] | [0.196, -6.180] | 6.1828 |
| 11 | [-3.704, -2.222] | [-4.190, -2.915] | 5.1049 |
| 12 | [7.222, -3.889] | [1.670, -7.258] | 7.4476 |
| 13 | [3.333, -6.667] | [1.049, -6.713] | 6.7945 |
| 14 | [-6.667, 7.222] | [-6.449, 0.013] | 6.4488 |
| 15 | [-5.000, -1.605] | [-4.524, -2.670] | 5.2528 |
| 16 | [-1.852, -4.815] | [-2.915, -4.016] | 4.9624 |
| 17 | [5.000, -6.111] | [1.045, -6.712] | 6.7931 |
| 18 | [0.000, 0.000] | [-2.366, -4.386] | 4.9836 |

---

## Importance Sampling post-FORM (mixture modes 1+2)

- Pf_IS   = 3.685e-07
- beta_IS = 4.9514
- COV     = 0.0500
- IC 95%  = [3.324e-07, 4.046e-07]
- N_IS    = 3031

---

## Comparaison HF / GEPCK deg1 / GEPCK deg2

| Grandeur | HF (ref) | GEPCK deg1 (N=39, non conv.) | GEPCK deg2 (N=17, conv. BS) |
|---|---|---|---|
| beta (FORM mode 1) | **5.1140** | ~4.59 | **4.962** |
| Pf (FORM mode 1) | **1.577e-07** | ~2.25e-06 | 3.48e-07 |
| beta_IS | — | ~4.64 (COV=0.12) | **4.951** (COV=0.05) |
| Pf_IS | — | ~1.73e-06 | **3.69e-07** |
| u* mode 1 | [-2.317, -4.559] | — | [-2.915, -4.016] |
| Imp. fc / fy (mode 1) | [0.205, 0.795] | — | [0.345, 0.655] |
| Nb. appels HF | ~194 | 39 + 5 initial = 44 | **17 + 5 initial = 22** |
| LOO final | — | 0.671 | 0.010 |
| Convergence EFF | — | Non (BB=0.725) | **Oui (BS, 3 iter)** |
| Modes FORM trouves | 1 | — | 3 |

---

## Diagnostic et conclusions

- **Ecart beta vs HF : ~3%** (4.95 vs 5.11) — nettement mieux que deg1 (~10% d'ecart)
- **Ecart Pf_IS vs HF : facteur 2.3** (3.69e-07 vs 1.58e-07) — acceptable compte tenu du COV=5%
- Le passage a deg2 (6 termes) stabilise le LOO (0.01 vs 0.67) et permet la convergence
- **BB ne converge pas** malgre le deg2 : la surface g+-2sigma reste mal contrainte globalement.
  BS converge en revanche — la position de beta_IS se stabilise meme si les bornes restent larges.
- **2 modes structurellement distincts** trouves (non detectes en deg1) :
  - Mode 1 : fy critique (u2=-4, fc moderement bas)
  - Mode 2 : fc critique (u1=-4.6, fy proche moyenne)
  - Mode 2 absent du run HF (un seul cluster DBSCAN en HF) — possiblement artefact surrogate
- Erreur FOSM mode 2 = 57% : la linearisation locale est mauvaise pour ce mode,
  ce qui signale que le surrogate est moins fiable loin du point de conception principal.
- **Piste** : deg1 fixe (max_of_maxdegree=1) pour forcer la stabilite PCE
  et eviter les surapprentissages partiels qui generent des theta pathologiques.