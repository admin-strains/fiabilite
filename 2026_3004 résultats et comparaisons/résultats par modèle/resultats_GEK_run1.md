# Résultats FORM GEK — Nouvelle géométrie — Run 1

**Output :** `output_3004_1447.txt`
**Config :** do_GEK=True, n0=35, fck=40 MPa, fyk=500 MPa

## Configuration du modèle

| Paramètre | Valeur |
|---|---|
| **— Géométrie —** | |
| b (m) | 0.5 |
| h (m) | 0.8 |
| L (m) | 5.0 |
| φ (mm) | 16.0 |
| Armatures | 2 lits de 3HA16 (z_lit1=0.328m, z_lit2=0.312m) |
| Block1 (0–4m) | ft1=0.1 MPa |
| Block2 (4–5m) | ft2=3.5 MPa |
| **— Chargement —** | |
| ‖F‖ (MN) | 0.1 |
| **— DOE —** | |
| n0 | 35 |
| DOE | LHS aléatoire (U_doe_fixed=None) |
| **— Distributions JCSS —** | |
| fck (MPa) | 40 |
| fyk (MPa) | 500 |

## Résultats FORM GEK (print_results)

| Paramètre | F=0.1 MN |
|---|---|
| **— Résultats FORM —** | |
| n_iter FORM | 1 |
| fc* (MPa) | 47.8323 |
| fy* (MPa) | 560.9651 |
| u* [u_fc, u_fy] | [0.0059, 0.3771] |
| dg/du_fc en u* | 0.000226 |
| dg/du_fy en u* | 0.051794 |
| Importance fc | 0.02% |
| Importance fy | 99.98% |
| β (FORM) | 0.3771 |
| Pf (FORM) | 6.4696e-01 |
| n_appels HF (DOE) | 35 |
| n_appels HF (FORM) | 0 |
| **— FOSM —** | |
| u* FOSM (GEK) | [0.0015, 0.3941] |
| Erreur FOSM | 4.65% |

## Observations

- FORM_all_modes (36 points de départ) : 1 seul mode détecté, tous convergent vers u*≈[0.002, 0.394], beta≈0.394.
- Note : discordance entre FORM_all_modes (beta=0.3940) et print_results/FORM_multistart (beta=0.3771, u*=[0.0059, 0.3771]). Les deux u* sont à distance <0.05 en U — même mode physique.
- Importance fc=0.02%, fy=99.98% : défaillance exclusivement par plastification des armatures, cohérent avec HF.
- beta=0.38 très faible (Pf=65%) : F=0.1 MN trop faible — structure quasi en état limite. F à augmenter.
- Erreur FOSM GEK = 4.65% vs 0.46% HF : linéarisation moins précise sur le GEK qu'en HF direct.

---

## Run 2 — Géométrie b=0.2, h=0.6, 2×1HA16

**Output :** `output_3004_1549.txt`
**Config :** do_GEK=True, n0=5, fck=28 MPa, fyk=550 MPa

### Configuration du modèle

| Paramètre | Valeur |
|---|---|
| **— Géométrie —** | |
| b (m) | 0.2 |
| h (m) | 0.6 |
| L (m) | 5.0 |
| φ (mm) | 16.0 |
| Armatures | 2 lits de 1HA16 centré (z_lit1=0.248m, z_lit2=0.232m) |
| Block1 (0–4m) | ft1=0.1 MPa |
| Block2 (4–5m) | ft2=3.5 MPa |
| **— Chargement —** | |
| ‖F‖ (MN) | 0.022 |
| **— DOE —** | |
| n0 | 5 |
| DOE | LHS aléatoire (U_doe_fixed=None) |
| **— Distributions JCSS —** | |
| fck (MPa) | 28 |
| fyk (MPa) | 550 |

### Résultats FORM GEK (print_results)

| Paramètre | F=0.022 MN |
|---|---|
| **— Résultats FORM —** | |
| n_iter FORM | 1 |
| fc* (MPa) | 35.6400 |
| fy* (MPa) | 504.1953 |
| u* [u_fc, u_fy] | [-0.0243, -3.1642] |
| dg/du_fc en u* | -0.000692 |
| dg/du_fy en u* | 0.054149 |
| Importance fc | 0.01% |
| Importance fy | 99.99% |
| β (FORM) | 3.1643 |
| Pf (FORM) | 7.7718e-04 |
| n_appels HF (DOE) | 5 |
| n_appels HF (FORM) | 0 |
| **— FOSM —** | |
| u* FOSM (GEK) | [-0.0076, -3.1699] |
| Erreur FOSM | 0.56% |

### Observations

- 1 mode détecté. u* = [-0.024, -3.164] : fy domine (u_fy négatif → fy faible → plastification).
- beta = 3.16 : charge bien calibrée pour cette géométrie.
- Importance fc=0.01%, fy=99.99% : même comportement que précédemment, défaillance par plastification acier.
- Erreur FOSM 0.56% : g très linéaire sur le GEK avec n0=5.

---

## Run 3 — Géométrie b=0.2, h=0.5, 3×1HA32

**Output :** `output_3004_1807.txt`
**Config :** do_GEK=True, n0=20, fck=63 MPa, fyk=550 MPa

### Configuration du modèle

| Paramètre | Valeur |
|---|---|
| **— Géométrie —** | |
| b (m) | 0.2 |
| h (m) | 0.5 |
| L (m) | 5.0 |
| φ (mm) | 32.0 |
| Armatures | 3 lits de 1HA32 (z_lit1=+0.202m, z_lit2=+0.170m, z_lit3=+0.138m) |
| Block1 (0–4m) | ft1=0.1 MPa |
| Block2 (4–5m) | ft2=3.5 MPa |
| **— Chargement —** | |
| ‖F‖ (MN) | 0.12 |
| **— DOE —** | |
| n0 | 20 |
| DOE | LHS OT graine fixe (print_DOE=True, U_doe_fixed=None) |
| **— Distributions JCSS —** | |
| fck (MPa) | 63 |
| fyk (MPa) | 550 |

### Résultats FORM GEK (print_results)

| Paramètre | F=0.12 MN |
|---|---|
| **— Résultats FORM —** | |
| n_iter FORM | 1 |
| fc* (MPa) | 72.4652 |
| fy* (MPa) | 664.7061 |
| u* [u_fc, u_fy] | [0.3271, 2.1596] |
| dg/du_fc en u* (HF) | 0.008781 |
| dg/du_fy en u* (HF) | 0.039239 |
| Importance fc | 2.24% |
| Importance fy | 97.76% |
| β (FORM) | 2.1842 |
| Pf (FORM) | 9.8553e-01 |
| n_appels HF (DOE) | 20 |
| n_appels HF (FORM) | 0 |
| **— FOSM —** | |
| u* FOSM (HF) | [0.4327, 2.2351] |
| Erreur FOSM | 5.94% |

### Observations

- FORM_all_modes : 1 mode, beta=2.279, u*=[0.5, 2.223] (cohérent avec print_results).
- u* = [0.327, 2.160] : POSITIF → origine en domaine de défaillance → F=0.12 MN trop grande.
  - À propriétés matériaux moyennes (u=0), la structure est déjà en défaillance.
  - beta = 2.18, Pf = 0.985 = Φ(+2.18) : régime "Pf inversé", pas le régime cible.
- Importance fc=2.24%, fy=97.76% : fy domine toujours, cohérent avec les autres géométries.
- Erreur FOSM 5.94% : acceptable.
- **Action requise :** diminuer F pour sortir de ce régime (viser u* négatif, Pf ≈ 10⁻³–10⁻⁴, beta ≈ 3–4).
