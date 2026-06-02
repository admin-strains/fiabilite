# Résultats FORM HF — Nouvelle géométrie — Run 1

**Output :** `output_3004_1203.txt`
**Config :** do_HF=True, point de départ [0,0], fck=40 MPa, fyk=500 MPa

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
| **— Distributions JCSS —** | |
| fck (MPa) | 40 |
| fyk (MPa) | 500 |

## Résultats FORM HF (print_results)

| Paramètre | F=0.1 MN |
|---|---|
| **— Résultats FORM —** | |
| n_iter FORM | 6 |
| fc* (MPa) | 47.8138 |
| fy* (MPa) | 561.3999 |
| u* [u_fc, u_fy] | [0.0016, 0.3915] |
| dg/du_fc en u* | 0.000220 |
| dg/du_fy en u* | 0.051833 |
| Importance fc | 0% |
| Importance fy | 100% |
| β (FORM) | 0.3915 |
| Pf (FORM) | 6.5229e-01 |
| n_appels HF | 1 |
| **— FOSM —** | |
| u* FOSM | [0.0015, 0.3933] |
| Erreur FOSM | 0.46% |

## Observations

- Importance fc = 0%, fy = 100% : défaillance exclusivement par plastification des armatures.
- beta = 0.39 très faible (Pf = 65%) : F=0.1 MN trop faible, la structure est quasi en état limite.
- g quasi-linéaire : erreur FOSM 0.46%.
- F à recalibrer pour obtenir beta ≈ 3–4.