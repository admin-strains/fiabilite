# Resume session 04 mai 2026 — Validation analytique + corrections code

---

## Partie 1 — Verification du code et premier run (output_0405_0718.txt)

### Contexte

Reprise de la session 30/04. Geometrie active : b=0.2m, h=0.5m, 3x1HA32, F=0.12 MN, fck=63, fyk=550.
Objectif du jour : verifier la cohérence de print_visu, lister les bugs potentiels, les corriger, et valider print_error_ana_hf.

### Verification print_visu + lecture du code

Lecture de AC_pure_flexion.py pour verification avant relance. Points notes :
- `d = h - z_centroide` dans calc_ana() → formule erronee (bug, corrige en Partie 2).
- `fac_c=1.1` dans la signature de `flexion_simple.__init__` → parametre inutilise (supprime en Partie 2).
- `n_visu` dans print_error_ana_hf → variable non definie dans OPTIONS (corrige par l'utilisatrice : remplace par `n_grid_hf`).
- `SIGMA` : importe et utilise dans print_error_ana_hf pour calculer u2_min. Non un bug.
- Grille HF dans print_visu sur [0, size_visu] (quadrant positif) : choix intentionnel de l'utilisatrice pour comparaison avec la frontiere analytique.

### Run 1 (output_0405_0718.txt)

**Config :** do_GEK=True, n0=20, fck=63, fyk=550, U_doe_fixed=None (graine OT fixe), print_ana=True, print_ana_hf_error=True, size_visu=5, n_grid_hf=7. Run de verification avant application des fixes.

| Parametre | Valeur |
|---|---|
| n_iter FORM | 1 |
| fc* (MPa) | 72.4645 |
| fy* (MPa) | 664.7030 |
| u* | [0.327, 2.1595] |
| dg/du_fc (HF@u*) | 0.008780 |
| dg/du_fy (HF@u*) | 0.039239 |
| Importance fc | 2.24% |
| Importance fy | 97.76% |
| beta | 2.1841 |
| Pf | 9.8552e-01 |
| u* FOSM (HF) | [0.4327, 2.2351] |
| Erreur FOSM | 5.95% |

FORM_all_modes : 1 mode (21 points de depart), beta=2.2789, u*=[0.503, 2.223].

**Observation :** regime inverse confirme (u* positif, Pf=98.5%). F=0.12 MN trop grande.

---

## Partie 2 — Corrections du code

### 1. Bug `d` dans `calc_ana()` (ligne 189)

**Probleme :** `d = h - sum(z_rebar) / len(z_rebar)`

Les coordonnees z_rebar sont extraites des appels `POINT(...)` dans dsCad.txt. Elles sont mesurees depuis l'axe neutre de la section (origine a h/2), pas depuis la fibre superieure comprimee.

La hauteur utile d (distance fibre comprimee → centroide des aciers) doit s'ecrire :
`d = h/2 + z_centroide`

**Fix applique :**
```python
# AVANT :
d = h - sum(z_rebar) / len(z_rebar)
# APRES :
d = h/2 + sum(z_rebar) / len(z_rebar)
```

Pour b=0.2, h=0.5, 3x1HA32 (z = +0.202, +0.170, +0.138m) : z_centroide = 0.170m.
- Avant fix : d = 0.5 - 0.170 = 0.330m (FAUX)
- Apres fix : d = 0.25 + 0.170 = 0.420m (CORRECT)

### 2. Parametre `fac_c=1.1` supprime

Retire de la signature `flexion_simple.__init__`. Inutilise dans le corps de la fonction.

```python
# AVANT :
def __init__(self, Med, As, b, h, d, Es, ecu, fc_otparams, fy_otparams, fac_c=1.1):
# APRES :
def __init__(self, Med, As, b, h, d, Es, ecu, fc_otparams, fy_otparams):
```

### 3. `print_error_ana_hf` — correction `n_visu` → `n_grid_hf`

La fonction ecrite par Claude utilisait `n_visu` (non defini dans OPTIONS). Corrige par `n_grid_hf` (valeur=7 dans OPTIONS). La normalisation par std(g_HF) a egalement ete supprimee ulterieurement — remplacee par err_abs_moy = mean(|g_HF - g_ana|), plus directement interpretable.

### 4. OPTIONS en vigueur apres corrections

```python
print_DOE = True
print_ana = True
print_ana_hf_error = True
size_visu = 5
n_grid_hf = 7
```

---

## Partie 3 — Run avec fixes + print_error_ana_hf (output_0405_0810.txt)

**Config :** identique au Run 1 + fix d + fix fac_c + n_visu→n_grid_hf. DOE identique (graine OT fixe).

| Parametre | Valeur |
|---|---|
| n_iter FORM | 1 |
| fc* (MPa) | 72.4642 |
| fy* (MPa) | 664.6959 |
| u* | [0.3269, 2.1592] |
| dg/du_fc (HF@u*) | 0.008780 |
| dg/du_fy (HF@u*) | 0.039239 |
| Importance fc | 2.24% |
| Importance fy | 97.76% |
| beta | 2.1842 |
| Pf | 9.8551e-01 |
| u* FOSM (HF) | [0.4327, 2.2351] |
| Erreur FOSM | 5.96% |

FORM_all_modes : 1 mode, beta=2.2785, u*=[0.503, 2.222].

### Resultats print_error_ana_hf

```
Points sur la frontière : 37
comp_grid : 19 points (1 sur 2)
--- Validation g_HF vs g_ana sur frontière ---
  pt  0 u=( -3.72,  4.45)  g_HF=+0.0206  g_ana=-0.0000  err_rel=8.3306
  pt  1 u=( -3.46,  4.31)  g_HF=+0.0207  g_ana=-0.0000  err_rel=8.3661
  pt  2 u=( -3.21,  4.18)  g_HF=+0.0199  g_ana=+0.0000  err_rel=8.0493
  pt  3 u=( -2.69,  3.94)  g_HF=+0.0202  g_ana=+0.0000  err_rel=8.1725
  pt  4 u=( -2.18,  3.72)  g_HF=+0.0202  g_ana=+0.0000  err_rel=8.1801
  pt  5 u=( -1.67,  3.51)  g_HF=+0.0214  g_ana=+0.0000  err_rel=8.6754
  pt  6 u=( -1.15,  3.32)  g_HF=+0.0230  g_ana=+0.0000  err_rel=9.3046
  pt  7 u=( -0.64,  3.14)  g_HF=+0.0246  g_ana=-0.0000  err_rel=9.9447
  pt  8 u=( -0.13,  2.98)  g_HF=+0.0240  g_ana=-0.0000  err_rel=9.6919
  pt  9 u=(  0.38,  2.82)  g_HF=+0.0222  g_ana=-0.0000  err_rel=8.9658
  pt 10 u=(  0.90,  2.67)  g_HF=+0.0210  g_ana=-0.0000  err_rel=8.4796
  pt 11 u=(  1.41,  2.54)  g_HF=+0.0203  g_ana=+0.0000  err_rel=8.2306
  pt 12 u=(  1.92,  2.41)  g_HF=+0.0193  g_ana=+0.0000  err_rel=7.8253
  pt 13 u=(  2.44,  2.29)  g_HF=+0.0190  g_ana=+0.0000  err_rel=7.6991
  pt 14 u=(  2.95,  2.17)  g_HF=+0.0178  g_ana=+0.0000  err_rel=7.1874
  pt 15 u=(  3.46,  2.06)  g_HF=+0.0175  g_ana=-0.0000  err_rel=7.0930
  pt 16 u=(  3.97,  1.96)  g_HF=+0.0165  g_ana=-0.0000  err_rel=6.6600
  pt 17 u=(  4.49,  1.86)  g_HF=+0.0162  g_ana=-0.0000  err_rel=6.5391
  pt 18 u=(  5.00,  1.77)  g_HF=+0.0153  g_ana=+0.0000  err_rel=6.1911
  → err_rel_moy = 8.0835  (std g_HF = 0.0025)
```

**Interpretation :**
- g_HF > 0 partout sur la frontiere analytique → la structure est "sure" pour STRAINS la ou g_ana=0 → STRAINS est plus permissif que la formule analytique.
- Decalage absolu ≈ +0.02 (≈ 2% de la valeur typique de g) : systematique sur toute la frontiere.
- err_rel grand (6-10) car std(g_HF)=0.0025 tres faible (les valeurs HF sont tres groupees).
- Conclusion : les deux modeles (STRAINS et formule analytique avec d=0.330m) sont d'accord en forme mais decales. Le fix de d (d=0.420m au lieu de 0.330m) va deplacer la frontiere analytique — a verifier au prochain run calibre.

---

## Partie 4 — Run de confirmation (output_0405_0819.txt)

**Config :** identique au Run 0810 (relance sans modification).

| Parametre | Valeur |
|---|---|
| n_iter FORM | 1 |
| fc* (MPa) | 72.4651 |
| fy* (MPa) | 664.7062 |
| u* | [0.3271, 2.1596] |
| dg/du_fc (HF@u*) | 0.008781 |
| dg/du_fy (HF@u*) | 0.039239 |
| Importance fc | 2.24% |
| Importance fy | 97.76% |
| beta | 2.1838 |
| Pf | 9.8551e-01 |
| u* FOSM (HF) | [0.4327, 2.2351] |
| Erreur FOSM | 5.94% |

print_error_ana_hf : resultats identiques au run 0810 (memes 19 points, err_rel_moy=8.0835, std=0.0025). Reproductible.

---

---

## Partie 5 — Nouvelle geometrie b=0.4, h=0.45, 2x2HA32

### Modifications dsCad/code avant ces runs

- **dsCad.txt :** b=0.4m, h=0.45m, phi=32mm, 2 lits de 2HA32. Lit 1 z=+0.165m (60mm du bord), Lit 2 z=+0.125m (100mm du bord). Centroide a 80mm du bord. Barres a y=±0.10m.
- **sensitivity_regions :** `["HA1","HA2","HA3","HA4"]`
- **print_error_ana_hf :** normalisation err_rel supprimee → err_abs = |g_HF - g_ana|, summary = err_abs_moy.
- **Corrections code :** `d = h/2 + sum(z_rebar)/len(z_rebar)` (ligne 189) ; `fac_c=1.1` supprime.

### Run 5a — output_0405_0853.txt

**Jeu de donnees :** b=0.4, h=0.45, 2x2HA32, n0=20, fck=63, fyk=550, F=0.12 MN.

| Parametre | Valeur |
|---|---|
| n_iter FORM | 1 |
| fc* (MPa) | 70.2676 |
| fy* (MPa) | 541.4991 |
| u* | [-0.113, -1.927] |
| Importance fc / fy | 0.34% / 99.66% |
| beta | 1.9303 |
| Pf | 2.68e-02 |
| Erreur FOSM | 1.61% |

print_error_ana_hf : 40 points → 20 retenus. **Regime normal** (u* negatif). beta trop faible.

### Run 5b — output_0405_0919.txt

**Jeu de donnees :** b=0.4, h=0.45, 2x2HA32, n0=5, fck=48, fyk=550, F=0.12 MN.

| Parametre | Valeur |
|---|---|
| n_iter FORM | 1 |
| fc* (MPa) | 55.4077 |
| fy* (MPa) | 546.8251 |
| u* | [-0.117, -1.750] |
| Importance fc / fy | 0.45% / 99.55% |
| beta | 1.7542 |
| Pf | 3.97e-02 |
| Erreur FOSM | 29.4% |

print_error_ana_hf : err_abs_moy = 0.034. Point suspect pt 1 u=(-4.74, 4.16) : g_ana=+0.093 (mauvaise branche brentq). FOSM 29% : n0=5 insuffisant.

### Runs interrompus (0923, 0927, 0933, 0939, 0949)

Relances avec modifications incrementales entre chaque run. Tous interrompus — resultats non obtenus.

### Run 5c — output_0405_0957.txt

**Jeu de donnees :** b=0.4, h=0.45, 2x2HA32, n0=5, fck=48, fyk=550, **F=0.11 MN**.

| Parametre | Valeur |
|---|---|
| n_iter FORM | 1 |
| fc* (MPa) | 30.2623 |
| fy* (MPa) | 564.6871 |
| u* | [-1.392, -1.158] |
| dg/du_fc (HF@u*) | 0.035511 |
| dg/du_fy (HF@u*) | 0.028845 |
| Importance fc / fy | 59.11% / 40.89% |
| beta | 1.8108 |
| Pf | 3.51e-02 |
| Erreur FOSM | 9.1% |

**Observation :** fc passe de <1% (F=0.12) a 59% (F=0.11) — la frontiere a bascule vers une zone ou fc est fortement sollicite. fc*=30.26 MPa tres faible (u_fc=-1.39). print_error_ana_hf interrompu avant completion.

### Run 5d — output_0405_1035.txt

**Jeu de donnees :** identique au Run 5c (F=0.11 MN). Code corrige : SyntaxError ligne 899 (`yt, all_grad, all_sensib = None, None, None`) + print listes ajouté en ligne 789.

| Parametre | Valeur |
|---|---|
| n_iter FORM | 1 |
| fc* (MPa) | 30.2604 |
| fy* (MPa) | 564.6717 |
| u* | [-1.393, -1.158] |
| dg/du_fc (HF@u*) | 0.035509 |
| dg/du_fy (HF@u*) | 0.028845 |
| Importance fc / fy | 59.11% / 40.89% |
| beta | 1.8108 (mode 1 : 1.9292) |
| Pf | 3.50e-02 |
| Erreur FOSM | 9.05% |

**print_error_ana_hf — 19 points, err_abs_moy = 0.0321 :**

| pt | u1 | u2 | g_HF | err_abs |
|---|---|---|---|---|
| 0 | -1.410 | -0.470 | 0.02521 | 0.02521 |
| 1 | -1.154 | -0.678 | 0.02654 | 0.02654 |
| 2 | -0.897 | -0.871 | 0.02877 | 0.02877 |
| 3 | -0.641 | -1.049 | 0.03159 | 0.03159 |
| 4 | -0.385 | -1.215 | 0.03465 | 0.03465 |
| 5 | -0.128 | -1.369 | 0.03666 | 0.03666 |
| 6 | +0.128 | -1.514 | 0.03708 | 0.03708 |
| 7 | +0.385 | -1.649 | 0.03731 | 0.03731 |
| 8 | +0.641 | -1.776 | 0.03724 | 0.03724 |
| 9 | +0.897 | -1.896 | 0.03737 | 0.03737 |
| 10 | +1.154 | -2.009 | 0.03784 | 0.03784 |
| 11 | +1.410 | -2.115 | 0.03814 | 0.03814 |
| 12 | +1.923 | -2.311 | 0.03945 | 0.03945 |
| 13 | +2.436 | -2.487 | 0.03846 | 0.03846 |
| 14 | +2.949 | -2.646 | 0.03407 | 0.03407 |
| 15 | +3.462 | -2.790 | 0.02946 | 0.02946 |
| 16 | +3.974 | -2.921 | 0.02500 | 0.02500 |
| 17 | +4.487 | -3.040 | 0.02125 | 0.02125 |
| 18 | +5.000 | -3.149 | 0.01747 | 0.01747 |

Biais non uniforme : max ≈ 0.039 autour de u1≈[1, 2], decroit vers les extremes. Superieur a l'ancienne geometrie (etait 0.020).

---

## Modifications code apportees en session (complement)

- **print_error_ana_hf ligne 789 :** print des listes brutes ajouté (`u_grid`, `g_HF_vals`, `g_ana_vals`).
- **Normalisation err_rel → err_abs :** suppression de la division par std(g_HF), remplacee par err_abs_moy = mean(|g_HF - g_ana|).
- **SyntaxError ligne 899 corrigee :** `yt, all_grad, all_sensib = None, None, None` (branche PCE+KRG inactive).
- **u1_bornes / u2_bornes :** parametres introduits dans OPTIONS pour borner la grille HF de print_visu.

---

## Partie 6 — Modifications print_error_ana_hf (apres-midi 04/05)

### 6.1 Ajout fc/fy dans le tableau de validation

Avant la boucle d'affichage, `dist_jointe()` et `T_inv` sont instancies une seule fois. Dans la boucle, `x = T_inv(ot.Point(list(pt)))` convertit chaque point u en espace physique.

```python
# AVANT :
for i, pt in enumerate(error_grid):
    print(f"  pt {i:2d} u=({pt[0]:6.2f},{pt[1]:6.2f})  "
        f"g_HF={g_HF_vals[i]:+.4f}  g_ana={g_ana_vals[i]:+.4f}  "
        f"err_abs={err_abs[i]:.4f}")

# APRES :
dist_X = dist_jointe()
T_inv  = dist_X.getInverseIsoProbabilisticTransformation()
for i, pt in enumerate(error_grid):
    x = T_inv(ot.Point(list(pt)))
    print(f"  pt {i:2d} u=({pt[0]:6.2f},{pt[1]:6.2f})  "
        f"fc={x[0]:6.2f}  fy={x[1]:6.2f}  "
        f"g_HF={g_HF_vals[i]:+.4f}  g_ana={g_ana_vals[i]:+.4f}  "
        f"err_abs={err_abs[i]:.4f}")
```

**Difference :** ajout de deux colonnes fc (MPa) et fy (MPa) dans le tableau. Permet de localiser les points de la frontiere directement en espace physique. Pas de changement de logique.

---

### 6.2 Inversion du scan u1/u2 : u2 varie en externe, decroissant

**Contexte :** dans la version precedente, u1 variait sur la grille externe et brentq cherchait u2. Modifie pour faire varier u2 sur la grille externe (decroissant de size_visu vers u2_low) et chercher u1 par brentq.

```python
# AVANT :
pts = []
u2_scan = np.linspace(u2_low, size_visu, n_scan)
for u1 in np.linspace(-size_visu, size_visu, 40):
    g_vals = [calc.g_ana([u1, u2]) for u2 in u2_scan]
    for i in range(len(u2_scan) - 1):
        if g_vals[i] * g_vals[i+1] < 0:
            u2_star = brentq(
                lambda u2: calc.g_ana([u1, u2]), u2_scan[i], u2_scan[i+1])
            pts.append([u1, u2_star])

# APRES :
pts = []
u1_scan = np.linspace(-size_visu, size_visu, n_scan)
for u2 in np.linspace(size_visu, u2_low, 40):  # u2 decroissant, du haut vers le bas
    g_vals = [calc.g_ana([u1, u2]) for u1 in u1_scan]
    for i in range(len(u1_scan) - 1):
        if g_vals[i] * g_vals[i+1] < 0:
            u1_star = brentq(
                lambda u1: calc.g_ana([u1, u2]), u1_scan[i], u1_scan[i+1])
            pts.append([u1_star, u2])
```

**Differences :**
- Variable externe : u1 fixe → u2 fixe (decroissant de 5 vers u2_low).
- Variable interne (cherchee par brentq) : u2 → u1.
- Ordre des points dans frontier_pts : ordonnes par u2 decroissant (des fy forts vers les fy faibles) au lieu de u1 croissant.
- Le format de sortie `[u1_star, u2]` reste inchange — le reste du code n'est pas affecte.

**Pourquoi ce changement :** pour que les points de la frontiere soient enumeres dans un ordre physiquement parlant (fy elevees en premier, fy faibles en dernier), ce qui correspond a parcourir la frontiere de haut en bas dans l'espace U.

**Consequence observee :** cette modification a revele un probleme : pour u2 eleve (fy ≥ 595 MPa), g_ana presente des discontinuites (transition regime plastique/non-plastique dans `test_plast`), detectees a tort comme des changements de signe. Cf. 6.3.

---

### 6.3 Filtre post-brentq : rejet des faux zeros (discontinuites)

**Contexte :** apres la modif 6.2, le tableau montrait 16 points dont 11 avec g_ana ≠ 0 (valeurs 0.006 a 0.17). Ces points ne sont pas sur la frontiere g_ana=0 — brentq convergeait sur des discontinuites de g_ana liees au switch binaire `test_plast` (transition regime plastique/non-plastique). Quand `test_plast` bascule de 1 a 0, les deux formules g_ana_plast et g_ana_nonplast ne sont pas egales, ce qui cree un saut de signe dans g_ana sans vrai zero.

```python
# AVANT :
                    u1_star = brentq(
                        lambda u1: calc.g_ana([u1, u2]), u1_scan[i], u1_scan[i+1])
                    pts.append([u1_star, u2])

# APRES :
                    u1_star = brentq(
                        lambda u1: calc.g_ana([u1, u2]), u1_scan[i], u1_scan[i+1])
                    if abs(calc.g_ana([u1_star, u2])) < 1e-6:
                        pts.append([u1_star, u2])
```

**Difference :** apres brentq, on reevalue g_ana au point trouve. Si |g_ana| >= 1e-6, le point est rejete (c'est une discontinuite, pas un vrai zero). Un vrai zero analytique atteint une precision bien inferieure a 1e-6 avec brentq. Cout : un appel supplementaire a g_ana par candidat (negligeable, fonction analytique).

**Pour revenir en arriere :** supprimer la ligne `if abs(calc.g_ana([u1_star, u2])) < 1e-6:` et dedenter `pts.append(...)`.

---

## Taches en suspens

- **Calibration F :** F=0.11 → beta=1.81, fc 59%. F trop faible ou geometrie a revoir. Cible beta ≈ 3-4.
- **n0 insuffisant :** n0=5 donne instabilite. Augmenter pour runs de production.
- **Biais STRAINS/analytique plus eleve (0.032 vs 0.020) :** a investiguer — formule analytique valide pour cette geometrie ?
- **sample_frontier + refonte visu HF :** plan documente dans global md section 9, non encore implemente.
