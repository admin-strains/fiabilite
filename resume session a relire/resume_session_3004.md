# Resume session 30 avril 2026 — Nouvelle geometrie + calibration charge

---

## Partie 1 — Runs exploratoires geometrie + charge (session du 30/04)

### Contexte

Passage a une nouvelle geometrie pour les tests de fiabilite GEK. Plusieurs geometries et charges testees au fil de la session.
Modifications systematiques de dsCad.txt (b, h, phi, rebars) et dsLoad.txt (F).
La plupart des runs intermediaires ont ete arretes manuellement avant la fin.

---

### Run 1 — b=0.5, h=0.8, 2×3HA16, F=0.1 MN (output_3004_1447.txt)

**Config :** do_GEK=True, n0=35, fck=40, fyk=500, U_doe_fixed=None, do_warm_start=False

| Parametre | Valeur |
|---|---|
| b (m) | 0.5 |
| h (m) | 0.8 |
| phi (mm) | 16.0 |
| Armatures | 2 lits 3HA16 (z_lit1=0.328m, z_lit2=0.312m) |
| F (MN) | 0.1 |
| n0 | 35 |
| fck (MPa) | 40 |
| fyk (MPa) | 500 |

**Resultats :** beta=0.3771, Pf=6.47e-01, u*=[0.006, 0.377], fc*=47.83, fy*=560.97, n_iter=1, Imp=[0.02%, 99.98%], Erreur FOSM=4.65%.
Ref resultats : `resultats_GEK_run1.md` (Run 1).

**Observation :** F=0.1 MN trop faible — beta tres bas.

---

### Run 2 — b=0.2, h=0.6, 2×1HA16, F=0.022 MN (output_3004_1549.txt)

**Config :** do_GEK=True, n0=5, fck=28, fyk=550, U_doe_fixed=None, do_warm_start=False

| Parametre | Valeur |
|---|---|
| b (m) | 0.2 |
| h (m) | 0.6 |
| phi (mm) | 16.0 |
| Armatures | 2 lits 1HA16 centre (z_lit1=0.248m, z_lit2=0.232m) |
| F (MN) | 0.022 |
| n0 | 5 |
| fck (MPa) | 28 |
| fyk (MPa) | 550 |

**Resultats :** beta=3.1643, Pf=7.77e-04, u*=[-0.024, -3.164], fc*=35.64, fy*=504.20, n_iter=1, Imp=[0.01%, 99.99%], Erreur FOSM=0.56%.
Ref resultats : `resultats_GEK_run1.md` (Run 2).

**Observation :** Bien calibre. u* negatif (regime normal).

---

### Runs intermediaires arretes — output_3004_1513 a output_3004_1753

Plusieurs runs arretes manuellement au cours de la session lors de l'exploration de differentes geometries (b=0.3, h=0.4, 2×2HA20 ; b=0.2, h=0.5, geometries variees). Non documentes en detail — resultats partiels ou nuls.

---

### Run 3 — b=0.2, h=0.5, 3×1HA32, F=0.12 MN (output_3004_1807.txt)

**Config :** do_GEK=True, n0=20, fck=63, fyk=550, U_doe_fixed=None (graine OT fixe), do_warm_start=False

| Parametre | Valeur |
|---|---|
| b (m) | 0.2 |
| h (m) | 0.5 |
| phi (mm) | 32.0 |
| Armatures | 3 lits 1HA32 (z=+0.202m, +0.170m, +0.138m depuis axe neutre) |
| F (MN) | 0.12 |
| n0 | 20 |
| fck (MPa) | 63 |
| fyk (MPa) | 550 |

DOE imprime (print_DOE=True) :
```python
U_doe_fixed = ot.Sample([
    [-0.4627998555892081, -0.9260041697193523],
    [-0.8219457421120840, -2.6393164348337010],
    [-0.3551373235826685, -0.2645708200879325],
    [-2.0608430899277224,  0.4491946950675937],
    [-1.4117780400639317, -0.7312385984989477],
    [ 1.2799006947726792, -0.4730655065029271],
    [ 1.6347475019570152,  0.3463639206935133],
    [-0.1336467833113242,  0.7962443064455704],
    [-0.5686415665276859,  0.1639912125691569],
    [-0.9909272956547241,  0.9049948538509544],
    [ 0.1579650138209510,  0.0847109798987154],
    [-1.2634674776365697, -0.1725992187133350],
    [ 0.5183621953593466,  0.5591539750810141],
    [ 0.7816806132486308, -0.0512159725069177],
    [ 0.8419977042800881, -1.3712348086175161],
    [ 0.0605710267694066, -1.0644060298959737],
    [ 0.3736737358998102,  1.5435472922371742],
    [ 1.7330974613641246,  1.1823289997171520],
    [ 0.5539183403720471, -0.5276627295149516],
    [-0.1055510289089516,  2.4466148569242270],
])
```

**Resultats :**

| Parametre | F=0.12 MN |
|---|---|
| n_iter FORM | 1 |
| fc* (MPa) | 72.4652 |
| fy* (MPa) | 664.7061 |
| u* | [0.3271, 2.1596] |
| dg/du_fc (HF@u*) | 0.008781 |
| dg/du_fy (HF@u*) | 0.039239 |
| Importance fc | 2.24% |
| Importance fy | 97.76% |
| beta | 2.1842 |
| Pf | 9.8553e-01 |
| n_appels HF DOE | 20 |
| n_appels HF FORM | 0 |
| u* FOSM (HF) | [0.4327, 2.2351] |
| Erreur FOSM | 5.94% |

FORM_all_modes : 1 mode, beta=2.279, Pf=0.9887, u*=[0.5, 2.223].

**Observations :**
- u* positif sur les deux composantes → origine en domaine de defaillance → regime inverse.
- F=0.12 MN trop grande : memes proprietes moyennes des materiaux → structure en defaillance.
- beta=2.18, Pf=0.985 = Phi(+2.18) : pas le regime cible (on veut Pf petite, u* negatif).
- Importance fy=97.76% : fy domine (coherent avec toutes les autres geometries).
- Erreur FOSM 5.94% : acceptable.

---

## Partie 2 — Modifications du code (session 30/04)

### print_results — vrais gradients HF en u*

`print_results` modifie pour utiliser `run_HF(u_star)` et `run_HF(u0)` au lieu des gradients du metamodele.
Blocs GEK et KRG : calculent les vraies sensibilites HF au point u* trouve par FORM (meme si u* est legerement faux).
Raison : les gradients du GEK peuvent etre bruites ; les gradients HF sont la reference physique.

### matplotlib TkAgg

`import matplotlib` + `matplotlib.use('TkAgg')` ajoutes avant `import matplotlib.pyplot` pour forcer le backend interactif et afficher la fenetre visu.

### size_visu

Variable `size_visu = 32` ajoutee dans OPTIONS. Utilisee dans `print_visu` pour `np.linspace(-size_visu, size_visu, ...)`.

### print_visu

Copie verbatim archivee dans `global_resume_session_2404.md` (section 9 — sample_frontier) pour reference avant modification future.

### Plan sample_frontier

Plan complet documente dans `global_resume_session_2404.md` (section 9). Integration non realisee ce jour.
