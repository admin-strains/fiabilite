# Résultats FORM KRG pur — Run 3
**Date :** 23 avril 2026  
**Configuration :** KRG pur, n0=15, DOE fixé, F=0.210 MN, noyau SquaredExponential, solver AbdoRackwitz  
**Référence HF :** β=3.784, Pf=7.73e-05, u*=[-0.526, -3.747]

---

### F = 0.210 MN (β_HF ≈ 3.78)

| Paramètre | Valeur |
|---|---|
| **— Résultats FORM KRG —** | |
| n points DOE | 15 (fixé) |
| fc* (MPa) | 26.7302 |
| fy* (MPa) | 475.1147 |
| u* | [-1.756, -4.750] |
| dg/du_fc en u* | 0.004838 |
| dg/du_fy en u* | 0.002553 |
| Importance fc | 12.03% |
| Importance fy | 87.97% |
| β (FORM) | 5.065 |
| Pf (FORM) | 2.05e-07 |
| n_appels HF (FORM) | 0 |
| n_iter FORM | 24 |
| **— Test GP au point de FORM —** | |
| g_HF(u*) | -4.96e-02 |
| g_KRG(u*) | +9.3e-05 |
| Erreur relative g | 100.2% |
| **— Comparaison HF —** | |
| β (FORM HF) | 3.784 |
| Ecart β (KRG vs HF) | +1.281 (33.9%) |
| **— Test linéarisation FOSM —** | |
| u* FORM | [-1.756, -4.750] |
| u* FOSM (depuis u=0) | [-0.643, -3.728] |
| Erreur relative u* FORM/FOSM | 29.84% |

---

### Comparaison avec runs précédents (F=0.210 MN, KRG pur)

| | **HF** | **Run2 n0=15 (LHS aléatoire)** | **Run3 n0=15 (DOE fixé)** |
|---|---|---|---|
| β | 3.784 | 5.687 | 5.065 |
| Erreur β | 0% | 50.3% | 33.9% |
| u* | [-0.53, -3.75] | [-1.65, -5.44] | [-1.76, -4.75] |
| g_HF(u*) | ≈ 0 | -7.76e-02 | -4.96e-02 |
| n_iter FORM | 21 | 51 | 24 |
| Erreur FOSM | — | 34.98% | 29.84% |

**Conclusion :** Le DOE fixé améliore KRG pur par rapport au LHS aléatoire (50.3% → 33.9%), mais KRG pur reste insuffisant à β≈3.78. g_HF(u*)=-0.050 confirme que la surface limite du métamodèle est mal placée. Motivation pour PCE-KRG confirmée.

---

### DOE fixé utilisé (n0=15, U-space)

```python
U_doe_fixed = ot.Sample([
    [ 1.0272625484832025,  0.3251235065050853],
    [ 0.2588934150948534, -1.6856336900013655],
    [-0.7900915845657982,  1.8047217395005692],
    [-0.0301755082064849,  1.3223984111477798],
    [-1.8073810055112547, -1.1012751718677385],
    [-0.2377471223963969, -0.4914312425631510],
    [ 0.7216266145109314,  1.0830320538875535],
    [ 0.4776729449462016, -0.2656508781535193],
    [-0.8730465106774573,  0.6497494474356423],
    [-1.1677174906609287,  0.0310652111349381],
    [ 1.1194425579629474, -0.7943643305093363],
    [ 0.1857520921586401,  0.4724170659386679],
    [-0.5669380193636159, -1.4858232340964800],
    [ 2.9454553139272623, -0.1582987245612891],
    [-0.2947626989079067,  0.1355018527305618],
])
```
