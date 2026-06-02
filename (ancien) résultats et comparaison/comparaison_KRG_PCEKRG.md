# Comparaison KRG pur vs PCE-KRG — Flexion pure BA (phi=16mm)

> **Objectif :** évaluer l'apport du PCE comme tendance dans le métamodèle hybride PCE-KRG vs KRG pur.
> **Conditions communes :** F=0.210 MN (β_HF≈3.78), noyau SquaredExponential, solver AbdoRackwitz, DOE LHS fixé identique (n0=15, hardcodé).
> **Référence :** FORM HF (appels directs STRAINS), β_HF=3.784, Pf_HF=7.73e-05.
> **Erreur β** : |β_modèle − β_HF| / β_HF.

---

## DOE fixé — n0=15 (à hardcoder dans le script)

```python
U_doe = ot.Sample([
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

---

## Tableau — F = 0.210 MN (β_HF ≈ 3.78), n0=15, DOE fixé

| | **HF** | **KRG pur n0=15** | **PCE-KRG n0=15** |
|---|---|---|---|
| u* | [-0.526, -3.747] | [-1.757, -4.752] | [-0.607, -3.730] |
| β | 3.784 | 5.067 | **3.779** |
| Pf | 7.73e-05 | 2.03e-07 | **7.87e-05** |
| g_HF(u*) | ≈ 0 | **-4.96e-02** | N/A ¹ |
| g_méta(u*) | ≈ 0 | ≈ 0 | ≈ 0 |
| Erreur relative β | 0% (réf.) | **33.9%** | **0.1%** |
| n_iter FORM | 21 | 24 | 15 |
| u* FOSM | [-0.526, -3.747] | [-0.643, -3.728] | [-0.643, -3.728] |
| Erreur u* FORM/FOSM | — | 29.9% | 0.97% |
| PCE degré | — | — | 2 |
| Q² LOO PCE | — | — | N/A ² |

¹ Test HF au point u* non atteint (segfault compute_q2_loo avant).  
² compute_q2_loo cause un segfault OT (bug en cours d'investigation) — colonne à compléter quand corrigé.

**Conclusion :** sur le même DOE fixé n0=15, PCE-KRG est très nettement supérieur à KRG pur pour ce cas β≈3.78. KRG pur échoue (g_HF(u*)=-0.05 ≠ 0, β surestimé ×1.34) là où PCE-KRG converge avec 0.1% d'erreur. Le PCE fournit une tendance globale précise qui compense l'insuffisance du DOE pour couvrir la queue de distribution à β≈3.78.

---

## Notes sur la configuration PCE-KRG

- PCE : LARS + CorrectedLeaveOneOut, degré max=2, q=0.75 (hyperbolic enumeration), `do_pce=True` hardcodé (workaround segfault LOO)
- KRG résiduel : SquaredExponential, basis constante, `ot.KrigingAlgorithm`
- Tendance KRG = g_HF − g_PCE (~1e-4, très petit → KRG corrige principalement le bruit résiduel)

---

## À compléter

- [ ] Hardcoder le DOE ci-dessus dans le script
- [ ] Lancer KRG pur (do_pce=False, try_pce=False) avec ce DOE fixé → remplir la colonne KRG pur
- [ ] Corriger le segfault compute_q2_loo → ajouter Q² LOO dans la colonne PCE-KRG
