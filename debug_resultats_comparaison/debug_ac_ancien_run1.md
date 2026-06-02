# Debug GEK — Resultats ac_ancien_ref runs 1-3 (Test A)
**Date :** 24 avril 2026 — 15h24
**Source :** `ac_ancien_ref.py` copie dans `AC_pure_flexion.py`
**Config :** do_GEK=True, do_GEK_analytic_grad=False (FD explicite), do_warm_start=False, n0=15, DOE fixe hardcode, F=0.210 MN
**Output :** `output/output_2404_1524.txt`
**Reference HF :** beta=3.784, u*=[-0.526, -3.747]

---

## Resultats FORM

| Parametre | Valeur |
|---|---|
| n points DOE | 15 (fixe hardcode) |
| fc* (MPa) | 33.154 |
| fy* (MPa) | 576.148 |
| u* [u_fc, u_fy] | [+0.286, -1.399] |
| Importance fc | 4.02% |
| Importance fy | 95.98% |
| beta (FORM) | **1.428218** |
| Pf (FORM) | 7.661e-02 |
| n_iter FORM | 18 |
| g_HF(u*) | 0.102979 |
| g_GEK(u*) | 0.103033 |
| Erreur relative g | 0.05% |
| u* FOSM | [-0.643, -3.728] |
| Erreur FOSM | 175.6% |
| Ecart beta vs HF | -2.356 (-62.3%) |

---

## Observations critiques

### 1. u* completement faux
u* = [+0.286, -1.399] : u_fc est POSITIF (cote sur de fc) alors que la defaillance necessite fc faible (u_fc negatif). FORM a converge dans la mauvaise direction. Reference : u*_HF ≈ [-0.526, -3.747].

### 2. Validation gradient GEK (bloc avant FORM)
```
Point test u = [-1.2, -3.0]
Var        Analytique     FD centre      Err rel
u_0      6.282623e-03   1.066816e-02      41.11%
u_1      4.162694e-02   3.398014e-02      22.50%
```
**Gradient analytique GEKPLS tres imprecis sur ce run.** Erreur 41% sur u_fc, 22% sur u_fy. Ce training a produit des theta donnant de mauvaises derivees. Cela confirme le non-determinisme GEKPLS : le modele interpole bien les valeurs (g_GEK(u*)=0.103033 vs g_HF(u*)=0.102979, erreur 0.05%) mais les gradients sont faux.

### 3. Test au point de reference HF
```
g_GEK(u*_HF=[-0.1595, -0.9398]) = 0.118921  (attendu : g ≈ 0)
```
Le metamodele ne predit pas g≈0 au voisinage du point de defaillance HF — confirme que le modele est globalement mauvais pour ce training.

Note : le point u*_HF = [-0.1595, -0.9398] hardcode dans l'ancien code correspond probablement a une ancienne reference (F different). Pour F=0.210 MN, u*_HF ≈ [-0.526, -3.747].

---

## Conclusion

Ce run confirme que l'ancien code avec FD explicite est aussi non-deterministe que le nouveau code. La GEKPLS training de ce run a produit des gradients inaccurats (41%, 22%) → FORM mal guide → convergence vers faux u*. Le beta=3.774 du run historique de l'ancien code (avec do_GEK_analytic_grad=True) etait soit un run chanceux avec bons theta, soit il y a une difference liee a la convention de shape du gradient.
