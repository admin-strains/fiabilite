# Debug GEPCK — Analyse des bugs (29/05/2026)

---

## Bug 1 — BLOQUANT : NaN Matern-5/2 dans `branche5.py:1120`

**Cause directe de l'échec du run (FORM beta=8.5 sur tous les starts).**

`branche5.py:1118-1124` — cas `[i,j]` avec `i≠j` dans `kernel_deriv_factory` :
```python
else:
    i, j = der, der_prime
    K_excl_ij = K_excl[:, :, i] / K_uni[:, :, j]   # ← 0/0 = NaN
```

Quand deux points d'entraînement sont éloignés, `K_uni[:,:,j] = (1+a+a²/3)·exp(-a) → 0` par underflow. `K_excl[:,:,i]` contient aussi ce facteur → `0/0 = NaN`. Ce NaN se propage dans R̃ → Rinv → `YMu = f0 @ beta + r0 @ (Rinv @ residual)` (`branche4.py:206`) → toutes les prédictions NaN → FORM dérive vers la frontière beta=8.5.

**Pour M=2 la valeur correcte est 1** (produit vide). Fix minimal :

```python
K_excl_ij = np.where(
    np.abs(K_uni[:, :, j]) > 1e-300,
    K_excl[:, :, i] / K_uni[:, :, j],
    0.0   # exp(-(ai+aj)) → 0 simultanément → terme global = 0
)
```

---

## Bug 2 — LOGIQUE : `run_EFF` utilise le surrogate à la place du modèle HF

`AC2_pure_flexion.py:1262-1267` :
```python
g_val    = g_ot(ot.Point(u_opt))[0]            # ← prédiction GEPCK, pas STRAINS
xt       = np.vstack([xt, [np.array(u_opt)]])
yt       = np.vstack([yt, [[g_val]]])
grad_ot  = g_ot.gradient(ot.Point(u_opt))       # ← gradient GEPCK, pas STRAINS
grad_val = np.array([[grad_ot[i, 0] for i in range(n_var)]])
```

L'EFF est censée enrichir le DOE avec de vraies évaluations HF aux points d'incertitude maximale. Ici elle ajoute ses propres prédictions comme observations, ce qui est circulaire — le surrogate refitté interpole ses propres valeurs, la variance baisse mais la précision ne s'améliore pas. Il faudrait :

```python
g_val, grad_U, _ = run_HF(np.array(u_opt))     # vrai appel STRAINS
xt       = np.vstack([xt, [np.array(u_opt)]])
yt       = np.vstack([yt, [[g_val]]])
grad_val = np.array([[float(grad_U[i]) for i in range(n_var)]])
```

---

## Bug 3 — MINEUR : second bloc `if do_GEPCK:` (ligne 1750) recrée un DOE from scratch

`AC2_pure_flexion.py:1751` :
```python
xt, yt, all_grad = init_surrogate()   # → build_DOE() → nouveau LHS aléatoire
```

Ce bloc (visu standalone du GEPCK) génère un second DOE indépendant, perdant le DOE EFF-enrichi du bloc FORM. Ce n'est pas la cause du run raté (le run s'arrête à `sys.exit(1)` ligne 1742 avant d'y arriver), mais si FORM réussit, il y a un double batch de calculs STRAINS inutiles.

---

## Résumé des corrections nécessaires

| # | Fichier | Ligne | Gravité | Action |
|---|---|---|---|---|
| 1 | branche5.py | 1120 | **BLOQUANT** | Remplacer `K_excl[:,:,i] / K_uni[:,:,j]` par `np.where(abs > 1e-300, ..., 0)` |
| 2 | AC2_pure_flexion.py | 1262-1267 | **Logique** | Remplacer `g_ot(u_opt)` et `g_ot.gradient(u_opt)` par `run_HF(u_opt)` |
| 3 | AC2_pure_flexion.py | 1751 | Mineur | Réutiliser `xt, yt, all_grad` du bloc FORM au lieu de recréer un DOE |
