# Debug GEPCK sigma — EFF mauvais comportement
**Date :** 20/05/2026 — session en cours

---

## 1. Problème central

`_exec_sigma` de `GEPCKFunction` et `GEKPLSFunction` fournit le sigma utilisé par EFF.
Quel que soit le sigma utilisé, EFF ajoute des points AUX COINS du domaine au lieu de la surface limite.

**Référence fonctionnelle :** PCKRG+EFF converge proprement en 4 pts EFF sur la surface limite → beta=7.966 (-0.16% vs HF=8.1235).

---

## 2. Historique des tentatives sigma

### Tentative 1 — predict_variances SMT (original)
```python
def _exec_sigma(self, u):
    return float(np.sqrt(self.sm.predict_variances(np.array(u).reshape(1, -1)).item()))
```
**Résultat :** sigma uniforme (~sqrt(sigma2)) partout, y compris aux points DOE.
**Cause :** theta=[20,20] (borne haute sans bounds) → portée corrélation très courte → sigma remonte à max à 0.4 unité de chaque DOE → EFF voit sigma quasi-constant → va aux coins.

### Tentative 2 — theta_bounds=[1,5] + predict_variances (output_2005_1156)
Bounds ajoutées dans build_metamodel_GEK. predict_variances inchangé.
**Résultat :** 6 EFF pts, beta=7.258 (-10.6%). Convergence mais erreur importante. max_of_maxdegree=1.

### Tentative 3 — Morris 1993 gradient-enhanced posterior variance
Implémenté dans _exec_sigma des deux classes. Formule :
```
th_eff[m] = Σ_l θ_l × W[m,l]²   (kernel ARD SE équivalent)
K_tot = [[K_ff, K_fd], [K_fd.T, K_dd]]
sigma²_GEK(x) = sigma2 × max(0, 1 − k^T K_tot^{-1} k)
```
Vérifié sur sources SMT distances.py : kernel SMT squar_exp = ARD SE avec th_eff[m] = Σ_l θ_l × W[m,l]².
**Sanity check :** à x=xi, k = K_tot[i,:] → B=0 → sigma=0 ✓

**Résultat :**
- Iterations 1-3 (degree=1, sigma2~0.004) : sigma=0 au DOE ✓, EFF sur surface limite ✓
- Iteration 4+ (upgrade degree 1→2, sigma2~0.20-0.45) : EFF va aux coins ✗

**Screenshots comparatifs PCKRG vs GEPCK au même stade (3 EFF pts, après upgrade) :**
- PCKRG sigma : max ~0.45, smooth, EFF RESTE sur surface limite ✓
- GEPCK Morris 1993 : max ~0.45, sigma anisotrope (réduit dans direction du gradient) → EFF voit surface limite comme "connue" → va aux coins ✗

**Diagnostic utilisatrice :** "ça ne vient pas du degré, ça vient de sigma"
Preuve : PCKRG a le MÊME upgrade degree et le MÊME LARS explosion. Mais OT Kriging sigma (isotrope) → EFF reste sur surface limite. Morris 1993 sigma (anisotrope, réduit dans direction gradient) → EFF va aux coins.

**Hypothèse mécaniste :** blocs K_fd/K_dd réduisent sigma dans la direction du gradient depuis chaque DOE. Or les DOE pointent vers la surface limite. Résultat : sigma faible près de la surface limite → EFF ne la considère plus comme "inconnue" → cherche ailleurs.

### Tentative 4 — Revert predict_variances avec theta_bounds=[1,5]
REJETÉE par l'utilisatrice : "non ce ne sont pas les bounds le problème."

### Tentative 5 — predict_variances + fix update_degree >= (21/05, output_2105_1013)

**Contexte :** deux modifications simultanées :
1. `_exec_sigma` revenu à `predict_variances` (dans GEPCKFunction et GEKPLSFunction)
2. Condition upgrade degree changée de `>` à `>=` → upgrade à DOE=7 au lieu de 8

```python
# GEPCKFunction et GEKPLSFunction — même code
def _exec_sigma(self, u):
    return float(np.sqrt(self.sm.predict_variances(np.array(u).reshape(1, -1)).item()))
```

**Résultat (output_2105_1013) :** EFF explose immédiatement après upgrade deg=1→2 à DOE=7.
- DOE 5→6 : sigmaG~0.004, u_opt sur surface limite ✓
- DOE 6→7 : sigmaG~0.004, u_opt sur surface limite ✓
- DOE 7→8 : upgrade deg=2 → sigmaG saute à ~0.095 → u_opt=[4.4, 8.9] (coin) ✗
- Suite : sigma monte à 0.10-0.14, EFF~0.09-0.14, tous les points aux coins
- **18 pts EFF ajoutés**, beta=7.587 (-6.6%), convergence lente par remplissage uniforme

**Cause :** identique à tentative 1. Après upgrade deg=2, le résidu GEKPLS est grand (PCE deg=2 sur 7 pts insuffisant) → sigma2 GEKPLS grand → predict_variances retourne sigma~sqrt(sigma2) uniforme partout → EFF voit sigma uniforme → va aux coins.

**Conclusion : predict_variances ne peut pas fonctionner avec GEPCK+EFF quelle que soit la condition d'upgrade, car sigma2 GEKPLS est trop grand après l'upgrade et predict_variances est quasi-constant.**

---

## 3. État actuel du code (21/05/2026)

**Fichier :** `C:\_workingDir\_SF\test flexion\AC_pure_flexion.py`

- `modele = 'GEPCK'`, `do_EFF = True`, `n0 = 5`, `max_of_maxdegree = 2`
- `theta_bounds = [1.0, 5.0]` dans les deux branches de `build_metamodel_GEK`
- `update_degree` : condition `>=` (upgrade à DOE=7, au lieu de `>` qui upgradait à DOE=8)
- `GEPCKFunction._exec_sigma` (ligne ~854) : **Morris 1993 (tentative 3, actuellement en place)**
- `GEKPLSFunction._exec_sigma` (ligne ~950) : **Morris 1993 (tentative 3, actuellement en place)**

### Code _exec_sigma Morris 1993 (actuellement en place — GEPCKFunction et GEKPLSFunction identiques)
```python
def _exec_sigma(self, u):
    sm = self.sm
    n  = sm.nt; d = sm.X_norma.shape[1]
    W  = sm.coeff_pls; th = sm.optimal_theta
    s2 = float(sm.optimal_par['sigma2'])
    th_eff = (W**2) @ th
    x_n = (np.array(u).reshape(-1) - sm.X_offset) / sm.X_scale
    Xn  = sm.X_norma
    df  = x_n[None, :] - Xn
    kf  = np.exp(-np.dot(df**2, th_eff))
    kd  = (2.0 * kf[:, None] * df * th_eff[None, :]).reshape(-1)
    dff = Xn[:, None, :] - Xn[None, :, :]
    K_ff = np.exp(-np.einsum('ijk,k->ij', dff**2, th_eff))
    K_fd = (2.0 * K_ff[:, :, None] * dff * th_eff[None, None, :]).reshape(n, n*d)
    B_mat = dff * th_eff[None, None, :]
    term1 = 2.0 * np.diag(th_eff)
    term2 = 4.0 * np.einsum('ija,ijb->ijab', B_mat, B_mat)
    K_dd  = (K_ff[:, :, None, None] * (term1 - term2)).transpose(0,2,1,3).reshape(n*d, n*d)
    K_tot = np.block([[K_ff, K_fd], [K_fd.T, K_dd]])
    K_tot += 1e-10 * np.eye(K_tot.shape[0])
    k = np.concatenate([kf, kd])
    try:
        B = max(0.0, 1.0 - k @ np.linalg.solve(K_tot, k))
        return float(np.sqrt(s2 * B))
    except np.linalg.LinAlgError:
        return float(np.sqrt(sm.predict_variances(np.array(u).reshape(1, -1)).item()))
```

### Code _exec_sigma predict_variances (tentatives 1, 4, 5 — REJETÉ)
```python
def _exec_sigma(self, u):
    return float(np.sqrt(self.sm.predict_variances(np.array(u).reshape(1, -1)).item()))
```

---

## 4. Ce qu'on sait sur predict_variances SMT (GEKPLS)

Formule dans `krg_based.py` ligne 1286 :
```python
C = self.optimal_par["C"]          # Cholesky de R
rt = solve_triangular(C, r.T)      # C^{-1} r(x) — (n, n_eval)
u  = solve_triangular(G.T,
     Ft.T @ rt - f(x_cont).T)      # correction trend — (p, n_eval)
B  = 1.0 - (rt**2).sum(axis=0) + (u**2).sum(axis=0)
s2 = sigma2 * B
```
- `r(x)` = vecteur corrélations valeur-valeur seulement (pas de gradients)
- `rt²` = corrélation normalisée par Cholesky (= R^{-1} r en notation matricielle)
- `u²` = correction d'incertitude sur le trend (terme de régression)
- Formule identique pour GEKPLS et KPLS (predict_variances jamais redéfinie)

**Les gradients ne sont PAS utilisés dans predict_variances** → sigma n'est pas réduit dans la direction du gradient.

---

## 5. Ce qu'on sait sur PCKRG sigma (ce qui marche)

`result_r.getConditionalMarginalVariance(u)` — OT Kriging exact interpolator :
- sigma=0 EXACTEMENT aux points DOE (pas de nugget, interpolation exacte)
- sigma croît LISENT et ISOTROPIQUEMENT depuis les points DOE
- Pas d'anisotropie due aux gradients → EFF distingue bien zone connue/inconnue

---

## 6. Question ouverte — calcul d'incertitude correct pour GEKPLS

L'utilisatrice demande : lire la doc GEKPLS pour voir comment l'incertitude est correctement calculée pour un modèle gradient-enhanced.

**Pistes à explorer :**
- predict_variances de SMT ne prend pas en compte les gradients → sous-estime la réduction d'incertitude (ce qui AIDE EFF mais empêche sigma=0 au DOE quand theta grand)
- predict_variances avec theta_bounds=[1,5] : sigma=0 au DOE ✓, isotrope ✓, mais erreur -10% sur beta
- Peut-être un sigma hybride : utiliser la corrélation valeur-valeur (r^T R^{-1} r, sans blocs gradient) mais recalculé manuellement avec th_eff correct pour éviter le problème theta=[20,20]
- Ou : lire exactement ce que SMT propose pour GEKPLS predict_variances et si une version "gradient-aware" existe

---

## 7. Résultats de référence

| Run | Config | EFF pts | beta | Ecart HF (8.1235) |
|-----|--------|---------|------|-------------------|
| output_2005_0956 | PCKRG n0=5, EFF, theta[1,5] | 4 | 7.9662 | -0.16% |
| output_2005_1156 | GEPCK n0=5, EFF, theta[1,5], deg=1 | 6 | 7.258 | -10.6% |
| output_2005_1745 | GEPCK n0=5, EFF, Morris 1993, deg=2 | 16+ | FAIL | — |
| output_2105_1003 | GEPCK n0=5, EFF, Morris 1993, upgrade>=, deg=2 | 8 | 7.7287 | -4.9% |
| output_2105_1013 | GEPCK n0=5, EFF, predict_variances, upgrade>=, deg=2 | 18 | 7.587 | -6.6% |

**Référence HF (sp=[0,0]) : beta=8.1235, u*=[-4.776, -6.571]**
