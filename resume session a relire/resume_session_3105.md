# Resume session 31/05/2026 — EFF, GEPCK, sigma, nugget

## Fichiers lus cette session
- `AC2_pure_flexion.py` (branche GEPCK, GEPCKFunction, run_EFF, EFFFunction)
- `branche1.py` (fit_gepck, predict_gepck, predict_gradient_gepck)
- Bichon 2008 (EGRA / EFF paper)
- Moustapha 2022 (Active learning benchmark, Structural Safety)

---

## 1. Ce que fait la branche `do_GEPCK` dans AC2_pure_flexion.py

**modele = 'GEPCK'** active le bloc `if do_GEPCK:` (ligne ~1762).

### Pipeline
1. `init_surrogate()` → `build_DOE()` : n0=5 points LHS + [0,0], appels STRAINS → `xt` (5x2), `yt` (5 valeurs g), `all_grad` (5x2 gradients dg/du)
2. `build_Y_aug(yt, all_grad)` : construit vecteur augmenté shape `(n0*(1+n_var),)` = `[g¹…g⁵, dg/du₁¹…⁵, dg/du₂¹…⁵]` — eq. 6 Zuhal et al.
3. `fit_gepck(xt, Y_aug, opts, marginals, copula)` depuis `branche1.py` : PC-Kriging gradient-enhanced (UQLab-style), entraîné simultanément sur g ET gradients, LARS, Hermite
4. `GEPCKFunction(fm)` : wrapper OT — `_exec` → `predict_gepck`, `_exec_sigma` → variance, `_gradient` → `predict_gradient_gepck`
5. Visu 2 panneaux (`notre_gepck_hf.png`) :
   - **Gauche** : contourf RdYlGn g_GEPCK, contour bleu g=0 GEPCK, contour rouge tiret g=0 HF (depuis `hf_2d_grid_fixed` hardcodé), courbe verte analytique, DOE noir, flèches gradient rouge
   - **Droite** : contourf Blues sigma_GEPCK, contour bleu g=0 GEPCK, DOE rouge
   - Titre : `N=5, LOO=..., n_poly=...`

### Différence old_GEPCK vs GEPCK
- `old_GEPCK` : PCE (OT) sur g, puis GEKPLS (SMT) sur résidu g-PCE. Deux modèles enchaînés.
- `GEPCK` (nouveau, branche1.py) : PCK gradient-enhanced natif, entraîné sur g + gradients simultanément.

---

## 2. Pourquoi run_EFF ajoute des points sur g_surrogate=0 (et pas g_HF=0)

### Par construction — formule EFF (Bichon 2008, eq. 17)

```python
epsilon = epsilon_factor * sigmaG   # = 2 * sigmaG
t1 = -muG / sigmaG
t2 = (epsilon + muG) / sigmaG
t3 = (epsilon - muG) / sigmaG
EFF = 2*muG*cdf(t1) - (epsilon+muG)*cdf(-t2) + (epsilon-muG)*cdf(t3)
    + sigmaG*(-2*pdf(t1) + pdf(t2) + pdf(t3))
```

EFF est **maximal** quand :
- `muG = g_surrogate(u) ≈ 0` → on est sur la frontière du **surrogate** (contour bleu)
- `sigmaG > 0` → le modèle est incertain là

**Le code ne voit jamais g_HF pendant l'optimisation de EFF.** HF n'intervient qu'après : une fois u_opt trouvé, `run_HF(u_opt)` est appelé pour évaluer la vraie valeur et l'ajouter au DOE.

---

## 3. Peut-on corriger la frontière avec EFF si le DOE initial est mauvais ?

### Mécanisme de correction (théorique)
Quand EFF ajoute u_opt sur le bleu avec g_HF(u_opt) ≠ 0, le surrogate apprend que "là où je croyais g=0, en réalité g=X≠0" → il déplace sa frontière.

**En théorie : oui, la boucle corrige progressivement la frontière bleue vers la rouge.**

### Pourquoi ça bloque en pratique

EFF s'arrête quand `max(EFF) < tol_EFF = 0.001`. Cette condition est atteinte quand **σG ≈ 0 sur toute la frontière** — pas quand la frontière est correcte.

Avec un surrogate quasi-interpolant (GEKPLS, GEPCK, nugget≈0) :
- σG = 0 aux points du DOE
- Après 3 points EFF ajoutés sur la frontière bleue → σG ≈ 0 sur toute la frontière
- EFF converge par effondrement de σG, pas par correction de la frontière

---

## 4. Notion de surrogate interpolant et nugget

### Surrogate interpolant
Passe **exactement** par tous les points du DOE : `ŷ(xᵢ) = yᵢ` pour tout i. GEKPLS et GEPCK SMT sont quasi-interpolants (nugget≈0).

### Nugget
Terme de variance ajouté sur la diagonale de la matrice de corrélation R. Représente un bruit de mesure.

| | Nugget = 0 | Nugget > 0 |
|---|---|---|
| Type de modèle | Interpolant (régression exacte) | Lissant (régression approchée) |
| σG aux points DOE | 0 | > 0 |
| Justifié quand | Simulateur déterministe | Données bruitées |
| Problème pour EFF | σG s'effondre vite sur la frontière | EFF continue d'explorer |

### Lien avec la variance Kriging (eq. 9 Bichon)
`σG²(u) = σ²_Z - [correction due aux données]`
- Au point du DOE uᵢ avec nugget=0 : correction = σ²_Z → σG²(uᵢ) = 0
- Au point du DOE uᵢ avec nugget>0 : correction < σ²_Z → σG²(uᵢ) = nugget > 0

---

## 5. EFF est-il adapté au Kriging standard ?

**OUI.** L'erreur de session est d'avoir généralisé. Pour le Kriging standard (OT, UQLab) :
- σG = 0 **uniquement aux points du DOE exactement**
- σG > 0 **partout ailleurs**, y compris sur la frontière g_surrogate=0

EFF fonctionne parfaitement avec le Kriging standard. Le problème est **spécifique à GEKPLS/SMT** dont `predict_variances` retourne ~0 partout (même loin du DOE). C'est une limitation de l'implémentation SMT, pas une propriété générale du Kriging interpolant.

---

## 6. Ce que disent les papiers

### Bichon 2008 (EGRA)
- EFF = adaptation de EIF (Expected Improvement Function) pour estimation de contour
- ε = 2σG (code : `epsilon_factor = 2`)
- Kriging standard interpolant, pas de nugget mentionné
- σG = 0 aux DOE, EFF = 0 aux DOE → OK, EFF cherche ailleurs sur la frontière
- Critère d'arrêt : `max(EFF) < 0.001`

### Moustapha 2022 (benchmark active learning)
- **Meilleure stratégie globale : PC-Kriging + Subset Simulation + U + β-bounds/combined**
- EFF est moins bon que le critère U (deviation number) dans la plupart des cas
- Section 3.1.1 : *"interpolation vs. regression — **The former are often preferred** in active learning schemes"* → **ne pas ajouter de nugget pour un simulateur déterministe**
- Pour surrogates sans variance built-in fiable : utiliser **bootstrap ou cross-validation** pour estimer σG
- EFF se dégrade avec la dimension (contrairement à U)

---

## 7. Recommandation pour le code

| Problème | Cause | Solution |
|---|---|---|
| EFF s'arrête trop tôt avec GEKPLS | `predict_variances` SMT ≈ 0 partout | Variance unreliable dans SMT |
| EFF s'arrête trop tôt avec GEPCK | Même problème via `predict_gepck(..., return_var=True)` | Vérifier implémentation dans branche1.py |
| Ajouter nugget ? | **NON** — simulateur déterministe, Moustapha dit interpolation préférable | Ne pas changer |
| Alternative | Bootstrap ou LOO pour estimer σG si variance SMT non fiable | Non trivial à implémenter |
| Meilleure piste long terme | KRG OT (variance OT fonctionnelle) + critère U + Subset Sim | Moustapha recommandation |

---

## 8. Etat du code au 31/05

```python
modele = 'GEPCK'   # branche active
n0 = 5
do_EFF = False     # EFF désactivé
max_degree = 0     # → liste vide → PCE degré 0 (constante seulement)
epsilon_factor = 2
tol_EFF = 1e-3
```

**Attention max_degree=0** : `list(range(1, 0+1))` = `[1]` → degree=1 actif (vérifier le comportement exact de `fit_gepck` avec cette valeur).

Fichier principal : `AC2_pure_flexion.py`
Branche1 : `branche1.py` (fit_gepck, predict_gepck, predict_gradient_gepck)

---

## 9. Graphe observé cette session

Run avec modele='GEPCK', n0=5, do_EFF=True (run antérieur) :
- Frontière bleue (g_GEPCK=0) très décalée de la frontière rouge (g_HF=0)
- 3 points EFF (triangles rouges) placés sur la frontière **bleue** (pas la rouge)
- u* FORM = [-7.25, -2.56], beta=7.691 — très différent de la référence HF
- Conclusion : DOE initial trop petit (n0=5), mauvaise frontière initiale, EFF rafine la mauvaise frontière
