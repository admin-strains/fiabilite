# Analyse : Rebound surface surrogate Kriging — pourquoi ça remonte ?

## Ce que fait réellement le code (trouvé dans AC_pure_flexion.py)

- **do_KRG = True / do_GEK = False** (l.86-89) → le surrogate actif est **ot.KrigingAlgorithm**, pas GEKPLS
- Kernel actif : `ot.SquaredExponential([1.0] * n_var)` — Matérn commenté (l.702)
- Tendance : `ot.ConstantBasisFactory(n_var).build()` → constante globale
- Hyperparamètres θ : optimisés par MLE via `algo_KRG.run()`
- DOE : 18 points LHS+SA dans [-10,10]²

---

## La formule de prédiction Kriging (source : doc OT + SMT)

```
ĝ(x) = f(x)ᵀβ + r(x)ᵀ R⁻¹ (Y - f(X)ᵀβ)
```
- `f(x)ᵀβ` = tendance (ici constante β₀)
- `r(x)` = vecteur de corrélation entre x et les points du DOE
- `R` = matrice de corrélation du DOE

### Comportement loin du DOE
Quand x s'éloigne du DOE → `r(x) → 0` (décroissance exponentielle du SE kernel) :

```
ĝ(x) → f(x)ᵀβ = β₀  (constante globale)
```

---

## Explication du rebound — mécanisme précis

### 1. Le kernel SE est un interpolateur exact + symétrique

Le kernel SquaredExponential :
```
R(xᵢ, xⱼ) = ∏ₗ exp(-θₗ · (xₗ⁽ⁱ⁾ - xₗ⁽ʲ⁾)²)
```
Propriétés :
- **Interpolation exacte** : le surrogate passe EXACTEMENT par les 18 points du DOE
- **Symétrie isotrope** : le noyau ne "sait" pas que g doit être monotone décroissant
- **Infiniment différentiable** : pour interpoler les points négatifs du DOE entourés
  de points positifs, le surrogate crée une "bosse inversée" localement

### 2. Géométrie du problème : une poche de valeurs négatives

Le DOE a probablement quelques points avec g<0 dans la zone centre-gauche,
entourés de points g>0 ailleurs. Le surrogate :
1. **Doit passer exactement par les points g<0** → crée un creux local
2. **Doit aussi passer par les points g>0 voisins** → remonte de chaque côté
3. **Hors du DOE** → r(x) → 0 → remonte vers β₀ > 0

→ **Le rebound n'est pas un artefact de zone vide : c'est la conséquence directe
de l'interpolation exacte par un noyau symétrique.**

### 3. Rôle du MLE dans le choix de θ

MLE optimise θ pour maximiser la vraisemblance sur les 18 points.
Avec des valeurs g très négatives dans le centre-gauche et g>0 ailleurs,
MLE converge vers un **θ relativement petit** (longueur de corrélation courte)
pour capturer ce contraste local.

Conséquence : la corrélation décroît rapidement → le surrogate "oublie" vite
la valeur négative et remonte dès qu'on s'éloigne de ce cluster.

### 4. La tendance constante aggrave le retour vers le haut

β₀ est estimé comme la moyenne pondérée des 18 évaluations HF.
Si la majorité des points DOE sont en zone g>0 (normal : LHS couvre tout [-10,10]²),
alors β₀ > 0.

→ **Plus on s'éloigne du cluster g<0, plus le surrogate revient vers β₀ > 0.**

---

## Ce qui se passe sur ton plot

```
Zone droite (u1>0) : g>0 → vert/jaune       ← points DOE positifs
Zone centre (u1≈-2) : g<0 → rouge           ← quelques points DOE négatifs
Zone gauche (u1<-5) : g remonte → vert       ← plus de DOE, r(x)→0, retour vers β₀>0
```

Le surrogate crée une **île** de g<0 au lieu d'une **région semi-infinie** comme le vrai modèle.
La courbe g=0 (pointillée bleue dans le plot) forme une boucle fermée au lieu d'une courbe ouverte.

---

## Pourquoi le HF n'a pas ce problème

La surface HF (3D) est le modèle analytique — pas de corrélation ni de tendance.
Elle représente la vraie physique : g monotone décroissant vers le bas-gauche.
Le surrogate n'a aucune façon de savoir ça sans points d'entraînement dans cette zone.

---

## Fichiers clés
- `c:\_workingDir\_SF\test flexion\AC_pure_flexion.py`
  - L.80 : `n0 = 18`
  - L.86-89 : `do_KRG = True`, `do_GEK = False`
  - L.699-708 : `build_metamodel_KRG` — SquaredExponential + ConstantBasis
  - L.702 : Matérn commenté (alternative possible)
