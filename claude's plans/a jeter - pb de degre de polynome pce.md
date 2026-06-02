# Plan — Analyse PCE : pourquoi degree=1 échoue et fallback degree=2→1

## Context

La fonction `build_metamodel_PCE` dans `AC_pure_flexion.py` utilise LARS + CorrectedLeaveOneOut
pour construire un PCE de Hermite. Avec les paramètres actuels (`max_degree=1`, `n0=3`, `n_var=2`),
la PCE semble ne pas se générer ou crasher. L'objectif est de comprendre pourquoi et de savoir
si passer à `max_degree=2` permet un fallback automatique vers un modèle de degré 1.

---

## Q1 — Pourquoi degree=1 échoue

### Calcul de basis_size

```
HyperbolicAnisotropicEnumerateFunction(n_var=2, q=0.75)
getBasisSizeFromTotalDegree(max_degree=1)
```

La q-norme q=0.75 ne change PAS le nombre total de termes jusqu'au degré 1 (elle change
l'ordre d'énumération). Résultat :

| Degré | Termes | basis_size |
|-------|--------|------------|
| 0     | {(0,0)} | 1 |
| ≤1    | {(0,0),(1,0),(0,1)} | **3** |
| ≤2    | {(0,0),(1,0),(0,1),(2,0),(0,2)} | **5** |

Note : le terme (1,1) est exclu par q=0.75 car sa q-norme = (1+1)^(1/0.75) = 2^(4/3) ≈ 2.52 > 2.

### Pourquoi ça crash : singularité du LOO

Avec `n0=3` et `basis_size=3` on a **n_samples = basis_size = 3**.

Le chemin LARS construit des modèles à 0, 1, 2, 3 termes actifs. À chaque étape,
`CorrectedLeaveOneOut` évalue :

```
CV_LOO = 1/(n) × Σ [ eᵢ / (1 − hᵢ) ]²
```

- `hᵢ` = diagonale de la hat matrix H = X(X'X)⁻¹X'
- À l'étape 3 (3 termes actifs, 3 points) : X est carré → H = I → **hᵢ = 1**
- Dénominateur `(1 − hᵢ) = 0` → **division par zéro → NaN ou exception C++**

OT 1.26 ne filtre pas ce NaN proprement lors de la comparaison des modèles du chemin LARS
(bug déjà documenté session 21/04 pour `FunctionalChaosValidation`). Résultat : `algo.run()`
lève une exception ou le sélecteur de modèle retourne un résultat invalide.

**Cause racine : n_samples = basis_size = 3 est la configuration critique.**
La configuration carrée rend le LOO algébriquement singulier à la dernière étape LARS.

---

## Q2 — Avec max_degree=2, fallback automatique vers degré 1 ?

### Nouvelles dimensions avec max_degree=2

`basis_size = 5` (5 candidats) avec `n0 = 3` (3 données) → **n_samples < basis_size**.

### Comportement de LARS

LARS ne connaît pas les "degrés" : il sélectionne les termes dans l'ordre de leur corrélation
avec le résidu courant. Il peut sélectionner 0, 1, 2, ou 3 termes maximum (au-delà, la matrice
devient singulière comme au Q1, mais LARS s'arrête au plus tôt selon le LOO).

Chemin LARS typique avec 5 candidats et 3 points :
- Étape 1 : 1 terme → n_active=1 < n=3 → LOO OK
- Étape 2 : 2 termes → n_active=2 < n=3 → LOO OK
- Étape 3 : 3 termes → n_active=3 = n=3 → LOO singulier (NaN)
- → LARS s'arrête à l'étape de LOO minimum (étape 1 ou 2)

**Différence clé vs degree=1 :** avec 5 candidats, LARS a de l'espace pour trouver un
modèle parcimonieux (1-2 termes) avant d'atteindre la singularité. Avec 3 candidats,
il est contraint d'explorer toutes les étapes jusqu'à la singularité.

**Réponse à la question :** Oui, avec `max_degree=2` et `n0=3`, LARS peut sélectionner
automatiquement seulement 1-2 termes (souvent les linéaires si la surface est douce),
ce qui constitue de facto un modèle de degré ≤ 1. Ce n'est **pas garanti** : si un terme
de degré 2 est plus corrélé avec y, il sera sélectionné en premier. C'est une sélection
statistique, pas un contrôle de degré.

---

## Modification à faire (si approuvée)

**Fichier :** `C:\_workingDir\_SF\test flexion\AC_pure_flexion.py`

**Ligne 112 :**
```python
# AVANT
max_degree = 1

# APRÈS
max_degree = 2
```

C'est le seul changement nécessaire. `getBasisSizeFromTotalDegree(2)` donnera basis_size=5,
LARS pourra construire un PCE parcimonieux sans singularité LOO avec n0=3.

### Alternatives plus robustes (non implémentées ici)

1. **CleaningStrategy** : sélection par seuil de signifiance des coefficients.
   Problème : nécessite une estimation initiale, difficile avec n0=3.

2. **AdaptiveStrategy** : commence à degré 0, ajoute des strates.
   Avantage : s'adapte automatiquement. Inconvénient : API plus complexe.

3. **Guard n_samples > basis_size** : ajouter dans `build_metamodel_PCE` :
   ```python
   if n0 <= basis_size:
       print(f"PCE ignoree : n0={n0} <= basis_size={basis_size}")
       return None
   ```
   Et dans `FORM_init`, tester si `g_ot_PCE is None` → fallback sur GEK pur.

---

## Vérification

Après le changement `max_degree=2` :
- Observer la ligne de print : `PCE construite : basis_size=5, coefficients actifs LARS=N`
- N attendu : 1 ou 2 (pas 3, qui serait la limite singulière)
- Si algo.run() ne plante plus → Q1 confirmé

