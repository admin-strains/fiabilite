# Plan : Correction des expressions du laplacien (diagonale Hessien) dans l'espace U

## Contexte

On veut calculer les dérivées secondes diagonales `∂²f/∂u_i²` (utilisées pour SORM) pour les méthodes `lap_f_plast` et `lap_f_nonplast`.

La règle de chaîne complète pour une dérivée seconde est :
```
∂²f/∂ui² = (∂²f/∂xi²)·(∂xi/∂ui)² + (∂f/∂xi)·(∂²xi/∂ui²)
```

Le **second terme est nul pour la normale** (∂²x/∂u² = 0) mais **non nul pour la lognormale** (∂²x1/∂u1² = σ_ln² · x1).

---

## Diagnostic du code actuel : deux erreurs

### `lap_f_plast` — DEUX expressions incorrectes

```python
# CODE ACTUEL (incorrect)
der_u1 = -(4*self.B**2*x2**4/x1**5) * self.fc_std   # ≠ ∂²f/∂u1²
der_u2 = (2*self.B/x1) * self.fy_std                  # ≠ ∂²f/∂u2² (manque un facteur fy_std)
```

**Valeurs correctes (après dérivation analytique complète) :**

Pour `f_plast = A·x2 + B·x2²/x1 + C` :
- `∂f/∂x1 = -B·x2²/x1²`         | `∂²f/∂x1² = 2·B·x2²/x1³`
- `∂f/∂x2 = A + 2·B·x2/x1`      | `∂²f/∂x2² = 2·B/x1`

Avec **x1 lognormale** (∂x1/∂u1 = σ_ln·x1, ∂²x1/∂u1² = σ_ln²·x1) et **x2 normale** (∂x2/∂u2 = σ_fy, ∂²x2/∂u2² = 0) :

```
∂²f_plast/∂u1² = (2·B·x2²/x1³)·(σ_ln·x1)² + (-B·x2²/x1²)·(σ_ln²·x1)
               = 2·B·x2²·σ_ln²/x1 - B·x2²·σ_ln²/x1
               = B · x2² · σ_ln² / x1            ← FORME SIMPLIFIÉE

∂²f_plast/∂u2² = (2·B/x1)·σ_fy² + 0
               = 2·B·σ_fy²/x1
```

**Remarque importante :** le terme `der_u2` du code actuel est `(2·B/x1)·σ_fy`, il manque un facteur `σ_fy`. C'était déjà faux pour la loi normale.

### `lap_f_nonplast` — expression à corriger

Pour `f_nonplast(x1)` (indépendant de x2), avec x1 lognormale :

```
∂²f_nonplast/∂u1² = σ_ln² · x1 · [x1·(∂²f/∂x1²) + (∂f/∂x1)]
∂²f_nonplast/∂u2² = 0  (inchangé)
```

Où :
```
∂f/∂x1   = Bp·[Ap/S - 0.1·(4·Ap·x1 - 2·S + 2)/(Ap²·x1²·S)]          (S = sqrt(1+4·Ap·x1))
∂²f/∂x1² = -2·Bp·Ap²/S³ - 0.1·Bp/(Ap²·x1³·S²)·[4·Ap·x1·S·(S-1) - 2·N·Q/S]
            avec N = 4·Ap·x1 - 2·S + 2  et  Q = S² + Ap·x1 = 1 + 5·Ap·x1
```

---

## Utilité d'OpenTURNS pour le laplacien

**NON nécessaire.** Les transformations sont connues analytiquement :
- `T_inv.hessian(u_pt)` existe dans OT mais utilise des différences finies internes pour certaines distributions → moins précis, moins rapide
- Pour normale et lognormale, les formules analytiques sont exactes et préférables

Le seul paramètre dont on a besoin est **`σ_ln`** (écart-type dans l'espace log de la lognormale de fc), à récupérer depuis la distribution dans `__init__` :

```python
# Dans __init__, après construction de dist_X :
self.sigma_ln = dist_X.getMarginal(0).getParameter()[1]  # σ_ln de la lognormale
```

---

## Code de remplacement

### `lap_f_plast`

```python
def lap_f_plast(self, u):
    x_point = self.T_inv(ot.Point(u))
    x1, x2  = x_point[0], x_point[1]
    sl  = self.sigma_ln   # σ_ln de la lognormale fc
    sfy = self.fy_std     # σ_fy de la normale fy (= 30 MPa)
    der_u1 = self.B * x2**2 * sl**2 / x1           # B·x2²·σ_ln²/x1
    der_u2 = 2 * self.B * sfy**2 / x1              # 2·B·σ_fy²/x1
    return anp.array([der_u1, der_u2])
```

### `lap_f_nonplast`

```python
def lap_f_nonplast(self, u):
    x1 = self.T_inv(ot.Point(u))[0]
    S  = (1 + 4 * self.Ap * x1)**0.5
    N  = 4 * self.Ap * x1 - 2 * S + 2
    Q  = S**2 + self.Ap * x1                  # = 1 + 5*Ap*x1

    # ∂f/∂x1 (déjà dans grad_f_nonplast)
    df_dx1  = self.Bp * (self.Ap / S - 0.1 * N / (self.Ap**2 * x1**2 * S))

    # ∂²f/∂x1²
    bracket = 4 * self.Ap * x1 * S * (S - 1) - 2 * N * Q / S
    d2f_dx1 = -2 * self.Bp * self.Ap**2 / S**3 \
              - 0.1 * self.Bp / (self.Ap**2 * x1**3 * S**2) * bracket

    sl = self.sigma_ln
    lap_u1 = sl**2 * x1 * (x1 * d2f_dx1 + df_dx1)
    return anp.array([lap_u1, 0.0])
```

### Ajout dans `__init__`

```python
# Après construction de dist_X :
self.sigma_ln = dist_X.getMarginal(0).getParameter()[1]  # σ_ln lognormale fc
```

---

## Fichier cible

`C:\_workingDir\_SF\test flexion\visu_ana\2026_3004_calcul_de_pf.py`

---

## Vérification (à faire quand SORM sera implémenté)

Comparer `lap_f` analytique avec différences finies centrées :
```
∂²f/∂ui² ≈ (f(u+h·ei) - 2·f(u) + f(u-h·ei)) / h²   avec h = 1e-5
```
Erreur attendue < 1e-6 (erreur de troncature des différences finies d'ordre 2).
