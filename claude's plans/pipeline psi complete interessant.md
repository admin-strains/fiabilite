# Plan : predict_deriv analytique pour GEPCK

## Contexte

La formule de prédiction GEPCK est :
```
ŷ(x*) = Ψ(x*)ᵀ β  +  r̃₀(x*)ᵀ α_pred
```
avec `α_pred = R̃⁻¹(ẏ - F̃β)`.

Pour FORM, il faut `∂ŷ/∂u_i` (gradient dans l'espace auxiliaire U) :
```
∂ŷ/∂u_i = [∂Ψ/∂u_i(u*)]ᵀ β  +  [∂r̃₀/∂u_i(u*)]ᵀ α_pred
```

- `∂r̃₀/∂u_i` : déjà implémenté via `uq_eval_deriv_global_Kernel` (branche5.py:1297).
- `∂Ψ/∂u_i` : calculé par `make_trend_handle_deriv` (branche3.py:1134), mais cette fonction est **nested** dans `uq_GEPCK_calculate_coefficients` — inaccessible depuis l'extérieur.

**Solution** : stocker les closures dérivées dans `fitted_kriging['F_deriv_handles']` au moment du fit, puis les appeler depuis les fonctions de prédiction.

---

## Pipeline Ψ — trace complète (bas → haut)

| Niveau | Fonction | Fichier | Ligne | Rôle |
|---|---|---|---|---|
| 0 | `uq_eval_hermite` / `uq_eval_legendre` | branche5.py | 286 / 210 | Polynômes univariés `(N, P+1)` |
| 0' | `uq_eval_hermite_deriv` / `uq_eval_legendre_deriv` | branche5.py | 311 / 240 | Dérivées univariées `(N, P+1)` |
| 1 | `uq_PCK_eval_unipoly(U, Indices, PolyTypes)` | branche5.py | 340 | → `(N, M, P+1)` |
| 2 | `uq_PCE_create_Psi(Indices, uv)` | branche5.py | 388 | Produit tensoriel → `(N, P)` |
| 3 | `make_trend_handle(...)` | branche3.py | 1126 | Closure : U → Ψ(U), `(N, P_sel)` |
| 3' | `make_trend_handle_deriv(..., der)` | branche3.py | 1134 | Closure : U → ∂Ψ/∂u_der(U). Remplace `uv[:, der, :]` par dérivées univariées + annule colonnes indépendantes de u_der. |
| 4 | `make_trend_global_handle(...)` | branche3.py | 1150 | Closure augmentée : U → [Ψ; ∂Ψ/∂u₀; ...; ∂Ψ/∂u_{M-1}], `(N*(M+1), P_sel)` |
| 5 | `fit_kriging_gepck(...)` → stocke `F_global_handle` | branche3.py | ~600 | |
| 6 | `uq_GEPCK_calculate_coefficients(...)` → `fm['Kriging'][0]` | branche3.py | 1009 | |
| 7 | `uq_GEPCK_eval_one_output(gepck_oo, U_test, U_train, Y_aug, F_tilde_train, CorrOptions)` | branche4.py | 156 | `f0 = F_global_handle(U_test)[:N_test, :]` |
| 8 | `uq_GEPCK_eval(fm, X_test)` | branche4.py | 239 | X → U (isoprobabiliste), loop outputs |
| 9 | `predict_gepck(fm, X_test)` | branche1.py | 310 | Wrapper mince |

**Champs clés de `fm`** :
- `fm['Kriging'][0]` → `theta`, `beta`, `sigmaSQ`, `auxMatrices` (cholR/Rinv), `R_tilde`, `F_global_handle`, `F_tilde`
- `fm['ExpDesign']['U']` → U_train `(N, Mred)`
- `fm['ExpDesign']['Y_aug']` → Y_aug `(N*(M+1),)`
- `fm['CorrOptions']` → `{'Handle': uq_eval_global_Kernel, 'Family': ..., 'Nugget': ...}`

---

## Modifications

### Modification 1 — branche3.py : stocker F_deriv_handles

**Localisation** : entre ligne 1212 (`fitted_kriging = best_fitted` / fin du mode optimal) et ligne 1214 (commentaire `# Résultat`).

```python
    # Indices finaux utilisés dans le modèle sélectionné
    final_idx = idx_ranked if mode == 'sequential' else idx_ranked[:best_ii]

    # Closures ∂Ψ/∂u_k pour k=0..Mred-1 — utilisées par predict_deriv_gepck
    fitted_kriging['F_deriv_handles'] = [
        make_trend_handle_deriv(final_idx, Indices_oo, PolyTypes_all[:Mred], k)
        for k in range(Mred)
    ]
```

**Pourquoi ici** : `make_trend_handle_deriv` est nested dans `uq_GEPCK_calculate_coefficients` — c'est la seule fenêtre où elle est visible. On stocke les closures pour les exposer à l'extérieur.

---

### Modification 2 — branche4.py : uq_GEPCK_eval_one_output_deriv

Ajouter après `uq_GEPCK_eval_one_output` (après ligne 232), avant `uq_GEPCK_eval` :

```python
def uq_GEPCK_eval_one_output_deriv(gepck_oo, U_test, U_train, Y_aug,
                                    F_tilde_train, CorrOptions, der_var):
    """
    ∂ŷ/∂u_{der_var} analytique.
    Returns shape (N_test,).
    """
    theta = gepck_oo['theta']
    beta  = gepck_oo['beta']
    am    = gepck_oo['auxMatrices']
    N_aug = gepck_oo['R_tilde'].shape[0]

    # alpha_pred = R̃⁻¹(ẏ - F̃β)
    cholR = am['cholR']
    Rinv  = (np.linalg.solve(cholR, np.linalg.solve(cholR.T, np.eye(N_aug)))
             if cholR is not None else am['Rinv'])
    alpha_pred = Rinv @ (Y_aug - F_tilde_train @ beta)   # (N_aug,)

    # Terme 1 : [∂Ψ/∂u_i]ᵀ β
    dPsi  = gepck_oo['F_deriv_handles'][der_var](U_test)  # (N_test, P)
    term1 = dPsi @ beta                                   # (N_test,)

    # Terme 2 : [∂r̃₀/∂u_i]ᵀ α_pred
    CrossCorOpts = {**CorrOptions, 'Nugget': 0.0}
    dr0   = uq_eval_deriv_global_Kernel(
        U_test, U_train, theta, CrossCorOpts, der_var)    # (N_test, N_aug)
    term2 = dr0 @ alpha_pred                              # (N_test,)

    return term1 + term2
```

---

### Modification 3 — branche4.py : uq_GEPCK_eval_deriv

Ajouter après `uq_GEPCK_eval` (après ligne 315) :

```python
def uq_GEPCK_eval_deriv(fitted_model, X_test, der_var):
    """
    ∂ŷ/∂u_{der_var} en chaque point de X_test (espace auxiliaire U).
    Returns shape (N_test, 1).
    """
    X_test   = np.atleast_2d(X_test).astype(float)
    Mred     = fitted_model['Mred']
    nonConst = fitted_model['nonConst']
    Xred_test = X_test[:, nonConst]

    red_marg = fitted_model['RedMarginals']
    aux_marg = fitted_model['AuxSpace']['Marginals']
    aux_cop  = fitted_model['AuxSpace']['Copula']
    red_cop  = {'Type': 'Independent', 'Parameters': np.eye(Mred)}
    U_test   = uq_GeneralIsopTransform(Xred_test, red_marg, red_cop, aux_marg, aux_cop)

    U_train     = fitted_model['ExpDesign']['U']
    Y_aug       = fitted_model['ExpDesign']['Y_aug']
    CorrOptions = fitted_model['CorrOptions']

    dYMu = np.zeros((U_test.shape[0], fitted_model['Nout']))
    for oo in range(fitted_model['Nout']):
        gepck_oo      = fitted_model['Kriging'][oo]
        F_tilde_train = gepck_oo['F_tilde']
        dYMu[:, oo] = uq_GEPCK_eval_one_output_deriv(
            gepck_oo, U_test, U_train, Y_aug, F_tilde_train, CorrOptions, der_var)
    return dYMu
```

---

### Modification 4 — branche1.py : predict_deriv_gepck et predict_gradient_gepck

Ajouter après `predict_gepck` (~ligne 318) :

```python
def predict_deriv_gepck(fitted_model, X_test, der_var):
    """
    ∂ŷ/∂u_{der_var} analytique (espace auxiliaire U, 0-indexed).
    Returns shape (N_test, 1).
    """
    return uq_GEPCK_eval_deriv(fitted_model, X_test, der_var)


def predict_gradient_gepck(fitted_model, X_test):
    """
    Gradient complet ∂ŷ/∂u dans l'espace auxiliaire.
    Returns shape (N_test, Mred).
    """
    Mred   = fitted_model['Mred']
    N_test = np.atleast_2d(X_test).shape[0]
    G = np.zeros((N_test, Mred))
    for i in range(Mred):
        G[:, i] = predict_deriv_gepck(fitted_model, X_test, i)[:, 0]
    return G
```

---

## Import à vérifier dans branche4.py

`uq_eval_deriv_global_Kernel` doit être importé depuis branche5.py en tête de branche4.py.

---

## Vérification — test par différences finies en U-space

Test à ajouter dans un script standalone (ou en bas de demo_gepck.py) :

```python
# Après fm = fit_gepck(X, Y_aug, ...)
gepck_oo      = fm['Kriging'][0]
U_train       = fm['ExpDesign']['U']
Y_aug         = fm['ExpDesign']['Y_aug']
F_tilde_train = gepck_oo['F_tilde']
CorrOptions   = fm['CorrOptions']

u0  = U_train[0:1, :].copy()
eps = 1e-5

for der_var in range(fm['Mred']):
    u_plus  = u0.copy(); u_plus[0,  der_var] += eps
    u_minus = u0.copy(); u_minus[0, der_var] -= eps

    y_plus  = uq_GEPCK_eval_one_output(gepck_oo, u_plus,  U_train, Y_aug, F_tilde_train, CorrOptions)
    y_minus = uq_GEPCK_eval_one_output(gepck_oo, u_minus, U_train, Y_aug, F_tilde_train, CorrOptions)
    fd = (y_plus[0] - y_minus[0]) / (2 * eps)

    analytic = uq_GEPCK_eval_one_output_deriv(
        gepck_oo, u0, U_train, Y_aug, F_tilde_train, CorrOptions, der_var)
    
    print(f"der_var={der_var}  FD={fd:.6e}  analytic={analytic[0]:.6e}  err={abs(fd-analytic[0]):.2e}")
    # erreur attendue : ~1e-9 à 1e-11
```

---

## Fichiers à modifier

| Fichier | Ligne | Modification |
|---|---|---|
| `branche3.py` | ~1213 | Mod 1 : `final_idx` + `F_deriv_handles` |
| `branche4.py` | ~233 | Mod 2 : `uq_GEPCK_eval_one_output_deriv` |
| `branche4.py` | ~316 | Mod 3 : `uq_GEPCK_eval_deriv` |
| `branche1.py` | ~319 | Mod 4 : `predict_deriv_gepck` + `predict_gradient_gepck` |
