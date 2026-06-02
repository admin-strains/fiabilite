# Plan — Enrichissement adaptatif du surrogate GEK par critère EFF

## Contexte

Le surrogate GEK (GEKPLS) est entraîné sur un DOE fixe de n0 points tirés aléatoirement.
Il n'y a aucune garantie que ces points sont proches de la surface de rupture g=0, là où FORM en a le plus besoin.
L'objectif est d'ajouter un enrichissement adaptatif après le DOE initial : à chaque itération, on choisit le point le plus informatif (celui qui maximise l'EFF) et on l'évalue avec STRAINS, jusqu'à ce que le surrogate soit suffisamment précis près de g=0.

---

## Formule EFF

L'EFF (Expected Feasibility Function, Bichon 2008) est définie pour T=0 (surface de rupture) par :

```
EFF(x) = 2μ·Φ(-μ/σ)
       - (ε+μ)·Φ(-(ε+μ)/σ)
       + (ε-μ)·Φ((ε-μ)/σ)
       + σ·[φ((ε+μ)/σ) - φ((ε-μ)/σ)]
```

où :
- μ = sm.predict_values(x)         → prédiction GEK en x
- σ = sqrt(sm.predict_variances(x)) → incertitude GEK en x
- ε = tolérance autour de g=0 (paramètre à fixer, voir ci-dessous)
- φ = scipy.stats.norm.pdf          → densité normale standard
- Φ = scipy.stats.norm.cdf          → CDF normale standard

## Bibliothèques utilisées

- `scipy.stats.norm.pdf` → φ
- `scipy.stats.norm.cdf` → Φ
- `sm.predict_variances(x)` → σ² (disponible dans GEKPLS/KRG de SMT)
- `numpy` → vectorisation sur la grille de candidats

---

## Algorithme EGRA (Efficient Global Reliability Analysis)

```
1. Entraîner GEK sur le DOE initial (n0 points)
2. Définir une grille/population de candidats dans l'espace U
3. Répéter jusqu'à convergence :
   a. Calculer EFF sur tous les candidats
   b. Si max(EFF) < EFF_threshold → STOP
   c. Sélectionner x* = argmax(EFF)
   d. Évaluer g_HF(x*) et dg(x*) avec STRAINS (run_HF)
   e. Ajouter x* au DOE, retrain GEK
4. Lancer FORM sur le GEK enrichi
```

---

## Fichier à modifier

`C:\_workingDir\_SF\test flexion\AC_pure_flexion.py`

### Ajouts

**1. Import en tête de fichier :**
```python
from scipy.stats import norm as sp_norm
```

**2. Fonction `compute_EFF` (après `build_metamodel_GEK`) :**
```python
def compute_EFF(sm, candidates, epsilon):
    """
    Calcule l'EFF (Expected Feasibility Function, Bichon 2008) sur une
    population de points candidats.
    sm         : modèle GEKPLS entraîné (SMT)
    candidates : np.array shape (n_cand, n_var) — points en espace U
    epsilon    : tolérance autour de g=0
    Retourne   : np.array shape (n_cand,) — valeur EFF en chaque candidat
    """
    mu  = sm.predict_values(candidates).ravel()          # (n_cand,)
    s2  = sm.predict_variances(candidates).ravel()       # (n_cand,)
    sig = np.sqrt(np.maximum(s2, 1e-12))                 # évite division par 0

    t1 = 2 * mu * sp_norm.cdf(-mu / sig)
    t2 = -(epsilon + mu) * sp_norm.cdf(-(epsilon + mu) / sig)
    t3 =  (epsilon - mu) * sp_norm.cdf( (epsilon - mu) / sig)
    t4 = sig * (sp_norm.pdf((epsilon + mu) / sig) - sp_norm.pdf((epsilon - mu) / sig))

    eff = t1 + t2 + t3 + t4
    return np.maximum(eff, 0.0)   # EFF >= 0 par construction
```

**3. Fonction `enrich_DOE_EFF` (après `compute_EFF`) :**
```python
def enrich_DOE_EFF(sm, xt, yt, all_grad, n_enrich_max, EFF_threshold, epsilon,
                   n_cand=2000):
    """
    Enrichissement adaptatif du DOE par critère EFF.
    À chaque itération : évalue STRAINS au point argmax(EFF), retrain GEK.
    Retourne le surrogate enrichi et le DOE mis à jour.
    """
    # Grille de candidats : LHS dans l'espace U [-5, 5]^n_var
    rng = np.random.default_rng(seed=0)
    candidates = rng.uniform(-5, 5, size=(n_cand, n_var))

    for k in range(n_enrich_max):
        eff_vals = compute_EFF(sm, candidates, epsilon)
        eff_max  = eff_vals.max()
        print(f"  [enrich {k+1}] max(EFF) = {eff_max:.4e}", flush=True)

        if eff_max < EFF_threshold:
            print(f"  Convergence EFF atteinte ({eff_max:.2e} < {EFF_threshold:.2e})", flush=True)
            break

        x_new = candidates[np.argmax(eff_vals)].reshape(1, -1)  # (1, n_var)

        # Évaluer STRAINS au point x_new
        g_new, grad_U_new, _ = run_HF(x_new.ravel())
        y_new = np.array([[g_new]])
        g_new_arr = np.array(grad_U_new).reshape(1, -1)

        # Mise à jour du DOE
        xt       = np.vstack([xt, x_new])
        yt       = np.vstack([yt, y_new])
        all_grad = np.vstack([all_grad, g_new_arr])

        # Retrain GEK
        sm = build_metamodel_GEK(xt, yt, all_grad)
        print(f"  DOE : {xt.shape[0]} points", flush=True)

    return sm, xt, yt, all_grad
```

**4. Paramètres à ajouter dans OPTIONS :**
```python
# EFF / enrichissement adaptatif
do_EFF        = True
n_enrich_max  = 10          # nombre max d'enrichissements
EFF_threshold = 1e-2        # critère d'arrêt sur max(EFF)
epsilon_EFF   = None        # None = calculé automatiquement depuis yt initial
```

**5. Intégration dans le flux principal** (après `build_metamodel_GEK`) :
```python
if do_GEK and not try_pce:
    xt, yt, all_grad = build_DOE()
    sm_GEK = build_metamodel_GEK(xt, yt, all_grad)

    if do_EFF:
        eps = epsilon_EFF if epsilon_EFF is not None else 2 * float(np.sqrt(
              sm_GEK.predict_variances(xt).mean()))
        print(f"EFF enrichissement — epsilon={eps:.4f}", flush=True)
        sm_GEK, xt, yt, all_grad = enrich_DOE_EFF(
            sm_GEK, xt, yt, all_grad,
            n_enrich_max=n_enrich_max,
            EFF_threshold=EFF_threshold,
            epsilon=eps)

    g_ot_GEK = ot.Function(GEKPLSFunction(sm_GEK))
    event = FORM_event(g_ot_GEK)
```

---

## Vérification

1. Lancer avec `do_EFF=False` → comportement identique à l'actuel (régression).
2. Lancer avec `do_EFF=True, n_enrich_max=3` → observer que 3 appels STRAINS
   s'ajoutent, que max(EFF) décroît à chaque itération, et que le DOE grandit.
3. Comparer u* et beta avec et sans enrichissement.
4. Vérifier que `compute_EFF` retourne des valeurs > 0 concentrées près de g=0
   en inspectant la visualisation `print_visu`.
