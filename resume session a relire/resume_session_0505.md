# Resume session 05 mai 2026 — DBSCAN pour identification des modes FORM

---

## Contexte

Discussion sur l'utilisation de DBSCAN pour identifier les modes de défaillance issus du multi-start FORM.
L'objectif : remplacer la logique de déduplication manuelle dans `FORM_all_modes` par un clustering géométrique plus robuste.

---

## Code actuel — `FORM_all_modes` (lignes 630-672 de AC_pure_flexion.py)

```python
def FORM_all_modes(starting_points, tol_all_modes):
    modes = []  # liste de best_result distincts

    for sp in starting_points:
        try:
            solver = ot.AbdoRackwitz()
            solver.setStartingPoint(sp.tolist())
            form_i = ot.FORM(solver, event)
            form_i.run()
            r_i = form_i.getResult()
            u_star = np.array(r_i.getStandardSpaceDesignPoint())
            sp_arr = np.array(sp)

            # Vérifier si ce u* est déjà connu (mode déjà trouvé)
            is_new = True
            for mode in modes:
                u_known = np.array(mode.getStandardSpaceDesignPoint())
                if np.linalg.norm(u_star - u_known) < tol_all_modes:
                    is_new = False
                    break

            if is_new:
                modes.append(r_i)
                print(f"  [sp={[round(v,3) for v in sp_arr]}, u*={[round(v,3) for v in u_star]}, beta={r_i.getHasoferReliabilityIndex():.4f}] NOUVEAU", flush=True)
            else:
                print(f"  [sp={[round(v,3) for v in sp_arr]}, u*={[round(v,3) for v in u_star]}, beta={r_i.getHasoferReliabilityIndex():.4f}]", flush=True)

        except Exception as e:
            sp_arr = np.array(sp)
            print(f"  [sp={[round(v,3) for v in sp_arr]}, ECHEC ({type(e).__name__})]", flush=True)

    modes.sort(key=lambda r: r.getHasoferReliabilityIndex())

    print(f"\n{len(modes)} mode(s) distinct(s) trouve(s) :", flush=True)
    for i, m in enumerate(modes):
        print(f"  mode {i+1} : beta={m.getHasoferReliabilityIndex():.4f}  Pf={m.getEventProbability():.3e}  u*={[round(v,3) for v in m.getStandardSpaceDesignPoint()]}", flush=True)

    return modes
```

---

## Code proposé — `FORM_all_modes` avec DBSCAN

Nécessite : `from sklearn.cluster import DBSCAN` en haut du fichier.

```python
from sklearn.cluster import DBSCAN

def FORM_all_modes(starting_points, tol_all_modes):
    """
    Multi-start FORM + DBSCAN pour identifier les modes de défaillance.
    - Collecte tous les u* (sans filtrage dans la boucle).
    - DBSCAN regroupe les u* proches en clusters = modes.
    - u* isolés (label -1) = descentes mal convergées, ignorées.
    - Pour chaque cluster : on garde le FORMResult avec le beta minimal.
    """
    all_u_star  = []   # u* de chaque run réussi
    all_results = []   # FORMResult correspondant

    for sp in starting_points:
        try:
            solver = ot.AbdoRackwitz()
            solver.setStartingPoint(sp.tolist())
            form_i = ot.FORM(solver, event)
            form_i.run()
            r_i    = form_i.getResult()
            u_star = np.array(r_i.getStandardSpaceDesignPoint())
            all_u_star.append(u_star)
            all_results.append(r_i)
            print(f"  [sp={[round(v,3) for v in np.array(sp)]}, "
                  f"u*={[round(v,3) for v in u_star]}, "
                  f"beta={r_i.getHasoferReliabilityIndex():.4f}]", flush=True)
        except Exception as e:
            print(f"  [sp={[round(v,3) for v in np.array(sp)]}, "
                  f"ECHEC ({type(e).__name__})]", flush=True)

    if not all_u_star:
        return []

    # --- DBSCAN sur tous les u* ---
    U_all  = np.array(all_u_star)           # shape (n_runs_ok, n_var)
    db     = DBSCAN(eps=tol_all_modes, min_samples=2).fit(U_all)
    labels = db.labels_

    n_noise = int(np.sum(labels == -1))
    if n_noise > 0:
        print(f"  {n_noise} descente(s) mal convergee(s) ignoree(s) (bruit DBSCAN)", flush=True)

    # --- Un mode par cluster : FORMResult avec beta minimal ---
    modes = []
    for lbl in sorted(set(labels) - {-1}):
        idx_cluster = [i for i, l in enumerate(labels) if l == lbl]
        best_i = min(idx_cluster,
                     key=lambda i: all_results[i].getHasoferReliabilityIndex())
        modes.append(all_results[best_i])

    modes.sort(key=lambda r: r.getHasoferReliabilityIndex())

    print(f"\n{len(modes)} mode(s) distinct(s) "
          f"(DBSCAN eps={tol_all_modes}, min_samples=2) :", flush=True)
    for i, m in enumerate(modes):
        u = [round(v, 3) for v in m.getStandardSpaceDesignPoint()]
        print(f"  mode {i+1} : beta={m.getHasoferReliabilityIndex():.4f}  "
              f"Pf={m.getEventProbability():.3e}  u*={u}", flush=True)

    return modes
```

---

## Comparaison des deux approches

### Logique de groupement

**Actuel :** dans la boucle FORM, chaque nouveau u* est comparé un par un aux modes déjà connus. Dès qu'un u* est à distance < `tol_all_modes` d'un mode existant, il est ignoré. Le premier u* qui arrive dans une région définit le mode.

**Proposé :** la boucle FORM ne fait que collecter. Tous les u* sont stockés. DBSCAN tourne une seule fois sur l'ensemble après la boucle, et groupe les u* proches en clusters. Le mode retenu pour chaque cluster est le FORMResult avec le **beta minimal** (le plus critique), pas le premier arrivé.

### Détection des descentes mal convergées

**Actuel :** pas de détection. Un u* aberrant (FORM a convergé vers un point incohérent) est accepté comme mode s'il est suffisamment loin des autres.

**Proposé :** DBSCAN avec `min_samples=2` étiquette `-1` (bruit) tout u* qui n'a pas au moins un voisin à distance < `eps`. Un u* totalement isolé = descente suspecte, ignorée. Note : si `min_samples=1`, cette détection est désactivée (tout point est son propre cluster) — revenir à 1 si les modes légitimes se retrouvent en bruit.

### Paramètres

Les deux approches utilisent `tol_all_modes` avec la même sémantique (distance en espace U pour distinguer deux modes). Dans la version DBSCAN, `eps = tol_all_modes` — pas de nouveau paramètre à calibrer.

### Tableau récapitulatif

| Aspect | Actuel | Proposé (DBSCAN) |
|---|---|---|
| Groupement | Boucle manuelle, premier arrivé | DBSCAN batch sur tous les u* |
| Mode retenu | Premier u* de la région | u* avec beta minimal du cluster |
| Descentes aberrantes | Acceptées si isolées | Rejetées (label -1, min_samples=2) |
| Paramètre | `tol_all_modes` directement | `eps = tol_all_modes` (même valeur) |
| Signature de la fonction | Identique | Identique — drop-in replacement |

---

## Pour revenir en arrière

Remplacer le bloc `FORM_all_modes` par le code actuel ci-dessus. Supprimer l'import `from sklearn.cluster import DBSCAN` si non utilisé ailleurs. Aucune autre modification du code principal.

---

---

# EFF — Enrichissement adaptatif du surrogate GEK

## Contexte

Le surrogate GEK est entraîné sur un DOE fixe (n0 points aléatoires). Ces points ne sont pas nécessairement proches de g=0 — là où FORM a besoin de précision. L'EFF (Expected Feasibility Function, Bichon 2008) permet de sélectionner itérativement les points les plus informatifs pour enrichir le DOE, en ciblant la surface de rupture.

---

## Formule EFF

```
EFF(x) = 2μ·Φ(-μ/σ)
       - (ε+μ)·Φ(-(ε+μ)/σ)
       + (ε-μ)·Φ((ε-μ)/σ)
       + σ·[φ((ε+μ)/σ) - φ((ε-μ)/σ)]
```

- μ = `sm.predict_values(x)`          → prédiction GEK
- σ = `sqrt(sm.predict_variances(x))` → incertitude GEK
- ε = tolérance autour de g=0
- φ = `scipy.stats.norm.pdf`
- Φ = `scipy.stats.norm.cdf`

EFF est grand quand μ est proche de 0 (près de g=0) ET/OU σ est grand (zone peu connue). Il vaut 0 loin de la surface de rupture.

**Choix de ε :** valeur de départ `ε = 2 * mean(σ_DOE)`. Ajuster si le critère ne converge pas.

**Critère d'arrêt :** `max(EFF) < EFF_threshold` (typiquement 1e-2 à 1e-3).

---

## Algorithme EGRA

```
1. Entraîner GEK sur le DOE initial (n0 points)
2. Définir un pool de candidats dans l'espace U (tirage uniforme)
3. Boucle jusqu'à convergence :
   a. Calculer EFF sur tous les candidats
   b. Si max(EFF) < EFF_threshold → STOP
   c. x* = argmax(EFF)
   d. Retirer x* du pool de candidats  ← IMPORTANT (évite re-sélection)
   e. Évaluer g_HF(x*) avec STRAINS
   f. Ajouter x* au DOE, retrain GEK
4. Lancer FORM sur le GEK enrichi
```

---

## Tests sur cas simples

### Test 1 — Vérification de la formule EFF (surrogate fictif 1D)

**Setup :**
- `g(x) = x - 1.5` (frontière g=0 en x=1.5)
- σ(x) = Gaussienne centrée en x=1.5 (incertitude max sur la frontière)
- ε = 0.2

**Résultat :**
```
argmax(EFF) = x = 1.286   (frontière g=0 en x=1.5)
max(EFF)    = 0.1308
mu  en argmax EFF = -0.214
sigma en argmax EFF = 0.343

EFF à x=1.5  : 0.086   ← proche de g=0, incertitude élevée
EFF à x=0.0  : 0.000   ← loin de g=0
EFF à x=3.0  : 0.000   ← loin de g=0
```

**Conclusion :** EFF est correctement nul loin de g=0 et maximal près de la frontière. L'argmax n'est pas exactement en x=1.5 car l'EFF pondère aussi l'incertitude σ — il sélectionne le meilleur compromis entre proximité à g=0 et incertitude élevée.

---

### Test 2 — Boucle EGRA complète avec vrai Krigeage SMT (cas 2D)

**Setup :**
- `g(u) = u1 + 0.5*u2 - 1.5` (frontière = droite dans l'espace U)
- DOE initial : 5 points aléatoires dans [-3,3]²
- Surrogate : KRG SMT
- Pool de candidats : 500 points uniformes dans [-3,3]²
- ε = 0.3, seuil = 1e-2

**Résultat de la boucle (6 itérations) :**
```
iter 1 : max(EFF)=0.29003  x*=[1.641, -0.262]  g_HF= 0.0100
iter 2 : max(EFF)=0.28748  x*=[1.001,  1.024]  g_HF= 0.0125
iter 3 : max(EFF)=0.28181  x*=[0.248,  2.540]  g_HF= 0.0182
iter 4 : max(EFF)=0.27883  x*=[1.344,  0.355]  g_HF= 0.0212
iter 5 : max(EFF)=0.27695  x*=[0.530,  1.986]  g_HF= 0.0230
iter 6 : max(EFF)=0.26883  x*=[1.358,  0.222]  g_HF=-0.0312
```

**Observations :**
- Tous les points ajoutés ont g proche de 0 : l'EFF cible exclusivement la surface de rupture.
- Les points sont répartis le long de la frontière (pas tous au même endroit).
- max(EFF) décroît à chaque itération : le surrogate devient progressivement plus précis près de g=0.
- Le seuil 1e-2 n'est pas atteint en 6 itérations avec ce pool de candidats — normal, la frontière est une droite infinie, il faudrait plus d'itérations ou un pool plus dense.

**Correction importante identifiée :** sans retirer x* du pool à chaque itération, l'algorithme re-sélectionne indéfiniment le même point (le Krigeage ne met pas à jour le pool de candidats). La ligne `candidates = np.delete(candidates, i_star, axis=0)` est obligatoire.

---

## Code à intégrer dans AC_pure_flexion.py

### Import à ajouter

```python
from scipy.stats import norm as sp_norm
```

### Fonction `compute_EFF`

```python
def compute_EFF(sm, candidates, epsilon):
    """
    EFF (Bichon 2008) sur une population de candidats.
    sm         : GEKPLS entraîné (SMT)
    candidates : np.array (n_cand, n_var) en espace U
    epsilon    : tolérance autour de g=0
    Retourne   : np.array (n_cand,)
    """
    mu  = sm.predict_values(candidates).ravel()
    s2  = sm.predict_variances(candidates).ravel()
    sig = np.sqrt(np.maximum(s2, 1e-12))

    t1 = 2 * mu * sp_norm.cdf(-mu / sig)
    t2 = -(epsilon + mu) * sp_norm.cdf(-(epsilon + mu) / sig)
    t3 =  (epsilon - mu) * sp_norm.cdf( (epsilon - mu) / sig)
    t4 = sig * (sp_norm.pdf((epsilon + mu) / sig) - sp_norm.pdf((epsilon - mu) / sig))

    return np.maximum(t1 + t2 + t3 + t4, 0.0)
```

### Fonction `enrich_DOE_EFF`

```python
def enrich_DOE_EFF(sm, xt, yt, all_grad, n_enrich_max, EFF_threshold, epsilon,
                   n_cand=2000):
    """
    Enrichissement adaptatif par EFF.
    Retourne (sm, xt, yt, all_grad) mis à jour.
    """
    rng = np.random.default_rng(seed=0)
    candidates = rng.uniform(-5, 5, size=(n_cand, n_var))

    for k in range(n_enrich_max):
        eff_vals = compute_EFF(sm, candidates, epsilon)
        i_star   = np.argmax(eff_vals)
        eff_max  = eff_vals[i_star]
        print(f"  [enrich {k+1}] max(EFF) = {eff_max:.4e}", flush=True)

        if eff_max < EFF_threshold:
            print(f"  Convergence EFF ({eff_max:.2e} < {EFF_threshold:.2e})", flush=True)
            break

        x_new      = candidates[i_star].reshape(1, -1)
        candidates = np.delete(candidates, i_star, axis=0)  # evite re-selection

        g_new, grad_U_new, _ = run_HF(x_new.ravel())
        xt       = np.vstack([xt, x_new])
        yt       = np.vstack([yt, [[g_new]]])
        all_grad = np.vstack([all_grad, np.array(grad_U_new).reshape(1, -1)])

        sm = build_metamodel_GEK(xt, yt, all_grad)
        print(f"  DOE : {xt.shape[0]} points", flush=True)

    return sm, xt, yt, all_grad
```

### Paramètres OPTIONS à ajouter

```python
do_EFF        = True
n_enrich_max  = 10
EFF_threshold = 1e-2
epsilon_EFF   = None   # None = 2 * mean(sigma_DOE)
```

### Intégration dans le flux principal

```python
if do_GEK and not try_pce:
    xt, yt, all_grad = build_DOE()
    sm_GEK = build_metamodel_GEK(xt, yt, all_grad)

    if do_EFF:
        eps = epsilon_EFF if epsilon_EFF is not None else \
              2 * float(np.sqrt(sm_GEK.predict_variances(xt).mean()))
        print(f"EFF enrichissement — epsilon={eps:.4f}", flush=True)
        sm_GEK, xt, yt, all_grad = enrich_DOE_EFF(
            sm_GEK, xt, yt, all_grad,
            n_enrich_max=n_enrich_max,
            EFF_threshold=EFF_threshold,
            epsilon=eps)

    g_ot_GEK = ot.Function(GEKPLSFunction(sm_GEK))
    event    = FORM_event(g_ot_GEK)
```

---

## Pour revenir en arrière (EFF)

Mettre `do_EFF = False` dans OPTIONS. Aucune autre modification nécessaire.

---

---

# Runs reels du 05/05/2026 — Nouvelle geometrie 2x3HA32 + exploration branche verticale

## Contexte

Reprise de session 04/05. Objectif : trouver une config ou la frontiere g_ana=0 a des zeros a u2 > -0.38 (branche verticale). Config de depart : b=0.4, h=0.45, 2x2HA32, F=0.11 MN → u2_min=-0.38.

---

## Modifications appliquees en debut de session

### dsCad.txt — passage a 2 lits de 3HA32

- b=0.4m, h=0.45m, phi=32mm
- Lit 1 : HA1(y=-0.10), HA2(y=0), HA3(y=+0.10), z=+0.161m (centroide a 0.08m du bord bas)
- Lit 2 : HA4(y=-0.10), HA5(y=0), HA6(y=+0.10), z=+0.145m

### AC_pure_flexion.py — FORM_all_modes avec DBSCAN

Remplacement de la logique manuelle de deduplication par DBSCAN (sklearn). `from sklearn.cluster import DBSCAN` ajoute a l'interieur du bloc `if __name__=='__main__'` (4 espaces d'indentation).

Fix IndentationError : output_0505_0746 = crash avant fix (DBSCAN import a indent=0).

### sensitivity_regions

Mis a jour avec les 6 barres : `["HA1","HA2","HA3","HA4","HA5","HA6"]` dans les deux occurrences.

---

## Partie A — Runs F=0.2, fyk=550, fck=28 (regime inverse)

### output_0505_0738 et output_0505_0747

**Config :** b=0.4, h=0.45, 2x3HA32, F=0.2 MN, fck=28, fyk=550, n0=20.

| Parametre | 0738 | 0747 |
|---|---|---|
| n_iter FORM | 1 | 1 |
| fc* (MPa) | 49.85 | 49.73 |
| fy* (MPa) | 643.33 | 643.11 |
| u* | [2.782, 1.451] | [2.763, 1.443] |
| Imp. fc/fy | 78.6%/21.4% | 78.6%/21.4% |
| beta | 3.137 | 3.117 |
| Pf | 9.99e-01 | 9.99e-01 |
| Regime | **Inverse** | **Inverse** |

F=0.2 MN > F_crit≈0.145 MN (capacite nominale 2x3HA32, fck=28). Regime inverse.

---

## Partie B — FORM fail avec F=0.094, fyk=550

### output_0505_0757 et output_0505_0759

**Config :** 2x3HA32, F=0.094 MN, fck=28, fyk=550, n0=20.

Tous les 21 points de depart → FORM exception : g_meta(u*)=0.404/0.409 > tol=0.2. Aucun FORM ne converge.

**Cause :** F_crit≈0.145 MN >> F=0.094 → g_GEK > 0 partout.

---

## Partie C — FORM converge avec fyk=300, F=0.094 (output_0505_0804)

**Config :** 2x3HA32, F=0.094 MN, fck=28, **fyk=300** MPa, n0=20.

| Parametre | Valeur |
|---|---|
| fc* (MPa) | 36.55 |
| fy* (MPa) | 290.35 |
| u* (multistart) | [0.187, -1.965] |
| Imp. fc/fy | 0.9%/99.1% |
| beta (multistart) | 1.974 |
| Pf | 2.42e-02 |
| mode 1 DBSCAN beta | 2.100, u*=[-0.241, -2.086] |
| err_abs_moy | 0.030 |

u2=-1.97 pour premier zero frontiere : toujours pas de branche a u2>-0.38. fyk=300 non physique.

---

## Partie D — Nouvelle geometrie b=0.5, h=0.8, 2x3HA16 (outputs 0842 et 0905)

**Config :** b=0.5m, h=0.8m, phi=16mm, 2 lits de 3HA16, z=0.328m et 0.312m. fck=48, fyk=550, n0=20.

| Parametre | output_0505_0842 | output_0505_0905 |
|---|---|---|
| fc* (MPa) | 55.88 | 55.88 |
| fy* (MPa) | 520.65 | 520.65 |
| u* | [0.004, -2.619] | [0.004, -2.619] |
| Imp. fc/fy | 0%/100% | 0%/100% |
| beta | 2.619 | 2.619 |
| Pf | 4.42e-03 | 4.41e-03 |
| dg/du_fc | 9.8e-05 | 9.8e-05 |
| dg/du_fy | 0.055852 | 0.055852 |
| Erreur FOSM | 0.42% | 0.40% |

Imp. fc=0% : section trop grande pour solliciter le beton. Pas de branche verticale. output_0905 = relance apres kill DLL.

---

## Partie E — Retour 2x3HA32, F=0.2, fck=28 (output_0505_0909)

**Config :** b=0.4, h=0.45, 2x3HA32, z=0.161/0.145, F=0.2 MN, fck=28, fyk=550, n0=20, print_pts=False.

| Parametre | Valeur |
|---|---|
| fc* (MPa) | 49.80 |
| fy* (MPa) | 643.24 |
| u* | [2.774, 1.448] |
| Imp. fc/fy | 78.6%/21.4% |
| beta | 3.129 |
| Pf | 9.99e-01 |
| dg/du_fc (HF@u*) | 0.032398 |
| dg/du_fy (HF@u*) | 0.031813 |
| u* FOSM (HF) | [3.307, 2.709] |
| Erreur FOSM | 0.438 |

print_error_ana_hf non capture (buffer Python non flush). 29 appels STRAINS confirmes. Regime inverse.

---

## Partie F — Retour config reference 2x2HA32 (output_0505_0929)

**Config :** b=0.4, h=0.45, 2x2HA32, z=0.165/0.125, F=0.11 MN, fck=28, fyk=550, n0=20, print_pts=False.

| Parametre | Valeur |
|---|---|
| fc* (MPa) | 31.31 |
| fy* (MPa) | 565.82 |
| u* (multistart) | [-1.108, -1.120] |
| Imp. fc/fy | 49.5%/50.5% |
| beta (multistart) | 1.576 |
| Pf | 5.76e-02 |
| mode 1 DBSCAN beta | 1.817, u*=[-1.286, -1.283] |

**print_error_ana_hf — 11 pts, err_abs_moy=0.0327 :** identique au run de reference output_0405_1446. Confirme que la geometrie est correctement restauree.

---

## Etat en fin de session

- dsCad : b=0.4, h=0.45, 2x2HA32, z=0.165/0.125, F=0.11 MN
- AC_pure_flexion.py : fck=28, fyk=550, do_GEK=True, n0=20, print_pts=False, tol_all_modes=0.01, sensitivity_regions HA1-HA4