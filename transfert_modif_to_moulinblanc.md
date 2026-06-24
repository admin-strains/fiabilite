# Transfert des modifications AC3 → moulin_blanc

Ce guide documente les modifications faites sur `AC3_pure_flexion.py` (branche `flexion`)
qui **divergent** de ce qui existe dans `AC_moulin_blanc_2fy.py` (branche `moulin_blanc`).
Le but : permettre au developpeur de moulin_blanc d'appliquer ces changements sur son code.

Seules les differences sont documentees ici. Les fix identiques (fix1 a fix6) ne sont pas
listes car ils sont deja en place dans moulin_blanc.

---

## Modif 1 — SOCP History : sous-dossiers au lieu de prefixes plats

**Commits AC3** : `345f585` (modif1) + `fe055ba` (modif1bis)

### Ce qui change par rapport a moulin_blanc

Dans moulin_blanc (`_save_socp_outputs`), les 5 fichiers SOCP sont copies **a plat**
dans `SOCP_history/` avec le prefix dans le nom de fichier :

```
SOCP_history/
  SOL_001_fy1235.0_fy2210.0_Yield_analysis0_0_kine.dsmetares
  SOL_001_fy1235.0_fy2210.0_Yield_analysis0_0_kine.dsmed
  SOL_001_fy1235.0_fy2210.0_Yield_analysis0_0_PL_cin_out.msh
  ...
```

Dans AC3, chaque appel SOCP a son propre **sous-dossier** :

```
SOCP_history/
  SOL_001_fc48.0_fy550.0/
    Yield_analysis0_0_kine.dsmetares
    Yield_analysis0_0_kine.dsmed
    Yield_analysis0_0_PL_cin_out.msh
    ...
  HF_006_u1-3.117_u2-7.345_fc38.2_fy480.1/
    ...
```

### Comment appliquer a moulin_blanc

Dans `_save_socp_outputs`, remplacer :

```python
# AVANT (moulin_blanc) :
save_dir = os.path.join(_socp_root, "SOCP_history")
os.makedirs(save_dir, exist_ok=True)
...
dst_name = f"{prefix_tag}{coords_str}_{f}"
dst = os.path.join(save_dir, dst_name)
```

par :

```python
# APRES :
sub_dir = os.path.join(_socp_root, "SOCP_history", f"{prefix_tag}{coords_str}")
os.makedirs(sub_dir, exist_ok=True)
...
dst = os.path.join(sub_dir, f)
```

### Parametres generiques (`p_vals`)

Dans moulin_blanc, la signature est `p1=None, p2=None` avec un format hardcode
`_fy1{p1:.1f}_fy2{p2:.1f}`. Dans AC3, on utilise `p_vals=None` (liste) avec un
format generique qui lit `params_names` :

```python
if p_vals is not None:
    coords_str += "_" + "_".join(f"{params_names[i]}{p_vals[i]:.1f}" for i in range(len(p_vals)))
```

Les appels deviennent :
- `run_one_SOL` : `p_vals=[float(SOL[i][p]) for p in params_names]`
- `run_HF` : `p_vals=[float(x_point[j]) for j in range(n_var)]`

Ca fonctionne pour 2fy (`fy1235.0_fy2210.0`), fc/fy (`fc48.0_fy550.0`), ou toute
autre combinaison future.

---

## Modif 1bis — Flag `save_history` pour conditionner la sauvegarde

**Commit AC3** : `fe055ba`

### Ce qui est ajoute

Variable globale dans les options utilisateur :

```python
save_history = True   # True = copie les fichiers SOCP dans SOCP_history/
```

Les appels `_save_socp_outputs` dans `run_one_SOL` et `run_HF` sont conditionnes :

```python
if save_history:
    _socp_call_counter[0] += 1
    _save_socp_outputs(...)
```

Moulin_blanc n'a pas ce flag — la sauvegarde est toujours active. Pour appliquer :
ajouter `save_history = True` dans les options et wrapper les 2 appels existants.

---

## Modif 3a — DOE cache sans signature, conditionne par `config_is_identical`

**Commit AC3** : `9b4b4f5`

### Ce qui change par rapport a moulin_blanc

Dans moulin_blanc (commit `9052a3c`), le cache DOE utilise une **signature automatique** :

```python
def _doe_cache_sig():
    return {"n0": n0, "params": list(params_names), "n_var": n_var,
            "g1": len(group1_names), "g2": len(group2_names), "modelname": modelname}
```

Le cache est invalide si la signature change (n0, params, groupes, modelname).

Dans AC3, on remplace ca par un **flag explicite utilisateur** :

```python
config_is_identical = True   # True = reutilise doe_cache.json si present (0 SOCP DOE)
```

- `True` + fichier existe → charge le cache sans verification
- `True` + fichier absent → calcule le DOE, sauvegarde le cache
- `False` → recalcule toujours (ignore le cache)

Le JSON sauvegarde ne contient pas de signature — juste `xt`, `yt`, `all_grad`.

### Pourquoi ce choix

La signature automatique n'est pas robuste : si on change `fcm`, `cov_fc`, le maillage,
ou tout autre parametre qui affecte le SOCP, la signature ne le detecte pas (elle ne
teste que n0/params/modelname). Mieux vaut laisser l'utilisatrice decider explicitement.

### Comment appliquer a moulin_blanc

Remplacer `_doe_cache_sig()` + la comparaison dans `_load_doe_cache` par :

```python
config_is_identical = True   # ajouter dans les options utilisateur

def _load_doe_cache():
    if not config_is_identical:
        return None
    if not os.path.exists(_DOE_CACHE_FILE):
        return None
    try:
        d = json.load(open(_DOE_CACHE_FILE))
        return np.array(d["xt"]), np.array(d["yt"]), np.array(d["all_grad"])
    except Exception:
        return None
```

Et simplifier `_save_doe_cache` pour ne plus ecrire de signature :

```python
def _save_doe_cache(xt, yt, all_grad):
    json.dump({"xt": np.asarray(xt).tolist(),
               "yt": np.asarray(yt).tolist(),
               "all_grad": np.asarray(all_grad).tolist()},
              open(_DOE_CACHE_FILE, "w"), indent=1)
```

---

## Modif 7 — Log incremental JSONL generique

**Commit AC3** : *(a venir)*

### Ce qui change par rapport a moulin_blanc

Dans moulin_blanc, `_append_point_log` hardcode les noms `fy1`/`fy2` :

```python
# moulin_blanc :
rec = {"phase": phase, "round": _point_log_round[0],
       "u1": ..., "u2": ...,
       "fy1": float(_x[0]), "fy2": float(_x[1]),
       "g": ..., "lambda": ...}
```

Dans AC3, les champs sont construits depuis `params_names` :

```python
# AC3 (generique) :
rec = {"phase": phase, "round": _point_log_round[0],
       "g": None if g is None else float(g),
       "lambda": None if g is None else float(g) + 1.0}
for i, p in enumerate(params_names):
    rec[f"u_{p}"] = float(_u[i]) if i < len(_u) else None
    rec[f"x_{p}"] = float(_x[i]) if i < len(_x) else None
```

Produit `u_fc, u_fy, x_fc, x_fy` pour AC3, ou `u_fy1, u_fy2, x_fy1, x_fy2`
pour 2fy. Fonctionne pour n'importe quelle combinaison de `params_names`.

Fichier : `points_log.jsonl` (au lieu de `points_log_2fy.jsonl`).

### Comment appliquer a moulin_blanc

Remplacer le bloc `rec = {...}` dans `_append_point_log` par la version generique
ci-dessus. Renommer le fichier en `points_log.jsonl` (ou garder `_2fy` si prefere).
Aucun autre changement — les points d'appel (`run_HF`, `build_DOE`, `run_EFF`,
`print_results`, bloc principal reset) sont identiques.