# Transfert des modifications AC3 → moulin_blanc

Ce guide documente les modifications faites sur `AC3_pure_flexion.py` (branche `flexion`)
qui **divergent** de ce qui existe dans `AC_moulin_blanc_2fy.py` (branche `moulin_blanc`).

**L'objectif final : les deux codes doivent etre IDENTIQUES.** Les seules differences
autorisees sont :

- `modelname`, `_path_ds`, chemins du projet
- `params_names` (`['fc','fy']` vs `['fy1','fy2']`)
- `dsCad.txt` / `dsLoad.txt` (geometrie, charges)
- Parametres physiques (`fcm`, `fym`, `cov_*`, etc.)
- Logique specifique 2fy (`group1_names`, `_sens_key_to_param`, etc.) vs fc/fy

Toutes les fonctions utilitaires (`_save_socp_outputs`, `_append_point_log`,
`_save_restart_state`, `_batch_mu_sigma`, `_doe_cache_sig`, etc.) doivent etre
**le meme code** dans les deux fichiers. Ce n'est pas une "adaptation pour AC3" —
c'est la version finale commune que moulin_blanc doit aussi adopter.

## Principe de generalisation : tout passe par `params_names`

Dans moulin_blanc actuellement, plusieurs fonctions hardcodent `fy1`/`fy2` :

```python
# _save_socp_outputs : signature hardcodee
coords_str += f"_fy1{p1:.1f}_fy2{p2:.1f}"

# _append_point_log : champs hardcodes
rec = {"fy1": float(_x[0]), "fy2": float(_x[1]), ...}

# _doe_cache_sig : taille des groupes hardcodee
return {"g1": len(group1_names), "g2": len(group2_names), ...}
```

Dans la version finale (celle d'AC3 et celle que moulin_blanc doit adopter), tout
passe par `params_names` :

```python
# _save_socp_outputs : generique
coords_str += "_" + "_".join(f"{params_names[i]}{p_vals[i]:.1f}" for i in range(len(p_vals)))

# _append_point_log : generique
for i, p in enumerate(params_names):
    rec[f"u_{p}"] = float(_u[i]) if i < len(_u) else None
    rec[f"x_{p}"] = float(_x[i]) if i < len(_x) else None

# _doe_cache_sig : generique (pas de g1/g2)
return {"n0": n0, "params": list(params_names), "n_var": n_var, "modelname": modelname}
```

Le meme code produit :
- Avec `params_names = ['fc','fy']` → `fc48.0_fy550.0`, champs `u_fc, x_fc, u_fy, x_fy`
- Avec `params_names = ['fy1','fy2']` → `fy1235.0_fy2210.0`, champs `u_fy1, x_fy1, u_fy2, x_fy2`
- Avec `params_names = ['fc','fy','F']` → marcherait aussi sans rien toucher

**Pourquoi retirer `g1`/`g2` de `_doe_cache_sig`** : ces champs servaient a invalider
le cache DOE si la taille des groupes changeait. Or la validation par signature est
remplacee par `config_is_identical` (flag explicite utilisateur). Le champ `signature`
dans le dump restart est purement informatif — `g1`/`g2` n'y apportent rien. Les
groupes eux-memes (`group1_names`, `_sens_key_to_param`, etc.) restent dans le code
moulin_blanc pour le parsing des sensibilites — ils disparaissent juste des fonctions
utilitaires partagees.

Les fix identiques (fix1 a fix6) ne sont pas listes car deja en place dans moulin_blanc.

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

## Modif 3a — DOE cache : validation par `config_is_identical` au lieu de signature

**Commit AC3** : `9b4b4f5`

### Ce qui change par rapport a moulin_blanc

Dans moulin_blanc (commit `9052a3c`), le cache DOE utilise une **signature automatique**
via `_doe_cache_sig()` pour decider si le cache est valide.

Dans AC3, le **DOE cache** ne se sert pas de `_doe_cache_sig()` pour la validation.
C'est un **flag explicite utilisateur** qui controle :

```python
config_is_identical = True   # True = reutilise doe_cache.json si present (0 SOCP DOE)
```

- `True` + fichier existe → charge le cache sans verification de signature
- `True` + fichier absent → calcule le DOE, sauvegarde le cache
- `False` → recalcule toujours (ignore le cache)

Le JSON sauvegarde ne contient pas de signature — juste `xt`, `yt`, `all_grad`.

> **NOTE** : la fonction `_doe_cache_sig()` n'est pas supprimee — elle est ajoutee
> dans la **modif 3b (restart state)** car le dump restart l'utilise pour ecrire
> un champ `signature` informatif dans le JSON. Elle est simplement **pas utilisee
> par le DOE cache** pour la validation (c'est `config_is_identical` qui decide).

### Pourquoi ce choix

La signature automatique n'est pas robuste : si on change `fcm`, `cov_fc`, le maillage,
ou tout autre parametre qui affecte le SOCP, la signature ne le detecte pas (elle ne
teste que n0/params/modelname). Mieux vaut laisser l'utilisatrice decider explicitement.

### Comment appliquer a moulin_blanc

1. Ajouter `config_is_identical = True` dans les options utilisateur.

2. Dans `_load_doe_cache`, remplacer la comparaison de signature par :
```python
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

3. Simplifier `_save_doe_cache` pour ne plus ecrire de signature :
```python
def _save_doe_cache(xt, yt, all_grad):
    json.dump({"xt": np.asarray(xt).tolist(),
               "yt": np.asarray(yt).tolist(),
               "all_grad": np.asarray(all_grad).tolist()},
              open(_DOE_CACHE_FILE, "w"), indent=1)
```

4. **Garder `_doe_cache_sig()`** (ne pas la supprimer) — elle est utilisee par le
   dump restart (modif 3b). La version generique (sans `g1`/`g2`) :
```python
def _doe_cache_sig():
    return {"n0": n0, "params": list(params_names), "n_var": n_var, "modelname": modelname}
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

---

## Modif 3b-prep — Renommage `n_max_EFF` + plafond `n_max_EFF_points`

**Commit AC3** : `eeca06a`

### Ce qui change par rapport a moulin_blanc

Dans moulin_blanc, `n_max_EFF` controle le budget NLopt (`setMaximumCallsNumber`) —
c'est le nombre d'evaluations du surrogate par NLopt pour trouver le max de EFF(u) a
chaque iteration. Ce n'est PAS un plafond sur le nombre de points EFF ajoutes au DOE.

Dans AC3, on clarifie :
- **`n_NLopt_EFF = 30`** : budget NLopt GN_DIRECT (renommage de `n_max_EFF`)
- **`n_max_EFF_points = 30`** : plafond de points EFF ajoutes (nouveau)

`len(xt_eff) < n_max_EFF_points` est ajoute au debut de **toutes les `_cond`** de
`run_EFF` (BB, BS, both, at_least_one, else). La boucle EFF s'arrete si :
- BB/BS converge (comme avant), **ou**
- le plafond `n_max_EFF_points` est atteint (nouveau)

### Pourquoi ce choix

Le plafond sert de filet de securite en run normal ET de mecanisme pour le mode
restart (modif 3b) : en reprise, `xt_eff` demarre non-vide (charge du dump), donc
`len(xt_eff) < n_max_EFF_points` laisse automatiquement le budget restant.
Pas besoin de `n_enrich_extra` comme dans moulin_blanc.

### Comment appliquer a moulin_blanc

1. Renommer `n_max_EFF` en `n_NLopt_EFF` dans les 2 appels `setMaximumCallsNumber`
2. Ajouter `n_max_EFF_points = 30` dans les options utilisateur
3. Ajouter `len(xt_eff) < n_max_EFF_points and` au debut de chaque `_cond`
4. Supprimer `n_enrich_extra` (devenu inutile — voir modif 3b)

---

## Modif 3b — Restart state (dump + reprise)

**Commit AC3** : `46991be`

Cette modif a deux parties : le dump (ecriture du JSON) et la reprise (chargement).

### Partie A — Le dump (`_save_restart_state`)

**Identique a moulin_blanc** sauf 2 points :

1. **`_doe_cache_sig()`** : la version AC3 n'a pas `g1`/`g2` (pas de groupes acier) :
```python
# moulin_blanc :
def _doe_cache_sig():
    return {"n0": n0, "params": list(params_names), "n_var": n_var,
            "g1": len(group1_names), "g2": len(group2_names), "modelname": modelname}

# AC3 :
def _doe_cache_sig():
    return {"n0": n0, "params": list(params_names), "n_var": n_var, "modelname": modelname}
```
Cette fonction est appelee par `_save_restart_state` pour le champ `signature` informatif.
Elle n'est PAS utilisee par le DOE cache (qui est controle par `config_is_identical`).

2. **Nom du fichier** : `restart_state.json` au lieu de `restart_state_2fy.json`.

**Tout le reste est identique** : les 22 champs du JSON, la fonction `_u_beta`, les
appels (incremental dans `run_EFF` apres chaque point, final apres `print_visu`),
la globale `_eff_history_beta_IS`, le snapshot en fin de `run_EFF`.

### Partie B — La reprise (`restart_enrich_only`)

Le bloc de chargement dans le code principal est **identique a moulin_blanc** :
charge le JSON, injecte xt/yt/all_grad/max_degree/hf_2d_grid/historiques, set
`_enrich_round`/`_round_sizes_prev`/`_restart_xt_eff`/`_point_log_round`, append
marqueur `_RESTART` dans le point log.

Le seeding dans `run_EFF` est **identique** :
- `xt_eff = list(_restart_xt_eff) if restart_enrich_only else []`
- `list_beta_IS` seede depuis `_eff_history_beta_IS` si restart
- Historiques non reset si restart

**Une divergence majeure : pas d'override de `_cond` en mode reprise.**

Dans moulin_blanc, la reprise ecrase `_cond` et ignore BB/BS :
```python
# moulin_blanc :
if restart_enrich_only:
    _cond = lambda: (len(xt_eff) - _n_eff_start) < n_enrich_extra
```

Dans AC3, `_cond` est **identique en mode normal et en mode reprise**. Les criteres
BB/BS restent actifs. Le plafond `n_max_EFF_points` (modif 3b-prep) controle le
budget total via `len(xt_eff) < n_max_EFF_points` dans chaque `_cond`.

| | Moulin blanc | AC3 |
|---|---|---|
| `_cond` en reprise | Override : force N points, ignore BB/BS | Pas d'override : meme `_cond` qu'en normal |
| Comment ajouter des points | `n_enrich_extra = 5` | Augmenter `n_max_EFF_points` (ex: 30 → 50) |
| BB/BS actifs en reprise | Non | Oui |
| `n_enrich_extra` | Oui (variable utilisateur) | Non (inutile) |

### Workflow typique en mode reprise (AC3)

1. Run initial (`n_max_EFF_points=30`) → sort par le plafond apres 30 pts, BB/BS pas converge
2. Pas contente → mettre `restart_enrich_only=True`, augmenter `n_max_EFF_points=50`
3. La reprise charge les 30 pts, repart dans EFF, continue jusqu'a BB/BS converge ou 50 atteint
4. Si sort encore par plafond → `n_max_EFF_points=70`, relancer

### Comment appliquer a moulin_blanc

1. Retirer `n_enrich_extra` et son override de `_cond` en mode restart
2. Ajouter `n_max_EFF_points` dans les `_cond` (modif 3b-prep ci-dessus)
3. Renommer `restart_state_2fy.json` en `restart_state.json`
4. Adapter `_doe_cache_sig()` (retirer `g1`/`g2` ou garder si pertinent pour 2fy)