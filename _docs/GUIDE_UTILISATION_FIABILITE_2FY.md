# Guide d'utilisation — Calcul de fiabilité 2-fy (IS parallélisable)

Calcul de l'indice de fiabilité β d'un ouvrage via **analyse limite cinématique (SOCP)** + **surrogate GEPCK enrichi par EFF** + **Importance Sampling (IS) parallélisable**.

Cas de référence : pont du Moulin Blanc, **2 variables aléatoires de limite élastique acier** (`fy1`, `fy2`), une par groupe d'aciers.

---

## 1. Récupérer le code

Tout le code de fiabilité est sur la branche **`moulin_blanc`** du dépôt `fiabilite` :

```bash
git fetch
git checkout moulin_blanc
git pull
```

### Fichiers récupérés (rien à copier à la main)

| Fichier | Rôle |
|---|---|
| `_run/_run_moulin_blanc_2fy.bat` | **Point d'entrée** (lance le calcul) |
| `launcher_moulin_blanc_2fy.py` | Launcher : redirige le log, prépare les DLL, lance l'AC |
| `AC_moulin_blanc_2fy.py` | **Le calcul** (DOE → GEPCK → EFF → FORM/IS → β) |
| `InitSolver.py` | Paramètres du solveur SOCP (MUMPS, tolérances) |
| `_lib/_parallel_is.py` | ⭐ **Module IS parallèle** (sonde + ramp-up) |
| `_lib/branche1.py … branche5.py`, `branche_lars.py` | Surrogate GEPCK (PCE + krigeage) — dépendances de `branche1` |

> ⚠️ Les 6 fichiers `branche*` sont **tous** nécessaires (`_parallel_is` → `branche1` → `branche2..5` → `branche_lars`). Sinon `ImportError`.

### Ce qui N'EST PAS dans la branche (3 couches externes à avoir séparément)

1. **STRAINS compilé** → `front/STRAINS/` (le moteur SOCP / maillage / CAO : `CetSOLV`, `CetMESH`, `CetCAD`, `CetOPTI`…).
2. **Environnement Python** → conda `strains` : `openturns`, `smt`, `autograd`, `scipy`, `scikit-learn`, `matplotlib`, `numpy`, `threadpoolctl`.
3. **Le projet `.ds`** → `storage/admin/.../<projet>.ds` contenant `dsCad.txt`, `dsLoad.txt`, le `.stp` (géométrie), `doe_cache.json`.

---

## 2. Adapter les chemins (obligatoire)

Le code utilise des **chemins absolus en dur** `C:\workspace\...`. Si ton arborescence diffère, il faut les ajuster :

| Où | Quoi ajuster |
|---|---|
| `_run/_run_moulin_blanc_2fy.bat` | `cd /d C:\workspace\fiabilite`, `PYTHONPATH`, le `python.exe` |
| `launcher_moulin_blanc_2fy.py` | chemin de l'AC, `sys.path` vers `_lib` |
| `AC_moulin_blanc_2fy.py` | catalogues (`front\STRAINS\common\Catalog\*.json`), dossier projet `storage\admin\...`, `InitSolver.py`, dossier de sortie PNG |

Le plus simple : **reproduire l'arborescence `C:\workspace`** (front, fiabilite, storage) — aucun chemin à toucher.

---

## 3. Lancer un calcul

> ### ⚠️ À FAIRE AVANT LE 1er LANCEMENT (sinon plantage immédiat)
> Dans `AC_moulin_blanc_2fy.py`, mettre **`modelname` = ton projet `.ds`** (qui doit **exister** dans `storage/admin/...`).
> La valeur par défaut (`Calcul_fiabilite_13k_2fy_membrure_inf_diagonal`) est un **exemple** : si ce dossier `.ds` n'existe pas sur ta machine, le calcul **échoue dès la création du log / lecture du `dsCad.txt`**.
> → Vérifie aussi que ton `.ds` contient bien `dsCad.txt`, `dsLoad.txt`, le `.stp`, et (optionnel) `doe_cache.json`.

```bat
_run\_run_moulin_blanc_2fy.bat
```

> La parallélisation de l'IS est **active par défaut** (`_IS_PARALLEL` ON). Pour revenir à l'IS séquentiel OpenTURNS : `set _IS_PARALLEL=0` avant de lancer.

Le log détaillé (y compris la sortie C++ de STRAINS) est écrit dans :
`<projet>.ds/log_2fy_<horodatage>.log`

### Le workflow interne

```
_run_moulin_blanc_2fy.bat
   └─> launcher_moulin_blanc_2fy.py   (log + DLL + sous-process)
          └─> AC_moulin_blanc_2fy.py
                 1. lit dsCad.txt / dsLoad.txt / doe_cache.json
                 2. DOE (depuis cache si signature OK → 0 SOCP)
                 3. fit surrogate GEPCK (branche1)
                 4. BOUCLE EFF (enrichissement adaptatif) :
                      a. argmax EFF (NLopt GN_DIRECT)  → point u*
                      b. SOCP HF en u* (CetMESH + CetSOLV) → g_HF + sensibilité dg/dfy
                      c. refit GEPCK
                      d. FORM + IS sur 3 bandes (μ, g+2σ, g-2σ) → β + critères BB/BS
                 5. convergence → β final + planches PNG
```

Par point HF : `patch dsCad.txt (nouveaux fy)` → `exec dsCad → modèle OCC` → `maillage` → `SOCP (~80 s)` → `lecture g_HF + sensibilité`.

---

## 4. La parallélisation (`_IS_PARALLEL`)

### Ce que ça active

Le flag `_IS_PARALLEL` (**ON par défaut**) fait passer l'IS par le module **`_lib/_parallel_is.py`** (sonde + ramp-up) au lieu de l'IS séquentiel d'OpenTURNS. Pour **désactiver** et revenir au comportement d'origine : `set _IS_PARALLEL=0` avant de lancer.

### Quelle partie est parallélisée

```
Pour CHAQUE bande (μ, sup=g+2σ, inf=g-2σ) :        ← les 3 bandes restent SÉQUENTIELLES entre elles
    IS de la bande :
        • SONDE : tire des blocs un par un, COV vérifié à chaque bloc
                  → si COV ≤ cible : STOP (séquentiel, aucun pool)   ← cas le plus fréquent
        • RAMP-UP (si la sonde échoue, bande "dure") :
                  → pool de K process, chaque ronde = K × CHUNK blocs EN PARALLÈLE   ← LA partie parallélisée
                  → recollage exact des sommes partielles, COV vérifié par ronde
```

**On parallélise la boucle interne d'échantillonnage de l'IS** (les blocs Monte-Carlo), une bande à la fois — **PAS** les 3 bandes entre elles. Le SOCP, lui, n'est pas concerné (il est séquentiel).

### Réglages (variables d'environnement, optionnels)

| Variable | Défaut | Rôle |
|---|---|---|
| `_IS_PARALLEL` | (off) | active la parallélisation |
| `_IS_K` | 16 | nb de workers du ramp-up |
| `_IS_CHUNK` | 8 | nb de blocs par worker par ronde |
| `_IS_PROBE` | 16 | nb de blocs de la sonde séquentielle |
| `_IS_CAP` | `n_IS` | plafond de blocs |

---

## 5. Les variables aléatoires : 2 aciers (`fy1`, `fy2`)

Le cas de référence utilise **2 limites élastiques acier**, une par **groupe géométrique** :

```python
params_names = ['fy1', 'fy2']      # AC ligne ~115
```

- **Les groupes sont lus directement dans `dsCad.txt`** via le grade des aciers :
  - `GRADE=fyd1` → groupe 1 (`fy1`)
  - `GRADE=fyd2` → groupe 2 (`fy2`)
- Lois marginales : `fy*` → **Normale** (acier), centrée sur `FY_MEAN` (235 MPa), σ selon le COV.
- `fc` (béton) est **fixe** (`COMPRESSIVE_STRENGTH` dans le dsCad).
- La sensibilité `dg/dfy1`, `dg/dfy2` vient de STRAINS (clé `YIELD_STRENGTH:...`), mappée aux groupes via des aciers « sentinelles ».

**Prérequis projet** : le `dsCad.txt` DOIT contenir des aciers `GRADE=fyd1` ET `GRADE=fyd2` (sinon assertion).

---

## 6. Adapter à d'autres variables (ex. `fc` + `fy`)

La sélection des lois est automatique selon `params_names` (AC, `_dist_list`) :

```python
for p in params_names:
    if p.startswith('fy'):  → loi_fy(...)   # Normale (acier)
    elif p.startswith('fc'): → loi_fc(...)  # LogNormale (béton)
```

### Pour passer en `fc` + `fy` (béton + acier, 1 groupe)

1. **`params_names = ['fc', 'fy']`** → les lois LogNormale (fc) + Normale (fy) sont choisies automatiquement.
2. **Détection des groupes** : la logique actuelle (`GRADE=fyd1`/`fyd2`) est spécifique au 2-fy. Pour fc/fy il faut adapter le mapping :
   - `fc` ↔ clé sensibilité STRAINS `COMPRESSIVE_STRENGTH` (un seul béton),
   - `fy` ↔ `YIELD_STRENGTH` (un seul groupe acier).
   (Voir le banc `cas_test/AC_cas_test.py` qui implémente déjà ce cas fc/fy.)
3. **Vérifier les valeurs caractéristiques** (`fcm`, `fym`, COV) et les distributions.

> En clair : changer `params_names` suffit pour les **lois**, mais il faut **adapter la détection des groupes/sentinelles** et le **mapping clé-sensibilité → variable** selon les variables choisies. Le cas fc/fy est déjà fait dans `cas_test/`.

---

## 7. Réglages principaux dans l'AC

| Variable | Rôle |
|---|---|
| `modelname` | nom du projet `.ds` à calculer (à mettre au tien) |
| `n0` | nombre de points DOE initiaux |
| `print_HF` | `True` = calcule la « courbe rouge » de référence (49 SOCP HF, **lent**) ; `False` = direct enrichissement |
| `EFF_criteria` | critère d'arrêt de l'enrichissement (`at_least_one`, `n_points`…) |
| `cov_IS`, `n_IS` | cible de COV et taille de bloc de l'IS |

---

## 8. Note sur la reproductibilité (threading MKL)

Le solveur SOCP C++ de STRAINS utilise **MKL multi-thread** (~32 cœurs). Par défaut (aucune variable MKL définie) :

- ✅ **Rapide** (tous les cœurs) et **β correct**.
- ❌ **Pas reproductible bit-à-bit** : la sensibilité `dg/dfy` porte un bruit FP ~1e-10 (ordre de réduction des threads variable), **amplifié** par la sélection discrète LARS + l'enrichissement glouton EFF → le **chemin** d'enrichissement (les points choisis) varie d'un run à l'autre.
- Le **β final reste robuste** : toutes les trajectoires convergent vers la même valeur.

**Si la reproductibilité du chemin est nécessaire** (débogage, comparaison de runs), poser **avant le lancement** (dans le `.bat`, car MKL se charge avant Python) :
- `set MKL_CBWR=COMPATIBLE` → garde le multi-thread, fixe l'ordre de réduction (rapide + déterministe), **ou**
- `set OMP_NUM_THREADS=1` + `set MKL_NUM_THREADS=1` → déterministe mais **lent**.

> Note : `numpy/scipy` côté Python utilisent **OpenBLAS** (pas MKL) et le fit du surrogate est déterministe. Le bruit vient bien du **solveur C++ (MKL/MUMPS)**.

---

## 9. Sorties

Dans le dossier `<projet>.ds/` :
- `log_2fy_<ts>.log` — log complet
- `restart_state.json`, `points_log.jsonl` — état (reprise possible)
- `.dscad` / `.dsmed` / `.dsmetares` — fichiers intermédiaires STRAINS

Dans `output/png_EFF_moulin_blanc/png_EFF_<ts>/` :
- planches EFF (critère + écart-type surrogate), graphes de convergence.
