# Fiabilité 2 variables fy — notes de passation (Moulin Blanc)

> Document de passation : à lire par un autre Claude / un nouvel utilisateur avant de reprendre
> les calculs de fiabilité à **2 variables fy** (une résistance d'acier par groupe).
> Rédigé le 2026-06-17.

---

## 1. Contexte : DEUX projets

On a abandonné l'ancienne approche (variables `fc` béton + `fy` acier) parce que **fc était
totalement inerte** : `dg/dfc = 0.0` exactement (le calcul à la rupture BA est piloté par
l'acier, pas par le béton). On la remplace par **2 variables `fy`** (une par groupe d'acier),
`fc` étant désormais **fixé** (20 MPa).

Deux projets `.ds` existent dans `C:\workspace\storage\admin\Moulin_Blanc\` :

| Projet | Groupe 1 (fy1) | Groupe 2 (fy2) | Membrure inférieure |
|---|---|---|---|
| `Calcul_fiabilite_13k_2fy_membrure_inf_tablier.ds`  | 14082 aciers | 1264 | **dans le tablier** (groupe 1) |
| `Calcul_fiabilite_13k_2fy_membrure_inf_diagonal.ds` | 13858 aciers | 1488 | **dans le treillis** (groupe 2) |

La différence : la membrure inférieure longitudinale est rattachée soit au tablier, soit au
treillis. Le découpage des aciers en groupes a été fait dans `rebar_grouping/` (cas1 = tablier,
cas2 = diagonal), à partir de box STEP + barycentres (les cadres/étriers fermés sont exclus de
la bascule, seuls les longitudinaux bougent).

**Recommandation : commencer par le cas DIAGONAL** (`...membrure_inf_diagonal`), c'est le cas
de validation déjà lancé. C'est le `modelname` actuellement actif dans l'AC.

---

## 2. Le script de calcul

- **AC** : `C:\workspace\fiabilite\AC_moulin_blanc_2fy.py` (copie de `AC_moulin_blanc_LM1.py` adaptée)
- **Launcher** : `launcher_moulin_blanc_2fy.py` + `_run_moulin_blanc_2fy.bat`
- Pour **changer de cas** : éditer `modelname` (ligne ~74 de l'AC) entre les 2 projets.

Ce qui a été adapté pour 2-fy :
- `params_names = ['fy1', 'fy2']` (fc retiré)
- groupes lus **directement du dsCad** : `REBAR ... GRADE=fyd1` → groupe 1, `GRADE=fyd2` → groupe 2
- 2 lois Normales acier (μ=235 MPa, σ=√(19²+22²+8²)=30.15 MPa, JCSS)
- sensibilités : 2 régions `YIELD_STRENGTH` (group1_names / group2_names) → `dg/dfy1`, `dg/dfy2`
- chargement : poids propre **ET** trafic LM1 tous deux en LIVE (amplifiés par λ), `DEAD_LOAD_CASES=[]`

---

## 3. ⚠️ Paramètre `n_max_EFF = 200` (ligne ~180) — IMPORTANT, question ouverte

`n_max_EFF` = budget d'appels de l'optimiseur EFF (NLopt GN_DIRECT) qui cherche chaque point
d'enrichissement du surrogate.

**Historique (sur les anciens calculs LM1/gravité)** :
- Avec `n_max_EFF = 30` (valeur flexion d'origine), le premier calcul n'a ajouté que **2 points**
  d'enrichissement (l'optimiseur ratait le pic EFF, étroit, sur la grille 300×300).
- En passant **30 → 200**, l'optimiseur a trouvé un point de plus → **3 enrichissements**.

**Pour le cas 2-fy : on ne sait pas encore si 200 est nécessaire ou si 30 suffit.** À évaluer :
- Si avec 200 il ajoute beaucoup de points utiles → garder 200.
- Si 200 fait surtout tourner l'optimiseur pour rien (peu de points ajoutés vs 30) → **repasser à 30**.

➡️ **Action suggérée au prochain utilisateur** : comparer le nombre d'enrichissements EFF et le β
obtenu avec `n_max_EFF = 200` vs `30`, et choisir. La valeur actuelle est **200**.

---

## 4. Paramètre `tol_EFF = 1e-3` (ligne ~174)

Critère d'arrêt de l'enrichissement EFF (on continue tant que `EFF(u_opt) > tol_EFF`).
- Valeur **flexion d'origine = 1e-3** → **remise à 1e-3** (état actuel).
- Avait été testée à **1e-5** (plus stricte, force plus d'itérations EFF) sur LM1. Si besoin de
  plus de précision sur le surrogate, on peut re-tester 1e-5.

---

## 5. Lecture/écriture de la courbe HF (rouge) — NOUVELLE LOGIQUE : cache JSON automatique

**Avant** : la grille HF (49 calculs SOCP, ~2h30) était imprimée dans le log et il fallait
**copier-coller à la main** le tableau dans le `.py` (`hf_2d_grid_fixed = {...}`). Fini.

**Maintenant (automatique)** : `_compute_hf_grid_with_progress` gère un **cache sidecar JSON** :
- au début → `_load_hf_cache()` lit le fichier si présent ET si la signature correspond
  (bornes u1/u2, `n_grid_hf`, `params_names`, tailles des 2 groupes) → **0 recalcul**
- à la fin → `_save_hf_cache()` écrit le fichier

**Emplacement du fichier JSON** (ligne ~1824) :
```
C:\workspace\storage\admin\Moulin_Blanc\<modelname>.ds\hf_grid_cache.json
```
soit, pour le cas diagonal :
```
C:\workspace\storage\admin\Moulin_Blanc\Calcul_fiabilite_13k_2fy_membrure_inf_diagonal.ds\hf_grid_cache.json
```
→ **1 cache par projet** (tablier ≠ diagonal, pas de mélange). La source `.py` reste
`hf_2d_grid_fixed = None` à jamais (plus de gros tableau collé).

**⚠️ À VÉRIFIER LA PREMIÈRE FOIS** : lancer avec `print_HF = True` (ligne ~195) et confirmer que
le fichier `hf_grid_cache.json` est **bien créé** dans le dossier `.ds` après les 49 calculs
(chercher dans le log la ligne `[HF CACHE] sauve dans ...`). Au run suivant, vérifier
`[HF CACHE] charge depuis ... (signature OK -> 0 calcul SOCP)`.

**`print_HF`** est l'interrupteur maître (veut-on la courbe rouge ?). Le cache ne décide QUE de
recalculer vs réutiliser. Si `print_HF = False` → aucune courbe, aucun JSON, et le calcul de
fiabilité (β) tourne quand même (la courbe rouge n'est QUE de la visualisation/vérification).
Pour le 1er run de validation 2-fy, `print_HF = False`.

---

## 6. Chemins à ADAPTER pour un nouvel utilisateur

En lisant le code, voici tous les chemins en dur :

### Dans `AC_moulin_blanc_2fy.py`
| Ligne | Chemin / élément | À adapter |
|---|---|---|
| ~67 | `from branche1 import ...` | `branche1.py` doit être sur le PYTHONPATH (= `C:\workspace\fiabilite`) |
| ~74 | `modelname = "..."` | nom du projet `.ds` (basculer tablier/diagonal) |
| ~78, 496, 634, 833 | `C:\workspace\storage\admin\Moulin_Blanc\` + modelname + `.ds` | **racine des projets** (storage) |
| ~353 | `C:\workspace\fiabilite\output\png_EFF_moulin_blanc` | dossier de sortie des PNG EFF |
| ~565, 698 | `C:\workspace\fiabilite\InitSolver.py` | fichier d'init solveur |
| ~1824 | `os.path.join(_path_ds, "hf_grid_cache.json")` | cache HF (dérivé de `_path_ds`, rien à changer) |

### Dans `launcher_moulin_blanc_2fy.py`
| Élément | À adapter |
|---|---|
| `dll_dirs` = `C:\workspace\front\STRAINS\rupt\core\bin`, `...\core`, `...\common\Dll`, `...\meshgems`, `...\mosek` | **install STRAINS** (DLLs) |
| `sys.path` = `C:\workspace\front`, `C:\workspace\fiabilite` | repo front + repo fiabilité |
| `exec(open(r'C:\workspace\fiabilite\AC_moulin_blanc_2fy.py'))` | chemin de l'AC |

### Dans `_run_moulin_blanc_2fy.bat`
| Élément | À adapter |
|---|---|
| `cd /d C:\workspace\fiabilite` | repo fiabilité |
| `PYTHONPATH=C:\workspace\front;C:\workspace\fiabilite` | front + fiabilité |
| `C:\python3\python.exe` | **interpréteur Python** |

### Dans le `dsCad.txt` de chaque projet
| Élément | À adapter |
|---|---|
| `EXTERNAL_FILE("External_file0", "...<modelname>.ds/pont_complet.stp")` | déjà pointé sur la copie locale du STP du projet |

### Résumé des 4 racines à changer pour porter sur une autre machine
1. `C:\workspace\storage\admin\Moulin_Blanc\` — projets `.ds`
2. `C:\workspace\fiabilite\` — scripts (AC, branche1, InitSolver, launcher, output)
3. `C:\workspace\front\STRAINS\` — install STRAINS (DLLs)
4. `C:\python3\python.exe` — Python

---

## 7. Comment lancer

```bat
cd /d C:\workspace\fiabilite
_run_moulin_blanc_2fy.bat   > _calc_2fy_diagonal.log 2>&1
```
(ou via le launcher directement). Vérifier en tête de log le bloc **`VALIDATION CONFIG 2-FY`** :
distributions (2 Normal 235/30.15), fc fixe, tailles groupes, et surtout
`CHARGEMENT : DEAD vide=True | LIVE=poids+trafic amplifies=True`.

Pendant le DOE, chaque point logue `[SENSIBILITES 2-fy] pObj=... g=... dg/dfy1=... dg/dfy2=...`
→ confirme que les 2 dérivées reviennent bien (preuve que la sensibilité par groupe marche).

---

## 8. État au 2026-06-17

- Run de validation **cas diagonal** lancé avec `print_HF=False`, `tol_EFF=1e-5` (au moment du run ;
  depuis remis à **1e-3** dans le fichier pour les prochains runs), `n_max_EFF=200`.
- Validé : 2 charges amplifiées (travail dead=0 vs −0.466 avant), `dg/dfy1`+`dg/dfy2` distincts
  (1er point : 0.00127 et 0.0119 → le treillis ~9× plus sensible que le tablier).
- À faire : laisser finir le run (β à 2 variables), puis décider n_max_EFF 200 vs 30, puis lancer
  le cas tablier, et éventuellement activer `print_HF=True` pour la courbe rouge (vérifier la
  création du JSON la 1ère fois).
