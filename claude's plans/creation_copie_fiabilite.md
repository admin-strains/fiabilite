# Guide : Créer une copie fiabilité pour un nouveau modèle STRAINS

**Objectif :** Adapter le pipeline de fiabilité (FORM + métamodèle) à un nouveau modèle STRAINS,
en partant du code de référence `AC2_pure_flexion.py` et du workflow EDF (`_exportRebar`) comme exemple concret déjà réalisé.

---

## 1. Comment fonctionne AC2 et son lancement

### Structure générale
`AC2_pure_flexion.py` est le script principal de fiabilité. Il contient :
- Un bloc `OPTIONS` en tête (dans `if __name__ == '__main__'`) : modèle actif, flags `do_KRG / do_GEK / do_HF / do_GEPCK`, n0, tolérances FORM, paramètres visu
- Des fonctions imbriquées définies à l'intérieur du `if __name__ == '__main__'` : `run_HF`, `run_one_SOL`, `build_DOE`, `build_metamodel_*`, `FORM_all_modes`, `print_visu`, etc.
- Un corps principal en fin de fichier qui orchestre le pipeline : DOE → EFF → FORM → résultats

Le code STRAINS est appelé à deux endroits : `run_HF` (1 point à la fois, avec gradient) et `run_one_SOL` (liste de points, avec gradient). Ce sont les seules fonctions qui touchent STRAINS.

### Lancement
**Ne jamais lancer `AC2_pure_flexion.py` directement.** Il faut passer par le launcher :

```
python launcher2.py
```

depuis le terminal conda habituel (l'environnement avec STRAINS installé).

### Rôle du launcher
`launcher2.py` (adresse : `C:\_workingDir\_SF\test flexion\launcher2.py`) fait trois choses :
1. Importe `openturns` **avant** STRAINS pour éviter le conflit MKL
2. Ajoute les DLL STRAINS au path Windows via `os.add_dll_directory`
3. Exécute le script principal via `exec(open(...).read(), {'__name__': '__main__'})`

Le `exec` avec `{'__name__': '__main__'}` est indispensable — il force l'exécution du bloc `if __name__ == '__main__'` du script cible.

---

## 2. Créer AC_[modele].py et launcher_[modele].py

### Principe
On part d'`AC2_pure_flexion.py` et `launcher2.py`. On les copie, on ne les modifie pas.

```
AC2_pure_flexion.py  →  AC_[modele].py
launcher2.py         →  launcher_[modele].py
```

Adresse des fichiers de référence :
- `C:\_workingDir\_SF\test flexion\AC2_pure_flexion.py`
- `C:\_workingDir\_SF\test flexion\launcher2.py`

### Point critique sur la copie du launcher
Dans `launcher_[modele].py`, il y a **une seule ligne à changer** après la copie :

```python
# avant (launcher2.py)
exec(open(r'C:\_workingDir\_SF\test flexion\AC2_pure_flexion.py').read(), {'__name__': '__main__'})

# après (launcher_[modele].py)
exec(open(r'C:\_workingDir\_SF\test flexion\AC_[modele].py').read(), {'__name__': '__main__'})
```

Et le print de démarrage pour s'y retrouver dans les logs :
```python
# avant
import sys; print("LAUNCHER2 START", flush=True)
# après
import sys; print("LAUNCHER_[MODELE] START", flush=True)
```

### DLL dans le launcher
Le launcher de référence (`launcher2.py`) contient ces DLL, suffisantes pour le modèle flexion pure :
```python
dll_dirs = [
    r'C:\workspace\front\STRAINS\rupt\core\bin',
    r'C:\workspace\front\STRAINS\rupt\core',
    r'C:\workspace\front\STRAINS\common\Dll',
    r'C:\workspace\front\STRAINS\rupt\core\bin\meshgems',
    r'C:\workspace\front\STRAINS\rupt\core\bin\mosek',
]
```

Certains modèles nécessitent des DLL supplémentaires. Pour le modèle EDF (voussoir), il a fallu **ajouter** (pas remplacer) :
```python
    r'C:\workspace\front\01_3RDPARTY\03_meshgems\lib\Win10_64_VC17',  # meshgems EDF
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin',   # GPU
```

**Comment savoir quelles DLL ajouter :** lire le `AC.py` du workflow STRAINS du nouveau modèle (voir section 3) — les `os.add_dll_directory` en tête de ce fichier donnent la liste exacte. Ces DLL peuvent être déclarées soit en tête du `AC.py` source (cas des scripts autonomes comme `_exportRebar\AC.py`), soit dans le launcher correspondant (cas de `launcher2.py` / `launcher_edf.py`). Vérifier les deux endroits.

---

## 3. Copie du dossier _exportRebar et du dossier .ds

### Contexte
Chaque modèle STRAINS est fourni avec :
- Un dossier `.ds` (ex: `voussoir_femelle_3.ds`) contenant `dsCad.txt`, `dsLoad.txt`, les fichiers compilés
- Un script de lancement autonome (`AC.py` dans `_exportRebar`) qui sait faire tourner ce modèle

Ces fichiers **ne doivent jamais être modifiés directement**. On travaille sur des copies.

### Copie du dossier .ds
Copier le dossier `.ds` dans le répertoire SF :
```
[source]  C:\workspace\storage\admin\EDF\voussoir_femelle_3.ds
[copie]   C:\workspace\storage\admin\SF\voussoir_femelle_3.ds
```

### Copie du dossier exportRebar
Copier le dossier du workflow STRAINS dans `_workingDir\_SF` :
```
[source]  C:\_workingDir\_exportRebar
[copie]   C:\_workingDir\_SF\_exportRebar
```

Attention : certains fichiers Rhino (`.3dm`) peuvent être verrouillés si ouverts. Les copier manuellement depuis l'explorateur si besoin.

### Modification dans la copie de AC.py
Dans la copie `C:\_workingDir\_SF\_exportRebar\AC.py`, **une seule ligne à changer** :

```python
# avant
path = "C:\\workspace\\storage\\admin\\EDF\\" + modelname + ".ds"
# après
path = "C:\\workspace\\storage\\admin\\SF\\" + modelname + ".ds"
```

Toutes les autres lignes (DLL, catalogues STRAINS, InitSolver) restent inchangées.

### Le EXTERNAL_FILE dans dsCad.txt
Le `dsCad.txt` contient souvent une ligne :
```
EXTERNAL_FILE("External_file0","C:\workspace\storage\admin\EDF\coffrage_v2.stp")
```
Ce chemin pointe vers le fichier géométrie `.stp`. Il peut **rester inchangé** dans la copie SF — le `.stp` n'a pas besoin d'être dupliqué, il est juste lu en entrée.

---

## 4. Modification du dsCad.txt (copie SF)

### Objectif
Rendre `fc` et `fy` paramétrables pour que `patch_params` dans `AC_[modele].py` puisse les modifier à chaque appel STRAINS.

### Ce qu'il faut lire en référence
`C:\workspace\storage\admin\SF\test_pure_flexion.ds\dsCad.txt` — c'est le modèle flexion pure qui montre la bonne structure : variables en tête, `fcd`/`fyd` calculés, utilisés dans `BLOCK` et `REBAR`.

### Modifications à faire sur le dsCad.txt de la copie SF

**Étape 1 — Ajouter le bloc de variables en tête du fichier** (avant `EXTERNAL_FILE` ou toute autre instruction) :
```python
# --- Materiaux ---
fc = 40.0          # valeur nominale (sera écrasée par patch_params à chaque appel)
fy = 435.0         # valeur nominale (sera écrasée par patch_params à chaque appel)

# --- Coefficients de securite ---
gamma_c = 1.0
gamma_s = 1.0
fcd = fc / gamma_c
fyd = fy / gamma_s
```

Les valeurs `fc` et `fy` sont les valeurs nominales du matériau (lire le dsCad original pour les trouver). `gamma_c` et `gamma_s` sont fixés à 1.0 pour la fiabilité (pas de coefficients partiels).

**Étape 2 — Remplacer la résistance béton dans IMPORT** :
```python
# avant
IMPORT('Import0', ..., COMPRESSIVE_STRENGTH='40', ...)
# après
IMPORT('Import0', ..., COMPRESSIVE_STRENGTH=str(fcd), ...)
```

**Étape 3 — Remplacer la limite élastique dans tous les REBAR** (replace_all) :
```python
# avant
REBAR('HA_20_0', ..., GRADE=435, ...)
# après
REBAR('HA_20_0', ..., GRADE=fyd, ...)
```

Si la valeur numérique dans `GRADE=` est différente de 435, adapter. Vérifier dans le dsCad original.

### Comment patch_params fonctionne
`patch_params` dans `AC_[modele].py` écrit directement dans `dsCad.txt` via regex :
```python
content = re.sub(r'^fc\s*=.*$', f'fc    = {value:.10f}', content, flags=re.MULTILINE)
```
Il cherche `fc = ...` et `fy = ...` au début des lignes. C'est pourquoi ces variables **doivent exister** à la racine du dsCad, pas dans une fonction.

---

## 5. Modifications de run_HF et run_one_SOL dans AC_[modele].py

### Méthode utilisée
Pour trouver quoi changer dans `run_HF` et `run_one_SOL`, comparer le contenu de `AC.py` du workflow STRAINS du nouveau modèle (copié dans `_workingDir\_SF\_exportRebar\AC.py`) avec le code de ces deux fonctions dans `AC2_pure_flexion.py` (`C:\_workingDir\_SF\test flexion\AC2_pure_flexion.py`).

**Lire côte à côte :**
- Le bloc `kwargs` dans `AC.py` (ce que le modèle a besoin)
- Le bloc `kwargs` dans `run_HF` et `run_one_SOL` de AC2

Tout ce qui est dans `AC.py` mais absent de `run_HF`/`run_one_SOL` doit être ajouté.

### 5.1 — Chemin du modèle (`path` et `modelname`)

Dans le bloc `if __name__ == '__main__'` en tête d'`AC_[modele].py` :
```python
# avant (modèle flexion)
modelname = "test_pure_flexion"
_path_ds = "C:\\workspace\\storage\\admin\\SF\\" + modelname + ".ds"

# après (nouveau modèle)
modelname = "voussoir_femelle_3"   # nom du dossier .ds
_path_ds = "C:\\workspace\\storage\\admin\\SF\\" + modelname + ".ds"
```

Le `path` dans `run_HF` et `run_one_SOL` est construit de la même façon et utilise aussi `modelname` — pas besoin de le changer séparément si `modelname` est bien mis à jour.

### 5.2 — Noms des armatures (`rebar_names`)

La ligne originale dans AC2 génère des noms séquentiels (`HA1`, `HA2`...) qui ne correspondent pas aux vrais noms du nouveau modèle :
```python
# avant — mauvais pour un nouveau modèle
n_rebars = len(re.findall(r'REBAR\(', _cad_txt))
rebar_names = [f"HA{i+1}" for i in range(n_rebars)]

# après — extrait les vrais noms depuis le dsCad
rebar_names = re.findall(r"REBAR\('([^']+)'", _cad_txt)
```

Cette version fonctionne pour tous les modèles quel que soit le format des noms (`HA1`, `HA_20_0`, etc.).

### 5.3 — `sensitivity_regions` (critique)

C'est le point le plus important. Si les noms ne correspondent pas, STRAINS ne renvoie pas les sensibilités et FORM plante (gradient = None).

**Comment trouver les bons noms :**
- Nom du solide béton : chercher `IMPORT(..., MATERIAL_TYPE='CONCRETE', ...)` dans le dsCad → c'est le `idname` entre crochets (ex: `['VF']` → nom = `"VF"`)
- Nom du solide dans AC2 original = `"Block1"` (modèle flexion pure)

```python
# avant (modèle flexion)
kwargs["sensitivity_regions"] = json.dumps([
    {"param": "COMPRESSIVE_STRENGTH", "solids": ["Block1"]},
    {"param": "YIELD_STRENGTH", "rebars": rebar_names},
])

# après (modèle voussoir EDF)
kwargs["sensitivity_regions"] = json.dumps([
    {"param": "COMPRESSIVE_STRENGTH", "solids": ["VF"]},
    {"param": "YIELD_STRENGTH", "rebars": rebar_names},
])
```

Ce changement est à faire **aux deux endroits** : dans `run_one_SOL` et dans `run_HF`.

### 5.4 — kwargs supplémentaires spécifiques au modèle

Certains modèles nécessitent des paramètres `kwargs` absents dans AC2. Les identifier en comparant le `kwargs` de `AC.py` avec celui de `run_HF`.

Pour le modèle voussoir EDF, il fallait ajouter **après** `kwargs["welds_throat_limit"] = True` (dans les deux fonctions) :
```python
kwargs["activated_thermic"] = False
kwargs["X0ConeC"] = 1.0e-2
kwargs["X0ConeT"] = 1.0e-2
kwargs["S0ConeC"] = 1.0e2
kwargs["S0ConeT"] = 1.0e2
```

Ces paramètres d'initialisation des cônes sont propres à ce modèle — ne pas les ajouter aveuglément à d'autres modèles sans avoir vérifié dans leur `AC.py`.

### 5.5 — Paramètres de maillage (`global_size` et `geo_min_approx`)

Ces deux paramètres sont définis dans le bloc `# PARAMETRES MESH` des OPTIONS (choix utilisateur) et injectés dans `Meshkwargs` dans `run_HF` et `run_one_SOL` :

```python
# PARAMETRES MESH
global_size     = 0.05   # global_physical_size
geo_min_approx  = 4      # geometric_approximation_min
```

Puis dans Meshkwargs (automatique, pas à toucher) :
```python
"global_physical_size": global_size,
"geometric_approximation_min": str(geo_min_approx),
```

Valeurs de référence :
- `global_size = 0.05`, `geo_min_approx = 4` → maillage rapide, adapté FORM (valeurs AC2)
- `global_size = 0.007`, `geo_min_approx = 4` → très fin (~400k nœuds, valeur `AC.py` EDF)
- `global_size = 0.5`, `geo_min_approx = 35` → grossier, pour un premier test rapide (valeurs AC_edf)

Lire le `AC.py` source du nouveau modèle pour connaître les valeurs utilisées en production.

---

## 6. InitSolver_[modele].py

### Pourquoi c'est important
`InitSolver.py` contient les paramètres du solveur (solveur PT INT, tolérances, nombre d'itérations max). Il est différent selon les modèles car il est calibré pour chacun.

Le fichier appelé par AC2 (`C:\_workingDir\_SF\test flexion\InitSolver.py`) est calibré pour le modèle de flexion pure. Il **ne convient pas** à un autre modèle.

### Méthode
1. Lire `InitSolver.py` du workflow source (ex: `C:\_workingDir\_SF\_exportRebar\InitSolver.py`)
2. Lire `InitSolver.py` dans `test flexion`
3. Comparer les deux — les différences clés sont dans `cinematic_params` (solveur, max itérations, tolérances)
4. Si différents : copier la version source sous un nouveau nom dans `test flexion` :

```
C:\_workingDir\_SF\_exportRebar\InitSolver.py
→ C:\_workingDir\_SF\test flexion\InitSolver_[modele].py
```

5. Dans `AC_[modele].py`, remplacer **toutes les occurrences** (2 : une dans `run_one_SOL`, une dans `run_HF`) :
```python
# avant
exec(open(r"C:\_workingDir\_SF\test flexion\InitSolver.py").read(), globals())
# après
exec(open(r"C:\_workingDir\_SF\test flexion\InitSolver_[modele].py").read(), globals())
```

### Différences constatées pour le modèle EDF (à titre d'exemple)

| Paramètre | InitSolver flexion | InitSolver EDF |
|---|---|---|
| cinematic [21] Solveur | `3` (MUMPS) | `4` (CuDss/GPU) |
| cinematic [23] Max iter | `50` | `300` |
| cinematic [20] Cone coef | `0.95` | `0.90` |
| cinematic [22] Tol. rel. | `1e-3` | `1e-4` |
| cinematic [28] special init | absent | `1.0` |

---

## Récapitulatif des fichiers créés pour le modèle EDF

| Fichier | Adresse | Créé depuis |
|---|---|---|
| `AC_edf.py` | `C:\_workingDir\_SF\test flexion\` | copie de `AC2_pure_flexion.py` |
| `launcher_edf.py` | `C:\_workingDir\_SF\test flexion\` | copie de `launcher2.py` |
| `InitSolver_edf.py` | `C:\_workingDir\_SF\test flexion\` | copie de `_exportRebar\InitSolver.py` |
| `voussoir_femelle_3.ds\` | `C:\workspace\storage\admin\SF\` | copie de `admin\EDF\voussoir_femelle_3.ds\` |
| `_exportRebar\` | `C:\_workingDir\_SF\` | copie de `C:\_workingDir\_exportRebar\` |

## Récapitulatif des modifications faites dans AC_edf.py

| Quoi | Où | Changement |
|---|---|---|
| `modelname` | ligne ~58 | `"test_pure_flexion"` → `"voussoir_femelle_3"` |
| `rebar_names` | ligne ~99 | génération séquentielle → `re.findall(r"REBAR\('([^']+)'", _cad_txt)` |
| `sensitivity_regions` | `run_one_SOL` | `"Block1"` → `"VF"` |
| `sensitivity_regions` | `run_HF` | `"Block1"` → `"VF"` |
| `kwargs` cone + thermic | `run_one_SOL` | ajout `activated_thermic`, `X0ConeC/T`, `S0ConeC/T` |
| `kwargs` cone + thermic | `run_HF` | ajout `activated_thermic`, `X0ConeC/T`, `S0ConeC/T` |
| `InitSolver` call | `run_one_SOL` + `run_HF` | `InitSolver.py` → `InitSolver_edf.py` |
