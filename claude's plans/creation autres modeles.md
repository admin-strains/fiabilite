# Plan : AC2_voussoir — sensibilités sur voussoir_femelle_3

## Context

AC.py fait tourner le solveur sur voussoir_femelle_3 (analyse yield). AC2_pure_flexion.py fait la fiabilité sur test_pure_flexion (sensibilités + métamodèles). L'objectif est de créer une version AC2_voussoir.py qui fait la même chose sur voussoir_femelle_3, en partageant le même dossier .ds que AC.py.

---

## Architecture cible

```
C:\_workingDir\_SF\Autres modeles\_exportRebar\
    AC.py                    ← inchangé
    AC2_voussoir.py          ← NOUVEAU (copie modifiée d'AC2)
    launcher2_voussoir.py    ← NOUVEAU (copie modifiée de launcher2)
    InitSolver2.py           ← NOUVEAU (fusion des deux InitSolver)
    InitSolver.py            ← inchangé (utilisé par AC.py)

C:\workspace\storage\admin\SF\Autres modeles\EDF\voussoir_femelle_3.ds\
    dsCad.txt                ← MODIFIÉ (ajout fc/fy/fcd/fyd, str(fcd), str(fyd))
    dsLoad.txt               ← inchangé
    [résultats...]
```

---

## Fichier 1 — dsCad.txt (modification)

**Chemin :** `C:\workspace\storage\admin\SF\Autres modeles\EDF\voussoir_femelle_3.ds\dsCad.txt`

**Ajouter en tête du fichier (avant EXTERNAL_FILE) :**
```python
# --- Materiaux ---
fc    = 31.4944265047
fy    = 640.4488805901

# --- Coefficients de securite ---
gamma_c = 1.0
gamma_s = 1.0
fcd = fc / gamma_c
fyd = fy / gamma_s

```

**Remplacer la ligne IMPORT Import0 (ligne 4 actuelle) :**
```
# Avant
IMPORT('Import0', 'External_file0', ('idname', ['VF']), MATERIAL_TYPE='CONCRETE', ICRITERION= 2, COMPRESSIVE_STRENGTH='40', TENSILE_STRENGTH='0.5', YOUNG_MODULUS='21', DENSITY='2.5')

# Après
IMPORT('Import0', 'External_file0', ('idname', ['VF']), MATERIAL_TYPE='CONCRETE', ICRITERION= 2, COMPRESSIVE_STRENGTH=str(fcd), TENSILE_STRENGTH='0.5', YOUNG_MODULUS='21', DENSITY='2.5')
```

**Remplacer tous les GRADE=435 (695 occurrences) :**
```
GRADE=435  →  GRADE=str(fyd)
```
→ utiliser `replace_all=True` dans Edit

**Pourquoi AC.py continue de marcher :** AC.py fait `exec(cadscript)` qui évalue le Python de dsCad.txt dans l'ordre. Les variables fc, fy, fcd, fyd sont définies avant d'être utilisées dans IMPORT et REBAR. ✓

**Caveat shared state :** Après un run AC2, `patch_params()` laisse dsCad.txt avec les dernières valeurs fc/fy utilisées. Si AC.py tourne ensuite, il utilise ces valeurs modifiées. C'est inhérent à l'architecture de fichier partagé — acceptable par design.

---

## Fichier 2 — InitSolver2.py (fusion)

**Chemin :** `C:\_workingDir\_SF\Autres modeles\_exportRebar\InitSolver2.py`

Combine les deux InitSolver avec les choix suivants :

| Paramètre | test flexion | _exportRebar | **InitSolver2** |
|---|---|---|---|
| static PT INT [11] | 3 (MUMPS) | 3 (MUMPS) | **3** |
| static WriteLog [12] | 1 | 0 | **0** |
| static Max iter [13] | 300 | 300 | **300** |
| static Freq Fact Sym [15] | absent | 100 | **100** |
| static Cone border [10] | 0.99 | 0.90 | **0.99** |
| static Tol Pobj-Dobj [12] | 5e-4 | 1e-2 | **5e-4** (strict) |
| static Tol Res [13-16] | 1e-12 | 1e-12 | **1e-12** |
| cinem PT INT [21] | 3 (MUMPS) | 4 (CuDss) | **4 (CuDss)** → GPU, 47s sur voussoir |
| cinem Max iter [23] | 50 | 300 | **100** (50 trop peu pour voussoir) |
| cinem Cone border [20] | 0.95 | 0.90 | **0.95** |
| cinem Tol Pobj-Dobj [22] | 1e-3 | 1e-4 | **1e-3** (strict, fiabilité) |
| cinem Tol Res [23-26] | 1e-8 | 1e-12 | **1e-8** (moins strict que 1e-12 → évite 100+ iter) |
| cinem special init CVX [28] | 1.0 | absent | **1.0** (requis fiabilité) |

---

## Fichier 3 — launcher2_voussoir.py

**Chemin :** `C:\_workingDir\_SF\Autres modeles\_exportRebar\launcher2_voussoir.py`

Copie de launcher2.py avec :
- `sys.path.insert(0, r'C:\_workingDir\_SF\test flexion')` → `r'C:\_workingDir\_SF\Autres modeles\_exportRebar'`
- `exec(open(r'C:\_workingDir\_SF\test flexion\AC2_pure_flexion.py').read(), ...)` → `exec(open(r'C:\_workingDir\_SF\Autres modeles\_exportRebar\AC2_voussoir.py').read(), ...)`
- Conserver `sys.path.insert(0, r'C:\_workingDir\_SF\fiabilite')` (modules fiabilité partagés)

---

## Fichier 4 — AC2_voussoir.py

**Chemin :** `C:\_workingDir\_SF\Autres modeles\_exportRebar\AC2_voussoir.py`

Copie de `C:\_workingDir\_SF\test flexion\AC2_pure_flexion.py` avec les modifications suivantes :

### Chemins du modèle (4 occurrences)

| Ligne | Avant | Après |
|---|---|---|
| ~57 | `modelname = "test_pure_flexion"` | `modelname = "voussoir_femelle_3"` |
| ~58 | `"C:\\workspace\\storage\\admin\\SF\\" + modelname + ".ds"` | `"C:\\workspace\\storage\\admin\\SF\\Autres modeles\\EDF\\" + modelname + ".ds"` |
| ~356 (run_one_SOL) | `"C:\\workspace\\storage\\admin\\SF\\" + modelname + ".ds"` | idem ci-dessus |
| ~460 (run_HF) | `"C:\\workspace\\storage\\admin\\SF\\" + modelname + ".ds"` | idem ci-dessus |
| ~628 | `os.path.join(r'C:\workspace\storage\admin\SF', modelname + '.ds')` | `os.path.join(r'C:\workspace\storage\admin\SF\Autres modeles\EDF', modelname + '.ds')` |

### InitSolver (2 occurrences)

| Ligne | Avant | Après |
|---|---|---|
| ~403 | `exec(open(r"C:\_workingDir\_SF\test flexion\InitSolver.py").read(), globals())` | `exec(open(r"C:\_workingDir\_SF\Autres modeles\_exportRebar\InitSolver2.py").read(), globals())` |
| ~507 | idem | idem |

### Paramètres mesh globaux

Chercher les variables `global_size` et `geo_min_approx` dans le scope global d'AC2 et les mettre à :
```python
global_size = 0.05
geo_min_approx = 20
```
(valeurs qui convergent en 47s sur voussoir_femelle_3)

### Sensitivity regions (⚠️ point à vérifier)

```python
# Avant (test_pure_flexion)
kwargs["sensitivity_regions"] = json.dumps([
    {"param": "COMPRESSIVE_STRENGTH", "solids": ["Block1"]},
    {"param": "YIELD_STRENGTH", "rebars": rebar_names},
])

# Après (voussoir_femelle_3) — nom du solide béton à vérifier
kwargs["sensitivity_regions"] = json.dumps([
    {"param": "COMPRESSIVE_STRENGTH", "solids": ["Import0"]},
    {"param": "YIELD_STRENGTH", "rebars": rebar_names},
])
```
→ **Le nom "Import0" est à vérifier** : c'est le nom donné dans `IMPORT('Import0', ...)` dans dsCad.txt. Peut aussi être `"VF"` (idname). À confirmer en lisant le premier `.dsmetares` généré.

### rebar_names pour voussoir

Dans test_pure_flexion, `rebar_names` est une petite liste de noms. Pour voussoir, il y a 695 armatures (HA_14, HA_16, HA_20, HA_25). Deux options :
- **Option A** : construire dynamiquement la liste depuis le modèle après chargement (via API STRAINS)
- **Option B** : lister les noms de groupe (HA_14, HA_16, HA_20_0 à HA_20_87, HA_25) si STRAINS supporte les groupes
→ **À clarifier avec l'exécution** : lancer une fois avec `sensitivity=False` d'abord pour valider la mécanique.

### Figure de sortie (~ligne 1861)

```python
# Avant
plt.savefig(r'C:\_workingDir\_SF\test flexion\notre_gepck_hf.png', dpi=150)
# Après
plt.savefig(r'C:\_workingDir\_SF\Autres modeles\_exportRebar\notre_gepck_hf.png', dpi=150)
```

---

## Réponse à la question 7 — les chemins sont-ils corrects ?

### Ce qui est correct ✓
1. AC.py → `SF\Autres modeles\EDF\voussoir_femelle_3.ds` ✓
2. dsCad.txt → `coffrage_v2.stp` dans `SF\Autres modeles\EDF\` ✓ (déjà corrigé)
3. AC2_voussoir → même chemin .ds que AC.py ✓
4. `patch_params()` → fonctionne car dsCad.txt aura `fc = ...` et `fy = ...` en tête, cible exacte du regex `^fc\s*=.*$` ✓
5. COMPRESSIVE_STRENGTH dans le JSON résultat → clé présente dans `.dsmetares` si `sensitivity_analysis=true` ✓

### Ce qui manque ⚠️
1. **Nom du solide béton** dans `sensitivity_regions` : "Block1" ≠ voussoir. Probablement "Import0" mais à confirmer.
2. **`rebar_names`** : liste à adapter (695 rebars vs quelques-uns dans test_pure_flexion). Solution : construire depuis le modèle ou utiliser les noms de groupe.
3. **Pas de chemin manquant côté launcher** : le launcher2 de test flexion ajoute `sys.path` pour les modules fiabilité (`C:\_workingDir\_SF\fiabilite`). Ce chemin doit rester dans launcher2_voussoir.

---

## Ordre d'exécution

1. Modifier `dsCad.txt` (ajout fc/fy, str(fcd), str(fyd))
2. Créer `InitSolver2.py`
3. Créer `launcher2_voussoir.py`
4. Créer `AC2_voussoir.py` (copie + toutes modifications ci-dessus)
5. Lancer AC.py seul pour vérifier que dsCad.txt modifié ne casse pas le run normal
6. Lancer launcher2_voussoir.py avec sensitivity=False pour valider les chemins
7. Activer sensitivity=True et vérifier le nom du solide et rebar_names dans le .dsmetares

---

## Fichiers critiques à modifier/créer

| Fichier | Action |
|---|---|
| `...\voussoir_femelle_3.ds\dsCad.txt` | Modifier (ajout tête + str(fcd) + str(fyd)) |
| `..._exportRebar\InitSolver2.py` | Créer |
| `..._exportRebar\launcher2_voussoir.py` | Créer |
| `..._exportRebar\AC2_voussoir.py` | Créer (copie modifiée) |
