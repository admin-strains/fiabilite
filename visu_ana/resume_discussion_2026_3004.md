# Résumé de discussion — Refactoring T_inv dans flexion_simple

---

## GUIDE D'UTILISATION DE CE FICHIER

**Ce que signifie "ajoute au md" :** quand l'utilisatrice demande "ajoute au md", elle veut que Claude mette à jour CE fichier (`resume_discussion_2026_3004.md`) en ajoutant une nouvelle section numérotée qui documente ce qui vient d'être fait dans la session : modifications de code, décisions prises, problèmes rencontrés, résultats obtenus. Le style attendu est précis et technique : noms de fichiers exacts, extraits de code avant/après, valeurs numériques. Ce fichier est relu en début de session suivante pour reprendre le contexte sans avoir à tout réexpliquer.

---

**Date de début :** 30 avril 2026  
**Fichier principal travaillé :** `C:\_workingDir\_SF\test flexion\visu_ana\2026_3004_calcul_de_pf.py` puis `2026_3004_calcul_de_pf_to_copy.py`  
**Fichier origine (lu en début de session) :** `C:\_workingDir\2026_1802_calcul_de_pf.py`

---

## 1. Contexte général

Le fichier `2026_1802_calcul_de_pf.py` (et sa copie `2026_3004_calcul_de_pf.py`) implémente une classe `flexion_simple` pour le calcul de probabilité de défaillance en flexion simple d'une section en béton armé. Il contient :
- Une **fonction de performance** `f(u)` dans l'espace standard U (variables gaussiennes centrées réduites)
- Ses **dérivées** analytiques : gradient (`grad_f`) et laplacien (`lap_f`)
- Deux régimes : aciers **plastifiés** et **non plastifiés**, sélectionnés par `test_plast(u)`
- Une **visualisation** de la frontière d'état limite dans l'espace U

Les distributions considérées :
- `fc` (résistance béton) : initialement `N(48, 4.8)` MPa → objectif : lognormale EC2 via `loi_fc`
- `fy` (limite d'élasticité acier) : initialement `N(550, 30)` MPa → objectif : normale JCSS via `loi_fy`

---

## 2. Problème principal

### Problème
Dans chaque méthode de la classe (`f_plast`, `grad_f_plast`, `lap_f_plast`, `f_nonplast`, `grad_f_nonplast`, `lap_f_nonplast`, `test_plast`), le passage de l'espace standard U → espace physique X était codé **manuellement** en supposant des lois normales :

```python
x1 = u[0] * self.fc_std + self.fc_mean   # x_fc
x2 = u[1] * self.fy_std + self.fy_mean   # x_fy
```

Ce code est **incorrect pour des lois non normales** (ex. lognormale pour fc). Il fallait généraliser en utilisant la **transformation isoprobabiliste inverse** d'OpenTURNS.

### Contexte de référence
Le fichier `C:\_workingDir\_SF\test flexion\AC_pure_flexion.py` utilise déjà ce pattern (lignes 346–354) :

```python
dist = []
if 'fc' in params_names:
    dist.append(loi_fc(fck, cov_fck))
if 'fy' in params_names:
    dist.append(loi_fy(fyk, cov_fyk))
dist_X   = ot.JointDistribution(dist)
T     = dist_X.getIsoProbabilisticTransformation()
T_inv = dist_X.getInverseIsoProbabilisticTransformation()
```

---

## 3. Solutions proposées et retenues

### 3.1 Remplacement des lignes x1/x2 — solution retenue

**Deux lignes de remplacement (quand x1 ET x2 sont nécessaires) :**
```python
x_point = self.T_inv(ot.Point(u))
x1, x2  = x_point[0], x_point[1]
```

**Une ligne (quand seul x1 est nécessaire — cas non plastifié) :**
```python
x1 = self.T_inv(ot.Point(u))[0]
```

**API OpenTURNS confirmée (recherche doc) :**
- `T_inv = dist_X.getInverseIsoProbabilisticTransformation()` → retourne un objet `ot.Function`
- Appel : `T_inv(ot.Point(u))` → retourne un `ot.Point`
- Accès : `x_point[0]`, `x_point[1]` → floats
- Interface **identique** pour normale et lognormale (OT choisit automatiquement Nataf ou Rosenblatt)

### 3.2 Où définir T_inv — solution retenue : dans `__init__`

**Option rejetée :** module-level (problème de portée si plusieurs instances)  
**Option retenue :** stockée comme `self.T_inv` dans `__init__`

### 3.3 Paramètres de loi_fc / loi_fy — problème de sémantique

**Problème identifié :** `fc_params=(48, 4.8)` signifiait `(mean, std)` pour l'ancienne formule, mais `loi_fc(fck, cov)` attend `(fck [MPa], cov [sans dimension])`. Passer `loi_fc(48, 4.8)` serait incorrect (`cov=4.8 = 480%`, aberrant).

**Solution retenue (temporaire) :** Ajout de DEUX NOUVEAUX paramètres `fc_otparams` et `fy_otparams` à `__init__`, distincts de `fc_params`/`fy_params` qui sont **conservés** car utilisés ailleurs dans le code pour `self.fc_mean`, `self.fc_std`, `self.fy_mean`, `self.fy_std` (notamment dans les termes Jacobiens des gradients).

```python
def __init__(self, ..., fc_params, fy_params, fc_otparams, fy_otparams, fac_c=1.1):
```

**Valeurs utilisées au site d'appel :**
- `fc_otparams=(40, 0.09)` → `loi_fc(fck=40 MPa, cov=0.09)` → lognormale avec `fcm=48 MPa`, sigma≈4.3 MPa
- `fy_otparams=(500,)` → `loi_fy(fyk=500 MPa)` → normale avec `mu≈549.4 MPa`, `sigma=30 MPa` (JCSS)

**Correspondance physique avec l'ancienne paramétrisation :**
| Ancien | Nouveau | Explication |
|---|---|---|
| `fc_mean=48` | `fck=40, fcm=fck+8=48` | EC2 : fcm = fck + 8 MPa |
| `fc_std=4.8` | `cov=0.09≈4.3/48` | CoV JCSS C35 |
| `fy_mean=550` | `fyk=500, mu=500+1.645×30≈549` | JCSS : mu = fyk + 1.645σ |
| `fy_std=30` | `sigma=30 (JCSS)` | codé en dur dans loi_fy |

**Note importante :** `fc_params` et `fy_params` seront SUPPRIMÉS ultérieurement, quand les autres lignes du code utilisant `self.fc_std`, `self.fy_std` (termes Jacobiens dans gradients) auront été mises à jour pour utiliser le Jacobien analytique de `T_inv`. Ce chantier est distinct et pas encore traité.

### 3.4 Construction de dist_X — pattern params_names retenu

Conformément au pattern de `AC_pure_flexion.py`, `dist_X` est construite avec le filtre `params_names` :

```python
dist = []
if 'fc' in params_names:
    dist.append(loi_fc(*fc_otparams))
if 'fy' in params_names:
    dist.append(loi_fy(*fy_otparams))
dist_X     = ot.JointDistribution(dist)
self.T_inv = dist_X.getInverseIsoProbabilisticTransformation()
```

`params_names` est défini **en global** avant la classe (pour permettre la réutilisation par `__init__` sans le passer en argument) :

```python
params_names = ['fc', 'fy']
```

---

## 4. Modifications apportées au code (détail) — session 1

### 4.1 Imports ajoutés (en haut du fichier, après `from scipy.optimize import minimize`)

```python
import sys
sys.path.insert(0, r'C:\_workingDir\_SF\fiabilite')
from config.jcss_fc import loi_fc
from config.jcss_fy import loi_fy
import openturns as ot
```

### 4.2 Variable globale ajoutée (avant la définition de la classe)

```python
params_names = ['fc', 'fy']
```

### 4.3 Signature `__init__` — ajout de deux paramètres

```python
# AVANT
def __init__(self, Med, As, b, h, d, Es, ecu, fc_params, fy_params, fac_c=1.1):

# APRÈS
def __init__(self, Med, As, b, h, d, Es, ecu, fc_params, fy_params, fc_otparams, fy_otparams, fac_c=1.1):
```

### 4.4 `__init__` — ajout de la construction de `self.T_inv`

Inséré après `self.fy_mean, self.fy_std = fy_params` et avant les constantes de performance :

```python
# Transformation isoprobabiliste (espace standard U → espace physique X)
dist = []
if 'fc' in params_names:
    dist.append(loi_fc(*fc_otparams))
if 'fy' in params_names:
    dist.append(loi_fy(*fy_otparams))
dist_X     = ot.JointDistribution(dist)
self.T_inv = dist_X.getInverseIsoProbabilisticTransformation()
```

### 4.5 Méthodes modifiées — remplacement des lignes x1/x2

**Méthodes avec x1 ET x2** (`f_plast`, `grad_f_plast`, `lap_f_plast`, `test_plast`) :
```python
# AVANT
x1 = u[0] * self.fc_std + self.fc_mean
x2 = u[1] * self.fy_std + self.fy_mean

# APRÈS
x_point = self.T_inv(ot.Point(u))
x1, x2  = x_point[0], x_point[1]
```

**Méthodes avec x1 uniquement** (`f_nonplast`, `grad_f_nonplast`, `lap_f_nonplast`) :
```python
# AVANT
x1 = u[0] * self.fc_std + self.fc_mean

# APRÈS
x1 = self.T_inv(ot.Point(u))[0]
```

### 4.6 Site d'appel mis à jour

```python
# AVANT
calc = flexion_simple(Med=0.40, As=11.3254e-4, b=0.5, h=0.8, d=0.72,
                    Es=200000, ecu=0.0035,
                    fc_params=(48, 4.8), fy_params=(550, 30))

# APRÈS
calc = flexion_simple(Med=0.40, As=11.3254e-4, b=0.5, h=0.8, d=0.72,
                    Es=200000, ecu=0.0035,
                    fc_params=(48, 4.8), fy_params=(550, 30),
                    fc_otparams=(40, 0.09), fy_otparams=(500,))
```

### 4.7 Prints de vérification ajoutés (après instanciation de `calc`)

```python
x0 = calc.T_inv(ot.Point([0.0, 0.0]))
print(f"T_inv([0,0]) : x_fc = {x0[0]:.4f} MPa  |  x_fy = {x0[1]:.4f} MPa")
print(f"  (mediane lognormale fc attendue ~ 47.8 MPa, mediane normale fy attendue ~ 549.4 MPa)")
print(f"f([0,0])     = {calc.f([0.0, 0.0]):.6f}  (> 0 = securite au point median)")
print(f"test_plast([0,0]) = {calc.test_plast([0.0, 0.0])}  (1.0 = aciers plastifies)")
x1_ref = calc.T_inv(ot.Point([1.0, 0.0]))
print(f"T_inv([1,0]) : x_fc = {x1_ref[0]:.4f} MPa  (doit etre > x_fc au point median)")
```

---

## 5. Résultats des tests de cohérence

### Test standalone (distributions normales, vérification mathématique pure)
```
u=[0,0]  =>  x1=48.000000 (attendu 48)   x2=550.000000 (attendu 550)  ✓
u=[1,0]  =>  x1=52.800000 (attendu 52.8) x2=550.000000 (attendu 550)  ✓
T(T_inv(u0)) = [1.2000000000, -0.8000000000]  (round-trip exact)       ✓
```

### Test sur classe réelle (loi_fc lognormale + loi_fy normale)
```
T_inv([0,0]) : x_fc = 47.8068 MPa  |  x_fy = 549.5961 MPa
  (mediane lognormale fc ~ 47.8 MPa ✓, mediane normale fy ~ 549.4 MPa ✓)
f([0,0])     = 0.040052  (> 0 = sécurité ✓)
test_plast([0,0]) = 1.0  (aciers plastifiés ✓)
T_inv([1,0]) : x_fc = 52.2995 MPa  (> 47.81 ✓)
Round-trip T(T_inv([1.2,-0.8])) = [1.2000000000, -0.8000000000]  ✓
```

---

## 6. Points en suspens / chantier futur

### 6.1 Termes Jacobiens dans `grad_f_plast`, `lap_f_plast`, `grad_f_nonplast`, `lap_f_nonplast`

Ces méthodes utilisent encore `self.fc_std` et `self.fy_std` comme termes de la règle de chaîne (∂x/∂u) :

```python
der_u1 = -(self.B * x2**2 / x1**2) * self.fc_std   # ← Jacobien approx. Normal
der_u2 = (self.A + 2 * self.B * x2 / x1) * self.fy_std
```

Pour une **loi normale** : `∂x/∂u = sigma` → `self.fc_std` est correct.  
Pour une **lognormale** : `∂x/∂u = sigma_ln * x1` → `self.fc_std` est **incorrect**.  
**Action future :** remplacer par le Jacobien analytique de `T_inv`, récupérable via `dist_X.getIsoProbabilisticTransformation().gradient(x_point)` ou calcul analytique selon la loi.

### 6.2 Suppression de fc_params / fy_params
Une fois les Jacobiens corrigés, `fc_params`, `fy_params`, `self.fc_mean`, `self.fc_std`, `self.fy_mean`, `self.fy_std` pourront être supprimés → passage de 6 à 4 paramètres d'entrée (`fc_otparams`, `fy_otparams` remplacent tout).

---

## 7. Fichiers clés référencés

| Fichier | Rôle |
|---|---|
| `C:\_workingDir\_SF\test flexion\visu_ana\2026_3004_calcul_de_pf_to_copy.py` | Fichier actif (session 2+) |
| `C:\_workingDir\_SF\test flexion\visu_ana\2026_3004_calcul_de_pf.py` | Version session 1 |
| `C:\_workingDir\2026_1802_calcul_de_pf.py` | Version antérieure |
| `C:\_workingDir\_SF\test flexion\AC_pure_flexion.py` | Référence du pattern T_inv / params_names / loi_fc / loi_fy |
| `C:\_workingDir\_SF\test flexion\launcher.py` | Référence pour sys.path |
| `C:\_workingDir\_SF\fiabilite\config\jcss_fc.py` | Définition de `loi_fc(fck, cov=None)` → `ot.LogNormal` EC2 |
| `C:\_workingDir\_SF\fiabilite\config\jcss_fy.py` | Définition de `loi_fy(fyk, cov=None)` → `ot.Normal` JCSS |
| `C:\workspace\storage\admin\SF\test_pure_flexion.ds\dsCad.txt` | Géométrie et matériaux du modèle DS |
| `C:\workspace\storage\admin\SF\test_pure_flexion.ds\dsLoad.txt` | Chargement du modèle DS |

### Signatures importantes

**`loi_fc(fck, cov=None)`** (jcss_fc.py) :
- `fck` : résistance caractéristique [MPa] (fractile 5%)
- `cov` : CoV (si None → déduit de la classe JCSS la plus proche : C35 → 0.09)
- Calcule `fcm = fck + 8` (EN 1992), retourne `ot.LogNormal`

**`loi_fy(fyk, cov=None)`** (jcss_fy.py) :
- `fyk` : limite d'élasticité caractéristique [MPa]
- `cov` : si None → `sigma = SIGMA = 30 MPa` (JCSS), `mu = fyk + 1.645 × 30`
- Retourne `ot.Normal`

**Note :** Les versions `loi_fc_jcss` et `loi_fy_jcss` sont **obsolètes**. Utiliser uniquement `loi_fc` et `loi_fy`.

---

## 8. Chemin réseau important

Le fichier ne peut pas être accédé via `C:\` depuis ce poste (session distante).  
Chemin fonctionnel depuis Python : `//tsclient/C/_workingDir/...`  
Chemin fonctionnel depuis shell Windows : `C:\_workingDir\...` (via `python "C:\..."`)

---

## 9. Session 2 — Nettoyage et adaptation au modèle DS (30 avril 2026)

### 9.1 Nouveau fichier de travail

`2026_3004_calcul_de_pf.py` → **"Save As"** → `2026_3004_calcul_de_pf_to_copy.py`  
Toutes les modifications de la session 2 portent sur ce nouveau fichier.

### 9.2 Suppression des méthodes gradient et laplacien

Les méthodes `grad_f_plast`, `grad_f_nonplast`, `grad_f`, `lap_f_plast`, `lap_f_nonplast`, `lap_f` ne sont pas utilisées dans la section résultats (ni dans les prints ni dans la visualisation). Elles ont été supprimées.

Conséquence : `self.fc_std`, `self.fy_std`, `self.fc_mean`, `self.fy_mean` n'étaient plus référencés nulle part → `fc_params` et `fy_params` supprimés aussi.

**Signature `__init__` après nettoyage :**
```python
def __init__(self, Med, As, b, h, d, Es, ecu, fc_otparams, fy_otparams, fac_c=1.1):
```

### 9.3 Renommage de la fonction de performance

Toutes les méthodes renommées pour cohérence avec la convention `g` du reste du projet :

| Avant | Après |
|---|---|
| `f_plast` | `g_ana_plast` |
| `f_nonplast` | `g_ana_nonplast` |
| `f` | `g_ana` |
| `F_val` (variable grille) | `G_ana` |

### 9.4 Lecture des paramètres géométriques depuis les fichiers DS

Bloc ajouté avant l'appel à `flexion_simple`, qui parse `dsCad.txt` et `dsLoad.txt` :

```python
import re
with open(r'C:\workspace\storage\admin\SF\test_pure_flexion.ds\dsCad.txt', encoding='utf-8') as f:
    _cad = f.read()
with open(r'C:\workspace\storage\admin\SF\test_pure_flexion.ds\dsLoad.txt', encoding='utf-8') as f:
    _load = f.read()

def _parse(text, name):
    return float(re.search(rf'(?m)^\s*{re.escape(name)}\s*=\s*([\d.]+)', text).group(1))

b   = _parse(_cad, 'b')    # 0.5 m
h   = _parse(_cad, 'h')    # 0.8 m
L   = _parse(_cad, 'L')    # 5.0 m
phi = _parse(_cad, 'phi')  # 16.0 mm

n_bars = len(re.findall(r'REBAR\(', _cad))          # 6
As = n_bars * math.pi * (phi / 2e3) ** 2            # 12.06 cm²

z_rebar = [float(v) for v in re.findall(
    r'pts\d+\.append\(POINT\([^,]+,\s*[^,]+,\s*([\d.]+)\)', _cad)]
d = h - sum(z_rebar) / len(z_rebar)                 # 0.480 m

F = abs(float(re.search(r"Z='(-?[\d.]+)'", _load).group(1)))  # 0.1 MN

Med = F * L   # 0.5 MN·m (moment en console)

Es  = 200000
ecu = 0.0035
```

**Valeurs parsées :** b=0.5m, h=0.8m, L=5.0m, phi=16mm, n_bars=6, As=12.06 cm², d=0.480m, F=0.1 MN.

### 9.5 Paramètres de distribution codés en dur avant l'appel

Conformément au pattern de `AC_pure_flexion.py` :
```python
fck, fyk = 40, 500          # MPa
cov_fck, cov_fyk = None, None

calc = flexion_simple(Med=Med, As=As, b=b, h=h, d=d,
                    Es=Es, ecu=ecu,
                    fc_otparams=(fck, cov_fck), fy_otparams=(fyk, cov_fyk))
```

`cov=None` → loi_fc et loi_fy utilisent les défauts JCSS (même résultat que cov=0.09 pour fck=40).  
`fc_otparams` et `fy_otparams` ne sont **pas** lus du dsCad car ce sont des paramètres de distribution probabiliste (EC2/JCSS), indépendants des valeurs nominales du modèle DS.

### 9.6 Diagnostic : frontière g=0 hors de la zone [-6, 6]

Avec Med=0.5 MN·m, la capacité nominale de la section est ≈ 0.31 MN·m → `g_ana([0,0]) = -0.191` → la section cède à la charge nominale.

La frontière g=0 se trouve à **u2 ≈ 11.84** (fy ≈ 907 MPa nécessaire) pour u1=0, bien au-delà de [-6, 6].

**Solution temporaire :** `borne = 16` codée en dur (la borne sera fournie par le code appelant dans le flux final). Grille réduite à **100×100** pour limiter le temps de calcul.

### 9.7 Pattern g_ana dans print_visu (AC_pure_flexion.py)

La fonction `print_visu` de `AC_pure_flexion.py` a été augmentée d'un paramètre `g_ana=None` :
- Bloc ajouté : contour pleine grille 100×100, couleur verte `'-.'`, label `'g=0 ana'`
- Usage : `print_visu(..., g_ana=calc.g_ana)` où `calc` est une instance de `flexion_simple`
- Le code de `print_visu` modifié a été fourni dans la discussion mais **pas encore écrit dans le fichier** `AC_pure_flexion.py`

---

## 10. Session 3 — Visualisation frontière, sample_frontier1/2, grid_hf (4 mai 2026)

### 10.1 Refactoring global des variables de visualisation

Renommages et ajouts dans `2026_3004_calcul_de_pf_to_copy.py` :

| Avant | Après |
|---|---|
| `borne = 32` | `size_visu = 32` |
| — | `n_visu = 3` (global, utilisé pour les sous-sélections) |

Import ajouté : `from config.jcss_fy import loi_fy, SIGMA` (pour le calcul analytique de u2_min).

### 10.2 Refactoring de `sample_frontier`

Suppression des paramètres `u1_range`, `n`, `u2_range` — remplacés par des globales ou calculés analytiquement :

**Signature avant :**
```python
def sample_frontier(calc, u1_range, n=40, u2_range=(-18, 40), n_scan=30):
```

**Signature après :**
```python
def sample_frontier(calc, n_scan=100):
```

- `u1_range` → `(-size_visu, size_visu)` (globale)
- `n=40` → codé en dur dans la boucle
- `u2_range` inférieur → calculé analytiquement depuis `fyk`, `cov_fyk`, `SIGMA` :
  - `cov_fyk is None` : `u2_min = -(fyk/SIGMA + 1.645)`
  - `cov_fyk` fourni : `u2_min = -1/cov_fyk`
  - `u2_low = max(-size_visu, u2_min)` ≈ -18.23

Cette formule est dérivée de la condition `fy ≥ 0` sur la normale JCSS (`fy = mu + sigma*u2 ≥ 0`), dont la simplification donne `u2 ≥ -1/cov` quand cov est fourni.

### 10.3 Ajout de `make_grid`

Fonction retournant n×n points avec contraintes physiques `fc ≥ 0` (toujours vérifié — lognormale) et `fy ≥ 0` :

```python
def make_grid(fck, cov_fck, fyk, cov_fyk, size_visu, n):
    dist_fy  = loi_fy(fyk, cov_fyk)
    u2_min   = max(-size_visu, -dist_fy.getMean()[0] / dist_fy.getStandardDeviation()[0])
    u1_vals  = np.linspace(-size_visu, size_visu, n)
    u2_vals  = np.linspace(u2_min, size_visu, n)
    U1g, U2g = np.meshgrid(u1_vals, u2_vals)
    return np.column_stack([U1g.ravel(), U2g.ravel()])
```

### 10.4 Sous-sélection `grid_hf` depuis `frontier_pts`

Sous-échantillonnage uniforme de `frontier_pts` pour obtenir ~`2*n_visu` points HF :

```python
n_hf_target = 2 * n_visu
step        = max(1, len(frontier_pts) // n_hf_target)
grid_hf     = frontier_pts[::step]
```

Ces points sont destinés à être évalués par `run_HF` dans `print_visu` (AC_pure_flexion.py) pour valider la cohérence HF/ana sur la frontière.

### 10.5 Bloc validation `run_HF` sur `grid_hf`

Bloc fourni dans la discussion (non écrit dans AC_pure_flexion.py, à intégrer dans `print_visu`) :

```python
g_HF_vals  = np.array([float(run_HF(pt)[0])       for pt in grid_hf])
g_ana_vals = np.array([float(calc.g_ana(list(pt))) for pt in grid_hf])
err_abs    = np.abs(g_HF_vals - g_ana_vals)

print(f"g_HF_vals  = {g_HF_vals.tolist()}")
print(f"g_ana_vals = {g_ana_vals.tolist()}")
print(f"  {'pt':>3}  {'u1':>8}  {'u2':>8}  {'g_ana':>10}  {'g_HF':>10}  {'err_abs':>10}")
for i, pt in enumerate(grid_hf):
    print(f"  {i:3d}  {pt[0]:8.3f}  {pt[1]:8.3f}  "
          f"{g_ana_vals[i]:+10.4f}  {g_HF_vals[i]:+10.4f}  {err_abs[i]:10.4f}")
print(f"  → err_abs_moy = {err_abs.mean():.4f}")
```

`run_HF(pt)[0]` extrait `g_HF` scalaire du tuple `(g_HF, grad_HF_U, grad_HF_X)`. Le `float()` garantit la conversion depuis JSON.

### 10.6 Bloc visuel HF dans `print_visu` — remplacement du contour 3×3

Le bloc contour original (grille 3×3 interpolée) est remplacé par un contour restreint à la zone frontière, avec les limites déduites de `grid_hf` :

```python
if g_hf is not None and grid_hf is not None and len(grid_hf):
    pt_start = frontier_pts[0]
    pt_end   = frontier_pts[-1]
    u1_hf = np.linspace(pt_start[0], pt_end[0], n_grid_hf)
    u2_hf = np.linspace(pt_start[1], pt_end[1], n_grid_hf)
    U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
    grid_mesh = np.column_stack([U1_hf.ravel(), U2_hf.ravel()])
    Z_true = np.array([float(run_HF(pt)[0]) for pt in grid_mesh]).reshape(n_grid_hf, n_grid_hf)
    ax.contour(U1_hf, U2_hf, Z_true, levels=[0], colors='red', linewidths=2, linestyles='--')
```

`frontier_pts` et `grid_hf` doivent être passés en paramètres à `print_visu`.

### 10.7 Nouvelles fonctions `sample_frontier1` et `sample_frontier2`

Deux fonctions ajoutées dans `2026_3004_calcul_de_pf_to_copy.py` après `grid_hf`, avec helpers `_u2_low()` et `_resample_arc()` :

**`sample_frontier1`** — branche verticale (varie u2 décroissant, cherche u1) :
- Fenêtre glissante `[u1_prev ± u1_tol]` pour rester sur la bonne branche
- Arrêt automatique quand aucun croisement dans la fenêtre
- Arc-length resampling → exactement `n_out=10` points

**`sample_frontier2`** — branche horizontale (varie u1 décroissant droite→gauche, cherche u2 bas→haut) :
- Prend le premier croisement u2 (le plus bas = branche horizontale)
- Arrêt automatique quand aucun croisement
- Arc-length resampling → exactement `n_out=10` points

**Résultats validés (run du 4 mai 2026) :**
```
sample_frontier1 : 10 pts  u1=[-0.10, 14.64]  u2=[1.21, 32.00]   ← branche verticale ✓
sample_frontier2 : 10 pts  u1=[0.16,  32.00]  u2=[0.53,  4.66]   ← branche horizontale ✓
```

Visualisation : triangles rouges (f1) sur la branche verticale, triangles violets (f2) sur la branche horizontale, bien séparés et couvrant chaque branche. Plot de validation sauvé dans `check_g_ana.png`.
