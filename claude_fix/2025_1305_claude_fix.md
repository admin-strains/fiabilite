# Fix MAJ Digital Structure — 13/05/2026

## Contexte

Suite a la mise a jour de Digital Structure du 12/05/2026, `python launcher.py` produisait :
```
ImportError: DLL load failed while importing _common: La procedure specifiee est introuvable.
```
OpenTURNS 1.26 ne chargeait plus depuis le launcher STRAINS.

---

## Architecture des trois repos

| Repo | Chemin | Role |
|------|--------|------|
| workspace | `C:\workspace\` | Racine, CMakeLists.txt top-level (compile front + back ensemble) |
| front | `C:\workspace\front\` | STRAINS (CAD, mesh, solveur, bindings Python) + gestion env via pixi |
| back | `C:\workspace\back\` | Bibliotheque C++ scientifique (rupt, qant) — depend de front/.pixi/ pour Eigen/SuiteSparse |

**Python systeme :** `C:\python3\` (Python 3.10, PAS conda). OpenTURNS installe dans ce Python via pip.

---

## Ce qui a change dans la MAJ du 12/05

Fichier `C:\workspace\front\pixi.lock` mis a jour a 20h57 (commit "env divergence pre-merge").

Changements cles :
- **MKL Windows :** rebuild 11 → rebuild 13 (`mkl-2025.3.1 hac47afa_11` → `hac47afa_13`)
- **vc14_runtime :** `>= 14.29.30139` → `>= 14.44.35208` (version runtime Visual Studio plus recente)
- **Eigen :** 3.4.0 → 5.0.1 (mise a jour majeure)
- **onemkl-license :** nouvelle dependance 2025.3.1

Ces nouvelles DLLs MKL (rebuild 13) sont deployees automatiquement lors du build dans :
```
C:\workspace\front\STRAINS\rupt\core\bin\   ← MKL, OCCT, HDF5, Boost
C:\workspace\front\STRAINS\rupt\core\bin\meshgems\   ← isole
C:\workspace\front\STRAINS\rupt\core\bin\mosek\      ← isole
```

---

## Cause du conflit OpenTURNS

`launcher.py` fait dans cet ordre :
1. `os.add_dll_directory(r'C:\workspace\front\STRAINS\rupt\core\bin')` — injecte les nouvelles DLLs MKL rebuild 13 dans le chemin de recherche Windows
2. `exec(AC_pure_flexion.py)` → `import openturns` → charge `_common.pyd` → cherche ses dependances MKL → trouve la version rebuild 13 incompatible avec OT 1.26 → crash

**OT importe bien seul** (`python -c "import openturns"` reussit) mais echoue via launcher car les DLLs STRAINS polluent le chemin de recherche avant que OT soit charge.

---

## Fixes appliques

### Fix 1 : Mise a jour OpenTURNS 1.26 → 1.27
```
pip install --upgrade openturns
```
OT 1.27 est desormais installe dans `C:\python3\lib\site-packages\` (version compatible avec MKL 2025.3.1).

### Fix 2 : Modification launcher.py
Ajout de `import openturns as ot` **avant** les `os.add_dll_directory()` :

```python
import sys; print("LAUNCHER START", flush=True); sys.stdout.flush()
import os, sys

# Import openturns BEFORE adding STRAINS DLL dirs pour eviter le conflit MKL
import openturns as ot

# Setup DLL search paths BEFORE any STRAINS import
dll_dirs = [
    r'C:\workspace\front\STRAINS\rupt\core\bin',
    ...
]
```

**Pourquoi ca marche :** en important OT avant d'ajouter les DLL dirs STRAINS, les DLLs OT sont chargees par Windows en premier. Quand STRAINS ajoute ensuite ses dirs MKL, les DLLs OT sont deja en memoire — Windows ne les recharge pas.

---

## Note sur libblas.dll (erreur d'acces)

Lors de `pip install --upgrade openturns`, erreur `[WinError 5] Acces refuse` sur `libblas.dll` dans `C:\python3\`. Ce fichier etait verrouille (en cours d'utilisation). La reinstallation a quand meme abouti (OT 1.27 installe correctement). Si ce probleme se reproduit : fermer tous les processus Python avant de relancer pip, ou utiliser `pip install --user`.

---

## Probleme DLL STRAINS isole (CLAUDE.md front)

Le `CLAUDE.md` de front documente un conflit DLL existant (independant de la MAJ) entre Meshgems, Mosek et les DLLs conda/MKL (`libiomp5md.dll`, `svml_dispmd.dll`, etc.). La strategie d'isolation par sous-dossiers (`bin/meshgems/`, `bin/mosek/`) est en place et geree automatiquement par le build system.

---

## Etat apres fix

- OT 1.27 importe sans erreur depuis le launcher
- Cache `hf_3d_grid_fixed` lu correctement, print 3D se declenche
- Erreur residuelle independante : `TypeError: unsupported operand type(s) for /: 'float' and 'tuple'` ligne 549 de `AC_pure_flexion.py` dans `flexion_claude.__init__` — bug dans le code utilisateur (virgule au lieu de point dans un litteral numerique), pas lie a la MAJ DS.
