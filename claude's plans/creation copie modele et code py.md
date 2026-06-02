# Plan : Création du doublon _exportRebar2 / EDF2

## Contexte

Les dossiers ont déjà été copiés manuellement :
- `C:\_workingDir\_SF\Autres modeles\_exportRebar2` (copie de `_exportRebar`)
- `C:\workspace\storage\admin\SF\Autres modeles\EDF2` (copie de `EDF`)

Les deux copies sont des doublons bit à bit de leurs sources. Tous les chemins hardcodés pointent encore vers les anciens emplacements (`_exportRebar` et `EDF`). L'objectif est de corriger exactement 12 chemins dans 5 fichiers pour que le tout soit cohérent et autonome.

## Architecture des dépendances

```
launcher2_voussoir.py
  └─ sys.path.insert → _exportRebar2/
  └─ exec(AC2_voussoir.py)
        └─ _path_ds / path → EDF2\voussoir_femelle_3.ds
        └─ exec(InitSolver2.py) → _exportRebar2/

AC.py
  └─ path → EDF2\voussoir_femelle_3.ds
  └─ exec("InitSolver.py ") ← RELATIF, OK
        └─ dsCad.txt (dans EDF2\voussoir_femelle_3.ds\)
              └─ EXTERNAL_FILE → EDF2\coffrage_v2.stp

AC_note.py
  └─ path → EDF2\voussoir_femelle_3.ds  (était resté sur EDF original)
```

---

## Modifications à effectuer

### Fichier 1 : `C:\_workingDir\_SF\Autres modeles\_exportRebar2\AC.py`

| Ligne | Ancienne valeur | Nouvelle valeur |
|-------|----------------|----------------|
| 61 | `"C:\\workspace\\storage\\admin\\SF\\Autres modeles\\EDF\\"` | `"C:\\workspace\\storage\\admin\\SF\\Autres modeles\\EDF2\\"` |

---

### Fichier 2 : `C:\_workingDir\_SF\Autres modeles\_exportRebar2\AC2_voussoir.py`

| Ligne | Ancienne valeur | Nouvelle valeur |
|-------|----------------|----------------|
| 59 | `"C:\\workspace\\storage\\admin\\SF\\Autres modeles\\EDF\\"` | `"C:\\workspace\\storage\\admin\\SF\\Autres modeles\\EDF2\\"` |
| 438 | `"C:\\workspace\\storage\\admin\\SF\\Autres modeles\\EDF\\"` | `"C:\\workspace\\storage\\admin\\SF\\Autres modeles\\EDF2\\"` |
| 486 | `r"C:\_workingDir\_SF\Autres modeles\_exportRebar\InitSolver2.py"` | `r"C:\_workingDir\_SF\Autres modeles\_exportRebar2\InitSolver2.py"` |
| 542 | `"C:\\workspace\\storage\\admin\\SF\\Autres modeles\\EDF\\"` | `"C:\\workspace\\storage\\admin\\SF\\Autres modeles\\EDF2\\"` |
| 589 | `r"C:\_workingDir\_SF\Autres modeles\_exportRebar\InitSolver2.py"` | `r"C:\_workingDir\_SF\Autres modeles\_exportRebar2\InitSolver2.py"` |
| 710 | `r'C:\workspace\storage\admin\SF\Autres modeles\EDF'` | `r'C:\workspace\storage\admin\SF\Autres modeles\EDF2'` |
| 1942 | `r'C:\_workingDir\_SF\Autres modeles\_exportRebar\notre_gepck_hf.png'` | `r'C:\_workingDir\_SF\Autres modeles\_exportRebar2\notre_gepck_hf.png'` (commenté, mis à jour par cohérence) |

---

### Fichier 3 : `C:\_workingDir\_SF\Autres modeles\_exportRebar2\launcher2_voussoir.py`

| Ligne | Ancienne valeur | Nouvelle valeur |
|-------|----------------|----------------|
| 23 | `r'C:\_workingDir\_SF\Autres modeles\_exportRebar'` | `r'C:\_workingDir\_SF\Autres modeles\_exportRebar2'` |
| 26 | `r'C:\_workingDir\_SF\Autres modeles\_exportRebar\AC2_voussoir.py'` | `r'C:\_workingDir\_SF\Autres modeles\_exportRebar2\AC2_voussoir.py'` |

---

### Fichier 4 : `C:\_workingDir\_SF\Autres modeles\_exportRebar2\AC_note.py`

| Ligne | Ancienne valeur | Nouvelle valeur |
|-------|----------------|----------------|
| 61 | `"C:\\workspace\\storage\\admin\\EDF\\"` | `"C:\\workspace\\storage\\admin\\SF\\Autres modeles\\EDF2\\"` |

> Note : Dans le doublon _exportRebar de référence, ce chemin était resté sur l'EDF original (jamais corrigé). On le corrige ici vers EDF2.

---

### Fichier 5 : `C:\workspace\storage\admin\SF\Autres modeles\EDF2\voussoir_femelle_3.ds\dsCad.txt`

| Ligne | Ancienne valeur | Nouvelle valeur |
|-------|----------------|----------------|
| 12 | `"C:\workspace\storage\\admin\SF\\Autres modeles\\EDF\coffrage_v2.stp"` | `"C:\workspace\storage\admin\SF\Autres modeles\EDF2\coffrage_v2.stp"` |

---

## Ce qu'on NE touche PAS

- `AC.py` ligne 152 : `exec(open("InitSolver.py ").read())` — chemin RELATIF, cherche InitSolver.py dans le répertoire courant, aucun changement
- `InitSolver.py` et `InitSolver2.py` — pas de chemins hardcodés, aucun changement
- Tous les autres `.ds` dans EDF2 (voussoir_femelle_0/1/2, test_3, test_export) — ne sont pas utilisés par AC.py/AC2 et leurs chemins vers EDF original ou "agnes" étaient déjà là dans la référence
- Fichiers de résultats (`.dslogloc`, `.pos`, etc.) — régénérés à chaque run
- `__pycache__/` — recompilé automatiquement

## Vérification post-modification

1. Vérifier par grep que plus aucune occurrence de `_exportRebar[^2]` ou `\\EDF\\` (sans le 2) n'existe dans les 5 fichiers modifiés de _exportRebar2
2. Vérifier par grep que `voussoir_femelle_3.ds\dsCad.txt` dans EDF2 ne contient plus de référence à `\\EDF\`
3. Lancer AC.py depuis _exportRebar2 avec `exescad=1, exesload=1, imesh=0, isolv=0` pour valider la lecture du modèle avant tout calcul
