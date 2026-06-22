# `_lib/` — bibliothèque GEPCK (modules `branche*`)

Modules du métamodèle **GEPCK** (PCE + Kriging) utilisés par les scripts de fiabilité (`AC_*.py`).
Déplacés ici depuis la racine le 2026-06-22 pour ranger le dépôt.

## Modules et dépendances
- **branche1.py** — API haut niveau : `fit_gepck`, `predict_gepck`, `predict_gradient_gepck`.
  → importe branche2, branche3, branche4, branche5.
- **branche2.py** — `uq_PCK_initialize` (init).
- **branche3.py** — `uq_PCK_calculate_coefficients`, `uq_GEPCK_calculate_coefficients` → importe branche5, branche_lars.
- **branche4.py** — `uq_PCK_eval`, `uq_GEPCK_eval`, `uq_GEPCK_eval_deriv` → importe branche3, branche5.
- **branche5.py** — noyaux : `uq_eval_Kernel`, `uq_eval_global_Kernel`, …
- **branche_lars.py** — `uq_lar` (LARS).

## Comment les scripts l'utilisent
Les importeurs (AC_*.py, _tools/regen/*) ajoutent ce dossier au `sys.path` **avant** d'importer :
```python
import sys; sys.path.insert(0, r"C:\workspace\fiabilite\_lib")
from branche1 import fit_gepck, predict_gepck, predict_gradient_gepck
```
Les imports internes entre `branche*` (`from branche2 import …`) fonctionnent automatiquement
une fois `_lib` sur le `sys.path`. Les modules `branche*` n'ont PAS été modifiés.
