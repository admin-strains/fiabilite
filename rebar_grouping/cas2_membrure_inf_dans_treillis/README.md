# Cas 2 — membrure inférieure dans le TREILLIS (groupe 2)

Variante du découpage des aciers en 2 groupes pour la fiabilité.

## Différence avec le cas 1
- **Cas 1** (`../cas1_membrure_inf_dans_tablier/`) : la membrure inférieure est
  dans le **groupe 1 (tablier)** car elle est géométriquement au niveau du tablier
  (capturée par la box `bounding_box_acier_tablier.stp`).
- **Cas 2** (ici) : on **retire la membrure inférieure du tablier** pour la mettre
  avec le **treillis (groupe 2)** — logique mécanique : la membrure inf + les
  diagonales forment un système connecté.

## Comment produire ce cas
1. Déposer dans ce dossier la **box modifiée** (`.stp`) = tablier SANS la zone de
   la membrure inférieure (à dessiner dans Rhino, ou box ajustée en Y).
   Les scripts détectent automatiquement le `.stp` du dossier (hors `GROUPE*`).
2. Copier ici les scripts du cas 1 (`step_solid.py`, `classify_rebars_in_box.py`,
   `view_individual.py`, `view3d_groups.py`, `export_groups_step.py`,
   `export_groups_stl.py`) — ils écrivent leurs sorties dans le dossier courant.
3. Lancer `python classify_rebars_in_box.py` → groupes + JSON + visu.

## Approche alternative (à définir)
L'utilisateur proposera une autre méthode pour identifier précisément les barres
de la membrure inférieure à basculer du groupe 1 vers le groupe 2.
