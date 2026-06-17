# Cas 2 — membrure inférieure dans le TREILLIS (groupe 2)

Variante du découpage : la **membrure inférieure longitudinale** est rattachée au
treillis (groupe 2) au lieu du tablier (groupe 1).

## Logique
- **Base** = cas 1 (`../cas1_membrure_inf_dans_tablier/rebar_groups.json`), G1=14082 / G2=1264.
- On bascule **G1 → G2** les barres dont le barycentre est dans l'un des **2 solides**
  de `boxes_membrures_inf.stp` ET qui sont **LONGITUDINALES** (polyligne ouverte).
- Les **CADRES** (étriers = boucles fermées, 1ᵉʳ point = dernier point) sont **IGNORÉS** :
  ils restent en groupe 1 (tablier). 1854 cadres dans les boîtes non basculés.

## Résultat
- **257 barres** basculées (membrure inf longitudinale) : HA_33 (65), HA_35 (8),
  HA_40 (120), HA_50 (64) — gros diamètres.
- **G1 tablier = 13825** | **G2 structure + membrure inf = 1521**.

## Fichiers
- `classify_cas2.py` — script (relancer pour reproduire)
- `boxes_membrures_inf.stp` — les 2 solides de la membrure inférieure
- `groupes_membrure_inf_dans_treillis.json` — listes + méta + barres basculées
- `rebar_groups_cas2.json` — format simple {group1, group2}
- `noms_groupe1_tablier.txt` / `noms_groupe2_structure_membrureinf.txt`
- `noms_bascules_membrure_inf.txt` — les 257 barres déplacées
- `cas2_groups.png` — visu (bascules en bleu)

## Détection des cadres
Cadre = polyligne fermée (`||p[0] - p[-1]|| < 1e-3 m`). Vérifié : HA_5/HA_7 tous
fermés (étriers), HA_8 majoritairement fermés ; les gros diamètres (HA_33/35/40/50)
sont ouverts = longitudinaux.
