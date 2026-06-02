# GEK / GEKPLS — Bibliothèques Python open source

Recherche effectuée le 20 avril 2026.

## Résultat : SMT est essentiellement unique

| Bibliothèque | GEK / dérivées | Notes |
|---|---|---|
| **SMT** | ✓ GEKPLS | Seule option clé-en-main en Python |
| **GPy** (Sheffield) | ✗ | 0 résultat pour "derivative" dans le code source |
| **GPyTorch / BoTorch** | ✗ | `HigherOrderGP` = outputs tensoriels, pas observations de dérivées |
| **scikit-learn** | ✗ | Feature request explicitement rejetée (issue #11481, "Not planned", juillet 2022) |
| **egobox** | ✗ | Gradient-free par design, Kriging + PLS mais pas GEK |
| **mogp-emulator** | ✗ | GP standard uniquement |
| **OpenTURNS** | ✗ | `KrigingAlgorithm` standard, pas d'injection de dérivées dans la matrice |

## Conséquence

Le **full GEK** (matrice de covariance augmentée avec blocs dérivées) **n'existe dans aucune bibliothèque Python open source**. Il doit être codé manuellement — ce qui justifie la direction prise dans le projet (full GEK avec matrice augmentée, gradients adjoints STRAINS).

## Contexte : pourquoi SMT/GEKPLS n'est pas le bon outil ici

- GEKPLS est conçu pour nx grand (50-100 variables, CFD aérodynamique) — PLS inutile pour nx=2
- GEKPLS restreint le noyau à `squar_exp` (gaussien) — Matérn 5/2 non supporté
- Le noyau gaussien suppose une régularité C∞ — non justifiée pour réponses élasto-plastiques (STRAINS)
- Full GEK avec Matérn + gradients adjoints est plus adapté au problème
