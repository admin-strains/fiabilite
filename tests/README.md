# Harness de non-regression -- branche `cleaning`

Filet de securite pose avant la remise a niveau du code de fiabilite. Son
role est simple : **rendre visible, a chaque iteration, tout changement de
comportement du coeur de calcul**, pour qu'un refactoring soit une decision
et non un pari.

## Lancer

```bat
run_tests.bat
```

Ne demande **ni STRAINS, ni OpenTURNS, ni l'environnement conda de
production** : seulement `numpy`, `scipy`, `pytest`. Environ 50 s.

```bat
run_tests.bat -m "not slow"      :: ~25 s, sans les suites historiques
run_tests.bat tests/test_40_chain_beta.py -v
set FIAB_PYTHON=C:\autre\python.exe & run_tests.bat
```

## Ce que le harness couvre -- et ce qu'il ne couvre pas

Couvert : `_lib/` en entier, c'est-a-dire le coeur mathematique (PCK, GEPCK,
noyaux, LARS, krigeage), qui est **identique sur les quatre branches** du
depot. Une modification de `_lib` se repercute partout : c'est la ou un
filet rapporte le plus.

Non couvert : les scripts `AC*.py` (appels STRAINS, maillage, SOCP, EFF, IS
parallele, figures). Ils exigent un poste de production. Le harness verifie
en revanche que **le mode d'appel du metamodele** reproduit exactement celui
de `AC3_pure_flexion.init_g_ot` (`tests/harness.py`) : si les deux divergent,
le harness devient un mensonge -- les maintenir ensemble.

## Les cinq etages

| Fichier | Question posee | Nature |
|---|---|---|
| `test_00_api.py` | les symboles importes par les scripts AC existent-ils toujours, avec la meme signature ? | contrat |
| `test_10_legacy_unit.py` | les 10 suites unitaires historiques passent-elles toujours ? | heritage |
| `test_20_reference.py` | les oracles du harness sont-ils eux-memes justes ? | auto-controle |
| `test_30_surrogate_golden.py` | les coefficients ont-ils bouge d'un iota ? | non-regression figee |
| `test_40_chain_beta.py` | le beta final est-il toujours JUSTE ? | verite metier |
| `test_50_known_defects.py` | les defauts connus sont-ils toujours la ? | dette tracee |

La distinction `test_30` / `test_40` est le point important :

- `test_30` compare a des **chiffres figes** (`tests/golden/`). Il tombe des
  qu'un resultat change, meme pour le mieux. C'est voulu : un refactoring
  cense etre neutre qui deplace la 12e decimale doit etre discute.
- `test_40` compare a une **verite analytique** independante du code. Il ne
  tombe que si le resultat devient FAUX. C'est lui qui autorise a mettre a
  jour un golden en confiance.

## Les oracles (`tests/reference/`)

N'importent rien de `_lib/`. Deux etats limites en espace standard :

- **`LinearLS`** -- `g(u) = beta_ref - a.u/||a||`, hyperplan : `beta` exact
  par construction. Invariant le plus dur : tout metamodele contenant le
  degre 1 doit le retrouver a la precision machine.
- **`FlexionLS`** -- section rectangulaire BA, aciers plastifies, reprise
  ligne a ligne de `flexion_claude` (`pure_flexion/AC3_pure_flexion.py`
  l.1191-1230) : `fc` lognormale, `fy` normale, `M_R = A.fy + B.fy^2/fc`.
  Valeurs : `beta = 3.4722741272`, `u* = (-0.243905, -3.463697)`.

`beta` y est obtenu par **deux chemins independants** qui se recoupent a
4e-15 : minimisation scalaire sur la courbe `g=0` parametree, et HL-RF
(`reference/form.py`). Un oracle isole serait invérifiable ; deux qui
concordent le sont.

## Etat mesure sur `cleaning` (25/08/2026)

| | PCK | GEPCK |
|---|---|---|
| LOO (flexion, DOE 24) | 1.390e-09 | 1.587e-09 |
| polynomes retenus | 9 | 7 |
| ecart sur `beta` | **0.011 %** | **1.30 %** |
| interpolation au DOE | 5.3e-07 | **3.0e-03** |

GEPCK dispose de **plus** d'information que PCK (valeurs + gradients) et
rend un resultat **cent fois moins precis**. Les deux lignes en gras sont
tracees dans `test_50_known_defects.py`. A traiter dans le plan de
nettoyage, pas en passant.

## Regenerer un golden

```bat
set PYTHONPATH= & C:\python3\python.exe tests\make_golden.py
```

Jamais en reflexe. La procedure est ecrite en tete de `make_golden.py` :
prouver d'abord par `test_20` et `test_40` que le nouveau comportement est
meilleur, puis regenerer, puis commiter le golden **avec** la modification
et justifier l'ecart dans le message de commit. Ecraser un golden sans cela
efface la memoire de ce que le code faisait avant.

## Ajouter un defaut connu

Dans `test_50_known_defects.py`, ecrire le test du comportement **attendu**
(pas du comportement actuel) et le marquer `xfail(strict=True)`. La suite
reste verte, et le jour ou le defaut est corrige le test devient `xpass`,
donc **rouge** : impossible de corriger en silence, impossible de regresser
sans bruit. Exiger un symptome reproductible, une localisation
`fichier:ligne`, et l'effet concret en production.

## Suites historiques (`tests/unit/`)

Dix fichiers restaures **sans modification** depuis la branche `fiabilite`,
ou ils avaient ete supprimes lors des reorganisations vers `flexion`,
`moulin_blanc` et `dir-fiabilite`. Ils totalisent ~270 assertions.

Ils sont ecrits en style script (compteur `PASS`/`FAIL`, aucun `assert`,
aucun code retour) : lances tels quels, ils affichent leurs echecs et
sortent en succes. `test_10_legacy_unit.py` les execute en sous-processus et
lit leur bilan, ce qui en refait une barriere sans toucher a leur contenu.

Deux sont en echec sur `cleaning` alors qu'ils passaient sur `fiabilite` --
`test_eval_global_kernel` et `test_predict_deriv_gepck` : voir le tableau
`STATUT_CONNU` en tete du fichier.
