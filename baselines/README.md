# Baselines — mesurer la stabilité au fil des modifications

Le chantier consiste à déplacer beaucoup de code sans changer un seul
résultat. Ces baselines sont l'appareil de mesure qui permet de le vérifier,
à chaque étape, et de répondre à la seule question qui compte pendant un
refactoring : **qu'est-ce qui a bougé, et de combien ?**

## Les deux baselines

| | `flexion_analytique` | `flexion_ds` |
|---|---|---|
| solveur | état limite analytique | Digital Structure |
| durée | **~7 s** | ~minutes |
| dépendances | numpy + scipy | Python 3.10 + DS + OpenTURNS |
| quand | **à chaque commit** (dans `pytest`) | à chaque fin de phase |

Les deux parcourent exactement les mêmes étapes et produisent le même format
de journal, donc le même comparateur les traite. La géométrie est celle du
vrai `test_pure_flexion` : b = h = 0,8 m, L = 5,0 m, 24 HA32, d = 0,691333 m.

## Utilisation

```bat
python tools\baseline_run.py                          :: rejoue et journalise
python tools\baseline_run.py --repeat 3               :: + plancher de bruit
python tools\baseline_compare.py --last flexion_analytique
python tools\baseline_compare.py reference.jsonl apres.jsonl
```

`run_tests.bat` exécute `tests/test_70_baseline.py`, qui rejoue la chaîne et
la compare à `flexion_analytique/reference.jsonl`. Aucune commande à lancer
à la main dans le cycle normal.

## Ce que trace le journal

Une ligne JSON par événement : plan d'expériences, appels au solveur,
coefficients du métamodèle (θ, β_PCE, LOO, indices retenus), grille du
critère EFF, itérations FORM, tirage d'importance, et les repères
indépendants du métamodèle (β exact, u\* exact).

Chaque grandeur est enregistrée avec **son hachage et ses statistiques**. Le
hachage répond à « identique au bit près ? », les statistiques à « de combien
ça a bougé ? ». Un hachage seul ne distingue pas une refonte qui casse tout
d'un dernier bit d'arrondi.

Les fonctions appelées des milliers de fois — les prédictions — ne sont pas
tracées une à une : elles sont résumées par un **condensat roulant** portant
la suite complète des entrées et sorties. Cela répond exactement à la bonne
question (« la séquence d'appels est-elle la même ? ») tout en évitant deux
défauts : un journal de plusieurs Mo, et un alignement fragile qui saute dès
qu'une modification change le nombre d'itérations FORM. Le journal fait 20 Ko
au lieu de 1,9 Mo.

## Le plancher de bruit

`plancher_de_bruit.json` enregistre la dispersion observée sur plusieurs
exécutions identiques. **Une baseline sans son plancher de bruit ne sert à
rien** : sans lui, un écart constaté après une modification est
ininterprétable — on ne sait pas s'il vient de la modification ou du run.

Mesuré le 25/08/2026 sur la chaîne analytique : **étendue relative
exactement nulle** sur β, Pf_FORM et Pf_IS. Tout écart non nul est donc
imputable à une modification, sans ambiguïté.

Cette reproductibilité tient à trois choses, dont deux sont fragiles :

- OpenTURNS part d'une **graine par défaut fixe**. Le déterminisme est donc
  *implicite* et sensible à l'ordre des tirages : un refactoring qui déplace
  un appel décale toute la suite. `telemetry.pin_seeds()` le rend explicite —
  `SetSeed(0)` reproduit exactement l'état par défaut, la mise en place ne
  change donc aucun résultat.
- `numpy.random` global n'est pas reproductible. Il n'est utilisé qu'à un
  endroit, `branche3.uq_Kriging_helper_create_randIdx`, et **uniquement quand
  K ≠ 1**. La configuration est en LOO (K = 1) : cet aléa est dormant, mais il
  se réveillera si quelqu'un passe en validation croisée K-fold.
- `_parallel_is` sème chaque bloc par son indice : reproductible.

## Sensibilité mesurée

L'instrumentation a été validée en injectant une perturbation relative de
**1e-12** dans l'assemblage du noyau (`branche5.uq_eval_global_Kernel`) :

| grandeur | écart relatif |
|---|---|
| perturbation injectée | 1 · 10⁻¹² |
| LOO du métamodèle | 2,6 · 10⁻³ |
| β | 4,1 · 10⁻⁴ |
| Pf | 9,7 · 10⁻³ |

Le comparateur a désigné `lib:fit_gepck / out_LOO` comme première divergence,
c'est-à-dire l'étape même où la perturbation avait été injectée, et la
restauration a rendu un journal identique au bit près.

Ce tableau dit deux choses. D'abord que l'appareil de mesure est assez fin
pour ce chantier. Ensuite, et c'est plus gênant, que **l'ajustement GEPCK
amplifie une perturbation de neuf ordres de grandeur** — c'est le
conditionnement de R̃ qui se manifeste, la même cause que les défauts 2 et 3
du plan de nettoyage, ici chiffrée.

## Regénérer une référence

Jamais en réflexe. La procédure est la même que pour les goldens :

1. prouver, par `test_20` et `test_40`, que le nouveau comportement est
   meilleur — pas seulement différent ;
2. `python tools\baseline_run.py --repeat 3` ;
3. copier le dernier `run_*.jsonl` en `reference.jsonl` ;
4. commiter la référence **avec** la modification, et justifier l'écart dans
   le message de commit, comparateur à l'appui.

Une référence écrasée sans cela efface la mémoire de ce que le code faisait
avant, et le filet ne vaut plus rien.
