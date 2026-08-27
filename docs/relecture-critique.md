# Relecture critique : partition et methode

> Demande d'Agnes, 26/08/2026 : partitionner le code pour une relecture
> approfondie et critique, phase par phase.
>
> Puis, en cours de partition : « il y a toujours un gros fichier AC ? ce n'est
> pas sain, AC doit quasiment etre reduit a 0 a mon sens. » La mesure lui donne
> raison, et cela change la nature du travail -- voir la phase 0.

## Pourquoi ce n'est PAS un decoupage par repertoire

Un decoupage par dossier suit l'arborescence, pas les defauts. Ce qui doit
gouverner la partition, c'est **la nature de la question de correction**, parce
qu'elle decide de la TECHNIQUE -- et qu'une relecture menee avec la mauvaise
technique ne trouve rien :

* chercher un defaut numerique en lisant le flot d'execution ne donne rien ;
* chercher un defaut d'etat en comparant a une solution analytique non plus.

## Ce que la journee du 26/08 a demontre

Six defauts trouves. Leur repartition dit ou chercher, et comment :

| trouve par | nombre | ou | se signalait |
|---|---|---|---|
| plantage du run | 2 | orchestration | oui |
| balayage systematique | 4 | orchestration, caches | **non** |
| campagne phases 6-7 | 2 | noyau numerique | **non** |

**Les defauts qui se signalent sont les moins graves.** Le plus couteux --
`restart_state.json` relu sans verification, 90 h d'enrichissement en jeu --
n'aurait jamais plante : il aurait produit un resultat.

Regle qui en decoule : **classer par capacite a etre faux EN SILENCE**, pas par
frequence de panne.

## Le terrain, mesure

```
_lib/                    5 203 l.   noyau numerique
scripts AC (x2)          5 974 l.   <- voir phase 0
_config/schema.py          679 l.
solver/                    825 l.
_reliability/              511 l.
_cache/                    390 l.
_model/lois.py             214 l.
launcher.py                231 l.
------------------------------------
total a relire          ~14 000 l.   contre 5 498 l. de tests (460 verts)
```

Deux fichiers seulement exigent une licence : `solver/digital_structure.py` et
`launcher.py`. **Tout le reste est verifiable sur un poste ordinaire.**

---

## Phase 0 -- VIDER LES SCRIPTS AC (prealable, pas une phase de relecture)

### La mesure

Les deux AC portent 58 fonctions communes, imbriquees dans `__main__` :

```
                                 flexion    Moulin Blanc
fichier                          2 976 l.      2 998 l.
  en-tete + configuration            62 l.         60 l.
  fonctions imbriquees            1 975 l.      2 075 l.   (58 / 60)
  reste (PARAM_CONFIG, orchestration) 939 l.       863 l.

sur les 1 975 lignes de fonctions communes :
  IDENTIQUES d'un fichier a l'autre   1 861 l.   94,2 %
  qui different vraiment                114 l.    5,8 %
```

**45 des 58 fonctions sont byte-identiques.** Les 13 restantes different de 1 a
31 lignes sur 50 a 335 : `build_DOE` fait 114 lignes pour **2 lignes** d'ecart,
`_hf_from_custom_points` 102 lignes pour 2 lignes d'ecart.

Ce n'est pas deux etudes. C'est **une implementation copiee deux fois**, plus
une centaine de lignes de vraie difference.

### Pourquoi c'est un prealable et non une phase

Trois raisons, et la troisieme est deja constatee :

1. **On ne relit pas une fermeture.** Une fonction definie dans `__main__` se
   ferme sur son espace de noms : elle n'est ni importable, ni testable, ni
   isolable. C'est le constat qui a fonde la phase 3, et il n'a pas change.
2. **Chaque defaut doit etre corrige deux fois, chaque test ecrit deux fois.**
   C'est ce qu'a coute chaque correction du 26/08 : six defauts, douze
   corrections.
3. **LES DEUX COPIES ONT DEJA DIVERGE.** Ce n'est pas un risque theorique :
   * `run_HF` et `run_one_SOL` ne maillaient pas pareil (phase 5) -- deux
     tailles de maille differentes alimentaient le MEME metamodele ;
   * les deux `InitSolver.py` ne demandaient pas le meme solveur lineaire
     (MUMPS contre CuDss), sans que rien ne le dise ;
   * `print_visu_EFF` et `print_visu_sigma` n'existent que dans un des deux.

### Ou va quoi

```
_reliability/eff.py         (existe)    5 fonctions    401 l.
_reliability/controle.py    (existe)    1 classe       128 l.
_reliability/arret.py       (existe)    1 classe       155 l.
_reliability/form.py        (existe)    7 fonctions     79 l.
_reliability/graphiques.py  (existe)   16 fonctions    780 l.
_cache/                     (existe)   10 fonctions     96 l.
_etapes/figurer.py          (existe)  1 classe + 8 fn  626 l.
_surrogate/wrappers.py      (existe)    6 classes      193 l.
_surrogate/ajuster.py       (existe)    9 fonctions    270 l.
_surrogate/projection.py    (existe)    1 fonction      37 l.
_doe/evaluation.py          (existe)    1 classe       116 l.
_doe/plan.py                (existe)    5 fonctions    137 l.
_doe/parallele.py           (existe)    5 fonctions    140 l.
_etapes/grille.py           (existe)    1 classe       281 l.
---------------------------------------------------------------
RESTE dans l'AC                         7 fonctions    108 l.
```

Ce qui reste legitimement dans un AC : `dist_jointe`, `_solveur`,
`_grad_vers_U`, `_etiquette_socp`, `_is_position_var`,
`_find_position_var_index` -- plus `PARAM_CONFIG`. (`_tracer_domaine_physique`
est partie dans `_etapes/figurer.py` le 27/08 : elle etait identique au
caractere pres dans les deux etudes.)
Autrement dit **le probleme pose**, et rien d'autre.

**Cible : un AC de 150 a 250 lignes**, contre 3 000.

### Le filet existe deja

`tools/extraction_temoin.py` recupere une fonction ENCORE IMBRIQUEE dans un AC,
sans executer le script, et la rend appelable. Le code de production sert
d'oracle a sa propre refonte. Son docstring dit l'intention d'origine :

> « Ce filet est TEMPORAIRE par construction : quand un script AC aura ete
> entierement vide de sa logique, il n'aura plus d'original a offrir. »

Le travail s'est arrete a mi-chemin. La phase 0 le termine.

### Avancement

| mesure | 26/08 matin | 27/08 | cible |
|---|---:|---:|---:|
| `AC3_pure_flexion.py` | 2 976 l. | **1 686 l.** | <= 250 |
| `AC3_moulinblanc.py` | 2 998 l. | **1 566 l.** | <= 250 |
| fonctions imbriquees | 58 / 60 | **42 / 42** | <= 8 |
| lignes de fonctions communes aux deux etudes | 1 975 | **803** | 0 |
| divergences declarees | 6 (trouvees a la main) | **2** (verifiees) | 0 |
| fonctions `print_*` pouvant appeler le solveur | 7 / 4 | **0 / 0** | 0 |
| machinerie importee par l'AC | 9 | 4 | 0 |

Le chiffre qui compte est le quatrieme, et il est desormais **nul et
verrouille** (`test_93`, plafond 0, plus de `xfail`). Trois soudures ont ete
defaites le 27/08 :

* `init_g_ot` portait **sept fois** la meme ligne cachee
  `if xt is None: xt, yt, all_grad = build_DOE()`, une par branche de
  surrogate. Ajuster un metamodele pouvait donc lancer le plan d'experiences
  entier -- c'est par la qu'une FIGURE (`print_globalplanche_EFF`) atteignait
  `run_DOE_parallel`. La ligne est remontee chez l'appelant, en clair, a
  l'endroit unique ou le plan doit reellement etre construit. Au passage, la
  branche `do_HF` recevait un TRIPLET dans une variable unique (`xt =
  build_DOE()`) : `xt` devenait un tuple, silencieusement.
* `print_results` imprimait FORM (gratuit) **puis** evaluait l'etat limite
  exact en deux points pour l'erreur FOSM. Scindee en `resume_FORM` (0 appel)
  et `erreur_FOSM` (2 appels par mode, desormais sous `CFG.erreur_fosm`).
* `print_3D_HF` calculait la grille haute fidelite -- `n_grid_hf^2` appels,
  soit 29 h sur le Moulin Blanc pour une grille 15x15 -- avant de la dessiner.
  Scindee en `grille_3D` (l'action) et `print_3D_HF(U1, U2, Z)` (le dessin).

`_etapes/figurer.py` est ne de ce mouvement : il recoit ce qui a deja ete
calcule et ne va jamais le chercher. `tests/test_93` construit son graphe
d'appel et verifie la fermeture transitive -- une seule arete vers le solveur,
meme a travers trois intermediaires, fait echouer la suite.

`_surrogate/wrappers.py` a suivi : les six enveloppes OpenTURNS des
metamodeles -- `HFFunction`, `PCKRGFunction`, `oldGEPCKFunction`,
`GEPCKFunction`, `PCKFunction`, `GEKPLSFunction` -- etaient definies dans le
`__main__` des DEUX scripts, 193 lignes chacune, **identiques au caractere
pres**, et n'avaient jamais ete couvertes par un test. Elles ne dependaient
pourtant pas de l'etude : leurs seules variables libres etaient `n_var`,
l'evaluateur haute fidelite et un interrupteur de trace. L'extraction a
revele deux defauts :

* `_exec_sigma` -- 25 lignes de variance posterieure a noyau augmente --
  etait recopiee dans `oldGEPCKFunction` **et** dans `GEKPLSFunction`, dans le
  meme fichier : quatre exemplaires au total. Une correction sur l'une
  n'aurait pas touche les trois autres. Elle est desormais `sigma_gek`, une
  fonction libre, ecrite une fois.
* quatre lignes de trace codaient `n_var == 2` en dur (`u[0]`, `u[1]`) : avec
  trois variables le journal tronquait, avec une seule il levait `IndexError`
  au milieu d'un run.

Les cinq ajustements de metamodeles ont suivi dans `_surrogate/ajuster.py`,
et l'extraction a trouve **une cinquieme divergence entre les deux copies** :
`build_metamodel_KRG` bornait l'optimisation des longueurs de correlation a
`[1, 100]` sur la flexion pure et a `[0, 100]` sur le Moulin Blanc. Une borne
inferieure nulle laisse l'optimiseur degenerer vers une correlation quasi
nulle -- le krigeage cesse de lisser et interpole le bruit. Rien ne signalait
l'ecart. Chaque etude garde sa valeur ; elle est desormais ecrite a l'appel.

Deux defauts plus discrets au passage : `build_metamodel_GEK` portait un `if`
dont les deux branches construisaient le meme objet au caractere pres, et
`calculate_PCE` recevait `y_hf` et `all_grad_hf` sans jamais les lire -- ce
qui laissait croire que la composante PCE etait comparee aux gradients
exacts.

Le renommage `fit.py` -> `ajuster.py` n'est pas cosmetique : `_lib/fit.py`
(le clone UQLab) est sur le meme chemin d'import et se faisait eclipser. Seul
l'echec de collecte de la suite l'a signale.

Enfin `init_g_ot` elle-meme -- 125 lignes de dispatch -- est devenue
`construire_surrogate`. Trois choses que la relecture a rendues visibles :

* les branches **GEPCK et PCK etaient deux fois la meme**, a trois details
  pres (la fonction d'ajustement, la forme de `NumberOfPoly` -- un entier
  d'un cote, un tableau de l'autre -- et l'etiquette imprimee) : 68 lignes
  pour trois differences ;
* `g_ot` et `sigma_func` etaient des **parametres d'entree que toutes les
  branches ecrasaient** : jamais lus ;
* `xt` ressortait toujours identique a ce qui entrait, et les branches PCKRG
  et old_GEPCK faisaient `y_hf = yt` puis `yt = y_hf` -- un aller-retour sans
  effet. Le contrat reel est `(metamodele, ecart-type, diagnostic)`.

Le detecteur d'imports morts de `test_91` a ete corrige au passage : il ne
regardait que la PREMIERE ligne d'un import, si bien qu'un
`from x import (a,` / ` b, c)` laissait `b` et `c` **se valider eux-memes** ;
et il comptait une mention en commentaire comme un usage. Corrige, il a
trouve **six imports morts de plus** -- dont trois lois qui ne survivaient
que par des lignes de `PARAM_CONFIG` mises en commentaire.

### L'instrument qui manquait : declarer les divergences

Les six derives entre les deux copies ont toutes ete trouvees A LA MAIN, en
lisant. Aucune n'aurait ete signalee par un test -- il n'y en avait pas.

`tests/test_97_divergences_declarees.py` compare desormais la STRUCTURE des
50 fonctions communes -- l'arbre syntaxique, donc sans commentaires, sans
docstrings, sans mise en page -- et exige que toute difference figure dans un
inventaire ecrit, avec sa raison. Deux gardes, pas un : une divergence non
declaree fait echouer la suite, et une divergence declaree QUI A DISPARU la
fait echouer aussi, pour que l'inventaire ne pourrisse pas en liste de
vieilles excuses.

Etat au 27/08/2026 : **six divergences declarees**, toutes de vraies
differences d'etude (la flexion pure superpose sa solution analytique de
reference, que le Moulin Blanc n'a pas ; `run_EFF` journalise davantage d'un
cote ; le fond haute fidelite n'est pas cadre pareil). Les deux derives sur
le DOMAINE DE TIRAGE ont ete corrigees au passage, gratuitement :
`build_DOE` et `build_starting_points` tiraient sur `+/- 7,5` CODE EN DUR
dans la flexion pure et sur `eff_bounds` dans le Moulin Blanc -- borner le
domaine n'avait donc d'effet que d'un cote. Les deux bornes coincidaient
encore, donc rien ne bouge numeriquement ; au prochain bornage, si.

Quatre fonctions ne different plus que par un COMMENTAIRE :
`_hf_from_custom_points` (102 l.), `_find_batch_EFF_points` (59 l.),
`run_DOE_parallel` (58 l.), `run_HF_grid_parallel` (57 l.). 276 lignes de
copie pure.

### Ordre d'extraction (du moins risque au plus risque)

1. **graphiques** (780 l.) -- ne changent aucun resultat ; l'echec est visible.
2. **caches** (96 l.) -- deja largement extraits, il reste les delegues.
3. **DOE** (332 l.) -- `build_DOE`, `run_one_SOL`, `run_HF`, les workers.
4. **surrogate** (279 l.) -- `init_g_ot`, `build_metamodel_*`, `calculate_PCE`.
5. **EFF** (401 l.) -- `run_EFF` fait 335 lignes a elle seule : la plus grosse
   fonction du depot, et celle qui porte les quatre criteres d'arret.

A chaque etape : temoin d'extraction, puis la chaine complete sur l'etat limite
**analytique** (bit-reproductible, sans licence) comme controle de
non-regression.

**Deux etudes analytiques, et pourquoi il en faut deux.**
`studies/pure_flexion_analytique.toml` exerce le plan, le metamodele,
l'enrichissement, FORM et le tirage d'importance -- mais elle tourne avec
`print_HF = false`, donc **pas la grille haute fidelite**. Ce trou a coute
cher : le doublon de grille du 27/08/2026 (29 h sur le Moulin Blanc) a du
etre mesure a la main, en retirant la garde et en relancant, parce
qu'aucune etude rejouable n'exercait ce chemin -- et la suite de tests
etait verte.

`studies/pure_flexion_grille.toml` le couvre desormais : meme chaine,
grille activee a 7x7, six secondes, aucune licence. `test_103` la lance et
compte ce qui se passe reellement. Ce n'est pas un test de VALEUR mais un
test de COUT : on paie ce qu'on croit payer, une fois et pas deux.

---

## Les six phases de relecture

Chacune porte une question, une technique et un oracle. Apres la phase 0, la
plupart s'appliquent a des modules importables plutot qu'a des fermetures.

### A - Ce qui definit le probleme (~900 l.)

`_config/schema.py`, `_model/lois.py`, `PARAM_CONFIG`.

**Question** : le probleme est-il bien pose ? Un parametre peut-il etre
accepte, classe, valide -- et pourtant demander au solveur quelque chose qui
n'a aucun sens physique ?

**Technique** : exhaustivite, et **analyse du domaine de valeurs** : traduire
chaque borne dans l'unite du metier avant de la juger.

**Precedent** : `eff_bounds = +/- 7,5` avait l'air d'une precaution. Traduit en
MPa, cela demandait un acier a 8,9 MPa -- et a tue le solveur. Personne n'avait
jamais fait la conversion.

**Non couvert** : le domaine de valeurs des AUTRES parametres ; la coherence
entre `PARAM_CONFIG` et les regions de sensibilite du modele.

### B - Le noyau numerique (~5 200 l.)

`_lib/`. **Le gisement de defauts silencieux.**

**Technique** : oracles independants, differences finies, invariants (symetrie,
definie-positivite, conditionnement), entrees degenerees.

**Precedent** : l'absence de pepite donnait **56,4 % d'erreur** avec 100 % de
tests verts.

**Non couvert** : `lars.py` (620 l.), `fit.py` (531 l.), `polynomials.py`
(439 l.) -- aucun n'a d'oracle analytique dedie.

### C - Les algorithmes de fiabilite (~1 200 l.)

`_reliability/` **apres extraction**, dont `run_EFF` et ses quatre criteres
d'arret imbriques dans 335 lignes. Un critere qui ne se declenche jamais, ou
trop tot, ne se verrait pas.

**Oracle** : `tests/reference/form.py`, beta = 4,77257.

### D - L'orchestration et l'etat (~1 400 l.)

**Deja couvert** : les huit points de reprise portent une signature comparee
(`POINTS_DE_REPRISE` dans `test_89`).

**Non couvert** : les **chemins d'echec**. Que se passe-t-il si un worker
meurt ? si le disque sature pendant une ecriture ? `run_DOE_parallel` n'a
JAMAIS tourne.

### E - Le couplage au solveur (~1 050 l.)

**Non couvert, question ouverte** : `NUMERICAL_ERROR` sur un point BIEN
conditionne (fy1/fy2 = 298/371, rapport 1,24) reste inexplique. Il y a au moins
deux modes de defaillance du solveur, un seul est compris.

### F - Les sorties (~780 l. apres extraction)

**Risque** : faible en correction, eleve en **consequence**. Une figure n'a pas
de garde-fou, et c'est elle qui part dans une note de calcul. Le cache HF
empoisonne du 26/08 se serait manifeste ici, et nulle part ailleurs.

---

## Ordre recommande

**0 -> A -> B -> C -> D -> E -> F.**

La phase 0 d'abord, parce qu'elle divise par deux le code a relire et rend
importable ce qui ne l'est pas. Puis A, qui conditionne tout et ne fait que
900 lignes : juger la justesse d'un calcul mene sur un domaine qui n'a pas de
sens physique est du temps perdu.

## Ce qu'une phase doit produire

1. un **inventaire exhaustif** de ce qui est examine, etabli par balayage et
   non au fil de la lecture ;
2. les **defauts trouves**, avec le mecanisme exact, le fichier et la ligne, et
   ce qui les rend silencieux ou bruyants ;
3. un **test qui echoue AVANT le correctif** -- un garde-fou qui ne casse pas
   sur le code d'avant ne prouve rien ;
4. ce qui a ete **cherche et non trouve**, qui borne ce que la relecture
   garantit.
