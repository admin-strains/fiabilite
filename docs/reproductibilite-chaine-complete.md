# Reproductibilite de la chaine complete — mesure du 25/08/2026

> Mesure faite pour repondre a une question simple : apres huit etapes de
> restructuration, le script AC rend-il encore les memes chiffres ?
> La reponse est oui. Mais la mesure en a donne une autre, plus importante.

## Ce qui a ete fait

`AC3_pure_flexion` execute **trois fois de bout en bout** sur Digital
Structure, avec la meme configuration d'essai (`n0 = 5`, budget
d'enrichissement 8, DOE sequentiel, caches desactives) :

| run | version du code |
|---|---|
| `origine` | revision `d0ef283`, avant toute restructuration |
| `actuel` | apres les phases 2, 3a-3f et 4a |
| `actuel2` | **la meme** que `actuel`, relancee |

Le troisieme run est celui qui compte : sans lui, tout ecart entre les deux
premiers aurait ete attribue au refactoring.

## Resultat 1 — la restructuration n'a rien casse

Les grandeurs deterministes sont **identiques** entre `origine` et `actuel` :

```
mode 1 : beta = 5.1309   Pf = 1.442e-07   u* = [-2.363, -4.555]
mode 2 : beta = 5.3599   Pf = 4.164e-08   u* = [-4.904, -2.162]
```

Le plan d'experiences final contient **le meme ensemble de points**, et les
quatre premieres iterations d'enrichissement coincident a quatre decimales.

Deux points d'enrichissement sont ajoutes dans un **ordre different** :

```
origine : ... [-5.0, -1.667]  [-1.481, -5.0]  [-6.667, 3.333]  [1.111, -5.556] ...
actuel  : ... [-5.0, -1.667]  [-6.667, 3.333]  [-1.481, -5.0]  [1.111, -5.556] ...
```

Ce n'est pas une divergence de logique : le critere EFF y presente deux
maxima quasi egaux, et un ecart numerique minuscule fait basculer lequel
l'emporte. L'ensemble final etant le meme, le metamodele final est le meme,
et FORM retrouve les memes modes.

**Le meme desordre s'observe entre `actuel` et `actuel2`** — deux executions
du meme code. Il n'a donc rien a voir avec la restructuration.

## Resultat 2 — la chaine n'est pas reproductible

C'est la trouvaille reelle. Le meme point, soumis trois fois au solveur :

```
alpha = 1.332124954037
alpha = 1.332124954018
alpha = 1.332124954056
```

Meme maillage (1 873 tetraedres), memes 23 iterations, meme statut
`OPTIMAL`. L'ecart est au **onzieme chiffre significatif**. Le maillage etant
identique, la source est le solveur lineaire, pas la geometrie.

Cet ecart traverse la chaine en s'amplifiant :

| etape | grandeur | etendue relative | amplification |
|---|---|---|---|
| solveur | `alpha` | 2,9 · 10⁻¹¹ | — |
| krigeage | `theta` | 1,5 · 10⁻⁶ | × 5 · 10⁴ |
| resultat | `Pf_IS` | **1,2 · 10⁻¹** | × 4 · 10⁹ |

```
Pf_IS : 2,4770e-07   2,3278e-07   2,1892e-07
```

**12,3 % d'etendue sur trois runs**, pour un critere d'arret fixe a
COV = 5 %. La dispersion d'un run a l'autre est donc plus grande que la
precision que l'algorithme croit atteindre.

## Ce que cela implique

**Pour la lecture des resultats.** Publier `Pf = 2,4770e-07` a quatre
chiffres significatifs suggere une precision qui n'existe pas. Deux chiffres
sont deja optimistes. La barre d'erreur a annoncer n'est pas le COV du
tirage seul : c'est la dispersion d'un run a l'autre, qui l'englobe.

**Pour toute comparaison A/B.** Aucune comparaison de deux versions du code
sur cette chaine ne peut conclure en dessous de ~12 % sur `Pf_IS`, ni en
dessous de ~10⁻⁶ sur `theta`. C'est le plancher de bruit, et il doit etre
mesure avant toute conclusion — c'est exactement ce que fait
`baselines/plancher_de_bruit.json` pour la baseline analytique, ou il vaut
zero.

**Pour l'amplification elle-meme.** Un facteur 4 · 10⁹ entre l'entree et la
sortie n'est pas une fatalite : il tient au conditionnement du krigeage,
deja identifie comme defaut 2 et 3 du plan de nettoyage. Une reduction du
conditionnement reduirait mecaniquement cette dispersion.

## Ce qui reste a verifier

Le chemin `n_workers_DOE > 1` n'a pas ete exerce : les runs sont
sequentiels. Le DOE parallele lance des sous-processus, dont l'ordre
d'achevement pourrait ajouter sa propre variabilite.

La grille de visualisation haute fidelite (`print_HF`) a ete desactivee :
49 appels solveur sans influence sur le resultat, mais non couverts.

Le script `AC3_moulinblanc` n'a pas ete execute : un appel solveur y coute
des minutes, non des secondes.

## Verification de la phase 4b (26/08/2026)

Un quatrieme run, apres le debranchement de la configuration, compare a la
repetition qui donne le plancher :

```bat
python tools\comparer_runs.py sortie_actuel.txt sortie_4b.txt ^
                              --repetition sortie_actuel2.txt
```

| grandeur | ecart 4b | plancher mesure | verdict |
|---|---|---|---|
| `beta FORM` | **0** | 4,7 · 10⁻² | identique |
| `u*` | **0** | 1,8 | identique |
| `COV` | **0** | 3,4 | identique |
| `theta` | 6,7 · 10⁻⁵ | 3,3 · 10⁻¹ | dans le bruit |
| `Pf_IS` | 1,7 · 10⁻³ | 1,1 | dans le bruit |
| `LOO` | 1,3 · 10⁻⁴ | 7,3 | dans le bruit |

Resultat final identique au chiffre imprime pres :
`Pf_IS = 2,3278e-07`, `beta_IS = 5,0400`. Les quatre runs s'etalent de
2,1892e-07 a 2,4770e-07 — les 12,3 % deja mesures.

Deux enseignements pour l'outil, tous deux corriges :

1. **Le seuil doit venir d'une repetition, pas d'une constante.** Avec le
   `rtol = 1e-9` d'origine, huit grandeurs etaient annoncees « ECART » — dont
   toutes celles ou deux runs du meme code s'ecartent davantage.
   `comparer_runs.py` prend desormais `--repetition` et derive le plancher de
   la mesure.
2. **Le plancher ne peut pas descendre sous la resolution d'impression.** Au
   premier essai, `EFF initial` a ete signale pour 3,2 · 10⁻⁶ — soit
   exactement un digit du dernier chiffre ecrit (1e-6 / 0,310336) : le
   journal ecrit `0.310336` d'un cote, `0.310337` de l'autre. La repetition
   donnait par hasard un plancher nul sur cette grandeur.

## Verification de la phase 5 (26/08/2026)

Cinquieme run complet, apres l'isolement de Digital Structure derriere
`solver/`. Le tableau des grandeurs intermediaires ne conclut rien — leur
ordre change d'un run a l'autre. Ce qui conclut, ce sont les **modes FORM
finaux** :

| run | mode 1 | mode 2 |
|---|---|---|
| `origine` (`d0ef283`) | β = 5,1309  u* = [−2,363 ; −4,555] | β = 5,3599  u* = [−4,904 ; −2,162] |
| `actuel` (phases 2 a 4a) | **idem** | **idem** |
| `4b` | **idem** | **idem** |
| `5` | **idem** | **idem** |
| `actuel2` — *repetition du meme code* | β = 5,1385  u* = [−2,353 ; −4,568] | β = 5,3639  u* = [−4,932 ; −2,108] |

Huit etapes de restructuration ne changent pas un chiffre ; **relancer le code
inchange en change**. C'est l'enonce le plus fort que cette chaine permette.

`tools/comparer_runs.py` distingue desormais ce RESULTAT du flux intermediaire,
parce que les 41 valeurs de `beta FORM` d'un journal incluent toutes les
iterations d'enrichissement, dont l'ordre bascule — les comparer une a une
signale un ecart la ou rien n'a bouge.

## La chaine analytique, elle, est reproductible

La meme chaine sur `solver/analytique.py`, deux executions :

```
mode 1 : beta=4.7527  Pf=1.004e-06  u*=[-2.793, -3.846]
Pf_IS = 1.5071e-06     beta_IS = 4.6698
```

Journaux **identiques hors chronometrage** (226 lignes). Et `beta_FORM =
4,7527` contre `4,77257` calcule sans metamodele ni FORM : **0,42 %**.

> Ces chiffres sont ceux de la phase 5. La phase 6 les a deplaces a
> `4,7516 / 1,4147e-06 / 4,6828`, et le gradient analytique du 02/09/2026 les
> a ramenes a `4,7527 / 1,4903e-06 / 4,6721` -- meme `beta_FORM` que la
> phase 5 par coincidence de quatrieme chiffre, `Pf_IS` et `beta_IS`
> differents. Voir les deux sections suivantes.

C'est la sortie utile de la phase 5. Un ecart observe sur cette chaine-la est
imputable au code, ce qui n'a jamais ete vrai sur Digital Structure.

## Effet de la phase 6 sur la chaine analytique (26/08/2026)

La correction des defauts 2 et 3 -- pepite par defaut a 1e-8 -- change les
chiffres. Sur la chaine analytique complete, ou la mesure est reproductible :

| | avant | apres | exact |
|---|---|---|---|
| `beta_FORM` | 4,7527 | 4,7516 | 4,77257 |
| `beta_IS` | 4,6698 | 4,6828 | 4,77257 |
| `Pf_IS` | 1,5071e-06 | 1,4147e-06 | — |

L'effet est modeste ICI, et c'est attendu : l'enrichissement EFF place ses
points le long de l'etat limite, la ou le metamodele etait deja bon avec
treize points. Le gain de la pepite se joue sur les grands plans
d'experiences, ou le conditionnement s'effondrait -- 56 % d'erreur sur beta a
quarante points, contre 0,0015 % apres. Voir `tools/mesure_pepite.py`.

La baseline analytique, elle, passe de **0,0771 % a 0,0492 %** d'erreur sur
`beta` contre le meme oracle exact.

## Effet du gradient analytique de la vraisemblance (02/09/2026)

La correction du gradient de `theta` -- `minimize` etait appele SANS `jac`,
et le pas de differences finies par defaut, absolu a 1,49e-08, donnait un
gradient 3 372 fois la vraie pente -- deplace `theta`, donc le metamodele,
donc le resultat.

Mesure sur la revision qui precede IMMEDIATEMENT la correction (`ac6fb8a`) et
sur `HEAD`, meme etude (`studies/pure_flexion_analytique.toml`), meme modele,
sur cette machine :

| | avant | apres | exact |
|---|---|---|---|
| `beta_FORM` | 4,7516 | **4,7527** | 4,77257 |
| `u*` | [−2,797 ; −3,841] | [−2,793 ; −3,846] | — |
| `Pf_FORM` | 1,0089e-06 | 1,0038e-06 | — |
| `beta_IS` | 4,6828 | 4,6721 | 4,77257 |
| `Pf_IS` | 1,4147e-06 | 1,4903e-06 | — |

L'ecart sur `beta_FORM` vaut **0,023 %**, et il va dans le bon sens : l'erreur
contre la valeur exacte passe de 0,0210 a 0,0199, soit **5,2 % de moins**.
La premiere divergence apparait des le PREMIER ajustement, sur le plan
initial de cinq points : `theta = [7,68468473 ; 9,31307410]` contre
`[7,68468409 ; 9,31307268]`, soit 1,5e-07 en relatif, pour un LOO identique a
sept chiffres (6,279100e-01). L'enrichissement EFF amplifie ensuite l'ecart
en placant ses points ailleurs.

Au passage, cette mesure etablit quelque chose de plus fort sur tout ce qui
separe la phase 7 du 26/08 de la correction du 02/09 -- la sortie de la
boucle EFF des etudes, les criteres d'arret, les historiques, l'ecriture non
destructrice, une quinzaine de commits : **la chaine analytique y rend des
chiffres IDENTIQUES**, `4,7516 / 1,0089e-06 / 4,6828 / 1,4147e-06` et le meme
`u*`, jusqu'au dernier chiffre imprime. Ces refactorisations n'ont deplace
aucun resultat.

**Ce deplacement n'avait ete ecrit nulle part.** Le commit du gradient
(`29c2e3d`) mesurait l'effet sur `theta`, sur les quatre LOO et sur la
reproductibilite, et notait `beta` INCHANGE -- ce qui etait vrai sur la
baseline GEPCK a 24 points (7,0e-07) et sur le Moulin Blanc, mais pas sur
cette chaine-ci. Aucun test ne fige le `beta` de la chaine analytique : la
baseline porte une autre configuration, et `test_103` verifiait qu'un `beta`
EXISTE, pas sa valeur. C'est corrige -- voir `test_103`.

## Effet de la phase 7 (26/08/2026)

Performance seule, resultat inchange. Sur la chaine analytique :

| | phase 6 | phase 7 |
|---|---|---|
| FORM + IS, cumul des 8 iterations | 42,0 s | **23,1 s** |
| `beta_FORM` | 4,7516 | 4,7516 |
| `u*` | [-2,797 ; -3,841] | idem |
| `beta_IS` | 4,6828 | 4,6828 |

C'est ce qu'une phase de performance doit produire : moitie moins de temps,
pas un chiffre deplace.

La justesse gagne aussi, mais ailleurs : la transformation isoprobabiliste
calculait `norm.ppf(norm.cdf(u))`, une identite que la double precision ne
tient pas au-dela de u = 7. Sur une grille couvrant [-7,5 ; 7,5]^2 et un etat
limite lineaire, l'erreur max passe de 2,510e-04 a 5,862e-14.

## Comment refaire la mesure

```bat
python tools\run_comparatif.py --patch pure_flexion\AC3_pure_flexion.py ^
                              --sortie C:\tmp\run --n-max-eff 8
python launcher.py pure_flexion\_run_comparatif.py > C:\tmp\sortie.txt
```

Puis comparer avec `tools\comparer_runs.py`. Toujours produire **trois**
runs : deux versions et une repetition. Sans `--repetition`, l'outil le dit
et refuse de presenter son tableau comme un verdict.
