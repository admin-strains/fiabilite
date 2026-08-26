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

## Comment refaire la mesure

```bat
python tools\run_comparatif.py --patch pure_flexion\AC3_pure_flexion.py ^
                              --sortie C:\tmp\run --n-max-eff 8
python launcher.py pure_flexion\_run_comparatif.py > C:\tmp\sortie.txt
```

Puis comparer avec `tools\comparer_runs.py`. Toujours produire **trois**
runs : deux versions et une repetition. Sans `--repetition`, l'outil le dit
et refuse de presenter son tableau comme un verdict.
