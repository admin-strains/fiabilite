# Fiabilite structurale par calcul a la rupture

Probabilite de defaillance d'un ouvrage, a partir d'un calcul a la rupture.
L'etat limite est `g = alpha - 1`, ou `alpha` est le multiplicateur de rupture
rendu par le solveur. La chaine construit un metamodele de `g`, l'enrichit la
ou il compte, puis cherche les modes de defaillance et integre la probabilite.

```
plan d'experiences  ->  metamodele PCK/GEPCK  ->  enrichissement EFF
                                                        |
                    probabilite <- tirage d'importance <- FORM multimodal
```

Auteurs du code d'origine : Semia Frikha et Mohamad Moussa. La branche
`cleaning` en fait un logiciel installable ailleurs que sur le poste de son
auteur -- voir [`docs/plan-nettoyage.md`](docs/plan-nettoyage.md).

---

## Installer

**Deux couches**, et c'est la distinction la plus utile de ce depot.

### La couche de calcul -- partout, sans licence

`numpy`, `scipy`, `pytest`. C'est la ou vivent le metamodele, FORM, le tirage
d'importance, et les 300 tests.

```bat
python -m pip install -r requirements\core.txt
run_tests.bat
```

### La couche des etudes -- pour evaluer un vrai ouvrage

OpenTURNS, scikit-learn, smt, autograd, matplotlib. Et, pour le solveur reel,
**Digital Structure**.

```bat
python -m pip install -r requirements\studies.txt
python launcher.py --check
```

Deux contraintes ont ete mesurees, et `launcher.py` les applique pour vous :

1. **Python 3.10 exactement.** Les modules compiles de Digital Structure sont
   lies a `python310.dll`. Un interpreteur 3.11+ echoue sur un
   « DLL load failed » trompeur qui n'a rien a voir avec une DLL manquante.
2. **OpenTURNS s'importe AVANT les repertoires DLL de Digital Structure.**
   Les anciens lanceurs parlaient d'un « conflit MKL ». C'est faux : trois DLL
   portent le meme nom des deux cotes avec des contenus differents --
   `liblapack.dll` fait 14,4 Mo chez OpenTURNS (OpenBLAS MinGW) contre 0,17 Mo
   chez Digital Structure.

Pour reproduire au paquet pres l'environnement ou les etudes ont tourne :
`requirements/constraints-reference.txt`.

---

## Lancer une etude

```bat
python launcher.py pure_flexion\AC3_pure_flexion.py
```

Une etude, c'est un fichier `.toml` dans `studies/`, valide par
[`_config/schema.py`](_config/schema.py) :

```toml
modelname = "test_pure_flexion"
storage   = 'C:\workspace\storage\admin\SF'
solveur   = "digital_structure"

modele           = "PCK"
n0               = 5
EFF_criteria     = "at_least_one"
n_max_EFF_points = 30
```

Il ne contient que ce qui s'ecarte des defauts -- une trentaine de lignes, la
ou une etude etait un fork de 3 500. Une cle mal orthographiee est **refusee**,
pas ignoree. Chaque run imprime sa configuration en tete de journal et la
depose en JSON a cote de ses figures : sans cela, un ecart de reglage et un
ecart de code se lisent pareil.

Changer d'etude sans toucher au script :

```bat
set FIABILITE_ETUDE=C:\chemin\vers\mon_etude.toml
python launcher.py pure_flexion\AC3_pure_flexion.py
```

### Sans licence

`solveur = "analytique"` remplace le calcul a la rupture par une forme fermee,
lue dans la **meme** geometrie. Toute la chaine s'exerce, sans GPU ni licence,
en **140 s** -- treize evaluations qui couteraient chacune une dizaine de
secondes de SOCP, et un journal de 226 lignes au lieu de 20 000 :

```bat
set FIABILITE_ETUDE=%CD%\studies\pure_flexion_analytique.toml
python launcher.py pure_flexion\AC3_pure_flexion.py
```

Resultat attendu : `beta_FORM = 4,7516` contre **4,77257** calcule sans
metamodele ni FORM, soit 0,42 %. Deux executions donnent des journaux
identiques hors chronometrage.

---

## Ce qu'il faut savoir avant de lire un resultat

**La chaine sur Digital Structure n'est pas reproductible au bit pres.** Meme
point, meme maillage, memes iterations, statut OPTIMAL : `alpha` differe au
onzieme chiffre. Amplifie le long de la chaine, cela donne **12,3 % d'etendue
sur `Pf_IS`** entre trois executions du meme code, pour un critere d'arret a
COV = 5 %.

Trois consequences pratiques :

- ne pas publier `Pf` a plus de deux chiffres significatifs ;
- toute comparaison de deux versions exige **trois** runs -- deux versions et
  une repetition -- sans quoi l'ecart est impute a tort au code ;
- les **modes FORM finaux**, eux, se sont reveles invariants a travers huit
  etapes de restructuration. C'est la grandeur a regarder.

Le detail, avec les mesures : [`docs/reproductibilite-chaine-complete.md`](docs/reproductibilite-chaine-complete.md).

---

## Le depot

| | |
|---|---|
| `_lib/` | metamodele : krigeage, base PCE, LARS, noyaux, prediction |
| `_model/` | lois de probabilite (JCSS) |
| `_cache/` | reprise d'un run interrompu |
| `_reliability/` | critere EFF, FORM multimodal, tirage d'importance, traces |
| `_config/` | schema de configuration d'une etude |
| `solver/` | **la seule frontiere avec Digital Structure** |
| `studies/` | un `.toml` par etude |
| `tests/` | 300 tests -- voir [`tests/README.md`](tests/README.md) |
| `tools/` | instrumentation et mesure -- voir ci-dessous |
| `docs/` | plan de nettoyage, reproductibilite |
| `historique/` | versions anterieures des scripts, conservees comme temoins |

`find_all_modes.py` et `projected_polyhedron.py`, a la racine, sont une
**exploration de recherche** : trouver tous les modes de defaillance par
l'algorithme du Projected Polyhedron plutot que par DBSCAN. Ils ne font pas
partie de la chaine et rien ne les appelle.

### L'outillage

Il ne sert pas a faire tourner la chaine, mais a savoir ce qu'elle fait.

| | |
|---|---|
| `baseline_run.py` / `baseline_compare.py` | rejoue une chaine de reference et dit **ou** un ecart apparait en premier |
| `comparer_runs.py` | compare deux journaux de run complet ; exige une repetition pour etablir le plancher de bruit |
| `mesure_pepite.py` | effet de la pepite sur conditionnement, interpolation et `beta` |
| `mesure_prediction.py` | cout d'un appel de prediction, decompose |
| `analyse_dependances.py` | graphe d'appels, partitions, cycles |
| `extraction_temoin.py` | recupere une fonction encore imbriquee dans un script, y compris a une revision git |
| `solve_one.py` | un point sur Digital Structure, ~10 s |

---

## Contribuer

Les regles qui comptent -- en particulier **quand et comment regenerer un
golden** -- sont dans [`CONTRIBUTING.md`](CONTRIBUTING.md).

### Integration continue

`.github/workflows/tests.yml` lance le harness a chaque push, sur quatre
combinaisons (Linux et Windows, Python 3.10 et 3.13) pour la couche noyau,
plus un job qui installe la couche des etudes.

La propriete qu'on ne peut pas tenir a l'oeil -- `_lib/` n'importe **que**
numpy et scipy -- est verifiee par un test, pas par une etape de CI : un
controle qui n'existe qu'en CI ne protege pas celui qui travaille en local.
Un import de trop y casserait la portabilite en silence, et jusqu'a la
phase 8 un tel import interrompait la COLLECTE de pytest -- la suite entiere
cessait de tourner au lieu de sauter un fichier.

Digital Structure n'etant pas un paquet pip, les tests qui en dependent
sautent proprement. Le run de bout en bout sur l'etat limite analytique ne
peut pas tourner en CI non plus : il lit sa geometrie dans un modele `.ds`
qui n'est pas dans le depot.
