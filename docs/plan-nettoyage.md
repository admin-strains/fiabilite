# Plan de remise a niveau du code de fiabilite

> Etat au 25/08/2026, branche `cleaning`. Le depot vient d'un travail de
> stage : il calcule juste, il est structure comme un brouillon. L'objectif
> est d'en faire un outil livrable, sans perdre un seul resultat acquis.
>
> **Terminologie** — le paquet Python `STRAINS` importe par les scripts AC
> **est Digital Structure**, le solveur des depots `front/` et `back/` du
> workspace. Ce n'est pas une brique tierce et ce n'est pas un paquet pip.
> Le present document dit « Digital Structure » (ou DS) ; `STRAINS` n'y
> apparait que lorsqu'il s'agit du nom litteral du module importe.

---

## 1. Constat mesure

Tout ce qui suit est mesure sur le depot, pas estime.

### 1.1 Le code est un script, pas un logiciel

| Mesure | `Moulinblanc/AC3_moulinblanc.py` |
|---|---|
| lignes | 3 515 |
| dont dans `if __name__ == '__main__':` | **3 456 (98,3 %)** |
| fonctions au niveau module | 2 |
| fonctions imbriquees dans `main` | 63 |
| classes imbriquees dans `main` | 8 |
| variables de configuration globales | 57 |
| `print(` / `logging` | 167 / **0** |
| `except Exception` generiques | 22 |
| chemins absolus en dur | 25, sur 3 racines differentes |

Consequence directe : **rien n'est importable, donc rien n'est testable**.
Les 63 fonctions et 8 classes se ferment sur les 57 variables de `main` ;
aucune ne peut etre appelee depuis un test sans rejouer tout le script,
donc sans Digital Structure.

### 1.2 Le code est duplique

13 variantes de script AC sur les 4 branches, de 2 159 a 3 689 lignes.
Entre les deux scripts actifs de la branche `cleaning` :

- **3 273 lignes identiques** entre `AC3_pure_flexion.py` (3 471 lignes) et
  `AC3_moulinblanc.py` (3 515), soit **93 %** ;
- **61 fonctions imbriquees portent le meme nom** dans les deux fichiers ;
  seules `print_visu_EFF` et `print_visu_sigma` sont propres a l'un.

Autrement dit : il n'existe pas deux etudes, il existe **une chaine de
calcul recopiee deux fois**, puis divergee ligne a ligne. Chaque correction
doit etre faite deux fois, et l'a rarement ete.

### 1.3 Le seul socle sain n'etait pas protege

`_lib/` (PCK, GEPCK, noyaux, LARS, krigeage) est **quasi identique sur les
quatre branches** : seuls `branche1`, `branche3` et `branche5` different
entre `flexion` et les autres, de quelques lignes. C'est le coeur de valeur
du depot, et le seul code deja factorise.

Il etait couvert par 10 suites unitaires (~270 assertions) sur la branche
`fiabilite`. Ces suites ont ete **supprimees** lors des reorganisations vers
`flexion`, `moulin_blanc` et `dir-fiabilite`. Restaurees telles quelles sur
`cleaning`, **8 passent, 2 tombent** : elles avaient donc detecte des
regressions, dans un depot ou plus personne ne les lancait.

Elles ne pouvaient de toute facon pas servir de barriere : ecrites en style
script (compteur `PASS`/`FAIL`, aucun `assert`, aucun code retour), elles
affichent leurs echecs **et sortent en succes**.

### 1.4 Quatre defauts, dont un plantage

Traces dans `tests/test_50_known_defects.py` et `tests/test_10_legacy_unit.py` :

1. **Plantage** — `predict_gepck(fm, X)` leve `ValueError` quand `X` est
   exactement le DOE. `branche5.uq_eval_global_Kernel` l.1360 dispatche sur
   `isGram = np.array_equal(X1, X2)` et renvoie `R_tilde` (N·(M+1), N·(M+1))
   au lieu de `r0_tilde` (N, N·(M+1)) ; `branche4` l.206 casse au broadcast.
   En production : toute evaluation sur un point deja au DOE plante. Ca ne se
   voit pas parce que les points tombent rarement pile dessus.
2. **Precision** — GEPCK n'interpole son propre DOE qu'a **3,0e-03**, contre
   **5,3e-07** pour PCK : 4 ordres de grandeur, sur un krigeage sans pepite
   qui devrait interpoler exactement. Les gradients, eux, interpolent a 6,9e-06.
3. **Justesse** — sur le meme DOE, `beta` est faux de **1,30 %** avec GEPCK
   contre **0,011 %** avec PCK. GEPCK dispose de **plus** d'information
   (valeurs + gradients) et rend un resultat cent fois moins precis. Meme
   cause probable que le point 2 : conditionnement de `R_tilde`.
4. **Convention** — `branche5.uq_eval_global_Kernel` **contredit sa propre
   docstring** (l.1336-1338) depuis la branche `flexion` : la doc annonce
   `der=cb-1, dp=None`, le code calcule `der=None, dp=cb-1`. Les deux suites
   historiques en echec pointent la.

### 1.5 Performance

Mesure sur `FlexionLS`, DOE de 24 points :

| | point par point | vectorise | rapport |
|---|---|---|---|
| PCK, 2 000 points | 0,78 ms/pt | 0,004 ms/pt | **×221** |
| GEPCK, 2 000 points | 1,28 ms/pt | 0,008 ms/pt | **×151** |
| GEPCK, DOE de 48 pts | 4,79 ms/pt | 0,021 ms/pt | **×227** |
| grille 300×300 (90 000 pts) | 121 s | 1,09 s | **×111** |

Les grilles sont **deja** vectorisees (`_batch_mu_sigma`), il ne faut donc
pas compter ce gain deux fois. Mais elles le sont par une introspection
fragile — `getattr(getattr(sigma_func, '__self__', None), 'fm', None)` —
avec repli sur une boucle pour tout modele qui n'est ni PCK ni GEPCK. Le
chemin point-par-point reste celui de FORM (`_exec`, `_gradient`) et de
`_exec_sigma`, qui n'a aucun equivalent par lot. Le lot doit devenir l'API,
et non un cas particulier devine par `getattr`.

Le cout d'un appel de prediction **sur un seul point** — le chemin exact de
FORM et du gradient — monte avec la taille du DOE :

| DOE | `N·(M+1)` | PCK | GEPCK | fit GEPCK |
|---|---|---|---|---|
| 24 | 72 | 0,74 ms | 1,36 ms | 6,2 s |
| 48 | 144 | 0,82 ms | 4,61 ms | 26,1 s |
| 80 | 240 | 1,00 ms | 9,02 ms | 76,0 s |
| 120 | 360 | 4,30 ms | **14,15 ms** | **169 s** |

Un point unique ne devrait rien couter de plus quand le DOE grandit, une
fois le modele ajuste. Il coute dix fois plus. La cause est identifiee :
`branche4` l.199-203 reconstruit `Rinv` par
`solve(cholR, solve(cholR.T, eye(N_aug)))` — une **inversion explicite en
`O(N_aug^3)`, refaite a chaque appel de prediction**, quel que soit le
nombre de points evalues. Le facteur de Cholesky est pourtant deja
disponible depuis l'ajustement.

Le parametre de production `n_max_EFF_points = 360` porte le DOE vers
~365 points, soit `N_aug = 1095`. L'extrapolation cubique de la derniere
ligne place alors le seul reajustement GEPCK autour de l'heure — et l'AC
reajuste a chaque iteration d'enrichissement. Ce chiffre est une
extrapolation, pas une mesure : a confirmer sur un vrai run avant d'en
faire un argument.

### 1.6 Le depot lui-meme

- 4 branches divergentes, aucune fusionnee dans une autre : `fiabilite`
  (43 commits, 11/06), `dir-fiabilite` (161, 30/07), `moulin_blanc`
  (164, 10/07), `flexion` (157, **25/08**).
- Chacune reorganise l'arborescence differemment : `_lib/` contre `lib/`,
  `code/modules fiabilite/` (avec une espace dans le nom de dossier).
- Fichiers parasites versionnes : `C:tmpform_out.txt` (3,7 Mo, ne d'une
  redirection ratee, present en double avec un `:` pleine largeur), plusieurs
  `output_*.txt` jusqu'a 1,6 Mo, et **204 fichiers de `.playwright_profile/`**
  sur `moulin_blanc`.
- Aucun `README`, aucun fichier de dependances, aucune integration continue.

---

## 2. Ce qui est deja fait

**Phase 0 — filet de securite.** Branche `cleaning` creee depuis `flexion`,
harness de non-regression en place (`tests/`, commit `8f6e229`) : 62 tests
verts, 4 defauts en `xfail(strict)`, environ 50 s, **sans Digital Structure, sans
OpenTURNS et sans environnement de production**. Contrat detaille
dans `tests/README.md`.

Rien de ce qui suit ne se fait sans que ce harness soit vert avant et apres.

**Phase 0 bis — l'environnement est reproductible.** Le depot n'avait aucun
fichier de dependances, sur aucune des quatre branches, et dix lanceurs
recopies portant chacun ses chemins absolus. Etabli le 25/08 :

- `requirements/core.txt` — numpy, scipy, threadpoolctl, pytest. Installable
  partout, sans DS ni licence. C'est la couche ou vivent 4 522 des ~7 000
  lignes du depot.
- `requirements/studies.txt` — la couche AC : openturns, smt, autograd,
  scikit-learn, matplotlib, ndsplines, psutil. `nlopt` n'en fait PAS partie,
  contrairement a ce que laissait croire le code : NLopt est utilise via
  `ot.NLopt`.
- `requirements/constraints-reference.txt` — versions exactes du serveur ou
  les etudes ont tourne, pour reproduire cet environnement au paquet pres.
- `launcher.py` — un lanceur portable, sans chemin en dur, qui remplace les
  dix precedents. `python launcher.py --check` valide une installation sans
  lancer de calcul.
- `tests/test_60_environnement.py` — verifie les deux contraintes ci-dessous ;
  saute proprement sur un poste sans DS.

Deux contraintes ont ete mesurees, la seconde corrigeant une croyance :

1. **Python 3.10 exactement.** Les `.pyd` de DS sont lies a `python310.dll`.
   Un interpreteur 3.11+ echoue sur un « DLL load failed » trompeur qui n'a
   rien a voir avec une DLL manquante.
2. **OpenTURNS doit etre importe avant l'ajout des repertoires DLL de DS.**
   Les anciens lanceurs parlaient d'un « conflit MKL ». C'est faux : trois
   DLL portent le meme nom des deux cotes avec des contenus differents —
   `liblapack.dll` fait **14,4 Mo** chez OpenTURNS (OpenBLAS MinGW) contre
   **0,17 Mo** chez DS, `libblas.dll` 0,75 contre 0,10 Mo, plus `zlib1.dll`.
   Quand `bin\` de DS passe devant, `libot.dll` resout le mauvais LAPACK.

Resultat : un environnement neuf a ete monte de zero en quelques minutes, et
la chaine d'import d'un script AC passe **26 controles sur 26**. Le harness
donne les memes goldens dans deux interpreteurs distincts.

**Phase 2 — renommage, FAIT (`b6c65a6`).** Quatre modules mono-sujet
renommes : `branche1`->`api`, `branche2`->`options`, `branche4`->`predict`,
`branche_lars`->`lars`. Renommage pur, verifie par diff : trois fichiers
identiques a l'octet pres, `api.py` a deux lignes d'import pres. Coquilles de
compatibilite a liaison tardive (`__getattr__`), pour que `tests/unit/` reste
intouche et que l'instrumentation reste visible a travers l'ancien nom.

**Phase 3a — scission, FAIT (`e417aa9`, `5cd2424`).** Les deux modules
composites sont decoupes, decoupage instruit par
`tools/analyse_dependances.py` avant d'ecrire une ligne :

    branche5 -> polynomials (8) + transform (12) + kernels (9)
                zero arete entre les trois
    branche3 -> kriging (15) + pce_basis (4) + fit (2)
                stratification fit -> kriging, fit -> pce_basis

Corps des fonctions VERBATIM. Le dernier import differe de `_lib` a ete
remonte en tete, dans un commit separe.

Les deux etapes n'ont regenere AUCUN golden et laisse la baseline identique
au bit pres.

**Phases 3b a 3f — sortie du code hors de `main`, FAIT.** Ce qui vivait en
fonctions imbriquees dans les deux scripts AC devient des modules, chacun
avec son golden preleve AVANT deplacement par `tools/extraction_temoin.py` :

| commit | ce qui sort de `main` | vers |
|---|---|---|
| `93c2f43` | les 7 lois JCSS et la loi jointe | `_model/lois.py` |
| `e530870` | 10 fonctions de cache, parametres explicites | `_cache/doe.py`, `hf.py` |
| `5da97e7` | le critere EFF | `_reliability/eff.py` + `eff_ot.py` |
| `a77eafc` | FORM multimodal et tirage d'importance | `_reliability/form.py` |
| `9cc2ffc` | les traces | `_reliability/graphiques.py` |

Deux enseignements de ces etapes, tous deux devenus des tests :

- **`eff.py` a d'abord menti dans sa docstring** : annonce « numpy et scipy
  seulement », il importait openturns. Neuf tests rouges dans un interpreteur
  sans OpenTURNS. D'ou la scission `eff.py` / `eff_ot.py`, et le test qui
  verifie la couche d'appartenance de chaque module.
- **`run_IS_proj` existait en TROIS exemplaires** : extrait dans `form.py`
  mais jamais retire des deux AC, alors que le message de commit affirmait le
  contraire (`6bdef67` corrige). Chaque extraction porte desormais un test qui
  exige que l'AC ne contienne plus qu'une delegation.

Duplication entre les deux scripts AC : **3 273 -> 2 760** lignes identiques.

**Phase 4a — schema de configuration, FAIT (`259425b`).** `_config/schema.py`
(dataclass gelee, 57 champs) et un `.toml` par etude. L'inventaire prealable a
montre que deux etudes ne different pas par soixante parametres mais par
**douze** : le reste appartient aux defauts. Sept `do_*` et deux corrections
tardives (`do_IS = do_IS and modele != 'HF'`) deviennent des valeurs derivees,
qui ne peuvent plus se contredire.

**Phase 4b — debranchement des scripts, FAIT.** Les 53 et 54 affectations
litterales quittent les scripts AC ; ne restent que 14 accumulateurs d'etat.
Trois consequences qui depassent la configuration :

1. **Zero chemin absolu du poste de l'auteur.** Ils etaient 25. Le dernier
   verrou etait `launcher3.py`, une copie du lanceur portant
   `C:\_workingDir\_SF\test flexion\_lib` : les workers de DOE parallele
   passaient par lui. Ce chemin de code **ne pouvait pas s'executer ailleurs
   que sur le poste de son auteur** — ce qui explique qu'il n'ait jamais ete
   couvert. Les deux copies sont supprimees au profit de `launcher.py
   --garder-cwd`.
2. **Chaque run porte sa configuration**, imprimee en tete de journal
   (`schema.resume`) et deposee en JSON a cote des figures
   (`schema.ecrire_trace`). Sans cela, un ecart de reglage et un ecart de code
   se lisent pareil dans un journal — cf. la mesure de reproductibilite.
3. **L'oracle a change de support.** Tant que les scripts portaient leurs
   valeurs, ils etaient la reference et `test_85` comparait le TOML au script.
   Une fois debranches, cette comparaison serait devenue vide : l'oracle a donc
   ete recopie a la revision anterieure dans `tests/golden/config_*.json` par
   `tools/golden_config.py`, et trois tests interdisent qu'il se vide en
   silence.

**Phase 5 — isolement de Digital Structure, FAIT.** La surface de couplage
etait minuscule et pourtant recopiee QUATRE fois : `run_one_SOL` et `run_HF`,
dans les deux scripts AC, ~90 lignes chacune. Tout le reste — 2 700 lignes de
plan d'experiences, metamodele, enrichissement, FORM et tirage d'importance —
etait deja du Python pur.

Le releve prealable par AST (`tools/golden_options_ds.py`) a montre que sur
22 options de maillage et 22 options de solveur, **deux avaient diverge** :

    global_physical_size          run_one_SOL : `global_size` (configuration)
                                  run_HF      : 0.05 (en dur)
    geometric_approximation_min   run_one_SOL : `str(geo_min_approx)`
                                  run_HF      : "4" (en dur)

Or `run_HF` sert aux points d'ENRICHISSEMENT EFF, qui rejoignent le plan
d'experiences : regler `global_size = 0.007` — ce que le commentaire d'origine
invitait a faire — aurait entraine le metamodele sur des points calcules avec
DEUX MAILLAGES DIFFERENTS, sans un mot. Le defaut etait dormant, les deux
etudes etant a 0.05. Une seule implementation ne peut plus diverger d'elle-meme.

Ce que la phase a produit :

- `solver/interface.py` — le contrat. Le solveur rend `g`, `alpha`, le gradient
  DANS L'ESPACE PHYSIQUE et son ETAT DE SANTE. Le passage en espace standard
  reste cote fiabilite : la transformation isoprobabiliste appartient a la loi
  jointe, pas au maillage.
- `solver/digital_structure.py` — la seule porte vers Digital Structure.
  Options recopiees valeur par valeur, verifiees contre le golden.
- `solver/analytique.py` — le meme contrat sur la forme fermee, geometrie lue
  dans les MEMES fichiers texte du `.ds`.
- `solver/fabrique.py` — ne charge que l'implementation demandee, ce qui est
  toute la raison pour laquelle les scripts AC n'ont plus besoin d'une licence.

**Les deux scripts AC n'importent plus Digital Structure du tout.** Une fois
les appels delegues, les seuls noms qu'ils lui empruntaient encore etaient
`INITCATALOG` — passe dans l'implementation — et `sys`, qui n'etait jamais
importe et ne marchait que parce que le `import *` le laissait fuiter
(defaut 7, corrige). L'espace de noms d'un script AC passe de **302 a 51
symboles** avant `__main__`.

Deux mesures pour finir :

1. **La chaine complete tourne sur l'etat limite analytique** et retrouve
   `beta_FORM = 4,7527` contre `4,77257` calcule sans metamodele et sans FORM
   (minimisation scalaire a 1e-12) : **0,42 %**. Le journal fait 226 lignes au
   lieu de 20 000.
2. **Elle est bit-reproductible.** Deux executions donnent des journaux
   identiques hors chronometrage — la ou la meme chaine sur Digital Structure
   affiche 12,3 % d'etendue sur `Pf_IS`. C'est ce qui permet enfin d'attribuer
   un ecart au code plutot qu'au solveur.

**La mesure la plus importante du chantier** n'est pas une etape du plan mais
son resultat : `docs/reproductibilite-chaine-complete.md`. Digital Structure
n'est pas reproductible au bit pres — meme point, meme maillage, memes 23
iterations, `alpha` differe au onzieme chiffre. Amplifie x4·10⁹ le long de la
chaine, cela donne **12,3 % d'etendue sur `Pf_IS`** entre trois executions du
meme code, pour un critere d'arret a COV = 5 %. Toute comparaison A/B sur
cette chaine exige donc **trois** runs, dont une repetition.

---

## 3. Cible

### 3.1 Un paquet, pas une collection de scripts

```
fiabilite/
  surrogate/          metamodeles -- ex-_lib, aucune dependance lourde
    options.py          <- branche2      lecture et validation des options
    fit.py              <- branche3      PCE + krigeage + LOO
    predict.py          <- branche4      evaluation, variance, gradients
    kernels.py          <- branche5      noyaux et derivees
    polynomials.py      <- branche5      Hermite, Legendre, base PCE
    transform.py        <- branche5      transformation isoprobabiliste
    lars.py             <- branche_lars
    api.py              <- branche1      fit_pck / fit_gepck / predict_*
  reliability/        FORM, IS, EFF, separation des modes -- sans DS
  model/              lois, PARAM_CONFIG, etat limite
  solver/             SEUL point de contact Digital Structure (SOCP, maillage, sensibilites)
  cache/              doe_cache, hf_cache, restart_state
  viz/                figures
  config/             schema de configuration d'etude
studies/
  pure_flexion.toml
  moulin_blanc_2fy.toml
  ...                 une etude = un fichier de config, pas un fork de 3 500 lignes
```

Les noms `branche1` a `branche5` datent du decoupage du portage UQLab en
lots de travail. Ils ne disent rien de ce que fait le code : c'est la
premiere chose que lit un nouvel arrivant, et elle ne l'informe pas.

### 3.2 La regle qui structure tout le reste

**Une seule frontiere avec Digital Structure.** Aujourd'hui le solveur est appele
depuis le milieu de `main`, ce qui rend toute la chaine non testable. Cible :
`solver/` expose une interface a trois methodes (`evaluate`,
`evaluate_batch`, `gradient`), avec deux implementations :

- `DigitalStructureSolver` — la vraie, qui appelle SOCP ;
- `AnalyticSolver` — un etat limite analytique, **deja ecrit**
  (`tests/reference/limit_states.py`).

Tout ce qui est en amont de cette frontiere devient testable sur un poste
sans Digital Structure, en secondes. C'est ce qui permet une base de tests unitaires
qui couvre autre chose que `_lib/`.

---

## 4. Phasage

Chaque phase se termine par : harness vert, goldens a jour et justifies,
commit unique et reversible.

### Phase 1 — Trancher la convergence des branches · DECISION, pas technique

Quatre branches divergentes, une seule peut devenir la base. `cleaning` part
de `flexion` (la plus recente, et sa version de `_lib` est en avance). Mais
`moulin_blanc` porte des elements qui n'existent nulle part ailleurs :
`tests/` (20 fichiers de validation physique), `_docs/` (6 documents dont la
passation 2-fy et la theorie du decouplage de position), `_tools/`, et
`rebar_grouping/` (le decoupage des 15 000 aciers en groupes).

**A arbitrer avec Semia et Mohamad** : ce qui est repris de `moulin_blanc`,
ce qui est archive, ce qui est abandonne. Tant que ce n'est pas tranche,
toute factorisation risque de porter sur du code qu'on jettera.

*Sortie : une liste explicite fichier par fichier. Rien d'autre.*

### Phase 2 — Renommer sans rien changer d'autre · 1 j

`branche1` a `branche5` vers des noms metier, imports mis a jour, **aucune
ligne de logique touchee**. Le harness doit rester vert **sans regenerer un
seul golden** : c'est la demonstration que le renommage est neutre, et le
premier vrai essai du filet.

Un module de compatibilite (`branche1.py` reexportant `api.py`) garde les
scripts AC fonctionnels pendant la transition.

*Sortie : 0 golden modifie.*

### Phase 3 — Extraire le noyau hors de `main` · 5 a 8 j

Le gros morceau. Dans l'ordre, du moins couple au plus couple :

1. **lois et configuration** (`loi_fc`, `loi_fy`, `loi_uni_approx`,
   `dist_jointe`, `PARAM_CONFIG`) — pur calcul, aucune dependance ;
2. **caches** (`_save`/`_load_doe_cache`, `_hf_cache`, `_restart_state`) —
   entrees-sorties pures, testables sur fichiers temporaires ;
3. **EFF** (`_eff_vectorized`, `_find_batch_EFF_points`) — deja vectorise,
   deja pur, mais enferme dans `main` ;
4. **FORM et separation des modes** (`FORM_all_modes`, DBSCAN,
   `FORM_warm_start`) ;
5. **IS** (`run_IS`, `run_IS_proj`, `adaptive_is`).

Chaque brique sortie de `main` recoit ses tests unitaires **dans le meme
commit**. C'est la que la base de tests se construit reellement : le harness
actuel protege `_lib/`, cette phase etend la couverture a la chaine de
fiabilite elle-meme.

*Sortie : les 61 fonctions communes aux deux AC existent en un seul
exemplaire, importables et testees.*

### Phase 4 — Configuration declarative · FAIT

Les 57 variables de `main` deviennent un schema valide (dataclass + fichier
`.toml` par etude). Une etude n'est plus un fork de 3 500 lignes mais un
fichier de configuration d'une trentaine de lignes.

L'effet de bord attendu s'est produit : les 25 chemins absolus ont disparu et
le depot s'installe ailleurs que sur le poste de son auteur. Voir §2 pour ce
qui a ete mesure en chemin.

Ce qui reste de la cible : les deux AC ne sont pas encore « deux fichiers de
configuration et un runner commun ». Le bloc de liaison
(`n0 = CFG.n0`, ...) subsiste tant que les 2 700 lignes suivantes lisent des
variables globales — il tombe avec la phase 5, quand `CFG` sera passe en
argument.

### Phase 5 — Isoler Digital Structure · FAIT

Interface `solver/` a deux implementations. Voir §2 pour le detail de ce qui a
ete mesure : la surface de couplage tenait en QUATRE copies du meme appel, qui
avaient diverge sur la taille de maille.

*Sortie obtenue : la chaine complete — plan d'experiences, metamodele,
enrichissement EFF, FORM multimodal, tirage d'importance — tourne sur l'etat
limite analytique, sans licence ni GPU, et retrouve `beta = 4,7527` contre
`4,77257` exact (0,42 %).*

Reste a faire pour boucler : porter ce run dans la CI, une fois la chaine
assez rapide pour y tenir (elle prend quelques minutes, dominees par les
iterations FORM+IS, pas par le solveur).

### Phase 6 — Corriger les defauts · FAIT

Menee dans l'ordre prevu, le quatrieme conditionnant les autres. Les criteres
de reussite avaient ete chiffres D'AVANCE — interpolation GEPCK sous 1e-6 et
ecart sur `beta` sous 0,05 % — et sont tenus avec de la marge.

**Defaut 4 — la convention `der`/`dp`. Le code avait raison, la
documentation avait tort.** Deux docstrings de `kernels.py` annonçaient la
derivee par rapport au premier argument la ou le code derivait le second.
Tranche par differences finies : la l-ieme observation augmentee est
`dy/dx_l` AU POINT D'APPRENTISSAGE, donc
`Cov(y(x*), dy/dx_l(x^j)) = dk(x*, x^j)/dx^j_l` — le second argument.
Les docstrings sont corrigees ; le test herite qui encode la convention
inverse reste `xfail` et INCHANGE, parce qu'un temoin ne se reecrit pas.
20 tests ancrent desormais la convention (`test_51_convention_derivees.py`).

**Defaut 1 — le dispatch `isGram` devinait.** `uq_eval_global_Kernel`
choisissait entre deux formes de retour, `(n(m+1), n(m+1))` et
`(n1, n2(m+1))`, en inspectant le CONTENU des tableaux. Evaluer le
metamodele sur un point deja present dans son plan d'experiences le faisait
donc basculer sur la mauvaise branche. L'appelant le DIT maintenant, via
`options['IsGram']`, et un `IsGram=True` sur deux jeux distincts est refuse.

**Defauts 2 et 3 — ce n'etait pas l'echelle des blocs, c'etait l'absence de
pepite.** L'hypothese inscrite au plan — « mise a l'echelle des blocs de
derivees » — est FAUSSE : l'equilibrage de Jacobi ne gagne rien
(1,64e15 → 1,63e15), pas plus que le raffinement iteratif ou `lstsq`.

La vraie cause est en amont. Les etats limites de fiabilite sont tres
lisses ; sans pepite, la vraisemblance croit indefiniment avec les longueurs
de correlation, parce qu'une matrice plus singuliere la gonfle
artificiellement. L'optimiseur allait donc au plafond — **deux des quatre cas
de reference y etaient colles, a theta = 100**. C'est l'estimation de `theta`
elle-meme qui etait cassee, pas seulement la resolution.

Ce que cela coutait, mesure par `tools/mesure_pepite.py` :

| etat limite | N | pepite 0 | pepite 1e-8 |
|---|---|---|---|
| flexion, GEPCK | 24 | 1,30 % | 0,0072 % |
| flexion, GEPCK | 40 | **56,4 %** | 0,0015 % |
| lineaire, GEPCK | 40 | **466 %** | exact |

L'erreur EMPIRE quand le plan grandit — a l'envers de ce qu'on attend, et la
boucle d'enrichissement EFF, elle, ajoute des points. Le cas lineaire est le
plus parlant : l'etat limite est un hyperplan que le metamodele contient
exactement, et sans pepite il rendait `beta = 19,8` au lieu de 3,5.

`kernels.PEPITE_PAR_DEFAUT = 1e-8`, valeur choisie par balayage sur 2 etats
limites x 4 tailles de plan x 2 metamodeles.

**Defaut 5 — conversion depreciee.** Quatre `float()` sur des tableaux 1x1,
que NumPy annonce comme une future ERREUR : le code aurait cesse de
fonctionner sans que rien n'ait change dans le depot. Corriges par `.item()`.
Un test statique recense les cas legitimes restants — il en a trouve un
cinquieme que la lecture avait manque.

**Ce qui change de chiffres.** C'est la premiere phase qui en change
volontairement. Goldens et baseline ont tout detecte, et la regeneration a
ete precedee de la demonstration exigee : la GENERALISATION s'ameliore aussi,
pas seulement l'interpolation — erreur aux points sonde divisee par 722
(GEPCK flexion), 561 (GEPCK lineaire), 94 et 4 pour PCK. La baseline passe
de 0,0771 % a 0,0492 % d'erreur sur `beta` contre le meme oracle exact.

`TOL_BETA` de `test_40` passe de 0,5 %/2 % — des seuils cales sur le defaut —
au critere du plan, 0,05 % pour les deux.

**Reste ouvert.** Le defaut 6 (les scripts lisaient `Primal_bound` sans
regarder `converged`) est instrumente en phase 5 : `Evaluation.sain` et
`exige_sain()` existent, et un point non converge est desormais SIGNALE dans
le journal — mais il entre encore au plan d'experiences, comme avant. En
faire une erreur est un choix d'exploitation, a arbitrer avec Semia et
Mohamad. Le defaut 7 (`sys` non importe) est corrige en phase 5.

### Phase 7 — Performance · 2 a 4 j

Une fois seulement la structure saine et les defauts corriges : optimiser
avant, c'est optimiser du code qu'on va deplacer.

- l'evaluation par lot devient l'API (`evaluate_batch`), le point unique
  devient le cas particulier — et non l'inverse ;
- supprimer l'inversion explicite de `R_tilde` sur le chemin chaud
  (`branche4` l.199-203 construit `Rinv` par `solve(cholR, solve(cholR.T, I))`
  a **chaque appel de prediction**) au profit d'un facteur pre-calcule au fit ;
- profiler ensuite, sur un vrai cas, avant de toucher a autre chose.

*Sortie : chiffres avant-apres sur un cas de reference, dans le commit.*

### Phase 8 — Documentation et integration continue · 2 j

`README` (installation, lancement, resultat attendu), fichier de
dependances, `CONTRIBUTING` avec la regle de regeneration des goldens, et la
CI qui lance le harness a chaque push. Sans CI, l'experience de ce depot
montre que les tests cessent d'etre lances en quelques semaines.

Nettoyage du depot : `C:tmpform_out.txt` et ses 3,7 Mo, les `output_*.txt`,
`.playwright_profile/`.

---

## 5. Ordre de grandeur

| Phase | Charge | Nature |
|---|---|---|
| 1 — convergence des branches | — | decision (Semia + Mohamad) |
| 2 — renommage | 1 j | mecanique, risque nul |
| 3 — extraction du noyau | 5-8 j | le coeur du chantier |
| 4 — configuration | 2 j | structurant |
| 5 — isolation Digital Structure | 3 j | debloque la CI |
| 6 — defauts | 3-5 j | qualite numerique |
| 7 — performance | 2-4 j | mesure avant-apres |
| 8 — doc et CI | 2 j | perennite |
| | **18-25 j** | hors phase 1 |

---

## 6. Regles du chantier

1. **Le harness est vert avant et apres chaque commit.** Un commit qui le
   laisse rouge n'entre pas.
2. **Un golden ne se regenere jamais en reflexe.** Prouver d'abord par les
   tests oracle que le nouveau comportement est meilleur, puis regenerer,
   puis justifier l'ecart dans le message de commit.
3. **Une phase egale un commit reversible.** Pas de refactoring qui traverse
   trois phases.
4. **Tout defaut trouve en chemin devient un `xfail(strict)` avant d'etre
   corrige.** Sinon il disparait des radars.
5. **Aucun resultat acquis n'est perdu.** Les valeurs de reference des etudes
   en cours (Moulin Blanc, flexion pure) sont figees en golden **avant** d'y
   toucher.

---

## 7. Hors perimetre

Ce plan ne traite ni la physique des etats limites, ni le choix des lois
JCSS, ni la strategie d'enrichissement EFF, ni le parametrage des etudes en
cours. Il ne change **aucun resultat** : si un chiffre bouge, c'est un echec
du plan, pas un progres — sauf pour les quatre defauts de la phase 6, ou le
changement est l'objectif et le critere est chiffre d'avance.
