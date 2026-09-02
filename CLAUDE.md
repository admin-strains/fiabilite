# Instructions pour un agent qui travaille dans ce depot

Ce depot calcule la probabilite de defaillance d'un ouvrage. Ses resultats
servent a dimensionner. La regle qui structure tout le reste, reprise de
[`CONTRIBUTING.md`](CONTRIBUTING.md) :

> **Une modification ne change pas un chiffre sans qu'on l'ait voulu, mesure,
> et ecrit.**

Lire d'abord [`README.md`](README.md) (ce que fait la chaine, comment lancer)
et [`CONTRIBUTING.md`](CONTRIBUTING.md) (les regles, en particulier la
regeneration d'un golden). Ce fichier ne les repete pas : il dit ce qu'un
agent doit savoir en plus, et ce qui coute cher a redecouvrir.

---

## Avant toute chose

```bat
run_tests.bat
```

Vert avant, vert apres. Si c'etait deja rouge, ce n'est pas le moment de
commencer.

---

## Les six pieges qui ont deja coute du temps

### 0. DEUX environnements, et la CI a le PLUS PAUVRE

`C:	mpenv_fiab310` a tout ; `C:	mpenv_ci` n'a que
`requirements/core.txt` -- numpy, scipy, pytest, tomli. **Les quatre jobs
`noyau` de l'integration continue tournent avec le second.** Un import
d'OpenTURNS ajoute dans `_lib/`, `_config/` ou `_cache/` y casse tout, et
passe inapercu en local.

Avant de pousser, les DEUX :

```bash
PYTHONPATH= C:/tmp/venv_fiab310/Scripts/python.exe -m pytest tests/ -q
PYTHONPATH= C:/tmp/venv_ci/Scripts/python.exe -m pytest tests/ -q
```

Mesure du 02/09/2026 : 1 264 verts avec la couche complete, 885 verts et 48
sautes avec le noyau seul. J'ai casse les quatre jobs `noyau` en faisant
valider la configuration contre le registre des lois, qui importe OpenTURNS
-- une commande de plus l'aurait dit avant le push.

### 1. `PYTHONPATH` global eclipse les environnements virtuels

Le `PYTHONPATH` de ce poste pointe vers `C:\python3\Lib\site-packages`.
Prefixer **toute** commande Python par `PYTHONPATH=` :

```bash
PYTHONPATH= C:/tmp/venv_fiab310/Scripts/python.exe -m pytest tests/ -q
```

Sans cela, on croit tester dans le venv et on teste ailleurs.

### 2. Ne pas installer OpenTURNS dans `C:\python3`

C'est le runtime de production de Digital Structure. OpenTURNS y apporte un
OpenBLAS MinGW complet dont trois DLL portent le nom de DLL de Digital
Structure (`liblapack.dll` : 14,4 Mo contre 0,17 Mo). L'environnement des
etudes est `C:\tmp\venv_fiab310`.

### 3. Un run REECRIT le modele

`patch_params` reecrit `dsCad.txt` et `dsLoad.txt` **a chaque evaluation**, et
le solveur depose ses maillages et ses caches dans le meme dossier -- 36
fichiers mesures sur le Moulin Blanc, 43 sur la flexion pure. Sauvegarder
le `.ds` avant tout run.

Le depot s'en protege : quand le `storage` configure n'existe pas,
`Configuration.chemin_ds` retombe sur `modeles/` du depot **par une copie**,
sous `_travail/`. La reference versionnee n'est jamais ecrite.

### 4. Le chemin du modele se calcule en UN SEUL endroit

`CFG.chemin_ds`. Jamais `os.path.join(storage, modelname + ".ds")`. Ce
recalcul s'est deja glisse a cinq endroits, et chaque fois il a diverge du
vrai chemin des que le repli jouait -- une fois dans les etudes, une fois
dans `_doe/parallele.py`, trois fois dans les tests, ou il faisait sauter
neuf tests en integration continue en annoncant un modele « absent » qui est
pourtant versionne.

### 5. Un appel solveur coute 466 s sur le Moulin Blanc

Tout ce qui peut etre verifie sans lui doit l'etre avant lui. C'est le role
de [`_config/coherence.py`](_config/coherence.py), appele en tete de chaque
run : 0,36 s pour dire ce qu'un premier appel solveur aurait dit apres
466 s, ou une chaine de DOE apres cinq heures.

---

## Ecrire une nouvelle etude

Une etude, c'est **un `.toml` dans `studies/`**, pas une copie du script.
Le schema est [`_config/schema.py`](_config/schema.py) : une cle mal
orthographiee est refusee, pas ignoree.

```bat
set FIABILITE_ETUDE=%CD%\studies\mon_etude.toml
python launcher.py pure_flexion\AC3_pure_flexion.py
```

### Les variables aleatoires s'y declarent aussi

```toml
[variables.fc]
loi     = "fc"                    # une loi du registre de _model/lois.py
args    = [48, 0.12]              # ses parametres, dans l'ordre
param   = "COMPRESSIVE_STRENGTH"  # le parametre que le solveur derive
solides = ["Block1"]              # un NOM est une donnee

[variables.fy]
loi   = "fy"
args  = [550]
param = "YIELD_STRENGTH"
```

Ce qui ne s'y declare PAS : la **selection** des elements. « Les armatures de
nuance fyd1 », c'est 13 858 noms et une propriete du modele, pas une donnee.
L'etude la calcule en une ligne :

```python
group1_names = _selection.armatures(_cad_txt, grade="fyd1")
```

C'est la seule ligne de Python qu'une nouvelle etude demande d'ecrire
(arbitrage d'Agnes du 02/09/2026, option B'). Pour savoir ce que contient un
modele avant de declarer : `_selection.nuances(cad)` rend les nuances
presentes et leur compte.

`region_key` n'est pas declarable : il vaut le nom de la variable. Le
declarer ouvrait la porte a un doublon -- deux variables ecrivant leur
sensibilite dans la meme region.

Ce que la declaration refuse, au CHARGEMENT et toutes fautes d'un coup : une
loi inconnue (avec la liste des choix), une clef obligatoire absente, une
clef inconnue, un `args` qui n'est pas une liste, et une variable qui ne
designe aucun element. `tests/test_135_ecrire_une_etude_en_toml.py` ecrit un
`.toml` neuf, le fait tourner, et verifie qu'une valeur declaree change
vraiment le resultat -- une declaration decorative serait pire qu'aucune.

### Regler d'abord, calculer ensuite

Commencer par `solveur = "analytique"` : toute la chaine s'exerce en 140 s,
sans licence ni GPU, et de maniere reproductible -- ce que le solveur reel ne
permet pas. Regler l'etude la, puis passer au solveur reel.

Les deux scripts `AC3_*.py` sont encore epais (1 080 et 949 lignes). Ils
**retrecissent** : `tests/test_91_ac_minces.py` porte un plafond de lignes qui
ne fait que descendre. Ne pas y ajouter de logique -- elle va dans un module,
d'ou elle sera testable.

---

## Ce qui est attendu d'une modification

**Mesurer, pas supposer.** Instrumenter avant d'expliquer. Se mefier de sa
propre sonde : quand deux mesures divergent, chercher laquelle a tort avant
de conclure. Reproductible n'est pas juste -- une variante a deja donne la
meilleure reproductibilite mesuree tout en convergeant vers un optimum
nettement moins bon.

**Ne pas contourner un defaut, le comprendre.** Un contournement qui rend le
symptome invisible laisse la cause en place. Exemple : le gradient de la
vraisemblance etait du bruit parce que `minimize` etait appele sans `jac` ;
passer a un optimiseur sans gradient aurait fait disparaitre le symptome sans
rien corriger. La derivation analytique, verifiee par differences finies, a
corrige la cause.

**Un temoin doit pouvoir tomber.** Un test qui passerait meme si ce qu'il
garde etait casse ne garde rien. Le verifier en cassant volontairement la
chose gardee.

**Un seuil se derive d'une mesure.** Un seuil qui tient a 10 % pres ne tient
pas. Et un temoin ne fige pas un compte dependant de la machine : ce qu'on
compare, c'est un CONTRASTE mesure sur la meme machine.

**Changer un contrat, c'est chercher qui l'affirme ailleurs.** Un `grep` sur
le nom avant de modifier son sens. Un contrat modifie sans cela a deja casse
le seul job vert de l'integration continue.

---

## Ce qu'on ne touche pas

- `tests/unit/` et `historique/` : ce sont des **temoins**. Quand l'un a tort,
  on le documente, on n'y touche pas.
- `solver/digital_structure.py` est la **seule** frontiere avec Digital
  Structure, et un test le verifie. Un besoin d'appel solveur ailleurs est le
  signe que le code appartient au solveur.
- `modeles/` : les modeles de reference versionnes. Un run travaille sur une
  copie (piege 3).

---

## Commits

Prefixe de module, puis **ce que la mesure a montre** -- les messages de ce
depot servent de journal de bord technique. Jamais de `git add -A`. Jamais de
commit tant que les tests ne passent pas.

---

## Ou en est le chantier

[`docs/plan-nettoyage.md`](docs/plan-nettoyage.md) donne les phases et leur
etat. Les questions ouvertes, avec leurs mesures, sont dans
[`docs/diagnostic-optimisation-theta.md`](docs/diagnostic-optimisation-theta.md).
