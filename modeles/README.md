# Modeles de test

Les deux modeles Digital Structure sur lesquels la chaine est validee. Ils sont
ici pour qu'une etude soit **rejouable ailleurs** : jusqu'a present ils vivaient
seulement dans le `storage` du poste de leur auteur, et le depot decrivait des
calculs que personne d'autre ne pouvait lancer.

| dossier | role | variables aleatoires |
|---|---|---|
| `test_pure_flexion.ds` | poutre BA 0,8 x 0,8 x 5,0 m, 3 lits de 8 HA32 | `fc` (48 MPa), `fy` (550 MPa) |
| `Calcul_fiabilite_G+LM1_13k_2fy_membrure_inf_diagonal.ds` | pont du Moulin Blanc, trafic LM1, 15 346 aciers | `fy1`, `fy2` (235 MPa, sigma JCSS 30,15) |

## Ce qu'il y a, et ce qu'il n'y a pas

**Presents** : les fichiers d'entree (`dsCad.txt`, `dsLoad.txt`, `dsNote.txt`,
`coupe.txt`) et la geometrie importee (`pont_complet.stp`).

**Absents, deliberement** : tout ce qu'un run PRODUIT -- `.dscad`, `.dsmed`,
`.dsmetares`, `.msh`, `.dslog`, ainsi que l'etat de run (`doe_cache.json`,
`points_log.jsonl`, `restart_state.json`). Sur la flexion pure cela represente
**42 des 43 Mo** du dossier d'origine : le modele lui-meme pese 0,01 Mo.

## Les valeurs des variables aleatoires sont NORMALISEES

`patch_params` reecrit `fy1`/`fy2` (ou `fc`/`fy`) **en place, dans le `dsCad.txt`
du modele**, a chaque evaluation. C'est le mecanisme meme par lequel la
geometrie est parametree -- pas un effet de bord.

Consequence : le fichier d'un modele qui a servi porte la valeur du **dernier
point evalue**, qui n'a aucun sens en soi. Les copies versionnees ici ont donc
ete remises a la **moyenne de la loi** de chaque variable, au format exact
qu'ecrit `patch_params` (`%.10f`). Le depot contient ainsi un etat de depart
deterministe, pas le residu d'un run.

## Avant de rejouer : deux points d'attention

**1. Sauvegarder.** Un run reecrit le `dsCad.txt` du modele qu'il evalue. Ne
pointez jamais une etude directement sur ces copies sans en avoir fait un
double, sinon le depot se salit au premier point calcule.

**2. Le chemin du STEP est absolu.** `dsCad.txt` du Moulin Blanc contient, en
ligne 8 :

```python
EXTERNAL_FILE("External_file0","C:/workspace/storage/admin/Moulin_Blanc/Calcul_fiabilite_G+LM1_13k_2fy_membrure_inf_diagonal.ds/pont_complet.stp")
```

C'est une donnee du modele, pas du code -- elle est recopiee telle quelle pour
rester fidele a l'original. **Il faut la reecrire** pour rejouer le modele
ailleurs. C'est aussi elle qui fait fonctionner le DOE parallele par accident :
les workers ne recopient que `dsCad.txt` et `dsLoad.txt` dans leur `.ds` isole,
jamais le STEP, et ne le trouvent que parce que ce chemin absolu les renvoie au
modele d'origine.

## Rejouer une etude sur ces copies

Les fichiers `studies/*.toml` designent le modele par `modelname` + `storage` :

```toml
modelname = "test_pure_flexion"
storage = 'C:\workspace\storage\admin\SF'
```

Pour travailler sur une copie, dupliquer le dossier hors du depot et pointer
`storage` dessus. Voir `docs/parametres-et-couts.md` : `storage` est un
parametre d'**etude**, pas de session.

## Cout d'un point

| modele | CAO | maillage | solveur | total |
|---|---:|---:|---:|---:|
| flexion pure | — | — | — | **~10 s** |
| Moulin Blanc | ~105 s | 9 s (13 804 tetraedres) | ~353 s | **466 s** |

Le rapport est de **46**. C'est pourquoi la chaine sait aussi tourner sur un
etat limite analytique (`solveur = "analytique"`, cf. `solver/fabrique.py`) :
elle s'y valide en secondes, sans licence.
