# Contribuer

Ce depot porte un calcul dont les resultats servent a dimensionner des
ouvrages. La regle qui structure tout le reste :

> **Une modification ne change pas un chiffre sans qu'on l'ait voulu, mesure,
> et ecrit.**

Le harness est la pour rendre cela verifiable, pas pour donner bonne
conscience.

---

## Avant de commencer

```bat
run_tests.bat
```

Vert avant, vert apres. Si c'etait deja rouge, ce n'est pas le moment de
commencer : trouvez pourquoi d'abord.

---

## Les trois filets, et ce que chacun attrape

| | attrape | ne voit pas |
|---|---|---|
| **oracle** (`test_20`, `test_40`) | un resultat FAUX -- compare a une verite analytique | un resultat different mais toujours juste |
| **golden** (`test_30`) | un resultat DIFFERENT -- compare a une valeur figee | si le nouveau est meilleur ou pire |
| **baseline** (`test_70`) | **ou** l'ecart apparait en premier dans la chaine | la cause |

Les trois sont complementaires. Un golden qui tombe ne dit pas que vous avez
tort ; il dit que quelque chose a bouge et qu'il faut le regarder.

---

## Regenerer un golden

C'est l'operation la plus dangereuse du depot : ecraser un golden efface la
memoire de ce que le code faisait avant. La procedure n'a pas d'exception.

1. **Un test golden tombe.** Ne pas regenerer tout de suite.
2. **Comprendre.** Quelle grandeur a bouge, de combien ?
   `tools/baseline_compare.py` donne la PREMIERE divergence -- le reste en
   decoule.
3. **Demontrer que le nouveau comportement est MEILLEUR**, pas seulement
   different. Les tests oracle doivent rester verts et, de preference,
   s'ameliorer. Une comparaison a la verite analytique sur une grille dense
   vaut mieux que sur cinq points.
4. **Regenerer** : `python tests/make_golden.py`.
5. **Commiter le golden AVEC la modification**, jamais separement, et
   **expliquer l'ecart chiffre dans le message**.

Exemple reel (phase 6, correction de la pepite) : avant de regenerer, il a
fallu montrer que l'erreur aux points sonde etait divisee par 722, que
l'interpolation passait de 2,96e-03 a 2,6e-09, et que la baseline gagnait en
justesse contre le meme oracle exact. Un golden regenere sans cette
demonstration ne prouve plus rien -- il enregistre la regression.

### Ce qui n'est PAS une raison de regenerer

- « ca ne change pas grand-chose » -- alors montrez de combien ;
- « le nouveau chiffre me parait plus logique » -- mesurez ;
- « ca fait tomber le test » -- c'est son travail.

---

## Comparer deux versions sur Digital Structure

**Toujours trois runs**, jamais deux : deux versions et une repetition de la
premiere. La chaine sur Digital Structure n'est pas reproductible au bit pres
-- 12,3 % d'etendue mesuree sur `Pf_IS` entre trois executions du meme code.
Sans repetition, tout ecart est impute a tort a votre modification.

```bat
python tools\run_comparatif.py --patch pure_flexion\AC3_pure_flexion.py ^
                              --sortie C:\tmp\run --n-max-eff 8
python launcher.py pure_flexion\_run_comparatif.py > C:\tmp\sortie.txt

python tools\comparer_runs.py origine.txt actuel.txt --repetition repetition.txt
```

Sans `--repetition`, l'outil le dit et refuse de presenter son tableau comme
un verdict.

**Sauvegardez le modele `.ds` avant tout run** : les scripts reecrivent
`dsCad.txt` a chaque evaluation.

---

## Les tests xfail

Un defaut connu s'ecrit comme un test qui decrit le comportement ATTENDU,
marque `xfail(strict=True)`. Consequence : le jour ou quelqu'un le corrige, le
test passe, donc `xpass`, donc **ECHEC** -- ce qui oblige a retirer le
marqueur et a acter la correction. Aucun defaut ne peut etre corrige en
silence, ni reapparaitre sans bruit.

N'ajoutez rien la sans : **symptome reproductible, localisation
fichier:ligne, et effet concret sur la production**.

---

## Ce qu'on ne touche pas

- **`tests/unit/`** -- les suites d'origine. Elles servent de TEMOIN : on ne
  reecrit pas un temoin. Quand l'une d'elles a tort, on le dit dans
  `STATUT_CONNU` de `test_10_legacy_unit.py`, avec la demonstration, et on
  ecrit un test correct ailleurs.
- **`historique/`** -- les versions anterieures des scripts, pour la meme
  raison.

---

## La frontiere avec Digital Structure

`solver/digital_structure.py` est le SEUL fichier du depot qui importe Digital
Structure pour calculer, et `tests/test_86_solveur.py` le verifie. Si votre
modification a besoin d'un appel au solveur ailleurs, c'est le signe qu'elle
appartient au solveur.

La regle vaut son prix : c'est elle qui permet d'exercer toute la chaine sur
un etat limite analytique, en secondes, sans licence -- et de maniere
reproductible, ce que Digital Structure ne permet pas.

---

## Style

- **Le francais est la langue du depot**, y compris dans les noms de
  fonctions nouvelles. Le code d'origine melange les deux ; on ne renomme pas
  l'existant pour le plaisir.
- **Un commentaire dit POURQUOI, pas QUOI.** « ajoute la pepite » est inutile ;
  « sans pepite la vraisemblance diverge et theta va au plafond » evite a
  quelqu'un de defaire la correction dans six mois.
- **Une affirmation porte son chiffre.** « c'est plus rapide » ne vaut rien ;
  « 1,227 ms -> 0,814 ms, meilleur temps sur dix essais » se verifie.
- Pas de `git add -A` : regardez ce que vous commitez.

---

## Message de commit

Prefixe la phase ou le module, puis **ce que la mesure a montre**. Les
messages de ce depot servent de journal de bord technique : c'est souvent
la seule trace de POURQUOI un chiffre a bouge.

```
phase 6: corriger les defauts 1, 2, 3, 4 et 5

DEFAUTS 2 ET 3 -- L'HYPOTHESE DU PLAN ETAIT FAUSSE
Le plan pariait sur la mise a l'echelle des blocs de derivees. Mesure :
l'equilibrage de Jacobi ne gagne RIEN (1,64e15 -> 1,63e15). La vraie cause
est l'absence de pepite [...]

    etat limite      N    pepite 0      pepite 1e-8
    flexion GEPCK    40   56,4 %        0,0015 %
```

Ecrivez ce qui vous a surpris. C'est ce qu'on relit.
