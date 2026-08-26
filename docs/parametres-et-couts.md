# Parametres : qui les regle, et ce qu'ils declenchent

> Question d'Agnes, 26/08/2026 : « quelles options sont liees a un run
> utilisateur, et lesquelles sont internes au code ? »
>
> La reponse ne l'etait pas. La phase 4 avait rassemble cinquante-trois
> affectations dans un fichier plat, sans dire laquelle definit l'ETUDE et
> laquelle decrit seulement la SESSION qui la calcule. Ce document repare
> cela, et ajoute ce qui manquait le plus : **le motif de calcul que chaque
> parametre declenche**.

## Ce qui a revele le probleme

`restart_enrich_only = true` figurait dans `studies/moulin_blanc.toml` parce
que le script le portait. Or c'est un mode de session -- « reprends
l'enrichissement la ou tu l'avais laisse » -- fige dans ce qui ressemble a une
definition d'etude. Consequence : l'etude etait **injouable sur tout poste
depourvu du `restart_state.json`**, qui vit dans le `.ds` et non dans le
depot.

Un parametre mal range ne fait pas qu'encombrer : il rend une etude
intransmissible.

---

## Le chiffre qui gouverne tout

Cout d'UNE evaluation de l'etat limite, mesure le 26/08/2026 :

| modele | construction CAD | maillage | solveur | **total** |
|---|---|---|---|---|
| flexion pure (1 873 tetra) | ~1 s | ~1 s | ~8 s | **~10 s** |
| Moulin Blanc (13 804 tetra) | ~105 s | 9 s | 353 s | **466 s** |
| etat limite analytique | — | — | — | **~0,0001 s** |

Tout ce qui suit se lit avec ce facteur **46 entre les deux modeles reels**, et
**4,6 millions** entre le Moulin Blanc et la forme fermee.

### Combien d'appels un run declenche

```
    n0                              plan d'experiences initial
  + n_EFF                           enrichissement, <= n_max_EFF_points
  + 1                               FOSM en u = 0        (mis en cache)
  + n_modes                         un run_HF(u*) par mode trouve
  + n_grid_hf^2      si print_HF        grille de contour
  + n_grid_hf^n_var  si print_fullHF    grille complete
  + n_sp             si print_grad_sp   gradients aux points de depart
```

Sur le Moulin Blanc regle a `n_max_EFF_points = 360` et `n_grid_hf = 15` :
365 appels de calcul (**47 h**) plus 225 appels **uniquement pour une figure**
(**29 h**). La figure coute donc plus de la moitie du calcul.

---

## ETUDE -- definit le probleme, change le resultat

Ce que l'ingenieur choisit. Se transmet avec l'etude, se cite dans une note de
calcul.

| parametre | motif de calcul declenche | cout |
|---|---|---|
| `modelname`, `storage` | designe le `.ds` a evaluer | — |
| `solveur` | `digital_structure` ou `analytique` | **facteur 4,6·10⁶** |
| `modele` | PCK, GEPCK, KRG, GEK, PCKRG, HF | GEPCK ajuste un systeme (m+1)x plus grand |
| `n0` | taille du plan initial | **n0 appels solveur** |
| `max_degree`, `q` | base PCE candidate, tri hyperbolique | ajustement seul |
| `global_size` | `global_physical_size` du maillage | **taille RELATIVE**, cf. ci-dessous |
| `geo_min_approx` | `geometric_approximation_min` | secteur angulaire en degres ; sur le Moulin Blanc c'est le seul des deux qui morde |
| `max_size` | borne haute du mailleur | relative elle aussi ; suit `global_size` par defaut |
| `do_EFF` | enrichissement actif | conditionne toute la boucle |
| `n_max_EFF_points` | plafond de points ajoutes | **jusqu'a autant d'appels solveur, en sequentiel** |
| `EFF_criteria`, `tol_BB`, `tol_BS`, `tol_EFF` | critere d'arret de l'enrichissement | decide n_EFF reel |
| `n_NLopt_EFF` | budget NLopt par recherche EFF | 30 evaluations du METAMODELE, pas du solveur |
| `epsilon_factor` | `eps = epsilon_factor · sigma` dans EFF | — |
| `n_batch_EFF` | 1 = sequentiel, >1 = Kriging Believer | >1 permet de paralleliser les appels |
| `eps_taylor` | PCK : points virtuels par Taylor ordre 1 | pas d'appel supplementaire |
| `do_IS`, `n_IS`, `cov_IS` | tirage d'importance | **n_IS evaluations du metamodele** par estimation |
| `n_max_FORM`, `tol_FORM` | iterations FORM | evaluations du metamodele |
| `do_multistart`, `start_from_LHS`, `n_sp` | points de depart FORM | n0+1 ou n_sp departs, metamodele |
| `tol_all_modes` | distance DBSCAN entre deux modes | decide n_modes, donc autant de `run_HF(u*)` |
| `do_warmstart`, `tol_warmstart` | reprise FORM si non convergence | — |
| `do_FORM_filter` | rejeter les u* hors bornes avant DBSCAN | reduit n_modes |
| `exclure_points_non_converges` | rejeter les points que le solveur declare douteux | change ce qui entre au plan d'experiences ; **false par defaut**, cf. ci-dessous |

### Les tailles de maille sont RELATIVES, pas en metres

Le piege coute cher, alors il est ecrit ici en toutes lettres.

`solver/digital_structure.py` ne passe pas `physical_size_type`. Le defaut du
mailleur n'est pas « ignorer la taille » : c'est
`CmnMESH_PhysicalSizeTypeRelative` (`back/rupt/02_CetMESH/CetMESH_SessionAbstract.cpp:140`).
En mode relatif, `RunCadSurf` suffixe la valeur d'un « r » avant de la donner a
MeshGems (ligne 666) : **une fraction de la diagonale de la boite englobante**.

L'echelle du modele decide donc de tout :

| modele | diagonale | `global_size = 0,05` vaut |
|---|---|---|
| flexion pure | quelques metres | quelques centimetres -- le parametre mord |
| Moulin Blanc | **98,1 m** (96,2 x 14,1 x 12,7) | **4,90 m** -- inerte |

Sur le Moulin Blanc, les trois consignes de taille sont hors d'echelle
(`min_size = "-1"` n'est meme jamais transmis, la garde `> 0` l'ecarte). Le
maillage tombe a son **plancher geometrique** : 13 804 tetraedres imposes par
la topologie des faces et la carte de courbure. Mesure du 26/08/2026 :

| `global_size` | `geo_min` | equivalent | tetraedres | duree |
|---|---|---|---|---|
| 0,05 | 4 | 4,90 m | 13 804 | 454 s |
| 0,15 | 20 | 14,71 m | 13 418 | 458 s |
| 0,30 | 35 | 29,42 m | 13 092 | 455 s |

Aucun gain de temps, et les 5 % de tetraedres en moins viennent de
`geo_min_approx`, pas de la taille. **On ne peut pas faire plus grossier.**
Pour piloter reellement la taille sur un grand modele il faudrait passer
`physical_size_type = "absolute"` et donner des metres : cela change le
maillage, donc le resultat -- c'est une decision, pas un reglage.

Cinq autres clefs du dictionnaire (`approach`, `is_iso`, `coeff_on_error`,
`remesh_type`, `old_size_factor`) ne sont lues qu'a partir de l'iteration 1,
par `CetMESH_SessionAnisoRemesh`. La chaine tourne a l'iteration 0 : elles
tombent dans le vide, sans avertissement.

---

## SESSION -- ce run, sur ce poste

Ne change **jamais** le resultat. Depend du nombre de coeurs, de l'espace
disque, de ce qu'on a deja calcule. **N'a rien a faire dans la definition
d'une etude** -- c'est la lecon de `restart_enrich_only`.

| parametre | motif de calcul declenche | remarque |
|---|---|---|
| `n_workers_DOE` | 1 = sequentiel ; >1 lance des sous-processus, un `.ds` copie par worker | ne parallelise QUE le plan initial, pas l'enrichissement |
| `config_is_identical` | relit `doe_cache.json` si present | **economise n0 appels** -- ou masque un plan perime |
| `restart_enrich_only` | repart d'un `restart_state.json` | exige ce dump ; sinon le run s'arrete avec un message |
| `save_history` | copie les sorties SOCP par point | 8,8 Mo par point en flexion, **424 Mo** sur le Moulin Blanc |
| `dossier_sortie` | ou vont figures et journaux | — |

---

## SORTIE -- ce qui est trace

Ne change jamais le resultat. Mais **cinq de ces parametres declenchent des
appels au solveur**, et c'est ce que le resume imprime en tete de journal.

| parametre | motif de calcul declenche | cout Moulin Blanc |
|---|---|---|
| `print_HF` + `n_grid_hf` | grille de contour : n_grid_hf² appels | 15² = 225 -> **29 h** |
| `print_fullHF` | grille complete : n_grid_hf^n_var | 15² en 2D, 3 375 en 3D |
| `do_custom_hf` | grille de points lue dans un fichier | autant d'appels que de points |
| `print_grad_sp` | gradients aux points de depart | un appel par point |
| `print_Pf` | 3 FORM+IS supplementaires **par iteration EFF** | metamodele seul, mais x3 |
| `u1_min/max`, `u2_min/max`, `n_grid` | domaine et finesse des traces | metamodele seul |
| `print_DOE`, `print_3D`, `print_ana`, `print_EFF_progres`, `print_gepck_calls` | figures et traces | gratuit |
| `hf_2d_grid_fixed`, `hf_3d_grid_fixed` | grille pre-calculee, evite de recalculer | **economise** n_grid_hf² appels |

---

## SANS EFFET -- aucun code vivant ne les lit

Mesure du 26/08/2026 : ces quatre noms n'apparaissent que dans le bloc de
liaison des scripts, jamais en lecture. `valider()` refuse de les regler
ailleurs qu'a leur defaut -- un parametre sans effet qui accepte une valeur
laisse croire qu'on a change quelque chose.

| parametre | pourquoi il ne sert plus |
|---|---|
| `reduc_PLS` | composantes PLS du GEK -- le chemin GEK n'est plus cable dans AC3 |
| `do_analytic_grad` | gradients analytiques du GEK -- meme raison |
| `max_of_maxdegree` | plafond de la montee en degre PCE, boucle presente dans AC et AC2, absente de AC3 |
| `seuil_pce` | seuil de validation de l'erreur PCE, lu par AC et AC2 seulement |

**Question ouverte pour Semia et Mohamad** : la boucle de montee en degre a-t-elle
ete abandonnee volontairement entre AC2 et AC3 ?

---

## VARIABLES INTERNES -- l'utilisateur n'y touche pas

Elles restent dans les scripts, en clair. Ce ne sont pas des reglages mais de
l'**etat** remis a zero a chaque run, ou de la **donnee d'etude**.

| variable | role |
|---|---|
| `_eff_history_EFF`, `_BB`, `_BS`, `_theta`, `_Pf`, `_beta_IS` | historiques de l'enrichissement, pour les traces et le dump de reprise |
| `_gepck_pce_label`, `_gepck_loo` | etiquette et erreur LOO du dernier ajustement |
| `_fosm_u0_cache` | evite de recalculer `run_HF([0,0])` pour chaque mode -- **economise n_modes-1 appels** |
| `_point_log_phase`, `_point_log_round` | phase courante, pour le journal incremental |
| `_enrich_round`, `_round_sizes_prev`, `_restart_xt_eff` | reprise d'enrichissement |
| `_socp_call_counter`, `_solveurs` | compteur d'appels et cache de solveurs (un par modele) |
| `slice_def_final` | coupe de trace, reecrite plus bas |
| `PARAM_CONFIG_CAD/LOAD` | **catalogue des variables aleatoires** : lois et regions de sensibilite. C'est la DONNEE de l'etude ; la phase 4 portait sur les reglages, pas sur elle |
| `FY_MEAN` | limite d'elasticite moyenne (Moulin Blanc) |
| `eff_bounds_min/max` | bornes de la recherche EFF. **Ce sont de vrais reglages** et ils ont vocation a rejoindre le schema : ils valent [-7,5 ; +7,5] dans les deux etudes |

---

## Ce que la classification a permis de voir

Elle n'est pas cosmetique. Trois faits n'etaient pas visibles avant :

1. **La figure coute plus que le calcul.** Sur le Moulin Blanc, 225 appels
   pour un contour contre 365 pour le calcul lui-meme.
2. **Un mode de session avait ete fige en definition d'etude**, rendant
   l'etude intransmissible.
3. **Quatre parametres ne servent a rien** et acceptaient pourtant des
   valeurs.

## Ou vit la classification

`_config/schema.py:CATEGORIES` associe chaque champ a `etude`, `session`,
`sortie` ou `sans_effet`, et `COUTE_DES_APPELS_SOLVEUR` recense ceux qui
declenchent des appels. `tests/test_85_configuration.py` exige que **tout**
champ soit classe : en ajouter un sans dire a qui il appartient fait echouer
la suite.

Le resume imprime en tete de chaque journal separe les trois et rappelle ce
qui coute -- avant le run, pas apres.
