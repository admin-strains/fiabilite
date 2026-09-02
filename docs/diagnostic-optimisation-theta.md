# L'optimisation de `theta` : diagnostic, et proposition de correctif

> Instrumentation des 01 et 02/09/2026, branche `cleaning`.
> Tout ce qui suit est **mesuré**. Les hypothèses écartées sont indiquées
> comme telles, avec le chiffre qui les a écartées.

---

## 1. Le point de départ

Les cinq jobs d'intégration continue rendaient cinq `theta` différents sur
les mêmes fichiers de référence. Trois explications ont été proposées puis
réfutées par la mesure avant d'arriver à la bonne.

| hypothèse | verdict | ce qui l'a écartée |
|---|---|---|
| « c'est Linux » | **fausse** | `noyau windows-latest py3.10` reproduit les goldens exactement |
| « c'est la version des bibliothèques » | **fausse** | goldens produits sous numpy 2.1.1, ils passent sous 2.2.6 |
| « c'est `differential_evolution` » | **fausse** | DE rend le **même** `theta` à 7 et à 1 thread, sur les quatre cas |
| « `theta` n'est pas identifiable » | **fausse** (sauf cas linéaire) | `J` a une pente réelle de 3e-04, stable sur trois décades |

---

## 2. La cause

`_lib/kriging.py:kriging_optimize_theta` appelle

```python
minimize(J, theta0, method='L-BFGS-B', bounds=bounds_list,
         options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8})
```

**sans `jac`**. Scipy différencie donc `J` lui-même, avec son pas par défaut
`eps ≈ 1.49e-08`, **absolu**. Or `theta` vit sur `[0.01, 100]` et `J` passe
par une factorisation de Cholesky mal conditionnée.

### 2.1 Le gradient reçu est du bruit

Dérivée de `J` à `theta0 = [47.6737, 21.9981]` :

| pas `h` | `dJ/dtheta_0` | `dJ/dtheta_1` |
|---|---|---|
| **1.49e-08** ← celui de scipy | **1.0283** | **−1.6137** |
| 1.00e-06 | 1.5888e-02 | −4.1327e-02 |
| 1.00e-04 | 3.0482e-04 | 4.8278e-04 |
| 1.00e-03 | 3.1950e-05 | 6.1622e-04 |

Une dérivée honnête a un plateau. La valeur au pas de scipy vaut **3 372
fois** celle obtenue à 1e-04, et change de signe.

Mesure du bruit : perturber `theta` au dernier bit déplace `J` de
**3.05e-08**. Divisé par 1.49e-08, cela donne un gradient parasite de
**2.05** — exactement l'ordre de ce que scipy calcule.

### 2.2 La pépite gouverne ce bruit

`cond(R) ≈ 2.4 / pépite`, et le bruit de `J` suit :

| pépite | cond(R) | bruit de `J` | bruit/pente |
|---|---|---|---|
| 0.0 | 3.75e+14 | 1.04e-03 | 9276 |
| 1e-8 *(actuel)* | 2.40e+09 | 3.05e-08 | 3343 |
| 1e-6 | 2.40e+07 | 5.50e-10 | **13** |
| 1e-5 | 2.40e+06 | 4.94e-11 | **3** |

### 2.3 La conséquence

`ABNORMAL_TERMINATION_IN_LNSRCH` — la recherche linéaire échoue — et
plusieurs appels ne font **aucune** itération. Chaîne de warm-start,
`flexion/PCK` :

```
etape  theta @7 threads      theta @1 thread       arret
ii=1   [47.6737, 21.9981]    [47.6737, 21.9981]    ABNORMAL (nit=3)
ii=3   [47.6737, 21.9981]    [47.6737, 21.9981]    ABNORMAL (nit=0)
ii=4   [47.6737, 21.9981]    [47.6737, 21.9981]    ABNORMAL (nit=0)
ii=5   [47.6737, 21.9981]    [ 6.5457,  6.1283]    ABNORMAL
```

---

## 3. Le défaut est d'origine, et le nettoyage l'a **révélé**

Vérifié contre `8f6e229~1`, avant toute intervention : l'appel à `minimize`,
les bornes `[[0.01]*M, [100]*M]` et `theta0` sont **identiques au caractère
près**. La phase 6 n'a pas touché `kriging_optimize_theta`.

Ce qui a changé est la pépite : `'Nugget': 0.0` à l'origine, `1e-8` depuis la
phase 6 — ajoutée pour corriger les défauts 2 et 3, avec des critères
chiffrés d'avance et tenus.

Appels à L-BFGS-B rendant leur point de départ **inchangé**, mode `optimal` :

| cas | pépite 0.0 | pépite 1e-8 |
|---|---|---|
| flexion/PCK | 8/9 | 3/9 |
| flexion/GEPCK | **9/9** | 2/9 |
| linear/PCK | **3/3** | 0/3 |
| linear/GEPCK | **3/3** | 3/3 |

**23 appels sur 24 immobiles à l'origine.** Le mode `optimal` était inerte :
`theta` valait la sortie de `differential_evolution` et traversait les neuf
étapes de la chaîne sans être touchée. Déterministe, donc reproductible — et
**jamais optimisée**.

En divisant le bruit par cinq ordres de grandeur, la pépite a rendu le
gradient *parfois* exploitable et L-BFGS-B s'est mis à avancer, vers des
points qui dépendent de l'arithmétique. L'irreproductibilité est le
sous-produit d'une correction légitime.

---

## 4. Le piège : reproductible n'est pas juste

Trois correctifs candidats ont été mesurés. `J` est **minimisée**.

| combinaison | `J` atteint | LOO | interpolation |
|---|---|---|---|
| 1e-8 / défaut *(actuel)* | **−910.51** | 7.66e-11 | 1.03e-08 |
| 1e-6 / pas relatif | −908.98 | 1.10e-10 | 7.40e-08 |
| 1e-6 / log-`theta` | −774.76 | 2.86e-09 | 1.65e-05 |
| 1e-8 / log-`theta` | −770.38 | 2.12e-09 | 1.36e-05 |

`log-theta` donnait la meilleure reproductibilité de toutes — **1.77e-11**,
et zéro échec de recherche linéaire. Il converge proprement **vers un point
136 unités pire**, et le métamodèle suit : LOO ×37, interpolation ×1600,
au-delà du critère de 1e-6 fixé en phase 6.

**Un correctif doit donc être jugé sur `J` et sur la qualité du métamodèle,
pas sur la reproductibilité.** Celle-ci s'achète en convergeant
systématiquement au mauvais endroit.

---

## 5. Proposition : supprimer le gradient plutôt que le réparer

Le défaut est dans le gradient. `differential_evolution` n'en utilise pas, et
la mesure montre qu'il rend le **même `theta` à 7 et à 1 thread sur les
quatre cas**. On le met donc à chaque troncature LARS, à la place du
L-BFGS-B warm-starté.

| | actuel | DE partout |
|---|---|---|
| **θ, 7 vs 1 thread** | | |
| flexion/PCK | 9.71e-01 | **2.01e-04** |
| flexion/GEPCK | 3.54e-02 | **1.59e-04** |
| linear/PCK | 7.32e-01 | 1.01e+03 |
| linear/GEPCK | 0.00e+00 | 4.46e-01 |
| **LOO** | | |
| flexion/PCK | 1.341e-09 | **5.051e-10** |
| flexion/GEPCK | 7.658e-11 | 1.469e-10 |
| linear/PCK | 2.907e-25 | **7.182e-31** |
| linear/GEPCK | 6.338e-26 | **9.464e-28** |
| **interpolation** | | |
| flexion/GEPCK | 1.03e-08 | **8.02e-10** |
| linear/PCK | 2.01e-16 | **0.00e+00** |
| **appels à `J`** | 2 235 | 6 588 |

**Le métamodèle est meilleur dans trois cas sur quatre**, spectaculairement
sur les deux cas linéaires, et la reproductibilité gagne trois à quatre
ordres sur les cas non dégénérés.

### Ce qu'il reste à mesurer avant d'appliquer

1. **Le coût à l'échelle réelle.** ×3 en appels à `J` sur un plan de 24
   points est indolore ; sur un plan de 360 points, où `R` est 360×360 (et
   1080×1080 en GEPCK), il faut le chiffrer.
2. **L'effet sur `beta` et `Pf`** à travers la chaîne complète — c'est le
   seul chiffre qui intéresse une étude.
3. **Les cas linéaires**, où la reproductibilité se dégrade. `theta` y est
   sans objet (la PCE représente l'état limite exactement, LOO ~1e-25) mais
   il faut le dire plutôt que le taire.

### Ce que ce correctif n'est pas

Ce n'est pas le correctif de fond. Le correctif de fond est un **gradient
analytique** de `J`, qui supprimerait la cause au lieu de la contourner. Il
demande d'écrire `dJ/dtheta` — un travail réel, à chiffrer séparément.

---

## 6. Traçabilité

| commit | ce qu'il change |
|---|---|
| `757a8e7` | bridage BLAS à un thread pendant les tests, goldens régénérés |
| `fa57dc4` | ⚠️ **conclusion fausse**, corrigée par le suivant |
| `9ed88aa` | la cause mesurée : le gradient est du bruit |
| `ac6fb8a` | le défaut est d'origine ; le nettoyage l'a révélé |

Aucun de ces commits ne modifie le code de calcul. Les témoins vivent dans
`tests/test_31_theta_non_identifiable.py`.

---

## 7. Le correctif appliqué — et une erreur de ma part, corrigée

Le gradient analytique de `J` a été écrit puis vérifié par différences finies,
pour **PCK** (commit `29c2e3d`) puis pour la Gram **augmentée de GEPCK**
(commit `6465c95`).

### Ce qui a rendu la dérivation de GEPCK sûre

Le noyau est séparable, donc chaque bloc de `R̃` est un produit sur les
dimensions et `d(bloc)/dθ_m` ne remplace **qu'un seul facteur**. Ce ne sont
pas douze formules par famille mais quatre fonctions scalaires — `u`,
`∂u/∂x₁`, `∂²u/∂x₁∂x₂`, et leurs dérivées en θ — plus une règle de produit
unique. Vérification sur la matrice entière, deux familles, deux dimensions,
trois échelles de θ : **écart maximal 5,62e-10**.

### ⚠️ Une réserve que j'ai écrite et qui était FAUSSE

Le message du commit `6465c95` dit : « cette baseline tourne en PCK ; ce que
le correctif fait à beta sur le Moulin Blanc, en GEPCK, n'est pas mesuré ».

**C'est faux.** `tools/baseline_run.py` porte son *propre* `CONFIG`,
indépendant des fichiers d'étude, et il déclare `"modele": "GEPCK"` —
`fit_gepck` est bien appelé, ligne 149. J'avais déduit le modèle de
`studies/pure_flexion_analytique.toml` sans vérifier que la baseline le
lisait. Elle ne le lit pas.

La mesure de bout en bout était donc **déjà** une mesure GEPCK :

| | avant | après |
|---|---|---|
| β | 4.77492513586 | 4.77492176778 |
| Pf_FORM | 8.98870596e-07 | 8.98885640e-07 |
| Pf_IS | 1.46965520e-06 | 1.46968341e-06 |
| durée | 4,8 s | 3,9 s |

### Ce qui reste réellement non mesuré

Le **Moulin Blanc**. Là ce n'est pas le métamodèle qui change de nature, mais
la taille du plan, la dimension, et un état limite qui vient d'un solveur et
non d'une formule. Aucune baseline analytique ne le prédit.

### Sur la baseline elle-même, une bonne nouvelle

Il n'y a pas de baseline GEPCK moins chère à construire : elle existe.
`baseline_run.py` instancie `FlexionLS` — section rectangulaire BA en flexion
simple, `M_R = A·fy + B·fy²/fc`, `Med = F·L` — avec la géométrie réelle de
`test_pure_flexion` (b = h = 0,80 m, L = 5,00 m, 24 HA32). C'est une console,
elle porte un **oracle** (`beta_exact()`, Brent à 1e-12, sans métamodèle ni
FORM), et elle tourne en **3,9 s** sans Digital Structure.

Ce qui manquerait est un **second** cas GEPCK, différent : un seul cas peut
cacher un défaut — `linear` et `flexion` se comportent d'ailleurs de façon
opposée sur tout ce diagnostic.

---

## 8. Le « choix du bassin » : ce que c'était vraiment

Le gradient analytique a rendu `theta` reproductible **à bassin donné**. Il ne
garantit pas *quel* bassin est atteint. Mesure du 02/09/2026 sur
`flexion/PCK`, même code, même plan :

| plateforme | θ | LOO |
|---|---|---|
| poste de référence (windows) | [0.3847 ; 100.0] | 1.265825e-09 |
| runner ubuntu py3.10 | [0.3847 ; 100.0] | 1.265825e-09 |
| runner windows py3.10 et py3.13 | [0.3847 ; 100.0] | 1.265825e-09 |
| **runner ubuntu py3.13** | **[0.0100 ; 6.55]** | **3.171187e-09** |

Quatre plateformes sur cinq trouvent le meilleur optimum, une trouve l'autre.
Les deux sont d'excellents métamodèles — 1e-09 des deux côtés, un facteur 2,5
entre eux — mais ce ne sont pas les mêmes.

**Ce n'est ni une tolérance trop serrée ni un défaut de code.** C'est un choix
discret, en amont de `theta` : il se joue dans `differential_evolution` ou
dans la chaîne de warm-start LARS, avant que L-BFGS-B n'affine.

`tests/test_30_surrogate_golden.py` le déclare
(`BASSIN_DEPENDANT_DE_LA_MACHINE`) et vérifie sur ce cas la **qualité** du
modèle — LOO sous 1e-8, ce qui est vrai des deux côtés — au lieu de le
comparer au golden. Il ne tranche pas la question.

### Fermée le 02/09/2026 au soir — et ce n'était pas un bassin

La mesure prévue ci-dessus a été faite, autrement : plutôt que d'instrumenter
deux plateformes, on relance l'optimiseur depuis **64 points de départ** sur
les paramètres du dernier ajustement, et on regarde où il tombe.

Il n'y a pas deux optima. Il y en a **au moins quatre** :

| θ atteint | J | départs (sur 64) | LOO du métamodèle |
|---|---|---|---|
| **[0.6873 ; 2.3884]** | **−314.012** | **22** | **5.047e-10** |
| [0.3847 ; 100.0] — *celui du golden* | −311.235 | 15 | 1.266e-09 |
| θ₁ = 0.01, θ₂ quelconque | −300.784 | 21 | 3.171e-09 |
| [17.9 ; 100] et [100 ; 100] | −285.97 / −269.94 | 6 | — |

**Le troisième n'est pas un bassin : c'est le BORD.** Le long de θ₁ = 0.01 —
la borne inférieure — J est constant au neuvième chiffre :

    theta = [0.01,   0.50]   J = -300.783830922
    theta = [0.01,   6.55]   J = -300.783830920
    theta = [0.01, 100.00]   J = -300.783830920

θ₂ y est **complètement non identifiable**. La première longueur de
corrélation s'effondre à sa borne, et la seconde cesse d'avoir un effet. Le
`[0.0100 ; 6.55]` du runner ubuntu py3.13 est donc un point ARBITRAIRE sur une
crête plate — pas le fond d'un second bassin. C'est ce qui explique qu'une
seule plateforme sur cinq le rapporte, et qu'elle rapporte *cette* valeur de
θ₂ plutôt qu'une autre.

### La question qui la remplace, et qui est plus intéressante

**La chaîne ne trouve pas le meilleur optimum.** Elle converge en dix
troncatures LARS, chacune repartant de la précédente, et J descend
régulièrement de −162,98 à −311,24 :

    de          [1.00, 1.00]        -> [47.67, 22.00]     J = -162.977
    gradbased   [47.66, 21.98]      -> [12.16, 20.51]     J = -190.818
    ...
    gradbased   [ 2.72,  2.19]      -> [ 0.3847, 100.0]   J = -311.235

Or un optimum à **J = −314,012** existe, et 22 des 64 départs froids le
trouvent — plus que tout autre. Et il ne s'agit pas seulement de
vraisemblance : c'est aussi un **meilleur métamodèle**.

| | LOO PCK | LOO GEPCK | erreur aux points sonde (PCK) |
|---|---|---|---|
| chaîne (golden) | 1.266e-09 | 4.956e-09 | 2.50e-05 |
| meilleur J | **5.047e-10** | **6.827e-10** | **1.53e-05** |

Soit **2,5 fois** mieux en PCK et **7 fois** mieux en GEPCK.

**Ce n'est pas une correction à appliquer sans arbitrage.** Chercher plus
largement — un multi-start, ou un `differential_evolution` à chaque
troncature plutôt qu'au seul premier ajustement — déplacerait `theta`, donc
tous les goldens, donc `beta`. C'est un changement de classe « phase 6 », qui
demande la procédure de `CONTRIBUTING.md` : démontrer que le nouveau
comportement est MEILLEUR, chiffres à l'appui, avant de régénérer quoi que ce
soit. Et une réserve mesurée va dans l'autre sens : sur GEPCK, l'erreur aux
points sonde se DÉGRADE (1,60e-05 → 2,65e-05) alors que le LOO s'améliore de
sept fois — les deux critères ne disent pas la même chose, et il faudrait
savoir lequel suivre avant de changer quoi que ce soit.

Le coût, lui, est connu : le balayage ci-dessus a pris quelques secondes sur
un plan de 24 points, mais l'optimisation de `theta` est appelée une fois par
troncature LARS et le Moulin Blanc en compte deux — donc peu — tandis que la
flexion pure en compte neuf.
