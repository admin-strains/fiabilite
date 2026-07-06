# Découplage de la position critique — théorie complète, de zéro

**2026-07-06, MM (synthèse d'une discussion de fond).**
Objectif : expliquer entièrement pourquoi et comment on trouve la position critique `s*`
d'une charge mobile *en dehors* du problème de fiabilité, avec la **justification de chaque
équation** (pourquoi on a le droit de l'écrire), et **toutes les options de résolution**.

Convention d'écriture : maths en ASCII (le terminal ne rend pas le LaTeX). `lambda`,
`theta`, `beta`, `sigma` = lettres grecques ; `*` = optimal ; `^T` = transposé.

---

## PARTIE I — Le décor (les objets et les variables)

### I.1 Ce qu'on calcule : l'indice de fiabilité `beta`
On veut la probabilité qu'une structure ruine, compte tenu des incertitudes. On la résume
par `beta` (indice de fiabilité) : plus `beta` est grand, plus c'est sûr. `Pf ~ Phi(-beta)`.

### I.2 Le multiplicateur de ruine `lambda`
Le solveur STRAINS (analyse limite) sort, pour une structure + un chargement donnés :
```
   lambda = facteur par lequel on peut multiplier la charge avant ruine
   lambda > 1 : tient    lambda = 1 : exactement a la ruine    lambda < 1 : deja ruine
```
`lambda = 1` définit l'**état-limite** (la frontière ruine / pas-ruine).

### I.3 LA distinction fondatrice : incertitude vs pire-cas
Deux natures de paramètres, à ne jamais confondre :
```
   theta = INCERTITUDE                       s = PIRE CAS (choix)
   (fy resistance, q magnitude, densite)     (position d'une charge mobile)
   "je ne connais pas la vraie valeur"       "la charge se met au plus defavorable" (EC1)
   -> on veut sa PROBABILITE -> on INTEGRE   -> on veut son MINIMUM -> on OPTIMISE puis FIGE
```
**Pourquoi cette distinction est légitime :** une incertitude est une propriété *de la nature*
qu'on ne maîtrise pas → il faut l'intégrer sur sa loi pour sortir une proba. Une position de
pire-cas est un *choix de dimensionnement* (le règlement EC1 dit : place la charge au plus
défavorable) → ce n'est pas une variable aléatoire, c'est le résultat d'une optimisation.
Figer une incertitude n'a pas de sens (détruit la proba) ; figer un pire-cas à son argmin
est exactement ce que demande le règlement.

---

## PARTIE II — Le SOCP d'analyse limite (ce qu'est `lambda(s)`, rigoureusement)

### II.1 Le théorème cinématique
Pour une position `s` FIXÉE, `lambda(s)` est la solution de :
```
   lambda(s) =  min      D(u)                     (1) minimiser la dissipation plastique
                 u
                s.c.     Fext(s)^T u = 1           (2) normalisation : travail externe = 1
                         B u = eps                  (3) compatibilite (deformations)
                         eps in K(fy)               (4) admissibilite plastique (cone SOCP)
```
**Pourquoi on peut écrire chaque ligne :**
- (1) `u` = champ de vitesses d'un mécanisme d'effondrement. Le théorème cinématique de
  l'analyse limite dit : le vrai facteur de ruine = le **min** sur tous les mécanismes
  admissibles de la dissipation. C'est un théorème de borne supérieure (Koiter).
- (2) On ne peut pas minimiser la dissipation « en absolu » (u=0 donne 0). On **normalise** le
  travail des charges externes à 1. C'est légitime car le problème est **homogène de degré 1
  en u** : multiplier `u` par un scalaire multiplie D et W par ce scalaire → le rapport D/W
  est invariant → on peut fixer l'échelle par `W=1`. `Fext(s)` = vecteur des forces à la
  position `s` (c'est le SEUL terme qui bouge avec `s`).
- (3) Les déformations dérivent des vitesses (opérateur de compatibilité `B`).
- (4) Le critère de plasticité (Mises, Rankine, béton…) définit un **cône convexe** `K` de
  déformations admissibles, dont la taille est fixée par `fy`. C'est ce qui fait du problème
  un **SOCP** (Second-Order Cone Program) — convexe, donc minimum global garanti.

### II.2 La sensibilité `dlambda/ds` (gratuite à chaque SOCP)
```
   dlambda/ds = - Lam . (dFext/ds)^T . u*         (5)
```
**Pourquoi on peut l'écrire (théorème d'enveloppe / Danskin) :** à l'optimum, la dérivée de
la valeur `lambda` par rapport à un paramètre = la dérivée **partielle** du Lagrangien, en
**gardant la solution optimale `u*` figée** (les termes en `du*/ds` s'annulent par
optimalité — KKT). `Lam` = multiplicateur de la contrainte de normalisation (2), `dFext/ds`
= comment le vecteur de charge bouge quand la charge se déplace (connu analytiquement via les
fonctions de forme de l'empreinte). **Validité : locale**, tant que l'ensemble actif (le
mécanisme) ne change pas — c'est le point central de la Partie III. Validée numériquement à
0.11% (FD 2e ordre).

---

## PARTIE III — Pourquoi la position se découple (le coeur théorique)

### III.1 `lambda` est un rapport dissipation / travail
Bilan d'énergie du mécanisme à la ruine (charge live + poids propre) :
```
   lambda * W_live(s) + W_dead = D         (bilan energetique du mecanisme)
   =>  lambda(s) = ( D - W_dead ) / W_live(s)      (6)
```
**Pourquoi :** c'est (1)-(2) réécrit. D = dissipation interne, W_live = travail de la charge
mobile (par unité de multiplicateur), W_dead = travail du poids propre (fixe).

### III.2 Séparabilité : matériaux au numérateur, position au dénominateur
```
   D        = fy * g            (dissipation lineaire en fy, g = geometrie du mecanisme)
   W_live   = q * w(s)          (magnitude q * facteur de travail geometrique w(s))
   W_dead   = constante
   =>  lambda(s, theta) = ( fy*g - W_dead ) / ( q * w(s) )
                        = [ (fy*g - W_dead)/q ] * [ 1/w(s) ]
                        = a(theta) * h(s)                          (7)   avec a>0
```
**Pourquoi on peut écrire la forme séparée `a(theta)*h(s)` :** parce que `theta` (fy, q,
poids propre) n'apparaît QUE dans le numérateur (dissipation) et `s` QUE dans le dénominateur
(travail géométrique). C'est ça, la **séparabilité** — la condition PRINCIPALE. Elle ne vient
PAS de la linéarité : même `D = fy^2 * g` se sépare (`a = fy^2/q`). Elle vient de ce que
matériaux et position agissent sur des termes physiquement distincts.

### III.3 L'invariance de `s*`
```
   s* = argmin_s [ a(theta) * h(s) ]  =  argmin_s h(s)            (8)
```
**Pourquoi :** `a(theta) > 0` MULTIPLIE la courbe (l'étire verticalement) et un éventuel
terme additif `b(theta)` la DÉCALE — ni l'un ni l'autre ne déplacent le point le plus bas.
Donc `s*` ne dépend que de `h(s)` = la GÉOMÉTRIE, pas de `theta`. **C'est le permis de
figer `s*`.** Intuition : rendre le béton plus fort renforce tout le pont du même facteur →
ne change pas OÙ est le point faible.

### III.4 Le vrai `lambda` est une ENVELOPPE de mécanismes
La forme (7) n'est valable que POUR UN mécanisme. Le vrai `lambda` = min sur tous :
```
   lambda(s, theta) = min_k [ a_k(theta) * h_k(s) ]              (9)
                       \___/
              le "min" = selection du mecanisme actif k*(s,theta)
```
**Pourquoi :** le théorème cinématique (1) minimise sur TOUS les mécanismes. Chaque mécanisme
`k` a ses propres constantes géométriques (`a_k`, `h_k`). Chaque terme est séparable →
`argmin_s` invariant. MAIS le `min` de produits séparables N'EST PAS séparable : l'indice
gagnant `k*(s,theta)` dépend de `s` ET de `theta`. **C'est LUI le coupable** qui recouple.

### III.5 Hiérarchie des conditions (ce qui fait vraiment marcher l'invariance)
```
   (1) SEPARABILITE (theta au numerateur, s au denominateur)  = condition PRINCIPALE
                     -> donne la forme a(theta)*h(s)
   (2) ENSEMBLE ACTIF STABLE (meme k*, memes multiplicateurs)  = le VRAI RISQUE
                     -> garde h_k(s) de forme fixe
   (3) linearite/homogeneite degre 1 en fy                     = apporte SEULEMENT
                     -> l'exactitude du SCALING UNIFORME (q, tous fy ensemble)
                        PAS l'invariance generale
```
Reformulation exacte : **l'invariance de `s*` = validité locale du théorème d'enveloppe =
stabilité de l'ensemble actif des multiplicateurs de Lagrange.**

### III.6 Deux bascules de mécanisme à ne pas confondre
```
   bascule le long de s (theta fixe)        bascule le long de theta (s fige)
   d(mecanisme)/ds                          d(mecanisme)/dtheta
   -> rend lambda(s) BOSSELEE (multimodale) -> fait SAUTER s*
   -> ATTENDUE, geree par le balayage       -> LE risque pour le figeage
```
**Conséquence importante :** si les matériaux sont FIXES, il n'y a pas de `theta` variable →
**aucun risque d'invariance** ; `s*` est juste un nombre à trouver, et la bascule le long de
`s` est bénigne (c'est la multimodalité normale).

---

## PARTIE IV — LE problème à résoudre pour `s*`

### IV.1 Formulation bi-niveau
```
   beta = min ||u_std||   s.c.  [ min_s lambda(s, theta(u_std)) ] = 1     (10)
           u_std                  \______ niveau interne ______/
          \___ externe (FORM) ___/
```
### IV.2 Découplé (grâce à l'invariance III.3)
```
   PROBLEME 1 (amont, UNE fois) :   s* = argmin_{s in [smin,smax]} lambda(s)   (11)
                                    condition interieure : dlambda/ds = 0
   ===== FIGER s = s* =====
   PROBLEME 2 (fiabilite) :         beta = min ||u_std|| s.c. lambda(u_std)=1,
                                           s = s* constant, u_std=(fy,q)        (12)
   VERIF finale (gratuite) au u* :  |dlambda/ds(u*)| * L / lambda(u*) <= eps   (13)
```
**Pourquoi on a le droit de séparer (10) en (11)+(12) :** exactement l'invariance (8) — `s*`
ne bouge pas quand `theta` varie, donc le niveau interne se résout une fois pour toutes au
lieu d'être ré-imbriqué à chaque pas. (13) vérifie a posteriori que l'hypothèse a tenu.

---

## PARTIE V — Les options de résolution du Problème 1 (justifiées)

### Option A — Balayage + sécante (naïf, RECOMMANDÉ pour démarrer)
```
   1) balayage grossier de s (3-5 pts/travee, TOUJOURS les 2 abouts)
      chaque SOCP -> (lambda, dlambda/ds) -> interpolation de Hermite -> reperer le bon creux
   2) secante sur dlambda/ds dans les 1-2 meilleurs creux :
         s_{k+1} = s_k - dlambda/ds(s_k) / [pente secante]
   3) s* = meilleur creux            (~10-20 SOCPs, UNE fois)
```
**Pourquoi ça marche :** balayage = gère la multimodalité (III.6, ~16 creux) ; sécante =
exploite la dérivée analytique (5) pour converger vite DANS un creux (où l'ensemble actif est
stable, donc dlambda/ds fiable). **Coût : ~1% du calcul de fiabilité total** (des centaines de
SOCPs) → déjà pratique. **Effort : quasi nul** (réutilise lambda + dlambda/ds déjà validés).

### Option B — Programmation paramétrique (formule fermée, optimisation)
Idée : à mécanisme figé, `lambda(s)` est EXPLICITE, plus besoin de re-solver.
```
   Un SOCP en s0 -> donne le mecanisme u* (la "forme") et la dissipation D(u*).
   Tant que u* reste optimal (region critique) :
        lambda(s) = D(u*) / ( Fext(s)^T u* )      (14)  FORMULE FERMEE en s
        (Fext(s) connu par les fonctions de forme -> on GLISSE sans re-solver)
        derivees a tous ordres gratuites (Newton quadratique dispo)
   Re-solver SEULEMENT aux FRONTIERES (changement de mecanisme).
   => nombre de SOCPs = nombre de MECANISMES (~16), pas nombre de points de balayage.
```
**Pourquoi on peut écrire (14) :** par homogénéité, `lambda(s) = min_u D(u)/W(s,u)` ; le `u*`
optimal reste optimal sur tout un voisinage de `s` (région critique) ; dans cette région,
avec `u*` figé, `lambda(s) = D(u*)/W(s,u*)` est explicite car `Fext(s)` est une fonction
connue de `s`. Cohérent avec (5) : dériver (14) redonne exactement `dlambda/ds =
-lambda * W'(s)/W(s)`.

#### Les 3 tests de franchissement de frontière (quand re-solver ?)
```
   TEST 1 predictor-corrector (ROBUSTE, defaut) :
        glisser avec (14), faire un vrai SOCP de controle de loin en loin,
        comparer |lambda_SOCP - lambda_formule|/lambda <= tol ?
        decroche -> on a change de mecanisme -> re-solver.
        + : ne demande rien de plus (compare des VALEURS convergees, nettes).

   TEST 2 marges + multiplicateurs (EXACT en theorie, FRAGILE avec l'IPM) :
        marge de plasticite min -> 0  (nouvelle rotule)  OU  mu_j -> 0 (rotule ferme).
        - : l'IPM n'atteint pas l'ensemble actif exact ; degenerescence a la bascule ;
            bruit de maillage ; faux positifs par tangence. Meme famille que le probleme
            "pas de certificat infeasible/unbounded" de STRAINS. A COUPLER au Test 1
            (alerte) ou ajouter un CROSSOVER simplexe pour le rendre exact.

   TEST 3 croisement des mecanismes concurrents (GLOBAL, multimodalite) :
        garder les 2-3 meilleures formes ; resoudre analytiquement ou
        const_1/delta_1(s) = const_2/delta_2(s) -> position de bascule sans SOCP.
        gere les sauts vers une forme LOINTAINE (pas voisine).
```
**Verdict tests :** Test 1 le plus sûr (valeurs nettes) ; Test 2 exact sur le papier mais pas
robuste avec le point intérieur de STRAINS sans crossover ; Test 3 pour la couverture globale.

### Option C — Bi-niveau imbriqué complet (référence, trop cher)
Ré-résoudre `min_s` à CHAQUE pas du FORM externe. Correct mais coûteux et instable. Sert de
référence conceptuelle ; on ne l'implémente pas car l'invariance rend A/B suffisants.

### Option D — Garder `s` en variable aléatoire Uniform (À NE PAS FAIRE)
Mettre `s` dans `u_std` avec une loi Uniform. **Échoue** : `lambda(s)` bosselée (III.6) casse
FORM (vécu 30/06, `u_s±3` diverge) et biaise le surrogate (`beta=2.10` vs vrai ~3.5). De plus,
sémantiquement **faux** pour un pire-cas EC1 (dilue le pire cas, non conservatif).

---

## PARTIE VI — Recommandation, décisions, extensions

### VI.1 Pipeline recommandé
```
   Phase 1 : Option A (balayage Hermite + secante) + Test 3 (croisement)
             -> robuste, code minimal, reutilise l'existant. Suffisant (1% du cout).
   Phase 2 (si n charges mobiles -> n-D coute cher) : Option B (formule fermee + Test 1).
```

### VI.2 Décision sémantique (à trancher avec Xavier / Agnès)
```
   pire-cas EC1 (le convoi se met au plus defavorable) -> bi-niveau (11)+(12) = CORRECT ;
                                                          Uniform serait FAUX.
   position vraiment aleatoire (statistique trafic)     -> figer = CONSERVATIF (+1% beta MB).
```
Dans les deux cas, découpler est justifié ; la seule question est l'écart (conservatif) qu'on
accepte si l'interprétation est « aléatoire ».

### VI.3 Plusieurs charges mobiles (n positions)
```
   positions <-> materiaux : DECOUPLABLE (meme invariance, positions au denominateur)   OUI
   positions <-> positions  : COUPLEES entre elles -> optimiser le VECTEUR (s1*..sn*)
                              conjointement (elles interagissent). Balayage n-D, cout monte,
                              multimodalite ~creux^n. -> preferer un convoi RIGIDE (n->1)
                              quand le reglement l'autorise.
```
**Faisable** dans tous les cas ; seul le coût du Problème 1 monte avec `n`.

### VI.4 Garde-fous permanents
```
   - balayage TOUJOURS aux 2 abouts (2 quasi-optima symetriques sur MB, beta 2.43 vs 2.48).
   - surveiller l'ecart entre les 2 meilleurs creux (indicateur de bascule le long de theta).
   - check KKT (13) au point de conception -> confirme que s* n'a pas bouge.
   - se tromper de 2 m sur x fige coute +17% sur beta (MB) -> raffiner le creux finement.
```

---

## Preuves empiriques (déjà faites, cf. mémoire)
```
   Moulin Blanc (grille HF fy1 x s, 182 pts) : Delta x* = 5 mm sur fy +-2 sigma ;
        invariance vs q EXACTE (meme point de grille) ; figer -> +1% beta (conservatif) ;
        ~16 creux + 2 quasi-optima symetriques aux abouts.
   Cantilever (35 SOCPs) : Delta s* = 0.000 sur 15 configs (fy x pression x dead) ;
        recherche gradient 2 SOCPs vs 7 ; dlambda/ds analytique plus exact que la FD.
```

## Banc A vs B sur cantilever (2026-07-06, tests/bench_optA_vs_optB_cantilever.py)
CORE PARTAGE, mesh 0.10, empreinte 0.6x0.3 mobile, course [0.8, 3.2].
```
   OPTION A (secante/gradient) : s* = 3.200  lam = 2.339  mode edge_droit  -> 2 SOCPs
   OPTION B (formule fermee)   : s* = 3.104  lam = 2.415  mode converge     -> 2 SOCPs
   Delta s* = 0.096 m   |   total 4 SOCPs   |   5 s
```
Enseignements : (1) B PREDIT s* depuis 1 SEUL SOCP (ancre s0=2.0 -> lit u* dans PL_cin_out.msh
-> W(x)=|vz moyen empreinte| MONOTONE croissant -> argmax=3.10) ; le 2e SOCP ne fait que
VERIFIER. (2) DRIFT MESURE = 2.51% : lam_formule(3.104)=2.355 vs lam_vrai=2.415 -> c'est
l'erreur tangente SOCP (CAS 2, angles d'ouverture qui derivent un peu du mecanisme d'ancre),
PETITE ici -> proche du CAS 1 -> corrector l'accepte (<3%). Demonstration numerique concrete
de la theorie. (3) LIMITE du test : cantilever = lambda(s) MONOTONE -> s* au BORD -> les deux
trouvent trivialement en 2 SOCPs, n'exerce NI la secante de A (pas de min interieur) NI le
re-solve multi-mecanisme de B. Pour differencier vraiment : cas a MINIMUM INTERIEUR (poutre
bi-encastree / sur 2 appuis, pire a mi-portee). Gotcha impl : le mecanisme u* n'est ecrit
(PL_cin_out.msh) que si write_debug_files='true' (doGmsh) -> param write_debug ajoute a
pl_harness.run_config (additif).

## Banc bi-encastre (minimum INTERIEUR) — LE resultat decisif (tests/bench_biencastre.py)
Poutre encastree aux 2 bouts, mesh 0.10, empreinte mobile, course [0.8, 3.2]. Min vrai a
mi-portee ~2.04 (symetrie).
```
   OPTION A (secante vrai dl/ds) : s*=2.041  lam=30.30  [CORRECT]   5 SOCPs
   OPTION B naif                 : s*=1.304  lam=33.93  [PIEGE]     2 SOCPs
   OPTION B + KKT gate           : s*=1.304  lam=33.93  [BLOQUE]    3 SOCPs
```
**B ECHOUE. Raison mecanique fondamentale = MECANISME QUI SUIT LA CHARGE (load-following).**
Bi-encastre : la rotule centrale se forme SOUS la charge -> le mecanisme@x deflechit le plus
EN x -> argmax_s W(mecanisme@x)(s) = x TOUJOURS -> la formule fermee lam(s)=lam0*W(s0)/W(s) a
son max A L'ANCRE -> B est un POINT FIXE partout -> ne glisse pas -> ne trouve jamais le vrai
min a 2.04. Le KKT gate (|dl/ds|*L/lam=0.637 >> 0 en 1.304) DETECTE que ce n'est pas un min
mais s_pred==ancre -> BLOQUE (ne peut pas reparer). Le B naif se PIEGE silencieusement (rel
1.45% < tol car le controle est PRES de l'ancre, meme mecanisme) -> annonce s*=1.304 FAUX
(erreur 0.74 m, lam 12% trop haut).

CANTILEVER (banc precedent) marchait car rotule a l'ENCASTREMENT (x=0), mecanisme GLOBAL
INDEPENDANT de la charge -> W(x) profil fixe monotone -> argmax=bord -> B predit d'1 SOCP.

**DISTINCTION PROPRE (reformule CAS 1/CAS 2 en termes mecaniques) :**
```
   mecanisme INDEPENDANT de la position charge (rotules fixes aux appuis, ex. cantilever)
       -> W(x) profil fixe -> Option B MARCHE (predit s* d'1 SOCP)                    = CAS 1
   mecanisme QUI SUIT la charge (rotule sous la charge, ex. bi-encastre, cas COURANT
   d'une charge mobile sur structure flexible)
       -> argmax W(mech@x) = x -> Option B POINT FIXE, ne glisse pas, ECHOUE          = CAS 2 extreme
```
**RECO REVISEE : Option A (secante sur le VRAI dl/ds) est robuste dans TOUS les cas** (elle
utilise le gradient exact qui pointe vers le min quel que soit le type de mecanisme).
**Option B n'est viable QUE pour des mecanismes globaux independants de la charge** (rare pour
une charge mobile). => pour le decouplage reel (convois mobiles sur tablier), utiliser
l'Option A. B abandonnee sauf structures a mecanisme fixe. Bonus utile : le check KKT
|dl/ds|*L/lam au point de conception (deja prevu, gratuit) distingue proprement un vrai min
d'un point fixe parasite.

## Liens mémoire
project_decouplage_position_critique_2026-07-04 ; project_sensibilite_position_charge_2026-06-29 ;
project_charge_ponctuelle_design_2026-07-03 ; project_socp_infeasible_unbounded (limite IPM,
pertinent pour le Test 2) ; project_charge_mobile_archi_lumping_2026-06-30 (pathologies Option D).
```
