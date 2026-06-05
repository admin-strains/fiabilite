# Resume global de session — Fiabilite flexion pure BA
**Date :** 2 juin 2026
**Couvre :** Etat du code AC2_pure_flexion.py, protocole de lancement, surveillance, remplissage des .md

---
## INSTRUCTIONS DE LANCEMENT OBLIGATOIRES

**Etape 1 — Obtenir l'heure exacte (dans un appel bash separe) :** (le launcher/ autres noms de fichiers .py dépendent du projet et sont précisés par le contexte / l'utilisatrice)
```bash
date +%d%m_%H%M
```
Note le nom du fichier output : `output_DDMM_HHMM.txt`.

**Etape 2 — Lancer en arriere-plan :**
```bash
cd "C:\_workingDir\_SF\test flexion"
python launcher2.py > "output/output_DDMM_HHMM.txt" 2>&1
```
Utiliser `run_in_background=True`.

**Etape 3 — Lancer immediatement la surveillance (run_in_background=True) :**
```bash
cd "C:\_workingDir\_SF\test flexion"
until grep -q "beta " "output/output_DDMM_HHMM.txt" 2>/dev/null; do sleep 15; done && grep "beta\|Pf\|u\*\|Imp\.\|Design\|iterations\|Warm start\|FOSM\|dg/du\|Erreur" "output/output_DDMM_HHMM.txt"
```


## PARTIE NON OBLIGATOIRE - ATTENDRE QUE LUTILISATRICE DEMANDE DE LES SUIVRE SPECIFIQUEMENT. 
**Etape 4 — Pendant que le run tourne, lire les options :**
Lire le bloc OPTIONS d'AC2_pure_flexion.py (lignes ~68–136) pour connaitre :
- `modele` actif
- `n0`, `do_EFF`, `do_warmstart`, `do_multistart`
- `U_doe_fixed` (None = DOE aleatoire ; sinon hardcode)

Cette lecture sert a interpreter les resultats quand ils arrivent — pas a bloquer le lancement.

---
## 0. Point d'entree

Ce fichier est le document de reference pour se remettre a jour en debut de session.

**Dossier resultats actif (test flexion) :** `C:\_workingDir\_SF\test flexion\2026_1205 résultats et comparaisons\`

**Dossier resultats actif (voussoir_femelle_3) :** `C:\_workingDir\_SF\Autres modeles\_exportRebar\résultats\`

---

## 1. Ce que fait le code AC2_pure_flexion.py

### Probleme traite

Calcul de fiabilite FORM sur une poutre beton arme en flexion pure, avec 2 variables aleatoires :
- **fc** (resistance beton) — LogNormale, fcm=48 MPa, CoV=0.12
- **fy** (limite elastique acier) — Normale, fym=550 MPa, sigma = sqrt(SIGMA_11^2 + SIGMA_12^2 + SIGMA_13^2) ≈ 30 MPa (si cov_fy=None)

### Fonction de performance

`g = Primal_bound - 1` (calculee par STRAINS, analyse limite cinematique)
- `g < 0` : defaillance
- `g > 0` : domaine sur

### Reference HF

Reference HF : `2026_1205 résultats et comparaisons\resultats_run_HF.md` (run 12/05/2026)

| Parametre | Valeur |
|---|---|
| F | 0.74 MN |
| fcm / cov_fc | 48 MPa / 0.12 |
| fym | 550 MPa |
| gamma_c, gamma_s | 1.0 |
| beta_HF | **7.9788** |
| Pf | 7.39e-16 |
| u* | [-3.117, -7.349] |
| fc* | 32.83 MPa |
| fy* | 328.55 MPa |
| Imp. | fc=15%, fy=85% |
| n_iter FORM | 7 |
| u* FOSM | [-4.149, -6.377] |
| Erreur FOSM | 17.74% |

### Fichiers du modele STRAINS

- `C:\workspace\storage\admin\SF\test_pure_flexion.ds\dsLoad.txt` — charge appliquee (Z='-F')
- `C:\workspace\storage\admin\SF\test_pure_flexion.ds\dsCad.txt` — geometrie + materiaux

### Launcher

`C:\_workingDir\_SF\test flexion\launcher2.py` — configure les DLL STRAINS. **Toujours lancer via ce fichier.**

---

## 2. Architecture du code AC2 — fonctions et pipeline

### Difference cle avec AC_pure_flexion.py

AC2 est une refactorisation complete : toutes les fonctions sont definies en premier (lignes ~300–1700), le code d'execution est a la fin (lignes ~1700+). Permet de reutiliser les fonctions independamment.

### Variables de controle (OPTIONS)

```python
modele = 'PCKRG'    # 'KRG' | 'GEK' | 'PCKRG' | 'old_GEPCK' | 'GEPCK' | 'HF'
do_EFF = True/False
n0 = 5              # taille DOE initial
params_names = ['fc', 'fy']
```

Flags derives automatiquement :
```python
do_KRG      = True si modele == 'KRG'
do_GEK      = True si modele == 'GEK'
do_HF       = True si modele == 'HF'
do_PCKRG    = True si modele == 'PCKRG'
do_old_GEPCK = True si modele == 'old_GEPCK'
do_GEPCK    = True si modele == 'GEPCK'
```

### Metamodeles disponibles

| modele | Description |
|---|---|
| `KRG` | Kriging simple (noyau squared-exponential, optimisation theta dans [1,5]) |
| `GEK` | GEKPLS (Gradient-Enhanced Kriging via smt, n_comp=2) |
| `PCKRG` | PCE (LARS + LOO, base Hermite, degre ≤ max_degree) + KRG sur le residu |
| `old_GEPCK` | PCE + GEK sur le residu (version intermediaire) |
| `GEPCK` | GEPCK complet via branche1.py (fit_gepck / predict_gepck) |
| `HF` | FORM direct haute-fidelite (appels STRAINS a chaque iteration FORM) |

### Pipeline complet (modele != 'GEPCK')

```
1. update_degree(n0)           → determine max_degree selon n0

2. init_g_ot(...)
    build_DOE()
        → LHS(n0) + SimulatedAnnealing en espace U
        → n0 appels STRAINS → (xt, yt, all_grad)
        → print_DOE=True : affiche le DOE sous forme copy-pastable
    construction du metamodele selon modele actif :
        PCKRG  : PCE (LARS/LOO) → residu → KRG sur residu → g_ot = PCE + KRG_res
        KRG    : KRG direct
        GEK    : GEKPLS direct
        old_GEPCK : PCE → residu → GEK sur residu
        HF     : HFFunction (wrapper OT, appels STRAINS + cache)

3. [si do_EFF:] run_EFF(...)
        → boucle : argmax(EFF) → appel HF → enrichissement DOE → reconstruction metamodele
        → critere arret : EFF(u_opt) < tol_EFF
        → update_degree si n0 augmente

4. init_FORM(...)              → cree l'event OT

5. FORM_all_modes(starting_points, ...)
        → AbdoRackwitz depuis chaque point du DOE + [0,0]
        → DBSCAN (eps=tol_all_modes=0.01, min_samples=2) → modes tries par beta

6. [si do_warmstart:] FORM_warm_start(...)
        → si |g_meta(u*)| > tol_warmstart : ajoute u* au DOE, reconstruit, re-FORM

7. print_results(best_result, g_ot)
        → n_iter FORM, fc*, fy*, u*, Imp., beta, Pf
        → 2 appels HF supplementaires : g_HF(u*) + run_HF([0,0]) pour FOSM

8. print_visu(...)             → figure 2D avec contours, DOE, u*, EFF pts
```

### Pipeline GEPCK (pipeline separé)

Si `modele == 'GEPCK'`, un second pipeline tourne apres le FORM :
- `init_surrogate()` → DOE + appels HF
- `fit_gepck(xt, Y_aug, ...)` → GEPCK 5 branches
- Prediction + sigma sur grille 100x100
- Figure 2 panneaux (surface + incertitude), sauvee dans `notre_gepck_hf.png`

### Modele analytique (`flexion_claude`)

Formule analytique pivot B (aciers plastifies vs non plastifies). Sert uniquement a la visualisation (contour vert).

### Gestion du DOE fixe

Si `U_doe_fixed` est defini (non None), le DOE est hardcode. Si `print_DOE=True`, le DOE tire aleatoirement est affiche sous forme copy-pastable :
```
U_doe_fixed = ot.Sample([
    [ x.xxxx,  y.yyyy],
    ...
])
```
**ATTENTION :** verifier que la ligne `U_doe_fixed =` non commentee est None ou non pour savoir si le DOE est aleatoire.

---



## 4. Lecture des resultats

Patterns grep apres convergence :

```bash
grep "beta "                  output_DDMM_HHMM.txt   # beta FORM
grep "Pf "                    output_DDMM_HHMM.txt   # Pf FORM
grep "u\* "                   output_DDMM_HHMM.txt   # u* (U-space)
grep "fc\*\|fy\*"             output_DDMM_HHMM.txt   # fc*, fy* (X-space MPa)
grep "Imp\."                  output_DDMM_HHMM.txt   # facteurs d'importance
grep "dg/du"                  output_DDMM_HHMM.txt   # gradients en u*
grep "n_iter FORM"            output_DDMM_HHMM.txt   # iterations FORM
grep "FOSM"                   output_DDMM_HHMM.txt   # u* FOSM + erreur FOSM
grep "Warm start"             output_DDMM_HHMM.txt   # warm start declenche O/N
grep "EFF"                    output_DDMM_HHMM.txt   # iterations EFF
grep "g\* "                   output_DDMM_HHMM.txt   # g_HF(u*) et g_meta(u*)
grep "Erreur relative"        output_DDMM_HHMM.txt   # erreur g et erreur FOSM
grep "U_doe_fixed"            output_DDMM_HHMM.txt   # DOE tire (si print_DOE=True)
```

**Prints exacts de `print_results` (AC2_pure_flexion.py) :**
```
n_iter FORM  = X
fc*          = X.XXXX
fy*          = X.XXXX
u*           = [...]
Imp.         = [...]
beta         = X.XXXX
Pf           = X.XXXXe-XX
dg/du_fc en u* (HF@u*GEK) = X.XXXXXX
dg/du_fy en u* (HF@u*GEK) = X.XXXXXX
u* FOSM (HF) = [...]
Erreur FOSM  = X.XXXX
```

---

## 5. Remplissage des fichiers .md resultats (SEULEMENT SI DEMANDE)

**Source unique :** sorties de `print_results` dans le fichier output.

**Lignes (ordre fixe) :**

```
n points DOE
fc* (MPa)             ← "fc*" dans output
fy* (MPa)             ← "fy*" dans output
u* [u_fc, u_fy]       ← "u*" dans output
dg/du_fc en u*        ← "dg/du_fc en u*" dans output
dg/du_fy en u*        ← "dg/du_fy en u*" dans output
Importance fc (%)     ← "Imp." dans output (indice 0)
Importance fy (%)     ← "Imp." dans output (indice 1)
beta (FORM)           ← "beta" dans output
Pf (FORM)             ← "Pf" dans output
n_appels HF (FORM)    ← 0 sur metamodele, 1 si warm start declenche
n_iter FORM           ← "n_iter FORM" dans output
--- Bloc test GP ---
g_HF(u*)              ← appel HF dans print_results
g_meta(u*)            ← valeur metamodele au point de FORM
Erreur relative g     ← "Erreur relative entre g* FORM et g* GP"
--- Bloc FOSM ---
u* FOSM               ← "u* FOSM" dans output
Erreur FOSM           ← "Erreur relative" apres u* FOSM dans output
```

**Quand creer un nouveau fichier `resultats_X_runN.md` :**
- Nouvelle methode principale (KRG→GEK→GEPCK etc.) → nouveau fichier
- Configuration fondamentalement differente (nouveau DOE fixe, nouveau setup) → nouveau fichier

**Quand ajouter une colonne :**
- n0 different, meme methode, meme F → colonne supplementaire dans fichier existant
- Ne jamais ecraser une colonne existante

**Quand creer une nouvelle section :**
- F change (= beta_HF change) → nouvelle section `### F = X MN (beta_HF ≈ Y.YY)`

---

## 6. Remplissage des fichiers .md comparaison (SEULEMENT SI DEMANDE)

**Source unique :** sorties de `print_results` (idem section 5).

**Lignes fixes :**

```
n points DOE (ex: 5 ou 5+1 si warm start)
n iter FORM
n_appels HF (FORM)    ← 0 metamodele, 1 si warm start
fc* (MPa)
fy* (MPa)
u* [u_fc, u_fy]
dg/du_fc en u*
dg/du_fy en u*
Importance fc (%)
Importance fy (%)
beta (FORM)
Pf (FORM)
g_meta(u*)
g_HF(u*)
Erreur relative g
u* FOSM
Erreur FOSM
Ecart beta vs HF (absolu + %)
do_warm_start (O/N)
```

**Colonnes :**
- Toujours commencer par colonne `HF reference`
- Une colonne par methode ou configuration testee

**Quand ajouter une colonne :** type de modele change, DOE change, variante meme methode
**Quand creer un nouveau tableau :** beta change (F change)
**Quand creer un nouveau fichier :** objectif d'etude different (impact noyau, impact DOE, methodes hybrides)

---


## 7. Tableau des resultats

| Date | Fichier output | modele | n0 | EFF pts | beta | Pf | g_meta(u*) | g_HF(u*) | n_iter | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 02/06 | output_0206_1320.txt | PCKRG | 5 | — | — | — | — | — | — | run en cours |
