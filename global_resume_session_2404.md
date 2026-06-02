# Resume global de session — Fiabilite flexion pure BA
**Date :** 24 avril 2026
**Couvre :** Sessions du 16/04 au 24/04/2026 + etat du code actuel

---

## 0. Protocole de mise a jour memoire 

Ce fichier est le point d'entree unique pour se remettre a jour. Ce qui peut servir: 

**1 — Ce fichier (global_resume_session_2404.md)**
Donne le contexte global, l'architecture, l'historique, les resultats, les bugs, et le protocole de travail.

**2 — Les resultats recents**
Le dossier actif unique est : `C:\_workingDir\_SF\test flexion\2026_1205 résultats et comparaisons\`
Ne jamais chercher dans d'autres dossiers resultats sans demande explicite.



## 1. Ce que fait le code

### Probleme traite
Calcul de la probabilite de defaillance d'une poutre en beton arme (BA) en flexion pure, par la methode FORM (First Order Reliability Method), en utilisant un metamodele de substitution entraine sur des appels au code de calcul STRAINS (analyse limite cinematique).

### Variables aleatoires
| Variable | Loi | Parametres | Fichier loi |
|---|---|---|---|
| fc (resistance beton C35) | Log-normale JCSS | moy≈31.67 MPa, CoV≈0.15 | `C:\_workingDir\_SF\fiabilite\config\jcss_fc.py` |
| fy (limite elastique acier, phi=16mm) | Normale JCSS | fy_nom=500 MPa | `C:\_workingDir\_SF\fiabilite\config\jcss_fy.py` |

### Fonction de performance
`g = alpha_plus - 1` ou `alpha_plus` est le multiplicateur de charge limite cinematique calcule par STRAINS.
- `g < 0` : defaillance (charge appliquee depasse la capacite)
- `g > 0` : domaine sur

### Charge calibree
F=0.210 MN → **beta_HF = 3.784**, Pf = 7.73e-05 (reference absolue, FORM direct HF, 21 iterations).

### Fichiers du modele STRAINS
- `C:\workspace\storage\admin\SF\test_pure_flexion.ds\dsLoad.txt` — charge (modifier Z='-0.210' pour changer F)
- `C:\workspace\storage\admin\SF\test_pure_flexion.ds\dsCad.txt` — geometrie (phi, b, h, L) + materiaux (ft, E)

**ATTENTION :** Le dossier .ds a ete deplace le 28/04/2026 de `C:\workspace\storage\semia\` vers `C:\workspace\storage\admin\SF\`. Mettre a jour le path dans `AC_pure_flexion.py` (variable `path` dans `run_one_SOL` et `run_HF`) si ce n'est pas encore fait.

---

## 2. Architecture du code

### Fichiers principaux
- `C:\_workingDir\_SF\test flexion\AC_pure_flexion.py` — script principal (toutes branches metamodele)
- `C:\_workingDir\_SF\test flexion\launcher.py` — wrapper DLL STRAINS (TOUJOURS utiliser ce fichier)

### Flags de controle (etat au 12/05)

**Flags principaux :**
- `do_HF` : True = FORM direct HF (HFFunction wrapper, appels STRAINS a chaque iteration)
- `do_KRG` : True = branche KRG
- `do_GEK` : True = branche GEK (actif par defaut)
- `do_multistart` : True = starting_points = LHS(n0) + [0,0] ; False = [0,0] uniquement
- `print_HF` : True = grille HF (n_grid_hf x n_grid_hf appels STRAINS) dans print_visu
- `print_ana` : True = courbe analytique (flexion_claude) dans print_visu

**Cas special visu sans FORM :** tous do_* = False + best_sol_modes_fixed non None → print_visu direct + sys.exit(0).

| `do_HF` | `do_KRG` | `do_GEK` | Branche active |
|---|---|---|---|
| True | — | — | HF direct |
| False | True | False | KRG pur |
| False | False | True | GEK pur |

### Options actuelles (etat code au 19/05) — INDICATIF UNIQUEMENT
**Toujours lire le bloc OPTIONS dans `AC_pure_flexion.py` avant tout run.**

**Flags metamodele (noms actuels — ont change depuis 12/05) :**
```python
do_KRG   = False   # KRG pur
do_GEK   = False   # GEK pur
do_HF    = True    # FORM direct HF
do_PCKRG = False   # KRG + PCE (ancien try_pce avec do_KRG)
do_GEPCK = False   # GEK + PCE (ancien try_pce avec do_GEK)
do_EFF   = False   # enrichissement adaptatif EFF
```
Note : `try_pce` (bool) a ete remplace par deux flags distincts `do_PCKRG` et `do_GEPCK`.

```python
n0 = 15; do_multistart = True; do_warmstart = False
max_degree = 2; epsilon_factor = 2; tol_EFF = 0.001; n_max_EFF = 1000
```
**dsCad actif :** b=h=0.8m, 3 lits 8HA32, phi=32mm, gamma_c=gamma_s=1.0, F=0.74 MN.

### Pipeline complet (etat au 12/05)
```
[si do_HF/do_GEK/do_KRG:]
    build_DOE()
        → LHS(n0) + SimulatedAnnealing en espace U
        → si not do_HF : n0 appels STRAINS → (xt, yt, all_grad)
        → si do_HF : retourne xt uniquement (pas d'appels STRAINS)

    [si do_GEK:] sm_GEK = build_metamodel_GEK(xt, yt, all_grad)
    [si do_KRG:] g_ot_KRG = build_metamodel_KRG(xt, yt)
    [si do_HF:]  g_ot_HF = ot.Function(HFFunction())
    event = FORM_event(g_ot)

    starting_points = vstack([xt,[0,0]]) si do_multistart sinon [[0,0]]
    modes, best_sps = FORM_all_modes(starting_points, tol_all_modes)
        → AbdoRackwitz depuis chaque point (setMaxIterations, setCheckStatus(False))
        → DBSCAN eps=tol_all_modes, min_samples=2 → clusters → modes tries par beta
        → retourne (modes: list[FORMResult], best_sps: list[sp])
    best_result = modes[0] ; best_sp = best_sps[0]

    print_results(best_result, ...)
    print_visu(best_result, best_sp, xt, sm_GEK, g_ot_KRG, modes)

[si tous do_* = False et best_sol_modes_fixed is not None:]
    print_visu(None, None, None, None, None, [])  → visu seule, sys.exit(0)
```
## 6. Processus de lancement du code

### 6.1 Lire les options avant lancement

**TOUJOURS lire le bloc OPTIONS (lignes ~199-231 de `AC_pure_flexion.py`) avant d'interpreter un run.**
Ces options definissent entierement le comportement du run : branche active (do_GEK, try_pce), DOE fixe ou aleatoire, warm start, n0, etc.

**TRES IMPORTANT — ne JAMAIS interpreter un run sans avoir lu les options au prealable.** Une interpretation faite sans lire les options est fausse par definition : on ne sait pas ce qui a ete teste. Exemple concret : interpreter un run comme "gradient analytique" alors que `do_analytic_grad=False` etait actif conduit a des conclusions erronees sur la cause des differences observees.

En particulier, verifier **`U_doe_fixed`** (lignes ~235-251) : si `U_doe_fixed = None`, le DOE est tire aleatoirement (LHS) ; sinon, le DOE fixe est hardcode et doit etre note dans le resume de run. Si **`print_DOE = True`**, le DOE tire aleatoirement sera affiche dans l'output — le recuperer et le donner OBLIGATOIREMENT dans la discussion sous forme de bloc copy-pastable (format `U_doe_fixed = ot.Sample([...])`), pour que l'utilisatrice puisse le hardcoder si elle veut fixer ce DOE pour un run suivant (cf. 6.2).

**ATTENTION — ne pas confondre DOE commente et DOE actif :** le bloc `U_doe_fixed` peut contenir un DOE hardcode qui est COMMENTE (lignes precedees de `#`). Dans ce cas, la ligne active est `U_doe_fixed = None` et le DOE est aleatoire. Toujours lire la ligne `U_doe_fixed =` qui n'est PAS commentee pour determiner si le DOE est fixe ou non.

Note : il arrive que les options soient modifiees pendant qu'un run tourne (preparation du run suivant). Ces modifications ne comptent pas pour l'interpretation du run en cours — seules les options lues AVANT le lancement sont pertinentes.

### 6.2 Lancement et lecture des resultats

> **!!! REGLE ABSOLUE AVANT TOUT LANCEMENT !!!**
> Executer `date +%d%m_%H%M` dans un appel bash SEPARE avant de lancer le code.
> Noter le nom exact du fichier output qui sera cree (`output_DDMM_HHMM.txt`).
> JAMAIS lancer sans avoir fait cette verification — sinon le fichier output est inconnu et doit etre cherche, ce qui est interdit.
> Cette regle s'applique dans toutes les sessions futures sans exception.

**Toujours lancer via launcher.py** (configure les DLL STRAINS) :
```
python launcher.py > "output/output_$(date +%d%m_%H%M).txt"
```

**Nommage obligatoire des outputs :** A chaque lancement, le fichier de sortie doit etre nomme avec la date et l'heure de lancement au format `output_DDMM_HHMM.txt`.
Exemple : lancement le 24 avril a 8h51 → `output_2404_0851.txt`.
Les fichiers sont a placer dans `C:\_workingDir\_SF\test flexion\output\`.
**IMPORTANT — heure systeme obligatoire :** Toujours utiliser `$(date +%d%m_%H%M)` dans la commande bash pour obtenir l'heure exacte du lancement. Ne jamais estimer l'heure manuellement — Claude n'a pas acces a l'horloge en temps reel et se trompe systematiquement.

**Surveillance et lecture des resultats — ROLE DE CLAUDE :**

> **!!! REGLE ABSOLUE — SURVEILLANCE ACTIVE ET PROACTIVE !!!**
> Claude NE DOIT PAS attendre que l'utilisatrice pose une question sur l'avancement du run.
> Claude DOIT surveiller activement l'output en cours de run et PREVENIR L'UTILISATRICE DE LUI-MEME :
> - Des que les resultats FORM apparaissent (beta, Pf, u*) → communiquer IMMEDIATEMENT, SANS ATTENDRE de message.
> - Avant le demarrage de la visu (do_visu=True) → signaler que FORM est termine et que la visu commence.
> - Si le run echoue ou crash → signaler immediatement.
> Ne JAMAIS laisser l'utilisatrice demander "alors ?" ou "tu as les resultats ?" — c'est un echec du protocole.
> Cette regle s'applique dans toutes les sessions futures sans exception.

C'est Claude qui surveille le run, lit les resultats et met a jour les .md. Protocole :

1. **Lancer en arriere-plan** via `run_in_background=True` (le run peut durer plusieurs minutes).
2. **Surveiller via `until grep`** (voir commande obligatoire ci-dessous). Les resultats FORM apparaissent avant la visu — communiquer immediatement.

> **!!! COMMANDE OBLIGATOIRE — SURVEILLANCE ACTIVE !!!**
> Immediatement apres le lancement en arriere-plan, lancer SYSTEMATIQUEMENT la commande suivante en arriere-plan (run_in_background=True) :
> ```
> until grep -q "beta =" "output/output_DDMM_HHMM.txt" 2>/dev/null; do sleep 15; done && grep "beta\|Pf\|u\*\|Imp\." "output/output_DDMM_HHMM.txt"
> ```
> Cette commande poll toutes les 15 secondes et extrait les resultats FORM des qu'ils apparaissent.
> Elle doit etre lancee dans le repertoire du run (`cd "C:\_workingDir\_SF\test flexion"`).
> Ne JAMAIS se contenter de greps manuels periodiques — utiliser obligatoirement ce until.
> **Si les prints ont change** (ex : renommage de "beta =" en autre chose, nouveau format de sortie) : lire `AC_pure_flexion.py` pour identifier les prints exacts produits par le code actif, puis adapter le pattern du `grep -q` et du `grep` final en consequence. Ne jamais copier-coller le until sans avoir verifie que le pattern correspond aux prints reels du code.

3. **Des que les resultats apparaissent**, les lire et les communiquer immediatement dans la conversation :
```
grep "Beta FORM"       output_2404_0851.txt
grep "Pf FORM"         output_2404_0851.txt
grep "Design point"    output_2404_0851.txt
grep "g\*"             output_2404_0851.txt
grep "Importance"      output_2404_0851.txt
grep "iterations FORM" output_2404_0851.txt
grep "Warm start"      output_2404_0851.txt
```
4. **Mettre a jour le .md correspondant** dans `2026_3004 résultats et comparaisons\` (cf. section 6.5) avec les valeurs extraites, sans attendre la fin de la visu.

### 6.3 Remplissage des fichiers .md resultats

**Source des valeurs :** toutes les lignes ci-dessous sont extraites des sorties de `print_results` dans le fichier output du run. C'est `print_results` qui est la source unique — ne pas chercher les valeurs ailleurs dans l'output.

**Lignes (ordre fixe dans tous les fichiers resultats) :**

```
n points DOE
fc* (MPa)             ← "Design point X : fc" dans output
fy* (MPa)             ← "Design point X : fy" dans output
u* [u_fc, u_fy]       ← "Design point U" dans output
dg/du_fc en u*        ← "dg/du_fc" dans output
dg/du_fy en u*        ← "dg/du_fy" dans output
Importance fc (%)     ← "Importance factor fc" dans output
Importance fy (%)     ← "Importance factor fy" dans output
beta (FORM)           ← "Beta FORM" dans output
Pf (FORM)             ← "Pf FORM" dans output
n_appels HF (FORM)    ← 0 sur metamodele, 1 si warm start declenche
n_iter FORM           ← "iterations FORM" dans output
--- Bloc test GP ---
g_HF(u*)              ← "g* FORM" dans output (bloc GP_HF_test)
g_meta(u*)            ← "g* GP" dans output (bloc GP_HF_test)
Erreur relative g     ← "Erreur relative entre g* FORM et g* GP"
--- Bloc FOSM ---
u* FOSM               ← "u* FOSM" dans output
Erreur FOSM           ← "Erreur relative" apres u* FOSM dans output
```

**Quand creer un nouveau fichier resultats_X_runN.md :**
- Nouvelle methode principale (KRG→GEK, GEK→GEPCK, etc.) → nouveau fichier
- Configuration fondamentalement differente (changement de DOE fixe, nouveau setup) → nouveau fichier
- Runs successifs meme methode, n0 variables → colonne supplementaire dans fichier existant

**Quand ajouter une colonne :**
- n0 different, meme methode, meme F → nouvelle colonne dans le tableau existant de cette F
- Ne jamais ecraser une colonne existante

**Quand creer une nouvelle section dans le meme fichier resultats :**
- F change (= beta_HF change) → nouvelle section `### F = X MN (beta_HF ≈ Y.YY)` avec son tableau

### 6.4 Remplissage des fichiers .md comparaison

**Source des valeurs :** idem section 6.3 — toutes les valeurs viennent des sorties de `print_results`.

**Lignes fixes (ordre dans tous les fichiers comparaison) :**

```
n points DOE (ex: 15 ou 15+1 si warm start)
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
- Toujours commencer par colonne `HF reference` (beta=3.784, u*=[-0.526,-3.747] pour F=0.210 MN)
- Puis une colonne par methode ou configuration testee

**Quand ajouter une colonne (dans le meme tableau) :**
- **Type de modele change** (ex: KRG→GEK→GEPCK) — meme F et meme etude comparative
- **DOE change** (ex: n0=20 vs n0=15, ou nouveau DOE fixe, meme methode)
- Variante d'une methode (avec/sans warm start)

**Quand creer un nouveau tableau (dans le meme fichier) :**
- **beta change** (= F change → nouveau beta_HF) → un tableau par valeur de F dans le meme fichier
- Exemple dans `comparaison_HF_KRG.md` : 3 tableaux (F=0.235, F=0.225, F=0.210)

**Quand creer un nouveau fichier de comparaison :**
- Objectif d'etude different (impact noyau, impact DOE fixe, methodes hybrides)
- Les colonnes ne sont pas comparables entre deux etudes differentes
- Exemples : `comparaison_Matern_exp.md` (impact noyau), `comparaison_a_DOE_fixe.md` (toutes methodes DOE fixe)

**Patterns grep pour extraire les valeurs depuis un output log :**
```
grep "Beta FORM"                     → beta FORM
grep "Pf FORM"                       → Pf FORM
grep "Design point U"                → u* (coordonnees U-space : u_fc, u_fy)
grep "Design point X"                → fc*, fy* (coordonnees X-space en MPa)
grep "g\* FORM"                      → g_HF(u*) (test HF au point de FORM)
grep "g\* GP"                        → g_meta(u*) (valeur metamodele au point de FORM)
grep "Importance factor"             → Importance fc et fy
grep "dg/du_"                        → gradients en u*
grep "iterations FORM"               → n_iter FORM
grep "u\* FOSM"                      → u* FOSM
grep "Erreur relative entre u\*"     → Erreur FOSM
grep "Warm start lance"              → warm start declenche O/N
```

### 6.5 Mise a jour de la documentation apres chaque run

**Apres chaque run, mettre a jour :**

**1. `global_resume_session_2404.md` (ce fichier) :**
- Section 4 (tableau des resultats) : ajouter une ligne avec la methode, beta, erreur beta, g_meta(u*), g_HF(u*), n_iter.
- Section 9 (taches en cours) si applicable : noter les observations importantes.

**2. `resume_session_DDMM.md` dans `resume session a relire\` :**
- Fichier du jour courant. Nommage : `resume_session_2404.md` pour le 24 avril, `resume_session_2504.md` pour le 25 avril, etc.
- Si le fichier du jour existe deja : ajouter une nouvelle Partie (Partie 2, Partie 3, ...) a la suite.
- Si c'est la premiere modification du jour : creer un nouveau fichier avec la date du jour.
- Contenu d'une partie : configuration complete du run (options lues avant lancement, DOE fixe si applicable), resultats au meme niveau de detail que les parties existantes, observations.
- Si `U_doe_fixed` n'etait pas None : noter le DOE hardcode utilise (bloc ot.Sample complet).

**3. Fichier .md dans `2026_3004 résultats et comparaisons\` :**
- resultats_X_runN.md → sous-dossier `résultats par modèle\` (cf. section 6.3)
- comparaison_X.md → sous-dossier `comparaisons entre modèles\` (cf. section 6.4)
- PNG visu → sous-dossier `png visu\`

---



















### Classe GEKPLSFunction
```python
class GEKPLSFunction(ot.OpenTURNSPythonFunction):
    def _exec(self, u):      return [sm.predict_values(...).item()]
    def _gradient(self, u):  return [[sm.predict_derivatives(u_np, kx).item()] for kx in range(n_var)]
    # format gradient OT : (n_var, 1) — reconnu nativement, pas de fallback FD
```

### Classe flexion_claude (fonction analytique — etat 12/05)
Lit `dsCad.txt` et `dsLoad.txt` (regex). Utilise `gamma_c_fic`/`gamma_s_fic` (pas gamma_c/gamma_s du dsCad).
```python
self.A = As * d / gamma_s_fic
self.B = -As**2 * gamma_c_fic / (2 * b * gamma_s_fic**2)
self.C = -Med
# u1_lim_plast : fc de plastification via equation quadratique en s=sqrt(1+4*Ap*x1)
def u2p_LS(self, u1): # formule directe, pas de brentq
```
Branche verticale : `u1 = u1_lim_plast`, tracee de `u2_lim` vers `u2_max`.

### Signature print_visu (etat 12/05)
```python
def print_visu(best_result, best_sp, xt, sm_GEK, g_ot_KRG, modes):
```
- `print_HF` (bool global) : grille n_grid_hf x n_grid_hf via `run_HF` directement
- `print_ana` (bool global) : courbe analytique via `flexion_claude()`
- `best_sol_modes_fixed` (dict global) : u* et sp fixes d'un run precedent, traces en 4 couleurs

---

## 3. Historique des modifications du code

Chaque entree cite le fichier de resume de session pour les details (bugs complets, code avant/apres, explications theoriques).

### Session 16/04 — Mise en place FORM HF
**Ref :** `C:\_workingDir\_SF\test flexion\resume session a relire\session_resume_1604.md`

- **Bug critique :** `dg/dfy = 0` pour tous les points DOE → FORM ne converge pas.
  Cause : cle `"solids"` au lieu de `"rebars"` pour les armatures dans `sensitivity_regions`.
  Fix : `{"param": "YIELD_STRENGTH", "rebars": ["HA1","HA2","HA3","HA4"]}`.
- Calibration charge : F=0.235 → beta=16 (trop fort) → F=0.60 → beta=5.74 → F=0.210 → beta=3.784.
- Architecture HFCache + AbdoRackwitz validee.
- Test linearisation FOSM : g quasi-lineaire (erreur FOSM < 0.5%).

### Session 17/04 apres-midi — Debogage FORM HF
**Ref :** `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_1704_aprem.md`

- `run_HF` retourne 3 valeurs : `(g_HF, grad_HF_U: ot.Point, grad_HF_X: list)` — toujours depackager a 3 valeurs.
- Garde-fou : ValueError si `grad_HF_U` contient None.
- **Convention gradient OT PythonFunction :** retourner `[[dg/du1, dg/du2]]` (shape `[1][n_var]`) et non `[[dg/du1], [dg/du2]]` (shape `[n_var][1]`) — erreur silencieuse, OT fallback FD sans warning.
- Pattern HFCache : evite double appel STRAINS (func + grad_func au meme point).
- `setCheckStatus(False)` sur AbdoRackwitz : permet de recuperer le resultat FORM meme si g(u*) > tolerance interne OT.

### Session 20/04 — FORM KRG + impact DOE + warm start
**Ref :** `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_2004.md`

- FORM KRG valide sur F=0.235 (beta_HF≈0.95) : excellent n0≥25, bon n0=16, degrade n0=8.
- F=0.210 (beta_HF≈3.78) : KRG necessite n0=60 pour erreur 8.6% ; n0=15 → erreur 50%.
- **Warm start proof of concept :** enrichir DOE avec u*_n15 → erreur 0.6% en 2 iterations.
- Warm start automatique implemente : test `g_meta(u*) > tol_warm_start` → ajout HF point + rebuild metamodele + rerun FORM.
- GEK etape 1 (Cobyla) : resultat non physique (u* pas sur surface limite) → solver remplace par AbdoRackwitz.

### Session 21/04 matin — PCE (LARS, theorie, construction)
**Ref :** `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_2104.md`

- PCE construit avec `FunctionalChaosAlgorithm` (LARS + CorrectedLeaveOneOut, polynomes Hermite).
- **Bug bloquant :** `FunctionalChaosValidation` crash en OT 1.26 avec LARS — check C++ `involvesModelSelection()` au constructeur, incontournable par heritage Python.
- Solution : LOO manuel `compute_q2_loo` (double CV non biaise) via `ot.MetaModelValidation`.
- Separation `try_pce` (intention utilisateur) vs `do_pce` (resultat apres validation Q2).

### Session 21/04 apres-midi — Fix LOO + premier run PCE-KRG
**Ref :** `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_2104.md` (suite dans meme fichier)

- `compute_q2_loo` implementee (boucle sur folds LOO, re-entraine LARS a chaque fold).
- Segfault lors de `algo_KRG.run()` avec yt residuel ~1e-4 → contourne par `if True:` (workaround temporaire, compute_q2_loo commentee).
- **Premier run PCE+KRG reussi :** beta=3.790 (erreur 0.16% vs HF=3.784).

### Session 22/04 — Refactoring fonctions + audit GEK
**Ref :** `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_2204.md`

- Refactoring complet : `init_GP`, `fill_sol`, `tirage_DOE`, `build_metamodel_*`, `FORM_*`, `resultats_GP`, `print_*`.
- **Aliasing numpy (bug critique) :** `yt = y_hf` sans `.copy()` → `yt -= y_PCE` modifie `y_hf` en place → corrompt les donnees de maniere silencieuse. Fix : `.copy()` dans `fill_inputGP` sur les 3 variables.
- Builder vs evaluateur KRG : `build_metamodel_KRG` retourne un objet OT ; l'evaluateur est `metamodel_KRG(u)`.
- `run_HF` : tous les appels mis a jour pour depackager 3 valeurs.
- `FORM_GEK` : SyntaxError argument sans defaut apres argument avec defaut → corrige.
- `hf_cache` renommage dans `do_visu`.
- Warm start KRG pur : `U_doe = ot.Sample(xt)` avant `.add(U_warm)` + `np.array()` wrapping.

### Session 23/04 matin — Audit PCE-KRG + KRG pur
**Ref :** `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_2304.md`

- `fill_PCE` : retourne `all_sensib_PCE`, `T_inv` defini localement, `y_PCE` en majuscule.
- `fill_inputGP` : `.copy()` sur les 3 variables (aliasing critique).
- `FORM_KRG` : parametre `metamodel` (pas `metamodel_KRG`) + `start_point` parametre obligatoire + `n_var` local.
- **Run KRG pur run3 (DOE fixe n0=15) :** beta=5.065 (+33.9% vs HF), g_HF(u*)=-0.050.

### Session 23/04 apres-midi — Audit GEPCK + GEK pur run1
**Ref :** `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_2304.md` (PARTIE 2 et 3)

- `build_metamodel_total` : n_var local, signature `(sm, metamodel_PCE=None)`, 1 seul `return`.
- `GP_HF_test` : `g_GP` → `g_GP_res`.
- Blocs GEPCK : `start_point` defini avant `FORM_GEK`.
- Encodage cp1252 : caractere fleche gauche dans commentaire → UnicodeDecodeError → remplace par `--`.
- **Run GEK pur run1 (DOE fixe n0=15, do_warm_start=False) :** beta=2.118 (-44% vs HF), g_GP_res=0.074 (FORM non converge sur surface limite), instabilite globale GEK avec n0=15.

### Session 24/04 — Run GEPCK DOE fixe + refactoring FORM_multi_start
**Ref :** `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_2404.md`

- Run GEPCK (do_GEK=True, try_pce=True, do_warm_start=True, n0=15 DOE fixe, F=0.210 MN).
- **Resultats :** beta=3.779 (erreur -0.1% vs HF=3.784), n_iter=15, u*=[-0.607, -3.730].
- g_GP_res=0.000006, g_HF(u*)=0.000345, fc*=30.174 MPa, fy*=505.870 MPa.

### Session 30/04 — Nouvelle geometrie + print_results + run HF direct
**Ref :** `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_3004.md`

- **Nouvelle geometrie dsCad :** b=0.5m, h=0.8m, L=5m, phi=16mm, 2 lits de 3HA16 (z=0.328m et z=0.312m), Block1 (0 a 4m, ft1=0.1), Block2 (4m a 5m, ft2=3.5), CONNECT GLUED. fc=48 MPa, fy=550 MPa (nominaux dsCad). Charge : F=0.1 MN. fck=40, fyk=500 (distributions JCSS).
- **FORM_all_modes :** diagnostic [sp, u*, beta] ajoute pour chaque point de depart — permet de verifier si plusieurs modes sont detectes.
- **Fix do_HF :** `build_DOE()` supprime du bloc `elif do_HF`, remplace par `xt = np.empty((0, n_var))`. Variables `g_ot_GEK = None` et `g_ot_HF = None` initialisees en dehors des blocs if/elif (evite NameError).
- **`print_results(best_result, g_ot_GEK, g_ot_KRG, g_ot_HF)` :** nouvelle fonction d'affichage. Toujours : n_iter, X-space (fc*, fy*), u*, importances, beta, Pf. Blocs conditionnels independants (meme structure que print_visu) : gradient dg/du en u* + FOSM pour chaque modele non-None.
- **Visu etendue [-16,16] :** grille surrogate et HF etendues de [-4,4] a [-16,16]. La grille HF (3x3) a des points extremes qui bloquent STRAINS — passer `g_hf=None` pour eviter.
- **Run HF direct (output_3004_1203.txt) :** do_HF=True, fck=40, fyk=500, F=0.1 MN, point de depart [0,0]. beta=0.3915, Pf=6.52e-01, u*=[0.0016, 0.391], fc*=47.81 MPa, fy*=561.40 MPa, n_iter=6, dg/du_fc=0.000220, dg/du_fy=0.051833, Erreur FOSM=0.46%.

### Session 04/05 — Validation analytique + corrections code
**Ref :** `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_0405.md`

- **Bug `d` dans `calc_ana()` corrige :** `d = h - z_centroide` → `d = h/2 + z_centroide` (ligne 189). Les z_rebar extraits de dsCad sont mesures depuis l'axe neutre (origine a h/2, pas a la fibre comprimee). Pour b=0.2, h=0.5, 3x1HA32 : d passait de 0.330m a 0.420m.
- **Parametre `fac_c=1.1` supprime :** retire de la signature `flexion_simple.__init__` (parametre inutilise).
- **`print_error_ana_hf(calc, n_scan=100)` implementee :** ecrite par Claude (lignes 763-812 du code). Trouve les zeros de g_ana (brentq), sous-echantillonne a 2*n_grid_hf points, evalue run_HF et g_ana sur ces points, affiche err_abs = |g_HF - g_ana| et err_abs_moy. Correction apportee : `n_visu` remplace par `n_grid_hf` ; normalisation par std(g_HF) supprimee (non pertinente — gonfle artificiellement le ratio).
- **OPTIONS mises a jour :** `print_ana_hf_error = True`, `size_visu = 5`, `n_grid_hf = 7`.
- **Resultat cle :** g_HF ≈ +0.02 uniformement sur les 19 points de la frontiere analytique (g_ana≈0), err_rel_moy = 8.08, std(g_HF) = 0.0025. Decalage systematique : STRAINS ≈ 2% plus permissif que la formule analytique. Les deux courbes sont paralleles mais decalees.
- **Ajout fc/fy dans le tableau print_error_ana_hf :** `dist_X = dist_jointe()` + `T_inv` instancies avant la boucle, `x = T_inv(ot.Point(list(pt)))` dans la boucle. Colonnes fc et fy (MPa) ajoutees a chaque ligne du tableau.
- **Inversion scan u1/u2 dans print_error_ana_hf :** u2 devient la variable externe (grille uniforme decroissante de size_visu vers u2_low, 40 points), u1 est cherche par brentq. Avant : u1 externe, brentq sur u2. Points ordonnes par u2 decroissant (fy fort → fy faible). Detail code : `resume_session_0405.md` Partie 6.2.
- **Filtre post-brentq (discontinuites test_plast) :** apres brentq, rejet si `abs(calc.g_ana([u1_star, u2])) >= 1e-6`. Cause des faux zeros : `test_plast` est un switch binaire (anp.where → 0.0 ou 1.0) entre g_ana_plast et g_ana_nonplast ; au point de transition, les deux formules ne sont pas egales → saut discontinu de signe dans g_ana, detecte a tort comme un zero. Brentq converge sur la discontinuite, pas sur un vrai zero. Le filtre post-evaluation elimine ces points. Detail code : `resume_session_0405.md` Partie 6.3.
- **Run output_0405_1446 :** 11 points propres (faux zeros filtres), tous g_ana=0.000, err_abs_moy=0.0327. Frontiere couvre fc∈[30, 58] MPa, fy∈[510, 588] MPa. Biais non uniforme : pic a 0.039 autour de fc≈47 MPa/fy≈526 MPa, decroit vers les extremes.
- **size_visu=7 (output_0405_1543) :** meme 11 points, aucun nouveau point sur la branche verticale. Analyse : la frontiere g_ana=0 a fy eleve (u2>-0.38) necessite fc tres faible (u1≈-5.7), hors de la fenetre [-7,7]. Pour voir la branche verticale : augmenter F (deplace la frontiere vers des fc moins extremes) plutot qu'agrandir la fenetre.
- **Nouvelle geometrie h=0.4, F=0.085 (output_0405_1549) :** FORM non converge (n_iter=50 max), u*=[-7.12,-3.45], beta≈7.9, Pf=1.24e-15. Cause : F=0.085 < F_crit≈0.094 MN (capacite nominale avec h=0.4, 2x2HA32, fck=28 → Mu≈0.471 MN.m, F_crit=Mu/L=0.094 MN). Poutre trop resistante, FORM part chercher du beton quasi-inexistant. Run interrompu. Charge a augmenter a F≈0.10-0.11 MN.
- **Corrections bloc principal (lignes 884-947) :** initialisation complete de toutes les variables a None avant les if/elif (`event=None`, `sm_GEPCK=None`, `g_ot_GEPCK=None`, `g_ot_PCKRG=None`) ; guard clause `if event is None: sys.exit(1)` ; early exit `if best_result is None: sys.exit(1)` ; `print_results` et `print_visu` desormais a plat (plus de `else:` imbrication). SyntaxError corrigee : `if event = None` → `if event is None`.
- **Multi-mode print (ajout utilisatrice, lignes 936-942) :** si `len(modes) > 1`, affiche les resultats du mode 2 avant le mode 1 (meilleur beta). `print_visu` ne trace que le mode 1 (best_result).
- **`print_pts = True` (ajout utilisatrice, ligne 106) :** flag qui fait sortir `print_error_ana_hf` apres le scan brentq sans lancer les appels HF. Affiche uniquement les coordonnees u des points de frontiere analytique. Objectif : explorer rapidement plusieurs jeux de donnees pour trouver une config ou la frontiere g_ana=0 a des u2 eleves. Fix : `flush=True` sur les prints du bloc print_pts pour eviter le buffering Python pendant les appels STRAINS de print_visu.
- **Exploration branche verticale — h=0.4, F=0.094, b=0.4 (output_0405_1628) :** 9 points frontiere, premier zero a u2=-1.97. Pire que h=0.45/F=0.11 (premier zero u2=-0.38). Cause : F=0.094≈F_crit → frontiere a u2≈0 necessite u1 << -7.
- **Exploration branche verticale — h=0.4, F=0.094, b=0.3 (output_0405_1724) :** 7 points frontiere, u2 ∈ [-1.92, -3.46], u1 ∈ [+1.89, +4.56]. Toujours pas de branche a u2 > 0. Reduire b ne resout pas le probleme.
- **print_visu etendue aux modes multiples :** signature  (parametre obligatoire). Modes supplementaires traces en magenta avec leur beta. Condition  sur modes[1:] pour eviter doublon avec best_result (mode 1).
- **Run output_0405_1807 (b=0.4, h=0.45, F=0.11, fck=28, fyk=550, n0=20, 2x2HA32) :** GEK, beta=3.182, Pf=7.32e-04, u*=[+0.65, -3.11], fc*=58.47 MPa, fy*=505.70 MPa, Imp. fc=4.2%/fy=95.8%, FOSM=31.9%. 6 points frontiere analytique, u2 ∈ [-1.92, -3.21], u1 ∈ [-4.75, +2.65].

### Session 06/05 — Refactoring lois + debut refonte analytique
**Ref :** `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_0605.md`

- Suppression imports jcss_fy/jcss_fc ; `loi_fy`, `loi_fc`, `SIGMA` definis localement.
- Refactoring `flexion_simple` : `g_ana_plast`/`g_ana_nonplast`/`g_ana` → `f_plast`/`f_nonplast`/`f`. Methode `print_f` ajoutee, fix NameError + vectorisation meshgrid.

### Session 07/05 — Refonte complete de la fonction analytique
**Ref :** aucun resume de session cree ce jour.

- **`flexion_simple` / `calc_ana` / `f_plast` / `f_nonplast` / `f` / brentq : SUPPRIMES.**
- **Nouvelle classe `flexion_claude` :** lit `dsCad.txt` et `dsLoad.txt` (regex), formule directe `u2p_LS(u1)` (pivot B, sans brentq), `u1_lim_plast` calcule analytiquement.
- **`print_visu_ana()` :** nouvelle fonction, trace courbe analytique seule.
- **`print_visu` mis a jour :** plus de parametre `g_ana`, courbe analytique calculee en interne. Nouveau dernier parametre `modes` (liste FORMResult pour les modes supplementaires).
- **Distributions :** `fcm/fym/cov_fc/cov_fy` ; `loi_fc` avec COV_TABLE ; `loi_fy` sigma depuis SIGMA_11/12/13.

### Session 11-12/05 — Refactoring FORM + runs HF direct (gamma=1.0, nouvelle geometrie)
**Ref :** aucun resume de session cree.

**Modifications code :**
- **`FORM_multistart` supprime :** `FORM_all_modes` gere tout. Retourne desormais `(modes, best_sps)` — `all_sp` tracke le point de depart de chaque run reussi, ordonne avec `modes` par beta croissant.
- **`starting_points` conditionne :** `np.vstack([xt, [[0,0]]])` si `do_multistart` sinon `np.array([[0,0]])`.
- **`FORM_all_modes` mis a jour :** `setMaximumIterationNumber(n_max_FORM)`, `setCheckStatus(False)`, `setMaximumConstraintError(tol_FORM)` ajoutes (identique a l'ancien `FORM_multistart`).
- **`print_HF` (bool) remplace parametre `g_hf` dans `print_visu` :** le bloc HF appelle directement `run_HF` sur grille n_grid_hf x n_grid_hf si `print_HF=True`.
- **`best_sol_modes_fixed` / `sol_modes_fixed` :** deux dictionnaires fixes definis dans les OPTIONS pour afficher les u* d'un run precedent sans relancer FORM.
- **`print_visu` mis a jour :** bloc `if best_sol_modes_fixed is not None` trace les 4 modes A/B/C/D en etoiles colorees (bleu/rouge/vert/or) + sp en croix memes couleurs.
- **`event is None` + `best_sol_modes_fixed is not None` :** saut direct vers `print_visu(None,None,None,None,None,[])` + `sys.exit(0)` — permet de tracer la visu sans relancer FORM.
- **dsCad mis a jour :** gamma_c=1.0, gamma_s=1.0 (precedemment 1.5/1.15), F=0.74 MN dans dsLoad.
- **Geometrie active :** b=h=0.8m, 3 lits 8HA32, phi=32mm, fcm=48, fym=550, cov_fc=0.12.
- **`sol_modes_fixed`/`best_sol_modes_fixed` conditionnes :** definis uniquement si `not do_HF and not do_GEK and not do_KRG`, sinon `None`. Evite d'afficher les u* du run precedent lors d'un vrai FORM.
- **Test gradient sentinelle (12/05) :** voir section 7 "Gradient HFFunction verifie".
- **`print_3D_HF()` ajoutee :** nouvelle fonction de visualisation 3D de la surface g_HF.
  - Evalue `run_HF` sur grille `n_grid_hf x n_grid_hf` (n_grid_hf^2 appels STRAINS).
  - Imprime dans l'output un bloc `hf_3d_grid_fixed = {'params':..., 'Z':[...]}` copy-pastable pour hardcoder les valeurs et eviter de relancer STRAINS.
  - Trace surface 3D (`plot_surface`), contour g=0 (rouge projete + noir sur surface), u* de `best_sol_modes_fixed` a z=0.
  - Declenchee par flag `print_3D = True` dans OPTIONS + `if print_3D: print_3D_HF(); sys.exit(0)` dans le corps principal.
  - **Etape 1 (FAIT 12/05) :** STRAINS a tourne sur la grille 7x7. `hf_3d_grid_fixed` hardcode dans les OPTIONS.
  - **Etape 2 (FAIT 12/05) :** `hf_3d_grid_fixed = {'params':..., 'Z':[...]}` hardcode dans les OPTIONS. Cache actif.
  - **Etape 3 (FAIT 12/05) :** `if hf_3d_grid_fixed is not None` ajoute dans `print_3D_HF` — court-circuite STRAINS.
  - **Etat actuel :** les etapes 1-2-3 sont implementees. La fonction trace surface HF + contours g=0 (rouge projete, noir sur surface) + u* (etoiles colorees A/B/C/D). sp non traces (voir ci-dessous).

**Modifications tentees puis revertees (session 12/05) :**
- **Scatter sp ajoute puis supprime :** `ax.scatter(u1_s, u2_s, 0.0, c=col, s=100, marker='x', linewidths=2, label=f'sp {lbl}')` avait ete ajoute dans la boucle `best_sol_modes_fixed`. Supprime car run impossible (environnement Python).
- **Contour analytique ajoute puis supprime :** bloc complet supprime car non teste. A remettre plus tard si besoin :
```python
        # --- Contour analytique g=0 (flexion_claude) ---
        if print_ana:
            calc = flexion_claude()
            u1_a = np.linspace(u1_min, u1_max, n_grid)
            u2_a = np.linspace(u2_min, u2_max, n_grid)
            U1_a, U2_a = np.meshgrid(u1_a, u2_a)
            Z_ana = np.array([calc.g(u1, u2)
                              for u1, u2 in zip(U1_a.ravel(), U2_a.ravel())]
                             ).reshape(n_grid, n_grid)
            ax.contour(U1_a, U2_a, Z_ana, levels=[0], colors='green', linewidths=2,
                       zdir='z', offset=float(Z.min()))
            ax.contour(U1_a, U2_a, Z_ana, levels=[0], colors='green', linewidths=2)
```
  A inserer dans `print_3D_HF` apres le bloc `ax.contour(..., colors='black', ...)` et avant le bloc `if best_sol_modes_fixed`.
  Prerequis : verifier que `flexion_claude.g(u1, u2)` est definie et accessible (methode instance, pas statique — appel via `calc = flexion_claude()`).

**Probleme environnement (session 12/05) :**
`python launcher.py` depuis le terminal Claude Code → `ModuleNotFoundError: No module named 'STRAINS.rupt.core.CetCAD'`. Cause : Python appele n'est pas l'environnement conda avec STRAINS. A lancer depuis le terminal utilisateur avec l'environnement conda habituel.

### Session 18/05 — Refactoring FORM + codage EFF

**Ref :** aucun resume de session cree.

**Refactoring architecture FORM :**
- `FORM_init` remplace par deux fonctions : `init_g_ot(g_ot, xt, yt, all_grad)` + `init_FORM(g_ot, xt, yt, all_grad)`.
- `init_g_ot` : construit le metamodele selon la branche active (KRG/GEK/PCKRG/GEPCK/HF), retourne `g_ot, xt, yt, all_grad` (g_ot generique, pas de variables specifiques separees).
- `init_FORM` : appelle `init_g_ot`, cree `X`, `Y`, `event`, retourne `event, g_ot, xt, yt, all_grad`.
- Initialisation corps principal : `event, g_ot, xt, yt, all_grad = [None] * 5` (remplace la liste a 11 variables).
- **Toutes les signatures simplifiees :** `print_results(best_result, g_ot)`, `print_visu(best_result, best_sp, xt, g_ot, modes)`, `FORM_warm_start(modes, best_sps, g_ot)`. Les fonctions utilisent les flags `do_GEK`/`do_KRG`/`do_HF` directement pour brancher sur g_ot unique.
- Tous les appels call sites mis a jour en coherence.

**Codage EFF :**
- `def EFF` supprimee, calcul fusionne dans `class EFFFunction(ot.OpenTURNSPythonFunction)` : `__init__(self, result)` stocke le KrigingResult, `_exec(u)` calcule le critere EFF directement. Pas de `_exec_sample` (DIRECT appelle point par point).
- `run_EFF(result, g_ot, xt, yt, all_grad)` partiellement codee : DIRECT via `ot.NLopt(problem, "GN_DIRECT")`, `problem.setMinimization(False)` pour maximiser, bornes `u1/u2_eff_min/max` independantes des bornes visu. Manquant : evaluation HF en u_opt, rebuild metamodele, boucle while, return.

**OT doc — `getImplementation()` :** retourne une copie `FunctionImplementation` C++, PAS l'objet Python original. Les attributs Python (`self.sm`) sont inaccessibles via cette voie. Pour recuperer `sm_GEK` depuis `g_ot`, il faut le passer en parametre separe.

**Run test 18/05 (output_1805_0902.txt) :** n0=3 DOE fixe, do_GEK=True. beta=1.5364, Pf=6.22e-02, u*=[-1.197, -0.963], Imp. fc=60.7%/fy=39.3%, n_iter=44. Code tourne sans crash.

**Session 18/05 (suite) — generalisation sigma_func + debug + EFF enrichi :**
- `sigma_func` generalise pour toutes branches : KRG (`result.getConditionalMarginalVariance`), GEK (`gek_impl._exec_sigma`), PCKRG (`result_r.getConditionalMarginalVariance`), GEPCK (`gepck_impl._exec_sigma`).
- `GEKPLSFunction._exec_sigma` et `GEPCKFunction._exec_sigma` implementees (predict_variances SMT).
- `EFFFunction.__init__(g_ot, sigma_func)` : refactorise pour prendre sigma_func generique au lieu de result KRG.
- `run_EFF` : signature `(g_ot, sigma_func, xt, yt, all_grad)`, retourne `..., xt_eff` (liste des points ajoutes par EFF).
- `run_EFF` : prints diagnostic (EFF initial, iterations while, convergence), `xt_eff` accumule les nouveaux points.
- `print_visu(best_result, best_sp, xt, g_ot, modes, xt_eff)` : scatter rouge triangles pour les points EFF ajoutes.
- Corps principal : `init_g_ot` -> `run_EFF` -> `init_FORM` (sequence avec enrichissement EFF avant FORM).
- Run test 18/05 (output_1805_1239.txt) : n0=3, do_GEK=True. beta=1.5364 identique. EFF=0.0000 des le depart — boucle while non declenchee, 0 point ajoute.
- Run test 18/05 (output_1805_1436.txt) : n0=3, do_KRG=True. Meme resultat : EFF=0.0000, 0 point ajoute.
- **Bug EFF identifie :** EFF=0 car sigmaG≈0. Pour GEK (GEKPLS SMT) : `predict_variances` retourne ~0 partout (modele interpolant, nugget≈0, limitation SMT independante de n0). Pour KRG (OT) : cause non encore confirmee — `getConditionalMarginalVariance` devrait etre non nul loin du DOE. Prints diagnostics sigmaG/muG/epsilon ajoutes dans `run_EFF` apres u_opt. Run en cours (output_1805_1442.txt, do_KRG=True) pour lire sigmaG.
- **Prints diagnostic ajoutes dans `run_EFF` :** sigmaG, muG, epsilon au u_opt initial + a chaque iteration while.

**Session 18/05 (fin de journee) — corrections EFF + tests PCKRG + auto-upgrade max_degree :**

- **Bug EFF formule corrige (utilisatrice) :** valeur absolue ajoutee dans la condition while : `abs(f(u_opt)[0]) > tol_EFF`. Sans ca, EFF negatif (ex: -3.95e-4) passait sous tol mais pour mauvaise raison.
- **Bug init_g_ot PCKRG/GEPCK corrige :** quand `xt is not None` (appel depuis run_EFF), `y_hf`/`all_grad_hf` non definis → `UnboundLocalError`. Fix : `else: y_hf, all_grad_hf = yt, all_grad` dans les deux branches PCE.
- **Print visu legende u* :** `u* mode1 beta=X` → `u*1 [x,y] beta=X` et `u*k [x,y] beta=X` pour les modes supplementaires.
- **`do_EFF` flag ajoute (utilisatrice) :** permet de desactiver EFF sans supprimer la fonction.
- **Prints debug convergence EFF :** bloc detaille apres boucle while (u_opt, sigmaG, muG, t1/t2/t3, chaque terme, EFF final) avec flush=True.
- **Auto-upgrade max_degree dans run_EFF :** dans la boucle while, avant chaque appel `init_g_ot`, si `len(xt) >= (n_var+1)*(n_var+2)//2 + 1` et `max_degree < 2` → `global max_degree = 2` + print flush. Evite segfault KRG sur residu nul quand PCE interpole exactement (n0 < basis_size).
- **Formule n0_min :** `n0_min(p, n_var) = C(n_var+p, p) + 1`. Pour n_var=2, p=2 : n0_min=7. En dessous → PCE interpole exactement → residu=0 → segfault algo_KRG.run().

**Runs session 18/05 (apres-midi/soir) :**

| Output | Config | EFF | beta | Ecart HF | u* | Note |
|---|---|---|---|---|---|---|
| output_1805_1457.txt | KRG n0=5, do_EFF=True | 1 pt [-9.998,-9.998] | 2.2931 | -71.8% | [-1.683,-1.558] | EFF negatif = bug abs |
| output_1805_1512.txt | KRG n0=5, debug prints | 1 pt | ~2.29 | — | — | Debug convergence |
| output_1805_1518.txt | KRG n0=10, do_EFF=True | 4 pts coins domaine | 6.3812 | -21.4% | [+1.085,-6.288] | 2 modes ; mode2 beta=4.60 |
| output_1805_1618.txt | PCKRG n0=5, max_deg=1, do_EFF=True | 16 pts sur surface limite | 7.2931 | -8.7% | [-4.22,-5.94] | muG≈0 a chaque it. EFF |
| output_1805_1630.txt | PCKRG n0=5, max_deg=2, do_EFF=False | 0 | 7.2981 | -8.5% | [-4.203,-5.967] | EFF n'apporte rien vs no-EFF |
| output_1805_1635.txt | PCKRG n0=5, max_deg=2, do_EFF=True | — | CRASH | — | — | Segfault : n0=5 < n0_min=7 |
| output_1805_1641.txt | PCKRG n0=7, max_deg=2, do_EFF=True | 0 pts (EFF<tol) | 7.7504 | -2.9% | [-6.512,-4.203] | 1 mode |

**Observations cles session 18/05 :**
- KRG+EFF : EFF ajoute aux coins du domaine (sigma max loin du DOE), pas sur la surface limite → peu utile pour FORM.
- PCKRG+EFF : EFF ajoute sur la surface limite du surrogate (muG≈0) → enrichissement cible, mais beta quasi identique avec/sans EFF a n0=5 (7.29 vs 7.30). Le PCKRG n0=5 avait deja sa surface limite bien positionnee, sigmaG tres faible.
- PCKRG n0=7 : EFF=0 deja satisfait des le depart (sigmaG=0.000135). DOE de 7 pts avec PCE degree=2 laisse 1 degre de liberte → residu KRG tres faible partout.
- Pour tester l'apport reel de EFF+PCKRG : utiliser n0 entre n0_min et n0_min+quelques pts (ex n0=4 avec max_degree=1 initial, upgrade auto vers degree=2 en cours d'EFF).

---

### Session 20/05 — Visualisations EFF/sigma + plt.show non-bloquant

**Nouvelles fonctions de visualisation :**
- `print_visu_EFF(g_ot, sigma_func, xt, xt_eff)` : carte 2D isocouleur du critere EFF (cmap viridis), overlay contour g=0 surrogate, scatter DOE (blanc) + points EFF ajoutes (triangles rouges).
- `print_visu_sigma(g_ot, sigma_func, xt, xt_eff)` : carte 2D isocouleur de l'ecart-type conditionnel sigma_func (cmap plasma), meme structure.
- Toutes deux avec `plt.show(block=False)` → non-bloquantes.

**Correction plt.show non-bloquant :**
- `print_visu_EFF` et `print_visu_sigma` : `plt.show()` → `plt.show(block=False)`.
- Corps principal bloc `if do_EFF` : deux paires (avant enrichissement `xt_eff=[]`, apres avec `xt_eff`), chacune suivie d'un `plt.show()` bloquant → deux fenetres s'ouvrent simultanement a chaque etape.
- **Retour arriere** : remettre `plt.show()` dans les deux fonctions, supprimer les `plt.show()` du corps principal.

**KRG : ajout setOptimizationBounds theta [1, 5] :**
- `build_metamodel_KRG` : `algo_KRG.setOptimizationBounds(ot.Interval([1.0]*n_var, [5.0]*n_var))` + print theta/sigma apres run.
- Raison : eviter que TNC trouve theta<1 (rebound fort) ; bornes [1,5] = portee suffisante sans overfitting.

**GEK : ajout theta_bounds [1.0, 5.0] dans GEKPLS.**

**Run 20/05 (output_2005_0956.txt) — PCKRG n0=5, do_EFF=True, theta bounds [1,5] :**

| Output | Config | EFF pts | beta | u* | Ecart HF ref (7.9788) |
|---|---|---|---|---|---|
| output_2005_0956.txt | PCKRG n0=5, EFF=True, theta∈[1,5] | 4 | **7.9662** | [-6.117, -5.103] | **-0.16%** |

- n_iter FORM = 14, fc*=22.93 MPa, fy*=396.16 MPa, Imp. fc/fy=59.0%/41.0%
- dg/du_fc=0.051988, dg/du_fy=0.044047, Erreur FOSM=29.4%
- Resultat quasi identique au run 1905_0901 (beta=7.9652) — theta bounds n'ont pas change le resultat.

**Runs 20/05 — Serie GEPCK+EFF (debug architecture) :**

| Output | Config | bounds GEK | max_of_maxdeg | n0 | EFF pts | beta | Ecart HF (8.1235) | Note |
|---|---|---|---|---|---|---|---|---|
| output_2005_1135.txt | GEPCK | non | 2 (update_deg) | 5 | 10 | FAIL | — | theta erratique (1e-5 a 20), DBSCAN 0 modes |
| output_2005_1151.txt | GEPCK | [1,5] | 2 (update_deg) | 5 | 16 | FAIL | — | DBSCAN 0 modes ; EFF explose 0.004→0.301 au DOE=8 |
| output_2005_1156.txt | GEPCK | [1,5] | **1** | 5 | 6 | 7.258 | -10.6% | theta bloque [5,5] ; 2 modes tres proches |
| output_2005_1158.txt | GEPCK | non | **1** | 5 | 13 | 7.278 | -10.4% | theta bloque [20,20] |
| output_2005_1227.txt | GEPCK | non | 2 | **7** | **0** | **7.750** | **-4.6%** | max_degree=2 initial, PCE quasi-interpolante (LARS=6/6), sigma=0.000117 → EFF converge d'emblee |
| output_2005_1355.txt | GEPCK | non | 2 (update_deg) | 5 | 10 | FAIL | — | EFF explose 0.004→0.493 au DOE=8 (passage deg=1→2), point EFF au coin [10,10], DBSCAN 0 modes |
| output_2005_1546.txt | GEPCK | [1,5] | **1** | 5 | 6 | — | — | max_of_maxdegree=1 par erreur — identique a 1156, pas d'upgrade, PCE degree=1 stable |
| output_2005_1558.txt | GEPCK | [1,5] | 2 (update_deg) | 5 | **20** | FAIL | — | Pire que 1151 : LARS oscille 2/6→5/6→2/6 apres upgrade, EFF n'arrive jamais a stabiliser le surrogate |

**Polynomes PCE progressifs — output_2005_1558.txt (GEPCK, bounds=[1,5], max_of_maxdeg=2, n0=5) :**

| DOE | deg | LARS | PCE expression | sigma_GEK | EFF |
|-----|-----|------|----------------|-----------|-----|
| 5 | 1 | 3/3 | +0.6142·1 +0.0485·H1(u1) +0.0688·H1(u2) | — | — |
| 6 | 1 | 3/3 | +0.6142·1 +0.0485·H1(u1) +0.0689·H1(u2) | 0.00379 | 0.00516 |
| 7 | 1 | 3/3 | +0.6142·1 +0.0486·H1(u1) +0.0689·H1(u2) | 0.00349 | 0.00470 |
| 8 | 1 | 3/3 | +0.6142·1 +0.0486·H1(u1) +0.0689·H1(u2) | 0.00356 | 0.00475 |
| **9** | **2** | **2/6** | **+0.5162·1 -0.0111·H2(u2)** | **0.225** | **0.308** |
| 10 | 2 | 5/6 | +0.6103·1 +0.0395·H1(u1) -0.0014·H2(u1) -0.0111·H2(u2) -0.0058·H1·H1(u2) | 0.069 | 0.093 |
| **11** | **2** | **2/6** | **+0.4423·1 -0.0067·H2(u1)** | **0.241** | **0.287** |
| 12–15 | 2 | 2-3/6 | +0.44..·1 + quelques H2(ui) | ~0.19–0.23 | ~0.12–0.24 |
| 16–19 | 2 | 3/6 | +0.518·1 -0.005·H2(u1) -0.006·H2(u2) | ~0.19 | ~0.08–0.14 |
| 20–24 | 2 | 6/6 | +0.58·1 +0.012·H1(u1) +0.022·H1(u2) -0.007·H2(u1) -0.006·H2(u2) -0.002·H1·H1(u2) | ~0.09 | ~0.02 |
| conv | 2 | 6/6 | idem | 0.089 | ≈0 (20 pts) |

**Observations cles 20/05 — GEPCK+EFF :**
- La reference HF depuis sp=[0,0] est beta=8.1235. PCKRG n0=5+4EFF donne 7.9662 (-0.16%).
- **Cause de l'explosion EFF au DOE=8 :** upgrade degree=1→2 → LARS passe de 3/3 a 2/6 termes actifs (LOO plus conservateur avec 6 candidats) → PCE moins bonne qu'avant → residu plus grand → theta GEK instable → sigma max aux coins → EFF va au coin [10,10].
- **LARS oscille (observation cle 1558) :** apres upgrade, LARS selectionne 2/6 puis 5/6 puis 2/6 selon les points EFF ajouyes (tous aux coins). Les coins perturbent la LOO sans couvrir la zone utile. Le polynome reste instable 20 iterations.
- **Comparaison PCKRG (0934) vs GEPCK (1558) :** les deux explosent au DOE=8 (LARS=2/6). PCKRG se stabilise en 1 iteration supplementaire (DOE=9 → LARS=6/6, sigma=0.00073). GEPCK oscille 20 iterations. Cause racine : les points step 1-3 sont differents → DOE=9 different → LARS voit des donnees differentes.
- **max_of_maxdegree=1** evite l'explosion mais bloque a -10% (PCE degree=1 trop simple).
- **n0=7 + max_degree=2 initial** : meilleur resultat GEPCK (-4.6%) mais PCE quasi-interpolante → GEK theta=[20,20] → EFF=0 pts. Le GEK n'apporte rien.
- **Conclusion :** l'architecture GEPCK est fondamentalement problematique : soit la PCE est trop simple (degree=1, -10%) soit elle interpole le DOE et le GEK s'entraine sur du bruit. PCKRG (KRG OT sur residu) reste superieur dans toutes les configs testees.

---

### Session 19/05 — Update degree dynamique + nettoyage code + visu trajectoires FORM

**Modifications code :**
- `try_pce` supprime partout (variable obsolete) : `elif do_KRG and try_pce:` → `elif do_PCKRG:`, `elif do_GEK and try_pce:` → `elif do_GEPCK:`. `do_pce = try_pce` supprime.
- `modele = 'KRG'` ajoute en OPTIONS : `do_KRG/GEK/HF/PCKRG/GEPCK` calcules par ternaire `True if modele == 'X' else False`.
- `n0_min(n_var, p)` + `update_degree(new_n0)` : calcul dynamique de max_degree depuis 0, s'arrete quand `new_n0 > n0_min(n_var, max_degree+1)` avec cap `max_of_maxdegree` (OPTIONS). `max_of_maxdegree = 2` par defaut.
- `update_degree` appele dans `run_EFF` a chaque point ajoute → upgrade automatique de degree en cours d'enrichissement.
- `traj_runs_fixed` hardcode en OPTIONS (4 modes A/B/C/D, 12-50 pts chacun, run 1805_1957). Affichage conditionne par flag `print_traj` dans `print_visu` : polylignes colorees + point de depart (cercle) + point final (etoile).

**Resultat cle — EFF+update_degree resout le pb des 16 points (PCKRG n0=5 session 18/05) :**
- Avant (max_degree fixe a 1) : EFF ajoutait 16 pts sur la surface limite (muG≈0, sigmaG faible mais non nul a chaque it.) → DOE explose.
- Apres (update_degree dynamique) : EFF ajoute **4 pts** puis converge. L'upgrade 1→2 au 4e point (DOE=8 > n0_min(2)=7) change radicalement le surrogate → sigmaG chute → EFF converge rapidement.

| Output | Config | EFF pts | max_deg final | beta | u* |
|---|---|---|---|---|---|
| output_1905_0856.txt | KRG n0=5, do_EFF=False | 0 | 1 | 7.2981 | [-4.203, -5.967] |
| output_1905_0901.txt | KRG n0=5, do_EFF=True | **4** | **2** (upgrade it.4) | **7.9652** | [-6.104, -5.117] |

---

### Session 15/05 — Analyse theorique artefact rebound surface KRG

**Ref :** aucun resume de session cree.

- **Artefact rebound surface KRG explique :** la zone rouge (g<0) formee par le surrogate est une "ile" fermee au lieu d'une region semi-infinie comme le HF. Formule : `g_hat(x) = beta0 + r(x)^T R^-1 (Y - beta0)`. Quand x s'eloigne du DOE, r(x)->0 (kernel SE), la correction s'annule, le surrogate revient vers beta0>0. Ce n'est pas un faux point invente : c'est le terme de correction qui disparait en absence de donnees.
- **beta0 :** estimateur GLS = `(1^T R^-1 1)^-1 * 1^T R^-1 Y` ≈ moyenne(g_DOE) > 0 car la majorite des points LHS tombent en zone g>0 (zone de defaillance petite relativement a [-10,10]^2).
- **Alternative `LinearBasisFactory` :** `ot.LinearBasisFactory(n_var).build()` a la place de `ConstantBasisFactory` (l.701 du script). Reversion vers plan lineaire beta0+beta1*u1+beta2*u2 au lieu de constante — si la regression lineaire capture la pente descendante, attenuation du rebound. Non teste.

### Session 19/05 — Tests leviers KrigingAlgorithm contre le rebound

**Ref detaillee :** `2026_1205 résultats et comparaisons\solution_krigeage_1905.md`

- **ConstantBasis baseline (n0=5, TNC) :** beta=3.856, u*=[-2.27,-3.12], Erreur FOSM=97.6%, ecart HF=-52.5%. Instable : meme config donnait 7.298 dans un run precedent (TNC converge vers des theta differents).
- **LinearBasisFactory (v1, n0=5, TNC) :** beta=7.298, u*=[-4.20,-5.97], Erreur FOSM=5.7%, ecart HF=-10.2%, n_iter=5 (vs 43). **Levier principal confirme.** Corrige le rebound, reproductible.
- **ConstantBasis + theta=10 gele (setOptimizeParameters=False) :** ECHEC TOTAL. sigma=1.0 aussi gele → surface mal calibree → FORM diverge hors domaine, 0 modes. A eviter.
- **ConstantBasis + theta=5 gele :** beta=5.463, u*=[-5.10,-1.95], Erreur FOSM=82.9%, ecart HF=-32.8%. Mieux que theta=10 (pas de divergence) mais u* faux, sigma toujours mal calibre.
- **Prochaine etape :** a definir par l'utilisatrice.

**Modifications code session 15/05 — activation branche PCE-KRG et GEPCK :**

- **`PCKRGFunction` ajoutee :** wrapper OT (`_exec` + `_exec_sample`) combinant `g_pce + g_krg` pour FORM. Corrige le bug critique : FORM travaillait sur le residu seul au lieu du modele complet.
- **`GEPCKFunction` ajoutee :** wrapper OT (`_exec` + `_exec_sample` + `_gradient`) combinant `g_pce + sm_gepck`. Gradient analytique = PCE.gradient() + sm.predict_derivatives().
- **`_exec_sample` ajoute a `GEKPLSFunction` :** appel vectorise `predict_values(np.array(U))` sur grille 300x300 — evite 90000 appels Python.
- **Bug NameError GEPCK corrige :** variable `smr_GEPCK` → `smr_GEK` dans `FORM_init` branche `do_GEK and try_pce`.
- **`FORM_init` branches PCE mises a jour :** utilise `build_PCE` (retourne y_PCE, all_grad_PCE) + calcul residus inline. `yt, all_grad = y_hf, all_grad_hf` (totaux, pas residus) pour future compatibilite warm start.
- **`print_visu` refactorise :**
  - Signature : `sm_GEK` → `g_ot_GEK`, ajout `g_ot_PCKRG`, `g_ot_GEPCK`
  - Fond colore : SMT API → OT API pour GEK ; blocs `elif g_ot_PCKRG` et `elif g_ot_GEPCK` ajoutes
  - Appel unique `print_visu(..., g_ot_GEK, g_ot_KRG, g_ot_PCKRG, g_ot_GEPCK, modes)` (variables globales = None par defaut)
  - Appel `best_sol_modes_fixed` mis a jour (2 None supplementaires)

**Modifications code suite session 15/05 (apres-midi) :**

- **`print_results` refactorise :** signature `(best_result, g_ot_GEK, g_ot_KRG, g_ot_PCKRG, g_ot_GEPCK, g_ot_HF)`. Blocs `if g_ot_PCKRG is not None` et `if g_ot_GEPCK is not None` ajoutes — meme structure que KRG (run_HF(u*) pour gradients + FOSM). Labels `HF@u*PCKRG` et `HF@u*GEPCK`. Call sites mis a jour (2 appels, modes 1 et 2).

**Referance HF pour comparaison surrogates :**
- **beta reference commune = 8.1235** (sp=[0,0] → u*=[-4.776,-6.571]) — point de depart commun a tous les runs metamodele. Toujours comparer les surrogates a cette valeur, pas au "meilleur" mode HF (7.9788 atteint seulement depuis certains sp).

**Runs comparatifs GEPCK vs PCKRG (n0 variable, sans warm start) :**

| n0 | GEPCK beta | GEPCK u* | PCKRG beta | PCKRG u* | HF ref (sp=[0,0]) |
|---|---|---|---|---|---|
| 3 | **1.5364 (-81%)** | [-1.197, -0.963] | ECHEC (aucun FORM) | — | 8.1235 |
| 6 | 7.6856 (-5.4%) | [-4.68, -6.10] | 7.6856 (-5.4%) | [-4.68, -6.10] | [-4.776, -6.571] |
| 8 | 7.4445 (-8.4%) | [-5.73, -4.75] | TBD | TBD | [-4.776, -6.571] |
| 12 | 8.5277 (+5.0%) | [-6.85, -5.08] | 8.5277 (+5.0%) | [-6.85, -5.08] | [-4.776, -6.571] |

**Observations cles :**
- n0=6 : GEPCK = PCKRG (meme DOE graine OT fixe, meme resultat) — differentiation non visible encore
- n0=3 PCKRG : tous FORM echouent (RuntimeError) — surrogate sans zero accessible. PCE degree 1 exactement determine sur 3 pts, residu KRG trop pauvre.
- n0=3 GEPCK : **TERMINE** (output_1505_1117) — beta=1.5364, PCE degeneree (LARS=1 coeff=constante), avantage GEPCK vient de GEKPLS seul sur le residu.
- n0=8 GEPCK : 7/9 bruit DBSCAN (surrogate degrade vs n0=6)
- n0=12 GEPCK=PCKRG : meme graine → meme DOE → meme resultat, 11/13 bruit
- **Diagnostic PCE n0=3 (print ajout 15/05) :** basis_size=3, LARS selectionne 1 coeff (constante). LARS/CorrectedLeaveOneOut rejette u1/u2 avec seulement 3 points — pas assez de donnees pour valider les termes lineaires. PCE = beta0 uniquement quel que soit le max_degree.

**Note graine OT :** OT PRNG fixe → LHS identique pour GEPCK et PCKRG au meme n0. Pour isoler l'apport des gradients, il faudrait comparer sur meme DOE fixe OU moyenner sur plusieurs graines.

**Detail :** `2026_1205 resultats et comparaisons\comparaison_GEK_KRG_1305.md` (colonnes PCKRG/GEPCK 1505).

---

### Session 13/05 — Surface 3D analytique + gradients aux sp

**Objectif :** comprendre geometriquement pourquoi FORM converge vers les 4 modes A/B/C/D observes le 12/05. Deux outils developpes :

**1. Double surface 3D dans `print_3D_HF` :**
Superposition de g_HF (STRAINS, grille 7x7 hardcodee) et g_ana (flexion_claude, grille n_grid x n_grid calculee a la volee) sur le meme axe 3D.
- g_HF : surface rouge transparente (`color='red', alpha=0.3`) + contour g=0 rouge projete et darkred sur surface
- g_ana : surface bleue transparente (`color='blue', alpha=0.3`) + contour g=0 vert projete et vert sur surface
- u* A/B/C/D : etoiles colorees a z=0 ; sp A/B/C/D : croix colorees a z=0
- Permet de voir visuellement l'ecart entre la surface HF et la surface analytique, et de localiser les 4 modes par rapport a la frontiere g=0.

**2. Gradients HF aux points de depart sp (4 appels STRAINS, output_1305_0937) :**
Bloc `if print_grad_sp:` dans le corps principal : evalue `run_HF` aux 4 sp, imprime g_HF(sp), grad(sp), -grad(sp).
Objectif : voir dans quelle direction FORM part depuis chaque sp, et comprendre vers quel mode il converge.
Resultats hardcodes dans `grad_sp_fixed` (meme bloc conditionnel que `best_sol_modes_fixed`) :

| Mode | sp | g_HF(sp) | grad(sp) | -grad(sp) |
|------|-----|----------|----------|-----------|
| A | (0.868, -1.460) | 0.550023 | [0.0395, 0.0723] | [-0.0395, -0.0723] |
| B | (0.745, 1.043) | 0.722565 | [0.0461, 0.0690] | [-0.0461, -0.0690] |
| C | (-0.843, -0.005) | 0.573155 | [0.0582, 0.0593] | [-0.0582, -0.0593] |
| D | (-0.694, 2.114) | 0.704360 | [0.0685, 0.0553] | [-0.0685, -0.0553] |

**Observation :** tous les -grad(sp) pointent en direction (u1<0, u2<0), coherent avec les u* tous dans le quadrant negatif. Les composantes varient selon le mode : mode A plus sensible a u2 (fy), mode D plus sensible a u1 (fc).

**3. Refactoring FORM_event → FORM_init + FORM_warm_start (13/05, apres-midi) :**

- **`FORM_init(sm_GEK, sm_GEPCK, g_ot_KRG, g_ot_GEK, g_ot_PCKRG, g_ot_GEPCK, g_ot_HF, xt, yt, all_grad)`** : regroupe build_DOE + build_metamodel + creation event. Si `xt is not None` (appel warm start), `build_DOE()` est saute — les valeurs passees sont utilisees directement. Retourne tous les objets mis a jour + `sm_GEK, sm_GEPCK`.
- **`FORM_warm_start(modes, best_sps, g_ot_GEK, g_ot_KRG, g_ot_HF)`** : si `g_meta(u*) > tol_warmstart`, augmente DOE avec u* (xt_ws, yt_ws, all_grad_ws), rappelle `FORM_init` avec les valeurs augmentees, relance `FORM_all_modes`.
- **Fix DBSCAN 1 point :** `FORM_all_modes` — si `len(all_u_star)==1`, retourne directement sans DBSCAN. Fix `return []` → `return [], []` pour le cas tous FORM echouent.
- **Initialisation variables :** `event, sm_GEK, sm_GEPCK, g_ot_GEK, ... = [None] * 11` dans le corps principal.
- **WARNING — bug latent try_pce + warm start :** dans `FORM_init`, branches `do_KRG and try_pce` et `do_GEK and try_pce` : si `xt` est non-None (appel warm start), `build_DOE()` est saute mais `y_hf` et `all_grad_hf` ne sont alors pas definis → NameError sur les lignes suivantes. Non bloquant car `try_pce=False` actuellement. A corriger avant de reactiver PCE.

- **Run GEK n0=10 (output_1305_1354) :** 2 modes. Mode 1 : beta=2.9923, u*=[-0.566,-2.938], Imp. fc/fy=3.6%/96.4%. Mode 2 : beta=3.112, u*=[-2.642,-1.644], Imp. fc/fy=72.1%/27.9%. Surrogate GEK avec n0=10 ne couvre pas la vraie frontiere g=0 (min g_GEK≈0.40 dans domaine) — FORM s'arrete a g=0.40 < tol_FORM=1.0.

---

## 4. Resultats obtenus

### 4.1 Ancienne geometrie — F=0.210 MN, n0=15, DOE fixe (sessions 23-24/04)

**Reference absolue :** `C:\_workingDir\_SF\test flexion\(ancien) résultats et comparaison\resultats_HF_run2.md`
**Detail complet :** `C:\_workingDir\_SF\test flexion\(ancien) résultats et comparaison\comparaison_a_DOE_fixe.md`

| Methode | beta | Erreur beta vs HF | g_meta(u*) | g_HF(u*) | n_iter |
|---|---|---|---|---|---|
| **HF reference** | **3.784** | 0% | ≈0 | ≈0 | 21 |
| PCE-KRG (session 2204) | 3.790 | +0.16% | ≈0 | ? | 15 |
| PCE-KRG (DOE fixe) | **3.779** | **-0.1%** | ≈0 | N/A | 15 |
| KRG pur (run3, DOE fixe) | 5.065 | +33.9% | +9.3e-05 | -4.96e-02 | 24 |
| GEK pur (run1, no WS, DOE fixe) | 2.118 | -44.0% | +0.074 | ~+0.074 | 37 |
| GEK pur (2404 run1, nouvelle version) | 3.509 | -7.3% | +0.015 | +0.015 | 22 |
| GEK pur (2404 run2, nouvelle version) | 2.154 | -43.1% | +0.070 | +0.071 | 32 |
| GEK pur (2404 run3, nouvelle version) | 1.487 | -60.7% | +0.098 | +0.098 | 16 |
| GEK pur (2404 run4, ancienne version) | **3.774** | **-0.26%** | **~0** | +0.001 | — |
| GEK+WS | 1.914 | -49.4% | +0.113 | +0.113 | 1 |
| PCE-KRG+WS | 1.644 | -56.6% | +0.091 | +0.091 | 1 |
| GEPCK+WS | 0.991 | -73.8% | +0.119 | +0.119 | 1 |
| GEPCK (session 2404) | **3.779** | **-0.1%** | 6e-06 | 3.45e-04 | 15 |

### 4.2 Nouvelle geometrie — b=0.5m, h=0.8m, 2 lits 3HA16, Block1+Block2, fck=40, fyk=500 (session 30/04)

**Config :** b=0.5m, h=0.8m, L=5m, phi=16mm, z_lit1=0.328m, z_lit2=0.312m, Block1 (ft1=0.1, 0-4m), Block2 (ft2=3.5, 4-5m), F=0.1 MN.

| Methode | beta | Pf | u* | fc* (MPa) | fy* (MPa) | n_iter | Erreur FOSM | Ecart beta vs HF |
|---|---|---|---|---|---|---|---|---|
| **HF direct (output_3004_1203)** | **0.3915** | 6.52e-01 | [0.0016, 0.3915] | 47.8138 | 561.3999 | 6 | 0.46% | — |
| GEK n0=35 (output_3004_1447) | 0.3771 | 6.47e-01 | [0.0059, 0.3771] | 47.8323 | 560.9651 | 1 | 4.65% | -3.68% |

**Observations :** Importance fc=0%, fy=100% — defaillance exclusivement par plastification acier. F=0.1 MN trop faible pour mobiliser le beton (beta=0.39, Pf=65%). g quasi-lineaire (FOSM 0.46%). Un seul mode detecte sur 36 points de depart GEK.

### 4.3 Géométrie b=0.2, h=0.5, 3×1HA32, phi=32mm — fck=63, fyk=550 (session 30/04)

**Config :** b=0.2m, h=0.5m, L=5m, phi=32mm, 3 lits 1HA32 (z=+0.202/+0.170/+0.138m), Block1 (ft1=0.1, 0-4m), Block2 (ft2=3.5, 4-5m).

| Methode | F (MN) | beta | Pf | u* | fc* (MPa) | fy* (MPa) | n_iter | Erreur FOSM | Regime |
|---|---|---|---|---|---|---|---|---|---|
| GEK n0=20 (output_3004_1807) | 0.12 | 2.1842 | 9.8553e-01 | [0.327, 2.160] | 72.47 | 664.71 | 1 | 5.94% | **Inverse** — pre-fix d |
| GEK n0=20 (output_0405_0718) | 0.12 | 2.1841 | 9.8552e-01 | [0.327, 2.1595] | 72.46 | 664.70 | 1 | 5.95% | **Inverse** — verification print_visu |
| GEK n0=20 (output_0405_0810) | 0.12 | 2.1842 | 9.8551e-01 | [0.3269, 2.1592] | 72.46 | 664.70 | 1 | 5.96% | **Inverse** — apres fixes + print_error_ana_hf |
| GEK n0=20 (output_0405_0819) | 0.12 | 2.1838 | 9.8551e-01 | [0.3271, 2.1596] | 72.47 | 664.71 | 1 | 5.94% | **Inverse** — confirmation |

**Observations :** u* positif sur les deux composantes → origine en domaine de defaillance → F=0.12 MN trop grande. beta=2.18, Pf=0.985 = Φ(+2.18) : regime inverse, pas le regime cible. Diminuer F pour viser u* negatif, beta≈3-4, Pf≈10⁻³–10⁻⁴.

**print_error_ana_hf (output_0405_0810 et 0405_0819) :** 37 points sur la frontiere analytique → 19 retenus (1 sur 2). g_HF ≈ +0.02 sur toute la frontiere, err_rel_moy=8.08, std(g_HF)=0.0025. Decalage systematique STRAINS/analytique de ≈2% (STRAINS plus permissif). Identique sur les deux runs → resultat reproductible.

### 4.4 Geometrie b=0.4, h=0.45, 2x2HA32, phi=32mm — session 04/05

**Config :** b=0.4m, h=0.45m, L=5m, phi=32mm, 2 lits de 2HA32 : Lit1 (HA1,HA2, z=+0.165m, 60mm du bord), Lit2 (HA3,HA4, z=+0.125m, 100mm du bord). Centroide acier z=0.145m (80mm du bord tendu). Block1 (ft1=0.1, 0-4m), Block2 (ft2=3.5, 4-5m), CONNECT GLUED. calc_ana avec d = h/2 + z_centroide = 0.225 + 0.145 = 0.370m.

| Run | Output | n0 | fck | F (MN) | beta | Pf | u* | fc* (MPa) | fy* (MPa) | Imp. fc/fy | FOSM | Regime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5a | output_0405_0853 | 20 | 63 | 0.12 | 1.9303 | — | — | — | — | <1%/~100% | 1.61% | Normal |
| 5b | output_0405_0919 | 5 | 48 | 0.12 | 1.7542 | — | — | — | — | — | 29.4% | Normal — n0=5 instable |
| 5c | output_0405_0957 | 5 | 48 | 0.11 | 1.8108 | 3.50e-02 | [-1.393, -1.158] | 30.26 | 564.67 | 59.1%/40.9% | 9.05% | Normal |
| 5d | output_0405_1035 | 5 | 48 | 0.11 | 1.8108 | 3.50e-02 | [-1.393, -1.158] | 30.26 | 564.67 | 59.1%/40.9% | 9.05% | Normal — + print_error_ana_hf |
| 5e | output_0405_1446 | 5 | 48 | 0.11 | 1.9271 | 2.70e-02 | [-1.393, -1.158] | 30.26 | 564.69 | 59.1%/40.9% | 9.05% | Normal — scan u2 + filtre brentq |

**dg/du en u* (runs 5c et 5d) :** dg/du_fc=0.035509, dg/du_fy=0.028845.

**Tableau print_error_ana_hf (run 5e, output_0405_1446) — b=0.4, h=0.45, F=0.11, fck=48, fyk=550 :**

| pt | u1 | u2 | fc (MPa) | fy (MPa) | g_HF | g_ana | err_abs |
|---|---|---|---|---|---|---|---|
| 0 | -1.51 | -0.38 | 29.84 | 588.00 | +0.0248 | 0.0000 | 0.0248 |
| 1 | -1.20 | -0.64 | 30.96 | 580.27 | +0.0262 | 0.0000 | 0.0262 |
| 2 | -0.86 | -0.90 | 32.25 | 572.54 | +0.0292 | 0.0000 | 0.0292 |
| 3 | -0.48 | -1.15 | 33.75 | 564.81 | +0.0333 | 0.0000 | 0.0333 |
| 4 | -0.06 | -1.41 | 35.50 | 557.08 | +0.0367 | 0.0000 | 0.0367 |
| 5 | +0.42 | -1.67 | 37.58 | 549.35 | +0.0373 | 0.0000 | 0.0373 |
| 6 | +0.96 | -1.92 | 40.08 | 541.62 | +0.0374 | 0.0000 | 0.0374 |
| 7 | +1.57 | -2.18 | 43.14 | 533.89 | +0.0388 | 0.0000 | 0.0388 |
| 8 | +2.28 | -2.44 | 46.95 | 526.15 | +0.0392 | 0.0000 | 0.0392 |
| 9 | +3.11 | -2.69 | 51.84 | 518.42 | +0.0322 | 0.0000 | 0.0322 |
| 10 | +4.09 | -2.95 | 58.30 | 510.69 | +0.0241 | 0.0000 | 0.0241 |

**err_abs_moy = 0.0327**

### 4.5 Exploration branche verticale — synthese des directions essayees (session 04/05)

**Objectif :** trouver un jeu de donnees ou la frontiere g_ana=0 a des zeros pour u2 > -0.38.

**Config de reference :** b=0.4, h=0.45, 2x2HA32, L=5m, F=0.11 MN, fck=28, fyk=550, n0=20. Premier zero frontiere analytique a u2=-0.38 (output_0405_1817, 11 pts).

| Direction | Modification | Output | n_pts frontiere | u2_min frontiere | Observation |
|---|---|---|---|---|---|
| Agrandir fenetre scan | size_visu : 5 → 7 | output_0405_1543 | 11 | -0.38 | Aucun nouveau point |
| Reduire h | h=0.45→0.4, F=0.094 MN | output_0405_1628 | 9 | -1.97 | Frontiere decalee vers u2 plus negatif |
| Reduire b | b=0.4→0.3, F=0.094 MN | output_0405_1724 | 7 | -1.92 | Frontiere comparable a h=0.4 |
| Modifier armatures | 3 lits 3HA25 (phi=25mm), F=0.15 MN | output_0405_1744 | 0 | — | Aucun zero trouve — regime inverse |

### 4.6 Session 05/05 — Nouvelle geometrie 2x3HA32 + exploration fyk/F

**Modifications dsCad (05/05) :** b=0.4, h=0.45, phi=32mm, **2 lits de 3HA32** (6 barres), z_lit1=0.161m, z_lit2=0.145m, y=±0.10 et 0. Centroide a 80mm du bord. sensitivity_regions : HA1–HA6. DBSCAN ajoute dans FORM_all_modes.

| Run | Output | F (MN) | fck | fyk | Config | beta | Pf | u* | Imp. fc/fy | Regime | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| — | output_0505_0738 | 0.2 | 28 | 550 | 2x3HA32 b=0.4 h=0.45 | 3.137 | 9.99e-01 | [2.782, 1.451] | 78.6%/21.4% | **Inverse** | — |
| — | output_0505_0746 | — | — | — | — | — | — | — | — | — | IndentationError DBSCAN |
| — | output_0505_0747 | 0.2 | 28 | 550 | 2x3HA32 b=0.4 h=0.45 | 3.117 | 9.99e-01 | [2.763, 1.443] | 78.6%/21.4% | **Inverse** | — |
| — | output_0505_0757 | 0.094 | 28 | 550 | 2x3HA32 b=0.4 h=0.45 | — | — | — | — | **Fail** | g_GEK=0.404 partout, section trop forte |
| — | output_0505_0759 | 0.094 | 28 | 550 | 2x3HA32 b=0.4 h=0.45 | — | — | — | — | **Fail** | g_GEK=0.409 partout |
| — | output_0505_0804 | 0.094 | 28 | **300** | 2x3HA32 b=0.4 h=0.45 | 1.974 (MS) / 2.100 (mode1) | 2.42e-02 | [0.187, -1.965] | 0.9%/99.1% | Normal | fyk=300 → FORM converge. err_abs_moy=0.030 |
| — | output_0505_0842 | 0.094? | 48 | 550 | **2x3HA16 b=0.5 h=0.8** | 2.619 | 4.42e-03 | [0.004, -2.619] | 0%/100% | Normal | Imp. fy=100%. FOSM 0.42% (g lineaire) |
| — | output_0505_0905 | 0.094? | 48 | 550 | **2x3HA16 b=0.5 h=0.8** | 2.619 | 4.41e-03 | [0.004, -2.619] | 0%/100% | Normal | Identique 0842 |
| — | output_0505_0909 | 0.2 | 28 | 550 | 2x3HA32 b=0.4 h=0.45 | 3.129 | 9.99e-01 | [2.774, 1.448] | 78.6%/21.4% | **Inverse** | print_error_ana_hf non capture (buffer) |
| — | output_0505_0929 | 0.11 | 28 | 550 | **2x2HA32** b=0.4 h=0.45 | 1.576 (MS) / 1.817 (mode1) | 5.76e-02 | [-1.108, -1.120] | 49.5%/50.5% | Normal | Config reference restauree. err_abs_moy=0.0327 = identique 0405_1446 |

**Observations session 05/05 :**
- F=0.2 avec fyk=550 et 2x3HA32 → regime inverse systematique (u* positif, Pf≈100%).
- F=0.094 avec fyk=550 → FORM fail (section trop forte, g_GEK>>0 partout). F_crit≈0.145 MN.
- F=0.094 avec **fyk=300** → FORM converge, branche verticale a u2=-1.97. Mais fyk=300 non physique.
- b=0.5/h=0.8/2x3HA16/fck=48/fyk=550 → beta=2.619, Imp. fy=100% (beton non sollicite). Pas de branche verticale visible.
- Conclusion : pour avoir u2>-0.38 avec fyk=550, il faut augmenter F au-dela de F_crit ou trouver une geometrie ou fc est plus sollicite.

**Detail complet :** resume_session_0505.md

---

### 4.7 Session 06/05 — Refactoring lois + calcul analytique + fix print_f

**Modifications code (06/05) :**
- **Suppression imports jcss_fy/jcss_fc :** `loi_fy`, `loi_fc`, `SIGMA` definis localement.
- **Refactoring flexion_simple :** `g_ana_plast`/`g_ana_nonplast`/`g_ana` → `f_plast`/`f_nonplast`/`f`. Methode `print_f` ajoutee. Fix NameError + vectorisation meshgrid.

| Run | Output | n0 | fck | fyk | Config | beta | Pf | u* | Imp. fc/fy | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| — | output_0605_1442 | 2 | 28 | 550 | 2x2HA32 b=0.4 h=0.45 F=0.11 | 2.117 | 1.71e-02 | [-1.381, -1.605] | 42.5%/57.5% | n0=2, run de test. 0 mode DBSCAN. Non significatif. |

### 4.8 Session 07/05 — Refonte complete de la fonction analytique

- **`flexion_simple` / `calc_ana` / `f_plast` / `f_nonplast` / `f` / brentq : SUPPRIMES.**
- **Nouvelle classe `flexion_claude` :** lit `dsCad.txt` et `dsLoad.txt` directement (regex), calcule les constantes A, B, C pour la branche pivot B (aciers plastifies), formule directe `u2p_LS(u1)` sans brentq. Limite de plastification `u1_lim_plast` calculee analytiquement.
- **`print_visu_ana()` :** nouvelle fonction qui trace la courbe analytique seule (branche plastifiee + branche verticale).
- **`print_visu` mis a jour :** signature `(best_result, best_sp, xt, sm_GEK, g_ot_KRG, modes)`. Courbe analytique calculee en interne via `flexion_claude()`. `print_HF` (bool global) controle la grille HF.
- **Distributions :** `fcm/fym/cov_fc/cov_fy` ; `loi_fc` avec COV_TABLE par classe ; `loi_fy` sigma JCSS depuis SIGMA_11/12/13 si `cov_fy=None`.

### 4.9 Session 12/05 — Run HF direct, gamma=1.0, geometrie 3 lits 8HA32

**Config :** b=h=0.8m, 3 lits 8HA32 (24 barres), phi=32mm, F=0.74 MN, gamma_c=gamma_s=1.0, fcm=48, fym=550, cov_fc=0.12, n0=15, do_HF=True, do_multistart=True.
**Output :** `output/output_1205_2032.txt` — **Detail :** `2026_1205 résultats et comparaisons\resultats_run_HF.md`

| Mode | beta | Pf | u* | fc* (MPa) | fy* (MPa) | Imp. fc/fy | FOSM |
|---|---|---|---|---|---|---|---|
| **1 (best)** | **7.9788** | 7.39e-16 | [-3.117, -7.345] | 32.83 | 328.55 | 15%/85% | 17.7% |

**Clusters u* identifies (16 points de depart, DBSCAN eps=0.01) :**
- **A** : u*≈[-3.0, -7.35], beta≈7.96–7.99 ← minimum (6 points)
- **B** : u*≈[-4.7, -6.6], beta≈8.10–8.12 (4 points)
- **C** : u*≈[-5.3, -6.2], beta≈8.15–8.17 (3 points)
- **D** : u*≈[-6.5, -5.0], beta≈8.17–8.18, u2 maximal (2 points)

**`best_sol_modes_fixed` (represenants A/B/C/D) :** stockes dans les OPTIONS du script pour affichage sans relancer FORM.

**Observations :** beta=7.98 tres eleve → coherent avec gamma=1.0 (pas de coefficients de securite). Tous u* en quadrant (u1<0, u2<0), branche horizontale pivot B. Surface limite non-lineaire (FOSM=17.7%). u1 du meilleur mode : fc*=32.8 MPa << fcm=48 MPa.

---

## 5. Inventaire des fichiers de resultats et comparaison

**Organisation des dossiers (mise a jour 20/05) :**

**DOSSIER ACTIF UNIQUE :** `C:\_workingDir\_SF\test flexion\2026_1205 résultats et comparaisons\`
Tous les nouveaux .md de resultats vont UNIQUEMENT dans ce dossier. Ne pas chercher ailleurs.

**Dossiers archives (ne pas modifier, ne pas consulter sans demande) :**
- Sessions 16/04-24/04 : `C:\_workingDir\_SF\test flexion\(ancien) résultats et comparaison\`
- Sessions 24/04-29/04 : `C:\_workingDir\_SF\test flexion\archive\nouveau résultats et comparaison\` (resultats incoherents)
- Sessions 30/04-11/05 : `C:\_workingDir\_SF\test flexion\2026_3004 résultats et comparaisons\`

### 5.1 Runs anciens

Tous les fichiers ci-dessous sont dans : `C:\_workingDir\_SF\test flexion\(ancien) résultats et comparaison\`

#### Fichiers resultats (lignes = metriques, colonnes = runs ou n0)

| Fichier | Objet | Etat |
|---|---|---|
| `resultats_HF.md` | FORM HF phi=32mm F=0.235 (run 1) | Obsolete — phi=32mm abandonne |
| `resultats_HF_run2.md` | FORM HF phi=16mm systematique (F=0.60 a 0.195 MN) | **Reference absolue** — complet |
| `resultats_KRG_run2.md` | FORM KRG n0 variable, 3 sections F=0.235/0.225/0.210 | Complet — sessions 17-20/04 |
| `resultats_KRG_run3.md` | KRG pur run3 DOE fixe n0=15 F=0.210 | Complet — session 23/04 |
| `resultats_comparaison.md` | Placeholder phi=16mm F=0.235 | Vide/inutilise |

### Fichiers comparaison (lignes = metriques, colonnes = methodes, tableaux = valeurs de F)

| Fichier | Objet | Colonnes actuelles |
|---|---|---|
| `comparaison_HF_KRG.md` | HF vs KRG, impact n0, plusieurs F | HF | KRG n0=60/40/25/20/15/15+1 (F=0.210), etc. |
| `comparaison_GEK_KRG.md` | GEK vs KRG, DOE identique | HF | KRG n0=20+1 | GEK n0=20+1 (et n0=5+1) |
| `comparaison_KRG_PCEKRG.md` | KRG pur vs PCE-KRG, DOE fixe n0=15 | HF | KRG pur n0=15 | PCE-KRG n0=15 |
| `comparaison_Matern_exp.md` | Matern 5/2 vs SquaredExponential | HF | KRG Matern | KRG squar_exp (3 tableaux) |
| `comparaison_a_DOE_fixe.md` | Toutes methodes sur DOE fixe n0=15 | HF | KRG | GEK | GEK+WS | PCE-KRG+WS | GEPCK+WS |

### 5.2 Runs actifs (a partir du 12/05)

Tous les nouveaux fichiers sont dans : `C:\_workingDir\_SF\test flexion\2026_1205 résultats et comparaisons\`

---


## 7. Bugs actifs et points techniques cles

### Bugs actifs

| # | Description | Bloquant ? | Etat |
|---|---|---|---|
| Bug A | RuntimeError FORM : g_meta(u*) > tol_FORM=0.2 → RuntimeError OT meme avec setCheckStatus(False) | Non bloquant — FORM_multistart attrape avec try/except | Contourne |
| Bug B | try_pce / branches PCE commentees | Non bloquant — PCE desactive intentionnellement dans version actuelle | Non bloquant |
| Bug C | build_metamodel_total commentee (GEPCK desactive) | Non bloquant — GEK pur uniquement pour l'instant | Non bloquant |

### Points techniques importants

**Espace de calcul :**
Tout est en espace U (N(0,1) standard). `result.getStandardSpaceDesignPoint()` retourne u*. Pas de T_inv necessaire dans FORM_multistart (FORM_event utilise directement N(0,1)^n_var).

**Gradient GEK pour OT — RESOLU (28/04) :**
`GEKPLSFunction._gradient` retourne `[[sm.predict_derivatives(u, kx).item()] for kx in range(n_var)]` = shape `(n_var, 1)` = format correct matrice Jacobienne OT pour f: R^n → R. Ce format est reconnu nativement — pas de fallback FD silencieux. L'ancienne hypothese sur `(1,n_var)` etait erronee.

**OT PRNG fixe (important) :**
OpenTURNS initialise son PRNG avec la meme graine a chaque demarrage de processus Python. `build_DOE` (LHS + SimulatedAnnealing) produit donc toujours le meme DOE. Pour un vrai DOE aleatoire : `ot.RandomGenerator.SetSeed(int(time.time()))` avant `build_DOE`.

**Encodage cp1252 :**
Le launcher lit `AC_pure_flexion.py` avec encodage Windows cp1252. Tout caractere Unicode non-ASCII → UnicodeDecodeError. Utiliser uniquement ASCII dans les commentaires.

**FORM_multistart vs FORM_all_modes :**
Les deux sont appeles sequentiellement sur les memes `starting_points`. `FORM_multistart` retourne le meilleur beta (pour affichage principal). `FORM_all_modes` retourne tous les modes distincts (pour analyse multi-modes). `tol_all_modes=0.5` est la distance U minimale pour considerer deux u* comme modes differents.

**run_HF :**
Retourne toujours 3 valeurs : `(g_HF: float, grad_HF_U: ot.Point, grad_HF_X: list)`. Toujours depackager a 3 valeurs.

**Role exact de tol_FORM et setCheckStatus — COMPRIS (13/05) :**
- `setMaximumConstraintError(tol_FORM)` : critere d'ARRET d'AbdoRackwitz. Quand `|g(u*)| < tol_FORM`, l'algo se declare SUCCESS.
- `setCheckStatus(False)` : empeche AbdoRackwitz de lever une exception si son statut est FAILED (ex : maxIterations atteint sans convergence).
- **Mais FORM fait son propre check interne :** si le statut du solver est FAILED, FORM leve sa propre RuntimeError, independamment de setCheckStatus. C'est Bug A : "RuntimeError OT meme avec setCheckStatus(False)".
- **Pourquoi tol_FORM=1.0 :** avec tol_FORM=1.0, AbdoRackwitz declare SUCCESS des que |g| < 1.0 → FORM ne leve pas de RuntimeError → `getResult()` accessible. Avec tol_FORM petit (ex 0.001, Test G), si g(u*)=0.008 > 0.001 → FAILED → RuntimeError → resultat perdu.
- **Combinaison dans le code :** tol_FORM=1.0 (evite RuntimeError) + setCheckStatus(False) (securite si maxIter) + try/except dans FORM_all_modes (attrape tout residu).
- **Effet de bord :** tol_FORM=1.0 peut provoquer un arret premature quand le surrogate n'a pas de vrai zero (GEK avec peu de points) — FORM s'arrete a g=0.40 < 1.0. Cf. note ligne test sentinelle : "beta=5.10 au lieu de 10 theorique car tol_FORM=1.0 trop lache".

**Rebound surface KRG — mecanisme compris (15/05) :**
`g_hat(x) = beta0 + r(x)^T R^-1 (Y - beta0)`. Loin du DOE, r(x)->0 (kernel SE) -> prediction -> beta0>0. Cree une ile g<0 fermee au lieu d'une region ouverte vers le bas-gauche. beta0 = moyenne GLS des evaluations DOE (> 0 car majorite des LHS en zone sure). Remede possible : `ot.LinearBasisFactory(n_var).build()` (reversion vers plan lineaire) ou ajout de points dans la zone vide.

**Gradient HFFunction verifie — CONFIRME (12/05) :**
`HFFunction._gradient` est bien appele par AbdoRackwitz a chaque iteration. Verifie par test sentinelle :
- `_exec` remplace par `g(u) = 1.0 + 0.1*u1 + 0.1*u2` (lineaire, pas de STRAINS)
- `_gradient` remplace par `[[1.0],[0.0]]` (sentinel — vrai gradient = `[[0.1],[0.1]]`)
- Prediction : si sentinel utilise → u* sur axe u1 (u2=0) ; si OT fait ses FD → u*=(-5,-5)
- Resultat : u*=(-5.103, 0.0), u2=0 exactement → sentinel utilise → gradient analytique actif.
- Note : beta=5.10 au lieu de 10 theorique car tol_FORM=1.0 trop lache (arret quand |g(u*)|<1.0).
- **Conclusion : les gradients STRAINS fournis par `_gradient` sont bien exploites par FORM. Les appels de sensibilite ne sont pas ignores.**

---

## 8. References fichiers (paths exacts post-reorganisation)

### Scripts
- `C:\_workingDir\_SF\test flexion\AC_pure_flexion.py`
- `C:\_workingDir\_SF\test flexion\launcher.py`

### Resumes de session (dans `resume session a relire\`)
- `C:\_workingDir\_SF\test flexion\resume session a relire\session_resume_1604.md`
- `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_1704_aprem.md`
- `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_2004.md`
- `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_2104.md`
- `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_2204.md`
- `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_2304.md`
- `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_2404.md`
- `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_3004.md`
- `C:\_workingDir\_SF\test flexion\resume session a relire\resume_session_0405.md` — **le plus recent** (04/05)

### Resultats (dans `(ancien) résultats et comparaison\`)
- `C:\_workingDir\_SF\test flexion\(ancien) résultats et comparaison\resultats_HF.md` — FORM HF phi=32mm (obsolete)
- `C:\_workingDir\_SF\test flexion\(ancien) résultats et comparaison\resultats_HF_run2.md` — **REFERENCE** : FORM HF phi=16mm, toutes charges
- `C:\_workingDir\_SF\test flexion\(ancien) résultats et comparaison\resultats_KRG_run2.md` — FORM KRG, n0 variable, 3 F
- `C:\_workingDir\_SF\test flexion\(ancien) résultats et comparaison\resultats_KRG_run3.md` — KRG pur run3, DOE fixe n0=15

### Comparaisons (dans `(ancien) résultats et comparaison\`)
- `C:\_workingDir\_SF\test flexion\(ancien) résultats et comparaison\comparaison_HF_KRG.md` — HF vs KRG, impact n0
- `C:\_workingDir\_SF\test flexion\(ancien) résultats et comparaison\comparaison_GEK_KRG.md` — GEK vs KRG, DOE identique
- `C:\_workingDir\_SF\test flexion\(ancien) résultats et comparaison\comparaison_KRG_PCEKRG.md` — KRG pur vs PCE-KRG, DOE fixe n0=15
- `C:\_workingDir\_SF\test flexion\(ancien) résultats et comparaison\comparaison_Matern_exp.md` — Matern 5/2 vs SquaredExponential
- `C:\_workingDir\_SF\test flexion\(ancien) résultats et comparaison\comparaison_a_DOE_fixe.md` — toutes methodes sur DOE fixe n0=15

---

## 9. Taches en cours

### FORM_multistart et FORM_all_modes — IMPLEMENTE (28/04)

Les deux fonctions sont implementees dans la version reecrite du code :
- `FORM_multistart(starting_points)` : lance AbdoRackwitz depuis chaque point, retourne `(best_result, best_sp)` = beta minimal.
- `FORM_all_modes(starting_points, tol_all_modes)` : tous modes distincts, tries par beta croissant.
- `starting_points = np.vstack([xt, [[0.0]*n_var]])` : n0+1 points de depart (DOE + origine).
- Les bugs P1/P3/P4/P5 identifies lors du refactoring ne s'appliquent plus — nouvelle implementation sans boucle while.

### print_results — IMPLEMENTE (30/04)

`print_results(best_result, g_ot_GEK=None, g_ot_KRG=None, g_ot_HF=None)` : affiche n_iter, fc*/fy* (X-space), u*, importances, beta, Pf. Blocs conditionnels independants pour gradient dg/du en u* et FOSM selon le modele fourni. Meme structure que print_visu. Valide sur run HF direct (output_3004_1203).

### Tache en cours — Calibration courbe analytique + verification HF (07/05)

**Objectif :** obtenir une courbe analytique (`flexion_claude`) qui a la bonne allure dans l'espace standard (U), puis verifier que STRAINS HF donne une courbe similaire via `print_visu`.

**Workflow :**
1. Lancer `print_visu_ana()` — trace la courbe analytique seule (branche plastifiee + branche verticale). Verifier l'allure : la courbe doit etre dans la zone plausible de l'espace U, avec la branche verticale du bon cote.
2. Lancer `print_visu(...)` avec `g_hf=run_HF` — superpose la grille HF (n_grid_hf x n_grid_hf appels STRAINS) et la courbe analytique. Verifier l'alignement des deux courbes.

**Pas de runs de production / FORM pour l'instant.** Phase de calibration/validation graphique uniquement.

Tache obsolete — `flexion_simple`/`calc_ana`/brentq supprimes. La comparaison analytique/HF se fait maintenant directement via `print_visu` (grille reguliere n_grid_hf x n_grid_hf).

---

### Config JCSS (note : imports supprimes, lois definies localement dans le script)
- `C:\_workingDir\_SF\fiabilite\config\jcss_fy.py` — reference theorique uniquement
- `C:\_workingDir\_SF\fiabilite\config\jcss_fc.py` — reference theorique uniquement

### Modele STRAINS
- `C:\workspace\storage\admin\SF\test_pure_flexion.ds\dsLoad.txt`
- `C:\workspace\storage\admin\SF\test_pure_flexion.ds\dsCad.txt`

---

## 10. Debug du GEK : comparaison ancien_nouveau_code

**STATUT : TERMINE (28/04/2026)**
Le GEK est desormais debogue et fonctionnel. La solution finale a ete une réécriture complète et reorganisation du code. `AC_pure_flexion.py` est la version de production active.

---

**Objectif initial :** Identifier pourquoi `ac_ancien_ref.py` (pre-refactoring) donne beta=3.774 stable, alors que `ac_nouveau_copie.py` (nouveau code refactorise) donne 1.487-5.043. Diagnostiquer et corriger le nouveau code.

### Fichiers de reference

| Fichier | Role | A modifier ? |
|---|---|---|
| `ac_ancien_ref.py` | Version ancienne de reference — ne jamais modifier | Non |
| `ac_nouveau_copie.py` | Copie de travail du nouveau code — ne jamais modifier | Non |
| `ac_nouveau_working1.py` | Premiere variante modifiee du nouveau code | Oui (debug) |
| `ac_nouveau_working2.py` | Deuxieme variante, etc. | Oui (debug) |
| `AC_pure_flexion.py` | **Fichier appele par launcher.py** — copie du fichier a tester | Remplace a chaque test |

### Principe de travail

1. Pour tester `ac_ancien_ref.py` : copier-coller son contenu dans `AC_pure_flexion.py`, lancer.
2. Pour tester `ac_nouveau_copie.py` : copier-coller son contenu dans `AC_pure_flexion.py`, lancer.
3. Pour tester une variante modifiee : creer `ac_nouveau_working1.py` (copie de `ac_nouveau_copie.py` + modification proposee), copier dans `AC_pure_flexion.py`, lancer.
4. Toujours noter dans le run quel fichier source a ete copie dans `AC_pure_flexion.py` et quelles options etaient actives.
5. Ne jamais modifier `ac_ancien_ref.py` ni `ac_nouveau_copie.py` — ce sont les references stables.
6. Incrementer le numero (`working1`, `working2`, ...) a chaque variante, meme si la modification est petite — pour garder la trace de chaque chemin explore.
7. **Quand l'utilisatrice annonce un nouveau test :** le decrire dans la section 10 (ordre des tests + nouvelle Partie dans le fichier .md de debug) AVANT de lancer. Ne jamais lancer un test qui n'est pas encore documente dans le global.

### Dossier de resultats de debug

Tous les fichiers `.md` de resultats et comparaison pour cette phase de debug sont dans :
`C:\_workingDir\_SF\test flexion\debug_resultats_comparaison\`

Ce dossier est distinct de `nouveau resultats et comparaison\` (resultats de production). Les fichiers de debug servent a la comparaison systematique entre variantes, pas a documenter des resultats definitifs.

**Format des tableaux de resultats de debug :** tous les runs d'un meme test (ex : Test A x3, Test B x3) sont regroupes dans **un seul tableau** avec une colonne par run. Ne jamais creer un tableau par run — cela rend la comparaison impossible et gaspille de la place. Exemple : `debug_ac_ancien_runs_TestA.md` contient un tableau avec colonnes Run 1 / Run 2 / Run 3.

### Hypothese mise a jour (session 27/04)

**Hypothese initiale invalidee :** le probleme n'est pas specifique a GEKPLS. SMT KRG (sans derivees) sur le meme DOE B.2 avec F=0.235 donne exactement le meme faux zero (beta≈0.13, u*≈[0.03,-0.12]). Le modele utilise (GEKPLS, SMT KRG, probablement OT KRG aussi) ne change rien.

**Diagnostic revise :** le probleme est dans la combinaison **DOE B.2 + F=0.235**.

- Pour F=0.235 : u*_HF=[-0.16,-0.94], norme=0.95. Ce point est DANS la zone couverte par le DOE B.2.
- Le metamodele cree un faux zero tres proche de l'origine (norme≈0.13) qui n'existe pas dans la vraie fonction g_HF. Ce faux zero est plus proche de l'origine que le vrai u* → FORM le trouve en premier, quel que soit le modele.
- Le faux zero est un artefact de l'interpolation : les 25 points DOE ont tous g_HF>0 (ils sont loin de la surface limite), le metamodele "invente" ou passe g=0 et se trompe.

**Ce qui etait correct dans l'hypothese initiale :**
- Le non-determinisme GEKPLS etait reel et cause par le parallelisme BLAS dans L-BFGS-B → confirme par Test H (BLAS single-thread = runs identiques).
- Ce non-determinisme explique la DISPERSION des betas entre runs (1.4 a 3.0) mais pas la MAUVAISE VALEUR moyenne.

**Hypothese sur les anciens bons resultats (F=0.210) :**
Pour F=0.210 : u*=[-0.526,-3.747], norme=3.78. Ce point est HORS de la zone couverte par le DOE. Le metamodele extrapole dans cette direction et le gradient pointe correctement vers u*. Il n'y a pas de faux zero a intercepter. C'est pourquoi OT KRG fonctionne bien pour F=0.210 (Test C) et echoue probablement pour F=0.235.

**Note sur le mode FD :** en FD explicite, FORM n'appelle jamais `predict_derivatives` — OT calcule le gradient par differences finies sur `predict_values` uniquement.

**Protocole de relance automatique :** quand plusieurs runs successifs sont demandes sur le meme fichier sans modification (ex : Test A x3, Test B x3), Claude relance le code immediatement apres chaque run sans attendre de message utilisateur. Claude note les resultats au fur et a mesure et communique un recapitulatif apres le dernier run.

**IMPORTANT — "lance le code" (ou equivalent : "lance", "relance", "go", etc.) = suivre a la lettre toutes les etapes de la section 6 :**
1. Lire les options du fichier actif (`AC_pure_flexion.py`) avant tout lancement — ne jamais interpreter un run sans avoir verifie les options. Verifier en particulier : `n0`, `U_doe_fixed` (fixe ou None), `do_GEK_analytic_grad` / `do_analytic_grad`, `do_warm_start`, `do_HF`, `do_KRG`, `do_GEK`, `try_pce`. Confirmer que les options correspondent bien a ce qui est attendu pour le test en cours (cf. section 10). Note : `do_GP` n'existe plus depuis le 27/04, remplace par `do_KRG` et `do_HF`.
2. Lancer en arriere-plan via `run_in_background=True`, nommer l'output `output_DDMM_HHMM.txt`.
3. Surveiller activement l'output et communiquer les resultats des qu'ils apparaissent.
4. **IMMEDIATEMENT apres la fin du run, sans attendre de message utilisateur :** creer ou mettre a jour le fichier `.md` de resultats correspondant — section ou tableau existant dans le bon fichier (production : `nouveau resultats et comparaison\` selon sections 6.3/6.4 ; debug : `debug_resultats_comparaison\` dans la partie correspondante). Ne jamais sauter cette etape, ne jamais la faire apres coup sur demande.
5. Mettre a jour `global_resume_session_2404.md` section 4 (tableau resultats) et section 9/10 si applicable.
6. Mettre a jour `resume_session_DDMM.md` du jour avec la configuration complete et les resultats.

**Ordre des tests :**

- **Test A** (TERMINE) : `ac_ancien_ref.py`, FD, n0=15, DOE fixe, 3 runs → non-deterministe confirme (betas 1.428 / 2.246 / 2.999). Ref : `debug_ac_ancien_runs.md` (Partie 1).

- **Test B.1** (TERMINE) : `ac_ancien_ref.py`, FD, n0=20, DOE aleatoire run 1 puis fixe runs 2&3.
  - Betas : 4.183 (run 1, DOE aleat.) / 2.108 (run 2, DOE fixe) / 2.654 (run 3, DOE fixe)
  - Conclusion : non-determinisme confirme a n0=20. n0=20 ne stabilise pas GEKPLS.
  - Ref resultats : `debug_resultats_comparaison\debug_ac_ancien_runs.md` (Partie 2 — Test B.1)

- **Test B.2** (TERMINE) : `ac_ancien_ref.py`, FD, n0=25, DOE aleatoire run 1 puis fixe runs 2&3.
  - Betas : 2.590 (run 1, DOE aleat.) / 2.424 (run 2, DOE fixe) / 1.268 (run 3, DOE fixe)
  - Conclusion : n0=25 ne stabilise pas GEKPLS. Run 3 pire de toute la serie (beta=1.268, grad u_fc 651%). Augmenter n0 n'aide pas.
  - Ref resultats : `debug_resultats_comparaison\debug_ac_ancien_runs.md` (Partie 3 — Test B.2)

- **Test B.3** (TERMINE) : `ac_ancien_ref.py`, FD, n0=25, do_warm_start=True, DOE fixe B.2 (tous les runs).
  - Betas : 0.420 / 3.397 / 0.560 — warm start declenche dans les 3 runs, n_iter=1 apres WS.
  - Conclusion : warm start aggrave le non-determinisme. Point WS variable → 2eme FORM converge vers faux minimum local (g_meta≈0 mais g_HF loin de 0). N'est pas une solution.
  - Ref resultats : `debug_resultats_comparaison\debug_ac_ancien_runs.md` (Partie 4 — Test B.3)

- **Test B.4** (PARTIEL — run 1 seul) : `ac_ancien_ref.py`, FD, n0=40, do_warm_start=True.
  - Run 1 : beta=0.531, meme pathologie que B.3 (WS declenche, n_iter=1 apres WS, g_HF=0.150). Arrete apres run 1.
  - Ref resultats : `debug_resultats_comparaison\debug_ac_ancien_runs.md` (Partie 5 — Test B.4)

- **Test C** (TERMINE) : `ac_ancien_ref.py`, **KRG** (do_GEK=False), n0=25, do_warm_start=True, DOE fixe (OT seed deterministe).
  - Betas : 3.797 (run 1, WS declenche) / 3.665 (run 2) / 3.672 (run 3) — ecart max 0.132.
  - Conclusion : **KRG est deterministe et stable.** Le non-determinisme est specifique a GEKPLS. Meme DOE que Test B.2 : GEK donnait 1.268-2.590, KRG donne 3.665-3.797.
  - Ref resultats : `debug_resultats_comparaison\debug_ac_ancien_runs.md` (Partie 6 — Test C)

- **Test F** (TERMINE — 2 runs) : `ac_ancien_ref.py`, GEK FD, n0=25, do_warm_start=True, **tol_FORM=0.05**, F=0.235 MN, DOE fixe B.2.
  - Betas : 0.471 (run 1) / 0.136 (run 2). Arrete apres 2 runs.
  - Ref resultats : `debug_resultats_comparaison\debug_ac_ancien_runs.md` (Partie 8 — Test F)

- **Test G** (CRASH — RuntimeError) : `ac_ancien_ref.py`, GEK FD, n0=25, do_warm_start=True, **tol_FORM=0.001**, F=0.235 MN, DOE fixe B.2.
  - g_meta(u*)=0.00860 > tol_FORM=0.001 → OT leve RuntimeError avant retour du resultat. Arrete.
  - Ref resultats : `debug_resultats_comparaison\debug_ac_ancien_runs.md` (Partie 9 — Test G)

- **Test H** (TERMINE) : `ac_ancien_ref.py` + **fix BLAS single-thread** (`OMP/MKL/OPENBLAS_NUM_THREADS=1`), GEK FD, n0=25, do_warm_start=True, tol_FORM=0.05, F=0.235 MN, DOE fixe B.2.
  - Betas : **0.172 / 0.172 / 0.172** — 3 runs bit-for-bit identiques.
  - **BLAS single-thread = determinisme parfait confirme.** Non-determinisme precedent etait exclusivement du au parallelisme BLAS dans L-BFGS-B.
  - Probleme residuel : theta trouve donne gradient 62%/52% d'erreur → beta toujours faux (0.172 vs HF≈0.95).
  - Ref resultats : `debug_resultats_comparaison\debug_ac_ancien_runs.md` (Partie 10 — Test H)

- **Test I** (EN COURS — 3 runs) : `ac_ancien_ref.py` + fix BLAS + **n_comp=1** (`reduc_PLS=1`), GEK FD, n0=25, do_warm_start=True, tol_FORM=0.05, F=0.235 MN, DOE fixe B.2.
  - Meme config que Test H sauf reduc_PLS=0→1 (n_comp passe de 2 a 1).
  - Objectif : n_comp=1 = theta 1D → paysage log-vraisemblance plus simple → meilleur theta → meilleur gradient → FORM converge vers u*.
  - Ref resultats : `debug_resultats_comparaison\debug_ac_ancien_runs.md` (Partie 11 — Test I)

- **Test D** (A FAIRE) : `ac_nouveau_copie.py`, FD (`do_analytic_grad=False`), n0=25, 3 runs → verifier si le nouveau code en FD se comporte comme l'ancien.
  - Objectif : confirmer que la difference ancien/nouveau n'est pas structurelle mais due au non-determinisme GEKPLS.

- **Test E** (TERMINE) : `ac_ancien_ref.py`, GEK FD, n0=25, do_warm_start=True, **F=0.235 MN** (beta_HF≈0.95), DOE fixe B.2 (tous les runs).
  - Betas : 0.267 / 0.223 / 0.177 (ref HF≈0.95) — gradient u_fc signe inverse dans les 3 runs.
  - Conclusion : non-determinisme GEKPLS independant de F. Probleme dans le training, pas dans le probleme.
  - F modifiee dans `dsLoad.txt` : Z='-0.235'. **Remettre Z='-0.210' avant le prochain test sur F=0.210.**
  - Ref resultats : `debug_resultats_comparaison\debug_ac_ancien_runs.md` (Partie 7 — Test E)

### Resultats Test A — ac_ancien_ref.py FD x3 (session 24/04 apres-midi)

**Ref :** `debug_resultats_comparaison\debug_ac_ancien_runs.md` (Partie 1)

| Run | Output | beta | u* | Grad err u_fc |
|---|---|---|---|---|
| 1 | output_2404_1524.txt | 1.428 | [+0.286, -1.399] | 41% |
| 2 | output_2404_1536.txt | 2.246 | [+0.324, -2.222] | 144% |
| 3 | output_2404_1541.txt | 2.999 | [-0.553, -2.947] | 860% |

**Conclusion Test A : NON-DETERMINISTE.** L'ancien code en FD explicite donne 1.428 a 2.999 (vs HF=3.784). Le beta=3.774 historique etait un run chanceux. Les erreurs de gradient GEKPLS sont enormes (41-860%) — le training L-BFGS-B produit des theta donnant de mauvaises derivees. La valeur predict_values reste correcte (g_GEK ≈ g_HF) mais predict_derivatives est inutilisable pour ce DOE n0=15.
