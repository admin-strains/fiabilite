# Résumé de session — FORM KRG Flexion Pure BA
**Date :** 17-20 avril 2026
**Objectif :** Mettre en place et valider le FORM KRG (métamodèle Krigeage) sur une poutre BA en flexion pure (phi=16mm), comparer à la référence HF (appels directs STRAINS), étudier l'impact de n0 sur la précision, et explorer l'enrichissement adaptatif du DOE.

---

## 1. Architecture du problème

- **Fonction de performance :** g = α⁺ − 1 (α⁺ = multiplicateur de charge limite STRAINS)
- **Variables aléatoires :** fc (béton, lognormale JCSS), fy (acier phi=16mm, lognormale JCSS)
- **Espace standard U :** transformation isoprobabiliste T : (fc, fy) → (u_fc, u_fy)
- **DOE :** LHS + recuit simulé (SimulatedAnnealingLHS + SpaceFillingMinDist), n0 points dans espace U
- **Métamodèle KRG :** ot.KrigingAlgorithm, covariance Matérn, base constante — gradient analytique intégré (pas de PythonFunction wrapper ni HFCache)
- **FORM :** AbdoRackwitz sur le métamodèle KRG, β = ‖u*‖, Pf = Φ(−β)
- **Référence HF :** FORM AbdoRackwitz avec appels directs STRAINS + gradient analytique STRAINS (HFCache pattern)

---

## 2. Problèmes rencontrés et solutions

### Problème 1 — setCheckStatus(False) ne fonctionne pas quand g >> 1e-5

**Symptôme :**
Pour certains runs KRG (n0 insuffisant ou β grand), FORM lève :
```
RuntimeError: Obtained design point is not on the limit state:
its image by the limit state function is X.XXX, incompatible with threshold 0, tolerance 1e-05
```
Malgré `solver.setCheckStatus(False)` déjà en place.

**Analyse :**
- `setCheckStatus(False)` supprime l'exception du solver ET la vérification finale de FORM **seulement si le solver s'est arrêté sur un critère de convergence** (MaximumAbsoluteError, MaximumRelativeError, MaximumConstraintError).
- Quand le solver atteint `MaximumIterationNumber` avec g encore grand (0.008, 0.016, 0.057…), FORM lève l'exception **indépendamment** de checkStatus.
- Distinction empirique : fonctionne pour g≈1.3e-5 (HF, solver convergé) ; ne fonctionne pas pour g=0.008+ (solver épuisé).

**Solution retenue :**
```python
solver.setMaximumConstraintError(1e-2)
```
Cela fait deux choses simultanément :
1. Le solver s'arrête dès g_KRG < 0.01 (critère de convergence → pas MaximumIterationNumber)
2. FORM accepte le résultat car g < seuil = 0.01

**Limite :** Résultat non physique si g=0.0083 avec gradient nul (métamodèle trop pauvre). Indiqué en note dans les tableaux.

---

### Problème 2 — Erreur de format `unsupported format string passed to Point.__format__`

**Symptôme :**
```
TypeError: unsupported format string passed to Point.__format__
```
à la ligne :
```python
print(f"  g* GP   = {g_GP:.6f}")
```

**Cause :** `metamodel_KRG(U_res)[0, 0]` retourne encore un Point OT, pas un float. Le double indexage `[0, 0]` sur un Point OT est invalide.

**Solution retenue :**
```python
g_GP = metamodel_KRG(U_res)[0]  # [0] suffit — retourne un float
```

---

### Problème 3 — KRG FORM manquait setCheckStatus et setMaximumIterationNumber

**Symptôme :** Le bloc KRG FORM original n'avait pas les mêmes paramètres que le bloc HF FORM.

**Solution retenue :** Ajout dans le bloc KRG (lignes ~529-532) :
```python
solver.setMaximumIterationNumber(n_max_FORM)
solver.setCheckStatus(False)
solver.setMaximumConstraintError(1e-2)  # ajouté plus tard pour le problème 1
```

---

### Problème 4 — KRG inexact pour β grand (design point dans la queue de distribution)

**Symptôme :** Pour F=0.210 (β_HF≈3.78) et F=0.195 (β_HF≈5.48), un DOE LHS standard (même avec n0=25-60) ne couvre pas la queue de distribution où se trouve u*. Le KRG est aveugle à cet endroit.

**Analyse :**
- u*_HF = [-0.526, -3.747] pour β=3.78 — probabilité d'un point LHS à cette distance : très faible
- u*_HF = [-0.829, -5.422] pour β=5.48 — probabilité quasi nulle (Φ(-5.4)≈3e-8)
- g_KRG(u*) restait grand même avec n0=60 (g=-0.013 → erreur β = 8.6%)

**Solutions testées :**
- Augmenter n0 progressivement (25→40→60) : amélioration lente, erreur 19%→12%→8.6%
- n0=50, 100 pour F=0.195 : toujours RuntimeError (g=0.029, 0.057) — pas traitable avec LHS standard

**Solution retenue — Enrichissement adaptatif :**
1. Lancer FORM KRG avec n0=15 (avec setMaximumConstraintError=1e-2 pour récupérer u*_n15)
2. Ajouter u*_n15 = [-1.65403, -5.4416] au DOE : `U_doe.add(ot.Point([-1.65403, -5.4416]))`
3. n0 devient 16 automatiquement : `n0 = U_doe.getSize()`
4. Warm start FORM depuis u*_n15
5. **Résultat : β=3.761 (erreur 0.6% vs 50.3% sans enrichissement), n_iter=2**

---

## 3. Modifications du code AC_pure_flexion.py

### 3a. Flag do_GP (ligne ~320)
```python
do_GP = True   # était False
```

### 3b. Bloc FORM KRG — paramètres solver (lignes ~529-534)
**Avant :**
```python
solver = ot.AbdoRackwitz()
solver.setStartingPoint([0.0] * n_var)
algo = ot.FORM(solver, event)
algo.run()
result = algo.getResult()
```
**Après :**
```python
solver = ot.AbdoRackwitz()
solver.setMaximumIterationNumber(n_max_FORM)
solver.setCheckStatus(False)
solver.setMaximumConstraintError(1e-2)
solver.setStartingPoint([-1.65403, -5.4416])  # warm start depuis u*_n15 (modifiable)
algo = ot.FORM(solver, event)
algo.run()
result = algo.getResult()
```

### 3c. n_max_FORM (ligne ~325)
```python
n_max_FORM = 50   # était 40
```

### 3d. n0 (ligne ~323)
```python
n0 = max(15, n_start)  # modifié progressivement selon les runs
```

### 3e. Enrichissement DOE (après ligne 328)
```python
U_doe.add(ot.Point([-1.65403, -5.4416]))  # enrichissement au point u*_n15
n0 = U_doe.getSize()  # mise à jour automatique (16)
```

### 3f. Correction indexage KRG (ligne ~618)
```python
g_GP = metamodel_KRG(U_res)[0]   # était [0, 0] → TypeError
```

### 3g. Test GP au point u*_n60 (après ligne 623)
```python
u_n60 = ot.Point([-0.8350, -4.0249])
g_GP_at_n60 = metamodel_KRG(u_n60)[0]
print(f"\nTest KRG au point u*_n60 = [-0.8350, -4.0249] :")
print(f"  g_KRG(u*_n60) = {g_GP_at_n60:.6f}")
```
→ Résultat : g_KRG_n15(u*_n60) = 0.019 (n0=15 ne "voit" pas la surface limite là-bas)

---

## 4. Résultats FORM KRG obtenus

### F = 0.235 MN (β_HF = 0.952)
| n0 | β | Erreur β | g_HF(u*) | n_iter |
|---|---|---|---|---|
| 25 | 0.9532 | 0.1% | -6.1e-05 | 11 |
| 10 | 0.9451 | 0.72% | +2.44e-04 | 11 |
| 5 | 0.8620 | 9.5% | +3.42e-03 | 12 |

### F = 0.225 MN (β_HF = 2.084)
| n0 | β | Erreur β | g_HF(u*) | n_iter |
|---|---|---|---|---|
| 25 | 2.0987 | 0.72% | -5.58e-04 | 14 |
| 16 | 2.0682 | 0.77% | +6.43e-04 | 14 |
| 8 | 2.2704 | 8.9% | -5.93e-03 | 16 |

### F = 0.210 MN (β_HF = 3.784)
| n0 | β | Erreur β | g_HF(u*) | n_iter |
|---|---|---|---|---|
| 60 | 4.1106 | 8.6% | -1.35e-02 | 19 |
| 40 | 4.2621 | 12.6% | -2.00e-02 | 19 |
| 25 | 4.5134 | 19.3% | -3.04e-02 | 21 |
| 20 | 4.9171 | 29.9% | -4.76e-02 | 23 |
| 15* | 5.6874 | 50.3% | -7.76e-02 | 51 |
| **15+1 enrichi** | **3.7614** | **0.6%** | **+1.24e-03** | **2** |

*Tolérance relâchée (MaximumConstraintError=1e-2), gradient nul — non physique.

---

## 5. Sujets abordés et apprentissages

### 5a. Comportement de setCheckStatus(False)
- Supprime l'exception du solver ET de FORM **uniquement** quand le solver s'arrête par un critère de convergence (pas par MaximumIterationNumber).
- Ne fonctionne pas quand g est trop grand (solver épuisé sans converger).
- Solution pour récupérer quand même un résultat : `solver.setMaximumConstraintError(1e-2)`.

### 5b. Limitation du LHS standard pour grand β
- Un DOE LHS dans l'espace U ne couvre pas la queue de distribution là où se trouve u* pour β>3.
- Erreur croissante : β=0.95 (0.1% avec n0=25) → β=2.08 (0.72% avec n0=25) → β=3.78 (19.3% avec n0=25).
- Pour β=5.5, même n0=50 ne converge pas.

### 5c. Enrichissement adaptatif
- **Principe :** Lancer FORM KRG avec n0 réduit pour trouver un premier u* (même imparfait), l'ajouter au DOE, relancer.
- **Résultat expérimental :** n0=15 → u*=[−1.654, −5.442] (β=5.69, erreur 50%). Enrichissement avec ce point → β=3.761 (erreur 0.6%) en seulement 2 itérations FORM.
- **Clé :** Le point ajouté informe le KRG sur la région critique de l'espace.
- **Observation :** g_KRG_n15(u*_n60) = 0.019 → le métamodèle n0=15 place la surface limite complètement ailleurs → warm start seul sans enrichissement ne suffit pas.

### 5d. Warm start FORM
- Démontré que démarrer depuis u*_n15 (au lieu de [0,0]) réduit n_iter de 51→30 pour le même résultat (même minimum local).
- Avec enrichissement + warm start : n_iter=2 (le métamodèle enrichi a sa surface limite près de u*_n15).

### 5e. Interprétation de g_HF(u*_KRG)
- g_HF(u*_KRG) > 0 → point dans domaine sûr (α⁺ > 1) : KRG a sous-estimé β
- g_HF(u*_KRG) < 0 → point dans domaine défaillant (α⁺ < 1) : KRG a sur-estimé β (cas général pour β grand)
- g_HF proche de 0 → bon accord KRG/HF
- Erreur relative g_HF vs g_KRG souvent > 90% car les deux valeurs sont proches de 0 (rapport instable)

### 5f. n_iter KRG vs HF
- HF : convergence en 1 iter pour β<2.5 (départ [0,0] très proche de u*)
- KRG : 11-16 iter pour β<2.5 (surface limite légèrement décalée → plus d'itérations)
- HF : 18-21 iter pour β≈3.8-5.5
- KRG (bon) : 19-21 iter pour β≈3.8 avec n0 suffisant
- KRG (enrichi) : 2 iter → warm start depuis u* déjà proche

---

## 6. Fichiers clés

| Fichier | Rôle | État |
|---|---|---|
| `AC_pure_flexion.py` | Script principal FORM HF + KRG | do_GP=True, n0=15+1 enrichi, n_max_FORM=50, MaxConstraintError=1e-2, setCheckStatus(False), starting point=[-1.65403,-5.4416] |
| `launcher.py` | Lance AC_pure_flexion.py avec DLL STRAINS | Inchangé |
| `dsLoad.txt` | Force appliquée (Z='-0.210') | F=0.210 MN |
| `resultats_HF_run2.md` | Résultats FORM HF référence (phi=16mm, phi=32mm) | Complet |
| `resultats_KRG_run2.md` | Résultats FORM KRG par F et n0 | 3 sections : F=0.235, F=0.225, F=0.210 |
| `comparaison_HF_KRG.md` | Tableaux synthèse HF vs KRG (β, Pf, g_HF, erreur, n_iter) | 3 tableaux |
| `resume_session_1704_aprem.md` | Résumé session précédente (problèmes HF, setCheckStatus) | Complet |

---

## 7. État du code au moment de l'arrêt de session

```python
# Flags
do_GP = True
do_GEK = False
do_pce = False
n_start = 1
n0 = max(15, n_start)         # → 15 points LHS
n_max_FORM = 50

# Enrichissement DOE (après sa.generate())
U_doe.add(ot.Point([-1.65403, -5.4416]))  # u*_n15
n0 = U_doe.getSize()          # → 16

# Bloc FORM KRG
solver = ot.AbdoRackwitz()
solver.setMaximumIterationNumber(n_max_FORM)
solver.setCheckStatus(False)
solver.setMaximumConstraintError(1e-2)
solver.setStartingPoint([-1.65403, -5.4416])  # warm start u*_n15

# Test GP
g_GP = metamodel_KRG(U_res)[0]   # [0] pas [0,0]
u_n60 = ot.Point([-0.8350, -4.0249])
g_GP_at_n60 = metamodel_KRG(u_n60)[0]

# dsLoad.txt : Z='-0.210' (F=0.210 MN)
```

---

## 8. Prochaines étapes suggérées

- Tester l'enrichissement depuis [0,0] (sans warm start) pour mesurer l'apport du warm start seul
- Appliquer l'enrichissement adaptatif itératif (plusieurs cycles u*→enrichissement→FORM) pour β=5.5
- Étendre l'étude à d'autres valeurs de F / beta
- Remettre `setMaximumConstraintError` à sa valeur par défaut (1e-5) pour les runs "normaux" et ne l'utiliser qu'en fallback
- Remettre le starting point à [0.0]*n_var pour les runs standard

---

## 9. Principe de remplissage des fichiers .md

### 9a. `comparaison_HF_KRG.md` — Tableaux de synthèse

**Structure :** Un tableau par valeur de F (= par niveau de β_HF). Les colonnes sont les méthodes testées (HF en premier, puis KRG n0=X par n0 croissant ou décroissant selon l'ordre de test). Les lignes sont les métriques clés.

**Colonnes :** HF | KRG n0=X1 | KRG n0=X2 | ...

**Lignes (ordre fixe) :**
1. `u*` — coordonnées [u_fc, u_fy] du design point
2. `β` — indice de fiabilité FORM
3. `Pf` — probabilité de défaillance (si disponible)
4. `g_HF(u*)` — valeur HF au design point KRG (0 par définition pour HF ; mesure l'écart pour KRG)
5. `g_KRG(u*)` — valeur KRG au design point (≈ 0 par construction FORM)
6. `Erreur relative β` — |β_KRG − β_HF| / β_HF
7. `n_iter FORM` — nombre d'itérations du solver AbdoRackwitz

**Règle :** La case g_HF(u*_HF) est "≈ 0" (par définition). La case g_KRG(u*_HF) n'est pas renseignée (KRG n'est pas évalué au point HF). Les notes ¹ ² en bas du tableau signalent les cas non-physiques ou les enrichissements.

**Quand remplir :** Après chaque run, ajouter une colonne (n0 correspondant). Ne pas écraser les colonnes existantes.

---

### 9b. `resultats_KRG_run2.md` — Résultats détaillés par run

**Structure :** Une section par valeur de F (titre `### F = X MN`). Dans chaque section, un tableau à colonnes multiples (une colonne par n0 testé).

**Colonnes :** Paramètre | n0=X1 | n0=X2 | ...

**Lignes (ordre fixe) :**
- **Bloc "Résultats FORM KRG"** : n points DOE, fc* (MPa), fy* (MPa), dg/du_fc en u*, dg/du_fy en u*, Importance fc (%), Importance fy (%), β (FORM), Pf (FORM), n_appels HF (FORM) [toujours 0 car métamodèle], n_iter FORM
- **Bloc "Test GP au point de FORM"** : g_HF(u*), g_KRG(u*) [≈ 0 par construction]
- **Bloc "Comparaison HF (Run 2)"** : β (FORM HF), Écart β (KRG vs HF)
- **Bloc "Test linéarisation FOSM"** : u* FORM, u* FOSM (depuis u=0), Erreur relative ‖u*_FOSM − u*_FORM‖/‖u*_FORM‖

**Source des données :** Output terminal du script, extrait par lecture attentive de la sortie. Les grandeurs physiques (fc*, fy*) sont les coordonnées du design point retransformées depuis l'espace U (via T_inv). Les gradients sont récupérés depuis `result.getHasoferReliabilityIndexSensitivity()` ou équivalent. g_HF est calculé par le bloc "Test GP" du code.

**Quand remplir :** Après chaque run réussi, ajouter la colonne correspondante. En cas de run non-physique, noter avec une note de bas de tableau (¹).

---

## 10. Liste exhaustive des runs effectués (session 17-20 avril 2026)

### F = 0.235 MN (β_HF = 0.952)

| n0 | β_KRG | Erreur β | g_HF(u*) | n_iter | Remarques |
|---|---|---|---|---|---|
| 25 | 0.9532 | 0.1% | -6.1e-05 | 11 | Premier run KRG, référence qualité |
| 10 | 0.9451 | 0.72% | +2.44e-04 | 11 | Bon résultat |
| 5 | 0.8620 | 9.5% | +3.42e-03 | 12 | Dégradation notable |

### F = 0.225 MN (β_HF = 2.084)

| n0 | β_KRG | Erreur β | g_HF(u*) | n_iter | Remarques |
|---|---|---|---|---|---|
| 25 | 2.0987 | 0.72% | -5.58e-04 | 14 | Bon résultat |
| 16 | 2.0682 | 0.77% | +6.43e-04 | 14 | Bon résultat |
| 8 | 2.2704 | 8.9% | -5.93e-03 | 16 | Dégradation |
| 5 | — | — | — | — | Non testé (décision d'arrêt) |

### F = 0.210 MN (β_HF = 3.784)

| n0 | β_KRG | Erreur β | g_HF(u*) | n_iter | Remarques |
|---|---|---|---|---|---|
| 25 | 4.5134 | 19.3% | -3.04e-02 | 21 | Premier run ce β |
| 40 | 4.2621 | 12.6% | -2.00e-02 | 19 | Amélioration lente |
| 60 | 4.1106 | 8.6% | -1.35e-02 | 19 | Encore 8.6% d'erreur |
| 20 | 4.9171 | 29.9% | -4.76e-02 | 23 | Dégradation attendue |
| 15 | 5.6874 | 50.3% | -7.76e-02 | 51 | Non physique (MaxConstraintError=1e-2, gradient nul) |
| 15+1 | **3.7614** | **0.6%** | **+1.24e-03** | **2** | DOE enrichi avec u*_n15, warm start |

### F = 0.195 MN (β_HF ≈ 5.48) — tentative abandonnée

| n0 | Résultat | Remarques |
|---|---|---|
| 50 | RuntimeError (g=0.029) | LHS standard insuffisant, β trop grand |
| 100 | RuntimeError (g=0.057) | Même problème — non résolu sans enrichissement |

---

## 11. Méthodologie pas-à-pas suivie durant la session

### Phase 0 — Mise en place initiale
1. Lire le code existant (`AC_pure_flexion.py`) pour comprendre l'architecture
2. Identifier le bloc GP (do_GP=False → mettre True)
3. Vérifier la cohérence du bloc KRG avec le bloc HF (solver params)
4. Ajouter `solver.setMaximumIterationNumber(n_max_FORM)` et `solver.setCheckStatus(False)` dans le bloc KRG

### Phase 1 — Test KRG pour β faible (F=0.235, β≈0.95)
1. Modifier `dsLoad.txt` : `Z='-0.235'`
2. Fixer `n0 = 25`, `do_GP = True`
3. Lancer via `launcher.py`
4. Lire le terminal : extraire β, Pf, u*, gradients, n_iter, g_HF(u*), g_KRG(u*)
5. Remplir colonne n0=25 dans `resultats_KRG_run2.md` (section F=0.235)
6. Remplir colonne KRG n0=25 dans `comparaison_HF_KRG.md` (Tableau 1)
7. Répéter avec n0=10, puis n0=5 (diminution progressive)
8. Arrêter quand l'erreur β dépasse ~10% ou la convergence échoue

### Phase 2 — Test KRG pour β moyen (F=0.225, β≈2.08)
1. Modifier `dsLoad.txt` : `Z='-0.225'`
2. Tester n0=25, puis 16, puis 8 (choix basé sur expérience phase 1 : éviter n0 trop petit)
3. Même procédure : lancer → extraire → remplir .md
4. Arrêter à n0=8 (n0=5 jugé trop petit a priori)

### Phase 3 — Test KRG pour β grand (F=0.210, β≈3.78)
1. Modifier `dsLoad.txt` : `Z='-0.210'`
2. Augmenter `n_max_FORM = 50` (HF converge en 21 iter → KRG besoin de plus)
3. Commencer à n0=25, puis augmenter (40, 60) pour voir si la convergence s'améliore
4. Constater que même n0=60 donne 8.6% d'erreur
5. Réduire n0 progressivement (25→20→15) pour caractériser la dégradation
6. Pour n0=15 : RuntimeError → résolu avec `setMaximumConstraintError(1e-2)`
7. Constater que le résultat n0=15 est non physique (gradient nul)

### Phase 4 — Enrichissement adaptatif
1. Récupérer u*_n15 = [-1.65403, -5.4416] depuis le run n0=15
2. Tester `metamodel_KRG(u_n60)` (évaluation du KRG n0=15 au point u*_n60) → g=0.019 → le KRG n0=15 ne "voit" pas la surface limite près de u*_n60
3. Décider d'ajouter u*_n15 au DOE : `U_doe.add(ot.Point([-1.65403, -5.4416]))`
4. Mettre `n0 = U_doe.getSize()` → 16 (mise à jour automatique)
5. Warm start FORM depuis u*_n15 : `solver.setStartingPoint([-1.65403, -5.4416])`
6. Relancer → β=3.7614 (erreur 0.6%), n_iter=2
7. Ajouter colonne n0=15+1 dans les deux .md

### Phase 5 — Documentation finale
1. Remplir tableaux de synthèse dans `comparaison_HF_KRG.md` (3 tableaux complets)
2. Remplir résultats détaillés dans `resultats_KRG_run2.md` (3 sections)
3. Générer `resume_session_1804.md` (ce fichier)

### Règles générales de la méthodologie
- **Jamais écraser** les résultats précédents dans les .md : toujours ajouter une colonne
- **Toujours noter** le n0 exact et les paramètres solver (MaxConstraintError, starting point, etc.)
- **Toujours vérifier** g_HF(u*_KRG) après chaque run pour valider la qualité
- **Signal d'arrêt** : erreur β > 20% OU RuntimeError non récupérable OU gradient nul
- **Ordre de remplissage** : d'abord résultats bruts dans `resultats_KRG_run2.md`, puis synthèse dans `comparaison_HF_KRG.md`

---

---

# Session du 20 avril 2026 (après-midi) — Warm start automatique + GEK/GEKPLS

**Fichier principal :** `C:\_workingDir\_SF\test flexion\AC_pure_flexion.py`
**Objectifs :**
1. Implémenter le warm start automatique (enrichissement DOE + re-training KRG + second FORM)
2. Analyser le bloc GEK existant, identifier les bugs, définir un plan d'activation en 4 étapes
3. Valider le warm start sur F=0.210

---

## A. Implémentation du warm start automatique

### A1. Principe du warm start automatique

Contrairement à l'enrichissement manuel de la session précédente (ajout d'un point fixe au DOE avant le lancement), le warm start automatique s'exécute **à l'intérieur du script**, après le premier FORM :

1. **Premier FORM** : AbdoRackwitz sur KRG, produit u*₁ et result₁
2. **Test de qualité** : si `float(metamodel_KRG(U_warm)[0]) > tol_warm_start` → KRG imprécis en u*₁
3. **Enrichissement** : ajout de u*₁ au DOE (`U_doe.add(U_warm)`), évaluation HF en u*₁, `y_hf = np.vstack([y_hf, [[y_to_add]]])`
4. **Re-training KRG** : `ot.KrigingAlgorithm(xt, yt, ...)` avec le DOE enrichi
5. **Second FORM** : AbdoRackwitz depuis u*₁ (`solver.setStartingPoint(U_warm)`) sur le KRG enrichi
6. **Résultat final** : `result` écrasé par le résultat du second FORM

**Condition de déclenchement :** `do_warm_start=True` ET `g_KRG(u*) > tol_warm_start` (ici 0.001)

### A2. Code warm start (lignes 547–575)

```python
U_warm = result.getPhysicalSpaceDesignPoint()
if do_warm_start and float(metamodel_KRG(U_warm)[0]) > tol_warm_start:
    # 1. Mise à jour DOE
    U_doe.add(U_warm)
    print(f"Warm start lancé avec point de départ U={list(U_warm)}")
    xt = np.array(U_doe)
    y_to_add, _ = run_HF(modelname, U_warm, params_names, T_inv, sensitivity=False)
    y_hf = np.vstack([y_hf, [[y_to_add]]])
    yt = y_hf
    if do_pce:
        yt -= y_pce
    # 2. Re-training KRG
    basis = ot.ConstantBasisFactory(n_var).build()
    covarianceModel = ot.MaternModel([1.0] * n_var, 2.5)
    algo_KRG = ot.KrigingAlgorithm(xt, yt, covarianceModel, basis)
    algo_KRG.run()
    result_KRG = algo_KRG.getResult()
    metamodel_KRG = result_KRG.getMetaModel()
    # 3. Second FORM
    output = ot.CompositeRandomVector(metamodel_KRG, vect)
    event = ot.ThresholdEvent(output, ot.Less(), 0.0)
    solver = ot.AbdoRackwitz()
    solver.setMaximumIterationNumber(n_max_FORM)
    solver.setCheckStatus(False)
    solver.setMaximumConstraintError(1e-2)
    solver.setStartingPoint(U_warm)   # départ depuis u*₁, pas [0,0]
    algo = ot.FORM(solver, event)
    algo.run()
    result = algo.getResult()
```

### A3. Problèmes rencontrés dans le warm start

#### Problème W1 — Indentation incorrecte du bloc `if do_pce:` (ligne 556–557)

**Symptôme :** le bloc `if do_pce: yt -= y_pce` était sur-indenté → rattaché au `if do_pce:` externe au lieu d'être dans le bloc warm start.

**Status :** identifié, non corrigé en fin de session (low priority, do_pce=False en permanence pour l'instant).

#### Problème W2 — `y_hf.append(y_to_add)` impossible sur numpy

**Question :** comment ajouter un scalaire à un array numpy `y_hf` de shape `(n0, 1)`.

**Analyse :**
- `np.append` produit un array 1D plat (perd la dimension 2D)
- `.append()` n'existe pas sur ndarray (c'est une méthode de list Python)

**Solution retenue :**
```python
y_hf = np.vstack([y_hf, [[y_to_add]]])   # conserve shape (n0+1, 1)
```

#### Problème W3 — Test GP utilisait l'ANCIEN metamodel_KRG après warm start

**Contexte :** Le bloc "Test GP" (lignes 660–667) évaluait `metamodel_KRG(U_res)`. Si warm start avait tourné, `metamodel_KRG` avait été remplacé par le KRG enrichi (ligne 564). La comparaison était donc faite avec le bon KRG, mais le message affiché pouvait prêter à confusion lors d'une lecture rapide des résultats.

**Vrai problème identifié :** lors d'un run précédent avec warm_start=True, la valeur affichée `g_GP=0.029` semblait anormale. Après investigation : le warm start **n'avait pas été déclenché** (g_KRG(u*) < tol=0.001 → condition non remplie), donc `metamodel_KRG` était l'original non enrichi, et `g_GP=0.029` correspondait à l'évaluation du KRG original au point u*. Ce n'était pas un bug de code mais une ambiguïté de lecture.

#### Problème W4 — NameError 'metamodel_KRG' quand do_GEK=True

**Symptôme :** si do_GEK=True, `metamodel_KRG` n'est jamais défini (le bloc `else:` de `if do_GEK:` ne s'exécute pas). Le bloc test GP qui appelle `metamodel_KRG(U_res)` levait `NameError`.

**Solution :** ajout de `and not do_GEK` à la garde (fix du user) :
```python
# AVANT
if do_GP_test and do_GP:

# APRÈS
if do_GP_test and do_GP and not do_GEK:
```

#### Problème W5 — n_iter=0 affiché pour le second FORM du warm start

**Observation :** quand le warm start se déclenche et que le second FORM part de u*₁ (déjà sur la surface limite), il converge en ~0 itérations. `result.getOptimizationResult().getIterationNumber()` retourne 0.

**Interprétation :** comportement physiquement cohérent (départ sur la surface limite → convergence immédiate). Ce n'est pas un bug mais peut prêter à confusion si on croit que le FORM n'a pas tourné.

---

## B. Analyse du bloc GEK — Bugs identifiés et plan d'activation

### B1. Analyse comparative : bloc GEK vs bloc HF

Le bloc GEK (`if do_GEK:`) avait été écrit mais jamais testé. Comparaison avec le bloc HF (référence fonctionnelle) :

| Aspect | Bloc HF (référence) | Bloc GEK (avant fix) | Bloc GEK (après fix) |
|---|---|---|---|
| Noyau | Matérn 5/2 (OT KRG) | `corr='matern52'` → **ValueError** | `corr='squar_exp'` |
| Gradient passé à OT | `gradient=grad_func` | absent → FD silencieux OT | absent (étape 4 future) |
| Solver FORM | AbdoRackwitz + MaxIter + CheckStatus + MaxConstraintError | Cobyla sans paramètres | Cobyla + MaxIter + CheckStatus + MaxConstraintError |
| Affichage post-FORM | dg/du + importance | **absent** (bloc manquant) | importance uniquement |

### B2. Recherche bibliographique : bibliothèques Python GEK open source

**Résultat :** aucune bibliothèque Python open source ne dispose d'un **full GEK** (matrice de covariance augmentée avec blocs dérivées). SMT/GEKPLS est la seule option existante, avec des limitations importantes.

| Bibliothèque | GEK / dérivées | Motif d'exclusion |
|---|---|---|
| **SMT (GEKPLS)** | ✓ partiel | PLS réduit nx→n_comp (inutile ici nx=2), noyau restreint à `squar_exp`/`abs_exp` uniquement |
| GPy (Sheffield) | ✗ | 0 résultat pour "derivative" dans le code source |
| GPyTorch / BoTorch | ✗ | `HigherOrderGP` = outputs tensoriels, pas observations de dérivées |
| scikit-learn | ✗ | Feature request explicitement rejetée (issue #11481, juillet 2022) |
| egobox | ✗ | Gradient-free par design |
| OpenTURNS | ✗ | `KrigingAlgorithm` standard, pas d'injection de dérivées dans la matrice |

**Conclusion :** full GEK avec Matérn + gradients adjoints doit être codé manuellement. Justifie la direction prise dans le projet.

**Fichier de synthèse créé :** `GEK_bibliotheques_open_source.md`

### B3. predict_derivatives dans GEKPLS — existence et instabilité

**Découverte (source : `krg_based.py` ligne 227) :**
- `KrgBased.supports["derivatives"] = True`
- `_predict_derivatives(x, kx)` existe, applique la règle de chaîne PLS
- Appel : `sm.predict_derivatives(u_np, kx)` → `ndarray(n_eval, 1)` — dérivée de g w.r.t. variable kx dans l'espace U original (pas espace PLS réduit)

**Instabilité connue (Issue SMT #186) :**
- `poly='linear'` → gradient faux (notre code utilise `poly='constant'` → a priori OK)
- Certaines configurations de `n_comp` provoquent `IndexError`
- Utilisation dans un optimiseur risquée sans validation préalable

**Protocole de validation (à coder en étape 3) :**
```python
x0 = np.atleast_2d([u_fc_val, u_fy_val])
eps = 1e-4
grad_sm = np.array([float(sm.predict_derivatives(x0, kx)[0, 0]) for kx in range(n_var)])
y0 = float(sm.predict_values(x0)[0, 0])
grad_fd = np.zeros(n_var)
for kx in range(n_var):
    xp = x0.copy(); xp[0, kx] += eps
    grad_fd[kx] = (float(sm.predict_values(xp)[0, 0]) - y0) / eps
err_rel = np.abs(grad_sm - grad_fd) / (np.abs(grad_fd) + 1e-12)
# CRITÈRE : err_rel < 1e-3 partout → gradient fiable pour étape 4
```

### B4. Discussion théorique : noyau gaussien vs Matérn pour réponses élasto-plastiques

**Question posée :** pourquoi les utilisateurs GEKPLS en CFD peuvent utiliser `squar_exp` (gaussien C∞) alors que les réponses EF ne sont pas forcément C∞ ?

**Réponse :**
- En **CFD incompressible** (régime sub-critique, pas de choc) : les réponses aérodynamiques sont C∞ par rapport aux paramètres de forme → `squar_exp` justifié
- Pour **réponses élasto-plastiques STRAINS** : le multiplicateur α⁺ n'est pas forcément C∞ (singularité possible si le mode de rupture change dans l'espace des paramètres) → Matérn 5/2 (C⁴, donc C² en dérivées) plus adapté
- **Cas PCE résidu** : si KRG est fait sur le résidu g − g_PCE, le résidu a la **même régularité que g** (la PCE est C∞, elle ne change pas la régularité du résidu). Donc squar_exp reste injustifié pour le résidu.
- **Conséquence pour le projet :** l'utilisation de `squar_exp` dans GEKPLS est une limitation méthodologique connue, à documenter dans la comparaison GEK vs KRG.

### B5. Plan d'activation GEK en 4 étapes

Fichier de plan complet : `plan_GEK_corrections.md`

**Étape 1 — Cobyla + FD (premier run fonctionnel, baseline)**
- Fix bloquant : `corr='squar_exp'`
- Cobyla + `setMaximumIterationNumber` + `setCheckStatus(False)` + `setMaximumConstraintError(1e-2)`
- Affichage post-FORM pour do_GEK=True (importance factors)
- `myFunction = ot.PythonFunction(n_var, 1, g_GEK)` → OT calcule gradient par FD (n_var+1 appels/iter)

**Étape 2 — AbdoRackwitz + FD (référence FD avec bon solver)**
- Seul changement vs étape 1 : `ot.Cobyla()` → `ot.AbdoRackwitz()`
- Tous paramètres identiques
- OT continue d'utiliser FD → n_var+1 = 3 appels g_GEK par itération

**Étape 3 — Validation de predict_derivatives (bloc test standalone)**
- Comparer `sm.predict_derivatives(x0, kx)` vs FD numérique sur point typique en espace U
- Critère : erreur relative < 1e-3 sur tous les kx
- Ne pas passer à l'étape 4 si ce critère n'est pas satisfait

**Étape 4 — AbdoRackwitz + predict_derivatives (gradient analytique)**
```python
def grad_GEK(u):
    u_np = np.array(u).reshape(1, -1)
    grad = [float(sm.predict_derivatives(u_np, kx)[0, 0]) for kx in range(n_var)]
    return [grad]   # shape [1][n_var] — même convention que grad_func HF

myFunction = ot.PythonFunction(n_var, 1, g_GEK, gradient=grad_GEK)
solver = ot.AbdoRackwitz()
# tous paramètres identiques étape 2
```

---

## C. Modifications du code — Session 20 avril

### C1. `corr='matern52'` → `'squar_exp'` (ligne 444) [BUG BLOQUANT]

```python
# AVANT
sm = GEKPLS(corr='matern52', ...)   # ValueError à la construction

# APRÈS
sm = GEKPLS(corr='squar_exp', ...)
```
**Cause :** GEKPLS redéclare l'option `corr` dans `gekpls.py` avec `values=("abs_exp", "squar_exp")` seulement. La validation `OptionsDictionary` refuse toute autre valeur.

### C2. Cobyla solver avec paramètres complets pour do_GEK (lignes 525–529) [BUG]

```python
# AVANT
solver = ot.Cobyla()
solver.setStartingPoint([0.0] * n_var)

# APRÈS
solver = ot.Cobyla()
solver.setMaximumIterationNumber(n_max_FORM)
solver.setCheckStatus(False)
solver.setMaximumConstraintError(1e-2)
solver.setStartingPoint([0.0] * n_var)
```

### C3. Affichage post-FORM pour do_GEK=True (lignes 629–640) [MANQUE]

```python
# AVANT : bloc manquant pour do_GEK=True
if do_GP:
    if do_GEK==False:
        grad_star = metamodel_KRG.gradient(U_res)
        ...

# APRÈS
if do_GP:
    if do_GEK:
        importance = result.getImportanceFactors()
        for i in range(n_var):
            print(f"  Importance factor {params_names[i]}: {importance[i]:.4f}")
    else:
        grad_star = metamodel_KRG.gradient(U_res)
        for i in range(n_var):
            print(f"  dg/du_{params_names[i]} en u* = {grad_star[i, 0]:.6f}")
        importance = result.getImportanceFactors()
        for i in range(n_var):
            print(f"  Importance factor {params_names[i]}: {importance[i]:.4f}")
```
**Note :** pas de gradients pour do_GEK=True (predict_derivatives non encore validé — étape 3).

### C4. Garde `not do_GEK` sur le bloc test GP (ligne 660) [NameError + clarté]

```python
# AVANT
if do_GP_test and do_GP:

# APRÈS (fix user)
if do_GP_test and do_GP and not do_GEK:
```
**Raisons :** (1) `metamodel_KRG` non défini si do_GEK=True → NameError ; (2) le test "g_KRG vs g_HF au point u*" n'a de sens que sur le chemin KRG.

### C5. Warm start : `y_hf` mis à jour avec `np.vstack` (ligne 554)

```python
# AVANT (tentative qui ne fonctionne pas)
y_hf.append(y_to_add)    # ndarray n'a pas .append()

# APRÈS
y_hf = np.vstack([y_hf, [[y_to_add]]])   # conserve shape (n0+1, 1)
```

### C6. Starting point du second FORM warm start = `U_warm` (ligne 572) [CORRECT]

```python
solver.setStartingPoint(U_warm)   # u* du premier FORM, pas [0,0]
```
**Note :** c'est la bonne valeur. Le résumé précédent indiquait à tort que `[0.0]*n_var` était en place ; c'était déjà `U_warm` dans le code final.

---

## D. Résultats du run F=0.210 en fin de session

**Flags au moment du run :**
```python
do_GEK = False
do_GP = True
do_pce = False
do_warm_start = True
tol_warm_start = 0.001
n0 ≈ 25  # user avait modifié (max(15,1)=15 dans le code mais output montre 25 alpha+ values)
n_max_FORM = 50
# dsLoad.txt : Z='-0.210'
```

**Résultats :**

| Grandeur | Valeur |
|---|---|
| β_KRG | 3.555065 |
| Pf | 1.889e-04 |
| u* | [-0.571, -3.509] |
| fc* | 30.29 MPa |
| fy* | 512.54 MPa |
| Importance fc | 2.6% |
| Importance fy | 97.4% |
| n_iter affiché | 0 |
| Warm start déclenché | Non (g_KRG(u*) < 0.001) |

**Analyse du n_iter=0 :** warm start non déclenché → seul le premier FORM AbdoRackwitz a tourné. Le 0 est soit un artefact `getIterationNumber()` pour AbdoRackwitz (OT ne rapporte pas toujours le bon compteur), soit une convergence immédiate. À investiguer.

**Comparaison :** β_KRG=3.555 vs β_HF=3.784 → écart 6.1% avec n0≈25.

---

## E. État du code en fin de session 20 avril

```python
# Flags
do_GEK = False        # GEK désactivé — étape 1 implémentée, pas encore lancée
do_GP = True
do_pce = False
do_warm_start = True
tol_warm_start = 0.001
n_start = 1
n0 = max(15, n_start)  # = 15 par défaut (user modifie selon le test)
n_max_FORM = 50

# FORM GEK (do_GEK=True) — étape 1 implémentée
solver = ot.Cobyla()
solver.setMaximumIterationNumber(n_max_FORM)
solver.setCheckStatus(False)
solver.setMaximumConstraintError(1e-2)
solver.setStartingPoint([0.0] * n_var)
myFunction = ot.PythonFunction(n_var, 1, g_GEK)  # FD automatique OT

# FORM KRG (do_GEK=False) — avec warm start automatique
solver = ot.AbdoRackwitz()
solver.setMaximumIterationNumber(n_max_FORM)
solver.setCheckStatus(False)
solver.setMaximumConstraintError(1e-2)
solver.setStartingPoint([0.0] * n_var)
# warm start : lignes 547–575 (déclenché si g_KRG(u*) > 0.001)

# dsLoad.txt : Z='-0.210' (F=0.210 MN)
```

**Bug warm start restant (non critique) :**
- Indentation `if do_pce: yt -= y_pce` (ligne 556) — sur-indenté ; sans effet si do_pce=False.

---

## F. Fichiers créés en session 20 avril

| Fichier | Contenu |
|---|---|
| `plan_GEK_corrections.md` | Plan détaillé 4 étapes GEK avec code snippets, vérifications, risques |
| `GEK_bibliotheques_open_source.md` | Tableau des bibliothèques Python GEK — seul SMT disponible, full GEK absent |
| `GEKPLS_OpenTURNS_FORM_contexte.md` | (fourni par user) Contexte instabilité predict_derivatives, patterns wrapper OT, protocole de validation — document de référence pour étapes 3-4 |

---

## G. Prochaines étapes

1. **Comprendre n_iter=0** : relancer avec do_warm_start=True, vérifier si le warm start se déclenche et si n_iter=0 est toujours là
2. **Lancer GEK étape 1** : do_GEK=True, do_warm_start=False, F=0.235 (β faible) → vérifier β_GEK vs β_KRG, pas de ValueError
3. **GEK étape 2** : `ot.Cobyla()` → `ot.AbdoRackwitz()` dans le bloc do_GEK, relancer
4. **GEK étape 3** : ajouter bloc validation `predict_derivatives` vs FD, vérifier err_rel < 1e-3
5. **GEK étape 4** : ajouter `gradient=grad_GEK` après validation étape 3

---

---

# Session du 20 avril 2026 (fin de journée) — Comparaison noyaux + GEK warm start + GEK vs KRG

**Fichier principal :** `C:\_workingDir\_SF\test flexion\AC_pure_flexion.py`
**Objectifs :**
1. Comparer KRG Matérn 5/2 vs KRG squar_exp sur F=0.210, n0=20, DOE fixé, avec et sans warm start → compléter `comparaison_Matern_exp.md`
2. Activer GEK étape 2 (AbdoRackwitz + FD) avec warm start
3. Comparer GEK vs KRG squar_exp à n0=20 et n0=5 (DOE fixé) → créer `comparaison_GEK_KRG.md`

---

## A. Comparaison Matérn vs squar_exp — F=0.210, n0=20, DOE fixé

### A1. Runs sans warm start (DOE fixé 20 pts)

**KRG Matérn sans warm start :**
- β=4.846, erreur=28%, u*=[-0.759, -4.786], g_HF=-4.46e-02, g_KRG=+8.0e-04
- n_iter=21, gradient non nul → converge proprement
- Le KRG pense être sur la surface limite (g_KRG≈0) mais g_HF=-0.045 → surface limite mal localisée

**KRG squar_exp sans warm start :**
- β=3.996, u*=[-1.143, -3.830], g_HF=-7.33e-03, g_KRG=+2.19e-02
- n_iter=51 (max épuisé), **gradient=0** → échec non-physique
- Cause : squar_exp décroît en e^{-r²} → corrélations nulles loin des données → gradient nul

**Conclusion sans warm start :** Matérn converge (résultat physique, 28% d'erreur due au DOE). squar_exp échoue (gradient nul) car sa décroissance super-exponentielle annule toute corrélation dans la queue de distribution.

### A2. Paradoxe warm start — tol_warm_start

Le warm start a une condition de déclenchement : `g_KRG(u*) > tol_warm_start`.

- **Matérn** : g_KRG(u*)=+8.0e-04 < tol=0.001 → warm start **non déclenché** (KRG croit être sur la limite). Résultat inchangé : β=4.846, 28%.
- **squar_exp** : g_KRG(u*)=+2.19e-02 > tol=0.001 → warm start **déclenché** → β=3.792, 0.2%.

**Fix appliqué :** `tol_warm_start` réduit de 0.001 à 0.0001 pour déclencher aussi le warm start pour Matérn (g_KRG=8e-04 > 0.0001).

### A3. Runs avec warm start

**KRG Matérn + warm start (tol=0.0001) :**
- β=3.682, erreur=2.7%, u*=[-0.643, -3.625], g_HF=+4.5e-03
- Fichier : `out_KRG_matern_F210_n20_fixedDOE_warmstart.txt`

**KRG squar_exp + warm start :**
- β=3.792, erreur=0.2%, u*=[-0.538, -3.754], g_HF=-3.3e-04
- Fichier : `out_KRG_squar_exp_F210_n20_fixedDOE_warmstart.txt`

**Conclusion avec warm start :** squar_exp plus précis (0.2% vs 2.7%). Résultat contre-intuitif : avec warm start, le noyau théoriquement moins adapté (C∞ vs C⁴) donne de meilleurs résultats.

### A4. Analyse mathématique des noyaux

**Squared Exponential :**
```
k_SE(x,xi) = sigma² * exp(-‖x-xi‖²/2l²)        → décroît en e^{-r²} (super-exponentielle)
dk_SE/dxj  = -(xj-xij)/l² * k_SE(x,xi)
```

**Matérn 5/2 :**
```
k_M52(x,xi) = sigma²*(1 + √5r/l + 5r²/3l²)*exp(-√5r/l)   → décroît en r*e^{-r} (plus lent)
dk_M52/dxj ∝ (xj-xij)*(1 + √5r/l)*exp(-√5r/l)
```

Le gradient du krigeage s'écrit `∇ĝ(x) = ∇k(x,X)ᵀ K⁻¹(y - μ·1)`. Quand x est loin de tous les xi, k_SE(x,xi)≈0 pour tout i → gradient nul. Matérn décroît plus lentement → gradient survit dans les zones creuses.

**Avec DOE riche :** les deux kernels se valent car les prédictions sont contraintes par de nombreux points proches. Avantage squar_exp : surface plus lisse → gradients plus réguliers pour AbdoRackwitz. **Avec DOE sparse :** Matérn plus robuste (pas de plateau gradient=0).

---

## B. Discussions théoriques — Warm start

### B1. Warm start : mécanisme atomique

Le warm start est indivisible : enrichissement DOE **et** départ du 2ème FORM depuis u*₁. Les deux moitiés n'ont de sens qu'ensemble. Si le point u*₁ est mauvais (pas éclaireur), les deux moitiés échouent ensemble — un demi-warm start n'aurait rien sauvé.

### B2. Warm start itératif — convergence sans cycling

Un enchaînement itératif de warm starts ne peut pas cycler : une fois zone A enrichie, la vraie valeur HF y est enregistrée → le KRG corrige sa fausse surface limite en zone A → le FORM ne peut pas retourner en zone A de la même façon. La progression est monotone (chaque zone enrichie est définitivement corrigée) mais pas garantie rapide : l'errance dans de nouvelles fausses zones est possible.

### B3. Point dégénéré comme éclaireur

Un point u* trouvé avec gradient=0 (squar_exp sans warm start) peut servir d'éclaireur si :
- Il est dans le bon voisinage (même quadrant, même ordre de β)
- L'enrichissement en ce point crée une corrélation non nulle dans la région → gradient non nul après re-training

Si le point est loin de la vraie surface limite (mauvais quadrant), l'enrichissement est inutile ET le 2ème FORM repart dans la mauvaise direction.

Dans notre cas (squar_exp n0=20 F=0.210) : u*₁=[-1.143, -3.830] était dans le bon voisinage → éclaireur suffisant.

---

## C. GEK étape 2 — AbdoRackwitz + FD, avec warm start

### C1. Run GEK FD sans warm start (F=0.210, n0=20, LHS)

- β=3.547, erreur=6.3%, u*=[-0.568, -3.502]
- g_HF(u*)=+0.010, g_GEK(u*)=+0.010
- g_GEK(u*_HF)=0.119 → GEK mal calé sur vraie surface limite
- Fichier : `out_GEK_abdoRackwitz_FD_nowarmstart_F210.txt`

**Note importante :** "sans FD" dans le contexte de ce projet = avec gradient analytique `predict_derivatives`. "avec FD" = gradient calculé automatiquement par OT par différences finies. Le run GEK AbdoRackwitz actuel est TOUJOURS avec FD (OT FD) tant que `predict_derivatives` n'est pas passé comme argument gradient à `ot.PythonFunction`.

### C2. Bugs dans le warm start GEK (corrigés)

**Bug 1 — sensitivity=False (ligne 553) :**
```python
# AVANT (bug)
y_to_add, all_grad_to_add = run_HF(..., sensitivity=False)
# APRÈS (fix)
y_to_add, all_grad_to_add = run_HF(..., sensitivity=True)
```
GEKPLS a besoin des vrais gradients adjoints STRAINS pour le re-training. Avec `sensitivity=False`, `all_grad_to_add` est vide/None.

**Bug 2 — shape 3D du vstack (ligne 556) :**
```python
# AVANT (bug) — crée tableau 3D (1,1,n_var) incompatible avec all_grad_U_g (n0,n_var)
all_grad_U_g = np.vstack([all_grad_U_g, [[all_grad_to_add]]])
# APRÈS (fix)
all_grad_U_g = np.vstack([all_grad_U_g, [all_grad_to_add]])
```

**Bug 3 — starting point du 2ème FORM (ligne 586) :**
```python
# AVANT (bug)
solver.setStartingPoint([0.0] * n_var)   # annule l'effet du warm start
# APRÈS (fix, fait par l'utilisatrice)
solver.setStartingPoint(U_warm)
```

**Bug 4 — MaximumConstraintError trop stricte (ligne 541) :**
```python
# AVANT → RuntimeError g=0.0119 > 0.01
solver.setMaximumConstraintError(1e-2)
# APRÈS (fix, fait par l'utilisatrice)
solver.setMaximumConstraintError(5e-2)
```

### C3. Run GEK FD avec warm start (F=0.210, n0=20)

- β=3.813, erreur=0.8%, u*=[-1.400, -3.546]
- Warm start déclenché : u*₁=[-0.338, -3.308]
- g_HF(u*)=+2.8e-03, g_GEK(u*)=+3.0e-03
- Fichier : `out_GEK_abdoRackwitz_FD_warmstart_F210.txt`

---

## D. Comparaison GEK vs KRG — Sweep n0 et DOE fixé

### D1. Sweep n0 KRG squar_exp + warm start (F=0.210, LHS aléatoire)

| n0 | β_KRG | Erreur β | g_HF(u*) | Warm start | Fichier |
|---|---|---|---|---|---|
| 20 | 3.792 | 0.2% | -3.3e-04 | oui | `out_KRG_squar_exp_F210_n20_fixedDOE_warmstart.txt` |
| 15 | 3.561 | 5.9% | +9.7e-03 | oui (u*₁=[-1.654,-5.442]) | `out_KRG_squar_exp_warmstart_F210_n15.txt` |
| 10 | 3.723 | 1.6% | +2.6e-03 | oui (u*₁=[-0.191,-5.855]) | `out_KRG_squar_exp_warmstart_F210_n10.txt` |
| 8 | 3.773 | 0.3% | +6.5e-04 | oui (u*₁=[-2.773,-3.872]) | `out_KRG_squar_exp_warmstart_F210_n8.txt` |
| **5** | — | — | — | **RuntimeError g=0.089** | `out_KRG_squar_exp_warmstart_F210_n5.txt` |

KRG squar_exp + warm start est remarquablement robuste jusqu'à n0=8. Rupture à n0=5.

**Note :** les résultats n0=15,10,8 utilisent des LHS aléatoires différents (pas DOE fixé) — variabilité possible entre runs.

### D2. DOE fixé n0=5 pour comparaison GEK vs KRG

DOE n0=5 F=0.210 (hardcodé dans le code) :
```python
U_doe = ot.Sample([
    [ 0.3230547119390826,  1.0835994509983533],
    [-0.4488634187720924,  0.4427899936470972],
    [ 0.9294010504086743, -0.0369984238561853],
    [-1.3637190976248115, -0.5897068143758146],
    [ 0.2494332098595843, -1.4365967484054665]
])
```

**KRG squar_exp n0=5 + warm start (DOE fixé) :**
- RuntimeError : g=0.089 > MaxConstraintError=0.05 → échec
- Fichier : `out_KRG_squar_exp_warmstart_F210_n5_fixedDOE.txt`

**GEK FD n0=5 + warm start (DOE fixé) :**
- β=3.784, erreur=**0.0%**, u*=[-0.466, -3.755]
- Warm start déclenché : u*₁=[-0.474, -3.806]
- g_HF(u*)=+3.0e-06, g_GEK(u*)=+1.0e-06 → quasi parfait
- Fichier : `out_GEK_abdoRackwitz_FD_warmstart_F210_n5_fixedDOE.txt`

**Interprétation :** GEK converge à n0=5 car chaque point d'entraînement apporte n_var+1=3 informations (1 valeur g + 2 gradients adjoints STRAINS). KRG n'a que 1 information par point. À n0=5, GEK dispose effectivement de ~15 contraintes vs ~5 pour KRG.

---

## E. État du code en fin de session

```python
# Flags
do_GEK = False        # à changer selon le test
do_GP = True
do_pce = False
do_warm_start = True
tol_warm_start = 0.0001   # réduit de 0.001 → déclenche aussi pour Matérn
n_start = 1
n0 = max(5, n_start)      # actuel : 5

# DOE fixé n0=5 F=0.210 (hardcodé après sa.generate())
U_doe = ot.Sample([
    [ 0.3230547119390826,  1.0835994509983533],
    [-0.4488634187720924,  0.4427899936470972],
    [ 0.9294010504086743, -0.0369984238561853],
    [-1.3637190976248115, -0.5897068143758146],
    [ 0.2494332098595843, -1.4365967484054665]
])

# Bloc FORM GEK (do_GEK=True)
solver = ot.AbdoRackwitz()
solver.setMaximumIterationNumber(n_max_FORM)    # 50
solver.setCheckStatus(False)
solver.setMaximumConstraintError(5e-2)          # 1er FORM
# Warm start GEK :
#   sensitivity=True pour run_HF (gradients pour re-training GEKPLS)
#   all_grad_U_g = np.vstack([all_grad_U_g, [all_grad_to_add]])   # shape (n0+1, n_var)
#   solver.setStartingPoint(U_warm)   # 2ème FORM
#   solver.setMaximumConstraintError(1e-2)   # 2ème FORM

# Bloc FORM KRG (do_GEK=False) — inchangé
solver = ot.AbdoRackwitz()
solver.setMaximumIterationNumber(n_max_FORM)
solver.setCheckStatus(False)
solver.setMaximumConstraintError(5e-2)
solver.setStartingPoint([0.0] * n_var)
# warm start KRG : condition g_KRG(u*) > tol_warm_start=0.0001

# dsLoad.txt : Z='-0.210' (F=0.210 MN)
```

---

## F. Fichiers créés/modifiés en cette session

| Fichier | Modifications |
|---|---|
| `comparaison_Matern_exp.md` | Tableau 3 ajouté (F=0.210, n0=20, DOE fixé, avec warm start) |
| `comparaison_GEK_KRG.md` | **Créé** — Tableau 1 (n0=20) et Tableau 2 (n0=5, DOE fixé) |
| `AC_pure_flexion.py` | sensitivity=True, vstack fix, starting point warm start GEK, MaxConstraintError 5e-2, tol_warm_start=0.0001, n0=max(5,...), DOE fixé n0=5 hardcodé |

**Fichiers output importants :**
- `out_KRG_matern_F210_n20_fixedDOE_warmstart.txt` : KRG Matérn n0=20+1 warm start → β=3.682
- `out_KRG_squar_exp_F210_n20_fixedDOE_warmstart.txt` : KRG squar_exp n0=20+1 warm start → β=3.792
- `out_GEK_abdoRackwitz_FD_warmstart_F210.txt` : GEK FD n0=20+1 warm start → β=3.813
- `out_GEK_abdoRackwitz_FD_warmstart_F210_n5_fixedDOE.txt` : GEK FD n0=5+1 warm start → β=3.784 (0.0%)
- `out_KRG_squar_exp_warmstart_F210_n5_fixedDOE.txt` : KRG squar_exp n0=5 warm start → RuntimeError

---

## G. Principe de remplissage de comparaison_GEK_KRG.md

**Structure :** un tableau par condition de test (combinaison n0 / DOE). Colonnes : HF (référence), KRG squar_exp, GEK FD. Lignes : u*, β, g_HF(u*), g_méta(u*), erreur β, u* warm start déclencheur.

**Règles :**
- Toujours noter si DOE fixé ou LHS aléatoire (comparaisons rigoureuses = DOE fixé identique)
- Toujours noter le gradient GEK utilisé (FD ou predict_derivatives — étape 2 ou 4)
- Case KRG vide (—) si RuntimeError
- Conclusion obligatoire sous chaque tableau avec interprétation physique
- Ne jamais écraser un tableau existant — ajouter un nouveau tableau

**Quand ajouter un tableau :** après chaque paire de runs (KRG + GEK) sur le même DOE fixé. Compléter aussi `comparaison_Matern_exp.md` si la comparaison de noyau est impliquée.

---

## H. Prochaines étapes

1. **Étape 4 GEK** : gradient analytique `predict_derivatives` → remplacer FD par `gradient=grad_GEK` dans `ot.PythonFunction` — comparer β_GEK_analytique vs β_GEK_FD sur DOE fixé n0=5 et n0=20
2. **Étape 3 optionnelle** : valider `predict_derivatives` vs FD numérique sur la surface GEK entraînée (err_rel < 1e-3) avant de passer à l'étape 4
3. **Étendre comparaison GEK vs KRG** : tester d'autres valeurs de F (β=0.95, β=2.08) pour vérifier si l'avantage GEK à faible n0 se confirme sur des cas plus faciles
4. **Remettre le DOE en LHS** pour les runs de production (commenter le DOE fixé hardcodé)

---

---

# Session du 20 avril 2026 (nuit) — Gradient analytique GEK + PC-KRG implémentation + Analyse Moustapha 2022

**Fichier principal :** `C:\_workingDir\_SF\test flexion\AC_pure_flexion.py`
**Objectifs :**
1. Implémenter le gradient analytique GEK (`do_GEK_analytic_grad`) et valider contre FD
2. Comparer GEK FD vs GEK gradient analytique sur DOE fixé n0=5+1 (warm start), F=0.210
3. Vérifier et corriger l'implémentation PC-KRG existante
4. Analyser la méthode PC-KRG de Moustapha et al. (2022) et comparer au code

---

## A. Ajout du flag `do_GEK_analytic_grad`

### A1. Contexte et motivation

Suite à la validation de `predict_derivatives` (étape 3 du plan GEK), l'utilisatrice a ajouté une option pour contrôler si le FORM GEK utilise le gradient analytique (`sm.predict_derivatives`) ou le gradient FD (calculé automatiquement par OT).

### A2. Flag ajouté (ligne 320)

```python
do_GEK_analytic_grad = True   # nouvelle ligne ajoutée par l'utilisatrice
```

Flags au moment de la session :
```python
do_GEK = True
do_GEK_analytic_grad = True   # NOUVEAU
do_GP = True
do_pce = False
do_warm_start = True
tol_warm_start = 0.0001
n0 = max(5, n_start)  # = 5
n_max_FORM = 50
```

### A3. Utilisation du flag dans le FORM GEK (lignes 557–571)

```python
if do_GP:
    if do_GEK:
        if do_GEK_analytic_grad:
            myFunction = ot.PythonFunction(
                n_var,
                1,
                g_GEK,
                gradient=grad_g_GEK
            )
        else:
            myFunction = ot.PythonFunction(
                n_var,
                1,
                g_GEK
            )
```

Quand `do_GEK_analytic_grad=True` : OT utilise `grad_g_GEK` (appelle `sm.predict_derivatives`) pour le gradient — 1 appel GEK par itération au lieu de n_var+1=3.
Quand `do_GEK_analytic_grad=False` : OT calcule le gradient par FD (n_var+1=3 appels GEK par itération).

---

## B. Bloc de validation gradient analytique vs FD (lignes 523–548)

### B1. Structure du bloc

Le bloc est inséré entre la construction du métamodèle GEK et la section FORM, déclenché uniquement si `do_GP and do_GEK`.

```python
if do_GP and do_GEK:
    u_test = np.array([-1.2, -3.0])   # point hors DOE
    h = 1e-4                           # pas FD centré

    grad_ana = grad_g_GEK(u_test)     # shape (n_var, 1) via predict_derivatives

    grad_fd = []
    for i in range(n_var):
        e = np.zeros(n_var)
        e[i] = h
        gp = g_GEK(u_test + e)[0]
        gm = g_GEK(u_test - e)[0]
        grad_fd.append((gp - gm) / (2 * h))

    print("\n=== Validation gradient GEK ===")
    print(f"Point test u = {u_test.tolist()}")
    print(f"{'Var':<6} {'Analytique':>14} {'FD centré':>14} {'Err rel':>12}")
    for i in range(n_var):
        ana = grad_ana[i][0]
        fd  = grad_fd[i]
        err = abs(ana - fd) / (abs(fd) + 1e-12)
        print(f"u_{i:<4} {ana:>14.6e} {fd:>14.6e} {err:>11.2%}")
    print("================================\n")
```

### B2. Résultat de la validation

**Résultat obtenu (point de test u=[-1.2, -3.0], DOE fixé n0=5, F=0.210) :**

| Var | Analytique | FD centré | Erreur rel |
|-----|-----------|-----------|-----------|
| u_0 | ~ | ~ | **0.00%** |
| u_1 | ~ | ~ | **0.00%** |

→ Erreur 0.00% sur les deux variables → gradient `predict_derivatives` parfaitement fiable pour ce DOE.

---

## C. Bug warm start GEK — `do_GEK_analytic_grad` non appliqué

### C1. Problème identifié

Avant la correction, le warm start GEK (second FORM, lignes ~610-625) reconstruisait `myFunction` **toujours avec gradient analytique**, indépendamment de `do_GEK_analytic_grad` :

```python
# AVANT (bug) — toujours gradient analytique dans le warm start
myFunction = ot.PythonFunction(n_var, 1, g_GEK, gradient=grad_g_GEK)
```

### C2. Correction appliquée (lignes 612–615)

```python
# APRÈS (correction)
if do_GEK_analytic_grad:
    myFunction = ot.PythonFunction(n_var, 1, g_GEK, gradient=grad_g_GEK)
else:
    myFunction = ot.PythonFunction(n_var, 1, g_GEK)
```

**Raison :** le warm start doit être cohérent avec le premier FORM. Si `do_GEK_analytic_grad=False`, le warm start doit aussi utiliser FD.

---

## D. Résultats comparatifs GEK FD vs GEK analytique

### D1. Tableau comparatif (F=0.210, n0=5, DOE fixé identique)

Référence : β_HF = 3.784, u*_HF = [-0.526, -3.747]

| | **HF** | **GEK FD n0=5+1** | **GEK analytique n0=5+1** |
|---|---|---|---|
| u* | [-0.526, -3.747] | [-0.466, -3.755] | [-0.466, -3.755] |
| β | 3.784 | **3.7838** | **3.7838** |
| g_HF(u*) | ≈ 0 | +3.0e-06 | +3.0e-06 |
| g_méta(u*) | ≈ 0 | +1.0e-06 | +1.0e-06 |
| Erreur relative β | 0% (réf.) | **0.0%** | **0.0%** |
| n_iter FORM | — | 1 | 1 |
| Warm start déclenché | — | oui | oui |
| u* warm start déclencheur | — | [-0.474, -3.806] | [-0.474, -3.806] |

**Conclusion :** résultats **strictement identiques** entre GEK FD et GEK analytique sur ce DOE fixé n0=5+1. L'avantage du gradient analytique est uniquement en coût de calcul (1 appel vs 3 par itération), pas en précision.

---

## E. Analyse et corrections de l'implémentation PC-KRG

### E1. Contexte

L'utilisatrice avait déjà codé la structure PC-KRG dans le script. Elle a demandé une révision complète avant activation (`do_pce=True`).

### E2. Structure PC-KRG dans le code

**Décomposition :**
```
g_PCKRG(u) = g_PCE(u) + g_KRG_résidu(u)
```

- KRG entraîné sur résidus : `yt = y_hf - y_pce` (ligne 450)
- Reconstruction du modèle complet : `metamodel_KRG += metamodel_pce` (ligne 492–493)

**PCE (lignes 385–435) :**
```python
q = 0.75
enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(n_var, q)
basis = ot.OrthogonalProductPolynomialFactory([ot.HermiteFactory()] * n_var, enumerateFunction)
max_degree = 2
basis_size = enumerateFunction.getBasisSizeFromTotalDegree(max_degree)
basisStrategy = ot.FixedStrategy(basis, basis_size)
selectionStrategy = ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut())
projectionStrategy = ot.LeastSquaresStrategy(selectionStrategy)
algo = ot.FunctionalChaosAlgorithm(inputSample, outputSample, dist_U, basisStrategy, projectionStrategy)
algo.run()
metamodel_pce = result.getMetaModel()
y_pce = np.array(metamodel_pce(U_doe))
```

**KRG sur résidus (lignes 484–493) :**
```python
basis = ot.ConstantBasisFactory(n_var).build()
covarianceModel = ot.SquaredExponential([1.0] * n_var)
algo_KRG = ot.KrigingAlgorithm(xt, yt, covarianceModel, basis)
algo_KRG.run()
result = algo_KRG.getResult()
metamodel_KRG = result.getMetaModel()
if do_pce:
    metamodel_KRG += metamodel_pce    # AJOUTÉ PAR L'UTILISATRICE
```

### E3. Question sur l'opérateur `+=`

**Question :** `metamodel_KRG = result.getMetaModel()` retourne quel type OT ? L'opération `metamodel_KRG += metamodel_pce` est-elle valide ?

**Réponse :**
- `KrigingResult.getMetaModel()` retourne un **`ot.Function`**
- `ot.Function` supporte les opérateurs arithmétiques : `+`, `-`, `*`, `+=`
- `f += g` crée un **nouvel objet** `ot.Function` évaluant `f(u) + g(u)` — pas de modification in-place, réaffectation de la variable → résultat : un `DualLinearCombinationFunction`
- Dimensions : les deux fonctions sont `(n_var → 1)` ✓

**Conclusion : l'opération est valide** et produit `g_PCKRG(u) = g_KRG_résidu(u) + g_PCE(u)`.

### E4. Vérification des dimensions

| Métamodèle | Entrée | Sortie | Source |
|-----------|--------|--------|--------|
| `metamodel_pce` | `n_var` = 2 | 1 | `ot.FunctionalChaosAlgorithm` |
| `metamodel_KRG` (avant `+=`) | `n_var` = 2 | 1 | `ot.KrigingAlgorithm` |
| `metamodel_KRG` (après `+=`) | 2 | 1 | `DualLinearCombinationFunction` |

Dimensions compatibles ✓.

### E5. Point critique restant — warm start + do_pce

Si `do_warm_start=True` ET `do_pce=True`, le warm start KRG (ligne ~657) reconstruit :
```python
metamodel_KRG = result_KRG.getMetaModel()   # retourne KRG résidu seulement
```
sans ré-appliquer `+= metamodel_pce`. Le PC-KRG post warm start évalue uniquement le résidu KRG, pas le modèle complet.

**Pour ce run (`do_warm_start=False`) : pas de problème. À corriger si ces deux flags sont combinés.**

---

## F. Analyse PC-KRG Moustapha 2022 vs code — Résultat de l'analyse en plan mode

### F1. Méthode PC-KRG dans le papier (Moustapha et al. 2022)

**Décomposition :**
```
g_PCKRG(u) = g_PCE(u) + g_KRG(u)
```
où `g_KRG` est entraîné sur `ε(u) = g_HF(u) - g_PCE(u)`.

**PCE :** base Hermite, degré 1–3 adaptatif, q-norme 0.8, calibration LARS, validation LOO (Q²_LOO).
**KRG résidu :** noyau gaussien (squar_exp), tendance constante, calibration MLE.
**Cadre global :** boucle ALR (Active Learning Reliability) avec 4 modules : métamodèle, algorithme de fiabilité (MCS/SuS/IS), fonction d'apprentissage (U-function/EFF/FBR), critère d'arrêt (β-bornes/β-stabilité).
**DOE :** `max(10, 2M)` points initiaux, enrichissement jusqu'à `100+10M` points max.

### F2. Tableau de comparaison code vs papier

| Composant | Papier | Code | Conformité |
|-----------|--------|------|-----------|
| Décomposition `g_PCE + g_KRG_résidu` | ✓ | `metamodel_KRG += metamodel_pce` (ligne 492) | ✅ |
| Résidu entraînement | `ε = g_HF - g_PCE` | `yt -= y_pce` (ligne 450) | ✅ |
| PCE base | Hermite (gaussiennes) | `ot.HermiteFactory()` × n_var | ✅ |
| PCE calibration | LARS | `ot.LARS()` + `ot.CorrectedLeaveOneOut()` | ✅ |
| PCE degré | 1–3 (adaptatif) | `max_degree = 2` (fixe) | ✅ (dans plage) |
| PCE q-norme | 0.8 | 0.75 | ⚠️ (légère diff) |
| PCE validation LOO | Q²_LOO | `computeR2Score()` | ✅ |
| KRG noyau | Gaussien (squar_exp) | `ot.SquaredExponential` | ✅ |
| KRG tendance | Constante | `ot.ConstantBasisFactory` | ✅ |
| KRG calibration | MLE | `ot.KrigingAlgorithm` (MLE par défaut) | ✅ |
| Algorithme de fiabilité | MCS / SuS / IS | **FORM** AbdoRackwitz | ⚠️ diff majeure |
| Boucle d'apprentissage actif | ALR itératif, learning function | Warm start unique conditionnel | ⚠️ simplifié |
| Critère d'arrêt | β-bornes / β-stabilité | Condition sur g_méta(u*) | ⚠️ absent |

**Conclusion :** le cœur PC-KRG (structure, PCE, KRG résidu) est conforme au papier. Les différences sont méthodologiques : FORM au lieu de MCS, et warm start unique au lieu d'ALR itératif. Ces choix sont cohérents avec l'objectif (calcul efficace, β modéré à élevé).

---

## G. Modifications du code — Session nuit 20 avril

### G1. Ajout flag `do_GEK_analytic_grad` (ligne 320) [NOUVEAU]

```python
do_GEK_analytic_grad = True   # ajouté par l'utilisatrice
```

### G2. Branchement sur `do_GEK_analytic_grad` dans le FORM GEK (lignes 557–571) [NOUVEAU]

```python
# AVANT (toujours FD)
myFunction = ot.PythonFunction(n_var, 1, g_GEK)

# APRÈS
if do_GEK_analytic_grad:
    myFunction = ot.PythonFunction(n_var, 1, g_GEK, gradient=grad_g_GEK)
else:
    myFunction = ot.PythonFunction(n_var, 1, g_GEK)
```

### G3. Correction warm start GEK — même branchement (lignes 612–615) [BUG CORRIGÉ]

```python
# AVANT (bug — toujours gradient analytique)
myFunction = ot.PythonFunction(n_var, 1, g_GEK, gradient=grad_g_GEK)

# APRÈS (corrigé)
if do_GEK_analytic_grad:
    myFunction = ot.PythonFunction(n_var, 1, g_GEK, gradient=grad_g_GEK)
else:
    myFunction = ot.PythonFunction(n_var, 1, g_GEK)
```

### G4. Bloc validation gradient (lignes 523–548) [NOUVEAU]

```python
if do_GP and do_GEK:
    u_test = np.array([-1.2, -3.0])
    h = 1e-4
    grad_ana = grad_g_GEK(u_test)
    grad_fd = []
    for i in range(n_var):
        e = np.zeros(n_var); e[i] = h
        gp = g_GEK(u_test + e)[0]
        gm = g_GEK(u_test - e)[0]
        grad_fd.append((gp - gm) / (2 * h))
    # affichage tableau analytique vs FD centré + erreur relative
```

### G5. `metamodel_KRG += metamodel_pce` (ligne 492–493) [AJOUTÉ PAR L'UTILISATRICE]

```python
if do_pce:
    metamodel_KRG += metamodel_pce   # reconstruit g_PCKRG complet
```

---

## H. État du code en fin de session

```python
# Flags
do_GEK = True
do_GEK_analytic_grad = True   # NOUVEAU — contrôle gradient analytique vs FD
do_GP = True
do_pce = False                # à activer pour PC-KRG
do_warm_start = True
tol_warm_start = 0.0001
n_start = 1
n0 = max(5, n_start)          # = 5
n_max_FORM = 50

# DOE fixé n0=5 F=0.210 (hardcodé)
U_doe = ot.Sample([
    [ 0.3230547119390826,  1.0835994509983533],
    [-0.4488634187720924,  0.4427899936470972],
    [ 0.9294010504086743, -0.0369984238561853],
    [-1.3637190976248115, -0.5897068143758146],
    [ 0.2494332098595843, -1.4365967484054665]
])

# Bloc validation gradient (actif si do_GP and do_GEK)
# → résultat : erreur 0.00% en u=[-1.2, -3.0] ✓

# Bloc FORM GEK
# Premier FORM (lignes 557–582) : branchement sur do_GEK_analytic_grad
# Warm start GEK (lignes 584–625) : branchement sur do_GEK_analytic_grad (CORRIGÉ)
#   - sensitivity=True pour run_HF
#   - all_grad_U_g = np.vstack([all_grad_U_g, [all_grad_to_add]])
#   - solver.setStartingPoint(U_warm)

# Bloc KRG (lignes 627–668) : inchangé, SquaredExponential
#   if do_pce: metamodel_KRG += metamodel_pce  (ligne 492–493)
```

**Note :** pour le prochain run PC-KRG, mettre `do_pce=True`, `do_GEK=False`, `do_warm_start=False`.

---

## I. Résultats clés de la session

| Run | Flags | β | Erreur β | g_HF(u*) | n_iter |
|-----|-------|---|---------|----------|--------|
| GEK FD n0=5+1 warm start | do_GEK_analytic_grad=False | 3.7838 | 0.0% | +3.0e-06 | 1 |
| GEK analytique n0=5+1 warm start | do_GEK_analytic_grad=True | 3.7838 | 0.0% | +3.0e-06 | 1 |

→ FD et analytique identiques en résultat. Avantage analytique : moins d'appels GEK par itération.

---

## J. Prochaines étapes

1. **Lancer PC-KRG** : `do_pce=True`, `do_GEK=False`, `do_warm_start=False`, F=0.210, n0=5
   - Vérifier Q²_LOO PCE (idéalement > 0.90)
   - Comparer β_PCKRG vs β_KRG (3.792) vs β_GEK (3.784) vs β_HF (3.784)
2. **Corriger warm start + do_pce** : si `do_warm_start=True` et `do_pce=True` combinés, ajouter `metamodel_KRG += metamodel_pce` après le re-training KRG dans le warm start (ligne ~657)
3. **Étendre PC-KRG** : tester d'autres n0 et F pour comparer PC-KRG vs KRG vs GEK

---

## K. Principe de remplissage des fichiers .md — Mise à jour

### K1. `comparaison_GEK_KRG.md`

Ajouter un tableau après chaque paire de runs GEK FD / GEK analytique sur DOE fixé identique. Colonnes : HF | GEK FD n0=X | GEK analytique n0=X. Toujours noter le DOE (fixé ou LHS) et si warm start déclenché.

### K2. `comparaison_Matern_exp.md`

Déjà complet pour les 3 tableaux de la session précédente. À compléter si nouveaux tests noyaux.

### K3. Tableau PC-KRG vs KRG vs GEK (nouveau fichier à créer)

**Nom suggéré :** `comparaison_PCKRG_KRG_GEK.md`
**Structure :** un tableau par n0 (colonnes : HF | KRG | GEK FD | PC-KRG). Lignes : u*, β, g_HF(u*), g_méta(u*), erreur β, Q²_LOO PCE (pour PC-KRG), warm start déclenché.
