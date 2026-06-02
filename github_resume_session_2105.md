# Résumé session du 21/05/2026

## Ce qu'on a fait

### 1. Vérification d'OpenTURNS
- OpenTURNS version **1.27** est installé via `pip install openturns`
- Installé dans `C:\python3\lib\site-packages\openturns\`
- `pip install` télécharge une wheel précompilée depuis PyPI — pas de clone GitHub, pas de code source

### 2. Compréhension de l'architecture
- `pip install` installe deux couches : C++ compilé (`.pyd`) + Python pur (`.py`)
- Quand le code fait `import openturns`, Python va chercher dans `site-packages\openturns\`
- Un clone du dépôt source GitHub n'est **pas** utilisé automatiquement par Python
- Modifier les fichiers `.py` dans `site-packages` ne nécessite **pas** de recompilation

### 3. Mise en place du dépôt git sur le bon dossier
- Le dossier `C:\python3\lib\site-packages\openturns\` a été initialisé comme dépôt git
- Branche `fiabilite` créée
- Fichiers binaires (`.pyd`, `.dll`, `.so`) exclus via `.gitignore`
- Commit initial des fichiers Python effectué

### 4. Création du repository GitHub et push
- Repository créé : `github.com/semiastrains/openturns`
- Owner : compte personnel `semiastrains`
- Visibilité : Public
- Branche `fiabilite` poussée avec succès

### 5. Authentification GitHub
- Personal Access Token (PAT classic) créé
- Scopes : `repo`, `workflow`
- Token configuré dans l'URL remote pour l'authentification

---

## Ce qu'on a appris

### Git et GitHub
- Un **dépôt git** = un dossier + un `.git` caché qui contient tout l'historique
- **GitHub** = serveur qui héberge des dépôts git en ligne
- Un **clone** = copie locale complète d'un dépôt distant
- Une **branche** = ligne de travail isolée — les modifications n'affectent pas les autres branches
- Un **push** = envoi du code local vers GitHub
- Un **commit** = snapshot des modifications à un instant donné

### Structure GitHub
- **Repository** = contient le code (fichiers, branches, commits)
- **Project** = tableau de gestion de tâches, pas de code
- **Organization** = regroupe plusieurs repositories sous un nom commun
- `admin-strains` est une organization existante — accès en lecture mais pas de droits de création
- `semiastrains` = compte personnel avec droits complets

### Authentification
- GitHub n'accepte plus les mots de passe pour `git push`
- Il faut un **Personal Access Token (PAT)** à la place
- Le token se configure dans l'URL remote : `https://username:token@github.com/...`
- Scope `workflow` nécessaire si le repo contient des fichiers `.github/workflows/`

### Python et packages
- `import openturns` → Python cherche dans `site-packages\openturns\`
- Modifier un `.py` dans `site-packages` = pris en compte immédiatement au prochain import
- Pas de recompilation nécessaire pour des modifications Python pures

---

## État final

### Repository `semiastrains/openturns`
- Dépôt git local : `C:\python3\lib\site-packages\openturns\` — branche active : `fiabilite`
- Branche `main` — version non modifiée d'OpenTURNS 1.27, figée
- Branche `fiabilite` — branche de travail pour les modifications
- Workflow : modifier `.py` → tester → `git add` → `git commit` → `git push`

### Repository `semiastrains/AC-pure-flexion`
- Dépôt git local : `C:\_workingDir\_SF\test flexion\`
- Branche `fiabilite` uniquement
- Contient : `AC_pure_flexion.py`

### Prochaine étape
- Créer `C:\python3\lib\site-packages\openturns\gepck.py` (classe GEPCKAlgorithm, GEPCK-GLS d'abord)

---

## CONVENTION : "met à jour le git md" ou "le md" = CE FICHIER
`C:\_workingDir\_SF\test flexion\github_resume_session_2105.md`
À chaque fois qu'on dit "met à jour le md" ou "met à jour le git md", c'est ce fichier qu'on modifie.
**Mettre à jour = mettre à jour ce qui est obsolète + ajouter ce qui est nouveau. Ne jamais supprimer ce qui existe encore.**
À chaque début de session après autocompactage : lire ce fichier pour reprendre le contexte complet.

---

## Architecture OpenTURNS 1.27

### Deux couches
- `pip install openturns` → wheel précompilée PyPI — **pas** de code source C++
- Couche C++ : `.pyd` + `.dll` — non modifiable sans recompilation
- Couche Python : `.py` dans `site-packages\openturns\` — modifiable, pris en compte immédiatement

### KrigingAlgorithm — `metamodel.py:5629`
- Classe SWIG (wrapper Python autour de C++)
- Signature : `KrigingAlgorithm(inputSample, outputSample, covarianceModel, basis=None)`
- Délègue **entièrement** le GLS à `GeneralLinearModelAlgorithm` en C++
- GLS en C++ : β = (FᵀC⁻¹F)⁻¹FᵀC⁻¹y — non modifiable en Python
- Prédiction : Ŷ(x) = μ(x) + Cov{Y(x), (Y(x₁),...,Y(xₙ))} · C⁻¹(y − m)
- **Pas de GEK, pas de GEPCK, pas de matrice augmentée**

### FunctionalChaosAlgorithm — `metamodel.py:3975`
- PCE avec LARS via `LeastSquaresMetaModelSelectionFactory(LARS(), CorrectedLeaveOneOut())`
- Sorties : `getReducedBasis()`, `getIndices()`, `getCoefficients()`

### LARS — `func.py:18152`
- **Peut être instancié en standalone** : `ot.LARS()`
- Méthode `.build(x, y, psi, indices)` → BasisSequence
- Son usage naturel reste dans FunctionalChaosAlgorithm

### Conséquence pour GEPCK
GLS de KrigingAlgorithm = C++, non modifiable.
→ GEPCK doit être implémenté en **Python pur (numpy/scipy)**, en utilisant OT uniquement pour les polynômes et le noyau.

---

## GEPCK — Mathématiques (Zuhal et al. 2021)

### Notations
- n : points d'entraînement, m : dimension entrée, xᵢ_l : composante l de xᵢ
- Ψₖ(x) : k-ème polynôme multivarié de Legendre, P : nombre total de bases
- k(xi, xj) = σ² · exp(−Σₗ (xᵢₗ − xⱼₗ)² / (2θₗ²)) — noyau Squared Exponential

### Vecteur ỹ (taille n(m+1) × 1)
```
ỹ = [y(x₁),...,y(xₙ),  ∂y/∂x₁(x₁),...,∂y/∂x₁(xₙ),  ...,  ∂y/∂xₘ(x₁),...,∂y/∂xₘ(xₙ)]
```
Index : ỹ[i] = y(xᵢ) ; ỹ[n + l*n + i] = ∂y/∂xₗ(xᵢ)

### Matrice R̃ (taille n(m+1) × n(m+1)) — Squared Exponential
Notation : Δxₗ = xᵢₗ − xⱼₗ, k = k(xᵢ, xⱼ)

| Bloc | Indices | Formule |
|------|---------|---------|
| valeur-valeur | [i, j] | k(xᵢ, xⱼ) |
| gradient-valeur (∂/∂xᵢₗ) | [n+l*n+i, j] | −Δxₗ/θₗ² · k |
| valeur-gradient (∂/∂xⱼₗ) | [i, n+l*n+j] | +Δxₗ/θₗ² · k |
| grad-grad même dim | [n+l*n+i, n+l*n+j] | k/θₗ² · (1 − Δxₗ²/θₗ²) |
| grad-grad dims croisées l≠p | [n+l*n+i, n+p*n+j] | −Δxₗ·Δxₚ/(θₗ²θₚ²) · k |

### GEK vs GEPCK — Différence unique : la matrice F̃ (taille n(m+1) × P)

| | GEK | GEPCK |
|--|-----|-------|
| Lignes valeurs F̃[i,k] | Ψₖ(xᵢ) | Ψₖ(xᵢ) |
| Lignes gradient F̃[n+l*n+i,k] | **0** | **∂Ψₖ/∂xₗ(xᵢ)** |
| β dépend de | y seulement | y ET ∇y |
| ∇μ(x) | 0 | ∇Ψ(x)β |

**GEPCK = GEK + remplacement des 0 par ∇Ψₖ dans F̃. C'est le seul changement.**

### Vecteur r̃(x) pour la prédiction (taille n(m+1) × 1)
```
r̃[j] = k(x, xⱼ)
r̃[n+l*n+j] = +(xₗ − xⱼₗ)/θₗ² · k(x,xⱼ)   ← signe + (dérivée par rapport au 2nd argument)
```

### Formules
- GLS : β = (F̃ᵀR̃⁻¹F̃)⁻¹F̃ᵀR̃⁻¹ỹ
- Prédiction : ŷ(x) = Ψ(x)ᵀβ + r̃(x)ᵀR̃⁻¹(ỹ − F̃β)

---

## Polynômes de Legendre (Table 1 de l'article)

| n | ψₙ(x) | ψₙ'(x) |
|---|--------|---------|
| 0 | 1 | 0 |
| 1 | x | 1 |
| 2 | (3x² − 1)/2 | 3x |
| 3 | (5x³ − 3x)/2 | (15x² − 3)/2 |
| 4 | (35x⁴ − 30x² + 3)/8 | (35x³ − 15x)/2 |

Multivarié : Ψₐ(x) = ∏ₗ ψₐₗ(xₗ)
Dérivée : ∂Ψₐ/∂xₗ(x) = ψ'ₐₗ(xₗ) · ∏_{j≠l} ψₐⱼ(xⱼ)

---

## GEPCK-LAR — Ce que fait l'article (≠ LARS ordinaire)

**β est TOUJOURS calculé par GLS**, que ce soit GEPCK-GLS ou GEPCK-LAR. LAR = sélection de bases uniquement, jamais de substitut à GLS.

L'article **ne fait pas** LARS-sélection puis GLS.

À chaque étape de la sélection :
1. On refit un **modèle de kriging complet** (GEPCK-GLS) avec la base courante
2. Le critère de sélection est le **eLOO kriging** (leave-one-out cross-validation kriging), pas la corrélation résiduelle LARS standard
3. La base qui minimise eLOO kriging est ajoutée
4. On s'arrête quand eLOO **augmente 3 fois de suite**
5. Optimisation hyperparamètres à chaque step : algorithme simplifié GA+BFGS

→ Options A (LARS Python pur) et B (FunctionalChaosAlgorithm) ≠ article car elles utilisent la corrélation LARS standard (ou LOO PCE), pas eLOO kriging.
→ GEPCK-LAR est plus coûteux que GEPCK-GLS. Implémenter GEPCK-GLS d'abord.

---

## Approche article vs idée utilisatrice

**Idée utilisatrice** (séquentielle) : PCE sur HF(x) → GEK sur résidu (HF − PCE)

**Article GEPCK** (jointe) : gradients enrichissent **simultanément** F̃ ET R̃ — β estimé en une seule fois sur [y ; ∇y]

Ces deux approches ne sont **pas équivalentes**. L'article estime trend et GP ensemble, l'approche séquentielle fait PCE d'abord (ignore la structure GP) puis GEK sur le résidu.

---

## Plan d'implémentation GEPCK

### Fichiers
| Fichier | Action |
|---------|--------|
| `C:\python3\lib\site-packages\openturns\gepck.py` | Créer — classe GEPCKAlgorithm |
| `C:\python3\lib\site-packages\openturns\__init__.py` | Modifier — `from .gepck import GEPCKAlgorithm` |

### Étape 1 — GEPCK-GLS
`GEPCKAlgorithm(X, y, grad_y, covarianceModel, polynomial_order)` → construire ỹ, R̃, F̃ → GLS numpy → optimiser θ,σ (scipy) → prédire

Algorithme : scanner p = 0, 1, ..., pmax. Pour chaque p, construire modèle GEPCK avec toutes les bases A^{m,p}, calculer eLOO. **Arrêt quand card(A^{m,p+1}) > n(m+1)** (système sous-déterminé). Sélectionner p qui minimise eLOO.

### eLOO (Leave-One-Out cross-validation kriging)
- `eLOO = Σᵢ εᵢ² / (n + nm)`  où la somme porte sur les n(m+1) observations augmentées
- Matrice B (taille n(m+1)+P × n(m+1)+P) : `B = [[σ²R̃, F̃], [F̃ᵀ, 0]]`
- LOOCV virtuel : `ŷ⁻ⁱ(xᵢ) = −Σⱼ Bᵢⱼ/Bᵢᵢ · ỹⱼ + ỹᵢ`  (formule analytique, pas de refit)
- εᵢ = ỹᵢ − ŷ⁻ⁱ(xᵢ)

### Étape 2 — GEPCK-LAR
Sélection itérative avec critère eLOO kriging à chaque step (après validation GEPCK-GLS).

### Ressources OT
- `ot.OrthogonalProductPolynomialFactory([ot.LegendreFactory()]*m)` : génère les Ψₖ
- `ot.LinearEnumerateFunction(m)` : multi-indices α jusqu'à ordre p
- `ot.SquaredExponential([theta]*m, [sigma])` : noyau Gaussien

### Vérification
1. Fonction 1D avec gradient connu : interpolation exacte aux points d'entraînement
2. GEPCK-GLS vs `ot.KrigingAlgorithm` classique sur même dataset
3. eLOO(GEPCK-LAR) ≤ eLOO(GEPCK-GLS)
4. Tester sur `AC_pure_flexion.py` (remplacer `build_metamodel_KRG`)
