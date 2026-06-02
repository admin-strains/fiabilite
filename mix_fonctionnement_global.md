# mix_fonctionnement_global.md
# Fichier de contexte cumulatif — à maintenir après chaque modification significative
# But : éviter de tout relire après un autocompactage

---

## PROJET : Fiabilité flexion pure béton — AC_pure_flexion.py

**Dossier principal :** `C:\_workingDir\_SF\test flexion\`
**Script principal :** `AC_pure_flexion.py`
**Modèle mécanique :** STRAINS — fichiers `C:\workspace\storage\admin\SF\test_pure_flexion.ds\{dsCad.txt, dsLoad.txt}`

---

## BIBLIOTHÈQUE PCK (Python clone UQLab)

### Fichiers de la librairie

| Fichier | Rôle | Statut |
|---|---|---|
| `branche1.py` | Point d'entrée : `fit_pck`, `predict_pck`, `generate_doe` (temporaire) | DONE — 51/51 tests |
| `branche2.py` | `uq_PCK_initialize` — parse options | DONE |
| `branche3.py` | `uq_PCK_calculate_coefficients` — cœur hybride PCE+Kriging | DONE — 75/75 tests |
| `branche4.py` | `uq_PCK_eval` — prédiction | DONE — 41/41 tests |
| `branche5.py` | `uq_eval_Kernel`, polynômes | DONE |

### API principale

```python
from branche1 import fit_pck, predict_pck, generate_doe

# Marginales
marginals = [{'Type': 'Gaussian', 'Parameters': [0.0, 1.0]}] * 2  # espace standard
copula    = {'Type': 'Independent', 'Parameters': np.eye(2)}

# DOE (temporaire, à supprimer plus tard)
X_doe = generate_doe(N, marginals, method='lhs', seed=42)
Y_doe = np.array([g(u[0], u[1]) for u in X_doe])

# Fit
opts = {'Mode': 'sequential', 'PCE': {'Degree': [1,2,3,4], 'Method': 'LARS'}}
fm = fit_pck(X_doe, Y_doe, opts, marginals, copula)

# Lire résultats
loo = fm['Error'][0]['LOO']         # LOO error
n_poly = fm['NumberOfPoly'][0]      # nb polynômes retenus
# fm['pck_config']['TrendMethod']   # 'pce' ou 'user'

# Prédiction
YMu, YVar = predict_pck(fm, X_test, return_var=True)  # YMu, YVar : (N_test, Nout)
```

---

## STRUCTURE D'AC_pure_flexion.py

### Variables / paramètres globaux (définis dans `if __name__ == '__main__':`)

| Variable | Valeur | Rôle |
|---|---|---|
| `modele` | `'GEPCK'` (actif) | sélecteur métamodèle |
| `n_var` | 2 | fc, fy |
| `fcm`, `cov_fc` | 48, 0.12 | béton — LogNormal |
| `fym`, `SIGMA` | 550, √(19²+22²+8²) ≈ 30 MPa | acier — Normal |
| `u1_min/max` | ±10 | espace standard u1 |
| `u2_min/max` | ±10 | espace standard u2 |
| `n_grid` | 300 | grille de visualisation |

### Flags `do_*`

```python
do_KRG   = True if modele == 'KRG'   else False
do_GEK   = True if modele == 'GEK'   else False
do_HF    = True if modele == 'HF'    else False
do_PCKRG = True if modele == 'PCKRG' else False
do_GEPCK = True if modele == 'GEPCK' else False
do_PCK_B1 = True if modele == 'PCK_B1' else False   # AJOUTÉ (Mod 2)
```

### Classe `flexion_claude` (lignes ~606–682)

- `__init__`: lit dsCad.txt + dsLoad.txt, construit la transformation isoproba OT
- `g(u1, u2)` : fonction de performance en **espace standard U ~ N(0,1)**
  - Appelle `self.T_inv(ot.Point([u1, u2]))` → (fc, fy) espace physique
  - 2 branches : aciers plastifiés / non plastifiés
  - Retourne scalaire (>0 = sûr, <0 = défaillance)
- `u2p_LS(u1)` : u2 sur la courbe g=0 (branche plastifiée)

### Fonctions de visualisation

| Fonction | Lignes | Description |
|---|---|---|
| `print_visu_ana()` | 684–714 | courbe g=0 analytique paramétrée par u2p_LS |
| `print_visu_pck_b1(...)` | **AJOUTÉ** après 714 | g=0 analytique + PCK (branche1) |
| `print_visu_EFF(...)` | 1250+ | carte EFF |
| `print_visu(...)` | 1379+ | figure principale post-FORM |

### Bloc d'exécution (lignes ~1613–1671)

```
update_degree(n0)
[if print_3D: ...]
[if print_grad_sp: ...]
[if do_PCK_B1: print_visu_pck_b1(...); sys.exit(0)]   ← AJOUTÉ (Mod 4)
g_ot, sigma_func, xt, yt, all_grad = init_g_ot(...)
if do_EFF: run_EFF(...)
event, ... = init_FORM(...)
→ FORM_all_modes → print_results → print_visu
```

---

## TÂCHE EN COURS : Intégration PCK_B1 dans AC_pure_flexion.py

### Objectif
Afficher la courbe g=0 du PCK (branche1) superposée à la courbe analytique.
**Pas encore de FORM** — uniquement visualisation.

### 4 modifications (état : EN COURS)

| Mod | Description | Statut |
|---|---|---|
| 1 | Import branche1 après `from math import comb` | ✅ FAIT |
| 2 | Ajout `do_PCK_B1 = ...` après `do_GEPCK = ...` | ✅ FAIT |
| 3 | Nouvelle fonction `print_visu_pck_b1` après `print_visu_ana` | ✅ FAIT |
| 4 | Appel `if do_PCK_B1: print_visu_pck_b1(...)` avant `init_g_ot` | ✅ FAIT |

### Choix architecturaux

- **Espace U** : le PCK est entraîné directement en espace standard N(0,1) × N(0,1)
  → marginales `{'Type': 'Gaussian', 'Parameters': [0.0, 1.0]}`
  → Hermite polynomials, cohérent avec `init_FORM`
- **DOE** : N=40 LHS, seed=42, espace U
- **Grille** : 150×150 (au lieu de 300×300 pour limiter les appels OT)
- **LOO attendu** : ~0.01–0.10 pour N=40

### Pour tester

Mettre `modele = 'PCK_B1'` dans AC_pure_flexion.py et lancer le script.
Résultat attendu :
- Print : `[PCK_B1] N_doe=40  LOO=...  n_poly=...`
- Figure : 2 courbes g=0 (noir=analytique, bleu tiret=PCK) + points DOE rouges

---

## RÉSULTATS IMPORTANTS

| Test | Résultat |
|---|---|
| B1 : 51/51 PASS | `test_branche1.py` — toutes les sections |
| B3 : 75/75 PASS | |
| B4 : 41/41 PASS | |
| Demo demo_pck.py | f(x)=sin(3πx)*exp(-x²), N=20, LOO=1.04e+00 (fonction trop complexe pour N=20) |
| **PCK_B1 sur g analytique flexion** | N=40, LOO=2.978e-03, n_poly=7 — **excellent** (27/05) |

---

## PROTOCOLE DE LANCEMENT (section 6 du global_resume)

### Avant tout lancement

1. **Toujours lire le bloc OPTIONS d'AC_pure_flexion.py avant un run** (ne jamais interpréter sans)
2. **Obtenir la date/heure** : `date +%d%m_%H%M` dans un bash SÉPARÉ avant de lancer

### Commande de lancement

```bash
cd "C:\_workingDir\_SF\test flexion"
python launcher.py > "output/output_$(date +%d%m_%H%M).txt"
```

- **Toujours lancer via launcher.py** (configure les DLL STRAINS)
- **Ne JAMAIS lancer AC_pure_flexion.py directement** (pas de STRAINS)
- **L'environnement conda habituel est requis** — ne fonctionne pas depuis le terminal Claude Code

### Surveillance du run (commande obligatoire)

Pour les runs normaux (recherche de `beta =`) :
```bash
until grep -q "beta =" "output/output_DDMM_HHMM.txt" 2>/dev/null; do sleep 15; done && grep "beta\|Pf\|u\*\|Imp\." "output/output_DDMM_HHMM.txt"
```

Pour un run PCK_B1 (recherche de `[PCK_B1]`) :
```bash
until grep -q "\[PCK_B1\]" "output/output_DDMM_HHMM.txt" 2>/dev/null; do sleep 5; done && grep "PCK_B1" "output/output_DDMM_HHMM.txt"
```

### Après le run

- **Mettre à jour ce fichier mix** + le `global_resume_session_2404.md` immédiatement
- Ne pas attendre que l'utilisatrice demande "tu as les résultats ?"

---

## RÈGLES DE TRAVAIL

1. **Toujours lire avant d'éditer** — ne jamais supposer une valeur
2. **Ne jamais supprimer de fichiers** sans confirmation explicite
3. **Mettre à jour ce fichier** après chaque modification significative
4. **LOO path** : `fm['Error'][0]['LOO']` (pas `fm['LOO']`)
5. **TrendMethod path** : `fm['pck_config']['TrendMethod']` (pas `fm['TrendMethod']`)
