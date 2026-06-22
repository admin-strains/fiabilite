# Migration de Semia vers `admin-strains/fiabilite` branche `flexion`

## Contexte

Semia bosse actuellement dans `C:\_workingDir\_SF\test flexion\`, qui est **son propre clone Git** (le dossier appartient à `semia.frikha`). Ce clone pointe a priori sur son repo perso GitHub `semiastrains/AC-pure-flexion`.

Le travail de fiabilité va désormais vivre sur le repo officiel Strains : `admin-strains/fiabilite`. Mohamad y a déjà créé une branche `moulin_blanc` pour son propre travail. **Semia doit maintenant créer sa branche `flexion`** et y migrer ses modifs locales.

**Modifs locales actuelles de Semia** (vérifiées le 11/06 à 14:12) : 7 fichiers `.py` édités après son dernier push, dont notamment **`AC3_pure_flexion.py` ligne 95 : `tol_all_modes = 0.9`** (modif fonctionnelle à préserver absolument).

**Objectif** :
- Faire passer son travail vers la branche `flexion` du repo `admin-strains/fiabilite`
- **Sans rien perdre** (les 7 modifs locales doivent migrer intactes)
- **Sans changer son dossier de travail** (elle garde `C:\_workingDir\_SF\test flexion\` comme working dir)

**Toutes les commandes doivent être lancées par Semia sous son compte `semia.frikha`** (Git refuse sinon pour cause de "dubious ownership" du repo).

---

## Étape 0 — Diagnostic initial

Avant de toucher à quoi que ce soit, comprendre où elle en est :

```powershell
cd "C:\_workingDir\_SF\test flexion"
git status                    # voir ses modifs non commitées
git remote -v                 # voir son remote actuel (probablement origin = semiastrains)
git branch -a                 # voir ses branches locales et distantes
git log --oneline -5          # voir ses 5 derniers commits locaux
```

**Vérifications attendues** :
- `git status` doit montrer **plusieurs fichiers `modified:`** (les 7 fichiers `.py` qu'elle a édités) :
  - `AC2_pure_flexion.py`, `AC3_pure_flexion.py`
  - `branche1.py`, `branche2.py`, `branche4.py`
  - `launcher2.py`, `launcher3.py`
- `git remote -v` doit afficher au moins un remote (probablement `origin → https://github.com/semiastrains/AC-pure-flexion.git`)
- `git log` doit afficher un historique d'au moins 43 commits

**Si ça plante avec "dubious ownership"** : c'est qu'elle a lancé sous le mauvais compte. Confirmer qu'elle est bien connectée sous `semia.frikha`.

**Si l'output diffère significativement** de l'attendu : noter et transmettre à Mohamad avant de continuer.

---

## Étape 1 — Sauvegarde préalable sur son repo perso (recommandé)

Pour avoir un backup de ses modifs locales avant la migration :

```powershell
git add AC2_pure_flexion.py AC3_pure_flexion.py branche1.py branche2.py branche4.py launcher2.py launcher3.py
git commit -m "Backup avant migration vers admin-strains (tol_all_modes=0.9, etc.)"
git push origin HEAD
```

**Vérifications** :
- `git status` après le commit doit afficher : `nothing to commit, working tree clean`
- Le push doit afficher quelque chose comme :
  ```
  To https://github.com/semiastrains/AC-pure-flexion.git
     12e94a4..abc1234  <sa-branche> -> <sa-branche>
  ```

**Note** : si elle ne veut PAS pousser sur son perso, sauter cette étape — mais alors elle perd le backup. À sa discrétion.

---

## Étape 2 — Ajouter `admin-strains` comme remote secondaire

On ne touche pas à `origin` (qui reste son perso), on ajoute un 2e remote nommé `strains` :

```powershell
git remote add strains https://github.com/admin-strains/fiabilite.git
git fetch strains
```

**Vérifications** :
- `git remote -v` doit maintenant afficher **2 remotes** :
  ```
  origin   https://github.com/semiastrains/AC-pure-flexion.git (fetch)
  origin   https://github.com/semiastrains/AC-pure-flexion.git (push)
  strains  https://github.com/admin-strains/fiabilite.git (fetch)
  strains  https://github.com/admin-strains/fiabilite.git (push)
  ```
- `git fetch strains` doit afficher quelque chose comme :
  ```
  From https://github.com/admin-strains/fiabilite
   * [new branch]      fiabilite     -> strains/fiabilite
   * [new branch]      moulin_blanc  -> strains/moulin_blanc
  ```

**Si le fetch plante** : c'est probablement un problème d'accès au repo Strains. Vérifier qu'elle est bien autorisée sur `admin-strains/fiabilite` côté GitHub (accepter l'invitation s'il y en a une en attente).

---

## Étape 3 — Vérifier la compatibilité historique entre les deux repos

Étape critique. S'assurer que son clone perso et `admin-strains/fiabilite` partagent une base commune (sinon le push sera rejeté).

```powershell
git log strains/fiabilite -5      # les 5 derniers commits côté Strains
git log HEAD -5                   # les 5 derniers commits côté local (perso)
```

**3 cas possibles** :

### ✅ Cas A — Les SHA des derniers commits matchent (idéal)
Si elle voit les mêmes SHA (`12e94a4`, `912d6f4`, etc.) des deux côtés → historiques compatibles, passer à l'étape 4.

### ✅ Cas B — Les SHA matchent jusqu'à un certain point, puis divergent
Genre : les 5 derniers commits de `strains/fiabilite` sont identiques aux commits 3 à 7 de `HEAD` (parce qu'elle a des commits perso après le mirror). C'est aussi OK — passer à l'étape 4.

### ❌ Cas C — Aucun SHA ne matche entre les 2
C'est le cas problématique : son perso et Strains ont des historiques totalement disjoints. **Arrêter, prévenir Mohamad** — on devra alors faire un `cherry-pick` plutôt que créer la branche directement à partir de `strains/fiabilite`.

---

## Étape 4 — Créer la branche `flexion` et y appliquer ses modifs

Si étape 3 = cas A ou B, on continue.

```powershell
# Créer la branche flexion à partir de strains/fiabilite (le dernier commit Strains)
git checkout -b flexion strains/fiabilite

# Vérifier qu'on est bien sur flexion
git status
```

→ Doit afficher `On branch flexion` et `nothing to commit, working tree clean`.

**⚠️ Attention** : à ce stade, **les fichiers sur disque sont remis à la version Strains** (sans ses 7 modifs). C'est NORMAL parce que `checkout -b flexion strains/fiabilite` matérialise les fichiers du commit Strains.

**MAIS ses 7 modifs locales ne sont pas perdues** : elle vient de les commiter à l'étape 1 sur son perso. Maintenant elle doit re-appliquer ces modifs sur la branche `flexion` via un cherry-pick.

### Re-appliquer le commit de backup sur flexion

```powershell
# Récupérer le SHA de son commit de backup créé à l'étape 1
git log origin/<sa-branche-perso> -1 --format="%H"
```
→ copier le SHA affiché (ex: `abc1234...`).

⚠️ **Remplacer `<sa-branche-perso>` par le nom réel de sa branche perso** (probablement `fiabilite`, à confirmer avec `git branch` à l'étape 0).

```powershell
# Cherry-pick ce commit sur flexion
git cherry-pick <SHA-copié>
```

**Vérifications après le cherry-pick** :
```powershell
git log --oneline -5
# → doit montrer son commit de backup en HEAD, puis les commits historiques de Strains

git diff strains/fiabilite HEAD
# → doit montrer les 7 fichiers modifiés (notamment AC3 ligne 95 : tol_all_modes 2.0 → 0.9)

git status
# → "On branch flexion", "nothing to commit"
```

Si elle voit son commit + les diffs corrects → ✅ tout est en place.

**Si le cherry-pick génère un conflit** (peu probable mais possible) : 
```powershell
git status                  # voir les fichiers en conflit
# résoudre manuellement les conflits dans chaque fichier
git add <fichier-résolu>
git cherry-pick --continue
```

---

## Étape 5 — Push sa branche `flexion` sur `admin-strains/fiabilite`

```powershell
git push -u strains flexion
```

**Vérifications attendues** :
- Sortie :
  ```
  To https://github.com/admin-strains/fiabilite.git
   * [new branch]      flexion -> flexion
  branch 'flexion' set up to track 'strains/flexion'.
  ```
- Aller sur `https://github.com/admin-strains/fiabilite/tree/flexion` dans le navigateur — vérifier que sa branche est bien là, avec son dernier commit visible.

**Si le push est rejeté** :
- Erreur `! [rejected] ... fetch first` → quelqu'un a poussé entre temps. Faire `git fetch strains` puis `git rebase strains/fiabilite` et re-push.
- Erreur `dubious ownership` → revérifier qu'elle est bien sous son compte `semia.frikha`.

---

## Étape 6 — Vérifications finales

```powershell
# La branche locale tracke bien la branche Strains ?
git branch -vv
```
→ doit afficher quelque chose comme :
```
* flexion   abc1234 [strains/flexion] Backup avant migration vers admin-strains (...)
  <ancienne-branche-perso> 12e94a4 [origin/<ancienne>] ...
```

```powershell
# Son historique est correct ?
git log --oneline -10
```
→ doit afficher son commit + les 43+ commits historiques.

```powershell
# Status clean ?
git status
```
→ `On branch flexion. Your branch is up to date with 'strains/flexion'. nothing to commit`

---

## Comprendre les branches et les remotes après la migration

### Son clone après migration

Le **dossier physique** : `C:\_workingDir\_SF\test flexion\` (le même qu'avant)

Les **branches locales** dans son clone :
```
* flexion                  ← active (= ce qui s'affiche dans le dossier)
  <son-ancienne-branche-perso>   ← son ancienne branche perso, toujours là
```

Les **remotes** :
```
origin   → github.com/semiastrains/AC-pure-flexion       (son perso)
  └── <ses branches perso>

strains  → github.com/admin-strains/fiabilite            (Strains)
  └── fiabilite (commune)
  └── moulin_blanc (Mohamad)
  └── flexion (à elle)
```

### Le dossier reflète UNE branche à la fois

```powershell
git checkout flexion       # → dossier affiche les fichiers de flexion (avec ses modifs)
git checkout <perso>       # → dossier affiche les fichiers de sa branche perso (sans ses modifs récentes)
```

À chaque `checkout`, les fichiers se transforment automatiquement. Rien n'est perdu — Git reconstruit les fichiers depuis l'historique.

### Push automatique selon la branche active

Grâce au `-u strains flexion` de l'étape 5 :

| Branche active | `git push` va vers |
|---|---|
| `flexion` | `strains/flexion` (côté Strains) |
| `<son-ancienne-branche>` | `origin/<son-ancienne-branche>` (côté perso) |

→ Plus de risque de mélanger les deux. Chaque branche pousse au bon endroit automatiquement.

---

## Workflow quotidien après migration

Elle continue à bosser dans `C:\_workingDir\_SF\test flexion\` exactement comme avant. Cycle typique :

```powershell
# 1. Vérifier qu'on est sur flexion (devrait être automatique après le checkout de l'étape 4)
git branch
# → * flexion

# 2. Modifier les fichiers normalement (VS Code, éditeur)

# 3. Sauver ses modifs
git add <fichiers modifiés>
git commit -m "..."
git push                        # → part automatiquement sur strains/flexion
```

### Récupérer le travail de Mohamad (`moulin_blanc`)

Si elle veut intégrer son travail :
```powershell
git fetch strains
git merge strains/moulin_blanc          # ou git rebase strains/moulin_blanc
```

### Récupérer les évolutions de `fiabilite` (la branche commune)

Si quelqu'un a fait avancer `fiabilite` :
```powershell
git fetch strains
git rebase strains/fiabilite            # rejoue ses commits par-dessus la nouvelle baseline
```

### Quand elle voudra que ses modifs entrent dans `fiabilite` (la branche commune)

Ouvrir une **Pull Request** sur GitHub : `https://github.com/admin-strains/fiabilite/pull/new/flexion`

→ Discussion + merge via UI GitHub. À ce moment-là seulement ses modifs deviendront visibles pour tout le monde.

---

## Récap commandes (à copier-coller dans l'ordre)

```powershell
# === Étape 0 : diagnostic ===
cd "C:\_workingDir\_SF\test flexion"
git status
git remote -v
git branch -a
git log --oneline -5

# === Étape 1 : sauvegarde perso ===
git add AC2_pure_flexion.py AC3_pure_flexion.py branche1.py branche2.py branche4.py launcher2.py launcher3.py
git commit -m "Backup avant migration vers admin-strains (tol_all_modes=0.9, etc.)"
git push origin HEAD

# === Étape 2 : ajout remote Strains ===
git remote add strains https://github.com/admin-strains/fiabilite.git
git fetch strains

# === Étape 3 : vérifier compatibilité historique (s'arrêter pour inspecter) ===
git log strains/fiabilite -5
git log HEAD -5

# === Étape 4 : créer flexion + cherry-pick le backup ===
git checkout -b flexion strains/fiabilite
git log origin/<sa-branche-perso> -1 --format="%H"
git cherry-pick <SHA>

# === Étape 5 : push sur strains ===
git push -u strains flexion

# === Étape 6 : vérif finales ===
git branch -vv
git log --oneline -10
git status
```

---

## Règles de sécurité

1. **Toujours sous le compte `semia.frikha`** (sinon "dubious ownership")
2. **Jamais de `git reset --hard`** sans comprendre ce qu'on fait (efface les modifs locales)
3. **Jamais de `git push --force`** sur `flexion` après le 1er push (sauf cas exceptionnel)
4. **Si une étape ne donne pas le résultat attendu** : s'arrêter immédiatement, transmettre l'erreur à Mohamad, ne pas improviser

---

## Glossaire express

| Terme | Sens |
|---|---|
| `origin` | Le 1er remote du clone (par défaut quand on clone) = son perso `semiastrains/AC-pure-flexion` |
| `strains` | Le 2e remote ajouté à l'étape 2 = `admin-strains/fiabilite` |
| `flexion` | La nouvelle branche perso de Semia sur `admin-strains` |
| `fiabilite` (côté Strains) | La branche commune partagée — ne pas push directement dessus |
| `moulin_blanc` | La branche perso de Mohamad sur `admin-strains` |
| Cherry-pick | Rejouer un commit existant (de la branche A) sur la branche B |
| Tracking | Lien automatique entre branche locale et branche distante (`-u` lors du 1er push) |
