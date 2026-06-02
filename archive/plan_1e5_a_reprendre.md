# Plan — Récupérer les résultats FORM malgré l'erreur de tolérance

## Contexte

AbdoRackwitz converge vers un design point avec g(u*)≈1.3e-05, mais FORM lève
`RuntimeError: design point image = 1.3e-05 > limit state tolerance 1e-05` (seuil C++ fixé à 1e-05).
Le résultat semble exploitable. On cherche à y accéder malgré l'erreur.

Sources consultées :
- https://openturns.github.io/openturns/1.26/user_manual/_generated/openturns.AbdoRackwitz.html
- https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.FORM.html

---

## Option C — setCheckStatus(False) [NOUVEAU — À TESTER EN PREMIER]

**Découverte** : AbdoRackwitz expose `setCheckStatus()` / `getCheckStatus()`.
Si ce flag contrôle la validation de convergence finale, le désactiver pourrait faire
tourner le solver jusqu'à n_max_FORM sans jamais lever l'erreur de tolérance.

```python
solver = ot.AbdoRackwitz()
solver.setMaximumIterationNumber(n_max_FORM)
solver.setCheckStatus(False)   # désactive la vérification du statut de convergence
solver.setStartingPoint([0.0] * n_var)
algo = ot.FORM(solver, event)
algo.run()   # ne devrait plus lever si checkStatus=False
result = algo.getResult()
```

**Avantage** : 1 ligne, propre, pas de try/except.
**Risque** : comportement exact de setCheckStatus non documenté — peut ne pas affecter le check FORM.
**À vérifier** : si c'est un flag interne du solver ou si FORM lit ce flag pour sa validation finale.

---

## Option A — try/except + algo.getResult()

**Base documentaire** : la doc FORM dit explicitement :
> "Evaluate the failure probability and **create a FORMResult**, the structure result which
> is **accessible with the method getResult()**."
→ La FORMResult est créée avant la levée de l'exception, getResult() devrait fonctionner.

```python
try:
    algo.run()
except RuntimeError as e:
    print(f"[FORM] RuntimeError intercepté : {e}")
result = algo.getResult()   # valide même après exception selon doc
```

**Avantage** : simple, 2 lignes, confirmé par la doc.
**Risque** : "create a FORMResult" pourrait être interrompu avant la fin si l'exception est levée
au milieu — non confirmé à 100%.

---

## Option B — solver.getResult() (fallback ultime)

**Base documentaire** : AbdoRackwitz expose `getResult()` → OptimizationResult.
Indépendant de FORM, stocké dans le solver.

```python
try:
    algo.run()
except RuntimeError:
    try:
        result = algo.getResult()
    except Exception:
        opt_result = solver.getResult()
        u_star = list(opt_result.getOptimalPoint())
        beta = float(np.linalg.norm(u_star))
        alpha = np.array(u_star) / beta
        print(f"β = {beta:.4f}, u* = {[round(v,4) for v in u_star]}")
        print(f"Importances (alpha²) : fc={alpha[0]**2:.4f}, fy={alpha[1]**2:.4f}")
```

**Avantage** : contourne tout, accès direct à u*.
**Risque** : importances FORM non disponibles (seulement direction cosines).

---

## Ordre de test recommandé

1. **Option C** : `solver.setCheckStatus(False)` — 1 ligne, le plus propre
2. **Option A** : try/except + `algo.getResult()` — confirmé par doc, 2 lignes
3. **Option B** : fallback `solver.getResult()` — si A échoue

**Fichier à modifier** : `C:\_workingDir\_SF\test flexion\AC_pure_flexion.py`, bloc FORM (~ligne 546)
