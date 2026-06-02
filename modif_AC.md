# Modifications à appliquer sur InitSolver et AC pour corriger la divergence du solveur kine

## Contexte

Comparaison entre `Yield_analysis0_kine.kwargs` (généré par Digital Structure, calcul converge : Primal≈1.589, 29 iter)
et les kwargs passés par AC.py + InitSolver.py (diverge : Primal~10^12–10^56, 3–300 iter).

---

## Modification 1 — InitSolver.py : IPARM0[21] = 3 au lieu de 4

**Fichier** : `C:\_workingDir\_exportRebar\InitSolver.py` (ou copie locale)

Dans `cinematic_params` :
```python
# AVANT
{"value": 4, "table": "IPARM0", "indices": [21]},   # PT INT (1 = MKL PARDISO, 3 = MUMPS)

# APRÈS
{"value": 3, "table": "IPARM0", "indices": [21]},   # PT INT (1 = MKL PARDISO, 3 = MUMPS)
```

**Effet observé** : le solveur passe de 3 itérations à 97–300 itérations. Sans ce fix, le point intérieur s'arrête immédiatement (valeur 4 = type solveur non documenté).

---

## Modification 2 — AC.py : supprimer l'initialisation des cônes

**Fichier** : `C:\_workingDir\_exportRebar\AC.py` (ou copie locale)

Supprimer les 4 lignes suivantes (absentes du kwargs DS) :
```python
kwargs["X0ConeC"] = 1.0e-2
kwargs["X0ConeT"] = 1.0e-2
kwargs["S0ConeC"] = 1.0e2
kwargs["S0ConeT"] = 1.0e2
```

**Effet observé** : Dual passe à ~0.667 (converge partiellement). Primal toujours divergé (~10^21).

---

## Modification 3 — À tester : DPARM0[20] cone border coef = 0.98

Dans `cinematic_params` de InitSolver :
```python
# AVANT
{"value": 0.90, "table": "DPARM0", "indices": [20]},  # Cone border coef

# APRÈS (valeur DS)
{"value": 0.98, "table": "DPARM0", "indices": [20]},  # Cone border coef
```

DS utilise 0.98. Non encore testé.

---

## Tableau de comparaison DS vs AC.py

| Paramètre | DS (`kine.kwargs`) | AC.py / InitSolver.py |
|---|---|---|
| IPARM0[21] PT INT | 3 (MUMPS) | 4 (invalide) |
| IPARM0[23] Max iter | 1000 | 300 |
| DPARM0[20] Cone border coef | 0.98 | 0.90 |
| DPARM0[21] Tol abs Alpha | 0.001 | 1e-4 |
| DPARM0[22] Tol rel Pobj-Dobj | 0.01 | 1e-4 |
| DPARM0[23-26] Tol rel Res | 1e-10 | 1e-12 |
| X0ConeC/T, S0ConeC/T | absent | présent (1e-2 / 1e2) |
| scaling | absent (cinematic_scaling="1") | scaling=1 |
| static_params | absent | présent |
| FullLorentz, LorentzToSdp... | absent | présent |
| hpc_params (4 threads OMP) | présent | absent |
