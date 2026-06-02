# Plan : PC-Kriging en Python (fidèle à UQLab)

## Contexte

PC-Kriging (Polynomial Chaos Kriging) est le meilleur surrogate model identifié dans le benchmark de Moustapha et al. (2022). UQLab implémente ce modèle en MATLAB. L'objectif est de le recoder en Python, testé sur une fonction simple (ex: x*sin(x)).

Le modèle combine :
- **Trend = PCE** : base polynomiale orthogonale (Hermite/Legendre) calibrée par LARS
- **Résidu = Kriging** : processus Gaussien sur les résidus du trend, avec noyau Matérn 5/2

## Formules mathématiques clés (extraites du source UQLab)

### Corrélation (ellipsoïdal, défaut)
```
h = seuclidean(x1, x2, θ)  → distance standardisée avec θ = vecteur longueurs de corrélation
K_matern52(h) = (1 + √5·h + 5/3·h²) · exp(-√5·h)
R = K(X_train, X_train) + nugget·I
r0 = K(X_test, X_train)  (nugget = 0 ici)
```

### Prédiction (BLUP)
```
β = G⁻¹ · Q1ᵀ · Ỹ          (QR-based, avec L=chol(R), Ỹ=L⁻ᵀY, F̃=L⁻ᵀF, [Q1,G]=qr(F̃))
Ŷ(x0) = f0·β + r0 · R⁻¹ · (Y - F·β)
D1 = diag(r0 · R⁻¹ · r0ᵀ)
u0 = Fᵀ·R⁻¹·r0ᵀ - f0ᵀ
D2 = diag(u0ᵀ · (Fᵀ·R⁻¹·F)⁻¹ · u0)
σ̂²(x0) = σ² · (1 - D1 + D2)
```

### MLE (objective à minimiser)
```
σ² = 1/N · ||Ỹ - Q1·Q1ᵀ·Ỹ||²
J(θ) = 0.5 · (N·log(2π·σ²) + log|R| + N)
```

### LOO error (critère sélection optimal mode)
```
LOO = Σ (eᵢ / (1 - Hᵢᵢ))² / Var(Y)   (formule analytique via hat matrix)
```

## Structure des fichiers

```
C:\_workingDir\_SF\test flexion\pck_python\
├── pck.py          # Classe PCKriging principale
├── kernels.py      # Fonctions kernel (Matérn 5/2, Gaussien, etc.)
├── pce_trend.py    # Trend PCE (chaospy + sklearn LARS)
└── test_pck.py     # Test sur f(x) = x·sin(x), x ∈ [0,15]
```

## Dépendances Python
- `numpy`, `scipy` : algèbre linéaire et optimisation MLE
- `chaospy` : base polynomiale orthogonale adaptée aux distributions (Legendre pour Uniform, Hermite pour Gaussian)
- `sklearn.linear_model` : `Lars` pour sparse regression

## Plan d'implémentation

### 1. `kernels.py`
```python
def eval_kernel(X1, X2, theta, family='matern-5_2', nugget=1e-4):
    """
    Calcule la matrice de corrélation K(X1, X2, θ).
    Ellipsoïdal : h = seuclidean(x1, x2, θ)
    Si X1==X2 : ajoute nugget sur la diagonale
    Supporte : 'matern-5_2', 'matern-3_2', 'gaussian', 'exponential'
    """
```

### 2. `pce_trend.py`
```python
class PCETrend:
    """
    Construit le trend PCE via chaospy + LARS.
    - Génère base polynomiale orthogonale adaptée aux distributions d'entrée
    - Régression LARS avec LOO early stopping
    - Retourne : indices multi-indices classés par importance, coefficients
    - Méthode eval(X) : retourne matrice design F (N × n_poly)
    """
    def fit(self, X, Y, distributions, degree_range=range(1, 4)):
        ...
    def eval(self, X):  # retourne F matrix
        ...
    def get_ranked_indices(self):  # pour mode sequential/optimal
        ...
```

### 3. `pck.py` — Classe principale
```python
class PCKriging:
    def __init__(self, mode='sequential', pce_degree=range(1,4),
                 corr_family='matern-5_2', nugget=1e-4, n_optim_starts=5):
        ...
    
    def fit(self, X, Y, distributions):
        """
        1. Calibre PCE trend (LARS sur la base orthogonale)
        2. Mode 'sequential': construit trend avec tous les polynômes
           Mode 'optimal': teste subsets croissants, sélectionne via LOO
        3. Optimise hyperparamètres θ par MLE (scipy.optimize.minimize, L-BFGS-B)
        4. Calcule matrices auxiliaires (chol, beta, sigmaSQ)
        """
    
    def predict(self, X_test, return_std=False):
        """
        Retourne (mean, variance_optionnel)
        Formule BLUP exacte du source MATLAB
        """
    
    def _mle_objective(self, log_theta, X, Y, F):
        """J(θ) = 0.5*(N*log(2π*σ²) + log|R| + N)"""
    
    def _compute_loo(self):
        """Calcul analytique de l'erreur LOO (pour mode optimal)"""
```

### 4. `test_pck.py`
```python
# Fonction: Y = X * sin(X), X ~ Uniform[0, 15]
# 10 points d'entraînement (Latin Hypercube)
# Comparer prediction vs vérité sur grille fine
# Tracer mean ± 2*std
# Comparer Sequential vs Optimal mode
```

## Mode Sequential vs Optimal

| Mode | Comportement |
|------|-------------|
| **Sequential** | LARS choisit les polynômes → tous utilisés comme trend en une seule calibration Kriging |
| **Optimal** | LARS choisit les polynômes → on teste tous les sous-ensembles [poly_1], [poly_1, poly_2], ... et on garde celui qui minimise LOO |

## Points d'attention

- Le scaling (normalisation des X) : UQLab transforme X → U via isoprobabilistic transform. En Python : pour Uniform[a,b] → Legendre, on mappe X sur [-1,1]. Pour Gaussian → Hermite, X est déjà dans l'espace standard. `chaospy` gère cela nativement.
- Le nugget : s'ajoute seulement sur la diagonale de R(X_train, X_train), pas dans r0.
- La cross-corrélation r0 = K(X_test, X_train) avec nugget=0 (fidèle au code MATLAB `CrossCorOpts.Nugget = 0`).
- theta optimisé dans l'espace log pour éviter les bornes : θ = exp(log_θ)

## Vérification
1. Tracer f(x) = x·sin(x) avec le surrogate sur [0,15] → visuellement vérifier que le mean suit la courbe vraie
2. Comparer LOO error Sequential vs Optimal
3. Vérifier que la variance est nulle aux points d'entraînement (cas interpolation)
4. Comparer avec `sklearn.gaussian_process.GaussianProcessRegressor` comme référence
