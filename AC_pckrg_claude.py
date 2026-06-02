"""
AC_pckrg_claude.py — Fichier principal PC-Kriging Python
=========================================================

Tests progressifs :
  TEST 1 : kernel.py   — vérification des formules de corrélation
  TEST 2 : pce_trend.py — vérification de la base polynomiale + LARS
  TEST 3 : pck.py       — modèle complet sur f(x) = x·sin(x), x ∈ [0, 15]
  TEST 4 : pck.py       — comparaison Sequential vs Optimal
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # pour environnements sans écran
import matplotlib.pyplot as plt

from kernels import eval_kernel
from pce_trend import PCETrend
from pck import PCKriging


# ============================================================
# TEST 1 — Fonctions kernel
# ============================================================

def test_kernels():
    """
    Vérifie les propriétés élémentaires des kernels :
      - K(x, x) = 1 pour un point isolé (sans nugget)
      - K symétrique
      - K(x1, x2) ∈ [0, 1]
      - Décroissance avec la distance
    """
    print("\n" + "="*55)
    print("TEST 1 — Kernels (kernels.py)")
    print("="*55)

    # Points 1D
    X = np.array([[0.0], [1.0], [3.0], [7.0]])
    theta = np.array([1.0])

    for family in ['matern-5_2', 'matern-3_2', 'gaussian', 'exponential']:
        R = eval_kernel(X, X, theta, family=family, nugget=0.0)
        diag_ok = np.allclose(np.diag(R), 1.0, atol=1e-10)
        sym_ok = np.allclose(R, R.T, atol=1e-10)
        bounded_ok = np.all(R >= 0) and np.all(R <= 1.0 + 1e-10)
        print(f"  {family:<15}  diag=1: {diag_ok}  symetrique: {sym_ok}  dans[0,1]: {bounded_ok}")

    # Vérification Matérn 5/2 à la main : K(h=0.5)
    h_test = 0.5
    sqrt5 = np.sqrt(5)
    expected = (1 + sqrt5*h_test + 5/3*h_test**2) * np.exp(-sqrt5*h_test)
    X1 = np.array([[0.0]])
    X2 = np.array([[h_test]])
    K_computed = eval_kernel(X1, X2, np.array([1.0]), family='matern-5_2', nugget=0.0)[0, 0]
    err = abs(K_computed - expected)
    print(f"\n  Matérn 5/2 à h=0.5 : attendu={expected:.6f}, calculé={K_computed:.6f}, err={err:.2e}")
    print(f"  → {'OK' if err < 1e-10 else 'ECHEC'}")

    # Nugget sur la diagonale
    R_nug = eval_kernel(X, X, theta, family='matern-5_2', nugget=1e-4)
    nugget_ok = np.allclose(np.diag(R_nug) - np.diag(R), 1e-4, atol=1e-12)
    print(f"\n  Nugget sur diagonale : {nugget_ok}")

    print("  [TEST 1 terminé]\n")


# ============================================================
# TEST 2 — PCE Trend
# ============================================================

def test_pce_trend():
    """
    Vérifie la construction du trend PCE + LARS sur une fonction polynomiale connue.
    Si Y = 3 + 2x + x² (polynomial exact de degré 2), le PCE doit bien
    capturer le trend sans résidu Kriging.
    """
    print("="*55)
    print("TEST 2 — PCE Trend (pce_trend.py)")
    print("="*55)

    np.random.seed(42)

    # Fonction polynomiale : Y = 1 + 2x + x², x ∈ [0, 3]
    # Base Legendre sur Uniform[0, 3]
    N = 20
    X = np.random.uniform(0, 3, size=(N, 1))
    Y = (1 + 2*X[:, 0] + X[:, 0]**2)

    distributions = [{'type': 'uniform', 'parameters': [0.0, 3.0]}]

    pce = PCETrend(distributions, degree=range(1, 5))
    pce.fit(X, Y)

    print(f"  Degrés testés : 1 à 4")
    print(f"  Nombre de polynômes sélectionnés : {pce.n_polynomials}")
    print(f"  LOO error : {pce.loo_error:.4e}")

    # Vérifier que le LOO error est faible (bon fit polynomial)
    print(f"  → LOO faible (< 0.1) : {'OK' if pce.loo_error < 0.1 else 'A vérifier'}")

    # Test d'évaluation
    X_test = np.array([[0.5], [1.0], [2.5]])
    F = pce.eval_active(X_test)
    print(f"  Shape F(X_test) : {F.shape}  (attendu : ({len(X_test)}, {pce.n_polynomials}))")
    print(f"  → {'OK' if F.shape == (len(X_test), pce.n_polynomials) else 'ECHEC'}")

    print("  [TEST 2 terminé]\n")


# ============================================================
# TEST 3 — PC-Kriging complet sur x·sin(x)
# ============================================================

def test_pck_xsinx(mode='sequential', plot=True):
    """
    Test principal : f(x) = x·sin(x) sur [0, 15], X ~ Uniform[0, 15].
    10 points d'entraînement (Latin Hypercube simplifié).
    Vérifie :
      - Variance nulle aux points d'entraînement (interpolation exacte)
      - Mean proche de la vraie valeur sur grille fine
    """
    print("="*55)
    print(f"TEST 3 — PCK '{mode}' sur f(x) = x·sin(x)")
    print("="*55)

    np.random.seed(101)

    # Design expérimental : 10 points (Latin Hypercube 1D simple)
    N = 10
    # LHS 1D
    strata = np.arange(N) / N + np.random.uniform(0, 1/N, N)
    X_train = (strata * 15.0).reshape(-1, 1)
    Y_train = X_train[:, 0] * np.sin(X_train[:, 0])

    distributions = [{'type': 'uniform', 'parameters': [0.0, 15.0]}]

    # Calibration
    pck = PCKriging(mode=mode, pce_degree=range(1, 8),
                    corr_family='matern-5_2', nugget=1e-4, n_optim_starts=10)
    pck.fit(X_train, Y_train, distributions)
    print(f"  {pck}")
    print(f"  LOO error : {pck.loo_error:.4e}")

    # Prédiction sur grille fine
    X_grid = np.linspace(0, 15, 300).reshape(-1, 1)
    Y_true = X_grid[:, 0] * np.sin(X_grid[:, 0])

    Y_pred, Y_std = pck.predict(X_grid, return_std=True)

    # Vérification : variance aux points d'entraînement ≈ 0 (interpolation)
    _, std_train = pck.predict(X_train, return_std=True)
    interp_ok = np.all(std_train < 0.1)
    print(f"  Variance nulle aux points d'entraînement : {'OK' if interp_ok else 'A vérifier'}"
          f"  (max std = {std_train.max():.4e})")

    # Erreur relative sur la grille
    rmse = np.sqrt(np.mean((Y_pred - Y_true)**2))
    print(f"  RMSE sur la grille : {rmse:.4f}")

    if plot:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(X_grid, Y_true, 'k-', lw=1.5, label='Vrai f(x) = x·sin(x)')
        ax.plot(X_grid, Y_pred, 'b-', lw=1.5, label=f'PCK {mode} (mean)')
        ax.fill_between(X_grid[:, 0],
                        Y_pred - 2*Y_std, Y_pred + 2*Y_std,
                        alpha=0.25, color='blue', label='±2σ')
        ax.scatter(X_train[:, 0], Y_train, c='red', zorder=5, s=50,
                   label='Points entraînement')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'PC-Kriging {mode} — f(x) = x·sin(x), N={N} pts')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fname = f'pck_{mode}_xsinx.png'
        fig.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Figure sauvegardée : {fname}")

    print(f"  [TEST 3 '{mode}' terminé]\n")
    return pck


# ============================================================
# TEST 4 — Sequential vs Optimal
# ============================================================

def test_comparison_seq_opt():
    """
    Compare Sequential et Optimal sur f(x) = x·sin(x).
    Affiche les LOO errors et RMSE des deux modes.
    """
    print("="*55)
    print("TEST 4 — Comparaison Sequential vs Optimal")
    print("="*55)

    np.random.seed(101)
    N = 10
    strata = np.arange(N) / N + np.random.uniform(0, 1/N, N)
    X_train = (strata * 15.0).reshape(-1, 1)
    Y_train = X_train[:, 0] * np.sin(X_train[:, 0])
    distributions = [{'type': 'uniform', 'parameters': [0.0, 15.0]}]

    X_grid = np.linspace(0, 15, 300).reshape(-1, 1)
    Y_true = X_grid[:, 0] * np.sin(X_grid[:, 0])

    results = {}
    for mode in ['sequential', 'optimal']:
        pck = PCKriging(mode=mode, pce_degree=range(1, 8),
                        corr_family='matern-5_2', nugget=1e-4, n_optim_starts=10)
        pck.fit(X_train, Y_train, distributions)
        Y_pred, Y_std = pck.predict(X_grid, return_std=True)
        rmse = np.sqrt(np.mean((Y_pred - Y_true)**2))
        results[mode] = {'pck': pck, 'rmse': rmse, 'Y_pred': Y_pred, 'Y_std': Y_std}
        print(f"  {mode:12s} | n_poly={pck._n_poly_used:2d} | "
              f"LOO={pck.loo_error:.3e} | RMSE={rmse:.4f}")

    # Figure comparative
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, mode in zip(axes, ['sequential', 'optimal']):
        r = results[mode]
        ax.plot(X_grid, Y_true, 'k-', lw=1.5, label='Vrai f(x)')
        ax.plot(X_grid, r['Y_pred'], 'b-', lw=1.5, label=f'PCK mean')
        ax.fill_between(X_grid[:, 0],
                        r['Y_pred'] - 2*r['Y_std'],
                        r['Y_pred'] + 2*r['Y_std'],
                        alpha=0.2, color='blue', label='±2σ')
        ax.scatter(X_train[:, 0], Y_train, c='red', zorder=5, s=50)
        ax.set_title(f'{mode} (LOO={r["pck"].loo_error:.2e}, RMSE={r["rmse"]:.3f})')
        ax.set_xlabel('x')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('f(x)')
    fig.suptitle("PC-Kriging : Sequential vs Optimal — f(x) = x·sin(x)", y=1.01)
    fig.savefig('pck_comparison.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print("  Figure sauvegardée : pck_comparison.png")
    print("  [TEST 4 terminé]\n")


# ============================================================
# ENTRÉE PRINCIPALE
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*55)
    print(" AC_pckrg_claude.py — Tests PC-Kriging Python")
    print("="*55)

    test_kernels()
    test_pce_trend()
    test_pck_xsinx(mode='sequential', plot=True)
    test_pck_xsinx(mode='optimal', plot=True)
    test_comparison_seq_opt()

    print("="*55)
    print(" Tous les tests terminés.")
    print("="*55)
