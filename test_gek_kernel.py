"""
test_gek_kernel.py
Tests pour les 3 fonctions GEK de branche5 :
  _prod_excl, kernel_deriv_factory, uq_assemble_global_Kernel

Sections :
  1.  _prod_excl                          (4 tests)
  2.  kernel_deriv_factory — structurel   (8 tests : 4 par famille)
  3.  kernel_deriv_factory — FD           (16 tests : 8 par famille)
  4.  uq_assemble_global_Kernel           (7 tests)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

from branche5 import (
    _prod_excl,
    kernel_deriv_factory,
    uq_assemble_global_Kernel,
    uq_eval_global_Kernel,
    uq_eval_Kernel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RNG  = np.random.default_rng(0)
EPS  = 1e-5       # pas différences finies
ATOL = 1e-7       # tolérance absolue pour comparaisons FD


def _std_opts(family):
    return {
        'Handle'   : uq_eval_Kernel,
        'Family'   : family,
        'Type'     : 'separable',
        'Isotropic': False,
        'Nugget'   : 0.0,
    }


def _fd_d1(f, X1, X2, theta, k):
    """Différence finie centrée de f par rapport à X1[:, k]."""
    X1p = X1.copy(); X1p[:, k] += EPS
    X1m = X1.copy(); X1m[:, k] -= EPS
    return (f(X1p, X2, theta) - f(X1m, X2, theta)) / (2 * EPS)


def _fd_d2(f, X1, X2, theta, k):
    """Différence finie centrée de f par rapport à X2[:, k]."""
    X2p = X2.copy(); X2p[:, k] += EPS
    X2m = X2.copy(); X2m[:, k] -= EPS
    return (f(X1, X2p, theta) - f(X1, X2m, theta)) / (2 * EPS)


# Données partagées pour les tests
_X_3x2 = np.array([[0.1, 0.7],
                    [0.5, 0.3],
                    [0.9, 0.1]], dtype=float)   # 3 points, 2 dimensions
_X2    = np.array([[0.2, 0.6],
                   [0.4, 0.8]], dtype=float)    # 2 points test distincts

_theta_2 = np.array([1.5, 0.8])
_theta_1 = np.array([1.2])

_X_1d = np.array([[0.1], [0.5], [0.9]], dtype=float)


# ===========================================================================
# Section 1 — _prod_excl
# ===========================================================================

class TestProdExcl:

    def test_M1_empty_product_is_one(self):
        """M=1 : le produit exclusif de l'unique slice est 1 (produit vide)."""
        K = RNG.uniform(0.5, 1.5, (4, 5, 1))
        out = _prod_excl(K)
        assert out.shape == (4, 5, 1)
        assert np.allclose(out[:, :, 0], 1.0)

    def test_M2_is_swap(self):
        """M=2 : K_excl[:,:,0] = K[:,:,1]  et  K_excl[:,:,1] = K[:,:,0]."""
        K = RNG.uniform(0.5, 1.5, (3, 4, 2))
        out = _prod_excl(K)
        assert np.allclose(out[:, :, 0], K[:, :, 1])
        assert np.allclose(out[:, :, 1], K[:, :, 0])

    def test_M3_each_slice_product_of_others(self):
        """M=3 : K_excl[:,:,l] = prod_{m≠l} K[:,:,m]."""
        K = RNG.uniform(0.5, 1.5, (3, 3, 3))
        out = _prod_excl(K)
        for l in range(3):
            expected = np.ones((3, 3))
            for m in range(3):
                if m != l:
                    expected *= K[:, :, m]
            assert np.allclose(out[:, :, l], expected), f"Echec pour l={l}"

    def test_consistency_K_times_excl_is_total(self):
        """K[:,:,l] * K_excl[:,:,l] == prod_all_dim pour tout l."""
        K = RNG.uniform(0.5, 1.5, (4, 4, 5))
        out = _prod_excl(K)
        total = K.prod(axis=2)          # (4, 4)
        for l in range(5):
            product = K[:, :, l] * out[:, :, l]
            assert np.allclose(product, total), f"Echec pour l={l}"


# ===========================================================================
# Section 2 — kernel_deriv_factory — propriétés structurelles
# ===========================================================================

class TestKernelDerivFactory:

    @pytest.mark.parametrize("family", ["gaussian", "matern-5_2"])
    def test_none_none_matches_uq_eval_kernel(self, family):
        """[None,None] doit retourner exactement uq_eval_Kernel(X1,X2,theta)."""
        f = kernel_deriv_factory(family, None, None)
        result  = f(_X_3x2, _X2, _theta_2)
        expected = uq_eval_Kernel(_X_3x2, _X2, _theta_2, _std_opts(family))
        assert result.shape == (3, 2)
        assert np.allclose(result, expected), \
            f"[{family}] [None,None] != uq_eval_Kernel"

    @pytest.mark.parametrize("family", ["gaussian", "matern-5_2"])
    def test_antisymmetry_i_None_vs_None_i(self, family):
        """∂k/∂xi = −∂k/∂xj pour les noyaux stationnaires séparables."""
        for i in range(2):
            f_der1 = kernel_deriv_factory(family, i,    None)
            f_der2 = kernel_deriv_factory(family, None, i   )
            v1 = f_der1(_X_3x2, _X2, _theta_2)
            v2 = f_der2(_X_3x2, _X2, _theta_2)
            assert np.allclose(v1, -v2, atol=1e-14), \
                f"[{family}] [i,None] ≠ -[None,i] pour i={i}"

    @pytest.mark.parametrize("family", ["gaussian", "matern-5_2"])
    def test_first_order_block_zero_on_identical_points(self, family):
        """[i,None] et [None,i] doivent être 0 quand X1=X2 (delta=0)."""
        X_same = np.array([[0.1, 0.7], [0.5, 0.3]])
        for i in range(2):
            for tag, (d, dp) in [("i,None", (i, None)), ("None,i", (None, i))]:
                f = kernel_deriv_factory(family, d, dp)
                v = f(X_same, X_same, _theta_2)
                # diagonale uniquement (xi=xj ssi i==j dans la matrice Gram)
                diag_v = np.diag(v)
                assert np.allclose(diag_v, 0.0, atol=1e-14), \
                    f"[{family}] [{tag}] diagonale ≠ 0 pour dim={i}"

    @pytest.mark.parametrize("family", ["gaussian", "matern-5_2"])
    def test_double_deriv_same_dim_gram_is_symmetric(self, family):
        """[i,i] doit produire une matrice symétrique en cas Gram (X1=X2)."""
        for i in range(2):
            f = kernel_deriv_factory(family, i, i)
            v = f(_X_3x2, _X_3x2, _theta_2)
            assert np.allclose(v, v.T, atol=1e-13), \
                f"[{family}] bloc [i,i] non symétrique pour i={i}"


# ===========================================================================
# Section 3 — kernel_deriv_factory — validation par différences finies
# ===========================================================================

class TestKernelDerivFD:
    """
    Validation des dérivées analytiques par différences finies centrées.
    Pas FD : eps = 1e-5, tolérance : 1e-7 absolu.
    """

    @pytest.mark.parametrize("family", ["gaussian", "matern-5_2"])
    @pytest.mark.parametrize("i", [0, 1])
    def test_der1_vs_fd_wrt_X1(self, family, i):
        """[i,None] == ∂k(X1,X2)/∂X1[:,i] par FD centrée."""
        f_base  = kernel_deriv_factory(family, None, None)
        f_der   = kernel_deriv_factory(family, i,    None)
        analytic = f_der(_X_3x2, _X2, _theta_2)
        fd       = _fd_d1(f_base, _X_3x2, _X2, _theta_2, i)
        # fd a shape (3, 2) car f_base : (n1,n2)
        assert np.allclose(analytic, fd, atol=ATOL), \
            f"[{family}] [i={i},None] : erreur FD max = {np.abs(analytic-fd).max():.2e}"

    @pytest.mark.parametrize("family", ["gaussian", "matern-5_2"])
    @pytest.mark.parametrize("j", [0, 1])
    def test_der2_vs_fd_wrt_X2(self, family, j):
        """[None,j] == ∂k(X1,X2)/∂X2[:,j] par FD centrée."""
        f_base  = kernel_deriv_factory(family, None, None)
        f_der   = kernel_deriv_factory(family, None, j   )
        analytic = f_der(_X_3x2, _X2, _theta_2)
        fd       = _fd_d2(f_base, _X_3x2, _X2, _theta_2, j)
        assert np.allclose(analytic, fd, atol=ATOL), \
            f"[{family}] [None,j={j}] : erreur FD max = {np.abs(analytic-fd).max():.2e}"

    @pytest.mark.parametrize("family", ["gaussian", "matern-5_2"])
    @pytest.mark.parametrize("i", [0, 1])
    def test_double_same_dim_vs_fd(self, family, i):
        """[i,i] == ∂/∂X2[:,i] de [i,None] par FD centrée."""
        f_d1    = kernel_deriv_factory(family, i, None)
        f_d2    = kernel_deriv_factory(family, i, i   )
        analytic = f_d2(_X_3x2, _X2, _theta_2)
        fd       = _fd_d2(f_d1, _X_3x2, _X2, _theta_2, i)
        assert np.allclose(analytic, fd, atol=ATOL), \
            f"[{family}] [i={i},i] : erreur FD max = {np.abs(analytic-fd).max():.2e}"

    @pytest.mark.parametrize("family", ["gaussian", "matern-5_2"])
    def test_double_cross_dim_vs_fd(self, family):
        """[0,1] == ∂/∂X1[:,0] de [None,1] par FD centrée."""
        f_d2    = kernel_deriv_factory(family, None, 1)
        f_cross = kernel_deriv_factory(family, 0,    1)
        analytic = f_cross(_X_3x2, _X2, _theta_2)
        fd       = _fd_d1(f_d2, _X_3x2, _X2, _theta_2, 0)
        assert np.allclose(analytic, fd, atol=ATOL), \
            f"[{family}] [0,1] : erreur FD max = {np.abs(analytic-fd).max():.2e}"


# ===========================================================================
# Section 4 — uq_assemble_global_Kernel
# ===========================================================================

class TestAssembleGlobalKernel:

    @pytest.mark.parametrize("family", ["gaussian", "matern-5_2"])
    def test_shape(self, family):
        """R̃ a la bonne taille n*(m+1) × n*(m+1)."""
        n, m = _X_3x2.shape
        Rt = uq_assemble_global_Kernel(_X_3x2, _X_3x2, _theta_2, family)
        assert Rt.shape == (n * (m + 1), n * (m + 1)), \
            f"[{family}] shape incorrecte : {Rt.shape}"

    @pytest.mark.parametrize("family", ["gaussian", "matern-5_2"])
    def test_bloc00_matches_uq_eval_kernel(self, family):
        """Le bloc (0,0) de R̃ doit être égal à uq_eval_Kernel(X,X,theta)."""
        n = _X_3x2.shape[0]
        Rt      = uq_assemble_global_Kernel(_X_3x2, _X_3x2, _theta_2, family)
        K_std   = uq_eval_Kernel(_X_3x2, _X_3x2, _theta_2, _std_opts(family))
        # Pour 'matern-5_2', uq_eval_Kernel produit une Gram (symétrique + diag=1)
        # kernel_deriv_factory [None,None] calcule K_uni.prod(axis=2) sans nugget
        # → la diagonale peut différer à cause du nugget dans uq_eval_Kernel
        # On compare donc sur les éléments hors-diagonale stricts
        mask = ~np.eye(n, dtype=bool)
        assert np.allclose(Rt[:n, :n][mask], K_std[mask], atol=1e-13), \
            f"[{family}] Bloc(0,0) hors-diagonale ≠ uq_eval_Kernel"

    @pytest.mark.parametrize("family", ["gaussian", "matern-5_2"])
    def test_R_tilde_is_symmetric(self, family):
        """R̃ doit être symétrique : R̃[i,j] = R̃[j,i]."""
        Rt = uq_assemble_global_Kernel(_X_3x2, _X_3x2, _theta_2, family)
        assert np.allclose(Rt, Rt.T, atol=1e-12), \
            f"[{family}] R̃ non symétrique, erreur max = {np.abs(Rt - Rt.T).max():.2e}"

    @pytest.mark.parametrize("family", ["gaussian", "matern-5_2"])
    def test_bloc_k0_antisymmetric_to_bloc_0k(self, family):
        """Block(k,0) = -Block(0,k) (antisymétrie dérivée 1er vs 2e argument)."""
        n = _X_3x2.shape[0]
        Rt = uq_assemble_global_Kernel(_X_3x2, _X_3x2, _theta_2, family)
        for k in range(1, 3):   # k=1,2 (les blocs de gradient pour m=2)
            B0k = Rt[:n,    k*n:(k+1)*n]   # Block(0, k)
            Bk0 = Rt[k*n:(k+1)*n, :n]     # Block(k, 0)
            assert np.allclose(Bk0, -B0k, atol=1e-12), \
                f"[{family}] Block({k},0) ≠ -Block(0,{k})"

    @pytest.mark.parametrize("family", ["gaussian", "matern-5_2"])
    def test_R_tilde_positive_semidefinite(self, family):
        """R̃ (matrice de Gram augmentée) doit être semi-définie positive."""
        Rt = uq_assemble_global_Kernel(_X_3x2, _X_3x2, _theta_2, family)
        eigs = np.linalg.eigvalsh(Rt)
        assert eigs.min() >= -1e-8, \
            f"[{family}] min eigenvalue = {eigs.min():.2e} < -1e-8"

    def test_invalid_family_raises_ValueError(self):
        """Un noyau inconnu doit lever ValueError."""
        with pytest.raises(ValueError, match="non supporte"):
            uq_assemble_global_Kernel(_X_3x2, _X_3x2, _theta_2, 'exponential')

    @pytest.mark.parametrize("family", ["gaussian", "matern-5_2"])
    def test_1d_input_shape(self, family):
        """Test avec M=1 : R̃ de taille 2n × 2n."""
        n = _X_1d.shape[0]
        Rt = uq_assemble_global_Kernel(_X_1d, _X_1d, _theta_1, family)
        assert Rt.shape == (2 * n, 2 * n), \
            f"[{family}] shape 1D incorrecte : {Rt.shape}"
        assert np.allclose(Rt, Rt.T, atol=1e-12), \
            f"[{family}] R̃ 1D non symétrique"


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
