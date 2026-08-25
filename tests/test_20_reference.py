"""
Auto-controle des oracles.

Le harness ne vaut que si ses references sont justes. Ces tests ne touchent
PAS a _lib/ : ils verifient que les etats limites analytiques et le FORM de
reference sont coherents entre eux, par deux chemins independants
(minimisation scalaire sur la courbe g=0 d'un cote, HL-RF de l'autre).
Si ce fichier tombe, ne pas chercher la cause dans la librairie.
"""

import numpy as np
import pytest

from reference.form import hlrf
from reference.limit_states import FlexionLS, LinearLS

pytestmark = pytest.mark.oracle


def test_flexion_valeurs_de_base(flexion_ls):
    ls = flexion_ls
    assert ls.As == pytest.approx(4 * np.pi * 0.010 ** 2, rel=1e-12)
    assert ls.Med == pytest.approx(0.297, rel=1e-12)
    # la section est sure a la moyenne, avec une marge realiste
    assert 0.15 < ls.g([[0.0, 0.0]])[0] < 0.35


def test_flexion_gradient_analytique_vs_fd(flexion_ls):
    """L'oracle de gradient sert de reference aux tests GEPCK : il doit etre juste."""
    U = np.array([[0.3, -1.2], [-1.0, 0.5], [1.7, -2.4], [0.0, 0.0]])
    G = flexion_ls.grad(U)
    h = 1e-6
    FD = np.zeros_like(G)
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        FD[:, j] = (flexion_ls.g(U + e) - flexion_ls.g(U - e)) / (2 * h)
    assert np.abs(G - FD).max() < 1e-8


def test_point_de_conception_est_sur_letat_limite(flexion_ls):
    u = flexion_ls.u_star_exact()
    assert abs(flexion_ls.g([u])[0]) < 1e-12


@pytest.mark.parametrize('cls', [LinearLS, FlexionLS])
def test_form_reference_retrouve_beta_exact(cls):
    """HL-RF (oracle 2) doit retrouver le beta de l'oracle 1 a 1e-10 pres."""
    ls = cls()
    r = hlrf(ls.g, ls.grad, np.zeros(2))
    assert r['converged'], f'{ls.name} : HL-RF de reference n a pas converge'
    assert r['beta'] == pytest.approx(ls.beta_exact(), abs=1e-10)
    assert np.allclose(r['u_star'], ls.u_star_exact(), atol=1e-8)


def test_form_reference_insensible_au_point_de_depart(flexion_ls):
    starts = np.array([[0.0, 0.0], [3.0, 3.0], [-4.0, 1.0], [0.5, -5.0]])
    betas = []
    for u0 in starts:
        r = hlrf(flexion_ls.g, flexion_ls.grad, u0)
        assert r['converged'], f'pas de convergence depuis {u0}'
        betas.append(r['beta'])
    assert np.ptp(betas) < 1e-8


def test_doe_reproductible():
    """Le DOE du harness doit etre identique d'une execution et d'une machine
    a l'autre : c'est ce qui rend les goldens comparables."""
    import harness
    a = harness.make_doe(24, 2)
    b = harness.make_doe(24, 2)
    assert np.array_equal(a, b)
    assert a.shape == (24, 2)
    # LHS centre : une seule observation par strate sur chaque axe
    for j in range(2):
        strates = np.floor((np.argsort(np.argsort(a[:, j])) + 0.5))
        assert len(np.unique(strates)) == 24
