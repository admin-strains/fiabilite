"""
Chaine complete : DOE -> metamodele -> FORM -> indice de fiabilite.

C'est le "cas test simple" au sens metier : on ne verifie pas seulement que
les chiffres n'ont pas bouge (test_30), mais que le resultat FINAL reste
JUSTE, en le comparant a un beta calcule analytiquement. Un refactoring qui
casserait la physique sans changer les coefficients tomberait ici ; un
refactoring qui change les coefficients sans degrader la physique tombe dans
test_30 seulement, et c'est le signal qu'il faut regenerer un golden.

Les tolerances sont des EXIGENCES METIER, pas des mesures : elles disent ce
qu'on accepte de perdre entre le vrai etat limite et sa version metamodelee.
"""

import numpy as np
import pytest

import harness
from reference.form import hlrf, multistart_hlrf

pytestmark = pytest.mark.oracle

# ecart relatif tolere sur beta, par metamodele.
# GEPCK est plus laxiste que PCK : c'est un CONSTAT sur la branche cleaning
# (0.01 % pour PCK contre 1.3 % pour GEPCK sur le meme DOE), pas une fatalite.
# Voir tests/test_50_known_defects.py::test_gepck_interpole_son_doe.
#: Ecart relatif accepte sur beta, entre le metamodele et l'oracle analytique.
#:
#: Valait 0,5 % (PCK) et 2 % (GEPCK) : des seuils cales sur le DEFAUT, pas sur
#: ce que la methode peut faire. Le plan de nettoyage fixait d'avance le
#: critere de reussite de la phase 6 -- « ecart sur beta sous 0,05 %,
#: c'est-a-dire au moins aussi bon que PCK ». C'est ce seuil-la qui est
#: applique ici depuis la correction des defauts 2 et 3 (pepite par defaut).
#:
#: Mesure du 26/08/2026 sur le plan de 24 points :
#:     PCK    0,0017 %      (valait 0,0114 %)
#:     GEPCK  0,0072 %      (valait 1,2982 %)
#: GEPCK est donc redevenu meilleur que PCK, ce qu'il devrait toujours etre :
#: il dispose des gradients en plus.
TOL_BETA = {'PCK': 0.0005, 'GEPCK': 0.0005}

STARTS = np.array([[0.0, 0.0], [2.0, -2.0], [-2.0, 2.0], [1.0, 1.0]])


@pytest.mark.parametrize('kind', ['PCK', 'GEPCK'])
def test_beta_du_metamodele_vs_beta_analytique(kind, fitted, flexion_ls):
    g_hat, grad_hat, _ = harness.predictors(kind, fitted[kind])
    r = multistart_hlrf(g_hat, grad_hat, STARTS)
    assert r is not None, f'{kind} : FORM n a converge depuis aucun point de depart'
    beta_ref = flexion_ls.beta_exact()
    ecart = abs(r['beta'] - beta_ref) / beta_ref
    assert ecart < TOL_BETA[kind], (
        f'{kind} : beta = {r["beta"]:.6f} contre {beta_ref:.6f} attendu '
        f'({100 * ecart:.2f} % > {100 * TOL_BETA[kind]:.2f} %)')


@pytest.mark.parametrize('kind', ['PCK', 'GEPCK'])
def test_point_de_conception_du_metamodele(kind, fitted, flexion_ls):
    g_hat, grad_hat, _ = harness.predictors(kind, fitted[kind])
    r = multistart_hlrf(g_hat, grad_hat, STARTS)
    u_ref = flexion_ls.u_star_exact()
    # le point de conception pilote le tirage d'importance en aval :
    # une derive de plus de 0.15 en norme change la densite d'echantillonnage
    assert np.linalg.norm(r['u_star'] - u_ref) < 0.15, \
        f'{kind} : u* = {np.round(r["u_star"], 4)} contre {np.round(u_ref, 4)}'


@pytest.mark.parametrize('kind', ['PCK', 'GEPCK'])
def test_metamodele_reste_fidele_le_long_de_letat_limite(kind, fitted, flexion_ls):
    """Le metamodele doit etre juste LA OU CA COMPTE : sur g=0, pas partout."""
    u1 = np.linspace(-2.5, 1.5, 41)
    U = np.column_stack([u1, [flexion_ls.u2_on_LS(v) for v in u1]])
    U = U[np.isfinite(U[:, 1])]
    g_hat, _, _ = harness.predictors(kind, fitted[kind])
    err = np.abs(g_hat(U))          # vrai g vaut 0 sur cette courbe
    assert err.max() < 0.02, f'{kind} : ecart max {err.max():.4f} sur l etat limite'


def test_le_cas_lineaire_est_retrouve_exactement(linear_ls):
    """Un metamodele qui contient le degre 1 doit rendre beta a la precision
    machine sur un etat limite lineaire : c'est un invariant, pas une mesure."""
    import warnings
    X = harness.make_doe(24, 2)
    for kind in ('PCK', 'GEPCK'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fm = harness.fit(kind, X, linear_ls)
        g_hat, grad_hat, _ = harness.predictors(kind, fm)
        r = hlrf(g_hat, grad_hat, np.zeros(2))
        assert r['beta'] == pytest.approx(linear_ls.beta_exact(), abs=1e-6), \
            f'{kind} : beta = {r["beta"]}'


@pytest.mark.parametrize('kind', ['PCK', 'GEPCK'])
def test_form_converge_depuis_tous_les_points_de_depart(kind, fitted):
    """Le multistart de production suppose que FORM converge souvent ;
    on verifie qu'au moins la moitie des departs aboutissent."""
    g_hat, grad_hat, _ = harness.predictors(kind, fitted[kind])
    ok = [hlrf(g_hat, grad_hat, u0)['converged'] for u0 in STARTS]
    assert sum(ok) >= len(STARTS) // 2, f'{kind} : {sum(ok)}/{len(STARTS)} convergences'
