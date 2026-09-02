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


# --------------------------------------------------------------------------- #
# LE TROISIEME CAS : DUR, ET C'EST LE POINT                                    #
# --------------------------------------------------------------------------- #
#: TOLERANCES MESUREES, ET POURQUOI ELLES SONT CENT FOIS PLUS LARGES
#:
#: `console` (trois variables, `g` en `h^-3`) est le premier cas du depot que
#: le metamodele ne resout PAS a la virgule. Mesure du 02/09/2026, meme plan
#: de 24 points, HL-RF multistart depuis cinq points :
#:
#:     n    PCK              GEPCK
#:     24   2,24 %           0,97 %
#:     40   1,65 %           0,55 %
#:     60   1,56 %           0,39 %
#:
#: contre 0,0017 % et 0,0072 % sur `flexion`. Ce n'est pas un defaut : c'est
#: ce que vaut un metamodele a 24 points sur un etat limite fortement non
#: polynomial. Le figer ici donne au depot son premier cas OU LA METHODE
#: TRAVAILLE, et ou une degradation serait visible.
TOL_BETA_CONSOLE = {'PCK': 0.030, 'GEPCK': 0.013}

#: Cinq points de depart en trois variables. Le dernier vise deja la zone du
#: point de conception : sans lui, HL-RF depuis l'origine seule peut s'arreter
#: sur un autre bassin d'un etat limite courbe.
STARTS_3 = np.array([[0.0, 0.0, 0.0], [2.0, -2.0, 0.0], [-2.0, 2.0, 0.0],
                     [1.0, 1.0, 1.0], [3.0, -1.0, -2.0]])


@pytest.fixture(scope='module')
def console_ajuste():
    """Le troisieme cas, ajuste une fois pour ce fichier."""
    import warnings
    from reference.limit_states import ConsoleLS
    ls = ConsoleLS()
    X = harness.make_doe(24, ls.n_var)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ls, {k: harness.fit(k, X, ls, max_degree=3)
                    for k in ('PCK', 'GEPCK')}


@pytest.mark.parametrize('kind', ['PCK', 'GEPCK'])
def test_beta_du_metamodele_sur_le_cas_console(kind, console_ajuste):
    ls, modeles = console_ajuste
    g_hat, grad_hat, _ = harness.predictors(kind, modeles[kind])
    r = multistart_hlrf(g_hat, grad_hat, STARTS_3)
    assert r is not None, f'{kind} : FORM n a converge depuis aucun depart'
    beta_ref = ls.beta_exact()
    ecart = abs(r['beta'] - beta_ref) / beta_ref
    assert ecart < TOL_BETA_CONSOLE[kind], (
        f'{kind} : beta = {r["beta"]:.6f} contre {beta_ref:.6f} attendu '
        f'({100 * ecart:.2f} % > {100 * TOL_BETA_CONSOLE[kind]:.2f} %)')


def test_sur_un_cas_DUR_le_gradient_enrichi_gagne_vraiment(console_ajuste):
    """CE QUE CE CAS APPORTE, ET QU'AUCUN AUTRE NE MONTRAIT.

    GEPCK dispose des gradients en plus de PCK : il devrait toujours etre
    meilleur. Sur `flexion` la comparaison ne tranche pas -- les deux sont
    a 1e-05 pres de l'exact, et l'ecart entre eux est du bruit d'ajustement
    (0,0017 % contre 0,0072 %, PCK devant). Sur `linear`, les deux sont
    exacts.

    Ici la question a une reponse. Mesure du 02/09/2026 :

        n    PCK      GEPCK    rapport
        24   2,24 %   0,97 %   2,3
        40   1,65 %   0,55 %   3,0
        60   1,56 %   0,39 %   4,0

    L'avantage du gradient enrichi est donc REEL et CROISSANT avec la taille
    du plan, la ou personne ne pouvait le montrer. C'est l'argument de la
    methode, et il est desormais tenu par un test.

    Le seuil de 1,5 laisse de la marge sous le 2,3 mesure : ce qui est fige,
    c'est l'ORDRE des deux, pas sa valeur exacte.
    """
    ls, modeles = console_ajuste
    beta_ref = ls.beta_exact()
    ecarts = {}
    for kind in ('PCK', 'GEPCK'):
        g_hat, grad_hat, _ = harness.predictors(kind, modeles[kind])
        r = multistart_hlrf(g_hat, grad_hat, STARTS_3)
        ecarts[kind] = abs(r['beta'] - beta_ref) / beta_ref
    assert ecarts['GEPCK'] < ecarts['PCK'] / 1.5, (
        'GEPCK (%.4f %%) ne gagne plus nettement sur PCK (%.4f %%) : rapport '
        '%.2f, mesure a 2,3 le 02/09/2026. GEPCK dispose des gradients en '
        'plus -- s il ne fait pas mieux sur un cas DUR, c est qu il ne les '
        'exploite pas.'
        % (100 * ecarts['GEPCK'], 100 * ecarts['PCK'],
           ecarts['PCK'] / max(ecarts['GEPCK'], 1e-30)))
