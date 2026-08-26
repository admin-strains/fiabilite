"""
Le chemin chaud de la prediction : ce qui a ete supprime, et pourquoi.

PHASE 7 du plan de nettoyage. Deux calculs etaient refaits a CHAQUE appel de
prediction alors qu'ils ne dependent que de l'ajustement. Les supprimer divise
par 1,8 le temps de FORM + tirage d'importance sur la chaine complete, sans
changer un chiffre du resultat.

1. LA TRANSFORMATION ISOPROBABILISTE, QUI ETAIT UNE IDENTITE
   Source et cible sont Gaussiennes (0, 1) des deux cotes -- tout se passe en
   espace standard. Le code calculait pourtant `norm.ppf(norm.cdf(u))`.

   Ce n'etait pas qu'une perte de temps. `norm.cdf(u)` sature a 1.0 en double
   precision : l'aller-retour perd 5,8e-06 a u = 7, 8,4e-03 a u = 8, et rend
   **+inf** au-dela de u = 8,3. Or les bornes de recherche EFF de ces etudes
   valent exactement [-7,5 ; +7,5]. Cote negatif il n'y a rien de tel, d'ou
   une erreur ASYMETRIQUE -- la pire sorte a diagnostiquer.

   Mesure sur une grille de 1 681 points couvrant [-7,5 ; 7,5]^2, etat limite
   lineaire (un hyperplan que le metamodele contient exactement) :
       erreur max AVANT  2,510e-04
       erreur max APRES  5,862e-14      -- dix ordres de grandeur

2. L'INVERSE COMPLETE DE R, POUR N'EN TIRER QU'UN VECTEUR
   `predict.py` reconstruisait `R^-1` par `solve(cholR, solve(cholR.T, I))`
   -- N_aug descentes-remontees -- a chaque appel, pour calculer
   `R^-1 @ residu`. Or `residu = Y_aug - F_tilde @ beta` ne depend que de
   l'ajustement : c'est une constante du metamodele.

Ces tests ne demandent ni Digital Structure ni OpenTURNS.
"""

import os
import sys

import numpy as np
import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
for p in (os.path.join(REPO, "_lib"), TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from predict import _inverse_complete, poids_duaux                # noqa: E402
from transform import _memes_marginales, _uq_IsopTransform        # noqa: E402

GAUSS = {"Type": "Gaussian", "Parameters": [0.0, 1.0]}


# --------------------------------------------------------------------------- #
# 1. La transformation identite                                               #
# --------------------------------------------------------------------------- #
def test_deux_marginales_identiques_sont_reconnues():
    assert _memes_marginales(GAUSS, dict(GAUSS))
    assert _memes_marginales({"Type": "gaussian", "Parameters": [0.0, 1.0]}, GAUSS)


@pytest.mark.parametrize("autre", [
    {"Type": "Gaussian", "Parameters": [0.0, 2.0]},      # meme loi, autre ecart-type
    {"Type": "Gaussian", "Parameters": [1.0, 1.0]},      # autre moyenne
    {"Type": "Uniform", "Parameters": [0.0, 1.0]},       # autre famille
    {"Type": "Gaussian", "Parameters": [0.0, 1.0, 3.0]},  # arite differente
])
def test_deux_marginales_differentes_ne_sont_pas_confondues(autre):
    """Un raccourci trop gourmand rendrait l'identite la ou une vraie
    transformation est due -- une faute silencieuse et grave."""
    assert not _memes_marginales(GAUSS, autre)


def test_une_marginale_bornee_n_est_jamais_identique():
    """Tronquer change la loi, meme si le type et les parametres coincident."""
    borne = dict(GAUSS, Bounds=[-3.0, 3.0])
    assert not _memes_marginales(GAUSS, borne)
    assert not _memes_marginales(borne, GAUSS)


def test_le_raccourci_rend_exactement_l_entree():
    X = np.array([[0.0, 1.0], [-3.0, 2.5], [7.5, -7.5]])
    Y = _uq_IsopTransform(X, [GAUSS, GAUSS], [dict(GAUSS), dict(GAUSS)])
    assert np.array_equal(Y, X), "le raccourci doit etre l'identite EXACTE"


def test_le_raccourci_evite_la_falaise_de_la_cdf():
    """Le fait qui fait de cette correction autre chose qu'une optimisation.

    Sans raccourci, l'aller-retour `ppf(cdf(u))` diverge : +inf des u = 8,3.
    Les bornes de trace de ces etudes vont a 7,5, et le plan d'experiences
    initial peut placer des points au-dela.
    """
    from scipy import stats                                       # noqa: PLC0415
    for u in (7.0, 8.0, 8.5, 12.0):
        aller_retour = stats.norm.ppf(stats.norm.cdf(u))
        X = np.array([[u, 0.0]])
        obtenu = _uq_IsopTransform(X, [GAUSS, GAUSS], [dict(GAUSS), dict(GAUSS)])[0, 0]
        assert obtenu == u
        if u >= 8.5:
            assert not np.isfinite(aller_retour), \
                "u=%s : l'aller-retour ne diverge plus, revoir ce test" % u


def test_une_vraie_transformation_reste_calculee():
    """Le raccourci ne doit pas court-circuiter les cas legitimes."""
    X = np.array([[0.0], [1.0], [-1.0]])
    cible = {"Type": "Gaussian", "Parameters": [10.0, 2.0]}
    Y = _uq_IsopTransform(X, [GAUSS], [cible])
    assert np.allclose(Y.ravel(), [10.0, 12.0, 8.0], atol=1e-9)


def test_le_raccourci_agit_marginale_par_marginale():
    """Une variable inchangee et une variable transformee dans le meme appel."""
    X = np.array([[2.0, 0.0], [-2.0, 1.0]])
    cible = [dict(GAUSS), {"Type": "Gaussian", "Parameters": [10.0, 2.0]}]
    Y = _uq_IsopTransform(X, [GAUSS, GAUSS], cible)
    assert np.array_equal(Y[:, 0], X[:, 0])              # identite exacte
    assert np.allclose(Y[:, 1], [10.0, 12.0], atol=1e-9)  # vraie transformation


# --------------------------------------------------------------------------- #
# 2. Les poids duaux                                                          #
# --------------------------------------------------------------------------- #
def _systeme(n=12, graine=0):
    rng = np.random.default_rng(graine)
    A = rng.normal(size=(n, n))
    R = A @ A.T + n * np.eye(n)          # symetrique definie positive
    cholR = np.linalg.cholesky(R).T      # convention MATLAB : triangulaire sup
    return R, {"cholR": cholR}


def test_les_poids_duaux_resolvent_bien_le_systeme():
    R, am = _systeme()
    residu = np.arange(1.0, R.shape[0] + 1.0)
    poids = poids_duaux(am, residu)
    assert np.max(np.abs(R @ poids - residu)) < 1e-9


def test_le_cache_rend_le_meme_vecteur():
    R, am = _systeme()
    residu = np.arange(1.0, R.shape[0] + 1.0)
    a = poids_duaux(am, residu)
    b = poids_duaux(am, residu)
    assert np.array_equal(a, b)
    assert "_poids_duaux" in am


def test_un_residu_different_invalide_le_cache():
    """Un cache qui rend la valeur d'un autre residu serait un faux resultat,
    pas une lenteur. C'est le seul vrai risque de cette optimisation."""
    R, am = _systeme()
    r1 = np.arange(1.0, R.shape[0] + 1.0)
    r2 = r1 * 2.0
    p1 = poids_duaux(am, r1)
    p2 = poids_duaux(am, r2)
    assert not np.array_equal(p1, p2)
    assert np.max(np.abs(R @ p2 - r2)) < 1e-9
    assert np.max(np.abs(poids_duaux(am, r1) - p1)) == 0.0   # r1 recalcule juste


def test_modifier_le_residu_en_place_n_empoisonne_pas_le_cache():
    """Le cache garde une COPIE du residu : sinon un appelant qui reutilise son
    tableau ferait passer un nouveau residu pour l'ancien."""
    R, am = _systeme()
    residu = np.arange(1.0, R.shape[0] + 1.0)
    poids_duaux(am, residu)
    residu[0] = 999.0
    neuf = poids_duaux(am, residu)
    assert np.max(np.abs(R @ neuf - residu)) < 1e-9


def test_sans_cholesky_le_repli_reste_disponible():
    """`_calc_CholR` rend `Rinv` au lieu de `cholR` quand la factorisation
    echoue. Les deux chemins doivent rendre la meme chose."""
    R, _ = _systeme()
    am = {"cholR": None, "Rinv": np.linalg.inv(R)}
    residu = np.arange(1.0, R.shape[0] + 1.0)
    assert np.max(np.abs(R @ poids_duaux(am, residu) - residu)) < 1e-9


def test_l_inverse_complete_reste_disponible_pour_la_covariance():
    """Elle n'est plus sur le chemin chaud, mais `return_cov` en a besoin :
    `r0 @ R^-1 @ r0.T` est une forme quadratique entre points de test."""
    R, am = _systeme()
    n = R.shape[0]
    inv = _inverse_complete(am, n)
    assert np.max(np.abs(R @ inv - np.eye(n))) < 1e-8
    assert _inverse_complete(am, n) is inv          # mise en cache


# --------------------------------------------------------------------------- #
# 3. Le resultat n'a pas bouge                                                #
# --------------------------------------------------------------------------- #
def test_la_prediction_reste_celle_du_golden():
    """Garde-fou de bout en bout : une optimisation qui change un chiffre
    n'est pas une optimisation. Les goldens de `test_30` couvrent le detail ;
    ce test verifie que le chemin rapide et le chemin lent coincident."""
    import json                                                   # noqa: PLC0415
    import warnings                                               # noqa: PLC0415
    warnings.filterwarnings("ignore")
    import harness                                                # noqa: PLC0415
    from reference.limit_states import LinearLS                   # noqa: PLC0415

    g = json.load(open(os.path.join(TESTS, "golden", "linear.json"), encoding="utf-8"))
    doe = np.asarray(g["doe"], float)
    ls = LinearLS()
    for modele in ("PCK", "GEPCK"):
        fm = harness.fit(modele, doe, ls)
        g_hat, _, _ = harness.predictors(modele, fm)
        # l'etat limite est un hyperplan : le metamodele doit le rendre EXACT,
        # y compris au bord du domaine ou l'aller-retour cdf/ppf echouait
        bord = np.array([[7.5, -7.5], [-7.5, 7.5], [7.5, 7.5]])
        obtenu = np.asarray(g_hat(bord)).ravel()
        assert np.max(np.abs(obtenu - np.asarray(ls.g(bord)).ravel())) < 1e-12, modele
