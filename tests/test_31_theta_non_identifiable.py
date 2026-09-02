r"""Pourquoi `theta` differe d'une machine a l'autre -- la cause, mesuree.

CE FICHIER A PORTE UNE CONCLUSION FAUSSE PENDANT UN JOUR
---------------------------------------------------------
Version du 01/09/2026 : « theta n'est pas determine par les donnees, aucun
choix d'optimiseur ne le rendra reproductible ». C'etait faux, et le
raisonnement qui y menait l'etait aussi : j'avais lu `sequential` rendant
theta0 = [1, 1] comme LA PREUVE d'un plateau de vraisemblance. C'est en
realite la preuve que L'OPTIMISEUR NE PEUT PAS BOUGER. Symptome pris pour
cause, et recherche arretee trop tot.

Agnes, 02/09/2026 : « vise plus large que dernier bit, tu as un a priori ».
Elle avait raison.

LA CAUSE, INSTRUMENTEE
-----------------------
`_lib/kriging.py:kriging_optimize_theta` appelle

    minimize(J, theta0, method='L-BFGS-B', bounds=...,
             options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8})

SANS `jac`. Scipy differencie donc J lui-meme, avec son pas par defaut
`eps ~ 1.49e-08`, ABSOLU. Or theta vit sur [0.01, 100] et J passe par une
factorisation de Cholesky dont le conditionnement mesure 2.4e+09.

Derivee de J par differences finies, a theta0 = [47.6737, 21.9981] :

    pas h        dJ/dtheta_0       dJ/dtheta_1
    1.49e-08     1.0283e+00        -1.6137e+00     <-- le pas de scipy
    1.00e-06     1.5888e-02        -4.1327e-02
    1.00e-04     3.0482e-04         4.8278e-04
    1.00e-03     3.1950e-05         6.1622e-04

Une derivee honnete a un plateau. Ici la valeur au pas de scipy vaut TROIS
MILLE FOIS celle obtenue a 1e-04, et change de signe. Mesure du bruit :
perturber theta au dernier bit deplace J de 3.05e-08 ; divise par 1.49e-08,
cela donne un gradient parasite de 2.05 -- exactement l'ordre de ce que
scipy calcule.

LE GRADIENT RECU EST DONC DU BRUIT. Sa norme n'est pas une propriete de J,
c'est le plancher de bruit de J divise par le pas.

CE QUE LA TRACE MONTRE, EN CONSEQUENCE
---------------------------------------
Chaine de warm-start du mode `optimal`, `flexion/PCK` :

    etape  theta @7 threads      theta @1 thread       arret
    ii=1   [47.6737, 21.9981]    [47.6737, 21.9981]    ABNORMAL (nit=3)
    ii=3   [47.6737, 21.9981]    [47.6737, 21.9981]    ABNORMAL (nit=0)
    ii=4   [47.6737, 21.9981]    [47.6737, 21.9981]    ABNORMAL (nit=0)
    ii=5   [47.6737, 21.9981]    [ 6.5457,  6.1283]    ABNORMAL

`ABNORMAL` est `ABNORMAL_TERMINATION_IN_LNSRCH` : la recherche lineaire
echoue, ce qui est le symptome direct d'un gradient qui ne pointe pas vers
le bas. A `nit = 0`, l'optimiseur n'a fait AUCUNE iteration.

CE QUI RESTE VRAI DE LA VERSION FAUSSE
---------------------------------------
Sur le cas LINEAIRE seulement. L'etat limite y est represente EXACTEMENT par
la PCE (LOO ~ 1e-25) : il ne reste aucun residu que le krigeage puisse
expliquer, et une portee de correlation ajustee sur un residu nul n'a pas de
valeur vraie. La, theta n'est effectivement pas identifiable -- et un pas
correct ne le rend pas reproductible non plus.

Sur la flexion, en revanche, J a une VRAIE pente : 3e-04, stable de 1e-04 a
1e-01. Il n'y avait rien de fatal.

CE QUI FERMERAIT LE SUJET, ET CE QUE CELA COUTE
------------------------------------------------
Reproductibilite entre 7 et 1 thread, trois parametrisations mesurees :

    cas              defaut      pas relatif   log10(theta)
    flexion/PCK      9.71e-01    1.65e-02      1.77e-11
    flexion/GEPCK    3.54e-02    3.86e-06      0.00e+00
    linear/PCK       7.32e-01    9.99e-01      3.46e-02
    linear/GEPCK     0.00e+00    3.98e-03      5.55e-04

Dix ordres de grandeur sur `flexion/PCK`, l'identite bit-a-bit sur
`flexion/GEPCK`. Mais le changement DEPLACE DES RESULTATS -- sur
`flexion/GEPCK`, le LOO passe de 7.62e-11 a 2.12e-09 -- et le plan de
nettoyage interdit cela sans decision. Ce fichier MESURE ; il ne tranche pas.
"""

import os
import sys
import warnings

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in ("_lib", "_model"):
    _c = os.path.join(_REPO, _p)
    if _c not in sys.path:
        sys.path.insert(0, _c)
if _ICI not in sys.path:
    sys.path.insert(0, _ICI)

np = pytest.importorskip("numpy")

import harness                                              # noqa: E402


#: Le pas de differenciation que scipy emploie par defaut pour L-BFGS-B,
#: quand aucun `jac` n'est fourni. ABSOLU, donc sans rapport avec l'echelle
#: de theta.
PAS_DE_SCIPY = 1.49e-8


def _params_et_theta0(case="flexion", kind="PCK"):
    """Capture les entrees du PREMIER appel gradbased d'un vrai ajustement.

    On ne fabrique pas un probleme de test : on prend celui que l'etude
    resout reellement.
    """
    import json
    import kriging as _kr
    import fit as _fit
    from reference.limit_states import CASES

    capture = {}
    vrai = _kr.kriging_optimize_theta

    def espion(params, theta0, bornes, method="gradbased"):
        if method.lower() == "gradbased" and "params" not in capture:
            capture["params"] = params
            capture["theta0"] = np.asarray(theta0, float).copy()
        return vrai(params, theta0, bornes, method)

    _kr.kriging_optimize_theta = espion
    _fit.kriging_optimize_theta = espion
    try:
        with open(os.path.join(_REPO, "tests", "golden", case + ".json"),
                  encoding="utf-8") as fh:
            ref = json.load(fh)
        opts = {"Mode": "optimal",
                "PCE": {"Degree": list(range(1, ref["max_degree"] + 1)),
                        "Method": "LARS"}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            harness.fit(kind, np.asarray(ref["doe"]), CASES[case](), opts=opts)
    finally:
        _kr.kriging_optimize_theta = vrai
        _fit.kriging_optimize_theta = vrai

    if "params" not in capture:
        pytest.skip("aucun appel gradbased capture")
    return capture["params"], capture["theta0"]


# --------------------------------------------------------------------- #
# 1. LE FAIT CENTRAL : le gradient recu est du bruit
# --------------------------------------------------------------------- #
def test_au_pas_de_scipy_le_gradient_est_du_bruit():
    """La derivee au pas de scipy n'a rien a voir avec la vraie pente.

    Une derivee saine a un PLATEAU en fonction du pas. On exige ici que la
    valeur au pas de scipy soit au moins cent fois celle obtenue a 1e-4 --
    mesure le 02/09/2026 : facteur 3 372 sur la premiere composante.

    Si ce temoin tombe un jour parce que le rapport devient PETIT, c'est une
    bonne nouvelle : le gradient serait devenu exploitable, et tout ce
    fichier serait a relire.
    """
    import kriging as _kr
    params, theta0 = _params_et_theta0()

    def J(t):
        return _kr.uq_Kriging_eval_J_of_theta_ML(np.asarray(t, float), params)

    J0 = J(theta0)

    def derivee(h, k):
        tp = np.asarray(theta0, float).copy()
        tp[k] += h
        return (J(tp) - J0) / h

    for k in range(len(theta0)):
        au_pas_scipy = abs(derivee(PAS_DE_SCIPY, k))
        vraie_pente = abs(derivee(1e-4, k))
        assert au_pas_scipy > 100 * vraie_pente, (
            "composante %d : |dJ| = %.3e au pas de scipy contre %.3e a 1e-4, "
            "soit un facteur %.0f. Attendu : au moins 100. Un rapport devenu "
            "petit voudrait dire que le gradient est redevenu exploitable."
            % (k, au_pas_scipy, vraie_pente, au_pas_scipy / max(vraie_pente, 1e-300)))


def test_le_bruit_de_J_explique_ce_gradient():
    """Le chiffre qui ferme le raisonnement.

    Perturber theta au dernier bit deplace J de ~3e-08. Divise par le pas de
    scipy, cela donne un gradient parasite de l'ordre de 2 -- c'est-a-dire
    l'ordre de ce que scipy calcule. Le « gradient » n'est donc pas une
    propriete de J : c'est son plancher de bruit divise par le pas.
    """
    import kriging as _kr
    params, theta0 = _params_et_theta0()

    valeurs = [_kr.uq_Kriging_eval_J_of_theta_ML(
        np.asarray(theta0, float) * (1.0 + k * np.finfo(float).eps), params)
        for k in range(12)]
    bruit = max(valeurs) - min(valeurs)
    gradient_parasite = bruit / PAS_DE_SCIPY

    assert bruit > 0.0, (
        "J ne bouge plus du tout sous une perturbation au dernier bit : le "
        "raisonnement de ce fichier suppose un bruit d'evaluation non nul.")
    assert gradient_parasite > 1e-2, (
        "gradient parasite %.3e, trop petit pour expliquer les valeurs "
        "observees (~1). Le bruit mesure vaut %.3e." % (gradient_parasite, bruit))


# --------------------------------------------------------------------- #
# 2. LA CONSEQUENCE : l'optimiseur n'avance pas
# --------------------------------------------------------------------- #
def test_l_optimiseur_echoue_en_recherche_lineaire():
    """`ABNORMAL_TERMINATION_IN_LNSRCH` -- le symptome direct.

    Sur un vrai ajustement, une part notable des appels a L-BFGS-B se
    termine par un echec de recherche lineaire, et certains sans avoir fait
    UNE SEULE iteration. Un optimiseur qui recevrait un vrai gradient ne
    ferait pas cela.
    """
    import json
    import kriging as _kr
    import fit as _fit
    from scipy.optimize import minimize
    from reference.limit_states import CASES

    arrets = []
    vrai = _kr.kriging_optimize_theta

    def espion(params, theta0, bornes, method="gradbased"):
        if method.lower() != "gradbased":
            return vrai(params, theta0, bornes, method)
        lb, ub = bornes[0, :], bornes[1, :]

        def J(t):
            return _kr.uq_Kriging_eval_J_of_theta_ML(t, params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(J, theta0, method="L-BFGS-B",
                           bounds=list(zip(lb, ub)),
                           options={"maxiter": 400, "ftol": 1e-12,
                                    "gtol": 1e-8})
        arrets.append((str(res.message), int(res.nit)))
        return res.x, res.fun, int(res.success)

    _kr.kriging_optimize_theta = espion
    _fit.kriging_optimize_theta = espion
    try:
        with open(os.path.join(_REPO, "tests", "golden", "flexion.json"),
                  encoding="utf-8") as fh:
            ref = json.load(fh)
        opts = {"Mode": "optimal",
                "PCE": {"Degree": list(range(1, ref["max_degree"] + 1)),
                        "Method": "LARS"}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            harness.fit("PCK", np.asarray(ref["doe"]), CASES["flexion"](),
                        opts=opts)
    finally:
        _kr.kriging_optimize_theta = vrai
        _fit.kriging_optimize_theta = vrai

    anormaux = [m for m, _ in arrets if "ABNORMAL" in m.upper()]
    immobiles = [n for _, n in arrets if n == 0]
    assert anormaux, (
        "plus aucun echec de recherche lineaire sur %d appels. Si le gradient "
        "a ete corrige, ce fichier est a relire entierement." % len(arrets))
    assert immobiles or len(anormaux) >= 2, (
        "un seul symptome sur %d appels : le diagnostic de ce fichier "
        "s'appuie sur leur recurrence." % len(arrets))


# --------------------------------------------------------------------- #
# 3. CE QUI RESTE VRAI DE LA VERSION FAUSSE -- le cas degenere
# --------------------------------------------------------------------- #
def test_sur_le_cas_lineaire_il_n_y_a_vraiment_rien_a_identifier():
    """La PCE represente l'etat limite EXACTEMENT : LOO ~ 1e-25.

    Il ne reste aucun residu que le krigeage puisse expliquer. Une portee de
    correlation ajustee sur un residu nul n'a pas de valeur vraie -- et,
    mesure le 02/09, un pas de differenciation correct ne rend pas ce cas
    reproductible non plus (3.46e-02 en log-theta, contre 1.77e-11 sur la
    flexion). C'est la SEULE part de la conclusion du 01/09 qui survive.
    """
    import json
    from reference.limit_states import CASES
    with open(os.path.join(_REPO, "tests", "golden", "linear.json"),
              encoding="utf-8") as fh:
        ref = json.load(fh)
    opts = {"Mode": "sequential",
            "PCE": {"Degree": list(range(1, ref["max_degree"] + 1)),
                    "Method": "LARS"}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm = harness.fit("PCK", np.asarray(ref["doe"]), CASES["linear"](),
                         opts=opts)
    loo = float(fm["Error"][0]["LOO"])
    assert loo < 1e-20, (
        "LOO = %.3e sur le cas lineaire : la PCE ne le represente plus "
        "exactement, et l'argument de degenerescence tombe." % loo)


# --------------------------------------------------------------------- #
# 4. LE DEFAUT EST D'ORIGINE -- et le nettoyage l'a REVELE, pas cree
# --------------------------------------------------------------------- #
#: Ce que `branche3.py` portait a `8f6e229~1`, AVANT tout nettoyage. Verifie
#: ligne a ligne le 02/09/2026 :
#:
#:   IDENTIQUE  `minimize(J, theta0, method='L-BFGS-B', bounds=...,
#:              options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8})`,
#:              SANS `jac`.
#:   IDENTIQUE  bornes [[0.01]*M, [100]*M], theta0 = moyenne geometrique.
#:   CHANGE     la pepite : `'Nugget': 0.0` a l'origine, 1e-8 depuis la
#:              phase 6 -- ajoutee pour corriger les defauts 2 et 3, avec
#:              des criteres chiffres d'avance et tenus.
PEPITE_D_ORIGINE = 0.0

#: Mesure du 02/09/2026, mode `optimal`, appels a L-BFGS-B rendant leur point
#: de depart INCHANGE :
#:
#:     cas              pepite 0.0      pepite 1e-8
#:     flexion/PCK      8/9             3/9
#:     flexion/GEPCK    9/9             2/9
#:     linear/PCK       3/3             0/3
#:     linear/GEPCK     3/3             3/3
#:
#: 23 appels sur 24 immobiles a l'origine, contre 8 sur 24 aujourd'hui. Et
#: le conditionnement de R passe de 3.75e+14 a 2.40e+09, le bruit de J de
#: 1.04e-03 a 3.05e-08.
IMMOBILES_A_L_ORIGINE = 23
IMMOBILES_SUR = 24


def _immobiles(pepite, case="flexion", kind="PCK"):
    """Combien d'appels a L-BFGS-B rendent leur point de depart inchange."""
    import json
    import kernels as _kern
    import fit as _fit
    import api as _api
    import kriging as _kr
    from scipy.optimize import minimize
    from reference.limit_states import CASES

    anciens = (_kern.PEPITE_PAR_DEFAUT, _fit.PEPITE_PAR_DEFAUT,
               _api.PEPITE_PAR_DEFAUT)
    vrai = _kr.kriging_optimize_theta
    etats = []

    def espion(params, theta0, bornes, method="gradbased"):
        if method.lower() != "gradbased":
            return vrai(params, theta0, bornes, method)
        theta0 = np.asarray(theta0, float)
        lb, ub = bornes[0, :], bornes[1, :]

        def J(t):
            return _kr.uq_Kriging_eval_J_of_theta_ML(t, params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(J, theta0, method="L-BFGS-B",
                           bounds=list(zip(lb, ub)),
                           options={"maxiter": 400, "ftol": 1e-12,
                                    "gtol": 1e-8})
        etats.append(float(np.max(np.abs(res.x - theta0)
                                  / np.maximum(np.abs(theta0), 1e-30))))
        return res.x, res.fun, int(res.success)

    _kern.PEPITE_PAR_DEFAUT = _fit.PEPITE_PAR_DEFAUT = pepite
    _api.PEPITE_PAR_DEFAUT = pepite
    _kr.kriging_optimize_theta = espion
    _fit.kriging_optimize_theta = espion
    try:
        with open(os.path.join(_REPO, "tests", "golden", case + ".json"),
                  encoding="utf-8") as fh:
            ref = json.load(fh)
        opts = {"Mode": "optimal",
                "PCE": {"Degree": list(range(1, ref["max_degree"] + 1)),
                        "Method": "LARS"}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            harness.fit(kind, np.asarray(ref["doe"]), CASES[case](), opts=opts)
    finally:
        _kern.PEPITE_PAR_DEFAUT, _fit.PEPITE_PAR_DEFAUT, _api.PEPITE_PAR_DEFAUT = anciens
        _kr.kriging_optimize_theta = vrai
        _fit.kriging_optimize_theta = vrai
    return sum(1 for b in etats if b < 1e-12), len(etats)


def test_a_l_origine_l_optimiseur_de_theta_ne_tournait_PAS():
    """LE fait qui repond a « le bug est-il d'origine ? ».

    Il l'est -- le code de l'optimiseur est identique au caractere pres. Mais
    sa consequence etait INVISIBLE : avec la pepite d'origine (0.0), R est
    conditionne a 3.75e+14, le bruit de J vaut 1.04e-03, le gradient au pas
    de scipy vaut 7.8e+04 contre une pente de 8.4 -- et L-BFGS-B echoue
    IMMEDIATEMENT a chaque appel.

    Le mode `optimal` etait donc inerte : theta valait la sortie de
    `differential_evolution`, traversant la chaine de warm-start sans etre
    touchee. Deterministe, donc reproductible -- et jamais optimisee.

    La pepite ajoutee en phase 6 pour corriger les defauts 2 et 3 a
    partiellement REVEILLE l'optimiseur. C'est la que
    l'irreproductibilite est apparue : le nettoyage n'a pas cree le defaut,
    il a rendu vivant ce qui etait mort.
    """
    # UN CONTRASTE, PAS UN COMPTE. Le nombre exact d'appels immobiles depend
    # de l'arithmetique -- ce dont ce fichier parle precisement. Mesures du
    # 02/09/2026, meme code, pepite 0.0 :
    #
    #     poste de reference (windows)   8 immobiles / 9
    #     runner ubuntu py3.10           7 / 9
    #     runner ubuntu py3.13           6 / 9
    #
    # Deux versions de ce temoin ont tombe en integration continue pour avoir
    # fige un seuil ABSOLU -- d'abord `>= total - 1`, puis `>= 0.7 * total`.
    # La propriete a figer n'est pas un compte : c'est que l'optimiseur bouge
    # STRICTEMENT MOINS avec la pepite d'origine qu'avec celle d'aujourd'hui.
    # Cela se compare sur la MEME machine, et ne depend donc d'aucun seuil.
    imm_origine, total = _immobiles(PEPITE_D_ORIGINE)
    imm_actuelle, total2 = _immobiles(1e-8)
    assert total == total2 > 0, (total, total2)
    assert imm_origine > imm_actuelle, (
        "avec la pepite d'origine (0.0), %d appels sur %d sont immobiles ; "
        "avec la pepite actuelle (1e-8), %d sur %d. L'optimiseur devrait "
        "bouger MOINS a l'origine -- c'est tout l'argument de ce fichier : la "
        "pepite ajoutee en phase 6 a divise le bruit de J par cinq ordres de "
        "grandeur et a reveille un optimiseur qui ne demarrait pas."
        % (imm_origine, total, imm_actuelle, total2))


def test_aujourd_hui_l_optimiseur_bouge_vraiment():
    """La contrepartie : avec la pepite actuelle, L-BFGS-B avance.

    C'est un PROGRES -- un optimiseur qui optimise vaut mieux qu'un
    optimiseur inerte -- et c'est aussi ce qui a rendu le resultat dependant
    de la machine, puisqu'il avance en suivant un gradient qui est du bruit.
    Les deux faits vont ensemble et doivent etre lus ensemble.
    """
    immobiles, total = _immobiles(1e-8)
    assert total > 0
    assert immobiles < total, (
        "les %d appels sont immobiles avec la pepite actuelle : l'optimiseur "
        "serait redevenu inerte comme a l'origine." % total)
