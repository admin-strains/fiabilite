"""
Defauts connus, ecrits comme des specifications executables.

CE FICHIER A CHANGE DE ROLE, ET C'EST UNE BONNE NOUVELLE
---------------------------------------------------------
Il a ete ecrit comme un registre de defauts OUVERTS : chaque test decrivait
le comportement ATTENDU -- pas le comportement d'alors -- et portait
`xfail(strict=True)`. Le jour ou quelqu'un corrigeait le defaut, le test
passait, donc `xpass`, donc ECHEC : il fallait retirer le marqueur et acter
la correction. Aucun defaut ne pouvait etre corrige en silence.

Le mecanisme a fonctionne jusqu'au bout : **il ne reste aucun `xfail` ici**,
les douze tests passent. Ce fichier est devenu un registre de defauts FERMES,
et chacun garde contre le retour du sien.

    defaut 1     le metamodele ne s'evaluait pas sur son propre plan
    defauts 2/3  GEPCK n'interpolait pas -- 3,0e-03 au plan, contre 2,8e-09
                 aujourd'hui, apres la pepite (phase 6) et le gradient
                 analytique de la vraisemblance (02/09)
    defaut 5     une conversion tableau->scalaire depreciee par numpy

Le seul defaut de cette semaine dont la consequence etait des CHIFFRES FAUX
plutot qu'un cout -- `patch_params` ignorant en silence un parametre absent
du modele -- est garde ailleurs, dans `test_125_patch_params.py`.

Ne rien remettre ici sans : symptome reproductible, localisation
fichier:ligne, et effet concret sur la production. Et si un defaut nouveau y
entre, il entre en `xfail(strict=True)` -- c'est ce marqueur qui interdit de
le corriger sans le dire.
"""

import numpy as np
import pytest

import harness


def test_gepck_sevalue_sur_son_propre_doe(fitted, doe24, flexion_ls):
    """DEFAUT 1 -- CORRIGE le 26/08/2026 (phase 6).

    `kernels.uq_eval_global_Kernel` choisissait entre R_tilde
    (N(M+1), N(M+1)) et r0_tilde (N, N(M+1)) en INSPECTANT le contenu des
    tableaux : `isGram = (n1 == n2) and np.array_equal(X1, X2)`. Evaluer le
    metamodele exactement sur son propre plan d'experiences faisait donc
    basculer la fonction sur la mauvaise branche, et `predict.py` levait
    « operands could not be broadcast together with shapes (N,) (N*(M+1),) ».

    Cela arrive pour de bon : une grille EFF qui passe par un point du DOE,
    une relecture de cache, une verification d'interpolation. Le
    contournement etait involontaire -- les points tombaient rarement pile
    sur le DOE.

    L'appelant DIT desormais ce qu'il veut (`options['IsGram']`). Ce test ne
    juge que cela : l'evaluation aboutit, avec la bonne forme. La PRECISION
    du resultat reste le defaut 2, teste juste en dessous -- les deux se
    recouvraient jusqu'ici, ce qui rendait impossible de constater la
    correction de l'un sans l'autre.
    """
    g_hat, _, _ = harness.predictors('GEPCK', fitted['GEPCK'])
    mu = np.asarray(g_hat(doe24)).ravel()
    assert mu.shape == (doe24.shape[0],), \
        "forme %s : le dispatch Gram/non-Gram est reparti de travers" % (mu.shape,)
    assert np.all(np.isfinite(mu))


def test_pck_sevalue_aussi_sur_son_propre_doe(fitted, doe24):
    """Le pendant PCK. Le noyau simple ne changeait pas de FORME selon la
    branche, donc il ne plantait pas -- mais il ajoutait l'identite et la
    pepite sans qu'on le lui demande. Le drapeau explicite vaut pour les deux."""
    g_hat, _, _ = harness.predictors('PCK', fitted['PCK'])
    mu = np.asarray(g_hat(doe24)).ravel()
    assert mu.shape == (doe24.shape[0],)
    assert np.all(np.isfinite(mu))


@pytest.mark.parametrize("modele", ["GEPCK", "PCK"])
def test_un_gram_demande_sur_deux_jeux_distincts_est_refuse(modele):
    """Corollaire du drapeau explicite : mieux vaut une erreur nette qu'une
    matrice silencieusement fausse."""
    import numpy as _np                                       # noqa: PLC0415
    from kernels import uq_eval_Kernel, uq_eval_global_Kernel  # noqa: PLC0415
    evalR = uq_eval_global_Kernel if modele == "GEPCK" else uq_eval_Kernel
    X1 = _np.array([[0.0, 0.0], [1.0, 1.0]])
    X2 = _np.array([[0.0, 0.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="IsGram"):
        evalR(X1, X2, _np.array([1.0, 1.0]),
              {"Family": "matern-5_2", "Nugget": 0.0, "IsGram": True,
               "Type": "separable", "Isotropic": False})


def test_gepck_interpole_son_doe(fitted, doe24, flexion_ls):
    """DEFAUTS 2 et 3 -- CORRIGES le 26/08/2026 (phase 6).

    Le krigeage doit interpoler ses points d'apprentissage. Il n'y arrivait
    qu'a 2,96e-03, contre 5,31e-07 pour PCK -- quatre ordres de grandeur
    d'ecart, alors que GEPCK dispose des gradients EN PLUS.

    La piste enregistree etait la bonne : le conditionnement de R_tilde,
    mesure a 1,7e15 sur ce plan, au bord des 4,5e15 que la double precision
    autorise. La cause de ce conditionnement, elle, n'etait pas celle qu'on
    croyait : ce n'est pas l'echelle des blocs de derivees -- l'equilibrage
    de Jacobi ne gagne rien (1,64e15 -> 1,63e15) -- mais l'absence de pepite,
    qui laissait le maximum de vraisemblance pousser les longueurs de
    correlation vers le haut jusqu'a rendre la matrice singuliere.

    Avec `kernels.PEPITE_PAR_DEFAUT = 1e-8`, l'interpolation tombe a 2,6e-09
    et l'erreur sur beta de 1,2982 % a 0,0072 % -- soit MIEUX que PCK.
    Mesure complete : `python tools/mesure_pepite.py`.
    """
    g_hat, _, _ = harness.predictors('GEPCK', fitted['GEPCK'])
    err = np.abs(g_hat(doe24 + 1e-9) - flexion_ls.g(doe24)).max()
    assert err < 1e-6, f'erreur d interpolation GEPCK = {err:.3e}'


def test_gepck_interpole_au_moins_aussi_bien_que_pck(fitted, doe24, flexion_ls):
    """Le critere que le plan de nettoyage fixait d'avance : « au moins aussi
    bon que PCK ». GEPCK voit la fonction ET ses derivees ; qu'il fasse pire
    etait le signe qu'il resolvait mal, pas qu'il en savait moins."""
    ref = flexion_ls.g(doe24)
    err = {}
    for modele in ('PCK', 'GEPCK'):
        g_hat, _, _ = harness.predictors(modele, fitted[modele])
        err[modele] = float(np.abs(g_hat(doe24 + 1e-9) - ref).max())
    assert err['GEPCK'] <= 1e-6, err
    assert err['PCK'] <= 1e-6, err


def test_la_pepite_par_defaut_n_est_pas_revenue_a_zero():
    """Une pepite nulle rend l'interpolation EXACTE en theorie et fausse en
    pratique. Sur un etat limite lineaire de 40 points -- un hyperplan que le
    metamodele contient pourtant exactement -- GEPCK rendait beta = 19,8 au
    lieu de 3,5. Et l'erreur EMPIRE quand le plan grandit, alors que la
    boucle d'enrichissement EFF, elle, ajoute des points."""
    from kernels import PEPITE_PAR_DEFAUT                      # noqa: PLC0415
    assert PEPITE_PAR_DEFAUT > 0.0
    assert 1e-10 <= PEPITE_PAR_DEFAUT <= 1e-6, (
        "hors de la plage mesuree comme sure (cf. tools/mesure_pepite.py) : "
        "en dessous de 1e-10 le conditionnement redevient hors de portee, "
        "au-dessus de 1e-6 le biais d'interpolation devient inutilement grand")


@pytest.mark.defect
def test_pck_interpole_son_doe(fitted, doe24, flexion_ls):
    """Temoin : le meme controle passe pour PCK. Sert a prouver que le test
    ci-dessus mesure un defaut de GEPCK et non un artefact du harness."""
    g_hat, _, _ = harness.predictors('PCK', fitted['PCK'])
    err = np.abs(g_hat(doe24 + 1e-9) - flexion_ls.g(doe24)).max()
    assert err < 1e-5, f'erreur d interpolation PCK = {err:.3e}'


@pytest.mark.defect
def test_gradient_gepck_est_juste(fitted, flexion_ls):
    """
    Contre-expertise du gradient analytique GEPCK.

    A ne pas confondre avec un test naif : l'ecart au FD AUGMENTE quand le pas
    diminue (9.7e-07 a h=1e-3, 4.1e-03 a h=1e-7), signature d'un bruit
    d'arrondi de la prediction et non d'un gradient faux. On compare donc au
    pas ou le FD est le plus fiable, et on verifie la decroissance.
    """
    g_hat, grad_ana, _ = harness.predictors('GEPCK', fitted['GEPCK'])
    U = np.array([[1.0, -1.0], [-2.0, 0.5], [0.25, 2.75]])
    G = grad_ana(U)

    def fd(h):
        F = np.zeros_like(G)
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            F[:, j] = (g_hat(U + e) - g_hat(U - e)) / (2 * h)
        return np.abs(G - F).max()

    assert fd(1e-3) < 1e-5, 'gradient analytique GEPCK faux au pas le plus fiable'
    assert fd(1e-3) < fd(1e-6), \
        'le FD ne se degrade plus quand h diminue : reexaminer le diagnostic'


@pytest.mark.parametrize("modele", ["GEPCK", "PCK"])
def test_pas_de_conversion_tableau_vers_scalaire_depreciee(doe24, flexion_ls, modele):
    """DEFAUT 5 -- CORRIGE le 26/08/2026 (phase 6).

    `kriging.py` convertissait un tableau 1x1 en scalaire par `float()`, en
    trois endroits (`(z.T @ z) / N` deux fois, `(z.T @ Rinv @ z) / N` une
    fois), plus un quatrieme dans `lars.py` (`float(C @ Ainv @ B)`). NumPy le
    deprecie depuis 1.25 et l'annonce comme une ERREUR dans une version
    future : le code aurait cesse de fonctionner sans que rien n'ait change
    dans le depot -- la pire facon de casser.

    Corrige par `.item()`, qui dit explicitement « ce tableau ne contient
    qu'une valeur, je la veux ». Aucun changement numerique : verifie par la
    baseline, a zero ecart.
    """
    import warnings
    import harness
    with warnings.catch_warnings(record=True) as captures:
        warnings.simplefilter('always', DeprecationWarning)
        harness.fit(modele, doe24, flexion_ls)
    ndim = [str(w.message) for w in captures
            if 'ndim > 0 to a scalar' in str(w.message)]
    assert not ndim, f'{len(ndim)} conversion(s) depreciee(s) : {ndim[0][:120]}'


def test_aucune_conversion_depreciee_ne_subsiste_dans_la_librairie():
    """Le pendant statique : la meme faute ailleurs se voit a la lecture.

    Un `float()` applique a un produit matriciel est suspect par nature -- le
    resultat est un tableau, pas un nombre. Les cas legitimes (produit de deux
    vecteurs 1-D, `np.trace`, `np.mean`) sont recenses ici.
    """
    import io
    import os
    import re
    LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib")
    #: (fichier, extrait) deja verifies comme rendant un scalaire 0-D
    TOLERES = {
        ("_parallel_is.py", "float(_W['ustar'] @ _W['ustar'])"),   # 1-D @ 1-D
        ("_parallel_is.py", "float(ustar @ ustar)"),               # 1-D @ 1-D
        ("lars.py", "float(np.trace(np.linalg.pinv(Psi.T @ Psi)))"),
        ("lars.py", "float(Psi[:, idx] @ Psi[:, idx])"),           # 1-D @ 1-D
        # s = np.sign(cj[a_arr]) est 1-D (k,), M_gram est (k, k) :
        # s @ M_gram @ s rend un scalaire 0-D. Confirme par les formes des
        # lignes suivantes -- w vaut (k,) et u vaut (N,).
        ("lars.py", "float(1.0 / np.sqrt(s @ M_gram @ s)"),
    }
    suspects = []
    for nom in sorted(os.listdir(LIB)):
        if not nom.endswith(".py"):
            continue
        for ligne in io.open(os.path.join(LIB, nom), encoding="utf-8",
                             errors="replace").read().splitlines():
            nu = ligne.split("#")[0].strip()
            for m in re.finditer(r"float\([^)]*@[^)]*\)", nu):
                extrait = m.group(0)
                if ".item()" in nu:
                    continue
                if any(nom == f and extrait.startswith(e[:20]) for f, e in TOLERES):
                    continue
                suspects.append("%s : %s" % (nom, extrait))
    assert not suspects, (
        "conversion(s) tableau->scalaire a verifier :\n  " + "\n  ".join(suspects)
        + "\nSi le resultat est bien 0-D, l'ajouter a TOLERES avec la raison ; "
          "sinon utiliser .item().")
