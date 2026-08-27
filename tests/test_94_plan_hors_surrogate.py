r"""Le PLAN, le SURROGATE, la GRILLE et la RESTITUTION sont quatre actions.

CE QUI ETAIT MELANGE -- 27/08/2026
-----------------------------------
Trois fonctions faisaient chacune deux choses, dont une coutait des appels
solveur invisibles a la lecture :

1. `init_g_ot` -- « initialise g_ot » -- portait SEPT fois la meme ligne
   cachee, une par branche de surrogate :

       if xt is None: xt, yt, all_grad = build_DOE()

   Toute fonction capable d'atteindre `init_g_ot` pouvait donc lancer le plan
   d'experiences entier. C'est par la que `print_globalplanche_EFF`, une
   FIGURE, atteignait `run_DOE_parallel`.

2. `print_results` imprimait les resultats FORM (gratuit) PUIS evaluait l'etat
   limite exact en deux points pour l'erreur FOSM (deux SOCP, soit un quart
   d'heure sur le Moulin Blanc) -- sous un nom qui dit « imprime ».

3. `print_3D_HF` calculait la grille haute fidelite (n_grid_hf^2 appels : 225
   pour une grille 15x15, soit 29 heures) puis la dessinait.

Ces tests interdisent que les coutures se refassent.
"""

import ast
import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _d in (os.path.join(_REPO, "_config"), os.path.join(_REPO, "_etapes")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# La couche etudes n'est pas installee partout : sans cela, la collecte de
# TOUTE la suite s'interrompt sur un poste minimal (test_05).
ot = pytest.importorskip("openturns")

import figurer                                       # noqa: E402
import schema                                        # noqa: E402

SCRIPTS = ["pure_flexion/AC3_pure_flexion.py", "Moulinblanc/AC3_moulinblanc.py"]


def _fonctions(script):
    """Les fonctions definies dans le bloc `__main__` d'un script d'etude."""
    chemin = os.path.join(_REPO, script)
    with open(chemin, encoding="utf-8", errors="replace") as fh:
        arbre = ast.parse(fh.read(), filename=chemin)
    main = [n for n in arbre.body if isinstance(n, ast.If)][0]
    return {n.name: n for n in main.body if isinstance(n, ast.FunctionDef)}


def _appelle(noeud, cible):
    for x in ast.walk(noeud):
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) \
                and x.func.id == cible:
            return True
    return False


# --------------------------------------------------------------------- #
# 1. ajuster un surrogate ne construit plus le plan
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("script", SCRIPTS)
def test_init_g_ot_ne_construit_plus_le_plan(script):
    """La ligne cachee sept fois est partie. Elle ne doit pas revenir : c'est
    ELLE qui rendait une figure capable de lancer n0 appels solveur."""
    fn = _fonctions(script)["init_g_ot"]
    assert not _appelle(fn, "build_DOE"), (
        "%s : `init_g_ot` appelle a nouveau `build_DOE`. Ajuster un surrogate "
        "et construire un plan d'experiences sont deux actions ; la seconde "
        "coute n0 appels solveur et doit rester visible chez l'appelant."
        % script)


@pytest.mark.parametrize("script", SCRIPTS)
def test_init_g_ot_refuse_un_plan_absent_au_lieu_de_le_fabriquer(script):
    """Le garde-fou doit LEVER, pas retomber silencieusement sur build_DOE."""
    fn = _fonctions(script)["init_g_ot"]
    leve = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
    assert leve, "%s : `init_g_ot` doit refuser xt=None explicitement" % script


@pytest.mark.parametrize("script", SCRIPTS)
def test_le_plan_est_construit_une_seule_fois_et_en_clair(script):
    """`build_DOE()` doit apparaitre au niveau du programme principal, pas
    seulement au fond d'une fonction."""
    chemin = os.path.join(_REPO, script)
    with open(chemin, encoding="utf-8", errors="replace") as fh:
        arbre = ast.parse(fh.read(), filename=chemin)
    main = [n for n in arbre.body if isinstance(n, ast.If)][0]
    direct = [n for n in main.body if not isinstance(n, ast.FunctionDef)]
    assert any(_appelle(n, "build_DOE") for n in direct), (
        "%s : aucun appel a `build_DOE` au niveau du programme principal. "
        "Le plan initial doit etre construit la, une seule fois." % script)


# --------------------------------------------------------------------- #
# 2. le resume FORM et l'erreur FOSM sont separes
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("script", SCRIPTS)
def test_l_erreur_FOSM_est_une_action_a_part(script):
    """Elle coute deux appels solveur : elle porte donc un nom qui le dit, et
    un interrupteur."""
    fns = _fonctions(script)
    assert "erreur_FOSM" in fns, "%s : `erreur_FOSM` attendue" % script
    assert "print_results" not in fns, (
        "%s : `print_results` est revenue. Elle melangeait un affichage "
        "gratuit et deux evaluations de l'etat limite." % script)
    with open(os.path.join(_REPO, script), encoding="utf-8",
              errors="replace") as fh:
        src = fh.read()
    assert "CFG.erreur_fosm" in src, (
        "%s : l'erreur FOSM doit etre desactivable -- deux SOCP par mode." % script)


def test_le_cout_de_l_erreur_FOSM_est_declare():
    assert "erreur_fosm" in schema.COUTE_DES_APPELS_SOLVEUR
    assert "erreur_fosm" in schema.CATEGORIES
    from dataclasses import fields
    defaut = {f.name: f.default for f in fields(schema.Configuration)}
    assert defaut["erreur_fosm"] is True, (
        "defaut historique : la mesure existait, elle etait juste invisible")


def test_le_resume_FORM_n_evalue_rien():
    """Un faux resultat FORM suffit : si la fonction touchait le solveur, elle
    ne pourrait pas s'en contenter."""
    ecrits = []

    class _Opt:
        def getIterationNumber(self):
            return 7

    class _Result:
        def getStandardSpaceDesignPoint(self):
            return ot.Point([-2.797, -3.841])

        def getOptimizationResult(self):
            return _Opt()

        def getImportanceFactors(self):
            return ot.Point([0.35, 0.65])

        def getHasoferReliabilityIndex(self):
            return 4.751643

        def getEventProbability(self):
            return 1.009e-06

    dist = ot.JointDistribution([ot.Normal(235.0, 30.15),
                                 ot.Normal(30.0, 4.5)])
    u = figurer.resume_FORM(_Result(), dist, ["fy", "fc"], ecrire=ecrits.append)
    assert list(u) == [-2.797, -3.841]
    texte = "\n".join(ecrits)
    assert "beta         = 4.7516" in texte
    assert "n_iter FORM  = 7" in texte
    # x* doit passer par la transformation isoprobabiliste, pas par un
    # aller-retour cdf/ppf : mu + u*sigma pour une marginale normale.
    ligne = next(l for l in ecrits if l.startswith("fy*"))
    assert float(ligne.split("=")[1]) == pytest.approx(235.0 - 2.797 * 30.15,
                                                       abs=1e-4)


# --------------------------------------------------------------------- #
# 3. le domaine physique : une valeur de retour, pas seulement du texte
# --------------------------------------------------------------------- #
def test_le_domaine_physique_est_celui_qu_on_evalue():
    """Les bornes rendues doivent coincider avec T_inv -- la transformation
    que l'evaluation de l'etat limite emploie reellement.

    La voie naive `computeQuantile(Normal().computeCDF(u))` derive de 11,3 MPa
    a u = +8,5 sur Normal(235 ; 30,15). C'est le defaut de la phase 7, et il a
    failli etre reintroduit ici le 26/08/2026.
    """
    dist = ot.JointDistribution([ot.Normal(235.0, 30.15),
                                 ot.Normal(500.0, 30.0)])
    extremes = figurer.tracer_domaine_physique(
        dist, ["fy1", "fy2"], -6.0, 6.0, ecrire=lambda _m: None)
    T_inv = dist.getInverseIsoProbabilisticTransformation()
    bas, haut = T_inv(ot.Point([-6.0, -6.0])), T_inv(ot.Point([6.0, 6.0]))
    for i, (lo, hi) in enumerate(extremes):
        assert lo == pytest.approx(float(bas[i]), abs=1e-9)
        assert hi == pytest.approx(float(haut[i]), abs=1e-9)


def test_le_domaine_physique_alerte_sur_le_rapport_aux_coins():
    """C'est le rapport a un COIN qui a tue le run du 26/08, pas la valeur des
    bornes. A 52, Digital Structure a termine le processus."""
    ecrits = []
    dist = ot.JointDistribution([ot.Normal(235.0, 30.15),
                                 ot.Normal(500.0, 30.0)])
    figurer.tracer_domaine_physique(dist, ["fy1", "fy2"], -7.5, 7.5,
                                    ecrire=ecrits.append)
    texte = "\n".join(ecrits)
    assert "rapport le plus defavorable aux coins" in texte
    # bas = 235 - 7.5*30.15 = 8.87 ; haut = 500 + 7.5*30 = 725 -> 81.7
    assert "81.7" in texte
    assert "ATTENTION" in texte


def test_un_domaine_sain_ne_declenche_aucune_alerte():
    ecrits = []
    dist = ot.JointDistribution([ot.Normal(235.0, 5.0), ot.Normal(240.0, 5.0)])
    figurer.tracer_domaine_physique(dist, ["fy1", "fy2"], -3.0, 3.0,
                                    ecrire=ecrits.append)
    assert "ATTENTION" not in "\n".join(ecrits)


# --------------------------------------------------------------------- #
# 4. la grille est une action, pas une figure
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("script", SCRIPTS)
def test_la_grille_3D_est_separee_de_son_dessin(script):
    fns = _fonctions(script)
    assert "grille_3D" in fns, (
        "%s : la grille 3D coute n_grid_hf^2 appels ; elle porte un nom "
        "d'action, pas un nom de trace." % script)
    args = [a.arg for a in fns["print_3D_HF"].args.args]
    assert args == ["U1_hf", "U2_hf", "Z"], (
        "%s : `print_3D_HF` doit recevoir la surface deja calculee (%s)"
        % (script, args))
    assert not _appelle(fns["print_3D_HF"], "run_HF")
    assert _appelle(fns["grille_3D"], "run_HF"), (
        "%s : c'est `grille_3D` qui paie les appels -- si elle n'en fait plus, "
        "verifier que le calcul n'est pas reparti ailleurs." % script)


# --------------------------------------------------------------------- #
# 5. les deux etudes partagent le meme module de restitution
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("script", SCRIPTS)
def test_les_deux_etudes_partagent_la_restitution(script):
    """94,2 % des lignes communes aux deux AC sont identiques au caractere
    pres. Chaque fonction remontee dans un module est une divergence future
    qui n'aura pas lieu."""
    with open(os.path.join(_REPO, script), encoding="utf-8",
              errors="replace") as fh:
        src = fh.read()
    assert "import figurer as _figurer" in src
    assert "_figurer.tracer_domaine_physique(" in src
    assert "_figurer.resume_FORM(" in src


def test_figurer_n_importe_pas_de_solveur():
    """Le module de restitution ne doit dependre ni du solveur, ni de la
    configuration : on doit pouvoir le charger sans licence et sans etude."""
    assert not hasattr(figurer, "run_HF")
    source = open(os.path.join(_REPO, "_etapes", "figurer.py"),
                  encoding="utf-8").read()
    for interdit in ("import fabrique", "import schema", "digital_structure",
                     "STRAINS"):
        assert interdit not in source, (
            "figurer.py importe %r : il doit rester chargeable sans licence "
            "et sans fichier d'etude." % interdit)
