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
    # Depuis le 27/08/2026 le calcul est dans `_etapes/grille.py` : l'etude
    # DELEGUE, elle ne paie plus elle-meme. Le cout doit donc etre la-bas.
    with open(os.path.join(_REPO, script), encoding="utf-8",
              errors="replace") as fh:
        src = fh.read()
    assert "_GRILLE.surface_3d(" in src, (
        "%s : la grille 3D doit passer par le module" % script)
    grille = open(os.path.join(_REPO, "_etapes", "grille.py"),
                  encoding="utf-8").read()
    assert "def surface_3d(" in grille and "self.evaluer(pt)" in grille, (
        "le calcul de la grille 3D a disparu au lieu d'etre deplace")


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


# --------------------------------------------------------------------- #
# LE CADRAGE DES FIGURES -- la cinquieme divergence
# --------------------------------------------------------------------- #
# Les deux etudes ne cadraient pas leurs figures pareil :
#
#     flexion pure : u1_min .. u1_max      (les bornes de la grille HF)
#     Moulin Blanc : eff_bounds +/- 1      (les bornes de recherche, elargies)
#
# L'ecart etait recopie dans QUATRE fonctions de trace, dix-huit lignes par
# fichier, sans qu'aucun commentaire ne le signale. C'est un REGLAGE : il vit
# desormais dans le fichier d'etude.
def test_le_cadrage_grille_reproduit_EXACTEMENT_l_ancien_code_flexion():
    """Preuve d'equivalence : `"grille"` doit rendre les bornes de la grille
    HF, telles quelles. C'est ce que `AC3_pure_flexion.py` ecrivait en dur."""
    assert figurer.cadre_des_figures(
        "grille", (-7.5, 7.5, -6.0, 6.0), [-7.5, -7.5], [7.5, 7.5], 1.0) \
        == (-7.5, 7.5, -6.0, 6.0)


def test_le_cadrage_elargi_reproduit_EXACTEMENT_l_ancien_code_moulin_blanc():
    """Preuve d'equivalence : `"bornes_elargies"` doit rendre
    `eff_bounds +/- marge`, ce que `AC3_moulinblanc.py` ecrivait en dur."""
    assert figurer.cadre_des_figures(
        "bornes_elargies", (-6.0, 6.0, -6.0, 6.0), [-6.0, -6.0], [6.0, 6.0], 1.0) \
        == (-7.0, 7.0, -7.0, 7.0)


def test_le_cadrage_elargi_suit_chaque_variable_separement():
    """Les deux variables peuvent avoir des bornes differentes ; le cadre doit
    suivre chacune, pas la premiere pour les deux."""
    assert figurer.cadre_des_figures(
        "bornes_elargies", (0, 0, 0, 0), [-2.0, -5.0], [3.0, 6.0], 0.5) \
        == (-2.5, 3.5, -5.5, 6.5)


def test_un_cadrage_inconnu_LEVE_au_lieu_de_choisir_pour_vous():
    """Une faute de frappe dans le fichier d'etude ne doit pas retomber
    silencieusement sur un cadrage par defaut : les figures seraient fausses
    sans que rien ne le dise."""
    with pytest.raises(ValueError) as err:
        figurer.cadre_des_figures("grile", (0, 1, 0, 1), [0], [1])
    assert "grille" in str(err.value) and "bornes_elargies" in str(err.value)


def test_le_cadrage_elargi_refuse_une_seule_variable():
    """Il lit `eff_min[1]` : avec une seule variable, l'original levait
    `IndexError` au milieu du trace."""
    with pytest.raises(ValueError) as err:
        figurer.cadre_des_figures("bornes_elargies", (0, 1, 0, 1), [-2.0], [2.0])
    assert "deux variables" in str(err.value)


@pytest.mark.parametrize("etude,attendu", [
    ("studies/pure_flexion.toml", "grille"),
    ("studies/pure_flexion_analytique.toml", "grille"),
    ("studies/moulin_blanc.toml", "bornes_elargies"),
    ("studies/moulin_blanc_fumee.toml", "bornes_elargies"),
])
def test_chaque_etude_garde_son_cadrage_historique(etude, attendu):
    """Changer le cadrage deplacerait toutes les figures publiees. Chaque
    etude garde donc le sien -- mais il est lisible."""
    cfg = schema.charger(os.path.join(_REPO, etude))
    assert cfg.cadre_figures == attendu


@pytest.mark.parametrize("script", SCRIPTS)
def test_le_cadrage_est_calcule_une_fois_et_une_seule(script):
    """Il etait recopie dans quatre fonctions de trace. Une seule expression
    doit subsister, et les traces doivent lire le resultat."""
    with open(os.path.join(_REPO, script), encoding="utf-8",
              errors="replace") as fh:
        src = fh.read()
    assert src.count("_figurer.cadre_des_figures(") == 1, (
        "%s : le cadrage doit etre calcule en UN endroit" % script)
    assert "eff_bounds_min[0] - 1" not in src, (
        "%s porte encore un cadrage litteral" % script)
    assert src.count("_CX0, _CX1, _CY0, _CY1 =") == 1


# --------------------------------------------------------------------- #
# LA GRILLE HAUTE FIDELITE ETAIT CALCULEE DEUX FOIS
# --------------------------------------------------------------------- #
# `slice_def` et `slice_def_final` valent TOUS DEUX (0, 1, {}) des qu'il y a
# deux variables -- mais ils etaient servis par deux fichiers de cache
# differents, `hf_grid_cache.json` et `hf_grid_cache_final.json`. La meme
# grille etait donc calculee deux fois.
#
# MESURE du 27/08/2026, etude analytique a n_grid_hf = 7, cache vide :
#     sans la garde : 2 grilles, 98 appels, deux fichiers de cache
#     avec la garde : 1 grille,  49 appels, un fichier
#
# Sur le Moulin Blanc regle a 15, cela fait 225 appels evites -- 29,1 heures
# a 466 s l'appel.
@pytest.mark.parametrize("script", SCRIPTS)
def test_une_coupe_identique_a_la_courante_ne_paie_pas_un_second_cache(script):
    """La garde qui evite le doublon. Sans elle, `n_grid_hf ** 2` appels
    solveur sont payes deux fois pour la MEME grille."""
    with open(os.path.join(_REPO, script), encoding="utf-8",
              errors="replace") as fh:
        src = fh.read()
    assert "_GRILLE.fond_de_figure(" in src, (
        "%s : le fond de figure ne passe plus par la grille, donc plus par "
        "la garde qui evite de le calculer deux fois." % script)
    assert "coupe_courante=slice_def" in src, (
        "%s : la grille ne connait plus la coupe COURANTE. Sans cette "
        "reference, elle ne peut pas reconnaitre qu'une coupe lui est egale, "
        "et la garde ne sert plus a rien." % script)

    # la garde elle-meme, un cran plus bas
    with open(os.path.join(_REPO, "_etapes", "grille.py"),
              encoding="utf-8") as fh:
        src_mod = fh.read()
    assert "if sd == self.coupe_courante:" in src_mod, (
        "la garde a disparu de `Grille.fond_de_figure` -- la grille haute "
        "fidelite sera calculee deux fois, une par fichier de cache.")
    assert "fichier, finale = self.fichier_cache, False" in src_mod, (
        "la garde doit rediriger vers le cache COURANT, sinon elle ne sert "
        "a rien.")


def test_la_garde_des_29_heures_est_EXERCEE_pas_seulement_lue(tmp_path):
    """Le garde ci-dessus lit du texte ; celui-ci COMPTE LES APPELS.

    Deux demandes de fond -- la coupe courante, puis une coupe qui lui est
    egale mais adressee a un second fichier de cache -- ne doivent couter
    qu'UNE grille. Sans la garde, c'est `cote^2` appels solveur en double :
    225 sur le Moulin Blanc, soit 29 heures.
    """
    import sys
    chemin = os.path.join(_REPO, "_etapes")
    if chemin not in sys.path:
        sys.path.insert(0, chemin)
    import grille as _grille

    appels = []

    def evaluer(u):
        appels.append(tuple(u))
        return 1.0, [0.0, 0.0], [0.0, 0.0]

    g = _grille.Grille(evaluer=evaluer, n_var=2, cote=4,
                       bornes=(-1.0, 1.0, -1.0, 1.0),
                       fichier_cache=str(tmp_path / "courant.json"),
                       fichier_cache_complet=str(tmp_path / "complet.json"),
                       coupe_courante=(0, 1, {}), tracer=lambda _m: None)
    g.fond_de_figure()                                     # la coupe courante
    n_apres_la_premiere = len(appels)
    g.fond_de_figure((0, 1, {}), fichier=str(tmp_path / "final.json"),
                     finale=True)
    assert n_apres_la_premiere == 16, n_apres_la_premiere
    assert len(appels) == 16, (
        "la meme coupe a ete recalculee : %d appels au lieu de 16. C'est le "
        "doublon qui coutait 29 heures sur le Moulin Blanc."
        % len(appels))


def test_une_coupe_VRAIMENT_differente_est_bien_calculee(tmp_path):
    """La garde ne doit pas confondre economie et perte : une autre coupe
    est une autre grille, et elle se paie."""
    import sys
    chemin = os.path.join(_REPO, "_etapes")
    if chemin not in sys.path:
        sys.path.insert(0, chemin)
    import grille as _grille

    appels = []
    g = _grille.Grille(
        evaluer=lambda u: (appels.append(tuple(u)), (1.0, [0.0] * 3, [0.0] * 3))[1],
        n_var=3, cote=4, bornes=(-1.0, 1.0, -1.0, 1.0),
        fichier_cache=str(tmp_path / "c.json"),
        fichier_cache_complet=str(tmp_path / "f.json"),
        coupe_courante=(0, 1, {2: 0.0}), tracer=lambda _m: None)
    g.fond_de_figure()
    g.fond_de_figure((0, 2, {1: 1.5}), fichier=None, finale=True)
    assert len(appels) == 32, len(appels)


@pytest.mark.parametrize("script", SCRIPTS)
def test_le_contournement_partiel_a_disparu(script):
    """Le Moulin Blanc portait un contournement qui evitait d'UTILISER la
    seconde grille -- mais elle etait calculee quand meme. Traiter la cause
    rend le contournement inutile ; le garder masquerait sa reapparition."""
    with open(os.path.join(_REPO, script), encoding="utf-8",
              errors="replace") as fh:
        src = fh.read()
    assert "if slice_def_final == slice_def:" not in src, (
        "%s : contournement revenu. La cause se traite dans "
        "`fond_hf_pour_figures`, pas dans les fonctions de trace." % script)
