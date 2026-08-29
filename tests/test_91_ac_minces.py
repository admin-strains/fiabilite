r"""Un script d'etude doit POSER UN PROBLEME, pas implementer la chaine.

CE QUE CE FICHIER MESURE -- phase 0, ouverte le 26/08/2026
-----------------------------------------------------------
Constat d'Agnes : « il y a toujours un gros fichier AC ? ce n'est pas sain,
AC doit quasiment etre reduit a 0 a mon sens. » La mesure lui donne raison :

    sur les 1 975 lignes de fonctions communes aux deux scripts,
    1 861 sont IDENTIQUES d'un fichier a l'autre -- 94,2 %.
    45 des 58 fonctions sont byte-identiques.
    `build_DOE` fait 114 lignes pour 2 lignes d'ecart.

Ce n'est pas deux etudes : c'est UNE implementation copiee deux fois. Et les
copies ont deja diverge trois fois -- le maillage entre `run_HF` et
`run_one_SOL`, le solveur lineaire entre les deux `InitSolver.py`, et deux
fonctions de trace presentes d'un seul cote.

CES TESTS ECHOUENT AUJOURD'HUI, ET C'EST VOULU
-----------------------------------------------
Ils ne constatent pas un acquis : ils MESURENT l'avancement de la phase 0 et
deviendront verts a mesure qu'elle progresse. Les seuils sont donc des
plafonds qu'on abaisse, jamais qu'on releve -- relever un plafond pour faire
passer un test reviendrait a supprimer la mesure.

Marques `xfail(strict=False)` : ils ne bloquent pas la suite, mais ils
passeront en `strict=True` -- donc en garde-fou -- des que la cible sera
atteinte.
"""

import ast
import os

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)

SCRIPTS = ["pure_flexion/AC3_pure_flexion.py", "Moulinblanc/AC3_moulinblanc.py"]

#: Ce qu'un script qui POSE UN PROBLEME a le droit d'importer : la
#: bibliotheque standard minimale, les deux socles numeriques, et les modules
#: du projet. Rien d'autre.
#:
#: DBSCAN, GEKPLS, `brentq`, matplotlib n'ont aucune raison d'apparaitre dans
#: un enonce de probleme. Leur presence est la preuve DIRECTE que la
#: separation des modes, l'entrainement du GEK, la recherche de racine et le
#: trace sont encore DANS le fichier plutot que derriere l'API.
TIERS_AUTORISES = {"os", "json", "sys", "openturns", "numpy"}

#: Les modules du projet -- l'API sur laquelle un AC doit s'appuyer.
PROJET = {
    "api", "lois", "doe", "hf", "eff", "eff_ot", "form", "graphiques",
    "schema", "fabrique", "interface", "params_ipm", "_parallel_is",
    "figurer",
    "launcher", "surrogate", "plan",
}

#: Cible de la phase 0 : un AC de 150 a 250 lignes. Le plafond descend a
#: chaque etape d'extraction ; il ne remonte jamais.
#: Le 26/08/2026 : 3 000 -> 3 000 refuse, les fichiers etaient a 3 013 et
#: 3 030 apres l'inversion des dependances de trace. Le cliquet a fait son
#: office : plutot que de relever le plafond -- ce qui aurait supprime la
#: mesure -- il a fallu SORTIR du code. 126 lignes de code MORT en sont
#: ressorties (`print_visu_EFF`, `print_visu_sigma`, `init_surrogate`),
#: invisibles depuis le debut du chantier parce que rien ne les mesurait.
#: 27/08 : le plan sort de `init_g_ot` (7 lignes cachees), `print_results`
#: se scinde en resume + erreur FOSM, `print_3D_HF` en grille + figure, et
#: `_etapes/figurer.py` recupere les deux restitutions communes.
#: 2995 -> 2974.
#: 27/08 (suite) : les six enveloppes OpenTURNS des metamodeles (193 l.
#: par fichier, identiques au caractere pres) partent dans
#: `_surrogate/wrappers.py`. 2974 -> 2783.
#: 27/08 (suite) : les cinq ajustements de metamodeles partent dans
#: `_surrogate/ajuster.py`. 2783 -> 2682, 55 -> 50 fonctions.
#: 27/08 (suite) : `init_g_ot` (125 l.) devient un appel a
#: `_surrogate.construire_surrogate`, et le detecteur d'imports morts,
#: corrige, en trouve six de plus. 2682 -> 2593.
#: 27/08 (suite) : `projection_surrogate` part dans
#: `_surrogate/projection.py`, et 80 lignes de bannieres annoncant des
#: sections desormais VIDES disparaissent. 2593 -> 2530.
#: 27/08 (suite) : la grille haute fidelite -- cache signe, reprise
#: apres interruption, coupes interpolees -- part dans
#: `_etapes/grille.py`. 2530 -> 2421, 50 -> 43 fonctions.
#: 27/08 (suite) : l'appel au solveur -- quatre exemplaires, un par
#: script x deux fonctions -- devient `_doe/evaluation.py`.
#: 2421 -> 2334, 43 -> 42 fonctions.
#: 27/08 (suite) : le tirage du plan, le tri des points sans gradient et
#: l'augmentation de Taylor partent dans `_doe/plan.py`. 2334 -> 2278.
#: 27/08 (suite) : les deux workers paralleles -- 115 lignes recopiees
#: quatre fois, dans un chemin de code qui n'avait jamais pu s'executer
#: -- deviennent `_doe/parallele.py`. 2278 -> 2194.
#: 27/08 (suite) : la grille de points CHOISIS -- 102 lignes identiques
#: dans les deux etudes, jamais couvertes -- rejoint `_etapes/grille.py`.
#: 2194 -> 2103.
#: 27/08 (suite) : le juge FORM+IS et les quatre criteres d'arret sortent
#: de `run_EFF`, qui passe de 321 a 225 lignes. 2103 -> 2009.
#: 27/08 (suite) : le cadrage des figures devient un reglage, et
#: `erreur_FOSM` rejoint `_reliability/controle.py`. 2009 -> 2000.
#: 27/08 (suite) : la grille 3D rejoint `_etapes/grille.py`, et la garde
#: qui evite de calculer la grille HF deux fois entre en service.
#: 2000 -> 1982.
#: 27/08 (suite) : `print_planche_EFF` (96 l.) devient un calcul de coupe
#: de 26 lignes ; le dessin part dans `_etapes/figurer.py` avec le decor
#: commun a toutes les figures. 1982 -> 1919.
#: 27/08 (suite) : `print_globalplanche_EFF` (103 l.) se scinde en un
#: reajustement du metamodele (33 l., du calcul, il reste) et un dessin
#: (parti dans `_etapes/figurer.py`). 1919 -> 1849.
#: 27/08 (suite) : `print_visu` (158 l.) part dans `_etapes/figurer.py`,
#: la surcouche analytique devient un ARGUMENT, et les deux coupes
#: passent de variables globales a l'etat de `Grille`.
#: 1849 -> 1710, 42 -> 42 fonctions (deux nommees, deux retirees).
#: 27/08 (suite) : `print_3D_HF` rejoint `_etapes/figurer.py`, et les
#: etudes cessent d'importer `pyplot` -- elles ne dessinent plus rien.
#: 1710 -> 1686.
#: 27/08 (suite) : l'etat de reprise -- ecriture ET relecture, 115 lignes
#: recopiees a l'identique -- part dans `_cache/reprise.py`. Le fichier le
#: plus cher du depot : jusqu'a 90 h d'enrichissement.
#: 1686 -> 1627.
#: 27/08 (suite) : l'evaluation du batch EFF -- le bloc de `run_EFF` qui
#: appelle le solveur et verse les points au plan -- part dans
#: `_doe/evaluation.py`. Sa branche PARALLELE est celle qui tourne sur le
#: Moulin Blanc et qu'aucune chaine de verification ne traverse : ses tests
#: unitaires sont son seul filet. 1627 -> 1598.
#: 27/08 (suite) : le bilan de fin d'enrichissement -- decomposition du
#: critere au point d'arret, et ce qui a arrete -- rejoint `_reliability`.
#: `scipy` disparait des deux etudes par la meme occasion. 1598 -> 1568.
#: 27/08 (suite) : `_find_batch_EFF_points` (59 l.) devient un appel de 6
#: lignes -- maximisation globale du critere et Kriging Believer sont dans
#: `_reliability/eff_ot.py`. 1568 -> 1523.
#: 27/08 (suite) : le montage du plan sort de `build_DOE` (72 -> 51 l.).
#: Ni la greffe d'une reprise ni le plan recopiable ne sont traverses par
#: la chaine analytique (`config_is_identical` et `print_DOE` y sont faux) :
#: exerces a part, sur un cache tronque a la main. 1523 -> 1502.
#: 27/08 (suite) : le groupe FORM -- evenement `g < 0`, points de depart,
#: warm start, choix de coupe -- part dans `_reliability/form.py`. L'evenement
#: etait construit a DEUX endroits du meme fichier ; il ne l'est plus nulle
#: part dans les etudes. 1502 -> 1489.
#: 27/08 (suite) : les historiques d'enrichissement sont vides EN PLACE au
#: lieu d'etre rebindes -- le contrat de partage part dans `ArretEFF`, et
#: l'etude n'en garde que deux lignes. 1489 -> 1488.
#: 27/08 (suite) : les trois champs d'une coupe (critere, ecart-type, etat
#: limite) etaient calcules DEUX FOIS par etude, a l'identique ; ils sont
#: dans `_reliability/eff_ot.champs_sur_coupe`. 1488 -> 1477.
#: 27/08 (suite) : la coupe finale se decide dans le flux AVANT les figures
#: (piege d'ordre, `test_112`) ; `print_visu` la recoit au lieu de la publier
#: par effet de bord, et son enveloppe locale disparait. Le nombre de lignes
#: ne bouge pas -- la correction coute deux lignes, les suppressions les
#: rendent. Fonctions : 42 -> 39.
#: 28/08 : la conduite d'une iteration d'enrichissement -- beta, l'encadrement
#: s'il est du, les compteurs -- rejoint `ControleurFORM.mesurer_iteration`.
#: La decision de payer l'encadrement s'ecrivait en deux `if` qui se
#: recouvraient ; c'est un `or`. 1477 -> 1460.
#: 28/08 (suite) : la preparation de l'enrichissement -- vider les
#: historiques EN PLACE, construire l'objet d'arret dessus, reprendre les
#: compteurs -- devient `ArretEFF.pour_un_run` ; l'amorce du plan initial
#: rejoint `ControleurFORM`. 1460 -> 1445.
#: 28/08 (suite) : le journal des points -- une ligne JSON par appel
#: solveur -- etait tenu en HUIT endroits : une fonction recopiee, deux
#: etats globaux en listes d'un element, deux ecritures directes. Il
#: devient `_cache/journal_points.JournalDesPoints`. 1445 -> 1428.
#: 28/08 (suite) : rejouer l'enrichissement etape par etape part dans
#: `_surrogate/ajuster.py`, et les TROIS globales de diagnostic du
#: metamodele (chaos, LOO, longueurs de correlation) deviennent un
#: objet `Diagnostics`. 1428 -> 1413.
#: 28/08 (suite) : `fond_hf_pour_figures` rejoint `Grille` -- avec la
#: garde des 29 heures, desormais EXERCEE par un test qui compte les
#: appels solveur au lieu de lire du texte. 1413 -> 1380.
#: 28/08 (suite) : la couche d'acces au solveur -- un solveur par modele,
#: memoise parce qu'il porte un compteur d'appels, et les deux
#: adaptateurs paralleles (plan et points). 1380 -> 1346.
PLAFOND_LIGNES = 1235
PLAFOND_FONCTIONS = 36


def _imports_tiers(chemin):
    """Les modules TIERS importes au premier niveau, hors projet et hors
    autorises."""
    with open(chemin, encoding="utf-8", errors="replace") as fh:
        arbre = ast.parse(fh.read(), filename=chemin)
    trouves = {}
    for n in ast.walk(arbre):
        if isinstance(n, ast.Import):
            for a in n.names:
                racine = a.name.split(".")[0]
                trouves.setdefault(racine, n.lineno)
        elif isinstance(n, ast.ImportFrom) and n.module:
            racine = n.module.split(".")[0]
            trouves.setdefault(racine, n.lineno)
    return {m: l for m, l in trouves.items()
            if m not in TIERS_AUTORISES and m not in PROJET}


def _fonctions_imbriquees(chemin):
    with open(chemin, encoding="utf-8", errors="replace") as fh:
        arbre = ast.parse(fh.read(), filename=chemin)
    mains = [n for n in arbre.body if isinstance(n, ast.If)]
    if not mains:
        return []
    return [(n.name, n.end_lineno - n.lineno + 1)
            for n in mains[0].body if isinstance(n, ast.FunctionDef)]


# --------------------------------------------------------------------- #
# ce qui est DEJA vrai -- ces tests doivent rester verts
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("script", SCRIPTS)
def test_l_AC_s_appuie_bien_sur_l_API(script):
    """Acquis des phases 2 a 5 : la configuration, le contrat solveur et les
    modules extraits sont importes, pas reimplemente."""
    with open(os.path.join(_REPO, script), encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    for attendu in ("import schema", "from fabrique import solveur",
                    "import lois", "import form", "from api import"):
        assert attendu in src, "%s : %r attendu" % (script, attendu)


@pytest.mark.parametrize("script", SCRIPTS)
def test_aucun_import_mort(script):
    """`anp` (autograd) et `comb` sont importes et jamais utilises. Un import
    mort fait croire a une dependance qui n'existe pas."""
    chemin = os.path.join(_REPO, script)
    with open(chemin, encoding="utf-8", errors="replace") as fh:
        src = fh.read()
        arbre = ast.parse(src, filename=chemin)
    lignes_import = set()
    noms = {}
    for n in ast.walk(arbre):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            # TOUTE l'etendue, pas seulement la premiere ligne : un
            # `from x import (a, b, c)` etale sur deux lignes laissait
            # `b` et `c` se valider
            # eux-memes, et trois imports morts sont passes au travers
            # le 27/08/2026.
            lignes_import.update(range(n.lineno, (n.end_lineno or n.lineno) + 1))
            for a in n.names:
                noms[(a.asname or a.name).split(".")[0]] = n.lineno
    # hors lignes d'import ET hors commentaires : une mention dans un
    # bloc commente n'est pas un usage. `loi_F_permanente` et
    # `loi_uni_approx` ne survivaient que par un PARAM_CONFIG mis en
    # commentaire.
    corps = [l.split("#")[0] for i, l in enumerate(src.splitlines(), 1)
             if i not in lignes_import]
    texte = "\n".join(corps)
    morts = sorted(nom for nom in noms
                   if not any(nom in ligne for ligne in corps)
                   and nom not in TIERS_AUTORISES)
    assert not morts, (
        "%s : imports jamais utilises %s -- ils font croire a une dependance "
        "qui n'existe pas." % (script, morts))


# --------------------------------------------------------------------- #
# ce qui MESURE la phase 0 -- echecs attendus, seuils qui descendent
# --------------------------------------------------------------------- #
@pytest.mark.xfail(strict=False, reason="phase 0 en cours : la chaine vit "
                                        "encore dans l'AC")
@pytest.mark.parametrize("script", SCRIPTS)
def test_un_AC_n_importe_aucune_machinerie(script):
    """LE critere d'acceptation de la phase 0.

    Un script qui POSE UN PROBLEME n'a aucune raison d'importer DBSCAN
    (separation des modes), GEKPLS (entrainement du GEK), `brentq` (recherche
    de racine) ou matplotlib (trace). Tant qu'ils sont la, la machinerie
    aussi.
    """
    tiers = _imports_tiers(os.path.join(_REPO, script))
    assert not tiers, (
        "%s importe encore de la machinerie : %s.\nCes bibliotheques "
        "appartiennent aux modules, pas a l'enonce d'une etude."
        % (script, ", ".join("%s (l.%d)" % (m, l) for m, l in sorted(tiers.items()))))


@pytest.mark.xfail(strict=False, reason="phase 0 en cours")
@pytest.mark.parametrize("script", SCRIPTS)
def test_un_AC_tient_en_250_lignes(script):
    n = sum(1 for _ in open(os.path.join(_REPO, script), encoding="utf-8",
                            errors="replace"))
    assert n <= 250, "%s fait %d lignes (cible : 150-250)" % (script, n)


@pytest.mark.xfail(strict=False, reason="phase 0 en cours")
@pytest.mark.parametrize("script", SCRIPTS)
def test_un_AC_ne_definit_presque_plus_de_fonctions(script):
    """Cible : les 7 fonctions qui decrivent le PROBLEME -- `dist_jointe`,
    `_solveur`, `_grad_vers_U`, `_etiquette_socp`, les deux helpers de
    position, `_tracer_domaine_physique`."""
    fns = _fonctions_imbriquees(os.path.join(_REPO, script))
    assert len(fns) <= 8, (
        "%s definit encore %d fonctions imbriquees (%d lignes) ; cible : 8"
        % (script, len(fns), sum(l for _, l in fns)))


# --------------------------------------------------------------------- #
# le cliquet : les plafonds ne remontent jamais
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("script", SCRIPTS)
def test_l_AC_ne_regrossit_pas(script):
    """Un plafond qu'on releve pour faire passer un test supprime la mesure.
    Celui-ci ne descend que par extraction."""
    n = sum(1 for _ in open(os.path.join(_REPO, script), encoding="utf-8",
                            errors="replace"))
    assert n <= PLAFOND_LIGNES, (
        "%s fait %d lignes, au-dela du plafond %d. Si c'est une extraction en "
        "cours, ABAISSER le plafond ; jamais le relever."
        % (script, n, PLAFOND_LIGNES))
    fns = _fonctions_imbriquees(os.path.join(_REPO, script))
    assert len(fns) <= PLAFOND_FONCTIONS, (
        "%s definit %d fonctions imbriquees, au-dela du plafond %d."
        % (script, len(fns), PLAFOND_FONCTIONS))
