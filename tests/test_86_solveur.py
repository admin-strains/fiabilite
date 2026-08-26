"""
La frontiere avec Digital Structure tient-elle ses promesses ?

PHASE 5. `solver/` est la SEULE porte vers Digital Structure. Ce fichier
verifie ce qui peut l'etre SANS Digital Structure -- c'est-a-dire presque
tout, parce que le contrat a ete concu pour cela.

TROIS QUESTIONS
---------------
1. Les options de calcul rassemblees sont-elles celles des scripts AC ?
   Comparaison au golden `tests/golden/options_ds.json`, releve par AST a la
   revision qui precede le regroupement. Ce sont des reglages : une valeur qui
   bouge change le resultat, sans que rien ne le signale.

2. Les quatre copies avaient-elles diverge, et ou ?
   Le golden le dit : `global_physical_size` et `geometric_approximation_min`,
   codees en dur dans `run_HF` et lues dans la configuration par
   `run_one_SOL`. Comme `run_HF` sert aux points d'ENRICHISSEMENT, qui
   rejoignent le plan d'experiences, un `global_size` autre que 0.05 aurait
   entraine le metamodele sur deux maillages differents.

3. Le contrat lui-meme se tient-il ? `Evaluation`, l'etat de sante, la
   discipline du gradient.

Le module `solver/digital_structure.py` importe Digital Structure : il ne peut
donc pas etre importe ici. Il est LU par AST, comme les scripts AC.
"""

import ast
import io
import json
import os
import sys

import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
for p in (os.path.join(REPO, "solver"), os.path.join(REPO, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from interface import Evaluation, Solveur, SolveurNonConverge  # noqa: E402

MODULE_DS = os.path.join(REPO, "solver", "digital_structure.py")
GOLDEN = os.path.join(TESTS, "golden", "options_ds.json")

#: la copie de reference : la plus complete des quatre, celle dont les options
#: ont ete reprises.
REFERENCE = "pure_flexion/run_one_SOL"


def _golden():
    assert os.path.isfile(GOLDEN), (
        "golden des options absent. Le regenerer NE VAUT RIEN maintenant que "
        "les scripts sont regroupes : il doit etre relu d'une revision "
        "anterieure.\n  python tools/golden_options_ds.py --revision 1d40af6")
    return json.load(io.open(GOLDEN, encoding="utf-8"))


def _dict_rendu_par(nom_fonction):
    """Le dictionnaire que retourne une fonction de `digital_structure.py`.

    Lecture par AST : le module importe Digital Structure et ne peut pas etre
    importe sur un poste sans licence.
    """
    import golden_options_ds as g                             # noqa: PLC0415

    arbre = ast.parse(io.open(MODULE_DS, encoding="utf-8").read())
    fonction = next(n for n in ast.walk(arbre)
                    if isinstance(n, ast.FunctionDef) and n.name == nom_fonction)
    out = {}
    for n in ast.walk(fonction):
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict):
            for cle, val in zip(n.value.keys, n.value.values):
                out[ast.literal_eval(cle)] = g._litteral_ou_expression(val)
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == "kwargs" and isinstance(n.value, ast.Dict):
            for cle, val in zip(n.value.keys, n.value.values):
                out[ast.literal_eval(cle)] = g._litteral_ou_expression(val)
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Subscript) \
                and getattr(n.targets[0].value, "id", None) == "kwargs":
            out[ast.literal_eval(n.targets[0].slice)] = g._litteral_ou_expression(n.value)
    return out


def _est_expression(v):
    return isinstance(v, dict) and "_expression" in v


# --------------------------------------------------------------------------- #
# 1. Les options rassemblees sont celles des scripts AC                       #
# --------------------------------------------------------------------------- #
#: (famille, cle) -> raison. Options DELIBEREMENT rendues configurables, alors
#: que les scripts les portaient en dur. Chaque entree doit dire pourquoi : un
#: reglage de calcul qui change sans raison ecrite est un resultat qui change
#: sans qu'on le sache.
EXPOSEES_VOLONTAIREMENT = {
    ("maillage", "max_size"):
        "Valait 0.05 en dur et PLAFONNAIT la taille des elements, quelle que "
        "soit la valeur de `global_physical_size`. Mesure du 26/08/2026 sur le "
        "Moulin Blanc : passer global_size de 0,05 a 0,15 ne retirait que "
        "2,8 % des tetraedres (13 804 -> 13 418) et ne faisait rien gagner sur "
        "la duree (454 s -> 458 s). Le levier n'etait pas expose. Par defaut "
        "`max_size` suit `global_size`, ce qui reproduit l'ancien "
        "comportement quand global_size vaut 0,05 -- sa valeur dans les deux "
        "etudes.",
}


@pytest.mark.parametrize("famille,fonction", [
    ("maillage", "options_maillage"),
    ("solveur", "options_solveur"),
])
def test_aucune_valeur_de_calcul_n_a_change(famille, fonction):
    """Les valeurs LITTERALES doivent etre identiques a celles des scripts.

    Une option restee en expression (`global_size`, `static_params`) ne peut
    pas etre comparee texto : sa forme a change en changeant de portee. Le
    test exige alors seulement qu'elle soit restee une expression -- et le
    test suivant verifie qu'aucune n'a ete perdue en route.

    Une option deliberement rendue configurable est declaree dans
    `EXPOSEES_VOLONTAIREMENT`, avec sa raison. Sans cette liste, la seule
    facon de faire passer le test serait de regenerer le golden -- c'est-a-dire
    d'effacer la trace de ce que le code faisait avant.
    """
    attendu = _golden()["copies"][REFERENCE][famille]
    obtenu = _dict_rendu_par(fonction)

    ecarts = []
    for cle, valeur in sorted(attendu.items()):
        if (famille, cle) in EXPOSEES_VOLONTAIREMENT:
            continue
        if cle not in obtenu:
            ecarts.append("%s : ABSENTE du module (valait %r)" % (cle, valeur))
            continue
        if _est_expression(valeur):
            if not _est_expression(obtenu[cle]):
                ecarts.append("%s : etait une expression %r, devient le litteral %r"
                              % (cle, valeur["_expression"], obtenu[cle]))
        elif obtenu[cle] != valeur:
            ecarts.append("%s : script=%r  module=%r" % (cle, valeur, obtenu[cle]))
    assert not ecarts, (
        "%s :\n  " % famille + "\n  ".join(ecarts)
        + "\n\nSi le changement est VOULU, l'inscrire dans "
          "EXPOSEES_VOLONTAIREMENT avec sa raison et la mesure qui la fonde.")


@pytest.mark.parametrize("famille,fonction", [
    ("maillage", "options_maillage"),
    ("solveur", "options_solveur"),
])
def test_les_options_exposees_le_sont_vraiment(famille, fonction):
    """Une entree de `EXPOSEES_VOLONTAIREMENT` qui ne correspond plus a un
    ecart est une exemption qui traine : elle masquerait le jour ou l'option
    changerait pour une AUTRE raison."""
    attendu = _golden()["copies"][REFERENCE][famille]
    obtenu = _dict_rendu_par(fonction)
    inutiles = []
    for (fam, cle) in EXPOSEES_VOLONTAIREMENT:
        if fam != famille or cle not in attendu or cle not in obtenu:
            continue
        if attendu[cle] == obtenu[cle]:
            inutiles.append(cle)
    assert not inutiles, "%s : exemption(s) devenue(s) inutile(s) : %s" % (famille, inutiles)


def test_chaque_option_exposee_porte_sa_mesure():
    """La raison doit contenir des chiffres : « ca ne servait a rien » n'est
    pas une justification pour changer un reglage de calcul."""
    import re as _re                                            # noqa: PLC0415
    maigres = [cle for cle, raison in EXPOSEES_VOLONTAIREMENT.items()
               if len(raison) < 100 or not _re.search(r"\d", raison)]
    assert not maigres, "raison sans mesure pour %s" % maigres


@pytest.mark.parametrize("famille,fonction", [
    ("maillage", "options_maillage"),
    ("solveur", "options_solveur"),
])
def test_aucune_option_n_a_ete_ajoutee_ni_perdue(famille, fonction):
    """Un jeu d'options n'est pas une liste de suggestions : Digital Structure
    lit ce qu'on lui donne. Une option en trop, ou en moins, change le calcul."""
    attendu = set(_golden()["copies"][REFERENCE][famille])
    obtenu = set(_dict_rendu_par(fonction))
    assert not (attendu - obtenu), "%s : option(s) perdue(s) : %s" % (
        famille, sorted(attendu - obtenu))
    assert not (obtenu - attendu), "%s : option(s) inventee(s) : %s" % (
        famille, sorted(obtenu - attendu))


def test_la_couverture_est_reelle():
    """Une comparaison portant sur trois options ne prouverait rien."""
    g = _golden()["copies"][REFERENCE]
    assert len(g["maillage"]) >= 20 and len(g["solveur"]) >= 20


# --------------------------------------------------------------------------- #
# 2. Les quatre copies avaient diverge -- et le golden le dit                 #
# --------------------------------------------------------------------------- #
def test_le_golden_enregistre_la_divergence_des_copies():
    """Le fait mesure qui justifie la phase 5 : `run_HF` codait en dur la
    taille de maille que `run_one_SOL` lisait dans la configuration, et les
    deux alimentent le MEME metamodele. Si ce test tombe, c'est que le golden
    a ete regenere apres coup -- il ne prouverait alors plus rien."""
    import golden_options_ds as g                             # noqa: PLC0415

    divergentes = {cle for _, cle, _ in g.divergences(_golden())}
    assert divergentes == {"global_physical_size", "geometric_approximation_min"}, \
        "divergences attendues non retrouvees : %s" % sorted(divergentes)


def test_la_divergence_portait_bien_sur_run_hf():
    g = _golden()["copies"]
    for etude in ("pure_flexion", "moulin_blanc"):
        assert g["%s/run_HF" % etude]["maillage"]["global_physical_size"] == 0.05
        assert g["%s/run_one_SOL" % etude]["maillage"]["global_physical_size"] \
            == {"_expression": "global_size"}


def test_le_module_ne_code_plus_la_taille_de_maille_en_dur():
    """La correction : une seule source pour les deux chemins."""
    obtenu = _dict_rendu_par("options_maillage")
    assert _est_expression(obtenu["global_physical_size"]), obtenu["global_physical_size"]
    assert _est_expression(obtenu["geometric_approximation_min"])


def test_un_seul_fichier_du_depot_importe_digital_structure_pour_calculer():
    """La raison d'etre de la phase 5. Les scripts AC gardent leurs imports
    tant que la phase 5b ne les a pas debranches : ils sont recenses ici, et
    la liste ne doit pas grossir."""
    autorises = {
        os.path.join("solver", "digital_structure.py"),
        os.path.join("tools", "solve_one.py"),          # prototype, garde comme temoin
        os.path.join("pure_flexion", "AC3_pure_flexion.py"),
        os.path.join("Moulinblanc", "AC3_moulinblanc.py"),
        "launcher.py",
    }
    coupables = []
    for racine, dossiers, fichiers in os.walk(REPO):
        # `tests/` est hors frontiere : verifier une installation demande de
        # toucher Digital Structure (test_60), et decrire la frontiere demande
        # d'en citer le nom. Ce sont des controles, pas la chaine de calcul.
        dossiers[:] = [d for d in dossiers
                       if d not in {".git", "__pycache__", "historique",
                                    ".pytest_cache", "tests"}]
        for f in fichiers:
            if not f.endswith(".py"):
                continue
            chemin = os.path.join(racine, f)
            rel = os.path.relpath(chemin, REPO)
            if rel in autorises:
                continue
            texte = io.open(chemin, encoding="utf-8", errors="replace").read()
            if "STRAINS.rupt" in texte and "import" in texte:
                coupables.append(rel)
    assert not coupables, (
        "fichier(s) important Digital Structure hors de la frontiere : %s" % coupables)


# --------------------------------------------------------------------------- #
# La fabrique : c'est elle qui rend la chaine independante du solveur         #
# --------------------------------------------------------------------------- #
def test_la_fabrique_ne_charge_que_l_implementation_demandee():
    """Le point entier de `fabrique.py`. Importer `digital_structure`, c'est
    importer Digital Structure -- donc exiger une licence. Le module lui-meme
    ne doit donc pas l'importer en tete."""
    texte = io.open(os.path.join(REPO, "solver", "fabrique.py"),
                    encoding="utf-8").read()
    arbre = ast.parse(texte)
    au_module = [n for n in arbre.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    noms = [a.name for n in au_module for a in n.names]
    assert not noms, "fabrique.py importe au niveau module : %s" % noms


def test_la_fabrique_refuse_un_solveur_inconnu():
    import fabrique                                           # noqa: PLC0415
    with pytest.raises(ValueError, match="GEPCK"):
        fabrique.solveur("GEPCK")


def test_le_schema_et_la_fabrique_listent_les_memes_solveurs():
    """Deux listes qui doivent coincider, dans deux fichiers : sans ce test,
    un solveur ajoute a la fabrique resterait refuse par la validation, ou
    l'inverse."""
    sys.path.insert(0, os.path.join(REPO, "_config"))
    import fabrique                                           # noqa: PLC0415
    from schema import SOLVEURS                               # noqa: PLC0415
    assert set(SOLVEURS) == set(fabrique.IMPLEMENTATIONS), \
        "schema.SOLVEURS=%s  fabrique=%s" % (sorted(SOLVEURS),
                                             sorted(fabrique.IMPLEMENTATIONS))


@pytest.mark.parametrize("nom", sorted(["digital_structure", "analytique"]))
def test_la_disponibilite_ne_leve_jamais(nom):
    """Sur un poste sans licence, `digital_structure` n'est pas disponible :
    c'est une information, pas une erreur."""
    import fabrique                                           # noqa: PLC0415
    assert isinstance(fabrique.disponible(nom), bool)


@pytest.mark.parametrize("etude", ["pure_flexion", "moulin_blanc"])
def test_les_scripts_ac_n_importent_plus_digital_structure(etude):
    """Le resultat mesurable de la phase 5. Une fois les appels delegues, les
    seuls noms que `AC3_pure_flexion` empruntait encore a Digital Structure
    etaient `INITCATALOG` -- passe dans l'implementation -- et `sys`, qui
    n'etait jamais importe et ne marchait que par la fuite du `import *`."""
    chemin = {"pure_flexion": os.path.join(REPO, "pure_flexion", "AC3_pure_flexion.py"),
              "moulin_blanc": os.path.join(REPO, "Moulinblanc", "AC3_moulinblanc.py")}[etude]
    src = io.open(chemin, encoding="utf-8", errors="replace").read()
    assert "STRAINS" not in src, "%s cite encore STRAINS" % etude
    assert "INITCATALOG" not in src
    arbre = ast.parse(src)
    assert any(isinstance(n, ast.Import) and any(a.name == "sys" for a in n.names)
               for n in arbre.body), "%s : `sys` toujours pas importe" % etude


# --------------------------------------------------------------------------- #
# 3. Le contrat                                                                #
# --------------------------------------------------------------------------- #
def test_une_evaluation_est_immuable():
    """Un resultat de solveur ne se retouche pas apres coup."""
    import dataclasses
    e = Evaluation(g=0.33, alpha=1.33)
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.g = 0.0


def test_g_et_alpha_disent_la_meme_chose():
    e = Evaluation(g=0.332124954037, alpha=1.332124954037)
    assert e.alpha - 1.0 == pytest.approx(e.g, rel=1e-15)


def test_un_gradient_incomplet_est_signale_comme_tel():
    """Digital Structure peut ne rendre qu'une partie des sensibilites. La
    chaine doit pouvoir le savoir sans inspecter le tuple elle-meme."""
    assert Evaluation(g=0.0, alpha=1.0, grad_x=(0.1, 0.2)).gradient_complet
    assert not Evaluation(g=0.0, alpha=1.0, grad_x=(0.1, None)).gradient_complet
    assert not Evaluation(g=0.0, alpha=1.0).gradient_complet


def test_un_point_non_converge_peut_etre_refuse():
    """Les scripts AC lisaient `Primal_bound` sans jamais regarder `converged`
    ni `solver_status` -- zero occurrence dans les deux fichiers. A
    `global_physical_size = 0.018`, le solveur sort NUMERICAL_ERROR avec
    alpha = 1.5197 au lieu de ~1.3188, et ce point entrait au plan
    d'experiences comme une evaluation valide."""
    malade = Evaluation(g=0.5197, alpha=1.5197, sain=False,
                        diagnostic={"solver_status": "NUMERICAL_ERROR"})
    with pytest.raises(SolveurNonConverge, match="NUMERICAL_ERROR"):
        malade.exige_sain("point du DOE")
    sain = Evaluation(g=0.3188, alpha=1.3188, sain=True)
    assert sain.exige_sain() is sain


def test_le_contrat_expose_le_cout_dun_appel():
    """Un appel SOCP coute des secondes ou des minutes : l'appelant doit
    pouvoir le savoir avant de lancer une grille de 49 points."""
    assert Solveur().cout_par_appel
    with pytest.raises(NotImplementedError):
        Solveur().evaluer({})


# --------------------------------------------------------------------------- #
# L'implementation DS, lue sans etre importee                                 #
# --------------------------------------------------------------------------- #
def test_l_implementation_ds_respecte_la_signature_du_contrat():
    arbre = ast.parse(io.open(MODULE_DS, encoding="utf-8").read())
    classe = next(n for n in ast.walk(arbre)
                  if isinstance(n, ast.ClassDef) and n.name == "SolveurDS")
    methodes = {n.name: n for n in classe.body if isinstance(n, ast.FunctionDef)}
    assert "evaluer" in methodes
    args = [a.arg for a in methodes["evaluer"].args.args]
    assert args[:2] == ["self", "valeurs"], args
    noms = args + [a.arg for a in methodes["evaluer"].args.kwonlyargs]
    for attendu in ("sensibilite", "etiquette"):
        assert attendu in noms, attendu


def test_l_implementation_ds_compte_ses_appels():
    """Le nombre d'appels SOCP est le cout reel d'une etude, et la seule
    grandeur que le budget d'enrichissement borne."""
    texte = io.open(MODULE_DS, encoding="utf-8").read()
    assert "_appels" in texte and "nb_appels" in texte
