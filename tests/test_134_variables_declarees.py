r"""Les variables aleatoires declarees en donnees : IDENTIQUES au litteral.

CE QUI EST EN JEU
------------------
Les deux etudes portaient leur catalogue en Python -- sept lignes, mais les
seules du script qui nommaient encore `fc` et `fy`. Il est desormais declare
dans `studies/*.toml`, et assemble par `_config/variables.construire`.

Une declaration qui produirait un catalogue LEGEREMENT different changerait
le resultat de l'etude sans que rien ne le dise : les lois, l'ordre des
colonnes du plan, les regions de sensibilite en dependent. Le premier test
de ce fichier est donc une EQUIVALENCE, ecrite avec le litteral d'avant.

Ce qui n'est PAS declarable, et pourquoi :

  * la SELECTION des elements (« les armatures de nuance fyd1 ») reste du
    code -- c'est une propriete du modele. Arbitrage d'Agnes, option B' ;
  * `region_key` vaut le nom de la variable. Les deux etudes l'ecrivaient a
    la main, identique au nom dans les quatre cas, avec le risque d'un
    doublon -- deux variables ecrivant leur sensibilite dans la meme region.
    Le deriver rend le doublon IMPOSSIBLE.
"""

import io
import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_config"), os.path.join(_REPO, "_model"),
           os.path.join(_REPO, "_lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("openturns", reason="la couche etudes n'est pas installee")

import numpy as np              # noqa: E402
import schema                   # noqa: E402
import selection as _sel        # noqa: E402
import variables as _variables  # noqa: E402
from lois import dist_jointe, loi_fc, loi_fy    # noqa: E402

ETUDES = ("pure_flexion", "moulin_blanc")


def _cfg(nom):
    return schema.charger(os.path.join(_REPO, "studies", nom + ".toml"))


def _cad(cfg):
    chemin = os.path.join(_REPO, "modeles", cfg.modelname + ".ds", "dsCad.txt")
    if not os.path.isfile(chemin):
        pytest.skip("le modele de %s n'est pas dans ce depot" % cfg.modelname)
    return io.open(chemin, encoding="utf-8", errors="replace").read()


def _litteral_et_construit(nom):
    """Le catalogue d'AVANT, ecrit a la main, et celui que le TOML produit."""
    cfg = _cfg(nom)
    cad = _cad(cfg)
    if nom == "pure_flexion":
        rebars = _sel.armatures(cad)
        litteral = {
            "fc": {"sens": {"param": "COMPRESSIVE_STRENGTH",
                            "solids": ["Block1"], "region_key": "fc"},
                   "loi": loi_fc, "args": (48, 0.12)},
            "fy": {"sens": {"param": "YIELD_STRENGTH", "rebars": rebars,
                            "region_key": "fy"},
                   "loi": loi_fy, "args": (550, None)},
        }
        construit = _variables.construire(
            cfg, elements={"fy": {"armatures": rebars}})
    else:
        g1 = _sel.armatures(cad, grade="fyd1")
        g2 = _sel.armatures(cad, grade="fyd2")
        litteral = {
            "fy1": {"sens": {"param": "YIELD_STRENGTH", "rebars": g1,
                             "region_key": "fy1"},
                    "loi": loi_fy, "args": (235.0, None)},
            "fy2": {"sens": {"param": "YIELD_STRENGTH", "rebars": g2,
                             "region_key": "fy2"},
                    "loi": loi_fy, "args": (235.0, None)},
        }
        construit = _variables.construire(
            cfg, elements={"fy1": {"armatures": g1}, "fy2": {"armatures": g2}})
    return litteral, construit


# --------------------------------------------------------------------------- #
# 1. L'EQUIVALENCE                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nom", ETUDES)
def test_l_ordre_des_variables_est_celui_du_litteral(nom):
    """`params_names` en derive, et avec lui l'ordre des colonnes du plan
    d'experiences. Le changer changerait des resultats sans le dire."""
    litteral, construit = _litteral_et_construit(nom)
    assert list(construit) == list(litteral)


@pytest.mark.parametrize("nom", ETUDES)
def test_les_regions_de_sensibilite_sont_identiques(nom):
    """`sens` designe au solveur ce qu'il doit deriver, et OU. Une difference
    ici -- un nom, un ordre, une clef -- change les gradients."""
    litteral, construit = _litteral_et_construit(nom)
    for cle in litteral:
        assert construit[cle]["sens"] == litteral[cle]["sens"], (
            "%s / %s : region de sensibilite differente" % (nom, cle))


@pytest.mark.parametrize("nom", ETUDES)
def test_c_est_la_MEME_fonction_de_loi(nom):
    """Pas une loi equivalente : LA fonction. Le registre `LOIS` de
    `_model/lois.py` rend l'objet lui-meme."""
    litteral, construit = _litteral_et_construit(nom)
    for cle in litteral:
        assert construit[cle]["loi"] is litteral[cle]["loi"], (
            "%s / %s : loi differente" % (nom, cle))


@pytest.mark.parametrize("nom", ETUDES)
def test_la_loi_JOINTE_est_identique_AU_BIT(nom):
    """LE test qui compte. `args` ne s'ecrit plus pareil -- `(550,)` au lieu
    de `(550, None)`, parce qu'un TOML n'a pas de valeur nulle et que le
    second argument de `loi_fy` vaut `None` par defaut. Ce qui doit etre
    identique n'est pas la forme du tuple, c'est la DISTRIBUTION.

    Verifie sur les quantiles de chaque marginale, egalite exacte.
    """
    litteral, construit = _litteral_et_construit(nom)
    noms = list(litteral)
    da, db = dist_jointe(litteral, noms), dist_jointe(construit, noms)
    assert da.getDimension() == db.getDimension() == len(noms)
    for j in range(len(noms)):
        qa = [da.getMarginal(j).computeQuantile(p)[0]
              for p in (0.001, 0.1, 0.5, 0.9, 0.999)]
        qb = [db.getMarginal(j).computeQuantile(p)[0]
              for p in (0.001, 0.1, 0.5, 0.9, 0.999)]
        assert qa == qb, (
            "%s / %s : la loi n'est plus la meme.\n  litteral  %s\n  construit %s"
            % (nom, noms[j], qa, qb))


@pytest.mark.parametrize("nom", ETUDES)
def test_region_key_vaut_le_nom_de_la_variable(nom):
    """Derivee, donc jamais dupliquee. Les etudes l'ecrivaient a la main."""
    _, construit = _litteral_et_construit(nom)
    clefs = [d["sens"]["region_key"] for d in construit.values()]
    assert clefs == list(construit)
    assert len(set(clefs)) == len(clefs)


# --------------------------------------------------------------------------- #
# 2. CE QUI DOIT ETRE REFUSE                                                   #
# --------------------------------------------------------------------------- #
def _config(**variables):
    return schema.Configuration(modelname="x", variables=variables)


def test_une_loi_inconnue_est_refusee_AU_CHARGEMENT():
    """Et non au premier tirage. Le message liste les choix."""
    cfg = _config(fy={"loi": "fyd", "args": [550], "param": "P"})
    with pytest.raises(ValueError) as capture:
        cfg.valider()
    texte = str(capture.value)
    assert "loi 'fyd' inconnue" in texte, texte
    assert "fy" in texte and "fc" in texte, "les choix ne sont pas listes"


def test_une_clef_obligatoire_absente_est_refusee():
    cfg = _config(fy={"loi": "fy", "args": [550]})       # pas de `param`
    with pytest.raises(ValueError, match="param"):
        cfg.valider()


def test_une_clef_inconnue_est_refusee():
    """Meme regle que pour les autres reglages : une faute de frappe est une
    erreur, pas un silence."""
    cfg = _config(fy={"loi": "fy", "args": [550], "param": "P",
                      "armature": ["HA1"]})             # sans le `s`
    with pytest.raises(ValueError, match="armature"):
        cfg.valider()


def test_args_doit_etre_une_liste():
    """`args = 550` au lieu de `args = [550]` : sans ce controle, le
    `tuple(...)` de la construction en ferait autre chose ou leverait loin de
    la cause."""
    cfg = _config(fy={"loi": "fy", "args": 550, "param": "P"})
    with pytest.raises(ValueError, match="liste"):
        cfg.valider()


def test_TOUTES_les_fautes_sont_dites_d_un_coup():
    """Un fichier d'etude avec trois fautes doit les montrer toutes."""
    cfg = _config(a={"loi": "zz", "args": 1},
                  b={"loi": "fy", "args": [1], "param": "P", "xx": 1})
    with pytest.raises(ValueError) as capture:
        cfg.valider()
    texte = str(capture.value)
    for attendu in ("loi 'zz' inconnue", "liste", "param", "xx"):
        assert attendu in texte, "%r absent de :\n%s" % (attendu, texte)


def test_une_variable_qui_ne_designe_RIEN_est_refusee():
    """Sa region de sensibilite serait vide, donc son gradient nul --
    indiscernable d'une insensibilite physique. Meme classe de defaut que la
    selection vide de `_model/selection.py`."""
    cfg = _config(fy={"loi": "fy", "args": [550], "param": "YIELD_STRENGTH"})
    cfg.valider()                      # la declaration est valide en soi
    with pytest.raises(ValueError) as capture:
        _variables.construire(cfg)     # mais l'etude n'a rien designe
    texte = str(capture.value)
    assert "AUCUN element" in texte, texte
    assert "gradient" in texte, "le refus ne dit pas la consequence"


def test_designer_des_elements_pour_une_variable_INCONNUE_est_refuse():
    """Une faute de frappe dans le nom cote etude : sans ce controle, la
    selection serait calculee, jetee, et la variable resterait sans elements
    -- ou pire, en aurait d'autres."""
    cfg = _config(fy={"loi": "fy", "args": [550], "param": "P",
                      "armatures": ["HA1"]})
    with pytest.raises(ValueError, match="fz"):
        _variables.construire(cfg, elements={"fz": {"armatures": ["HA1"]}})


def test_aucune_variable_declaree_est_refuse():
    cfg = _config()
    with pytest.raises(ValueError, match="aucune variable"):
        _variables.construire(cfg)


# --------------------------------------------------------------------------- #
# 3. CE QUE LA DECLARATION PERMET, ET QUI N'EXISTAIT PAS                       #
# --------------------------------------------------------------------------- #
def test_une_variable_de_chargement_se_declare_comme_les_autres():
    """L'exemple du convoi vivait en CODE COMMENTE dans l'etude du Moulin
    Blanc. Il se declare, avec ses deux clefs propres."""
    cfg = _config(s_convoi={"loi": "uni_approx", "args": [0.0, 1.0, 0.15],
                            "param": "LIVE_LOAD",
                            "cas_de_charge": "LC_convoi", "axe": "position"})
    cfg.valider()
    cat = _variables.construire(cfg)
    sens = cat["s_convoi"]["sens"]
    assert sens["load_case"] == "LC_convoi"
    assert sens["axis"] == "position"
    assert sens["region_key"] == "s_convoi"
    assert cat["s_convoi"]["args"] == (0.0, 1.0, 0.15)


def test_les_noms_LITTERAUX_du_toml_sont_repris():
    """Un NOM est une donnee : `solides = ["Block1"]` se declare."""
    cfg = _config(fc={"loi": "fc", "args": [48, 0.12],
                      "param": "COMPRESSIVE_STRENGTH",
                      "solides": ["Block1"]})
    assert _variables.construire(cfg)["fc"]["sens"]["solids"] == ["Block1"]


def test_ce_que_l_etude_selectionne_L_EMPORTE_sur_le_toml():
    """Le TOML peut porter un nom d'exemple ; la selection de l'etude, faite
    sur le modele reel, doit gagner."""
    cfg = _config(fy={"loi": "fy", "args": [550], "param": "P",
                      "armatures": ["HA_exemple"]})
    cat = _variables.construire(cfg, elements={"fy": {"armatures": ["HA1", "HA2"]}})
    assert cat["fy"]["sens"]["rebars"] == ["HA1", "HA2"]


def test_le_catalogue_s_ANNONCE():
    """Un journal de run doit dire quelles variables ont ete construites, avec
    quelle loi et sur combien d'elements : c'est la premiere chose qu'un
    ingenieur verifie."""
    cfg = _config(fy={"loi": "fy", "args": [550], "param": "YIELD_STRENGTH",
                      "armatures": ["HA1", "HA2", "HA3"]})
    dits = []
    _variables.construire(cfg, tracer=dits.append)
    assert len(dits) == 1
    for attendu in ("1 variable", "fy", "loi_fy", "YIELD_STRENGTH", "3 element"):
        assert attendu in dits[0], "%r absent de %r" % (attendu, dits[0])


# --------------------------------------------------------------------------- #
# 4. LES ETUDES NE PORTENT PLUS LEUR CATALOGUE                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_l_etude_ne_porte_plus_de_catalogue_litteral(script):
    src = io.open(os.path.join(_REPO, script), encoding="utf-8",
                  errors="replace").read()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    for interdit in ("PARAM_CONFIG_CAD", "PARAM_CONFIG_LOAD",
                     "region_key", "'loi':", '"loi":'):
        assert interdit not in code, (
            "%s : %r est revenu dans l'etude. Les variables se declarent dans "
            "`studies/*.toml`, section `[variables]`." % (script, interdit))
    assert "_variables.construire(" in code, (
        "%s ne construit pas son catalogue depuis la declaration" % script)


@pytest.mark.parametrize("nom", ETUDES)
def test_le_fichier_d_etude_declare_bien_ses_variables(nom):
    cfg = _cfg(nom)
    assert cfg.variables, "%s ne declare aucune variable" % nom
    for decl in cfg.variables.values():
        for clef in ("loi", "args", "param"):
            assert clef in decl, "%s : %r manque" % (nom, clef)
