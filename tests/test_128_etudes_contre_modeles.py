r"""Ce que les etudes AFFIRMENT du modele, verifie contre le modele.

POURQUOI
---------
Une etude declare ses variables aleatoires dans `PARAM_CONFIG` et designe les
elements sur lesquels le solveur doit calculer des sensibilites. Rien ne
verifiait que ces noms existent dans le `.ds`. Les consequences etaient
silencieuses ou tardives :

* un parametre absent du modele n'etait PAS ecrit par `patch_params`, et le
  solveur evaluait un point qui n'etait pas celui demande. Ferme le
  29/08/2026 -- mais le refus arrive au PREMIER appel solveur, soit 466 s sur
  le Moulin Blanc. C'est desormais dit en 0,36 s, cf. plus bas ;
* la flexion pure FABRIQUAIT les noms d'armatures -- `[f"HA{i+1}" for i in
  range(n)]` -- au lieu de les lire, la ou le Moulin Blanc les lit. Les deux
  listes coincident sur le modele d'aujourd'hui (verifie : 24 noms, identiques
  dans le meme ordre), mais rien ne le garantissait : renommer une armature
  aurait fait designer au solveur des elements inexistants. Depuis le
  02/09/2026 les deux etudes passent par `_model/selection.py`, qui REFUSE
  une selection vide au lieu de rendre `[]`.

CE QUI A ETE VERIFIE ET TROUVE SAIN
------------------------------------
Les deux groupes d'acier du Moulin Blanc PARTITIONNENT les armatures :

    REBAR total       15 346
    groupe 1 (fyd1)   13 858
    groupe 2 (fyd2)    1 488
    dans aucun groupe      0

Une armature hors des deux groupes garderait sa limite d'elasticite nominale
sans etre une variable aleatoire -- un trou de modelisation invisible dans les
resultats.

LE CONTROLE DES VARIABLES A DEMENAGE DANS `_config/coherence.py`, appele en
tete de chaque run et verifie par `tests/test_131_coherence_etude_modele.py`.
Il y est plus fort : il lit le dictionnaire REELLEMENT construit, la ou ce
fichier ne voit que ce qui est ECRIT dans le source. Le garder ici aussi
aurait donne DEUX implementations de la meme regle, qui divergeraient.

Ce qui reste ici ne se verifie que sur le SOURCE des etudes -- qu'un nom
d'armature soit LU du modele et non fabrique -- ou sur le modele seul.

CES TESTS PORTENT SUR DES DONNEES, PAS SUR DU CODE : ils ont besoin du `.ds`.
Ils le prennent par `Configuration.chemin_ds`, qui retombe sur les modeles
versionnes du depot quand le storage du poste est absent -- ils tournent donc
aussi en integration continue. Ils ne se taisent que si le modele n'est ni
dans le storage ni dans `modeles/`.
"""

import ast
import io
import os
import re
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
if os.path.join(_REPO, "_config") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "_config"))

ETUDES = {
    "pure_flexion": ("pure_flexion/AC3_pure_flexion.py",
                     "studies/pure_flexion.toml"),
    "moulin_blanc": ("Moulinblanc/AC3_moulinblanc.py",
                     "studies/moulin_blanc.toml"),
}


def _modele(toml):
    """Le dossier `.ds` de l'etude, ou None s'il n'est nulle part."""
    import schema
    cfg = schema.charger(os.path.join(_REPO, toml))
    # `cfg.chemin_ds` et non `join(storage, ...)`. Mesure du 02/09/2026 :
    # ce recalcul faisait sauter NEUF tests de ce fichier sur les cinq jobs
    # d'integration continue, avec le message « le modele n'est pas sur ce
    # poste » -- alors que les deux modeles sont versionnes dans `modeles/`
    # et donnent exactement les memes reponses (15 346 armatures, 13 858 +
    # 1 488, aucune orpheline).
    dossier = cfg.chemin_ds
    return dossier if os.path.isdir(dossier) else None


def _lire(dossier, nom):
    chemin = os.path.join(dossier, nom)
    if not os.path.exists(chemin):
        return ""
    return io.open(chemin, encoding="utf-8", errors="replace").read()


def _variables_declarees(script):
    """Les clefs de `PARAM_CONFIG_CAD` et `PARAM_CONFIG_LOAD` de l'etude.

    Lues sur l'ARBRE : executer le script demanderait Digital Structure, et
    les commentaires en portent d'anciennes versions qu'il ne faut pas
    ramasser.
    """
    src = io.open(os.path.join(_REPO, script), encoding="utf-8",
                  errors="replace").read()
    noms = []
    for n in ast.walk(ast.parse(src)):
        if not (isinstance(n, ast.Assign) and len(n.targets) == 1):
            continue
        cible = n.targets[0]
        if not (isinstance(cible, ast.Name)
                and cible.id in ("PARAM_CONFIG_CAD", "PARAM_CONFIG_LOAD")):
            continue
        if isinstance(n.value, ast.Dict):
            noms += [k.value for k in n.value.keys
                     if isinstance(k, ast.Constant)]
    return noms


@pytest.fixture(params=sorted(ETUDES))
def etude(request):
    script, toml = ETUDES[request.param]
    dossier = _modele(toml)
    if dossier is None:
        pytest.skip("le modele de %s n'est ni dans le storage ni dans "
                    "`modeles/` du depot" % request.param)
    return request.param, script, dossier


# --------------------------------------------------------------------------- #
# 1. LES ARMATURES DESIGNEES EXISTENT                                          #
# --------------------------------------------------------------------------- #
def test_les_noms_d_armatures_sont_LUS_et_non_fabriques(etude):
    """La flexion pure les fabriquait -- `[f"HA{i+1}" ...]`. Les deux listes
    coincidaient sur le modele d'aujourd'hui, mais renommer une armature
    aurait fait designer au solveur des elements inexistants."""
    nom, script, _ = etude
    src = io.open(os.path.join(_REPO, script), encoding="utf-8",
                  errors="replace").read()
    assert '_selection.armatures(' in src, (
        "%s : les noms d'armatures doivent etre LUS dans le modele, par "
        "`_model/selection.py`" % nom)
    assert 'for i in range(n_rebars)' not in src, (
        "%s : des noms d'armatures sont fabriques" % nom)
    assert 'REBAR' not in src.replace("REBAR('", "@"), (
        "%s : une expression reguliere sur `REBAR` est revenue dans l'etude. "
        "Elle rend `[]` en silence quand elle ne trouve rien -- region de "
        "sensibilite vide, gradient nul, indiscernable d'une insensibilite "
        "physique. Passer par `_selection.armatures`." % nom)


def test_le_modele_porte_bien_des_armatures_nommees(etude):
    nom, _, dossier = etude
    texte = _lire(dossier, "dsCad.txt")
    nommees = re.findall(r"REBAR\('([^']+)'", texte)
    total = len(re.findall(r"REBAR\(", texte))
    assert nommees, "%s : aucune armature nommee dans le modele" % nom
    assert len(nommees) == total, (
        "%s : %d armature(s) sur %d sans nom entre apostrophes -- elles "
        "seraient absentes des regions de sensibilite."
        % (nom, total - len(nommees), total))


# --------------------------------------------------------------------------- #
# 2. LE MOULIN BLANC : LES DEUX GROUPES PARTITIONNENT LES ARMATURES            #
# --------------------------------------------------------------------------- #
def test_les_deux_groupes_d_acier_couvrent_TOUTES_les_armatures():
    """Une armature hors des deux groupes garderait sa limite nominale sans
    etre une variable aleatoire : un trou de modelisation invisible dans les
    resultats.

    Mesure du 29/08/2026 : 13 858 + 1 488 = 15 346, aucune laissee de cote.
    """
    dossier = _modele("studies/moulin_blanc.toml")
    if dossier is None:
        pytest.skip("le modele du Moulin Blanc n'est ni dans le storage ni "
                    "dans `modeles/` du depot")
    texte = _lire(dossier, "dsCad.txt")
    tous = set(re.findall(r"REBAR\('([^']+)'", texte))
    g1 = set(re.findall(r"REBAR\('([^']+)',[^\n]*GRADE=fyd1,", texte))
    g2 = set(re.findall(r"REBAR\('([^']+)',[^\n]*GRADE=fyd2,", texte))
    assert g1 and g2, "un des deux groupes d'acier est vide"
    assert not (g1 & g2), "%d armature(s) dans les DEUX groupes" % len(g1 & g2)
    orphelines = tous - g1 - g2
    assert not orphelines, (
        "%d armature(s) dans aucun groupe (%s...) : leur limite d'elasticite "
        "resterait nominale, sans etre une variable aleatoire."
        % (len(orphelines), ", ".join(sorted(orphelines)[:3])))
