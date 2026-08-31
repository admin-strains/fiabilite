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
  le Moulin Blanc. Ce fichier le dit en une seconde ;
* la flexion pure FABRIQUAIT les noms d'armatures -- `[f"HA{i+1}" for i in
  range(n)]` -- au lieu de les lire, la ou le Moulin Blanc les lit. Les deux
  listes coincident sur le modele d'aujourd'hui (verifie : 24 noms, identiques
  dans le meme ordre), mais rien ne le garantissait : renommer une armature
  aurait fait designer au solveur des elements inexistants.

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

CES TESTS SE TAISENT SI LE MODELE N'EST PAS SUR LE POSTE. C'est le prix d'un
controle qui porte sur des donnees, pas sur du code.
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
    """Le dossier `.ds` de l'etude, ou None s'il n'est pas sur ce poste."""
    import schema
    cfg = schema.charger(os.path.join(_REPO, toml))
    dossier = os.path.join(cfg.storage, cfg.modelname + ".ds")
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
        pytest.skip("le modele de %s n'est pas sur ce poste" % request.param)
    return request.param, script, dossier


# --------------------------------------------------------------------------- #
# 1. CHAQUE VARIABLE DECLAREE EXISTE DANS LE MODELE                            #
# --------------------------------------------------------------------------- #
def test_chaque_variable_est_un_parametre_du_modele(etude):
    """La precondition de `patch_params`, verifiee en une seconde plutot
    qu'au premier appel solveur -- 466 s sur le Moulin Blanc."""
    nom, script, dossier = etude
    declarees = _variables_declarees(script)
    assert declarees, "aucune variable trouvee : l'analyse a rate sa cible"

    texte = _lire(dossier, "dsCad.txt") + "\n" + _lire(dossier, "dsLoad.txt")
    absentes = [v for v in declarees
                if not re.search(r"(?m)^" + re.escape(v) + r"\s*=", texte)]
    assert not absentes, (
        "%s : %s declaree(s) dans PARAM_CONFIG et absente(s) du modele.\n"
        "`patch_params` refuserait, mais seulement au premier appel solveur."
        % (nom, ", ".join(absentes)))


def test_aucune_variable_n_est_definie_deux_fois(etude):
    """`patch_params` ne reecrit que la PREMIERE occurrence de chaque
    fichier : deux definitions, et on ne sait pas laquelle le solveur lit."""
    nom, script, dossier = etude
    for fichier in ("dsCad.txt", "dsLoad.txt"):
        texte = _lire(dossier, fichier)
        for v in _variables_declarees(script):
            n = len(re.findall(r"(?m)^" + re.escape(v) + r"\s*=", texte))
            assert n <= 1, "%s : %s defini %d fois dans %s" % (nom, v, n, fichier)


# --------------------------------------------------------------------------- #
# 2. LES ARMATURES DESIGNEES EXISTENT                                          #
# --------------------------------------------------------------------------- #
def test_les_noms_d_armatures_sont_LUS_et_non_fabriques(etude):
    """La flexion pure les fabriquait -- `[f"HA{i+1}" ...]`. Les deux listes
    coincidaient sur le modele d'aujourd'hui, mais renommer une armature
    aurait fait designer au solveur des elements inexistants."""
    nom, script, _ = etude
    src = io.open(os.path.join(_REPO, script), encoding="utf-8",
                  errors="replace").read()
    assert 'rebar_names = re.findall' in src, (
        "%s : les noms d'armatures doivent etre LUS dans le modele" % nom)
    assert 'for i in range(n_rebars)' not in src, (
        "%s : des noms d'armatures sont fabriques" % nom)


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
# 3. LE MOULIN BLANC : LES DEUX GROUPES PARTITIONNENT LES ARMATURES            #
# --------------------------------------------------------------------------- #
def test_les_deux_groupes_d_acier_couvrent_TOUTES_les_armatures():
    """Une armature hors des deux groupes garderait sa limite nominale sans
    etre une variable aleatoire : un trou de modelisation invisible dans les
    resultats.

    Mesure du 29/08/2026 : 13 858 + 1 488 = 15 346, aucune laissee de cote.
    """
    dossier = _modele("studies/moulin_blanc.toml")
    if dossier is None:
        pytest.skip("le modele du Moulin Blanc n'est pas sur ce poste")
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
