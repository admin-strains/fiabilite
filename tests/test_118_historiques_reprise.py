r"""Aucun historique ne doit disparaitre a la reprise -- la REGLE, pas le cas.

LE DEFAUT, MESURE -- 28/08/2026
--------------------------------
Une etude tient cinq historiques d'enrichissement. Quatre passaient dans le
dump de reprise ; le cinquieme, `_eff_history_Pf`, n'y passait pas. Rien ne
le disait, et les deux figures produites cote a cote par le meme run ne
couvraient donc pas la meme periode.

Mesure sur `pure_flexion_analytique` (print_Pf active, budget 4 puis 6) :

    run initial, 4 iterations   hist_BB = 5   hist_Pf = 5
    reprise,     2 iterations   hist_BB = 8   hist_Pf = 3   <-- depuis la
                                                                 reprise seule

`convergence_EFF.png` montrait alors tout le run, `Pf_evolution.png` sa
derniere portion, avec le meme titre et sur le meme dossier de sortie.

CE QUE CELA COUTAIT
--------------------
Le triple `mid/sup/inf` est PAYE : deux FORM+IS de plus par iteration, trois
pour le plan initial. Ce qui precedait l'interruption etait donc calcule,
puis jete de la figure. La decision d'arret, elle, n'a jamais dependu de cet
historique -- `ArretEFF` ne fait que le vider ; c'est bien la courbe, et elle
seule, qui etait fausse.

POURQUOI LA REGLE, ET PAS LE CAS
---------------------------------
Corriger `hist_Pf` seul aurait ferme un cas. Ce fichier ferme la CLASSE :
tout nom `_eff_history_*` d'une etude doit etre a la fois ECRIT dans le dump
et RELU a la reprise. Un sixieme historique ajoute demain sans ces deux
gestes fait echouer ce test avant d'atteindre une figure.
"""

import ast
import io
import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_cache"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import reprise as _reprise   # noqa: E402

ETUDES = ("pure_flexion/AC3_pure_flexion.py", "Moulinblanc/AC3_moulinblanc.py")

#: le dump et la relecture emploient les memes clefs courtes
CLEFS = ("EFF", "BB", "BS", "theta", "beta_IS", "Pf")


def _source(script):
    return io.open(os.path.join(_REPO, script), encoding="utf-8",
                   errors="replace").read()


# --------------------------------------------------------------------------- #
# 1. l'aller-retour, sur des valeurs                                          #
# --------------------------------------------------------------------------- #
def _etat(historiques):
    return _reprise.construire_etat(
        signature="s", signature_solveur="ss", modele="PCK", timestamp="t",
        max_degree=0, n0=5, xt=[[0.0, 0.0]], yt=[0.0], all_grad=[[0.0, 0.0]],
        xt_eff=[], enrich_round=0, round_sizes_prev=[], historiques=historiques,
        coupe_hf=None, best_result=None, best_sp=None, modes=[], result_IS=None)


def _historiques(**surcharges):
    base = {"EFF": [], "BB": [], "BS": [], "theta": [], "beta_IS": [], "Pf": []}
    base.update(surcharges)
    return base


def test_le_triple_Pf_fait_l_aller_retour():
    """Ce qui est ecrit se relit identique -- valeurs ET trous."""
    paye = [{'mid': 1e-6, 'sup': 2e-7, 'inf': 4e-6},
            {'mid': 3.0, 'sup': None, 'inf': 5.0}]
    etat = _etat(_historiques(Pf=paye))
    assert "hist_Pf" in etat, (
        "le triple Pf n'est pas dans le dump : une reprise le perdra, comme "
        "avant le 28/08/2026")
    relu = _reprise.historiques_de(etat)["Pf"]
    assert relu == paye, (relu, paye)


def test_un_trou_reste_un_trou():
    """`None` ne devient pas 0.0 : une iteration sans mesure n'est pas une Pf nulle."""
    etat = _etat(_historiques(Pf=[None, {'mid': 1.0, 'sup': 2.0, 'inf': 3.0}]))
    assert _reprise.historiques_de(etat)["Pf"][0] is None


def test_un_dump_ancien_se_relit_sans_lever():
    """Les dumps ecrits avant le 28/08/2026 n'ont pas la clef."""
    etat = _etat(_historiques())
    del etat["hist_Pf"]
    assert _reprise.historiques_de(etat)["Pf"] == []


def test_un_appelant_sans_Pf_ne_leve_pas():
    """`construire_etat` accepte un jeu d'historiques sans Pf."""
    sans = {"EFF": [], "BB": [], "BS": [], "theta": [], "beta_IS": []}
    assert _etat(sans)["hist_Pf"] == []


def test_les_six_clefs_sont_les_memes_des_deux_cotes():
    """Ecriture et relecture ne peuvent pas diverger en silence."""
    lues = set(_reprise.historiques_de(_etat(_historiques())))
    assert lues == set(CLEFS), lues


# --------------------------------------------------------------------------- #
# 2. LA REGLE : aucun historique d'etude hors du dump                         #
# --------------------------------------------------------------------------- #
def _historiques_declares(script):
    """Les noms `_eff_history_*` affectes au niveau du flux de l'etude."""
    arbre = ast.parse(_source(script))
    main = [n for n in arbre.body if isinstance(n, ast.If)][0]
    noms = set()
    for n in ast.walk(main):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                for x in ast.walk(t):
                    if isinstance(x, ast.Name) and x.id.startswith("_eff_history_"):
                        noms.add(x.id)
    return noms


def _clefs_du_dump(script):
    """Les clefs passees en `historiques=` a `construire_etat`."""
    arbre = ast.parse(_source(script))
    for n in ast.walk(arbre):
        if isinstance(n, ast.Call):
            for kw in n.keywords:
                if kw.arg == "historiques" and isinstance(kw.value, ast.Dict):
                    return {k.value for k in kw.value.keys
                            if isinstance(k, ast.Constant)}
    return set()


@pytest.mark.parametrize("script", ETUDES)
def test_tout_historique_declare_est_dumpe(script):
    """Un historique qui n'est pas dans le dump est perdu a la reprise.

    On rapproche les noms `_eff_history_X` des clefs du dump. `theta` n'a pas
    de nom de cette forme (il vit dans `_DIAG`), d'ou la comparaison dans ce
    sens : chaque nom declare doit avoir sa clef.
    """
    declares = {n[len("_eff_history_"):] for n in _historiques_declares(script)}
    clefs = _clefs_du_dump(script)
    assert declares, "aucun historique trouve : l'analyse a rate sa cible"
    manquants = sorted(declares - clefs)
    assert not manquants, (
        "%s : %s tenu(s) pendant le run mais absent(s) du dump.\n"
        "Une reprise repartira d'un historique vide, et la figure qui le trace "
        "ne couvrira que la periode d'apres l'interruption -- sans le dire.\n"
        "C'est exactement ce que faisait `Pf` jusqu'au 28/08/2026."
        % (script, ", ".join(manquants)))


@pytest.mark.parametrize("script", ETUDES)
def test_tout_historique_dumpe_est_relu_a_la_reprise(script):
    """Ecrire sans relire ne sert a rien -- c'est le meme defaut, en aval.

    Le bloc de reprise doit toucher chaque nom `_eff_history_*`. On lit les
    lignes du bloc `if restart_enrich_only:` qui suivent l'appel a
    `historiques_de`.
    """
    src = _source(script)
    declares = _historiques_declares(script)
    # le bloc de reprise : depuis `_h = _reprise.historiques_de(` jusqu'a la
    # fin de la branche, reperee par la premiere ligne moins indentee.
    lignes = src.splitlines()
    debut = next(i for i, l in enumerate(lignes) if "_reprise.historiques_de(" in l)
    indent = len(lignes[debut]) - len(lignes[debut].lstrip())
    fin = debut + 1
    while fin < len(lignes) and (not lignes[fin].strip()
                                 or len(lignes[fin]) - len(lignes[fin].lstrip()) >= indent):
        fin += 1
    bloc = "\n".join(lignes[debut:fin])

    absents = sorted(n for n in declares if n not in bloc)
    assert not absents, (
        "%s : %s dumpe(s) mais jamais relu(s) a la reprise.\n"
        "Le dump les porte, le run repart sans eux : le calcul est paye deux "
        "fois, ou la courbe est amputee."
        % (script, ", ".join(absents)))
