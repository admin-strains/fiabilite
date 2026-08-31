r"""Les gradients rendus par Digital Structure, apparies a leurs variables.

L'APPARIEMENT EST POSITIONNEL
------------------------------
`SolveurDS` recoit `params_names` et `regions` -- un dict `{'param',
'region_key'}` par variable -- et les apparie par position. Les etudes les
construisent bien alignes :

    regions=[PARAM_CONFIG[p]['sens'] for p in params_names]

mais RIEN ne le verifiait. Un `zip` plus court aurait tronque en silence, et
les dernieres variables se seraient retrouvees sans gradient.

LE MESSAGE DOIT DESIGNER LE COUPABLE -- 29/08/2026
---------------------------------------------------
Quand une `region_key` ne correspond pas a ce que rend le solveur, TOUS les
gradients restent None, et la chaine annonce :

    run_HF en u=[...] : le solveur n'a rendu aucun gradient (grad_HF_X=[None,
    None])

Ce qui accuse le solveur -- alors que le calcul a reussi, que les
sensibilites SONT dans le fichier de resultats, et que l'erreur est dans la
table des variables de l'etude. Sur un modele a 466 s l'appel, envoyer
quelqu'un chercher du cote du solveur coute cher.

CE QUE CE FICHIER EXERCE
-------------------------
`solver/digital_structure.py` importe les APIs `STRAINS` des sa premiere
ligne : il ne s'importe pas hors d'un poste equipe. On reconstruit donc la
classe depuis le SOURCE, ce qui a l'avantage de tester ce qui est ecrit.
"""

import io
import json
import os
import re

import pytest


def _classe_solveur():
    """`SolveurDS` reconstruite sans Digital Structure.

    Seules `__init__`, `_cle_vers_param` et `_lire_resultat` sont exercees :
    elles ne touchent ni au mailleur ni au solveur.
    """
    ici = os.path.dirname(os.path.abspath(__file__))
    chemin = os.path.join(os.path.dirname(ici), "solver", "digital_structure.py")
    src = io.open(chemin, encoding="utf-8").read()

    debut = src.index("class SolveurDS:")
    corps = src[debut:]          # la classe va jusqu'a la fin du fichier

    # les methodes qui parlent a Digital Structure sont remplacees par un
    # refus explicite : ce fichier ne doit jamais les atteindre.
    corps = re.sub(r"(?ms)^    def evaluer\(self.*?(?=^    # ---)", "", corps)
    corps = corps.replace("initialiser_catalogues()", "pass")

    class _Ev:
        def __init__(self, g, alpha, grad_x=(), sain=True, diagnostic=None):
            self.g, self.alpha, self.grad_x = g, alpha, tuple(grad_x)
            self.sain, self.diagnostic = sain, diagnostic or {}

    espace = {"os": os, "json": json, "re": re, "Evaluation": _Ev,
              "time": __import__("time")}
    exec(compile(corps, chemin, "exec"), espace)      # noqa: S102
    return espace["SolveurDS"]


SolveurDS = _classe_solveur()

REGIONS = [{"param": "COMPRESSIVE_STRENGTH", "region_key": "fc"},
           {"param": "YIELD_STRENGTH", "region_key": "fy"}]


def _solveur(tmp_path, params=("fc", "fy"), regions=None, **kw):
    return SolveurDS(str(tmp_path), str(tmp_path), params,
                     REGIONS if regions is None else regions, **kw)


def _resultat(tmp_path, sensibilite=None, alpha=1.42):
    """Ecrit un `.dsmetares` comme le solveur en produit."""
    info = {"Primal_bound": [alpha], "Dual_bound": [alpha],
            "solver_status": "OPTIMAL", "converged": True}
    if sensibilite is not None:
        info["Sensitivity"] = sensibilite
    chemin = os.path.join(str(tmp_path), "Yield_analysis0_0_kine.dsmetares")
    io.open(chemin, "w", encoding="utf-8").write(json.dumps({"info": info}))


# --------------------------------------------------------------------------- #
# 1. L'APPARIEMENT EST VERIFIE A L'OUVERTURE                                   #
# --------------------------------------------------------------------------- #
def test_le_cas_nominal_s_ouvre(tmp_path):
    s = _solveur(tmp_path)
    assert s._cle_vers_param("COMPRESSIVE_STRENGTH:fc") == "fc"
    assert s._cle_vers_param("YIELD_STRENGTH:fy") == "fy"
    assert s._cle_vers_param("AUTRE:chose") is None


def test_moins_de_regions_que_de_variables_LEVE(tmp_path):
    """`zip` aurait tronque en silence, privant les dernieres variables de
    gradient pour une raison introuvable."""
    with pytest.raises(ValueError, match="region"):
        _solveur(tmp_path, regions=REGIONS[:1])


def test_plus_de_regions_que_de_variables_LEVE(tmp_path):
    with pytest.raises(ValueError, match="region"):
        _solveur(tmp_path, regions=REGIONS + [dict(REGIONS[0])])


def test_deux_variables_sur_la_meme_cle_LEVENT(tmp_path):
    """Digital Structure ne rendrait qu'un gradient pour les deux, et
    l'appariement en attribuerait un au hasard."""
    doublon = [dict(REGIONS[0]), dict(REGIONS[0])]
    with pytest.raises(ValueError, match="meme cle"):
        _solveur(tmp_path, regions=doublon)


# --------------------------------------------------------------------------- #
# 2. LES GRADIENTS ARRIVENT SUR LEUR VARIABLE                                  #
# --------------------------------------------------------------------------- #
def test_chaque_gradient_va_a_sa_variable(tmp_path):
    """L'ordre des clefs rendues par le solveur n'a aucune raison d'etre celui
    des variables : c'est la CLEF qui decide, pas la position."""
    _resultat(tmp_path, {"YIELD_STRENGTH:fy": 2.0,
                         "COMPRESSIVE_STRENGTH:fc": 1.0})
    ev = _solveur(tmp_path)._lire_resultat(str(tmp_path), True, None, 0.0, 0.0)
    assert ev.grad_x == (1.0, 2.0)
    assert ev.g == pytest.approx(0.42)


def test_une_cle_etrangere_est_ignoree(tmp_path):
    _resultat(tmp_path, {"COMPRESSIVE_STRENGTH:fc": 1.0,
                         "YIELD_STRENGTH:fy": 2.0,
                         "AUTRE:chose": 9.0})
    ev = _solveur(tmp_path)._lire_resultat(str(tmp_path), True, None, 0.0, 0.0)
    assert ev.grad_x == (1.0, 2.0)


def test_sans_sensibilite_demandee_aucun_gradient(tmp_path):
    _resultat(tmp_path, {"COMPRESSIVE_STRENGTH:fc": 1.0})
    ev = _solveur(tmp_path)._lire_resultat(str(tmp_path), False, None, 0.0, 0.0)
    assert ev.grad_x == (None, None)


# --------------------------------------------------------------------------- #
# 3. QUAND IL MANQUE UN GRADIENT, LE MESSAGE DESIGNE LE COUPABLE               #
# --------------------------------------------------------------------------- #
def test_une_region_key_qui_ne_correspond_pas_est_DIAGNOSTIQUEE(tmp_path, capsys):
    """Le cas reel : le calcul reussit, les sensibilites sont la, et rien ne
    correspond parce que la table des variables ne dit pas la meme chose."""
    _resultat(tmp_path, {"YIELD_STRENGTH:fy_groupe1": 2.0,
                         "COMPRESSIVE_STRENGTH:beton": 1.0})
    ev = _solveur(tmp_path, verbeux=False)._lire_resultat(
        str(tmp_path), True, None, 0.0, 0.0)
    assert ev.grad_x == (None, None)
    sortie = capsys.readouterr().out
    assert "SENSIBILITE" in sortie
    assert "fc" in sortie and "fy" in sortie
    assert "fy_groupe1" in sortie and "beton" in sortie, (
        "le message doit montrer les cles RENDUES, sinon il n'aide pas")
    assert "COMPRESSIVE_STRENGTH:fc" in sortie, (
        "le message doit montrer les cles ATTENDUES, pour la comparaison")


def test_un_seul_gradient_manquant_est_nomme(tmp_path, capsys):
    _resultat(tmp_path, {"COMPRESSIVE_STRENGTH:fc": 1.0})
    ev = _solveur(tmp_path, verbeux=False)._lire_resultat(
        str(tmp_path), True, None, 0.0, 0.0)
    assert ev.grad_x == (1.0, None)
    sortie = capsys.readouterr().out
    assert "aucun gradient pour fy" in sortie


def test_rien_n_est_dit_quand_tout_va_bien(tmp_path, capsys):
    """Un diagnostic qui parle a chaque appel n'est plus un diagnostic : sur
    le Moulin Blanc, il y a jusqu'a 592 appels."""
    _resultat(tmp_path, {"COMPRESSIVE_STRENGTH:fc": 1.0,
                         "YIELD_STRENGTH:fy": 2.0})
    _solveur(tmp_path, verbeux=False)._lire_resultat(
        str(tmp_path), True, None, 0.0, 0.0)
    assert "SENSIBILITE" not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 4. CE QUE LE DIAGNOSTIC RAPPORTE PAR AILLEURS                                #
# --------------------------------------------------------------------------- #
def test_un_solveur_non_convergent_n_est_pas_sain(tmp_path):
    chemin = os.path.join(str(tmp_path), "Yield_analysis0_0_kine.dsmetares")
    io.open(chemin, "w", encoding="utf-8").write(json.dumps({"info": {
        "Primal_bound": [1.42], "solver_status": "STALLED", "converged": False}}))
    ev = _solveur(tmp_path)._lire_resultat(str(tmp_path), False, None, 0.0, 0.0)
    assert ev.sain is False


def test_un_statut_absent_ne_declare_pas_le_point_malade(tmp_path):
    """Une version ancienne du `.dsmetares` ne porte ni `converged` ni
    `solver_status` : on ne SAIT pas, on ne condamne pas."""
    chemin = os.path.join(str(tmp_path), "Yield_analysis0_0_kine.dsmetares")
    io.open(chemin, "w", encoding="utf-8").write(
        json.dumps({"info": {"Primal_bound": [1.42]}}))
    ev = _solveur(tmp_path)._lire_resultat(str(tmp_path), False, None, 0.0, 0.0)
    assert ev.sain is True
