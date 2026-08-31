r"""Placer un point : le seul geste dont l'echec donne des CHIFFRES FAUX.

LE DEFAUT -- MESURE LE 29/08/2026
----------------------------------
`patch_params` reecrit `dsCad.txt` et `dsLoad.txt` : c'est le mecanisme par
lequel TOUTE la chaine place ses points. Elle le faisait par `re.sub`, qui ne
dit rien quand il ne trouve pas son motif.

    demande  fc=30, fyd2=400        (fyd2 absent du modele)
    ecrit    fc   = 30.0000000000
             fyd1 = 550.0000000000  <- inchange, et fyd2 nulle part

Le solveur evaluait alors un point qui n'est pas celui demande, et `g`, les
gradients, le metamodele, beta et Pf se construisaient dessus. Tous les autres
defauts de la semaine coutaient des heures ; celui-la donnait des resultats
faux sans le dire.

Il n'etait pas atteint sur les modeles d'aujourd'hui -- sur le Moulin Blanc,
`fyd1` et `fyd2` existent bien, une fois chacun. Il attendait un parametre
renomme dans le `.ds`, une faute de frappe dans `PARAM_CONFIG`, ou une etude
pointee sur un autre modele.

POURQUOI LE CONTROLE PORTE SUR LES DEUX FICHIERS REUNIS
--------------------------------------------------------
Un parametre de geometrie ne vit que dans `dsCad.txt`, un chargement que dans
`dsLoad.txt`. « Absent d'un fichier » est donc le cas NORMAL : sur le Moulin
Blanc, `fyd1` et `fyd2` ne sont dans aucun `dsLoad.txt`. Exiger la presence
dans chaque fichier ferait echouer tous les runs.
"""

import io
import os
import re

import pytest


def patch_params(path, **params):
    """La fonction sous test, importee sans Digital Structure.

    `solver/digital_structure.py` importe les APIs `STRAINS` des sa premiere
    ligne : il ne s'importe pas hors d'un poste equipe. On extrait donc la
    fonction du source, ce qui a l'avantage de tester CE QUI EST ECRIT.
    """
    return _CHARGEE(path, **params)


def _charger():
    ici = os.path.dirname(os.path.abspath(__file__))
    chemin = os.path.join(os.path.dirname(ici), "solver", "digital_structure.py")
    src = io.open(chemin, encoding="utf-8").read()
    debut = src.index("def patch_params(")
    fin = src.index("\n#: fichiers recopies par", debut)
    espace = {"os": os, "re": re}
    exec(compile(src[debut:fin], chemin, "exec"), espace)   # noqa: S102
    return espace["patch_params"]


_CHARGEE = _charger()

CONTENU_CAD = ("fc    = 48.0000000000\n"
               "fyd1  = 550.0000000000\n"
               "gamma_c = 1.0\n")
CONTENU_LOAD = "Z='-1.0'\nq_perm  = 12.5000000000\n"


def _modele(tmp_path, cad=CONTENU_CAD, load=CONTENU_LOAD):
    io.open(str(tmp_path / "dsCad.txt"), "w", encoding="utf-8").write(cad)
    io.open(str(tmp_path / "dsLoad.txt"), "w", encoding="utf-8").write(load)
    return str(tmp_path)


def _lire(chemin, nom):
    return io.open(os.path.join(chemin, nom), encoding="utf-8").read()


# --------------------------------------------------------------------------- #
# 1. LE CAS NOMINAL N'A PAS BOUGE                                              #
# --------------------------------------------------------------------------- #
def test_un_parametre_de_geometrie_est_reecrit(tmp_path):
    d = _modele(tmp_path)
    patch_params(d, fc=30.0)
    assert "fc    = 30.0000000000" in _lire(d, "dsCad.txt")
    assert "fyd1  = 550.0000000000" in _lire(d, "dsCad.txt")


def test_un_parametre_de_chargement_est_reecrit(tmp_path):
    """Il ne vit que dans `dsLoad.txt` : son absence de `dsCad.txt` est
    normale et ne doit rien declencher."""
    d = _modele(tmp_path)
    patch_params(d, q_perm=20.0)
    assert "q_perm    = 20.0000000000" in _lire(d, "dsLoad.txt")


def test_plusieurs_parametres_a_la_fois(tmp_path):
    d = _modele(tmp_path)
    patch_params(d, fc=30.0, fyd1=400.0, q_perm=20.0)
    cad = _lire(d, "dsCad.txt")
    assert "fc    = 30.0000000000" in cad and "fyd1    = 400.0000000000" in cad
    assert "q_perm    = 20.0000000000" in _lire(d, "dsLoad.txt")


def test_le_format_est_celui_que_le_modele_attend(tmp_path):
    """Dix decimales : le solveur relit ce fichier, le format n'est pas libre."""
    d = _modele(tmp_path)
    patch_params(d, fc=1.0 / 3.0)
    assert "fc    = 0.3333333333" in _lire(d, "dsCad.txt")


# --------------------------------------------------------------------------- #
# 2. CE QU'ON NE SAIT PAS PLACER FAIT LEVER                                    #
# --------------------------------------------------------------------------- #
def test_un_parametre_absent_des_deux_fichiers_LEVE(tmp_path):
    d = _modele(tmp_path)
    with pytest.raises(ValueError, match="introuvable"):
        patch_params(d, fyd2=400.0)


def test_le_message_nomme_le_parametre_et_le_modele(tmp_path):
    d = _modele(tmp_path)
    with pytest.raises(ValueError) as exc:
        patch_params(d, fyd2=400.0, autre=1.0)
    message = str(exc.value)
    assert "fyd2" in message and "autre" in message
    assert d in message
    assert "PARAM_CONFIG" in message, (
        "le message doit dire OU regarder : la table des variables de l'etude")


def test_le_modele_n_est_pas_touche_quand_un_parametre_manque(tmp_path):
    """Un modele a moitie patche serait pire que le defaut qu'on ferme : la
    chaine repartirait d'un `.ds` melangeant deux points."""
    d = _modele(tmp_path)
    avant_cad, avant_load = _lire(d, "dsCad.txt"), _lire(d, "dsLoad.txt")
    with pytest.raises(ValueError):
        patch_params(d, fc=30.0, fyd2=400.0)
    assert _lire(d, "dsCad.txt") == avant_cad
    assert _lire(d, "dsLoad.txt") == avant_load


# --------------------------------------------------------------------------- #
# 3. LES PIEGES DE FORME                                                       #
# --------------------------------------------------------------------------- #
def test_un_prefixe_ne_compte_pas_comme_une_definition(tmp_path):
    """`fy` ne doit pas se croire trouve parce que `fyd1` existe."""
    d = _modele(tmp_path, cad="fyd1  = 550.0000000000\n")
    with pytest.raises(ValueError, match="fy"):
        patch_params(d, fy=400.0)


def test_un_nom_a_caracteres_speciaux_ne_devient_pas_un_motif(tmp_path):
    """Le nom etait interpole SANS echappement dans l'expression reguliere."""
    d = _modele(tmp_path, cad="f.c   = 48.0\nfxc   = 1.0\n")
    patch_params(d, **{"f.c": 30.0})
    contenu = _lire(d, "dsCad.txt")
    assert "f.c    = 30.0000000000" in contenu
    assert "fxc   = 1.0" in contenu, "le point a servi de joker"


def test_une_definition_en_double_est_signalee(tmp_path, capsys):
    """Diagnostic, pas correctif : seule la premiere occurrence est reecrite,
    et on ne sait pas laquelle le solveur retient."""
    d = _modele(tmp_path, cad="fc    = 48.0\nfc    = 12.0\n")
    patch_params(d, fc=30.0)
    sortie = capsys.readouterr().out
    assert "fc" in sortie and "2 fois" in sortie
    contenu = _lire(d, "dsCad.txt")
    assert "fc    = 30.0000000000" in contenu and "fc    = 12.0" in contenu


def test_un_parametre_present_dans_les_DEUX_fichiers_est_reecrit_partout(tmp_path):
    d = _modele(tmp_path, cad="p  = 1.0\n", load="p  = 2.0\n")
    patch_params(d, p=9.0)
    assert "p    = 9.0000000000" in _lire(d, "dsCad.txt")
    assert "p    = 9.0000000000" in _lire(d, "dsLoad.txt")
