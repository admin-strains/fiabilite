r"""Selectionner des elements du modele : la meme chose que les regex, en
lisible -- et un refus la ou elles se taisaient.

CE QUI EST VERIFIE
-------------------
1. EQUIVALENCE STRICTE avec les expressions regulieres que les etudes
   portaient, sur les DEUX modeles versionnes, l'ORDRE compris. C'est la
   condition pour que la substitution ne change aucun resultat : les regions
   de sensibilite suivent cet ordre.
2. Le refus d'une selection VIDE. C'est l'apport : `re.findall` rendait `[]`,
   la region de sensibilite etait vide, son gradient valait zero -- et un
   zero de gradient ne se distingue pas d'une insensibilite physique. Une
   faute de frappe dans une nuance rendait donc un resultat plausible et
   faux.
3. Le refus d'un critere qui n'existe pas dans le modele, plutot qu'un filtre
   qui ne retient rien.
"""

import io
import os
import re
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_model"), os.path.join(_REPO, "_config")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import selection as _sel        # noqa: E402

#: Les deux modeles versionnes du depot, et ce qu'ils contiennent -- mesure du
#: 02/09/2026. Ces comptes sont ceux de `test_128` et du journal des etudes.
MODELES = {
    "test_pure_flexion": {"armatures": 24, "nuances": {"fyd": 24}},
    "Calcul_fiabilite_G+LM1_13k_2fy_membrure_inf_diagonal":
        {"armatures": 15346, "nuances": {"fyd1": 13858, "fyd2": 1488}},
}


def _cad(nom):
    chemin = os.path.join(_REPO, "modeles", nom + ".ds", "dsCad.txt")
    if not os.path.isfile(chemin):
        pytest.skip("le modele %s n'est pas dans ce depot" % nom)
    return io.open(chemin, encoding="utf-8", errors="replace").read()


@pytest.fixture(params=sorted(MODELES))
def modele(request):
    return request.param, _cad(request.param), MODELES[request.param]


# --------------------------------------------------------------------------- #
# 1. LA MEME CHOSE QUE LES REGEX, L'ORDRE COMPRIS                              #
# --------------------------------------------------------------------------- #
def test_toutes_les_armatures_a_l_identique(modele):
    """L'expression que les deux etudes portaient :
    `re.findall(r"REBAR\\('([^']+)'", _cad_txt)`."""
    nom, cad, attendu = modele
    ancien = re.findall(r"REBAR\('([^']+)'", cad)
    assert _sel.armatures(cad) == ancien, (
        "%s : la selection ne rend plus exactement ce que rendait "
        "l'expression reguliere -- ou pas dans le meme ordre. Les regions de "
        "sensibilite suivent cet ordre." % nom)
    assert len(ancien) == attendu["armatures"]


def test_les_deux_groupes_du_moulin_blanc_a_l_identique():
    """Les deux expressions du Moulin Blanc :
    `re.findall(r"REBAR\\('([^']+)',[^\\n]*GRADE=fyd1,", _cad_txt)`."""
    nom = "Calcul_fiabilite_G+LM1_13k_2fy_membrure_inf_diagonal"
    cad = _cad(nom)
    for nuance, compte in (("fyd1", 13858), ("fyd2", 1488)):
        ancien = re.findall(r"REBAR\('([^']+)',[^\n]*GRADE=%s," % nuance, cad)
        obtenu = _sel.armatures(cad, grade=nuance)
        assert obtenu == ancien, "nuance %s : selection differente" % nuance
        assert len(obtenu) == compte


def test_les_nuances_partitionnent_les_armatures(modele):
    """Meme propriete que `test_128` verifie sur le modele, vue d'ici : la
    somme des nuances doit faire le total. Une armature sans nuance garderait
    sa limite nominale sans etre une variable aleatoire."""
    nom, cad, attendu = modele
    compte = _sel.nuances(cad)
    assert compte == attendu["nuances"], nom
    assert sum(compte.values()) == attendu["armatures"], (
        "%s : %d armatures nuancees pour %d au total -- il en manque."
        % (nom, sum(compte.values()), attendu["armatures"]))


def test_un_critere_selectionne_un_SOUS_ensemble(modele):
    """Garde-fou de la sonde : un filtre doit filtrer. Sans cela, une
    selection qui rend tout passerait pour un filtre qui marche."""
    nom, cad, attendu = modele
    toutes = set(_sel.armatures(cad))
    for nuance in attendu["nuances"]:
        sous = set(_sel.armatures(cad, grade=nuance))
        assert sous, nuance
        assert sous <= toutes, "%s : la nuance %s sort du total" % (nom, nuance)
    if len(attendu["nuances"]) > 1:
        parts = [set(_sel.armatures(cad, grade=g)) for g in attendu["nuances"]]
        assert set.union(*parts) == toutes
        assert not set.intersection(*parts), (
            "%s : une armature est dans deux nuances" % nom)


def test_les_solides_de_la_flexion_pure():
    cad = _cad("test_pure_flexion")
    assert _sel.solides(cad) == ["Block1", "Block2"]


def test_un_modele_de_coques_n_a_pas_de_solides_ET_LE_DIT():
    """Le Moulin Blanc n'a aucun appel `BLOCK(` -- c'est un modele de coques.
    La bonne reponse n'est pas une liste vide (que l'appelant passerait au
    solveur) mais un refus qui nomme la cause."""
    cad = _cad("Calcul_fiabilite_G+LM1_13k_2fy_membrure_inf_diagonal")
    with pytest.raises(ValueError, match="aucun appel BLOCK"):
        _sel.solides(cad)


# --------------------------------------------------------------------------- #
# 2. CE QUE LES REGEX NE DISAIENT PAS                                          #
# --------------------------------------------------------------------------- #
def test_une_nuance_mal_orthographiee_est_REFUSEE(modele):
    """L'APPORT de ce module. `re.findall` rendait `[]` ; la region de
    sensibilite etait vide, son gradient valait zero, et un zero de gradient
    ne se distingue pas d'une insensibilite physique."""
    nom, cad, attendu = modele
    fautive = sorted(attendu["nuances"])[0].replace("fyd", "fdy")
    assert fautive not in attendu["nuances"]
    with pytest.raises(ValueError) as capture:
        _sel.armatures(cad, grade=fautive)
    texte = str(capture.value)
    assert "VIDE" in texte, texte
    # le message doit dire CE QUI EXISTE, sans quoi il faut ouvrir le modele
    for presente in attendu["nuances"]:
        assert presente in texte, (
            "le refus ne dit pas quelles nuances existent : %s" % texte)


def test_un_critere_inconnu_du_modele_est_REFUSE(modele):
    """`grades=` au lieu de `grade=` : un filtre sur un attribut absent ne
    retiendrait rien, et l'utilisateur chercherait la cause dans son
    modele."""
    nom, cad, _ = modele
    with pytest.raises(ValueError) as capture:
        _sel.armatures(cad, grades="fyd1")
    texte = str(capture.value)
    assert "GRADES" in texte and "inconnu" in texte, texte
    assert "GRADE" in texte, "le refus doit lister les attributs presents"


def test_le_refus_ne_depend_pas_du_modele():
    """Sur un texte fabrique, pour que ces refus soient verifies meme sans les
    modeles du depot."""
    faux = ("REBAR('A', DIAMETER=5, GRADE=g1, DISTANCE=0)\n"
            "REBAR('B', DIAMETER=8, GRADE=g2, DISTANCE=0)\n")
    assert _sel.armatures(faux) == ["A", "B"]
    assert _sel.armatures(faux, grade="g1") == ["A"]
    assert _sel.armatures(faux, diametre="8") == ["B"]
    assert _sel.nuances(faux) == {"g1": 1, "g2": 1}
    with pytest.raises(ValueError, match="VIDE"):
        _sel.armatures(faux, grade="g3")
    with pytest.raises(ValueError, match="aucun appel REBAR"):
        _sel.armatures("rien du tout")


def test_deux_criteres_se_combinent():
    faux = ("REBAR('A', DIAMETER=5, GRADE=g1)\n"
            "REBAR('B', DIAMETER=8, GRADE=g1)\n"
            "REBAR('C', DIAMETER=5, GRADE=g2)\n")
    assert _sel.armatures(faux, grade="g1", diametre="5") == ["A"]


def test_les_valeurs_sont_comparees_COMME_ECRITES():
    """Dans la flexion pure, `DIAMETER=phi` : le diametre est une VARIABLE du
    modele, pas un nombre. Convertir en flottant leverait ; comparer des
    chaines laisse selectionner dessus."""
    faux = "REBAR('A', DIAMETER=phi, GRADE=fyd)\n"
    assert _sel.armatures(faux, diametre="phi") == ["A"]


# --------------------------------------------------------------------------- #
# 3. LES ETUDES S'EN SERVENT                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_l_etude_ne_porte_plus_d_expression_reguliere_sur_le_modele(script):
    """La ligne que ce module remplace. Si elle revient, il y a de nouveau
    deux facons de selectionner -- et une seule refuse le vide."""
    src = io.open(os.path.join(_REPO, script), encoding="utf-8",
                  errors="replace").read()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    assert 'REBAR\\(' not in code and "REBAR\\('" not in code, (
        "%s selectionne encore les armatures par une expression reguliere : "
        "elle rend `[]` en silence quand elle ne trouve rien." % script)
    assert "_selection.armatures(" in code, (
        "%s ne passe pas par `_model/selection.py`" % script)
