r"""La verification en tete de run : elle attrape, et elle laisse passer.

Un controle qui ne tombe jamais ne garde rien. Chaque temoin d'ici casse
DELIBEREMENT une declaration d'etude et exige que `coherence` le dise --
puis un dernier verifie que les deux etudes reelles passent, pour qu'un
controle trop zele ne devienne pas un obstacle.

`tests/test_128_etudes_contre_modeles.py` pose les memes questions sur
l'ARBRE SYNTAXIQUE des etudes, sans les executer. Les deux se completent :
128 voit ce qui est ECRIT, 131 ce qui est CONSTRUIT.
"""

import os
import re
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_config"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import coherence          # noqa: E402
import schema             # noqa: E402


@pytest.fixture(scope="module")
def modele():
    """Le `.ds` de la flexion pure -- 24 armatures, un solide, un cas de
    charge. Assez petit pour que chaque temoin coute une milliseconde."""
    cfg = schema.charger(os.path.join(_REPO, "studies", "pure_flexion.toml"))
    if not os.path.isdir(cfg.chemin_ds):
        pytest.skip("le modele de la flexion pure n'est ni dans le storage "
                    "ni dans `modeles/` du depot")
    return cfg.chemin_ds


@pytest.fixture(scope="module")
def armatures(modele):
    texte = open(os.path.join(modele, "dsCad.txt"), encoding="utf-8",
                 errors="replace").read()
    noms = re.findall(r"REBAR\('([^']+)'", texte)
    assert noms, "le modele de reference ne porte aucune armature nommee"
    return noms


@pytest.fixture
def sain(armatures):
    """Une declaration correcte, dont chaque temoin derive sa version cassee."""
    return {
        "fc": {"sens": {"param": "COMPRESSIVE_STRENGTH", "solids": ["Block1"],
                        "region_key": "fc"}},
        "fy": {"sens": {"param": "YIELD_STRENGTH", "rebars": list(armatures),
                        "region_key": "fy"}},
    }


def _casser(sain, param, clef, valeur):
    casse = {p: {"sens": dict(v["sens"])} for p, v in sain.items()}
    casse[param]["sens"][clef] = valeur
    return casse


# --------------------------------------------------------------------------- #
# CE QUI DOIT PASSER                                                           #
# --------------------------------------------------------------------------- #
def test_une_declaration_saine_passe(sain, modele):
    assert coherence.anomalies(sain, list(sain), modele) == []


def test_les_deux_etudes_REELLES_passent():
    """Le controle porte sur les etudes du depot telles qu'elles sont.

    Il tourne a chaque run : s'il refusait a tort, il rendrait les etudes
    inlancables. Ce temoin lit les declarations comme le fait `test_128` --
    sur l'arbre -- et les confronte au modele.
    """
    import test_128_etudes_contre_modeles as t128
    for nom, (script, toml) in t128.ETUDES.items():
        dossier = t128._modele(toml)
        if dossier is None:
            pytest.skip("le modele de %s est absent" % nom)
        noms = t128._variables_declarees(script)
        faux = coherence.anomalies({p: {"sens": {"region_key": p}} for p in noms},
                                   noms, dossier)
        assert faux == [], "%s : %s" % (nom, faux)


# --------------------------------------------------------------------------- #
# CE QUI DOIT TOMBER                                                           #
# --------------------------------------------------------------------------- #
def test_une_variable_absente_du_modele_est_refusee(sain, modele):
    """Le defaut 19 : `patch_params` ignorait en SILENCE un parametre absent,
    et le solveur evaluait un point qui n'etait pas celui demande. Le refus
    existe depuis le 29/08/2026 -- mais au PREMIER APPEL SOLVEUR."""
    casse = dict(sain)
    casse["fz_inconnu"] = {"sens": {"region_key": "fz"}}
    mauvais = coherence.anomalies(casse, list(casse), modele)
    assert any("fz_inconnu" in m and "patch_params" in m for m in mauvais), mauvais


def test_une_armature_inexistante_est_refusee(sain, modele):
    """Une region de sensibilite vide rend un gradient NUL -- indiscernable
    d'une insensibilite physique. Rien ne le signalait."""
    casse = _casser(sain, "fy", "rebars", ["HA_qui_n_existe_pas"])
    mauvais = coherence.anomalies(casse, list(casse), modele)
    assert any("HA_qui_n_existe_pas" in m for m in mauvais), mauvais


def test_un_solide_inexistant_est_refuse(sain, modele):
    casse = _casser(sain, "fc", "solids", ["Block_fantome"])
    mauvais = coherence.anomalies(casse, list(casse), modele)
    assert any("Block_fantome" in m for m in mauvais), mauvais


def test_un_cas_de_charge_inexistant_est_refuse(sain, modele):
    """`load_case` porte une CHAINE et non une liste : le controle doit
    traiter les deux formes, sans confondre une chaine avec la sequence de
    ses caracteres."""
    casse = _casser(sain, "fc", "load_case", "LC_qui_n_existe_pas")
    mauvais = coherence.anomalies(casse, list(casse), modele)
    assert any("LC_qui_n_existe_pas" in m for m in mauvais), mauvais
    assert not any(len(m) > 400 for m in mauvais), (
        "la chaine a ete parcourue caractere par caractere : %s" % mauvais)


def test_un_cas_de_charge_EXISTANT_passe(sain, modele):
    """Le pendant du precedent : le modele porte `Load_case0`."""
    bon = _casser(sain, "fc", "load_case", "Load_case0")
    assert coherence.anomalies(bon, list(bon), modele) == []


def test_un_region_key_manquant_est_refuse(sain, modele):
    casse = {p: {"sens": {k: v for k, v in d["sens"].items()
                          if k != "region_key"}}
             for p, d in sain.items()}
    mauvais = coherence.anomalies(casse, list(casse), modele)
    assert any("region_key manquant" in m for m in mauvais), mauvais


def test_deux_region_key_identiques_sont_refuses(sain, modele):
    """Deux variables ecriraient leur sensibilite dans la meme region : la
    seconde ecraserait la premiere, sans que rien ne le dise."""
    casse = _casser(sain, "fy", "region_key", "fc")
    mauvais = coherence.anomalies(casse, list(casse), modele)
    assert any("deux fois" in m for m in mauvais), mauvais


def test_un_modele_illisible_est_refuse(sain, tmp_path):
    """Un dossier vide n'est pas un modele. Sans ce cas, toutes les autres
    verifications passeraient -- aucun nom n'y est absent, puisqu'il n'y a
    aucun nom."""
    mauvais = coherence.anomalies(sain, list(sain), str(tmp_path))
    assert any("illisible ou vide" in m for m in mauvais), mauvais


# --------------------------------------------------------------------------- #
# LA FORME DU REFUS                                                            #
# --------------------------------------------------------------------------- #
def test_TOUTES_les_anomalies_sont_rendues_d_un_coup(sain, modele):
    """Refuser au premier probleme obligerait a relancer autant de fois
    qu'il y a de defauts -- et chaque relance coute un appel solveur."""
    casse = _casser(sain, "fy", "rebars", ["HA_fantome"])
    casse["fz_inconnu"] = {"sens": {"region_key": "fz"}}
    mauvais = coherence.anomalies(casse, list(casse), modele)
    assert len(mauvais) >= 2, mauvais
    assert any("HA_fantome" in m for m in mauvais)
    assert any("fz_inconnu" in m for m in mauvais)


def test_le_refus_est_un_SystemExit_qui_nomme_le_modele(sain, modele):
    """Une etude mal declaree n'est pas un defaut de code : le message doit
    etre lisible sans pile d'appels, et dire SUR QUEL modele le desaccord a
    ete constate -- le repli sur `modeles/` fait qu'on n'en est jamais sur
    d'avance."""
    casse = _casser(sain, "fy", "rebars", ["HA_fantome"])
    with pytest.raises(SystemExit) as capture:
        coherence.verifier(casse, list(casse), modele, tracer=lambda m: None)
    texte = str(capture.value)
    assert "HA_fantome" in texte, texte
    assert modele in texte, texte


def test_le_passage_s_ANNONCE(sain, modele):
    """Un controle muet est un controle dont on ne sait pas s'il a tourne."""
    dits = []
    coherence.verifier(sain, list(sain), modele, tracer=dits.append)
    assert len(dits) == 1, dits
    assert "2 variable(s)" in dits[0], dits
    # 24 armatures + 1 solide sur ce modele
    assert re.search(r"(\d+) element\(s\) designe", dits[0]), dits[0]
    assert int(re.search(r"(\d+) element\(s\) designe", dits[0]).group(1)) > 1


# --------------------------------------------------------------------------- #
# LES ETUDES L'APPELLENT VRAIMENT                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_l_etude_verifie_AVANT_de_construire_son_solveur(script):
    """Un controle place apres le premier appel solveur ne fait plus gagner
    les 466 s qui le justifient."""
    src = open(os.path.join(_REPO, script), encoding="utf-8",
               errors="replace").read()
    appel = src.find("_coherence.verifier(")
    assert appel > 0, "%s n'appelle pas la verification de coherence" % script
    for plus_tard in ("_fabriquer_solveur(", "run_DOE", "construire_plan_initial"):
        pos = src.find(plus_tard)
        if pos > 0:
            assert appel < pos, (
                "%s : la verification arrive APRES %s" % (script, plus_tard))
