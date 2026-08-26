r"""Le solveur lineaire du point interieur, devenu un parametre d'etude.

CE QUI A MOTIVE CE FICHIER -- 26/08/2026
-----------------------------------------
Le choix du solveur lineaire (MUMPS / CuDss) vivait dans l'`InitSolver.py` de
chaque etude, en clair mais sans que rien ne le remonte : ni le resume de
journal, ni la configuration, ni les caches n'en savaient quoi que ce soit.

Les deux etudes du depot avaient donc DIVERGE en silence :

    pure_flexion/InitSolver.py    IPARM0[21] = 3   MUMPS
    Moulinblanc/InitSolver.py     IPARM0[21] = 4   CuDss

Et c'est exactement la que se separent les deux reproductibilites mesurees :
2,9e-11 sur la flexion pure, 7,7e-06 sur le Moulin Blanc. Ce n'est pas une
preuve -- les modeles n'ont ni la meme taille ni le meme conditionnement --
mais on ne pouvait pas meme formuler l'hypothese.

Ces tests ne demandent ni licence, ni GPU : `solver/params_ipm.py` existe
precisement pour que ce reglage soit verifiable sur un poste ordinaire.
"""

import copy
import json
import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _d in (os.path.join(_REPO, "solver"), os.path.join(_REPO, "_config"), _REPO):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import params_ipm                                    # noqa: E402
import schema                                        # noqa: E402
from _cache import doe as cache_doe                  # noqa: E402


#: un `cinematic_params` de la forme exacte des `InitSolver.py` du depot
def _cinematic(valeur_21=4):
    return [
        {"value": 0,   "table": "IPARM0", "indices": [20]},
        {"value": valeur_21, "table": "IPARM0", "indices": [21]},
        {"value": 0,   "table": "IPARM0", "indices": [22]},
        {"value": 50,  "table": "IPARM0", "indices": [23]},
        {"value": 0.95, "table": "DPARM0", "indices": [20]},
    ]


# --------------------------------------------------------------------- #
# le remplacement lui-meme
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("nom,attendu", [("mumps", 3), ("cudss", 4)])
def test_impose_la_bonne_valeur(nom, attendu):
    params = _cinematic(valeur_21=4)
    params_ipm.imposer_solveur_lineaire(params, nom, tracer=None)
    entree = [e for e in params if e.get("indices") == [21]][0]
    assert entree["value"] == attendu


def test_none_ne_touche_a_rien():
    """C'est le comportement d'avant l'existence du parametre, et le defaut."""
    avant = _cinematic(valeur_21=4)
    apres = copy.deepcopy(avant)
    params_ipm.imposer_solveur_lineaire(apres, None, tracer=None)
    assert apres == avant


def test_ne_touche_qu_a_l_indice_21():
    """Une erreur d'indice enverrait un numero de solveur dans le nombre max
    d'iterations ou le coefficient de bord de cone."""
    params = _cinematic(valeur_21=4)
    autres_avant = [dict(e) for e in params if e.get("indices") != [21]]
    params_ipm.imposer_solveur_lineaire(params, "mumps", tracer=None)
    autres_apres = [dict(e) for e in params if e.get("indices") != [21]]
    assert autres_apres == autres_avant


def test_pardiso_est_refuse():
    """Pardiso est deprecie (Agnes, 26/08/2026) : la valeur 1 existe dans le
    solveur mais n'est pas proposee."""
    with pytest.raises(ValueError, match="deprecie"):
        params_ipm.imposer_solveur_lineaire(_cinematic(), "pardiso", tracer=None)
    with pytest.raises(ValueError):
        params_ipm.imposer_solveur_lineaire(_cinematic(), "mkl_pardiso", tracer=None)


def test_entree_absente_leve_au_lieu_de_l_ajouter():
    """Ajouter silencieusement un reglage dans un fichier de forme inconnue
    serait pire que s'arreter."""
    sans_21 = [e for e in _cinematic() if e.get("indices") != [21]]
    with pytest.raises(ValueError, match="aucune entree IPARM0"):
        params_ipm.imposer_solveur_lineaire(sans_21, "mumps", tracer=None)


def test_le_changement_est_annonce():
    messages = []
    params_ipm.imposer_solveur_lineaire(_cinematic(4), "mumps", tracer=messages.append)
    assert messages and "IPARM0[21]" in messages[0] and "mumps" in messages[0]
    # et quand il n'y a rien a changer, on le dit aussi
    messages.clear()
    params_ipm.imposer_solveur_lineaire(_cinematic(3), "mumps", tracer=messages.append)
    assert messages and "deja" in messages[0]


# --------------------------------------------------------------------- #
# alignement des tables et de la configuration
# --------------------------------------------------------------------- #
def test_les_deux_tables_sont_alignees():
    """Deux copies d'une meme table finissent toujours par diverger -- c'est
    exactement ce qui est arrive aux deux `InitSolver.py`."""
    assert params_ipm.SOLVEURS_LINEAIRES == schema.SOLVEURS_LINEAIRES
    assert params_ipm.IPARM0_SOLVEUR_LINEAIRE \
        == schema.IPARM0_SOLVEUR_LINEAIRE_CINEMATIQUE


def test_la_configuration_refuse_un_nom_inconnu():
    with pytest.raises(ValueError, match="solveur_lineaire"):
        schema.Configuration(modelname="x", solveur_lineaire="pardiso").valider()


def test_la_configuration_accepte_none_et_les_noms_connus():
    for valeur in [None] + sorted(schema.SOLVEURS_LINEAIRES):
        schema.Configuration(modelname="x", solveur_lineaire=valeur).valider()


def test_le_resume_montre_le_solveur_lineaire_en_tete():
    """Un `solveur_lineaire=None` perdu dans un bloc de cinquante valeurs ne
    se remarque pas : la ligne doit etre en tete et dire ce que None veut
    dire."""
    txt = schema.resume(schema.Configuration(modelname="x", solveur_lineaire="mumps"))
    assert "lineaire mumps (IPARM0[21] = 3" in txt
    tete = txt.split("ETUDE --")[0]
    assert "lineaire" in tete, "la ligne doit preceder le bloc ETUDE"

    txt_none = schema.resume(schema.Configuration(modelname="x"))
    assert "non impose" in txt_none and "InitSolver.py" in txt_none


def test_il_est_classe_comme_parametre_d_etude():
    """Il change les nombres : ce n'est pas un reglage de session."""
    assert schema.CATEGORIES["solveur_lineaire"] == "etude"


# --------------------------------------------------------------------- #
# le cache de DOE ne doit plus melanger deux backends
# --------------------------------------------------------------------- #
def test_les_champs_invalidants_existent_vraiment():
    noms = {f.name for f in schema.fields(schema.Configuration)}
    inconnus = [n for n in schema.CHAMPS_QUI_INVALIDENT_UN_POINT if n not in noms]
    assert not inconnus, "champs inexistants : %s" % inconnus


def test_la_signature_porte_le_solveur_lineaire():
    sig = schema.Configuration(modelname="x", solveur_lineaire="mumps").signature_solveur()
    assert sig["solveur_lineaire"] == "mumps"
    assert "global_size" in sig and "modelname" in sig


def _ecrire_cache(chemin, signature, n0=2):
    with open(chemin, "w") as fh:
        json.dump({"n0": n0, "complet": True, "signature": signature,
                   "xt": [[0.0, 0.0], [1.0, 1.0]],
                   "yt": [[0.1], [0.2]],
                   "all_grad": [[0.0, 0.0], [0.0, 0.0]]}, fh)


def test_le_cache_est_relu_quand_la_signature_coincide(tmp_path):
    sig = {"solveur_lineaire": "mumps", "global_size": 0.05}
    f = str(tmp_path / "doe.json")
    _ecrire_cache(f, sig)
    assert cache_doe.load_doe_cache(f, 2, True, signature=sig) is not None


def test_le_cache_est_refuse_quand_le_backend_a_change(tmp_path):
    """LE cas du 26/08 : basculer CuDss -> MUMPS ne doit pas relire des points
    calcules par l'autre backend."""
    f = str(tmp_path / "doe.json")
    _ecrire_cache(f, {"solveur_lineaire": "cudss", "global_size": 0.05})
    got = cache_doe.load_doe_cache(
        f, 2, True, signature={"solveur_lineaire": "mumps", "global_size": 0.05})
    assert got is None


def test_un_cache_sans_signature_est_refuse(tmp_path):
    """Le cout est un recalcul du plan initial ; le prix de l'alternative est
    un resultat faux et silencieux."""
    f = str(tmp_path / "doe.json")
    _ecrire_cache(f, None)
    got = cache_doe.load_doe_cache(
        f, 2, True, signature={"solveur_lineaire": "mumps"})
    assert got is None


def test_sans_signature_demandee_le_comportement_est_l_ancien(tmp_path):
    """Les appels qui ne passent pas de signature ne doivent pas changer de
    comportement -- c'est ce que les temoins d'origine attendent."""
    f = str(tmp_path / "doe.json")
    _ecrire_cache(f, None)
    assert cache_doe.load_doe_cache(f, 2, True) is not None


def test_ce_qui_est_ecrit_est_relisible(tmp_path):
    """Aller-retour complet : ce que `save` ecrit, `load` doit l'accepter."""
    sig = schema.Configuration(modelname="x", solveur_lineaire="cudss").signature_solveur()
    f = str(tmp_path / "doe.json")
    cache_doe.save_doe_cache(f, 2, [[0.0, 0.0], [1.0, 1.0]], [[0.1], [0.2]],
                             [[0.0, 0.0], [0.0, 0.0]], signature=sig)
    assert cache_doe.load_doe_cache(f, 2, True, signature=sig) is not None
    # ... et refuse des qu'un seul champ bouge
    autre = dict(sig, solveur_lineaire="mumps")
    assert cache_doe.load_doe_cache(f, 2, True, signature=autre) is None


# --------------------------------------------------------------------- #
# ce que portent les etudes du depot
# --------------------------------------------------------------------- #
def test_les_deux_InitSolver_du_depot_sont_lisibles_et_documentes():
    """Constat fige : les deux etudes ne demandaient pas le meme backend.

    Ce test n'exige pas qu'elles soient identiques -- ce serait une decision
    d'etude. Il exige que l'entree existe, pour que
    `imposer_solveur_lineaire` ait toujours quelque chose a remplacer.
    """
    import re
    for rel in ("pure_flexion/InitSolver.py", "Moulinblanc/InitSolver.py"):
        chemin = os.path.join(_REPO, rel)
        with open(chemin, "r", encoding="utf-8", errors="replace") as fh:
            texte = fh.read()
        m = re.search(r'\{"value":\s*(\d+)\s*,\s*"table":\s*"IPARM0"\s*,'
                      r'\s*"indices":\s*\[21\]\s*\}', texte)
        assert m, "%s : pas d'entree IPARM0 indices=[21]" % rel
        assert int(m.group(1)) in set(params_ipm.SOLVEURS_LINEAIRES.values()), (
            "%s : IPARM0[21] = %s, valeur hors des backends proposes (Pardiso "
            "est deprecie)" % (rel, m.group(1)))
