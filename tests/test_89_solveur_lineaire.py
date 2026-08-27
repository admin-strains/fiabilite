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

import numpy as np
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
    inconnus = [n for n in schema.CHAMPS_QUI_INVALIDENT_LE_CACHE if n not in noms]
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
def test_toute_etude_sur_DS_declare_son_backend():
    """Une etude qui ne le declare pas retombe sur son `InitSolver.py` -- et
    c'est precisement ce silence qui avait laisse les deux etudes diverger.

    Les etudes qui n'emploient pas Digital Structure (solveur analytique) sont
    hors sujet : il n'y a pas de point interieur a regler.
    """
    import glob
    muettes = []
    for chemin in sorted(glob.glob(os.path.join(_REPO, "studies", "*.toml"))):
        cfg = schema.charger(chemin)
        if cfg.solveur != "digital_structure":
            continue
        if cfg.solveur_lineaire is None:
            muettes.append(os.path.basename(chemin))
    assert not muettes, (
        "ces etudes tournent sur Digital Structure sans dire quel solveur "
        "lineaire elles emploient : %s.\nLa valeur reste alors celle de "
        "l'InitSolver.py, que personne ne lit." % muettes)


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


# --------------------------------------------------------------------- #
# bornes du domaine de recherche
# --------------------------------------------------------------------- #
def test_les_bornes_du_domaine_invalident_le_cache():
    """Elles ne changent pas la VALEUR d'un point, mais le CHOIX des points :
    le plan est tire par `ot.Uniform(eff_bounds_min, eff_bounds_max)`. Un
    cache tire sur +/- 7,5 ne couvre pas le domaine +/- 6,0."""
    assert "eff_bound_min" in schema.CHAMPS_QUI_INVALIDENT_LE_CACHE
    assert "eff_bound_max" in schema.CHAMPS_QUI_INVALIDENT_LE_CACHE
    large = schema.Configuration(modelname="x").signature_solveur()
    serre = schema.Configuration(modelname="x", eff_bound_min=-6.0,
                                 eff_bound_max=6.0).signature_solveur()
    assert large != serre


def test_un_domaine_vide_est_refuse():
    with pytest.raises(ValueError, match="domaine vide"):
        schema.Configuration(modelname="x", eff_bound_min=2.0,
                             eff_bound_max=1.0).valider()


def test_le_defaut_reproduit_la_valeur_codee_en_dur():
    """+/- 7,5 est ce que les deux scripts portaient. Le defaut ne doit rien
    changer aux etudes qui ne se prononcent pas -- la flexion pure notamment,
    ou `fy ~ Normal(550 ; 30,15)` vaut encore 324 MPa a u = -7,5."""
    cfg = schema.Configuration(modelname="x")
    assert (cfg.eff_bound_min, cfg.eff_bound_max) == (-7.5, 7.5)


def test_les_etudes_moulin_blanc_sont_bornees():
    """Le domaine +/- 7,5 y donne fy = 8,88 MPa et a tue le solveur."""
    import glob
    for chemin in sorted(glob.glob(os.path.join(_REPO, "studies", "moulin_blanc*.toml"))):
        cfg = schema.charger(chemin)
        nom = os.path.basename(chemin)
        assert cfg.eff_bound_min >= -6.5, "%s : borne inf %s trop large" % (nom, cfg.eff_bound_min)
        assert cfg.eff_bound_max <= 6.5, "%s : borne sup %s trop large" % (nom, cfg.eff_bound_max)
        # la grille HF a ses propres bornes : elles doivent suivre, sinon la
        # figure retourne balayer les coins qui ont tue le run.
        assert cfg.u1_min >= cfg.eff_bound_min and cfg.u1_max <= cfg.eff_bound_max, (
            "%s : la grille HF (u1) sort du domaine de recherche" % nom)
        assert cfg.u2_min >= cfg.eff_bound_min and cfg.u2_max <= cfg.eff_bound_max, (
            "%s : la grille HF (u2) sort du domaine de recherche" % nom)


def test_les_bornes_sont_classees_comme_etude():
    assert schema.CATEGORIES["eff_bound_min"] == "etude"
    assert schema.CATEGORIES["eff_bound_max"] == "etude"


# --------------------------------------------------------------------- #
# points sans gradient -- defaut du 26/08/2026
# --------------------------------------------------------------------- #
def test_le_parametre_existe_et_est_classe():
    """Digital Structure peut rendre `Sensitivity = {fy1: None, fy2: None}`
    sur un point NUMERICAL_ERROR. Le plan partait alors en
    `TypeError: unsupported format string passed to NoneType.__format__`."""
    cfg = schema.Configuration(modelname="x")
    assert cfg.exclure_points_sans_gradient is True
    assert schema.CATEGORIES["exclure_points_sans_gradient"] == "etude"


def test_c_est_bien_DEUX_decisions_distinctes():
    """`exclure_points_non_converges` porte sur l'OPINION du solveur, jugee
    peu fiable donc ignoree (Agnes, 26/08). `exclure_points_sans_gradient`
    porte sur une ABSENCE de donnee. Les confondre reviendrait a fabriquer un
    gradient nul, que le metamodele ajusterait."""
    cfg = schema.Configuration(modelname="x")
    assert cfg.exclure_points_non_converges is False, "l'opinion du solveur est ignoree"
    assert cfg.exclure_points_sans_gradient is True, "l'absence de donnee ne l'est pas"


@pytest.mark.parametrize("script", ["Moulinblanc/AC3_moulinblanc.py",
                                    "pure_flexion/AC3_pure_flexion.py"])
def test_le_plan_ecarte_au_lieu_de_fabriquer(script):
    """Garde-fou statique : `.get(f'dg_{p}', 0.0)` ne protegeait RIEN -- la
    clef existe, avec la valeur None, donc le defaut 0.0 n'etait jamais
    utilise. Le motif ne doit pas revenir dans l'assemblage du plan."""
    import re
    chemin = os.path.join(_REPO, script)
    with open(chemin, "r", encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    assert "exclure_points_sans_gradient" in src, (
        "%s n'honore pas le parametre" % script)
    # le motif trompeur ne doit plus servir a construire all_grad du plan
    assert not re.search(r"all_grad = np\.array\(\[\[SOL\[i\]\.get\(f'dg_\{p\}', 0\.0\)", src), (
        "%s reconstruit all_grad avec le defaut 0.0 illusoire" % script)


@pytest.mark.parametrize("script", ["Moulinblanc/AC3_moulinblanc.py",
                                    "pure_flexion/AC3_pure_flexion.py"])
def test_run_HF_leve_et_explique_l_asymetrie(script):
    """`run_HF` LEVE la ou le plan ECARTE, et le message doit dire pourquoi :
    un point d'enrichissement est demande PARCE QUE l'algorithme le veut la."""
    chemin = os.path.join(_REPO, script)
    with open(chemin, "r", encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    assert "exclure_points_sans_gradient`." in src or \
           "`exclure_points_sans_gradient`" in src, (
        "%s : le message de run_HF ne renvoie pas au parametre" % script)


# --------------------------------------------------------------------- #
# reprise d'un plan interrompu -- defaut de robustesse du 26/08/2026
# --------------------------------------------------------------------- #
def _ecrire_partiel(chemin, n0, xt, yt, ag, n_done, signature=None):
    with open(chemin, "w") as fh:
        json.dump({"n0": n0, "complet": False, "n_completed": n_done,
                   "signature": signature, "xt": xt, "yt": yt,
                   "all_grad": ag}, fh)


_XT3 = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
_YT3 = [[1.0], [2.0], [3.0]]
_AG3 = [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]]


def test_un_plan_interrompu_est_repris(tmp_path):
    """LE defaut : le cache incremental etait ecrit apres chaque point et
    JAMAIS relu. Trois interruptions dans la journee, ~75 min perdues a
    chaque fois."""
    f = str(tmp_path / "doe.json")
    _ecrire_partiel(f, 5, _XT3, _YT3, _AG3, 3)
    got = cache_doe.charger_doe_partiel(f, 5, xt_attendu=_XT3)
    assert got is not None
    xt, yt, ag, n = got
    assert n == 3 and len(xt) == 3
    assert yt[2][0] == 3.0 and ag[2][1] == 0.06


def test_la_reprise_VERIFIE_que_le_tirage_n_a_pas_change(tmp_path):
    """On ne SUPPOSE pas que le LHS redonne les memes points : on controle.
    Sinon on melangerait deux plans differents, sans trace."""
    f = str(tmp_path / "doe.json")
    _ecrire_partiel(f, 5, _XT3, _YT3, _AG3, 3)
    autre = [[0.1, 0.2], [9.9, 9.9], [0.5, 0.6]]     # 2e point deplace
    assert cache_doe.charger_doe_partiel(f, 5, xt_attendu=autre) is None


def test_la_reprise_respecte_la_signature(tmp_path):
    f = str(tmp_path / "doe.json")
    _ecrire_partiel(f, 5, _XT3, _YT3, _AG3, 3,
                    signature={"solveur_lineaire": "cudss"})
    assert cache_doe.charger_doe_partiel(
        f, 5, signature={"solveur_lineaire": "mumps"}, xt_attendu=_XT3) is None
    assert cache_doe.charger_doe_partiel(
        f, 5, signature={"solveur_lineaire": "cudss"}, xt_attendu=_XT3) is not None


def test_un_gradient_absent_tronque_la_reprise(tmp_path):
    """`null` devient `nan` SANS RIEN DIRE (`np.asarray([[None]], float)`).
    Reprendre tel quel injecterait des NaN dans le metamodele."""
    f = str(tmp_path / "doe.json")
    ag = [[0.01, 0.02], [None, None], [0.05, 0.06]]
    _ecrire_partiel(f, 5, _XT3, _YT3, ag, 3)
    got = cache_doe.charger_doe_partiel(f, 5, xt_attendu=_XT3)
    assert got is not None
    _, _, ag_lu, n = got
    assert n == 1, "la reprise doit s'arreter AVANT le point sans gradient"
    assert bool(np.all(np.isfinite(ag_lu))), "aucun NaN ne doit ressortir"


def test_un_premier_point_sans_gradient_annule_la_reprise(tmp_path):
    f = str(tmp_path / "doe.json")
    _ecrire_partiel(f, 5, _XT3, _YT3, [[None, None]] + _AG3[1:], 3)
    assert cache_doe.charger_doe_partiel(f, 5, xt_attendu=_XT3) is None


def test_n0_different_annule_la_reprise(tmp_path):
    f = str(tmp_path / "doe.json")
    _ecrire_partiel(f, 5, _XT3, _YT3, _AG3, 3)
    assert cache_doe.charger_doe_partiel(f, 9, xt_attendu=_XT3) is None


def test_l_ecrivain_incremental_ne_fabrique_plus_de_zero(tmp_path):
    """Le defaut `.get(f'dg_{p}', 0.0)` n'attendait qu'une clef manquante pour
    ecrire un gradient nul -- qui affirme que l'etat limite est plat."""
    f = str(tmp_path / "doe.json")
    SOL = [{"_u": [0.1, 0.2], "g": 1.0},            # AUCUNE clef dg_*
           {"_u": [0.3, 0.4], "g": 2.0, "dg_fc": 0.03, "dg_fy": 0.04}]
    cache_doe.save_doe_cache_incremental(f, 5, ["fc", "fy"], SOL, 2)
    with open(f) as fh:
        ecrit = json.load(fh)
    assert ecrit["all_grad"][0] == [None, None], (
        "une clef absente doit ressortir null, jamais 0.0 : %r"
        % (ecrit["all_grad"][0],))


@pytest.mark.parametrize("script", ["Moulinblanc/AC3_moulinblanc.py",
                                    "pure_flexion/AC3_pure_flexion.py"])
def test_le_script_reprend_et_saute_les_points_connus(script):
    with open(os.path.join(_REPO, script), encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    assert "charger_doe_partiel" in src, "%s ne reprend pas un plan interrompu" % script
    assert "if 'g' in SOL[i]:" in src, (
        "%s : run_one_SOL doit SAUTER un point deja calcule, sinon la reprise "
        "ne fait rien gagner" % script)
