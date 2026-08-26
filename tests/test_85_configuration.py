"""
La configuration declarative reproduit-elle celle des scripts AC ?

PHASE 4. `_config/schema.py` remplace les ~50 variables globales du bloc
`OPTIONS UTILISATEUR`. Le test le plus important de ce fichier est
`test_les_valeurs_sont_celles_des_scripts_ac` : il lit les affectations
encore presentes dans les deux scripts et les compare champ par champ a la
configuration chargee depuis les `.toml`.

C'est une comparaison a l'ETAT ACTUEL, pas a un golden : tant que les scripts
AC portent leur configuration, ils sont l'oracle. Le jour ou ils ne
l'auront plus, ce test devra etre remplace par une comparaison au golden --
et il tombera pour le dire, au lieu de passer a vide.

Ces tests ne demandent que `tomli` (stdlib a partir de Python 3.11).
"""

import ast
import io
import os
import sys

import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
for p in (os.path.join(REPO, "_config"),):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("tomli", reason="lecture TOML (tomllib a partir de 3.11)") \
    if sys.version_info < (3, 11) else None

from schema import CRITERES_EFF, MODELES, Configuration, charger  # noqa: E402

ETUDES = {
    "pure_flexion": os.path.join(REPO, "pure_flexion", "AC3_pure_flexion.py"),
    "moulin_blanc": os.path.join(REPO, "Moulinblanc", "AC3_moulinblanc.py"),
}

#: champs qui n'ont pas d'equivalent litteral dans les scripts AC
HORS_COMPARAISON = {
    "storage",            # etait enfoui dans une concatenation de chemin
    "dossier_sortie",     # remplace `path_dir`, chemin absolu du poste de l'auteur
    "hf_2d_grid_fixed",   # None des deux cotes, mais parfois ecrit en expression
    "hf_3d_grid_fixed",
}


def _config_du_script(chemin):
    """Affectations litterales au premier niveau de `main`, avant la 1re def."""
    src = io.open(chemin, encoding="utf-8", errors="replace").read()
    main = [n for n in ast.parse(src).body
            if isinstance(n, ast.If) and getattr(n.test.left, "id", None) == "__name__"]
    assert main, "%s n'a pas de bloc __main__" % chemin
    out = {}
    for n in main[0].body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            break
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except Exception:
                pass
    return out


@pytest.fixture(scope="module", params=sorted(ETUDES))
def etude(request):
    nom = request.param
    return nom, charger(os.path.join(REPO, "studies", nom + ".toml")), \
        _config_du_script(ETUDES[nom])


# --------------------------------------------------------------------------- #
# Le test qui compte                                                          #
# --------------------------------------------------------------------------- #
def test_les_valeurs_sont_celles_des_scripts_ac(etude):
    """Aucun parametre ne doit changer de valeur en passant au fichier TOML.

    Si ce test tombe, soit le TOML est faux, soit un defaut du schema ne
    correspond pas a ce que faisait le script -- dans les deux cas c'est un
    changement de comportement deguise en refactoring.
    """
    nom, cfg, ac = etude
    ecarts = []
    for champ in cfg.en_dict():
        if champ in HORS_COMPARAISON or champ not in ac:
            continue
        attendu, obtenu = ac[champ], getattr(cfg, champ)
        if isinstance(attendu, float) or isinstance(obtenu, float):
            egal = obtenu == pytest.approx(attendu, rel=1e-15)
        else:
            egal = obtenu == attendu
        if not egal:
            ecarts.append("%s : script=%r  config=%r" % (champ, attendu, obtenu))
    assert not ecarts, "%s :\n  " % nom + "\n  ".join(ecarts)


def test_la_couverture_est_reelle(etude):
    """Un test de comparaison qui ne compare rien passerait sans rien prouver.
    On exige donc qu'il porte sur au moins trente champs."""
    nom, cfg, ac = etude
    compares = [c for c in cfg.en_dict() if c not in HORS_COMPARAISON and c in ac]
    assert len(compares) >= 30, "%s : seulement %d champs compares" % (nom, len(compares))


def test_les_champs_du_schema_existent_dans_les_scripts(etude):
    """Un champ du schema qui ne correspond a rien dans les scripts AC est
    soit une invention, soit un renommage a documenter."""
    nom, cfg, ac = etude
    connus_ailleurs = {"max_of_maxdegree", "seuil_pce", "print_ana", "modelname"}
    inventes = [c for c in cfg.en_dict()
                if c not in ac and c not in HORS_COMPARAISON and c not in connus_ailleurs]
    assert not inventes, "%s : champs sans equivalent : %s" % (nom, inventes)


# --------------------------------------------------------------------------- #
# Les valeurs derivees                                                        #
# --------------------------------------------------------------------------- #
def test_un_seul_drapeau_de_modele_est_vrai(etude):
    """Les sept `do_*` etaient sept variables ecrites a la main : elles
    pouvaient se contredire. Derivees, c'est impossible."""
    _, cfg, _ = etude
    drapeaux = [cfg.do_GEPCK, cfg.do_PCK, cfg.do_PCKRG, cfg.do_KRG,
                cfg.do_GEK, cfg.do_HF, cfg.do_old_GEPCK]
    assert sum(drapeaux) == 1


@pytest.mark.parametrize("modele", MODELES)
def test_les_drapeaux_suivent_le_modele(modele):
    cfg = Configuration(modelname="x", modele=modele)
    attendu = {"GEPCK": "do_GEPCK", "PCK": "do_PCK", "PCKRG": "do_PCKRG",
               "KRG": "do_KRG", "GEK": "do_GEK", "HF": "do_HF",
               "old_GEPCK": "do_old_GEPCK"}[modele]
    for nom in ("do_GEPCK", "do_PCK", "do_PCKRG", "do_KRG", "do_GEK",
                "do_HF", "do_old_GEPCK"):
        assert getattr(cfg, nom) is (nom == attendu)


def test_en_haute_fidelite_enrichissement_et_tirage_sont_desactives():
    """Les scripts AC reecrivaient `do_EFF` et `do_IS` cinquante lignes apres
    les avoir lus de l'utilisateur. Le schema garde l'intention dans le champ
    et l'effet dans la propriete, ce qui rend la correction visible."""
    cfg = Configuration(modelname="x", modele="HF", do_EFF=True, do_IS=True)
    assert cfg.do_EFF is True and cfg.do_IS is True         # l'intention
    assert cfg.eff_actif is False and cfg.is_actif is False  # l'effet
    autre = Configuration(modelname="x", modele="GEPCK", do_EFF=True, do_IS=True)
    assert autre.eff_actif is True and autre.is_actif is True


def test_chemin_du_modele(etude):
    _, cfg, _ = etude
    assert cfg.chemin_ds.endswith(cfg.modelname + ".ds")
    assert cfg.storage in cfg.chemin_ds


# --------------------------------------------------------------------------- #
# La validation, que les scripts AC ne faisaient pas                          #
# --------------------------------------------------------------------------- #
def test_un_modele_inconnu_est_refuse():
    """Dans les scripts AC, une faute de frappe sur `modele` mettait les sept
    drapeaux a False et le calcul partait sans metamodele, sans un mot."""
    with pytest.raises(ValueError, match="modele"):
        Configuration(modelname="x", modele="GEPKC").valider()


@pytest.mark.parametrize("modif,motif", [
    ({"EFF_criteria": "BQ"}, "EFF_criteria"),
    ({"max_degree": 5, "max_of_maxdegree": 2}, "max_of_maxdegree"),
    ({"n0": 0}, "n0"),
    ({"q": 0.0}, "q"),
    ({"q": 1.5}, "q"),
    ({"n_workers_DOE": 0}, "n_workers_DOE"),
    ({"u1_min": 1.0, "u1_max": -1.0}, "bornes"),
    ({"tol_EFF": 0.0}, "tol_EFF"),
    ({"cov_IS": -0.1}, "cov_IS"),
    ({"n_batch_EFF": 4, "eps_taylor": 0.1}, "eps_taylor"),
])
def test_les_configurations_incoherentes_sont_refusees(modif, motif):
    with pytest.raises(ValueError, match=motif):
        Configuration(modelname="x", **modif).valider()


def test_le_message_d_erreur_liste_tous_les_problemes():
    """Corriger une faute a la fois sur dix lancements serait absurde."""
    with pytest.raises(ValueError) as exc:
        Configuration(modelname="x", modele="XX", n0=0, q=2.0).valider()
    msg = str(exc.value)
    assert msg.count("- ") >= 3, msg


def test_une_cle_inconnue_dans_le_toml_est_refusee(tmp_path):
    """Une faute de frappe dans un fichier d'etude doit s'arreter la, pas
    produire un parametre silencieusement ignore."""
    f = tmp_path / "x.toml"
    f.write_text('modelname = "x"\ntol_FROM = 0.05\n', encoding="utf-8")
    with pytest.raises(ValueError, match="tol_FROM"):
        charger(str(f))


def test_la_configuration_est_immuable(etude):
    """Un run ne doit pas pouvoir modifier sa propre configuration en cours
    de route -- c'est ce qui rendait `do_IS` difficile a suivre."""
    import dataclasses
    _, cfg, _ = etude
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.n0 = 99


def test_remplace_valide_le_resultat(etude):
    _, cfg, _ = etude
    assert cfg.remplace(n0=12).n0 == 12
    with pytest.raises(ValueError, match="n0"):
        cfg.remplace(n0=-1)


# --------------------------------------------------------------------------- #
def test_les_deux_etudes_ne_diffferent_que_par_leur_contenu_propre():
    """Mesure qui justifie l'existence de la phase 4 : l'inventaire des
    scripts AC donnait 77 variables communes dont 65 de meme valeur. Les
    fichiers d'etude doivent rester du meme ordre -- une dizaine de lignes,
    pas cinquante."""
    a = charger(os.path.join(REPO, "studies", "pure_flexion.toml"))
    b = charger(os.path.join(REPO, "studies", "moulin_blanc.toml"))
    differents = [c for c in a.en_dict()
                  if getattr(a, c) != getattr(b, c)]
    assert 5 <= len(differents) <= 25, \
        "%d champs different entre les deux etudes : %s" % (len(differents), differents)
