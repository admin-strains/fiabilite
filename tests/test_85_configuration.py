"""
La configuration declarative reproduit-elle celle des scripts AC ?

PHASE 4. `_config/schema.py` remplace les ~50 variables globales du bloc
`OPTIONS UTILISATEUR`.

CHANGEMENT D'ORACLE (phase 4b)
------------------------------
En phase 4a, les scripts AC portaient encore leurs affectations litterales :
ils etaient l'oracle, et ce fichier comparait le TOML a CE QUE FAISAIT LE
SCRIPT. C'est cette comparaison qui a prouve que la phase 4a ne changeait
aucune valeur.

La phase 4b debranche les scripts. L'oracle a donc ete recopie dans
`tests/golden/config_*.json` par `tools/golden_config.py`, A LA REVISION QUI
PRECEDE LE DEBRANCHEMENT (e80ef8b). C'est desormais lui la reference, et
`test_les_valeurs_sont_celles_des_scripts_ac` compare le TOML au golden.

Le risque d'un tel changement est connu : une comparaison a un oracle vide
passe sans rien prouver. Trois tests l'interdisent --
`test_la_couverture_est_reelle` exige au moins trente champs compares,
`test_le_golden_vient_bien_des_scripts_ac` verifie la provenance, et
`test_les_scripts_ac_ne_portent_plus_leur_configuration` verifie que le
debranchement a bien eu lieu (sans quoi le golden serait une copie inutile).

Ces tests ne demandent que `tomli` (stdlib a partir de Python 3.11).
"""

import ast
import io
import json
import os
import re
import sys

import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
for p in (os.path.join(REPO, "_config"),):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("tomli", reason="lecture TOML (tomllib a partir de 3.11)") \
    if sys.version_info < (3, 11) else None

from schema import (CRITERES_EFF, MODELES,  # noqa: E402
                    Configuration, charger, ecrire_trace, resume)

ETUDES = {
    "pure_flexion": os.path.join(REPO, "pure_flexion", "AC3_pure_flexion.py"),
    "moulin_blanc": os.path.join(REPO, "Moulinblanc", "AC3_moulinblanc.py"),
}

#: champs qui n'avaient pas d'equivalent litteral dans les scripts AC
HORS_COMPARAISON = {
    "storage",            # etait enfoui dans une concatenation de chemin
    "dossier_sortie",     # remplace `path_dir`, chemin absolu du poste de l'auteur
    "hf_2d_grid_fixed",   # None des deux cotes, mais parfois ecrit en expression
    "hf_3d_grid_fixed",
}

#: (etude, champ) -> raison. Ecarts VOULUS entre le script d'origine et le
#: fichier d'etude. La liste doit rester courte, et chaque entree porter sa
#: raison : c'est ici qu'on verra, dans six mois, ce qui a ete decide et par
#: qui. Un ecart non declare reste un echec.
ECARTS_ASSUMES = {
    ("moulin_blanc", "restart_enrich_only"):
        "Agnes, 26/08/2026 : le script portait `true`, l'etat de TRAVAIL de "
        "son auteur, qui reprenait un enrichissement depuis un "
        "`restart_state.json`. Ce dump vit dans le `.ds` et n'est pas dans le "
        "depot : l'etude etait injouable ailleurs. L'intention est de REJOUER "
        "LE RUN COMPLET, depuis le plan d'experiences.",
}

#: Le domaine de recherche, reduit de +/- 7,5 a +/- 6,0 sur le Moulin Blanc.
#: Meme raison pour les quatre champs, donc une seule redaction.
_RAISON_DOMAINE = (
    "Agnes, 26/08/2026 : BORNER LE DOMAINE. `fy ~ Normal(235 ; 30,15)` n'est "
    "pas bornee en bas -- u = -7,5 donne fy = 8,88 MPa (plus faible que le "
    "beton) et fy s'annule a u = -7,79. Le run du 26/08 est mort au point "
    "u = [+7,5 ; -7,5] : fy1/fy2 = 461/8,9, soit un rapport de 52 entre les "
    "deux groupes d'aciers, quand docs/mesh/ donne ~5 comme seuil de mauvais "
    "conditionnement des cones SOCP. Digital Structure a termine le processus "
    "sans exception ni trace, apres deux heures. 6,0 est un compromis assume : "
    "contenir le point de conception (|u| ~ 5,3) exigeait b >= 5,5, garder le "
    "rapport <= 5 exigeait b <= 5,2 -- aucune valeur ne satisfait les deux. "
    "Il reste 7,7 aux coins. CECI CHANGE LE RESULTAT DE L'ETUDE, et c'est "
    "voulu : le domaine precedent contenait des points que le modele "
    "mecanique ne sait pas representer. Tableau complet dans "
    "studies/moulin_blanc_fumee.toml.")
for _champ in ("u1_min", "u1_max", "u2_min", "u2_max"):
    ECARTS_ASSUMES[("moulin_blanc", _champ)] = _RAISON_DOMAINE
del _champ


def _golden(nom):
    chemin = os.path.join(TESTS, "golden", "config_%s.json" % nom)
    assert os.path.isfile(chemin), (
        "golden de configuration absent : %s\n"
        "Le regenerer NE VAUT RIEN une fois les scripts debranches : il doit "
        "etre relu d'une revision anterieure.\n"
        "  python tools/golden_config.py --revision e80ef8b" % chemin)
    return json.load(io.open(chemin, encoding="utf-8"))


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
        _golden(nom)["valeurs"]


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
        if (nom, champ) in ECARTS_ASSUMES:
            continue
        attendu, obtenu = ac[champ], getattr(cfg, champ)
        if isinstance(attendu, float) or isinstance(obtenu, float):
            egal = obtenu == pytest.approx(attendu, rel=1e-15)
        else:
            egal = obtenu == attendu
        if not egal:
            ecarts.append("%s : script=%r  config=%r" % (champ, attendu, obtenu))
    assert not ecarts, (
        "%s :\n  " % nom + "\n  ".join(ecarts)
        + "\n\nSi l'ecart est VOULU, l'inscrire dans ECARTS_ASSUMES avec sa "
          "raison et qui l'a decide.")


def test_les_ecarts_assumes_en_sont_vraiment(etude):
    """Le pendant : une entree de `ECARTS_ASSUMES` qui ne correspond plus a un
    ecart reel est une exemption qui traine. Elle masquerait le jour ou le
    champ se remettrait a diverger pour une AUTRE raison."""
    nom, cfg, ac = etude
    inutiles = []
    for (etude_nom, champ), _raison in ECARTS_ASSUMES.items():
        if etude_nom != nom or champ not in ac:
            continue
        if getattr(cfg, champ) == ac[champ]:
            inutiles.append(champ)
    assert not inutiles, (
        "%s : ECARTS_ASSUMES declare %s, mais le fichier d'etude a repris la "
        "valeur du script. Retirer l'entree." % (nom, inutiles))


def test_chaque_ecart_assume_porte_sa_raison():
    """Une exemption sans raison est une exemption qu'on ne pourra pas
    reexaminer."""
    maigres = [cle for cle, raison in ECARTS_ASSUMES.items() if len(raison) < 80]
    assert not maigres, "raison trop courte pour %s" % maigres


def test_la_couverture_est_reelle(etude):
    """Un test de comparaison qui ne compare rien passerait sans rien prouver.
    On exige donc qu'il porte sur au moins trente champs."""
    nom, cfg, ac = etude
    compares = [c for c in cfg.en_dict() if c not in HORS_COMPARAISON and c in ac]
    assert len(compares) >= 30, "%s : seulement %d champs compares" % (nom, len(compares))


def test_le_golden_vient_bien_des_scripts_ac(etude):
    """La provenance est ce qui fait la valeur du golden : il doit avoir ete
    releve sur le script AC, a une revision nommee, pas ecrit a la main."""
    nom, _, _ = etude
    meta = _golden(nom)
    assert meta["_script"].endswith(".py")
    assert os.path.isfile(os.path.join(REPO, meta["_script"].replace("/", os.sep)))
    assert re.fullmatch(r"[0-9a-f]{7,40}", meta["_revision"]), \
        "revision %r : le golden doit venir d'une revision git figee" % meta["_revision"]
    assert meta["_n_affectations"] >= 50


# --------------------------------------------------------------------------- #
# Le debranchement a-t-il eu lieu ?                                           #
# --------------------------------------------------------------------------- #
#: noms encore affectes litteralement dans les scripts : ce ne sont pas des
#: parametres mais des accumulateurs d'etat, remis a zero a chaque run.
ETAT_ATTENDU = {
    "_gepck_pce_label", "_gepck_loo", "_eff_history_EFF", "_eff_history_BB",
    "_eff_history_BS", "_eff_history_theta", "_eff_history_Pf",
    "_eff_history_beta_IS", "_fosm", "_point_log_phase",
    "_point_log_round", "_enrich_round", "_round_sizes_prev", "_restart_xt_eff",
    "_socp_call_counter", "_solveurs",     # phase 5 : compteur et cache de solveurs
    "slice_def_final",                     # valeur de depart, reecrite plus bas
}

#: ce qui reste dans les scripts sans etre ni de l'etat ni un reglage : la
#: DONNEE de l'etude. La phase 4 portait sur les ~50 parametres de reglage,
#: pas sur le catalogue des variables aleatoires ni sur les proprietes du
#: materiau. Ces noms sont recenses pour que la liste ne grossisse pas en
#: silence, pas parce qu'ils seraient a leur place definitive.
DONNEES_D_ETUDE = {
    "PARAM_CONFIG_LOAD",       # lois et regions de sensibilite des variables
    "FY_MEAN",                 # limite d'elasticite moyenne (Moulin Blanc)
    # eff_bounds_* : bornes de la recherche EFF. Ce sont bel et bien des
    # reglages, et ils ont vocation a rejoindre le schema -- ils valent
    # [-7.5, +7.5] dans les deux etudes. Non fait : ils n'apparaissent comme
    # litteraux que dans un des deux scripts (l'autre les ecrit [-7.5]*n_var),
    # ce qui demande de choisir une forme commune d'abord.
    "eff_bounds_min", "eff_bounds_max",
}


@pytest.mark.parametrize("nom", sorted(ETUDES))
def test_les_scripts_ac_ne_portent_plus_leur_configuration(nom):
    """Un parametre qui reste ecrit en dur dans le script court-circuite le
    fichier d'etude : deux sources de verite, dont une invisible."""
    champs = {f for f in Configuration(modelname="x").en_dict()}
    restants = sorted(set(_config_du_script(ETUDES[nom])) & champs)
    assert not restants, (
        "%s porte encore %d parametre(s) en dur : %s\n"
        "Les deplacer dans studies/%s.toml." % (nom, len(restants), restants, nom))


@pytest.mark.parametrize("nom", sorted(ETUDES))
def test_seuls_les_accumulateurs_d_etat_restent(nom):
    """Le pendant du test precedent : ce qui reste doit etre de l'etat ou de
    la donnee d'etude, et cet inventaire ne doit pas grossir en douce."""
    restants = set(_config_du_script(ETUDES[nom]))
    inattendus = sorted(restants - ETAT_ATTENDU - DONNEES_D_ETUDE)
    assert not inattendus, (
        "%s : affectations litterales inattendues : %s\n"
        "Soit ce sont des reglages -- ils vont dans studies/%s.toml -- soit "
        "c'est de l'etat ou de la donnee, et il faut les recenser ici en "
        "disant lequel." % (nom, inattendus, nom))


@pytest.mark.parametrize("nom", sorted(ETUDES))
def test_la_configuration_d_essai_reste_injectable(nom):
    """`tools/run_comparatif.py` impose une configuration d'essai en inserant
    un `CFG = CFG.remplace(...)` juste apres le chargement du fichier d'etude.
    Si ce point d'ancrage disparait, l'outil ne patcherait plus rien -- et les
    deux versions comparees tourneraient avec des reglages differents, ce qui
    invaliderait la comparaison sans le dire.

    Le test va jusqu'au bout : il produit le script patche, verifie qu'il se
    compile, et que la surcharge precede bien l'impression du resume. C'est
    exactement le defaut qu'avait la premiere version de l'outil -- le journal
    annoncait `n_max_EFF_points=30` pendant que le run en appliquait 8."""
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import run_comparatif                                    # noqa: PLC0415

    src = io.open(ETUDES[nom], encoding="utf-8", errors="replace").read()
    m = run_comparatif.ANCRE.search(src.replace("\r\n", "\n"))
    assert m, "%s : point d'injection de run_comparatif.py introuvable" % nom

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        # `cible` explicite : ecrire a cote de l'AC ecraserait -- et, au
        # nettoyage, supprimerait -- le script d'un run en cours.
        patche = run_comparatif.patcher(ETUDES[nom], 8, os.path.join(tmp, "out"),
                                        cible=os.path.join(tmp, "patche.py"))
        texte = io.open(patche, encoding="utf-8").read()
        ast.parse(texte)                            # le script patche doit compiler
        i_remplace = texte.index("CFG = CFG.remplace(")
        i_resume = texte.index("_schema.resume(CFG)")
        assert i_remplace < i_resume, \
            "%s : la surcharge est appliquee APRES l'impression du resume" % nom
        for attendu in ("n_max_EFF_points=8", "n_workers_DOE=1", "save_history=False"):
            assert attendu in texte.replace(" ", ""), attendu


# --------------------------------------------------------------------------- #
# Les parametres sans effet                                                   #
# --------------------------------------------------------------------------- #
def _noms_lus(chemin):
    """Noms lus par le script, hors bloc de liaison `nom = CFG.nom`.

    Le bloc de liaison est retire AVANT la recherche : sans cela, la ligne
    `reduc_PLS = CFG.reduc_PLS` compterait comme une lecture et un parametre
    mort passerait pour vivant.
    """
    src = io.open(chemin, encoding="utf-8", errors="replace").read()
    return re.sub(r"(?m)^\s+\w+\s+=\s+CFG\.\w+\s*$", "", src)


def test_la_liste_des_parametres_sans_effet_est_exacte():
    """Un parametre qu'aucun code ne lit et qui accepte quand meme une valeur
    est un piege : l'utilisateur croit avoir change quelque chose.

    Cette liste est tenue par la MESURE, pas a la main. Si quelqu'un recable
    l'un de ces chemins, ce test tombe et demande de retirer le nom de
    `SANS_EFFET` -- au lieu de laisser le refus de `valider()` bloquer un
    parametre redevenu utile.
    """
    from schema import SANS_EFFET                            # noqa: PLC0415
    ressuscites = []
    for nom, chemin in sorted(ETUDES.items()):
        corps = _noms_lus(chemin)
        for champ in SANS_EFFET:
            if re.search(r"\b%s\b" % re.escape(champ), corps):
                ressuscites.append("%s : %s est de nouveau lu" % (nom, champ))
    assert not ressuscites, "\n  ".join(ressuscites)


def test_aucun_autre_parametre_n_est_mort_en_silence():
    """Le pendant : un champ du schema que plus aucun script ne lit doit
    rejoindre `SANS_EFFET`, pas rester un reglage sans effet."""
    from schema import SANS_EFFET                            # noqa: PLC0415
    champs = set(Configuration(modelname="x").en_dict())
    # champs consommes par l'outillage et non par les scripts AC
    HORS_SCRIPTS = {"modelname", "storage", "dossier_sortie",
                    "hf_2d_grid_fixed", "hf_3d_grid_fixed"}
    corps = "\n".join(_noms_lus(c) for c in ETUDES.values())
    morts = sorted(champ for champ in champs - HORS_SCRIPTS - set(SANS_EFFET)
                   if not re.search(r"\b%s\b" % re.escape(champ), corps))
    assert not morts, (
        "parametre(s) que plus aucun script ne lit : %s\n"
        "Les ajouter a SANS_EFFET (avec la raison) ou recabler leur usage." % morts)


@pytest.mark.parametrize("champ", ["reduc_PLS", "do_analytic_grad",
                                   "max_of_maxdegree", "seuil_pce"])
def test_regler_un_parametre_sans_effet_est_refuse(champ):
    from schema import SANS_EFFET, _DEFAUTS                  # noqa: PLC0415
    assert champ in SANS_EFFET
    defaut = _DEFAUTS[champ]
    autre = (defaut + 1) if isinstance(defaut, (int, float)) and not isinstance(defaut, bool) \
        else (not defaut)
    with pytest.raises(ValueError, match=champ):
        Configuration(modelname="x", **{champ: autre}).valider()


def test_les_etudes_reelles_restent_acceptees():
    """Corollaire : le refus ci-dessus ne doit pas casser les deux etudes, qui
    posent ces champs a leur valeur par defaut."""
    for nom in ETUDES:
        charger(os.path.join(REPO, "studies", nom + ".toml"))


@pytest.mark.parametrize("nom", sorted(ETUDES))
def test_aucun_chemin_absolu_du_poste_de_l_auteur(nom):
    """Le but de la reprise est que ce code s'installe n'importe ou. Un chemin
    en dur suffit a le rendre inexecutable ailleurs -- c'est ce qui rendait le
    DOE parallele intestable : ses workers pointaient sur `launcher3.py`, une
    copie du lanceur portant `C:\\_workingDir\\_SF\\test flexion\\_lib`."""
    src = io.open(ETUDES[nom], encoding="utf-8", errors="replace").read()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    for interdit in (r"C:\_workingDir", r"C:\workspace\front",
                     r"C:\workspace\storage"):
        assert interdit not in code, "%s : chemin en dur %s" % (nom, interdit)


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


@pytest.mark.parametrize("nom", sorted(ETUDES))
def test_les_scripts_lisent_l_effet_et_non_l_intention(nom):
    """Corollaire du test precedent : le bloc de liaison doit lier `do_EFF` a
    `eff_actif`, pas a `do_EFF`. Sinon la correction en haute fidelite, que
    les scripts appliquaient a la main, serait silencieusement perdue."""
    src = io.open(ETUDES[nom], encoding="utf-8", errors="replace").read()
    assert re.search(r"(?m)^\s+do_EFF\s*=\s*CFG\.eff_actif\b", src), nom
    assert re.search(r"(?m)^\s+do_IS\s*=\s*CFG\.is_actif\b", src), nom


def test_chemin_du_modele(etude):
    _, cfg, _ = etude
    assert cfg.chemin_ds.endswith(cfg.modelname + ".ds")
    assert cfg.storage in cfg.chemin_ds


# --------------------------------------------------------------------------- #
# Tracabilite d'un run                                                        #
# --------------------------------------------------------------------------- #
def test_le_resume_porte_les_champs_decisifs(etude):
    """Un journal de run qui ne porte pas sa configuration ne peut pas etre
    compare a un autre. Mesure du 25/08/2026 : sur cette chaine, un ecart de
    configuration et un ecart de code se lisent pareil.

    « Decisif » veut dire : tout ce qui definit l'ETUDE, et tout ce qui decrit
    la SESSION. Les parametres sans effet, eux, n'ont rien a faire dans un
    journal -- les y mettre laisserait croire qu'ils comptent.
    """
    from schema import CATEGORIES                              # noqa: PLC0415
    nom, cfg, _ = etude
    texte = resume(cfg)
    for champ, categorie in CATEGORIES.items():
        if categorie in ("etude", "session") and champ not in ("modelname", "storage"):
            assert champ in texte, "%s : %s absent du resume" % (nom, champ)
    for champ, categorie in CATEGORIES.items():
        if categorie == "sans_effet":
            assert champ not in texte, \
                "%s : %s est sans effet, il ne doit pas figurer au journal" % (nom, champ)
    assert cfg.modelname in texte or cfg.chemin_ds in texte


def test_le_resume_signale_une_intention_corrigee():
    """En haute fidelite, `do_EFF = True` n'enrichit rien : le resume doit le
    dire, sinon le journal affiche une intention que le run ne suit pas."""
    cfg = Configuration(modelname="x", modele="HF", do_EFF=True, do_IS=True)
    assert "CORRIGE" in resume(cfg)
    assert "CORRIGE" not in resume(Configuration(modelname="x", modele="PCK"))


def test_la_trace_json_est_relisible(etude, tmp_path):
    nom, cfg, _ = etude
    cible = ecrire_trace(cfg, str(tmp_path))
    relu = json.load(io.open(cible, encoding="utf-8"))
    for champ, valeur in cfg.en_dict().items():
        if isinstance(valeur, (str, int, float, bool)) or valeur is None:
            assert relu[champ] == valeur, champ
    assert relu["_derive"]["eff_actif"] == cfg.eff_actif
    assert relu["_origine"].endswith("%s.toml" % nom)


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


# --------------------------------------------------------------------------- #
# A qui appartient chaque parametre                                           #
# --------------------------------------------------------------------------- #
def test_chaque_champ_est_classe():
    """Question d'Agnes, 26/08/2026 : quelles options sont liees a un run
    utilisateur, et lesquelles sont internes au code ? Elle ne l'etait pas.

    Un champ non classe est un champ dont personne ne sait s'il definit
    l'ETUDE, decrit la SESSION, ou ne change que la SORTIE. C'est ainsi que
    `restart_enrich_only` -- un mode de session -- s'est retrouve fige dans la
    definition d'une etude, la rendant injouable ailleurs que sur le poste ou
    le dump de reprise existait.
    """
    from schema import CATEGORIES                              # noqa: PLC0415
    champs = set(Configuration(modelname="x").en_dict())
    non_classes = sorted(champs - set(CATEGORIES))
    assert not non_classes, (
        "champ(s) sans categorie : %s\n"
        "Choisir dans CATEGORIES de _config/schema.py : 'etude' (definit le "
        "resultat), 'session' (ce run sur ce poste), 'sortie' (ce qui est "
        "trace), 'sans_effet'." % non_classes)


def test_aucune_categorie_inventee():
    from schema import CATEGORIES                              # noqa: PLC0415
    champs = set(Configuration(modelname="x").en_dict())
    assert set(CATEGORIES) <= champs, \
        "CATEGORIES nomme des champs inexistants : %s" % sorted(set(CATEGORIES) - champs)
    valeurs = set(CATEGORIES.values())
    assert valeurs <= {"etude", "session", "sortie", "sans_effet"}, valeurs


def test_les_sans_effet_sont_les_memes_des_deux_cotes():
    """Deux listes qui decrivent le meme fait ne doivent pas diverger."""
    from schema import CATEGORIES, SANS_EFFET                  # noqa: PLC0415
    classes = {c for c, v in CATEGORIES.items() if v == "sans_effet"}
    assert classes == set(SANS_EFFET), \
        "CATEGORIES=%s  SANS_EFFET=%s" % (sorted(classes), sorted(SANS_EFFET))


def test_aucun_parametre_de_session_ne_change_le_resultat():
    """Le critere qui definit la categorie « session » : deux runs qui ne
    different que par elle doivent rendre le meme resultat. On ne peut pas le
    prouver ici sans lancer un calcul -- on verifie donc la propriete qui le
    rend possible : aucun de ces champs n'entre dans les valeurs derivees."""
    from schema import CATEGORIES                              # noqa: PLC0415
    session = {c for c, v in CATEGORIES.items() if v == "session"}
    a = Configuration(modelname="x")
    b = Configuration(modelname="x", n_workers_DOE=1, config_is_identical=False,
                      restart_enrich_only=True, save_history=True,
                      dossier_sortie="/ailleurs")
    for derive in ("do_GEPCK", "do_PCK", "do_HF", "eff_actif", "is_actif",
                   "bornes_u"):
        assert getattr(a, derive) == getattr(b, derive), derive
    assert session  # la categorie n'est pas vide


def test_les_sorties_qui_coutent_des_appels_solveur_sont_recensees():
    """Un parametre de trace qui declenche des centaines d'appels au solveur
    n'est pas anodin : sur le Moulin Blanc, un appel coute 466 s, donc une
    grille 15x15 represente 29 heures POUR UNE FIGURE. Le resume de journal
    les affiche pour qu'on le sache avant de lancer, pas apres."""
    from schema import CATEGORIES, COUTE_DES_APPELS_SOLVEUR    # noqa: PLC0415
    for nom in COUTE_DES_APPELS_SOLVEUR:
        assert CATEGORIES.get(nom) == "sortie", \
            "%s coute des appels solveur mais n'est pas une sortie" % nom
        assert len(COUTE_DES_APPELS_SOLVEUR[nom]) > 15, \
            "%s : dire CE QUE ca coute, pas seulement que ca coute" % nom


def test_le_resume_separe_l_etude_de_la_session(etude):
    """Le resume doit rendre la distinction lisible dans le journal : c'est
    la seule trace qui restera quand on relira un run dans six mois."""
    _, cfg, _ = etude
    texte = resume(cfg)
    assert "ETUDE" in texte and "SESSION" in texte
    assert texte.index("ETUDE") < texte.index("SESSION"), \
        "l'etude se lit avant la session : c'est elle qui definit le resultat"
