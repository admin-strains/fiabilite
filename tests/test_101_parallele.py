r"""L'evaluation en parallele : un code qui n'avait jamais tourne.

CE QU'IL ETAIT
---------------
`run_DOE_parallel` et `run_HF_grid_parallel`, dans chacun des deux scripts :
QUATRE copies de 115 lignes qui ne differaient que par le nom des
sous-dossiers et la facon de ranger le resultat.

Et surtout : **ce chemin de code ne pouvait pas s'executer**. Les workers
passaient par `launcher3.py`, une copie du lanceur portant en dur les chemins
du poste de l'auteur (`C:\_workingDir\_SF\test flexion\_lib`). Ce chemin
n'existant nulle part ailleurs, la branche parallele echouait partout sauf
la -- ce qui explique qu'elle n'ait jamais ete couverte, ni sans doute
executee depuis.

POURQUOI ELLE EST TESTABLE MAINTENANT
--------------------------------------
Le lanceur de processus est un ARGUMENT. Ces tests verifient tout ce qui se
passe autour du solveur -- repartition, copies isolees, fichiers de tache,
variables d'environnement, collecte, detection d'un worker mort -- sans en
lancer un seul.

DEUX GARDES QUI N'EXISTAIENT PAS
---------------------------------
* un worker qui ne rend AUCUNE sortie leve, en nommant son journal ;
* un plan TROUE -- des points sans resultat alors que tous les workers ont
  rendu quelque chose -- leve aussi. L'original remplissait `SOL` avec ce
  qu'il avait recu et laissait le trou se manifester plus loin, sous forme
  de `KeyError`.
"""

import json
import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
if os.path.join(_REPO, "_doe") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "_doe"))

import parallele as _parallele                        # noqa: E402

PARAMS = ["fc", "fy"]


class _FauxProcessus:
    """Un worker qui ecrit sa sortie sans rien calculer."""

    def __init__(self, argv, env, cwd, journal, resultats=None, rc=0):
        self.argv, self.env, self.cwd, self.journal = argv, env, cwd, journal
        self.rc = rc
        self._resultats = resultats

    def wait(self, *a, **kw):
        if self._resultats is not None:
            with open(self.env["_DOE_OUT"], "w") as fh:
                json.dump(self._resultats, fh)
        return self.rc


def _lanceur(calcul=None, rc=0, muet=False):
    """Fabrique un `lancer(...)` qui note ce qu'on lui demande."""
    trace = []

    def lancer(argv, env, cwd, journal):
        tache = json.load(open(env["_DOE_WORKER"]))
        trace.append({"argv": argv, "env": env, "cwd": cwd,
                      "journal": journal, "points": tache["points"]})
        if muet:
            return _FauxProcessus(argv, env, cwd, journal, None, rc)
        res = {str(p["idx"]): (calcul(p) if calcul
                               else {"g": float(p["idx"]), "dg_fc": 0.5,
                                     "dg_fy": 0.25})
               for p in tache["points"]}
        return _FauxProcessus(argv, env, cwd, journal, res, rc)

    lancer.trace = trace
    return lancer


def _modele(tmp_path, nom="essai"):
    """Un `.ds` minimal : les deux fichiers que le solveur reecrit."""
    base = tmp_path / (nom + ".ds")
    base.mkdir()
    (base / "dsCad.txt").write_text("b = 1.0\n", encoding="utf-8")
    (base / "dsLoad.txt").write_text("Z='-1.0'\n", encoding="utf-8")
    return str(tmp_path), nom


def _points(n):
    return [{"fc": 40.0 + i, "fy": 500.0 + i} for i in range(n)]


# --------------------------------------------------------------------- #
# la repartition
# --------------------------------------------------------------------- #
def test_les_points_sont_repartis_en_tourniquet():
    """Et non par tranches contigues : le cout d'un point varie fortement
    avec sa position dans le domaine, et des tranches donneraient un worker
    charge et trois oisifs."""
    assert _parallele.repartir(7, 3) == [[0, 3, 6], [1, 4], [2, 5]]


def test_on_ne_lance_pas_plus_de_workers_que_de_points():
    lots = _parallele.repartir(2, 8)
    assert len(lots) == 2 and sum(len(l) for l in lots) == 2


def test_un_seul_worker_prend_tout():
    assert _parallele.repartir(4, 1) == [[0, 1, 2, 3]]


def test_un_plan_vide_ne_fait_pas_diviser_par_zero():
    assert _parallele.repartir(0, 4) == [[]]


# --------------------------------------------------------------------- #
# la copie isolee du modele
# --------------------------------------------------------------------- #
def test_chaque_worker_recoit_SA_copie_du_modele(tmp_path):
    """Digital Structure REECRIT `dsCad.txt` a chaque evaluation : deux
    workers dans le meme dossier se detruiraient l'un l'autre."""
    storage, nom = _modele(tmp_path)
    lancer = _lanceur()
    _parallele.evaluer_en_parallele(_points(4), PARAMS, storage, nom, 2,
                                    script_etude="etude.py", repo=_REPO,
                                    lancer=lancer, tracer=lambda _m: None)
    dossiers = {t["cwd"] for t in lancer.trace}
    assert len(dossiers) == 2, "les workers partagent un dossier"
    for d in dossiers:
        assert os.path.exists(os.path.join(d, "dsCad.txt"))
        assert os.path.exists(os.path.join(d, "dsLoad.txt"))
        assert os.path.abspath(d) != os.path.abspath(
            os.path.join(storage, nom + ".ds")), (
            "un worker travaille dans le modele PRINCIPAL")


def test_une_sortie_d_un_run_precedent_est_effacee(tmp_path):
    """Sinon elle serait relue comme celle de ce run-ci -- des valeurs d'un
    autre domaine, servies comme etant celles-ci."""
    storage, nom = _modele(tmp_path)
    wds, _tache, sortie = _parallele.preparer_worker(
        storage, os.path.join(storage, nom + ".ds"), "w0",
        {0: {"fc": 40.0, "fy": 500.0}}, PARAMS)
    with open(sortie, "w") as fh:
        json.dump({"0": {"g": -999.0}}, fh)
    _parallele.preparer_worker(storage, os.path.join(storage, nom + ".ds"),
                               "w0", {0: {"fc": 40.0, "fy": 500.0}}, PARAMS)
    assert not os.path.exists(sortie)


def test_la_tache_porte_l_indice_et_les_variables(tmp_path):
    """Sans l'indice, les resultats ne se recollent pas dans le bon ordre."""
    storage, nom = _modele(tmp_path)
    _wds, tache, _sortie = _parallele.preparer_worker(
        storage, os.path.join(storage, nom + ".ds"), "w0",
        {3: {"fc": 41.0, "fy": 501.0}, 7: {"fc": 42.0, "fy": 502.0}}, PARAMS)
    points = json.load(open(tache))["points"]
    assert [p["idx"] for p in points] == [3, 7]
    assert points[0]["fc"] == 41.0 and points[1]["fy"] == 502.0


# --------------------------------------------------------------------- #
# l'environnement du worker
# --------------------------------------------------------------------- #
def test_la_charge_MKL_est_repartie_entre_les_workers(tmp_path):
    """Sans cela chaque worker croit disposer de toute la machine, et quatre
    solveurs se disputent les memes coeurs."""
    storage, nom = _modele(tmp_path)
    lancer = _lanceur()
    _parallele.evaluer_en_parallele(_points(8), PARAMS, storage, nom, 4,
                                    script_etude="etude.py", repo=_REPO,
                                    lancer=lancer, tracer=lambda _m: None)
    attendu = str(_parallele.FILS_MKL_DISPONIBLES // 4)
    for t in lancer.trace:
        assert t["env"]["MKL_NUM_THREADS"] == attendu
        assert t["env"]["OMP_NUM_THREADS"] == attendu


def test_le_worker_sait_ou_est_le_modele_principal(tmp_path):
    """Seuls `dsCad` et `dsLoad` sont copies ; le STEP et le reste sont relus
    depuis le modele principal."""
    storage, nom = _modele(tmp_path)
    lancer = _lanceur()
    _parallele.evaluer_en_parallele(_points(2), PARAMS, storage, nom, 2,
                                    script_etude="etude.py", repo=_REPO,
                                    lancer=lancer, tracer=lambda _m: None)
    for t in lancer.trace:
        assert t["env"]["_DOE_MAIN_DS"] == os.path.join(storage, nom + ".ds")
        assert t["env"]["_FIAB_LOG_REDIRECTED"] == "1", (
            "sans ce drapeau le worker ouvre une fenetre matplotlib")


def test_le_worker_est_lance_par_le_lanceur_du_depot(tmp_path):
    """C'est LE defaut historique : les workers passaient par `launcher3.py`,
    qui portait en dur les chemins du poste de l'auteur."""
    storage, nom = _modele(tmp_path)
    lancer = _lanceur()
    _parallele.evaluer_en_parallele(_points(2), PARAMS, storage, nom, 1,
                                    script_etude="mon_etude.py", repo=_REPO,
                                    lancer=lancer, tracer=lambda _m: None)
    argv = lancer.trace[0]["argv"]
    assert argv[1] == os.path.join(_REPO, "launcher.py"), (
        "le worker doit passer par le lanceur du DEPOT, jamais par une copie")
    assert "--garder-cwd" in argv, (
        "le worker travaille dans SA copie : le lanceur ne doit pas revenir "
        "au dossier de l'etude")
    assert argv[-1] == "mon_etude.py"


# --------------------------------------------------------------------- #
# la collecte
# --------------------------------------------------------------------- #
def test_les_resultats_reviennent_dans_le_bon_ordre(tmp_path):
    """Le tourniquet melange les indices : c'est la collecte par indice qui
    les remet en place."""
    storage, nom = _modele(tmp_path)
    lancer = _lanceur(calcul=lambda p: {"g": p["fc"], "dg_fc": 1.0, "dg_fy": 2.0})
    res = _parallele.evaluer_en_parallele(_points(5), PARAMS, storage, nom, 3,
                                          script_etude="e.py", repo=_REPO,
                                          lancer=lancer, tracer=lambda _m: None)
    assert sorted(res) == [0, 1, 2, 3, 4]
    for i in range(5):
        assert res[i]["g"] == pytest.approx(40.0 + i)


def test_un_worker_sans_sortie_LEVE_et_nomme_son_journal(tmp_path):
    """Un worker qui meurt en silence rendrait un plan incomplet sans le
    dire."""
    storage, nom = _modele(tmp_path)
    with pytest.raises(RuntimeError) as err:
        _parallele.evaluer_en_parallele(_points(3), PARAMS, storage, nom, 2,
                                        script_etude="e.py", repo=_REPO,
                                        lancer=_lanceur(muet=True),
                                        tracer=lambda _m: None)
    assert "sans sortie" in str(err.value)
    assert "_doe_worker.log" in str(err.value), (
        "le message doit dire OU regarder")


def test_un_plan_troue_LEVE_au_lieu_de_passer_pour_complet(tmp_path):
    """Nouveau garde : l'original remplissait `SOL` avec ce qu'il avait recu
    et laissait le trou se manifester plus loin, en `KeyError`."""
    storage, nom = _modele(tmp_path)

    def lancer(argv, env, cwd, journal):
        tache = json.load(open(env["_DOE_WORKER"]))
        # le worker « oublie » son premier point
        res = {str(p["idx"]): {"g": 0.0} for p in tache["points"][1:]}
        return _FauxProcessus(argv, env, cwd, journal, res)

    with pytest.raises(RuntimeError) as err:
        _parallele.evaluer_en_parallele(_points(6), PARAMS, storage, nom, 2,
                                        script_etude="e.py", repo=_REPO,
                                        lancer=lancer, tracer=lambda _m: None)
    assert "sans resultat" in str(err.value)


def test_le_journal_dit_combien_de_workers_et_de_points(tmp_path):
    messages = []
    storage, nom = _modele(tmp_path)
    _parallele.evaluer_en_parallele(_points(6), PARAMS, storage, nom, 3,
                                    script_etude="e.py", repo=_REPO,
                                    lancer=_lanceur(), tracer=messages.append)
    texte = "\n".join(messages)
    assert "6 pts -> 3 workers" in texte
    assert texte.count("<- worker") == 3, "chaque worker doit rendre son code"


# --------------------------------------------------------------------- #
# les deux etudes partagent le meme mecanisme
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_les_scripts_ne_recopient_plus_le_mecanisme(script):
    src = open(os.path.join(_REPO, script), encoding="utf-8",
               errors="replace").read()
    # Les etudes passent desormais par les ADAPTATEURS -- le plan et la
    # grille -- qui portent chacun leur mise en forme. Le mecanisme, lui,
    # reste unique dans `_doe/parallele.py`.
    assert ("_parallele.evaluer_plan_en_parallele(" in src
            and "_parallele.evaluer_points_en_parallele(" in src), (
        "%s : le parallele ne passe plus par `_doe/parallele.py`" % script)
    for parti in ("_DOE_WORKER=", "MKL_NUM_THREADS", "shutil.copy2",
                  "_sp.Popen"):
        assert parti not in src, (
            "%s porte encore `%s` : le mecanisme appartient a "
            "`_doe/parallele.py`." % (script, parti))


# --------------------------------------------------------------------- #
# LES DEUX ADAPTATEURS ET LA FABRIQUE (28/08/2026)
# --------------------------------------------------------------------- #
def test_un_plan_reparti_est_RECOPIE_dans_la_liste_de_l_appelant():
    """Les workers rendent un dictionnaire indexe ; l'appelant travaille sur
    la liste qu'il a construite. L'adaptateur fait le pont, en place."""
    SOL = [{"fc": 1.0, "fy": 2.0}, {"fc": 3.0, "fy": 4.0}]
    rendu = {0: {"g": 0.5, "dg_fc": 1.0, "dg_fy": 2.0},
             1: {"g": 1.5, "dg_fc": 3.0, "dg_fy": 4.0}}
    vu = {}

    def faux_pool(SOL_, params, storage, base, n, script_etude=None, repo=None,
                  lancer=None, **reste):
        # `**reste` : cette doublure suit une signature reelle, et le jour ou
        # elle en diverge le test doit tomber sur SON sujet -- le pont entre
        # le dictionnaire indexe et la liste -- pas sur un mot-clef de plus.
        vu["n"] = n
        return rendu

    reel = _parallele.evaluer_en_parallele
    _parallele.evaluer_en_parallele = faux_pool
    try:
        out = _parallele.evaluer_plan_en_parallele(
            SOL, ["fc", "fy"], "sto", "modele", 3,
            script_etude="etude.py", repo="repo")
    finally:
        _parallele.evaluer_en_parallele = reel
    assert out is SOL, "la liste doit etre remplie EN PLACE"
    assert SOL[0]["g"] == 0.5 and SOL[1]["dg_fy"] == 4.0
    assert vu["n"] == 3


def test_un_gradient_absent_reste_None_et_ne_devient_pas_zero():
    """C'est ce que le solveur a dit. Un zero affirmerait un etat limite
    plat -- `_doe/plan.assembler_plan` decide ensuite quoi en faire."""
    SOL = [{"fc": 1.0, "fy": 2.0}]

    def faux_pool(*a, **k):
        return {0: {"g": 0.5}}          # aucun gradient

    reel = _parallele.evaluer_en_parallele
    _parallele.evaluer_en_parallele = faux_pool
    try:
        _parallele.evaluer_plan_en_parallele(SOL, ["fc", "fy"], "s", "m", 2,
                                             script_etude="e", repo="r")
    finally:
        _parallele.evaluer_en_parallele = reel
    assert SOL[0]["dg_fc"] is None and SOL[0]["dg_fy"] is None


def test_des_points_STANDARD_passent_par_la_transformation_avant_le_pool():
    """Le solveur ne connait que les variables physiques."""
    ot = pytest.importorskip("openturns")
    dist = ot.JointDistribution([ot.Normal(235.0, 30.0), ot.Normal(30.0, 4.5)])
    recu = {}

    def faux_pool(points, params, storage, modelname, n, **k):
        recu["points"] = [dict(p) for p in points]
        return {i: {"g": float(i)} for i in range(len(points))}

    reel = _parallele.evaluer_en_parallele
    _parallele.evaluer_en_parallele = faux_pool
    try:
        g = _parallele.evaluer_points_en_parallele(
            [[0.0, 0.0], [1.0, -1.0]], dist, ["fy", "fc"], "s", "m", 2,
            script_etude="e", repo="r")
    finally:
        _parallele.evaluer_en_parallele = reel
    assert recu["points"][0] == {"fy": 235.0, "fc": 30.0}
    assert recu["points"][1]["fy"] == pytest.approx(265.0)
    assert recu["points"][1]["fc"] == pytest.approx(25.5)
    assert g == [0.0, 1.0], "les g doivent revenir DANS L'ORDRE demande"


def test_ce_module_n_a_aucune_dependance_a_OpenTURNS():
    """Il ne fait que du sous-processus et du fichier. La transformation
    isoprobabiliste lui arrive DEJA construite, et `T_inv(list(u))` suffit --
    inutile de lui imposer OpenTURNS pour un `ot.Point`."""
    import io as _io
    src = _io.open(_parallele.__file__, encoding="utf-8").read()
    assert "import openturns" not in src


def test_la_fabrique_memoise_pour_garder_le_compteur_d_appels():
    """La memoisation n'est pas une optimisation : le solveur porte un
    compteur d'appels, et en reconstruire un le remettrait a zero."""
    construits = []

    def fabriquer(chemin_ds=None, **reglages):
        construits.append((chemin_ds, reglages))
        return "solveur-%s" % chemin_ds

    solveur = _parallele.fabrique_memoisee(
        fabriquer, "principal", lambda nom: nom + ".ds", taille=0.05)
    assert solveur() == "solveur-principal.ds"
    assert solveur() is solveur()
    assert len(construits) == 1, "le solveur a ete reconstruit"
    assert construits[0][1] == {"taille": 0.05}


def test_chaque_worker_a_SON_solveur():
    """Un worker travaille sur SA copie du `.ds` : le nom lui est impose par
    le processus pere."""
    solveur = _parallele.fabrique_memoisee(
        lambda chemin_ds=None, **k: chemin_ds, "principal",
        lambda nom: nom + ".ds")
    assert solveur("w0") == "w0.ds"
    assert solveur("w1") == "w1.ds"
    assert solveur() == "principal.ds"
    assert set(solveur.cache) == {"w0", "w1", "principal"}


# --------------------------------------------------------------------- #
# LA REPRISE : ce qui est deja paye ne repart pas au solveur
#
# DEFAUT MESURE LE 29/08/2026. `evaluer_plan` saute depuis toujours une
# entree qui porte deja `g` -- c'est tout l'interet du cache partiel. Le pool,
# lui, ne sautait rien : le worker RECONSTRUIT son dictionnaire a partir des
# seuls parametres physiques, et le `g` greffe disparaissait en route.
#
# L'appelant, lui, annoncait la reprise :
#
#     [DOE PARTIEL] 3/5 points repris du cache -> autant de SOCP evites
#
# et payait les cinq. Un journal qui affirme une economie qu'il ne fait pas
# est pire qu'un journal muet.
#
#     voie sequentielle   2 points payes sur 5
#     voie parallele      5 points payes sur 5
#
# Sur le Moulin Blanc (`n0 = 5`, six workers, `config_is_identical`), cinq
# appels solveur -- environ 39 minutes.
# --------------------------------------------------------------------- #
def _points_dont_deja(n, deja):
    """`n` points, dont les indices `deja` portent leur resultat."""
    pts = _points(n)
    for i in deja:
        pts[i]["g"] = 100.0 + i
        pts[i]["dg_fc"], pts[i]["dg_fy"] = 1.0 + i, 2.0 + i
    return pts


def test_un_point_deja_calcule_ne_repart_pas_au_solveur(tmp_path):
    storage, nom = _modele(tmp_path)
    lancer = _lanceur()
    _parallele.evaluer_en_parallele(
        _points_dont_deja(5, [0, 1, 2]), PARAMS, storage, nom, 6,
        script_etude="etude.py", repo=_REPO, lancer=lancer,
        tracer=lambda _m: None)
    envoyes = sorted(p["idx"] for t in lancer.trace for p in t["points"])
    assert envoyes == [3, 4], (
        "le pool a repaye des points deja calcules : %s" % envoyes)


def test_le_resultat_deja_connu_est_rendu_tel_quel(tmp_path):
    """Il doit ressortir a son INDICE GLOBAL, melange aux points calcules."""
    storage, nom = _modele(tmp_path)
    res = _parallele.evaluer_en_parallele(
        _points_dont_deja(5, [0, 2]), PARAMS, storage, nom, 6,
        script_etude="etude.py", repo=_REPO, lancer=_lanceur(),
        tracer=lambda _m: None)
    assert sorted(res) == [0, 1, 2, 3, 4]
    assert res[0] == {"g": 100.0, "dg_fc": 1.0, "dg_fy": 2.0}
    assert res[2] == {"g": 102.0, "dg_fc": 3.0, "dg_fy": 4.0}
    assert res[1]["g"] == 1.0, "un point a payer n'a pas ete calcule"


def test_un_plan_entierement_connu_ne_lance_aucun_worker(tmp_path):
    """Le cas d'une reprise juste apres la fin du plan : monter des
    sous-processus pour ne rien calculer coute des secondes et une copie du
    modele par worker."""
    storage, nom = _modele(tmp_path)
    lancer = _lanceur()
    res = _parallele.evaluer_en_parallele(
        _points_dont_deja(4, [0, 1, 2, 3]), PARAMS, storage, nom, 4,
        script_etude="etude.py", repo=_REPO, lancer=lancer,
        tracer=lambda _m: None)
    assert lancer.trace == [], "des workers ont ete lances pour rien"
    assert sorted(res) == [0, 1, 2, 3]


def test_les_workers_se_partagent_les_SEULS_points_a_payer(tmp_path):
    """La repartition porte sur ce qui reste, pas sur le plan entier --
    sinon un worker recevrait un lot vide et un autre deux points."""
    storage, nom = _modele(tmp_path)
    lancer = _lanceur()
    _parallele.evaluer_en_parallele(
        _points_dont_deja(6, [0, 1, 2, 3]), PARAMS, storage, nom, 2,
        script_etude="etude.py", repo=_REPO, lancer=lancer,
        tracer=lambda _m: None)
    lots = sorted(sorted(p["idx"] for p in t["points"]) for t in lancer.trace)
    assert lots == [[4], [5]], lots


def test_la_reprise_est_annoncee(tmp_path):
    storage, nom = _modele(tmp_path)
    messages = []
    _parallele.evaluer_en_parallele(
        _points_dont_deja(5, [0, 1, 2]), PARAMS, storage, nom, 6,
        script_etude="etude.py", repo=_REPO, lancer=_lanceur(),
        tracer=messages.append)
    assert any("3/5" in m and "deja connus" in m for m in messages), messages
    assert any("2 pts ->" in m for m in messages), (
        "le journal du dispatch doit compter les points A PAYER, pas le "
        "plan entier : %s" % messages)


def test_le_controle_de_completude_porte_sur_le_plan_ENTIER(tmp_path):
    """Un worker qui meurt doit toujours faire lever, meme si des points
    grefes comblent les trous."""
    storage, nom = _modele(tmp_path)
    with pytest.raises(RuntimeError, match="sans sortie"):
        _parallele.evaluer_en_parallele(
            _points_dont_deja(5, [0, 1, 2]), PARAMS, storage, nom, 6,
            script_etude="etude.py", repo=_REPO,
            lancer=_lanceur(muet=True), tracer=lambda _m: None)


# --------------------------------------------------------------------- #
# LA MOISSON : ce qu'un plan parallele INTERROMPU avait deja paye
#
# DEFAUT n°14, MESURE LE 29/08/2026. Le filet de reprise du plan
# (`save_doe_cache_incremental`, ecrit apres CHAQUE point) s'execute DANS le
# worker, donc dans SA copie du `.ds`. Le pere ne relit que le sien.
# Interruption apres 3 points sur 5 :
#
#     voie sequentielle   filet chez le pere : oui   3 points repris
#     voie parallele      filet chez le pere : non   0 point repris
#                         (le fichier etait dans _doe_workers/doew0.ds/)
#
# Sur le Moulin Blanc -- n0 = 5, six workers -- jusqu'a cinq appels solveur,
# environ 39 minutes.
#
# ON VERIFIE, ON NE SUPPOSE PAS. Une sortie de worker ne porte que
# `{idx: {g, dg_*}}` : la greffer sur le seul indice serait exactement le
# defaut du cache de points libres, ou des valeurs se retrouvaient appariees a
# de mauvaises coordonnees. Le fichier de TACHE est a cote et porte les
# parametres ; on compare les deux.
# --------------------------------------------------------------------- #
def _run_interrompu(tmp_path, faits, total=5):
    """Un run parallele dont `faits` indices sont alles au bout."""
    storage, nom = _modele(tmp_path)
    points = _points(total)
    base = os.path.join(storage, nom + ".ds")
    for k, i in enumerate(faits):
        wds = os.path.join(base, "_doe_workers", "doew%d.ds" % k)
        os.makedirs(wds, exist_ok=True)
        json.dump({"points": [dict({"idx": i}, **points[i])]},
                  open(os.path.join(wds, "_doe_task.json"), "w"))
        json.dump({str(i): {"g": 10.0 + i, "dg_fc": 1.0, "dg_fy": 2.0}},
                  open(os.path.join(wds, "_doe_out.json"), "w"))
    return storage, nom, points


def test_les_points_menes_au_bout_sont_recuperes(tmp_path):
    storage, nom, points = _run_interrompu(tmp_path, [0, 2])
    recolte = _parallele.moissonner_sorties(points, PARAMS, storage, nom,
                                            tracer=lambda _m: None)
    assert sorted(recolte) == [0, 2]
    assert recolte[0]["g"] == 10.0 and recolte[2]["g"] == 12.0


def test_sans_dossier_de_workers_la_moisson_est_vide(tmp_path):
    storage, nom = _modele(tmp_path)
    assert _parallele.moissonner_sorties(_points(3), PARAMS, storage, nom,
                                         tracer=lambda _m: None) == {}


def test_un_worker_sans_sortie_n_apporte_rien(tmp_path):
    """Il n'etait pas alle au bout : c'est le cas normal a une interruption."""
    storage, nom, points = _run_interrompu(tmp_path, [1])
    os.remove(os.path.join(storage, nom + ".ds", "_doe_workers",
                           "doew0.ds", "_doe_out.json"))
    assert _parallele.moissonner_sorties(points, PARAMS, storage, nom,
                                         tracer=lambda _m: None) == {}


def test_une_sortie_dont_les_COORDONNEES_diffrent_est_ECARTEE(tmp_path):
    """Le controle qui fait toute la difference.

    `preparer_worker` efface la sortie d'un run precedent precisement parce
    qu'elle serait relue comme celle-ci. La moisson, elle, va la chercher : il
    lui faut donc son propre controle, sinon elle rouvre le trou.
    """
    storage, nom, points = _run_interrompu(tmp_path, [0, 1])
    points[1]["fc"] = 999.0            # le plan courant n'est plus le meme
    recolte = _parallele.moissonner_sorties(points, PARAMS, storage, nom,
                                            tracer=lambda _m: None)
    assert sorted(recolte) == [0], (
        "un point d'un AUTRE tirage a ete greffe : %s" % sorted(recolte))


def test_la_moisson_dit_ce_qu_elle_recupere_et_ce_qu_elle_ecarte(tmp_path):
    storage, nom, points = _run_interrompu(tmp_path, [0, 1])
    points[1]["fy"] = 999.0
    messages = []
    _parallele.moissonner_sorties(points, PARAMS, storage, nom,
                                  tracer=messages.append)
    assert any("1 point(s) recuperes" in m and "1 ecarte" in m
               for m in messages), messages


def test_une_sortie_illisible_est_ignoree_en_le_DISANT(tmp_path):
    storage, nom, points = _run_interrompu(tmp_path, [0])
    chemin = os.path.join(storage, nom + ".ds", "_doe_workers",
                          "doew0.ds", "_doe_out.json")
    open(chemin, "w").write("{ ceci n'est pas du JSON")
    messages = []
    assert _parallele.moissonner_sorties(points, PARAMS, storage, nom,
                                         tracer=messages.append) == {}
    assert any("illisible" in m for m in messages), messages


def test_la_moisson_et_le_saut_du_pool_se_completent(tmp_path):
    """Bout a bout : ce qui est moissonne ne repart pas au solveur."""
    storage, nom, points = _run_interrompu(tmp_path, [0, 2])
    for i, rendu in _parallele.moissonner_sorties(
            points, PARAMS, storage, nom, tracer=lambda _m: None).items():
        points[i].update(rendu)
    lancer = _lanceur()
    _parallele.evaluer_en_parallele(points, PARAMS, storage, nom, 6,
                                    script_etude="etude.py", repo=_REPO,
                                    lancer=lancer, tracer=lambda _m: None)
    envoyes = sorted(p["idx"] for t in lancer.trace for p in t["points"])
    assert envoyes == [1, 3, 4], envoyes
