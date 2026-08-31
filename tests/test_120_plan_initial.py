r"""L'enchainement du plan initial, exerce sans solveur.

CE QUE CE FICHIER COUVRE, ET QUE LES RUNS NE COUVRENT PAS
-----------------------------------------------------------
`build_DOE` etait recopie dans les deux etudes -- cinquante-deux lignes de
chaque cote. Il est dans `_doe/plan.py` depuis le 29/08/2026, sous le nom
`construire_plan_initial`.

Quatre chemins ont ete rejoues bout en bout sur l'etude analytique, avant et
apres, avec des journaux identiques : le cas courant, `do_HF = true`, et
`config_is_identical = true` a froid puis a chaud. Restent deux choses qu'un
run ne montre pas :

* le chemin PARALLELE. `n_workers_DOE` vaut 1 dans toutes les etudes
  analytiques et 6 sur le Moulin Blanc : la branche qui tourne EN PRODUCTION
  est precisement celle qu'aucun run de controle ne traverse ;
* l'ARITE. La fonction rend toujours un triplet, y compris en haute fidelite
  pure ou `yt` et `all_grad` valent None. `build_DOE` rendait tantot un
  triplet, tantot le seul `xt`, et son propre commentaire gardait la trace de
  ce que cela avait coute : « l'ancienne branche faisait `xt = build_DOE()`
  et recevait un TRIPLET -- xt devenait un tuple, silencieusement ». Une
  arite qui depend d'un drapeau ne se voit pas sur un run qui passe.
"""

import os
import sys

import numpy as np
import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_doe"), os.path.join(_REPO, "_cache")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ot = pytest.importorskip("openturns", reason="la couche etudes n'est pas installee")

import plan as _plan   # noqa: E402

PARAMS = ["fc", "fy"]


class _Cfg:
    """Les seuls reglages que l'enchainement lit."""

    def __init__(self, **kw):
        self.do_HF = False
        self.do_PCK = False
        self.print_DOE = False
        self.config_is_identical = False
        self.n_workers_DOE = 1
        self.eps_taylor = 0.01
        self.exclure_points_sans_gradient = True
        self.__dict__.update(kw)


def _dist():
    return ot.JointDistribution([ot.Normal(48.0, 5.0), ot.Normal(550.0, 30.0)])


def _sol_complet(SOL):
    """Le solveur de papier : chaque point rend une valeur et un gradient."""
    for i, point in enumerate(SOL):
        point['g'] = 0.1 * (i + 1)
        for p in PARAMS:
            point['dg_%s' % p] = 0.01 * (i + 1)
    return SOL


def _construire(tmp_path, cfg=None, executer_plan=_sol_complet,
                executer_en_parallele=None, n_doe=5, journaliser=None,
                tracer=None, graine=None):
    """`tmp_path` est OBLIGATOIRE : le cache s'ecrit sur disque.

    `graine` fixe le tirage. Deux appels dans le meme process n'ont AUCUNE
    raison de tirer le meme hypercube -- le generateur avance. C'est ce qui
    rend le cache partiel intestable sans elle, et c'est aussi la raison
    d'etre de `xt_attendu` : la reprise ne SUPPOSE pas que le tirage
    recommence a l'identique, elle le VERIFIE.
    """
    cfg = cfg or _Cfg()
    if graine is not None:
        ot.RandomGenerator.SetSeed(graine)
    fichier = os.path.join(str(tmp_path), "doe_cache.json")
    return _plan.construire_plan_initial(
        cfg, n_doe, dist_jointe=_dist, params_names=PARAMS,
        bornes_min=[-4.0, -4.0], bornes_max=[4.0, 4.0],
        fichier_cache=fichier, signature={"solveur": "papier"},
        executer_plan=executer_plan,
        executer_en_parallele=executer_en_parallele,
        # une moisson NEUTRE : ce fichier eprouve l'enchainement, pas la
        # recuperation d'un run parallele interrompu (cf. test_101).
        moissonner=lambda SOL, noms: {},
        journaliser=journaliser, tracer=(tracer or (lambda m: None)))


# --------------------------------------------------------------------------- #
# 1. L'ARITE : toujours trois sorties                                          #
# --------------------------------------------------------------------------- #
def test_le_cas_courant_rend_un_triplet(tmp_path):
    xt, yt, all_grad = _construire(tmp_path)
    assert xt.shape == (5, 2)
    assert yt.shape == (5, 1)
    assert all_grad.shape == (5, 2)


def test_la_haute_fidelite_pure_rend_le_meme_triplet(tmp_path):
    """Pas de valeurs ni de gradients a rendre : deux `None`, pas une autre
    signature. C'est ce qui faisait de `xt` un tuple, silencieusement."""
    xt, yt, all_grad = _construire(tmp_path, cfg=_Cfg(do_HF=True))
    assert isinstance(xt, np.ndarray) and xt.shape == (5, 2)
    assert yt is None and all_grad is None


def test_la_haute_fidelite_pure_ne_paie_aucun_appel(tmp_path):
    """Le plan sert de POINTS : les appels viendront un par un, plus tard."""
    appels = []

    def _compter(SOL):
        appels.append(len(SOL))
        return _sol_complet(SOL)

    _construire(tmp_path, cfg=_Cfg(do_HF=True), executer_plan=_compter)
    assert appels == []


def test_la_haute_fidelite_pure_ne_lit_pas_le_cache(tmp_path):
    """`xt` seul ne se relit pas d'un fichier qui porte aussi `yt`."""
    cfg_cache = _Cfg(config_is_identical=True)
    _construire(tmp_path, cfg=cfg_cache)          # ecrit le cache
    xt, yt, all_grad = _construire(
        tmp_path, cfg=_Cfg(do_HF=True, config_is_identical=True))
    assert yt is None and all_grad is None


# --------------------------------------------------------------------------- #
# 2. LE CHEMIN PARALLELE -- celui qui tourne en production                     #
# --------------------------------------------------------------------------- #
def test_a_plusieurs_workers_le_plan_part_au_pool(tmp_path):
    vus = {}

    def _pool(SOL, n_workers):
        vus['n'] = n_workers
        vus['taille'] = len(SOL)
        return _sol_complet(SOL)

    def _jamais(SOL):
        raise AssertionError("le chemin sequentiel a ete pris malgre 6 workers")

    xt, yt, _ = _construire(tmp_path, cfg=_Cfg(n_workers_DOE=6),
                            executer_plan=_jamais, executer_en_parallele=_pool)
    assert vus == {'n': 6, 'taille': 5}
    assert yt.shape == (5, 1)


def test_a_un_seul_worker_le_plan_reste_sequentiel(tmp_path):
    def _jamais(SOL, n_workers):
        raise AssertionError("le pool a ete lance pour un seul worker")

    _construire(tmp_path, cfg=_Cfg(n_workers_DOE=1),
                executer_en_parallele=_jamais)


# --------------------------------------------------------------------------- #
# 3. LES DEUX CACHES, QUI NE FONT PAS LA MEME CHOSE                            #
# --------------------------------------------------------------------------- #
def test_le_cache_complet_evite_tout_appel_solveur(tmp_path):
    cfg = _Cfg(config_is_identical=True)
    premier = _construire(tmp_path, cfg=cfg)

    def _jamais(SOL):
        raise AssertionError("le solveur a ete appele malgre un cache complet")

    second = _construire(tmp_path, cfg=cfg, executer_plan=_jamais)
    for a, b in zip(premier, second):
        assert np.allclose(np.asarray(a, float), np.asarray(b, float))


def test_sans_config_identique_le_cache_n_est_pas_relu(tmp_path):
    """`config_is_identical = false` veut dire « recalcule », pas « recalcule
    si le fichier manque »."""
    _construire(tmp_path, cfg=_Cfg(config_is_identical=True))
    appels = []

    def _compter(SOL):
        appels.append(len(SOL))
        return _sol_complet(SOL)

    _construire(tmp_path, cfg=_Cfg(config_is_identical=False),
                executer_plan=_compter)
    assert appels == [5]


def _interrompre_apres(tmp_path, k, graine):
    """Ecrit le filet d'un plan interrompu apres `k` points.

    C'est exactement ce que fait l'evaluateur point par point : le fichier
    porte `complet: False`, et il etait ECRIT puis IGNORE avant le
    26/08/2026 -- trois interruptions dans la meme journee, ~75 minutes de
    solveur perdues chaque fois.
    """
    import doe as _cache_doe
    ot.RandomGenerator.SetSeed(graine)
    dist_X = _dist()
    _U, X_doe, xt = _plan.tirer_plan_lhs(dist_X, 5, [-4.0, -4.0], [4.0, 4.0])
    SOL = [{PARAMS[j]: X_doe[i][j] for j in range(len(PARAMS))} for i in range(5)]
    _sol_complet(SOL)
    # `_u` : les coordonnees en espace standard. C'est `run_one_SOL` qui les
    # pose en production ; sans elles la sauvegarde incrementale echoue -- en
    # le DISANT, mais elle echoue.
    for i, point in enumerate(SOL):
        point['_u'] = [float(v) for v in xt[i]]
    _cache_doe.save_doe_cache_incremental(
        os.path.join(str(tmp_path), "doe_cache.json"), 5, PARAMS, SOL, k,
        signature={"solveur": "papier"})
    return xt


def test_un_plan_interrompu_ne_repaie_que_ce_qui_manque(tmp_path):
    """La greffe : trois points deja payes, deux a payer."""
    _interrompre_apres(tmp_path, 3, graine=12345)
    a_payer = []

    def _compter(SOL):
        a_payer.append(sum(1 for p in SOL if 'g' not in p))
        return _sol_complet(SOL)

    _construire(tmp_path, cfg=_Cfg(config_is_identical=True),
                executer_plan=_compter, graine=12345)
    assert a_payer == [2], (
        "%s point(s) restaient a payer au lieu de 2 : la greffe du plan "
        "interrompu n'a pas eu lieu." % a_payer)


def test_un_plan_interrompu_AILLEURS_est_refuse_en_entier(tmp_path):
    """On VERIFIE, on ne SUPPOSE PAS.

    Le filet porte les coordonnees des points deja payes. Si le tirage courant
    ne redonne pas les memes -- graine differente, version d'OpenTURNS
    differente -- les greffer apparierait des valeurs a de mauvais points, et
    rien ne le montrerait. Ce controle est celui qui manquait au cache des
    points libres, et qui lui a coute une surface de reference entiere.
    """
    _interrompre_apres(tmp_path, 3, graine=12345)
    a_payer = []

    def _compter(SOL):
        a_payer.append(sum(1 for p in SOL if 'g' not in p))
        return _sol_complet(SOL)

    _construire(tmp_path, cfg=_Cfg(config_is_identical=True),
                executer_plan=_compter, graine=999)
    assert a_payer == [5], (
        "des points d'un AUTRE tirage ont ete greffes (%s a payer au lieu de "
        "5) : la verification des coordonnees ne mord pas." % a_payer)


def test_sans_config_identique_le_plan_interrompu_est_ignore(tmp_path):
    """`config_is_identical = false` veut dire « recalcule tout »."""
    _interrompre_apres(tmp_path, 3, graine=12345)
    a_payer = []

    def _compter(SOL):
        a_payer.append(sum(1 for p in SOL if 'g' not in p))
        return _sol_complet(SOL)

    _construire(tmp_path, cfg=_Cfg(config_is_identical=False),
                executer_plan=_compter, graine=12345)
    assert a_payer == [5]


# --------------------------------------------------------------------------- #
# 4. LE JOURNAL ET LES POINTS ECARTES                                          #
# --------------------------------------------------------------------------- #
def test_chaque_point_complet_passe_au_journal(tmp_path):
    lignes = []
    _construire(tmp_path,
                journaliser=lambda u, x, g, phase: lignes.append((phase, g)))
    assert len(lignes) == 5
    assert {p for p, _ in lignes} == {"DOE"}


def test_sans_journal_l_enchainement_ne_leve_pas(tmp_path):
    """`journaliser` est facultatif, et eprouve par `is not None`."""
    xt, _, _ = _construire(tmp_path, journaliser=None)
    assert len(xt) == 5


def test_un_point_sans_gradient_est_ecarte_et_ne_va_pas_au_journal(tmp_path):
    """Le parametre gouverne les deux : le plan ET sa trace."""
    def _un_trou(SOL):
        SOL = _sol_complet(SOL)
        SOL[2]['dg_fy'] = None
        return SOL

    lignes = []
    xt, yt, all_grad = _construire(
        tmp_path, executer_plan=_un_trou,
        journaliser=lambda u, x, g, phase: lignes.append(g))
    assert len(xt) == 4 and len(yt) == 4 and len(all_grad) == 4
    assert len(lignes) == 4


# --------------------------------------------------------------------------- #
# 5. LES POINTS VIRTUELS DE TAYLOR                                             #
# --------------------------------------------------------------------------- #
def test_le_PCK_recoit_des_points_virtuels(tmp_path):
    """Un PCK n'exploite pas les gradients : on les lui donne en points."""
    sans = _construire(tmp_path, cfg=_Cfg(do_PCK=False))
    avec = _construire(tmp_path, cfg=_Cfg(do_PCK=True))
    assert len(sans[0]) == 5
    assert len(avec[0]) == 5 * (1 + len(PARAMS)), len(avec[0])
