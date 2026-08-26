"""
Les caches extraits se comportent-ils comme ceux des scripts AC ?

Dix fonctions sorties de `if __name__ == '__main__':` vers `_cache/doe.py` et
`_cache/hf.py`. L'oracle est `tests/golden/caches_originaux.json`, produit par
`tools/golden_caches.py` a partir des definitions telles qu'elles etaient
avant l'extraction, lues a une revision git.

L'essentiel de ces fonctions n'est pas d'ecrire du JSON : c'est de REFUSER un
cache qui ne correspond pas -- n0 different, coupe differente, cache
incomplet, dimensions differentes, fichier absent, configuration declaree
differente. Un cache accepte a tort reinjecte des resultats perimes dans une
etude sans que rien ne le signale. Les six refus sont donc testes autant que
les six ecritures.

Aucune dependance lourde : ces tests tournent partout ou tourne le harness.
"""

import json
import os
import sys

import numpy as np
import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
for p in (os.path.join(REPO, "_cache"), os.path.join(REPO, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import doe as cache_doe        # noqa: E402
import hf as cache_hf          # noqa: E402

GOLDEN = os.path.join(TESTS, "golden", "caches_originaux.json")

pytestmark = pytest.mark.skipif(not os.path.isfile(GOLDEN),
                                reason="oracle des caches d'origine absent")

XT = [[0.1, -0.2], [1.5, 0.3], [-2.0, 1.1]]
YT = [[0.5], [-0.1], [0.9]]
AG = [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]]
SOL = [{"_u": [0.1, -0.2], "g": 0.5, "dg_fc": 0.01, "dg_fy": 0.02},
       {"_u": [1.5, 0.3], "g": -0.1, "dg_fc": 0.03, "dg_fy": 0.04},
       {"_u": [-2.0, 1.1], "g": 0.9}]
Z = np.array([[1.0, 2.0], [3.0, 4.0]])
SD = (0, 1, {2: 0.0})
NOMS = ["fc", "fy"]


@pytest.fixture(scope="module")
def attendu():
    with open(GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)["cas"]


def _lire(chemin):
    with open(chemin, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# DOE                                                                          #
# --------------------------------------------------------------------------- #
def test_signature_de_configuration(attendu):
    assert cache_doe.doe_cache_sig(3, NOMS, 2, "test_pure_flexion") \
        == attendu["doe_cache_sig"]


def test_ecriture_doe_complet(attendu, tmp_path):
    f = str(tmp_path / "doe.json")
    cache_doe.save_doe_cache(f, 3, XT, YT, AG)
    assert _lire(f) == attendu["save_doe_cache"]


def test_relecture_doe_complet(attendu, tmp_path):
    f = str(tmp_path / "doe.json")
    cache_doe.save_doe_cache(f, 3, XT, YT, AG)
    obtenu = [np.asarray(x).tolist() for x in cache_doe.load_doe_cache(f, 3)]
    assert obtenu == attendu["load_doe_cache"]


def test_ecriture_doe_incrementale(attendu, tmp_path):
    f = str(tmp_path / "doe.json")
    cache_doe.save_doe_cache_incremental(f, 3, NOMS, SOL, 2)
    assert _lire(f) == attendu["save_doe_cache_incremental"]


@pytest.mark.parametrize("cas", ["n0_different", "config_differente",
                                 "incomplet", "absent"])
def test_les_refus_du_cache_doe(attendu, tmp_path, cas):
    """Un cache accepte a tort reinjecte des resultats perimes dans une etude."""
    f = str(tmp_path / "doe.json")
    if cas == "n0_different":
        cache_doe.save_doe_cache(f, 3, XT, YT, AG)
        obtenu = cache_doe.load_doe_cache(f, 5)
    elif cas == "config_differente":
        cache_doe.save_doe_cache(f, 3, XT, YT, AG)
        obtenu = cache_doe.load_doe_cache(f, 3, config_is_identical=False)
    elif cas == "incomplet":
        cache_doe.save_doe_cache_incremental(f, 3, NOMS, SOL, 2)
        obtenu = cache_doe.load_doe_cache(f, 3)
    else:
        obtenu = cache_doe.load_doe_cache(f, 3)
    assert obtenu is attendu["load_doe_cache_" + cas] is None


# --------------------------------------------------------------------------- #
# Grille haute fidelite                                                        #
# --------------------------------------------------------------------------- #
def test_ecriture_grille_hf(attendu, tmp_path):
    f = str(tmp_path / "hf.json")
    cache_hf.save_hf_cache(Z, 2, f, SD)
    assert _lire(f) == attendu["save_hf_cache"]


def test_relecture_grille_hf(attendu, tmp_path):
    f = str(tmp_path / "hf.json")
    cache_hf.save_hf_cache(Z, 2, f, SD)
    assert cache_hf.load_hf_cache(2, f, SD).tolist() == attendu["load_hf_cache"]


def test_refus_grille_hf_coupe_differente(attendu, tmp_path):
    f = str(tmp_path / "hf.json")
    cache_hf.save_hf_cache(Z, 2, f, SD)
    assert cache_hf.load_hf_cache(2, f, (0, 2, {})) is None
    assert attendu["load_hf_cache_coupe_differente"] is None


def test_grille_hf_partielle(attendu, tmp_path):
    """Reprise d'un calcul interrompu : les points non calcules valent None."""
    f = str(tmp_path / "hf.json")
    cache_hf.save_hf_cache_partial([1.0, None, 3.0, None], 4, f, SD)
    assert _lire(f + ".partial") == attendu["save_hf_cache_partial"]
    assert cache_hf.load_hf_cache_partial(f, SD, 4) == attendu["load_hf_cache_partial"]
    assert cache_hf.load_hf_cache_partial(f, SD, 9) is None


def test_grille_hf_complete(attendu, tmp_path):
    f = str(tmp_path / "hf_full.json")
    cache_hf.save_hf_grid_full(f, Z, 2, 2)
    assert _lire(f) == attendu["save_hf_grid_full"]
    assert cache_hf.load_hf_grid_full(f, 2, 2).tolist() == attendu["load_hf_grid_full"]
    assert cache_hf.load_hf_grid_full(f, 2, 7) is None
    assert cache_hf.load_hf_grid_full(f, 2, 2, config_is_identical=False) is None


# --------------------------------------------------------------------------- #
# Ce que l'extraction devait accomplir                                        #
# --------------------------------------------------------------------------- #
def test_plus_aucune_variable_libre():
    from extraction_temoin import variables_libres
    autorises = {"np", "json", "os"}
    for module, noms in (("doe", ["doe_cache_sig", "save_doe_cache", "load_doe_cache",
                                  "save_doe_cache_incremental"]),
                         ("hf", ["save_hf_cache", "load_hf_cache",
                                 "save_hf_cache_partial", "load_hf_cache_partial",
                                 "save_hf_grid_full", "load_hf_grid_full"])):
        chemin = os.path.join(REPO, "_cache", module + ".py")
        for nom in noms:
            restantes = set(variables_libres(chemin, nom)) - autorises
            assert not restantes, f"{module}.{nom} depend encore de {sorted(restantes)}"


EXTRAITES = ["_doe_cache_sig", "_save_doe_cache", "_load_doe_cache",
             "_save_doe_cache_incremental", "_save_hf_cache", "_load_hf_cache",
             "_save_hf_cache_partial", "_load_hf_cache_partial",
             "_save_hf_grid_full", "_load_hf_grid_full"]


@pytest.mark.parametrize("rel", ["pure_flexion/AC3_pure_flexion.py",
                                 "Moulinblanc/AC3_moulinblanc.py"])
def test_les_scripts_ac_ne_portent_plus_que_des_delegues(rel):
    """Les noms subsistent dans les scripts AC -- ce sont les delegues qui
    gardent les sites d'appel intacts. Ce qui ne doit plus s'y trouver, c'est
    la LOGIQUE : un delegue tient en trois lignes et transmet a `_cache`.

    Verifier l'absence du nom serait plus simple, et faux : ce test doit
    distinguer un delegue d'une copie oubliee, pas compter des `def`."""
    import ast
    chemin = os.path.join(REPO, rel)
    with open(chemin, encoding="utf-8", errors="replace") as fh:
        source = fh.read()
    arbre = ast.parse(source, chemin)
    vus = set()
    for n in ast.walk(arbre):
        if not isinstance(n, ast.FunctionDef) or n.name not in EXTRAITES:
            continue
        vus.add(n.name)
        corps = [x for x in n.body if not isinstance(x, ast.Expr)
                 or not isinstance(x.value, ast.Constant)]
        assert len(corps) == 1 and isinstance(corps[0], ast.Return),             f"{rel} : {n.name} n est pas un delegue (corps de {len(corps)} instructions)"
        appelle = [d.value.id for d in ast.walk(corps[0])
                   if isinstance(d, ast.Attribute) and isinstance(d.value, ast.Name)]
        assert any(a in ("_cache_doe", "_cache_hf") for a in appelle),             f"{rel} : {n.name} ne transmet pas a _cache"
    assert vus == set(EXTRAITES), f"{rel} : manquants {sorted(set(EXTRAITES) - vus)}"


def test_ce_qui_reste_dans_main_est_documente():
    """Ce qui n'a PAS ete extrait du bloc cache. Ce test empeche de l'oublier
    -- et tombe le jour ou c'est fait, pour qu'on mette a jour le plan.

    `_save_socp_outputs` est parti en phase 5, comme annonce : il manipulait
    les sorties du solveur, il vit maintenant dans
    `solver/digital_structure.py:SolveurDS.archiver_sorties`.
    """
    import re
    src = open(os.path.join(REPO, "pure_flexion", "AC3_pure_flexion.py"),
               encoding="utf-8", errors="replace").read()
    # _save_restart_state : 14 variables libres, c'est un instantane de main,
    #                       pas un cache -- il suivra la refonte de la config.
    for nom in ("_save_restart_state",):
        assert re.search(r"(?m)^\s+def %s\(" % nom, src), \
            f"{nom} a ete extrait : mettre a jour ce test et le plan"
    # ... et celle qui est bien partie ne doit pas revenir.
    assert not re.search(r"(?m)^\s+def _save_socp_outputs\(", src), \
        "_save_socp_outputs est reapparu dans le script AC"
    solveur = open(os.path.join(REPO, "solver", "digital_structure.py"),
                   encoding="utf-8", errors="replace").read()
    assert "def archiver_sorties" in solveur
