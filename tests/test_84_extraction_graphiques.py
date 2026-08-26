"""
Les graphiques extraits sont-ils les memes images ?

Trois fonctions de suivi de l'enrichissement sorties de
`if __name__ == '__main__':` vers `_reliability/graphiques.py`.

La verification est directe et sans compromis : l'original, recupere a une
revision git figee, et l'extrait produisent chacun leur PNG, et les fichiers
sont compares **octet par octet**. Matplotlib avec le backend Agg est
deterministe ; il n'y a donc aucune raison d'accepter le moindre ecart, et
aucune tolerance a choisir.

`print_Pf_evolution` et `print_logPf_evolution` etaient identiques a 54 % --
la difference tenait a l'echelle des ordonnees. `tracer_pf_evolution` les
remplace toutes deux, et les deux images doivent rester identiques a celles
des deux originaux.

Ces tests ne demandent que matplotlib et numpy : ils tournent partout ou
tourne le harness.
"""

import hashlib
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pytest

# matplotlib appartient a la couche des ETUDES, pas au noyau : sur un poste
# ou seul `requirements/core.txt` est installe -- la CI, par exemple -- ce
# fichier doit se sauter proprement, et non interrompre la COLLECTE de toute
# la suite. C'est ce qui arrivait avant la phase 8.
matplotlib = pytest.importorskip(
    "matplotlib", reason="couche etudes (requirements/studies.txt)")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
for p in (os.path.join(REPO, "_reliability"), os.path.join(REPO, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import graphiques as G  # noqa: E402
from extraction_temoin import AC_FLEXION, fonction_originale  # noqa: E402

#: revision d'AVANT le retrait des definitions des scripts AC
REVISION = "a77eafc"

HIST_PF = [{"mid": 1.2e-3, "sup": 2.0e-3, "inf": 7.0e-4},
           {"mid": 9.0e-4, "sup": 1.5e-3, "inf": 5.0e-4},
           {"mid": None, "sup": None, "inf": None},      # iteration sans mesure
           {"mid": 8.0e-4, "sup": 1.1e-3, "inf": 6.0e-4}]
HIST_EFF = [1.0, 0.3, 0.05, 0.001, 0.0]                  # 0.0 : passage par le clip
HIST_BB = [0.4, 0.2, None, 0.03]
HIST_BS = [0.09, 0.02, 0.005, 0.001]
HIST_THETA = [[10.0, 20.0], [30.0, 25.0], [66.7, 75.4]]
NOMS = ["fc", "fy"]

pytestmark = pytest.mark.slow


def _revision_disponible():
    try:
        fonction_originale(AC_FLEXION, "print_Pf_evolution",
                           {"np": np, "plt": plt, "os": os, "_eff_history_Pf": [],
                            "modele": "x", "EFF_criteria": "x", "out_dir_eff": ".",
                            "timestamp": "T"}, revision=REVISION)
        return True
    except Exception:
        return False


besoin_revision = pytest.mark.skipif(
    not _revision_disponible(),
    reason="revision %s introuvable (historique reecrit ?)" % REVISION)


def _md5(chemin):
    with open(chemin, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _original(nom, libres, dossier):
    env = {"np": np, "plt": plt, "os": os, "out_dir_eff": str(dossier), "timestamp": "T"}
    env.update(libres)
    f = fonction_originale(AC_FLEXION, nom, env, revision=REVISION)
    with redirect_stdout(io.StringIO()):
        f()


def _muet(fn, *a, **kw):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# --------------------------------------------------------------------------- #
@besoin_revision
@pytest.mark.parametrize("echelle,nom_origine,fichier",
                         [("lineaire", "print_Pf_evolution", "Pf_evolution_T.png"),
                          ("log", "print_logPf_evolution", "logPf_evolution_T.png")])
def test_pf_evolution_image_identique(tmp_path, echelle, nom_origine, fichier):
    """Une seule fonction remplace les deux originaux : les deux images
    doivent rester identiques, sinon l'unification a change quelque chose."""
    a, b = tmp_path / "orig", tmp_path / "neuf"
    a.mkdir(); b.mkdir()
    _original(nom_origine, {"_eff_history_Pf": HIST_PF, "modele": "GEPCK",
                            "EFF_criteria": "BS"}, a)
    rendu = _muet(G.tracer_pf_evolution, HIST_PF, "GEPCK", "BS", str(b), "T", echelle)
    assert rendu == fichier
    assert _md5(a / fichier) == _md5(b / fichier), "les images different"


@besoin_revision
def test_convergence_eff_image_identique(tmp_path):
    a, b = tmp_path / "orig", tmp_path / "neuf"
    a.mkdir(); b.mkdir()
    _original("print_EFF_graphs",
              {"_eff_history_EFF": HIST_EFF, "_eff_history_BB": HIST_BB,
               "_eff_history_BS": HIST_BS, "_eff_history_theta": HIST_THETA,
               "params_names": NOMS, "tol_EFF": 1e-3, "tol_BB": 0.05,
               "tol_BS": 0.01}, a)
    rendu = _muet(G.tracer_convergence_eff, HIST_EFF, HIST_BB, HIST_BS, HIST_THETA,
                  NOMS, 1e-3, 0.05, 0.01, str(b), "T")
    assert rendu == "EFF_graphs_T.png"
    assert _md5(a / "EFF_graphs_T.png") == _md5(b / "EFF_graphs_T.png")


# --------------------------------------------------------------------------- #
# Comportement propre, independamment des originaux                           #
# --------------------------------------------------------------------------- #
def test_rien_a_tracer_ne_produit_aucun_fichier(tmp_path):
    assert _muet(G.tracer_pf_evolution, [], "GEPCK", "BS", str(tmp_path), "T") is None
    assert _muet(G.tracer_convergence_eff, [], [], [], [], NOMS,
                 1e-3, 0.05, 0.01, str(tmp_path), "T") is None
    assert not list(tmp_path.iterdir())


def test_les_deux_echelles_donnent_des_images_differentes(tmp_path):
    """Garde-fou du parametre : si `echelle` etait ignore, les deux appels
    rendraient la meme image et l'unification aurait perdu une fonction."""
    lin = _muet(G.tracer_pf_evolution, HIST_PF, "GEPCK", "BS", str(tmp_path), "T", "lineaire")
    log = _muet(G.tracer_pf_evolution, HIST_PF, "GEPCK", "BS", str(tmp_path), "T", "log")
    assert lin != log
    assert _md5(tmp_path / lin) != _md5(tmp_path / log)


def test_echelle_inconnue_refusee(tmp_path):
    with pytest.raises(ValueError, match="echelle"):
        G.tracer_pf_evolution(HIST_PF, "GEPCK", "BS", str(tmp_path), "T", "semilog")


def test_les_iterations_sans_mesure_ne_font_pas_echouer(tmp_path):
    """L'historique contient des None quand une iteration n'a pas mesure Pf."""
    assert _muet(G.tracer_pf_evolution, HIST_PF, "GEPCK", "BS", str(tmp_path), "T") is not None


def test_plus_aucune_variable_libre():
    from extraction_temoin import variables_libres
    autorises = {"np", "plt", "os", "CLIP"}
    chemin = os.path.join(REPO, "_reliability", "graphiques.py")
    for nom in ("tracer_convergence_eff", "tracer_pf_evolution"):
        restantes = set(variables_libres(chemin, nom)) - autorises
        assert not restantes, f"{nom} depend encore de {sorted(restantes)}"
