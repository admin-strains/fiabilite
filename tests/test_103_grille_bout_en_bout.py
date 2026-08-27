r"""La grille haute fidelite, exercee bout en bout -- le chemin qui n'etait pas couvert.

LE TROU, ET CE QU'IL A COUTE
------------------------------
`pure_flexion_analytique.toml` tourne avec `print_HF = false`. Elle exerce le
plan d'experiences, le metamodele, l'enrichissement, FORM et le tirage
d'importance -- mais **pas la grille haute fidelite**.

Le 27/08/2026, on a decouvert que cette grille etait calculee DEUX FOIS :
`slice_def` et `slice_def_final` valent tous deux `(0, 1, {})` des qu'il y a
deux variables, mais ils etaient servis par deux fichiers de cache
differents. Sur le Moulin Blanc regle a 15, cela faisait 225 appels solveur
en double -- 29,1 heures a 466 s l'appel.

Le defaut a du etre MESURE A LA MAIN, en retirant la garde et en relancant,
parce qu'aucune etude rejouable n'exercait ce chemin. La suite de tests, elle,
etait verte.

CE QUE FAIT CE FICHIER
-----------------------
Il lance `studies/pure_flexion_grille.toml` -- la meme chaine, avec la grille
activee a 7x7 -- et compte ce qui se passe reellement. Six secondes, 49
evaluations d'une formule de section, aucune licence.

Ce n'est pas un test de valeur mais un test de COUT : il verifie qu'on paie
ce qu'on croit payer, une fois et pas deux.
"""

import os
import re
import subprocess
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)

ETUDE = os.path.join(_REPO, "studies", "pure_flexion_grille.toml")
COTE_ATTENDU = 7

pytest.importorskip("openturns", reason="la couche etudes n'est pas installee")
pytest.importorskip("smt", reason="la couche etudes n'est pas installee")


def _dossier_du_modele():
    """Le `.ds` de l'etude, ou None si le modele n'est pas sur ce poste."""
    sys.path.insert(0, os.path.join(_REPO, "_config"))
    import schema
    cfg = schema.charger(ETUDE)
    dossier = os.path.join(cfg.storage, cfg.modelname + ".ds")
    return dossier if os.path.isdir(dossier) else None


def _fichiers_de_cache():
    dossier = _dossier_du_modele()
    if dossier is None:
        return ()
    return (os.path.join(dossier, "hf_grid_cache.json"),
            os.path.join(dossier, "hf_grid_cache.json.partial"),
            os.path.join(dossier, "hf_grid_cache_final.json"))


@pytest.fixture(scope="module")
def journal():
    """Le journal d'un run complet avec la grille activee.

    L'etude porte `config_is_identical = false` : aucun cache n'est relu,
    donc le compte d'appels est le meme a chaque execution. Les fichiers de
    cache sont tout de meme effaces avant, pour que le test qui verifie leur
    ABSENCE observe ce run et non l'historique du poste.
    """
    if not os.path.isfile(ETUDE):
        pytest.skip("etude %s absente" % ETUDE)
    # Un `hf_grid_cache_final.json` laisse par un run ANTERIEUR ferait echouer
    # `test_aucun_second_fichier_de_cache_n_est_ecrit` a tort. Le test doit
    # observer CE run, pas l'historique du poste.
    for chemin in _fichiers_de_cache():
        if os.path.exists(chemin):
            os.remove(chemin)
    env = dict(os.environ, FIABILITE_ETUDE=ETUDE, PYTHONPATH="",
               MPLBACKEND="Agg", _FIAB_LOG_REDIRECTED="1")
    proc = subprocess.run(
        [sys.executable, os.path.join(_REPO, "launcher.py"),
         os.path.join("pure_flexion", "AC3_pure_flexion.py")],
        cwd=_REPO, env=env, capture_output=True, text=True,
        errors="replace", timeout=600)
    if proc.returncode != 0:
        pytest.fail("l'etude a echoue (code %d) :\n%s"
                    % (proc.returncode, proc.stdout[-3000:] + proc.stderr[-2000:]))
    return proc.stdout


# --------------------------------------------------------------------- #
# LE test : on paie UNE grille, pas deux
# --------------------------------------------------------------------- #
def test_la_grille_haute_fidelite_n_est_calculee_QU_UNE_FOIS(journal):
    """LE defaut du 27/08/2026, rendu impossible.

    Deux grilles au lieu d'une, c'est `n_grid_hf ** 2` appels solveur payes
    pour rien : 225 sur le Moulin Blanc, soit 29 heures.
    """
    departs = journal.count("HF GRID START")
    assert departs == 1, (
        "%d grilles haute fidelite calculees au lieu d'une.\n"
        "C'est le defaut du 27/08/2026 : `slice_def` et `slice_def_final` "
        "designent la MEME coupe mais etaient servis par deux fichiers de "
        "cache differents. Sur le Moulin Blanc, cela coute 29 heures."
        % departs)


def test_la_grille_coute_exactement_son_carre(journal):
    """Le cout annonce doit etre le cout paye."""
    m = re.search(r"HF GRID START: (\d+)x(\d+) = (\d+) points solveur", journal)
    assert m, "le journal n'annonce pas le cout de la grille"
    cote_x, cote_y, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
    assert (cote_x, cote_y) == (COTE_ATTENDU, COTE_ATTENDU)
    assert total == COTE_ATTENDU ** 2

    fait = re.search(r"HF GRID DONE in [\d.]+ min \((\d+) appels solveur", journal)
    assert fait, "le journal ne dit pas ce qui a ete paye"
    assert int(fait.group(1)) == COTE_ATTENDU ** 2, (
        "annonce %d appels, %s payes" % (COTE_ATTENDU ** 2, fait.group(1)))


def test_aucun_second_fichier_de_cache_n_est_ecrit(journal):
    """`hf_grid_cache_final.json` etait la trace de la seconde grille.

    Son absence est une preuve independante du comptage : meme si le journal
    changeait de forme, un second fichier signalerait le retour du doublon.

    Depend de `journal` a dessein : le fichier est efface AVANT le run, si
    bien que ce test observe CE run et non l'historique du poste.
    """
    dossier = _dossier_du_modele()
    if dossier is None:
        pytest.skip("modele absent de ce poste")
    final = os.path.join(dossier, "hf_grid_cache_final.json")
    assert not os.path.exists(final), (
        "%s existe : une SECONDE grille a ete calculee pour la meme coupe."
        % final)


# --------------------------------------------------------------------- #
# la chaine va bien jusqu'au bout
# --------------------------------------------------------------------- #
def test_l_etude_va_jusqu_au_resultat(journal):
    """Un run qui s'arrete avant FORM ne prouverait rien sur la grille."""
    assert "beta         =" in journal, "l'etude n'a pas produit de beta"
    assert re.search(r"Pf           = [\d.]+e[-+]\d+", journal), \
        "l'etude n'a pas produit de probabilite de defaillance"


def test_la_grille_sert_effectivement_de_fond(journal):
    """Payer la grille et ne pas s'en servir serait le meme gachis, sous une
    autre forme."""
    assert "[HF CACHE]" in journal or "HF GRID DONE" in journal
    assert "Traceback" not in journal


# --------------------------------------------------------------------- #
# l'etude elle-meme reste dans son role
# --------------------------------------------------------------------- #
def test_l_etude_reste_assez_petite_pour_la_suite_de_tests():
    """Le cout croit comme le carre du cote : a 15 elle ne serait plus
    utilisable ici, et le chemin redeviendrait non couvert."""
    import sys as _sys
    _sys.path.insert(0, os.path.join(_REPO, "_config"))
    import schema
    cfg = schema.charger(ETUDE)
    assert cfg.print_HF is True, "l'etude doit exercer la grille"
    assert cfg.n_grid_hf <= 9, (
        "n_grid_hf = %d : %d evaluations. Au-dela, cette etude sort de la "
        "suite de tests et le chemin cesse d'etre couvert."
        % (cfg.n_grid_hf, cfg.n_grid_hf ** 2))
    assert cfg.solveur == "analytique", "aucune licence ne doit etre requise"
