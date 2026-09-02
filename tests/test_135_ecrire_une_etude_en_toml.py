r"""Ecrire une etude SANS TOUCHER AU PYTHON : la promesse, verifiee.

CE QUE CE FICHIER PROUVE
-------------------------
Depuis le 02/09/2026, les variables aleatoires d'une etude se declarent en
donnees dans son `.toml`. La promesse qui va avec -- « un collegue ecrit une
etude en ecrivant un fichier » -- ne vaut que si elle est EXERCEE : une
declaration que rien ne relit serait un reglage sans effet, exactement la
classe de defaut que `dossier_sortie` a portee pendant des mois.

Ce fichier ecrit donc un `.toml` NEUF dans un dossier temporaire, le donne au
lanceur, et regarde le resultat. Deux exigences, et la seconde est la vraie :

1. l'etude tourne jusqu'a son `beta` avec un fichier que la suite vient
   d'ecrire -- aucune ligne de Python touchee ;
2. changer une VALEUR de la declaration change le RESULTAT. Sans quoi la
   declaration serait decorative : elle serait lue, validee, imprimee au
   journal, et le calcul se ferait sur autre chose.

Le run coute une dizaine de secondes : c'est l'etude `pure_flexion_grille`,
la meme que `test_103`, avec sa grille 7x7 sur solveur analytique.
"""

import io
import os
import re
import subprocess
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)

pytest.importorskip("openturns", reason="la couche etudes n'est pas installee")
pytest.importorskip("smt", reason="la couche etudes n'est pas installee")

#: L'etude de reference, dont on derive les variantes.
MODELE_DE_TOML = os.path.join(_REPO, "studies", "pure_flexion_grille.toml")

#: Ce que `test_103` fige pour cette etude : le resultat a declaration
#: INCHANGEE. Reproduit ici pour que ce fichier dise ce qu'il compare.
BETA_DE_REFERENCE = 4.6793


def _ecrire_variante(dossier, remplacements):
    """Un `.toml` neuf, derive du fichier d'etude, avec des substitutions."""
    texte = io.open(MODELE_DE_TOML, encoding="utf-8").read()
    for avant, apres in remplacements:
        assert avant in texte, "ancre %r absente du fichier d'etude" % avant
        texte = texte.replace(avant, apres)
    chemin = os.path.join(str(dossier), "etude_ecrite_par_le_test.toml")
    io.open(chemin, "w", encoding="utf-8", newline="").write(texte)
    return chemin


def _lancer(chemin_toml):
    """Le run complet, comme `test_103`, et son journal."""
    env = dict(os.environ, FIABILITE_ETUDE=chemin_toml, PYTHONPATH="",
               MPLBACKEND="Agg", _FIAB_LOG_REDIRECTED="1")
    proc = subprocess.run(
        [sys.executable, os.path.join(_REPO, "launcher.py"),
         os.path.join("pure_flexion", "AC3_pure_flexion.py")],
        cwd=_REPO, env=env, capture_output=True, text=True,
        errors="replace", timeout=600)
    if proc.returncode != 0:
        pytest.fail("l'etude a echoue (code %d) :\n%s"
                    % (proc.returncode,
                       proc.stdout[-3000:] + proc.stderr[-2000:]))
    return proc.stdout


def _beta(journal):
    m = re.search(r"(?m)^beta         = ([\d.]+)", journal)
    assert m, "le journal n'imprime pas `beta` :\n%s" % journal[-2000:]
    return float(m.group(1))


# --------------------------------------------------------------------------- #
# 1. UN FICHIER NEUF SUFFIT                                                    #
# --------------------------------------------------------------------------- #
def test_un_toml_ecrit_a_l_instant_donne_le_meme_resultat(tmp_path):
    """Le fichier est recopie ailleurs, sous un autre nom, dans un dossier qui
    n'existait pas. Rien d'autre ne change -- donc rien ne doit changer."""
    journal = _lancer(_ecrire_variante(tmp_path, []))
    assert _beta(journal) == pytest.approx(BETA_DE_REFERENCE, rel=1e-5)
    assert "[variables] 2 variable(s)" in journal, (
        "le journal n'annonce pas les variables construites depuis la "
        "declaration :\n%s" % journal[:2000])


# --------------------------------------------------------------------------- #
# 2. ET LA DECLARATION AGIT VRAIMENT                                           #
# --------------------------------------------------------------------------- #
def test_changer_la_MOYENNE_declaree_change_le_resultat(tmp_path):
    """LA verification qui compte.

    `fc` passe de 48 a 30 MPa : un beton nettement plus faible, donc un
    moment resistant plus faible, donc un `beta` plus petit. Si le resultat
    ne bougeait pas, la declaration serait decorative -- lue, validee,
    imprimee, et ignoree par le calcul.

    On n'exige pas une valeur : on exige le SENS et un ordre de grandeur. Ce
    qui est verifie ici est le CABLAGE, pas la physique -- celle-ci a ses
    propres oracles dans `test_20` et `test_40`.
    """
    journal = _lancer(_ecrire_variante(tmp_path, [
        ('args    = [48, 0.12]', 'args    = [30, 0.12]')]))
    beta = _beta(journal)
    assert beta != pytest.approx(BETA_DE_REFERENCE, rel=1e-3), (
        "beta = %.4f, inchange alors que la moyenne de `fc` passe de 48 a "
        "30 MPa : la declaration ne pilote pas le calcul." % beta)
    assert beta < BETA_DE_REFERENCE, (
        "beta = %.4f contre %.4f avec un beton PLUS RESISTANT : un beton "
        "plus faible doit abaisser l'indice de fiabilite."
        % (beta, BETA_DE_REFERENCE))
    assert "args=(30, 0.12)" in journal, (
        "le journal n'annonce pas la valeur declaree")


def test_changer_la_LOI_declaree_est_visible_au_journal(tmp_path):
    """Une etude peut choisir une autre loi pour la meme variable. Le journal
    doit le dire : c'est la premiere chose qu'un ingenieur verifie avant de
    lire un resultat."""
    journal = _lancer(_ecrire_variante(tmp_path, [
        ('[variables.fc]\nloi     = "fc"\nargs    = [48, 0.12]',
         '[variables.fc]\nloi     = "F_permanente"\nargs    = [48, 0.12]')]))
    assert "loi_F_permanente" in journal, (
        "le journal annonce toujours `loi_fc` alors que la declaration a "
        "change de loi :\n%s" % journal[:2500])


# --------------------------------------------------------------------------- #
# 3. UN FICHIER FAUX EST REFUSE, ET TOT                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("faute,attendu", [
    (('loi     = "fc"', 'loi     = "fdc"'), "loi 'fdc' inconnue"),
    (('args    = [48, 0.12]', 'args    = 48'), "liste"),
    (('param   = "COMPRESSIVE_STRENGTH"', 'parm    = "COMPRESSIVE_STRENGTH"'),
     "parm"),
])
def test_une_declaration_fautive_arrete_le_run_avec_un_message(tmp_path, faute,
                                                               attendu):
    """Avant le premier appel solveur, et en nommant la faute. Une etude qui
    demarre puis meurt au bout de 466 s sur le Moulin Blanc a deja coute."""
    chemin = _ecrire_variante(tmp_path, [faute])
    env = dict(os.environ, FIABILITE_ETUDE=chemin, PYTHONPATH="",
               MPLBACKEND="Agg", _FIAB_LOG_REDIRECTED="1")
    proc = subprocess.run(
        [sys.executable, os.path.join(_REPO, "launcher.py"),
         os.path.join("pure_flexion", "AC3_pure_flexion.py")],
        cwd=_REPO, env=env, capture_output=True, text=True,
        errors="replace", timeout=300)
    assert proc.returncode != 0, (
        "une declaration fautive (%s) a laisse le run demarrer" % (faute,))
    sortie = proc.stdout + proc.stderr
    assert attendu in sortie, (
        "le refus ne nomme pas la faute %r :\n%s" % (attendu, sortie[-2500:]))
    assert "HF GRID START" not in sortie, (
        "le refus arrive APRES le debut des appels solveur")
